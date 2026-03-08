"""
PPO Fine-tuning for Paragraph Retrieval Selection.

Key improvement over the tool-selection variant:
  * Dense reward — each retrieval step gets immediate feedback
    (supporting fact → positive, distractor → negative)
  * Causal learning signal — reading supporting paragraphs
    directly causes better LLM answers
  * Clean action space — 11 actions (read_0..read_9, answer)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
import json
import re
import time
import os
import string
import numpy as np
from collections import defaultdict

import requests
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from multi_agent_baseline import (
    NUM_PARAGRAPHS, AgentStep, AgentTrajectory, RetrievalAgent, _STOP_WORDS,
)


# ======================================================================
#  Data structures
# ======================================================================

@dataclass
class RetrievalDecision:
    """A single retrieval decision point in a trajectory."""
    task_id: str
    agent_id: int
    step_id: int
    task_description: str
    context: str
    action_name: str          # "read_0" .. "read_9" or "answer"
    action_idx: int           # 0-9 for read, 10 for answer
    reward: float
    log_prob: float = 0.0


@dataclass
class TrajectoryWithReward:
    """Trajectory annotated with per-step rewards."""
    task_id: str
    agent_id: int
    task_description: str
    decisions: List[RetrievalDecision]
    final_answer: Optional[str]
    correct: bool
    final_reward: float
    num_supporting_read: int
    total_reads: int


# ======================================================================
#  Scoring
# ======================================================================

class TaskScorer:
    """Score task outcomes: text matching + LLM-as-judge fallback."""

    def __init__(self, llm_base_url: str = "http://localhost:11434",
                 judge_model: str = "qwen3:8b"):
        self.task_patterns: Dict[str, str] = {}
        self.llm_base_url = llm_base_url
        self.judge_model = judge_model
        self._cache: Dict[Tuple[str, str], float] = {}

    def register_ground_truth(self, task_id: str, correct_answer: str):
        self.task_patterns[task_id] = correct_answer

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = [t for t in text.split() if t not in {"a", "an", "the"}]
        return " ".join(tokens).strip()

    def _llm_judge(self, prediction: str, ground_truth: str) -> float:
        key = (prediction.lower().strip(), ground_truth.lower().strip())
        if key in self._cache:
            return self._cache[key]
        prompt = (
            "/nothink You are a strict answer judge. Does the predicted answer "
            "match the ground truth? They need not be identical, but must refer "
            "to the same entity/fact.\n\n"
            f"Ground truth: {ground_truth}\n"
            f"Prediction: {prediction}\n\n"
            "Reply with ONLY one word: CORRECT or INCORRECT."
        )
        try:
            r = requests.post(
                f"{self.llm_base_url}/api/generate",
                json={"model": self.judge_model, "prompt": prompt,
                      "temperature": 0.0, "num_predict": 16, "stream": False},
                timeout=30,
            )
            resp = r.json().get("response", "").strip().upper()
            score = 1.0 if "CORRECT" in resp and "INCORRECT" not in resp else 0.0
        except Exception:
            score = 0.0
        self._cache[key] = score
        return score

    def score_answer(self, task_id: str, answer: str) -> float:
        """Cascade: exact match → containment → LLM judge."""
        if task_id not in self.task_patterns:
            return 0.0
        correct = self.task_patterns[task_id]
        na = self._normalize(answer)
        nc = self._normalize(correct)
        if na == nc:
            return 1.0
        if nc in na:
            return 1.0
        return self._llm_judge(answer, correct)


# ======================================================================
#  Policy Network
# ======================================================================

NUM_ACTIONS = NUM_PARAGRAPHS + 1   # read_0..read_9, answer


class RetrievalSelector(nn.Module):
    """Residual MLP policy for paragraph retrieval selection.

    Input:  544-dim features (512 BoW hash + 32 structured)
    Output: 11 action logits  +  1 value estimate
    """

    def __init__(self, input_dim: int = 544, hidden_dim: int = 128,
                 num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions

        self.proj = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.res1_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.res1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.res2_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.res2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

        # Prior: slightly favour reading over answering early
        with torch.no_grad():
            bias = torch.zeros(num_actions)
            bias[:NUM_PARAGRAPHS] = 0.5    # read_i
            bias[NUM_PARAGRAPHS] = -0.5    # answer
            self.action_head.bias.data = bias

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.ln1(self.proj(x)))

        res = h
        h = F.relu(self.res1_fc1(h))
        h = self.res1_fc2(h)
        h = F.relu(self.ln2(h + res))

        res = h
        h = F.relu(self.res2_fc1(h))
        h = self.res2_fc2(h)
        h = F.relu(self.ln3(h + res))

        logits = self.action_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value


# ======================================================================
#  PPO Trainer
# ======================================================================

class PPOTrainer:
    """PPO optimiser for the retrieval selector."""

    def __init__(self, model: nn.Module, lr: float = 3e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2, entropy_coeff: float = 0.05,
                 target_kl: float = 0.02, device: str = "cpu"):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.target_kl = target_kl
        self.device = device
        self.model.to(device)

    # ------ GAE ------
    def compute_gae(self, rewards: List[float],
                    values: List[float]) -> Tuple[List[float], List[float]]:
        advantages, returns = [], []
        adv = 0.0
        for t in reversed(range(len(rewards))):
            nv = values[t + 1] if t + 1 < len(values) else 0.0
            delta = rewards[t] + self.gamma * nv - values[t]
            adv = delta + self.gamma * self.gae_lambda * adv
            advantages.insert(0, adv)
            returns.insert(0, adv + values[t])
        return advantages, returns

    # ------ single PPO update ------
    def train_step(self, batch: Dict, old_log_probs: torch.Tensor,
                   advantages: torch.Tensor, returns: torch.Tensor,
                   old_values: Optional[torch.Tensor] = None,
                   num_epochs: int = 3) -> Dict[str, float]:
        feats = batch["features"].to(self.device)
        actions = batch["actions"].to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        if old_values is not None:
            old_values = old_values.to(self.device)

        metrics: Dict[str, float] = defaultdict(float)
        n_epochs_run = 0
        for _ in range(num_epochs):
            logits, values = self.model(feats)
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)

            action_lp = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
            entropy = -(log_probs * probs).sum(dim=-1).mean()

            ratio = torch.exp(action_lp - old_log_probs)
            s1 = ratio * advantages
            s2 = torch.clamp(ratio, 1 - self.clip_ratio,
                             1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(s1, s2).mean()

            if old_values is not None:
                v_clipped = old_values + torch.clamp(
                    values - old_values, -self.clip_ratio, self.clip_ratio)
                value_loss = 0.5 * torch.max(
                    (values - returns) ** 2,
                    (v_clipped - returns) ** 2,
                ).mean()
            else:
                value_loss = F.mse_loss(values, returns)

            loss = policy_loss + 0.5 * value_loss - self.entropy_coeff * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            n_epochs_run += 1
            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"] += value_loss.item()
            metrics["entropy"] += entropy.item()

            with torch.no_grad():
                approx_kl = (old_log_probs - action_lp).mean().item()
            if self.target_kl and approx_kl > self.target_kl:
                break

        for k in metrics:
            metrics[k] /= max(1, n_epochs_run)
        return dict(metrics)

    # ------ feature extraction ------
    def extract_features(self, context: str) -> torch.Tensor:
        """512-dim BoW hash + 32-dim structured features = 544.

        Dims 0-18:  basic retrieval state (step, read flags, title overlap …)
        Dims 19-31: bridge-entity features — sequential dependency signal.
                    After reading paragraph A, content words from A may match
                    titles of unread paragraphs, revealing chain-reasoning
                    targets that only become visible *after* the read.
        """
        bow = self._bow_hash(context, dim=512)
        extra = torch.zeros(32)

        # --- parse shared fields once ---
        steps_found = re.findall(r'Step: (\d+)', context)
        n_steps = int(steps_found[-1]) if steps_found else 0

        task_match = re.search(r'Task: (.+?)(?:\n|$)', context)
        titles_match = re.search(r'Titles: (.+?)(?:\n|$)', context)
        read_match = re.search(r'Read: (.+?)(?:\nStep|$)', context, re.DOTALL)

        q_words: set = set()
        if task_match:
            q_words = set(task_match.group(1).lower().split()) - _STOP_WORDS

        raw_titles: list = []
        if titles_match:
            raw_titles = titles_match.group(1).split(" | ")

        read_text = ""
        has_read = False
        if read_match and read_match.group(1).strip() != "Nothing yet":
            read_text = read_match.group(1)
            has_read = True

        clean_read = re.sub(r'\[.*?\]', '', read_text) if read_text else ""
        read_words = set(clean_read.lower().split()) - _STOP_WORDS if clean_read.strip() else set()

        n_read = context.count("[READ]")

        # 0: step count (normalised)
        extra[0] = n_steps / 6.0
        # 1: num [READ] tags
        extra[1] = n_read / NUM_PARAGRAPHS
        # 2: has read anything
        extra[2] = 1.0 if has_read else 0.0

        # 3-12: per-paragraph title–question word overlap
        for i, t in enumerate(raw_titles[:NUM_PARAGRAPHS]):
            clean = t.replace("[READ] ", "").replace("[READ]", "").strip()
            t_words = set(clean.lower().split()) - _STOP_WORDS
            if q_words and t_words:
                extra[3 + i] = len(q_words & t_words) / len(q_words)

        # 13: question–read-content word overlap
        if q_words and read_words:
            extra[13] = len(q_words & read_words) / len(q_words)

        # 14-17: step position flags
        for j in range(4):
            extra[14 + j] = float(n_steps >= j + 1)

        # 18: read content length (normalised)
        if read_text:
            extra[18] = min(1.0, len(read_text) / 500.0)

        # --- bridge-entity features (19-31) ---
        # These create genuine sequential dependency: reading paragraph A
        # changes the overlap signal for every other paragraph, so the
        # next decision is meaningfully conditioned on past reads.
        n_bridge = 0
        max_bridge = 0.0
        for i, t in enumerate(raw_titles[:NUM_PARAGRAPHS]):
            is_already_read = "[READ]" in t
            clean = t.replace("[READ] ", "").replace("[READ]", "").strip()
            t_words = set(clean.lower().split()) - _STOP_WORDS
            if t_words and read_words:
                overlap = len(read_words & t_words) / len(t_words)
            else:
                overlap = 0.0
            # 19-28: read-content → paragraph-title overlap per paragraph
            extra[19 + i] = overlap
            if not is_already_read and overlap > 0:
                n_bridge += 1
                max_bridge = max(max_bridge, overlap)

        # 29: fraction of *unread* paragraphs with bridge overlap
        n_unread = max(1, NUM_PARAGRAPHS - n_read)
        extra[29] = n_bridge / n_unread

        # 30: max bridge overlap among unread paragraphs
        extra[30] = max_bridge

        # 31: total question-word coverage from all read information
        if q_words and (read_words or n_read > 0):
            all_info = set(read_words)
            for i, t in enumerate(raw_titles[:NUM_PARAGRAPHS]):
                if "[READ]" in t:
                    clean = t.replace("[READ] ", "").replace("[READ]", "").strip()
                    all_info |= set(clean.lower().split()) - _STOP_WORDS
            extra[31] = len(q_words & all_info) / len(q_words)

        return torch.cat([bow, extra])

    @staticmethod
    def _bow_hash(text: str, dim: int = 512) -> torch.Tensor:
        words = text.lower().split()
        vec = np.zeros(dim)
        for w in words:
            vec[hash(w) % dim] += 1.0
        s = vec.sum()
        if s > 0:
            vec /= s
        return torch.FloatTensor(vec)


# ======================================================================
#  Decision Collector  (dense reward)
# ======================================================================

class DecisionCollector:
    """Collect retrieval decisions and assign dense per-step rewards.

    Reward scheme:
      read supporting paragraph (first time) : +0.3
      read distractor paragraph  (first time): -0.1
      re-read already read paragraph         : -0.2
      answer correctly                       : +1.0
      answer incorrectly                     :  0.0
    """

    REWARD_SUPPORTING = 0.3
    REWARD_DISTRACTOR = -0.1
    REWARD_REREAD = -0.2
    REWARD_CORRECT = 1.0
    REWARD_WRONG = 0.0

    def __init__(self, scorer: TaskScorer):
        self.scorer = scorer
        self.trajectories: List[TrajectoryWithReward] = []

    def collect(self, traj: AgentTrajectory, question: str,
                paragraphs: List[Tuple[str, List[str]]],
                supporting_titles: Set[str]) -> TrajectoryWithReward:
        """Convert an AgentTrajectory into reward-annotated decisions."""
        answer_score = self.scorer.score_answer(
            traj.task_id, traj.final_answer or "")
        is_correct = answer_score > 0.8

        decisions: List[RetrievalDecision] = []
        read_set: Set[int] = set()
        n = min(len(paragraphs), NUM_PARAGRAPHS)

        for step in traj.steps:
            # Reconstruct context (same as solve_with_policy)
            read_paras = [(paragraphs[i][0], paragraphs[i][1])
                          for i in sorted(read_set)]
            context = (
                f"Task: {question}\n"
                f"Titles: {' | '.join(('[READ] ' if i in read_set else '') + paragraphs[i][0] for i in range(n))}\n"
                f"Read: {' '.join('[' + t + '] ' + ' '.join(s) for t, s in read_paras) if read_paras else 'Nothing yet'}\n"
                f"Step: {len(read_set)}"
            )

            if step.action == "answer":
                reward = self.REWARD_CORRECT if is_correct else self.REWARD_WRONG
                action_idx = NUM_PARAGRAPHS  # answer index
            else:
                pidx = step.paragraph_idx
                action_idx = pidx
                if step.already_read:
                    reward = self.REWARD_REREAD
                elif step.is_supporting:
                    reward = self.REWARD_SUPPORTING
                else:
                    reward = self.REWARD_DISTRACTOR

                if not step.already_read:
                    read_set.add(pidx)

            decisions.append(RetrievalDecision(
                task_id=traj.task_id,
                agent_id=traj.agent_id,
                step_id=step.step_id,
                task_description=question,
                context=context,
                action_name=step.action,
                action_idx=action_idx,
                reward=reward,
            ))

        twr = TrajectoryWithReward(
            task_id=traj.task_id,
            agent_id=traj.agent_id,
            task_description=question,
            decisions=decisions,
            final_answer=traj.final_answer,
            correct=is_correct,
            final_reward=answer_score,
            num_supporting_read=traj.num_supporting_read,
            total_reads=traj.total_reads,
        )
        self.trajectories.append(twr)
        return twr


# ======================================================================
#  PPO Fine-Tuner  (on-policy)
# ======================================================================

class PPOFineTuner:
    """On-policy PPO training for paragraph retrieval selection."""

    def __init__(self, scorer: TaskScorer, device: str = "cpu"):
        self.scorer = scorer
        self.device = device
        self.num_actions = NUM_ACTIONS
        self.action_names = [f"read_{i}" for i in range(NUM_PARAGRAPHS)] + ["answer"]

        self.model = RetrievalSelector(input_dim=544, num_actions=self.num_actions)
        self.trainer = PPOTrainer(self.model, device=device)
        self.collector = DecisionCollector(scorer)
        self.training_history: List[Dict] = []

    # ---- action selection (called by RetrievalAgent.solve_with_policy) ----
    def select_action(self, context: str):
        """Policy selects next action given context string.

        Returns (action_name, action_idx, log_prob, value, features).
        """
        features = self.trainer.extract_features(context)
        with torch.no_grad():
            logits, value = self.model(features.unsqueeze(0))
        dist = torch.distributions.Categorical(F.softmax(logits, dim=-1))
        action = dist.sample()
        lp = dist.log_prob(action)
        idx = action.item()
        name = self.action_names[idx] if idx < len(self.action_names) else "answer"
        return name, idx, lp.item(), value.item(), features

    # ---- PPO update on collected decisions ----
    def _ppo_update(self, num_epochs: int = 3,
                    batch_size: int = 16,
                    ppo_epochs: int = 3) -> Dict:
        trajs = self.collector.trajectories
        if not trajs:
            return {}

        all_features, all_actions, all_old_lp = [], [], []
        all_old_values = []
        all_advantages, all_returns = [], []

        for twr in trajs:
            if not twr.decisions:
                continue
            t_feats, t_acts, t_rewards, t_values, t_lps = [], [], [], [], []
            for d in twr.decisions:
                feat = self.trainer.extract_features(d.context)
                with torch.no_grad():
                    logits, val = self.model(feat.unsqueeze(0))
                    lp = F.log_softmax(logits, dim=-1)[0, d.action_idx]
                t_feats.append(feat)
                t_acts.append(d.action_idx)
                t_rewards.append(d.reward)
                t_values.append(val.item())
                t_lps.append(lp.item())

            advs, rets = self.trainer.compute_gae(t_rewards, t_values)
            all_features.extend(t_feats)
            all_actions.extend(t_acts)
            all_old_lp.extend(t_lps)
            all_old_values.extend(t_values)
            all_advantages.extend(advs)
            all_returns.extend(rets)

        if not all_features:
            return {}

        feats_t = torch.stack(all_features)
        acts_t = torch.LongTensor(all_actions)
        old_lp_t = torch.FloatTensor(all_old_lp)
        old_val_t = torch.FloatTensor(all_old_values)
        adv_t = torch.FloatTensor(all_advantages)
        ret_t = torch.FloatTensor(all_returns)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        N = len(all_features)
        history = []
        for _ in range(num_epochs):
            idx = torch.randperm(N)
            epoch_m: Dict[str, float] = defaultdict(float)
            nb = 0
            for s in range(0, N, batch_size):
                bi = idx[s:s + batch_size]
                m = self.trainer.train_step(
                    {"features": feats_t[bi], "actions": acts_t[bi]},
                    old_lp_t[bi], adv_t[bi], ret_t[bi],
                    old_values=old_val_t[bi],
                    num_epochs=ppo_epochs,
                )
                for k, v in m.items():
                    epoch_m[k] += v
                nb += 1
            for k in epoch_m:
                epoch_m[k] /= max(1, nb)
            history.append(dict(epoch_m))
            self.training_history.append(dict(epoch_m))
        return {"epochs": num_epochs, "history": history}

    # ---- on-policy training loop ----
    def on_policy_train(self, examples: Dict[str, Dict],
                        num_iterations: int = 8,
                        max_steps: int = 5,
                        ppo_epochs: int = 3,
                        batch_size: int = 16) -> List[Dict]:
        """
        True on-policy PPO training.
        Each iteration: collect fresh trajectories → PPO update.
        """
        all_metrics: List[Dict] = []

        for it in range(num_iterations):
            print(f"\n{'='*60}")
            print(f"On-Policy Iteration {it+1}/{num_iterations}")
            print(f"{'='*60}")

            # Fresh collector
            self.collector = DecisionCollector(self.scorer)

            n_correct = 0
            n_total = 0
            n_supp = 0
            n_reads = 0

            for q_id, ex in examples.items():
                question = ex["question"]
                paragraphs = ex["paragraphs"]
                supp_titles = ex["supporting_titles"]

                agent = RetrievalAgent(agent_id=0, model="qwen3:8b")
                traj = agent.solve_with_policy(
                    q_id, question, paragraphs, supp_titles,
                    policy=self, max_steps=max_steps,
                )
                twr = self.collector.collect(
                    traj, question, paragraphs, supp_titles,
                )

                n_total += 1
                if twr.correct:
                    n_correct += 1
                n_supp += twr.num_supporting_read
                n_reads += twr.total_reads

                ans = (traj.final_answer or "N/A")[:40]
                tag = "✓" if twr.correct else "✗"
                print(f"  {tag} [{q_id}] {ans}  "
                      f"(reads={twr.total_reads}, supp={twr.num_supporting_read})")

            acc = n_correct / max(1, n_total)
            avg_supp = n_supp / max(1, n_total)
            avg_reads = n_reads / max(1, n_total)
            print(f"\n  accuracy={acc:.1%}  avg_reads={avg_reads:.1f}  "
                  f"avg_supp_found={avg_supp:.1f}")

            train_m = self._ppo_update(
                num_epochs=3, batch_size=batch_size, ppo_epochs=ppo_epochs,
            )
            all_metrics.append({
                "iteration": it + 1,
                "accuracy": acc,
                "correct": n_correct,
                "total": n_total,
                "avg_reads": avg_reads,
                "avg_supporting_found": avg_supp,
                "num_decisions": len(self.collector.trajectories),
                "training": train_m,
            })
        return all_metrics

    # ---- non-sequential ablation helper ----
    def rank_paragraphs_static(self, context: str, n: int) -> List[int]:
        """Rank paragraphs by action logits from the initial (pre-reading) state.

        Used by the Static Top-K baseline to show that sequential state
        updates are necessary: the same model, when forced to decide all
        reads at once from step-0 features, cannot exploit bridge entities.
        """
        features = self.trainer.extract_features(context)
        with torch.no_grad():
            logits, _ = self.model(features.unsqueeze(0))
        return logits[0, :n].argsort(descending=True).tolist()

    # ---- persistence ----
    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        self.model.load_state_dict(
            torch.load(path, map_location=self.device))

    def save_trajectories(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = []
        for twr in self.collector.trajectories:
            data.append({
                "task_id": twr.task_id,
                "task_description": twr.task_description,
                "final_answer": twr.final_answer,
                "correct": twr.correct,
                "final_reward": twr.final_reward,
                "num_supporting_read": twr.num_supporting_read,
                "total_reads": twr.total_reads,
                "decisions": [
                    {"step": d.step_id, "action": d.action_name,
                     "reward": round(d.reward, 3)}
                    for d in twr.decisions
                ],
            })
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def save_training_results(self, path: str, baseline_metrics: Dict = None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        results = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_trajectories": len(self.collector.trajectories),
                "correct": sum(1 for t in self.collector.trajectories if t.correct),
            },
            "baseline_metrics": baseline_metrics or {},
            "training_history": self.training_history,
        }
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
