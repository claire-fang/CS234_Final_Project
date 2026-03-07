"""
PPO Fine-tuning for Tool Selection in Multi-Agent Baseline.

Focuses on optimizing tool selection decisions using task correctness as reward.
Lightweight implementation suitable for CPU training.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import json
import re
import time
import os
import numpy as np
from collections import defaultdict

import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Try importing from transformers/trl, fallback to custom implementation
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
    HAS_TRANSFORMERS = True
except:
    HAS_TRANSFORMERS = False


STOP_WORDS = {'what', 'is', 'the', 'of', 'in', 'a', 'an', 'which', 'who',
              'where', 'when', 'how', 'did', 'does', 'was', 'were', 'do',
              'that', 'this', 'are', 'it', 'its', 'for', 'on', 'at', 'to',
              'and', 'or', 'by', 'with', 'from', 'as', 'has', 'had', 'have'}


@dataclass
class ToolSelectionDecision:
    """A single tool selection point in a trajectory."""
    task_id: str
    agent_id: int
    step_id: int
    task_description: str
    context: str  # Previous steps/history
    chosen_tool: str
    available_tools: List[str] = field(default_factory=lambda: ["no_tool", "web_search", "web_fetch", "extract", "calculator"])
    reward: float = 0.0  # 1.0 if led to correct answer, else scaled down
    log_prob: float = 0.0  # Log probability of tool choice


@dataclass
class TrajectoryWithReward:
    """Trajectory annotated with task correctness reward."""
    task_id: str
    agent_id: int
    task_description: str
    decisions: List[ToolSelectionDecision]
    final_answer: Optional[str]
    correct: bool  # Whether task was solved correctly
    final_reward: float  # Reward signal (0-1)


class TaskScorer:
    """Scores task outcomes using text matching + LLM-as-judge fallback."""
    
    def __init__(self, llm_base_url: str = "http://localhost:11434",
                 judge_model: str = "qwen3:8b"):
        self.task_patterns = {}
        self.llm_base_url = llm_base_url
        self.judge_model = judge_model
        self._judge_cache = {}  # cache to avoid repeat LLM calls
    
    def register_ground_truth(self, task_id: str, correct_answer: str):
        """Register ground truth for a task."""
        self.task_patterns[task_id] = correct_answer
    
    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text: lowercase, remove punctuation and articles."""
        import string
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = [t for t in text.split() if t not in {"a", "an", "the"}]
        return " ".join(tokens).strip()
    
    def _llm_judge(self, prediction: str, ground_truth: str) -> float:
        """Use LLM to judge if prediction matches ground truth semantically."""
        cache_key = (prediction.lower().strip(), ground_truth.lower().strip())
        if cache_key in self._judge_cache:
            return self._judge_cache[cache_key]
        
        prompt = (
            "/nothink You are a strict answer judge. Does the predicted answer match the ground truth answer? "
            "They don't need to be word-for-word identical, but must refer to the same entity/fact.\n\n"
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
            response = r.json().get("response", "").strip().upper()
            # Parse response
            score = 1.0 if "CORRECT" in response and "INCORRECT" not in response else 0.0
        except Exception:
            score = 0.0  # fallback: treat as wrong if LLM unavailable
        
        self._judge_cache[cache_key] = score
        return score
    
    def score_answer(self, task_id: str, answer: str) -> float:
        """Score answer with fast text checks first, LLM judge as fallback.
        
        Scoring cascade:
          1. Exact match (after normalization) → 1.0
          2. Containment (GT appears in answer) → 1.0
          3. LLM judge (semantic equivalence)  → 1.0 or 0.0
        """
        if task_id not in self.task_patterns:
            return 0.0
        
        correct = self.task_patterns[task_id]
        norm_answer = self._normalize(answer)
        norm_correct = self._normalize(correct)
        
        # Level 1: Exact match
        if norm_answer == norm_correct:
            return 1.0
        
        # Level 2: Containment (GT is a substring of the answer)
        if norm_correct in norm_answer:
            return 1.0
        
        # Level 3: LLM judge for semantic equivalence
        return self._llm_judge(answer, correct)


class ToolSelectionDataset(Dataset):
    """Dataset of tool selection decisions for training."""
    
    def __init__(self, decisions: List[ToolSelectionDecision], tokenizer=None, max_len: int = 512):
        self.decisions = decisions
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.available_tools = ["no_tool", "web_search", "web_fetch", "extract", "calculator"]
        self.tool_to_idx = {t: i for i, t in enumerate(self.available_tools)}
    
    def __len__(self):
        return len(self.decisions)
    
    def __getitem__(self, idx):
        dec = self.decisions[idx]
        
        # Create context string
        context_text = f"""Task: {dec.task_description}
Context: {dec.context}
Available tools: {", ".join(dec.available_tools)}
Choose best tool:"""
        
        chosen_idx = self.tool_to_idx.get(dec.chosen_tool, 0)
        
        if self.tokenizer:
            # Tokenize for model training
            encoding = self.tokenizer(
                context_text,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "tool_idx": chosen_idx,
                "reward": dec.reward,
            }
        else:
            # Simple bag-of-words features
            return {
                "context": context_text,
                "tool_idx": chosen_idx,
                "reward": dec.reward,
            }


class SimpleToolSelector(nn.Module):
    """Simple neural network for tool selection (CPU-friendly)."""
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 128, 
                 num_tools: int = 5, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_tools = num_tools
        
        # Embedding and encoding
        self.embed = nn.Embedding(1000, 64)  # Simple embedding for tokens
        
        # Neural network for tool prediction
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.tool_logits = nn.Linear(hidden_dim // 2, num_tools)
        
        # Value head for PPO
        self.value_head = nn.Linear(hidden_dim // 2, 1)
    
    def forward(self, context_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            context_features: (batch_size, input_dim)
        
        Returns:
            tool_logits: (batch_size, num_tools)
            values: (batch_size, 1)
        """
        h = F.relu(self.fc1(context_features))
        h = self.dropout1(h)
        h = F.relu(self.fc2(h))
        h = self.dropout2(h)
        
        logits = self.tool_logits(h)
        values = self.value_head(h)
        
        return logits, values.squeeze(-1)


class DeepToolSelector(nn.Module):
    """Deeper policy network with residual connections and layer normalization.

    Compared to SimpleToolSelector:
    - Residual connections prevent gradient vanishing
    - Layer normalization improves training stability
    - More capacity to learn multi-step decision patterns
    """

    def __init__(self, input_dim: int = 544, hidden_dim: int = 128, num_tools: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.num_tools = num_tools

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.res1_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.res1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.res2_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.res2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

        self.tool_logits = nn.Linear(hidden_dim, num_tools)
        self.value_head = nn.Linear(hidden_dim, 1)

        # Initialize with prior: strongly favor web_search for QA tasks
        # [no_tool, web_search, web_fetch, extract, calculator]
        with torch.no_grad():
            self.tool_logits.bias.data = torch.tensor([1.0, 2.0, -2.0, -2.0, -2.0])

    def forward(self, context_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.ln1(self.input_proj(context_features)))

        res = h
        h = F.relu(self.res1_fc1(h))
        h = self.res1_fc2(h)
        h = F.relu(self.ln2(h + res))

        res = h
        h = F.relu(self.res2_fc1(h))
        h = self.res2_fc2(h)
        h = F.relu(self.ln3(h + res))

        logits = self.tool_logits(h)
        values = self.value_head(h)
        return logits, values.squeeze(-1)


class PPOTrainer:
    """PPO-style trainer for tool selection."""
    
    def __init__(self, model: nn.Module, learning_rate: float = 3e-4, 
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2, entropy_coeff: float = 0.05,
                 device: str = "cpu", feature_mode: str = "bow"):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.device = device
        self.feature_mode = feature_mode
        self.model.to(device)
    
    def compute_gae(self, rewards: List[float], values: List[float]) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        returns = []
        
        advantage = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            advantage = delta + self.gamma * self.gae_lambda * advantage
            
            advantages.insert(0, advantage)
            returns.insert(0, advantage + values[t])
        
        return advantages, returns
    
    def train_step(self, batch: Dict, old_log_probs: torch.Tensor, 
                   advantages: torch.Tensor, returns: torch.Tensor,
                   num_epochs: int = 3) -> Dict[str, float]:
        """Single PPO training step."""
        context_features = batch["context_features"].to(self.device)
        action_indices = batch["action_indices"].to(self.device)
        
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        
        metrics = defaultdict(float)
        
        for epoch in range(num_epochs):
            # Forward pass
            logits, values = self.model(context_features)
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Get log probs of taken actions
            action_log_probs = log_probs.gather(1, action_indices.unsqueeze(-1)).squeeze(-1)
            entropy = -(log_probs * probs).sum(dim=-1).mean()
            
            # PPO clipping
            ratio = torch.exp(action_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, returns)
            
            loss = policy_loss + 0.5 * value_loss - self.entropy_coeff * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            
            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"] += value_loss.item()
            metrics["entropy"] += entropy.item()
        
        # Average over epochs
        for key in metrics:
            metrics[key] /= num_epochs
        
        return metrics
    
    def extract_features(self, context: str) -> torch.Tensor:
        """Extract features from context text.

        Modes:
            "bow": 512-dim bag-of-words hash (baseline)
            "structured": 512 BoW + 32 handcrafted features = 544-dim
                Handcrafted features capture step count, tool usage,
                question-history word overlap, and progress signals
                that pure BoW cannot represent.
        """
        bow = self._bow_hash(context)

        if self.feature_mode == "bow":
            return bow

        # --- Structured mode: BoW + handcrafted signals ---
        extra = torch.zeros(32)

        # Step count (normalized)
        steps_found = re.findall(r'Step \d+:', context)
        n_steps = len(steps_found)
        extra[0] = n_steps / 8.0

        # Per-tool mention count (normalized by steps)
        tools = ["no_tool", "web_search", "web_fetch", "extract", "calculator"]
        for i, t in enumerate(tools):
            extra[1 + i] = context.lower().count(t) / max(1.0, n_steps + 1)

        # Question-history word overlap
        task_match = re.search(r'Task: (.+?)(?:\n|$)', context)
        if task_match and 'History:' in context:
            q_words = set(task_match.group(1).lower().split()) - STOP_WORDS
            history_text = context.split('History:')[-1].lower()
            h_words = set(history_text.split())
            if q_words:
                extra[6] = len(q_words & h_words) / len(q_words)

        # Has history
        extra[7] = 0.0 if "No steps yet" in context else 1.0

        # History text length (normalized)
        if 'History:' in context:
            h_text = context.split('History:')[-1]
            extra[8] = min(1.0, len(h_text) / 500.0)

        # Step position indicators (binary)
        extra[9] = float(n_steps >= 1)
        extra[10] = float(n_steps >= 2)
        extra[11] = float(n_steps >= 3)
        extra[12] = float(n_steps >= 4)

        # Error/failure signals in context
        extra[13] = context.lower().count('error') / max(1.0, n_steps + 1)
        extra[14] = context.lower().count('failed') / max(1.0, n_steps + 1)

        return torch.cat([bow, extra])

    def _bow_hash(self, context: str, dim: int = 512) -> torch.Tensor:
        """Bag-of-words with hashing."""
        words = context.lower().split()
        feature_vec = np.zeros(dim)
        for word in words:
            idx = hash(word) % dim
            feature_vec[idx] += 1.0
        if feature_vec.sum() > 0:
            feature_vec /= feature_vec.sum()
        return torch.FloatTensor(feature_vec)


class DecisionCollector:
    """Collects tool selection decisions from agent trajectories."""
    
    def __init__(self, scorer: TaskScorer, step_penalty: float = -0.02,
                 tool_fail_penalty: float = -0.05):
        """
        Initialize collector with per-step reward configuration.
        
        Args:
            scorer: Task scorer for evaluating answers
            step_penalty: Small negative reward per step to incentivize efficiency
            tool_fail_penalty: Additional penalty when a tool call fails
        """
        self.scorer = scorer
        self.step_penalty = step_penalty
        self.tool_fail_penalty = tool_fail_penalty
        self.trajectories_with_rewards: List[TrajectoryWithReward] = []
    
    def collect_from_trajectory(self, trajectory, task_description: str, 
                               final_answer: Optional[str]) -> TrajectoryWithReward:
        """
        Convert an agent trajectory to reward-annotated decisions.
        
        Per-step reward structure (addresses credit assignment):
        - Each step receives step_penalty (small negative, incentivizes efficiency)
        - Failed tool calls receive additional tool_fail_penalty
        - The LAST step receives the answer correctness score (0-1)
        This creates a sequential reward signal suitable for GAE.
        """
        # Score the final answer
        answer_score = self.scorer.score_answer(trajectory.task_id, final_answer or "")
        is_correct = answer_score > 0.8
        
        decisions = []
        num_steps = len(trajectory.steps)
        
        for i, step in enumerate(trajectory.steps):
            # Reconstruct context matching solve_with_policy format exactly
            prev_steps = trajectory.steps[:i][-3:]  # last 3 previous steps
            history_str = "\n".join([
                f"Step {j+1}: {s.tool_called} -> {s.tool_result.content[:100]}..."
                for j, s in enumerate(prev_steps)
            ])
            context = f"Task: {task_description}\nHistory: {history_str if history_str else 'No steps yet'}"
            
            # Per-step reward: step penalty + optional tool failure penalty
            reward = self.step_penalty
            if not step.tool_result.ok:
                reward += self.tool_fail_penalty
            # Final step also gets the answer correctness score
            if i == num_steps - 1:
                reward += answer_score
            
            decision = ToolSelectionDecision(
                task_id=trajectory.task_id,
                agent_id=trajectory.agent_id,
                step_id=step.step_id,
                task_description=task_description,
                context=context,
                chosen_tool=step.tool_called,
                reward=reward,
            )
            decisions.append(decision)
        
        traj_with_reward = TrajectoryWithReward(
            task_id=trajectory.task_id,
            agent_id=trajectory.agent_id,
            task_description=task_description,
            decisions=decisions,
            final_answer=final_answer,
            correct=is_correct,
            final_reward=answer_score,
        )
        
        self.trajectories_with_rewards.append(traj_with_reward)
        return traj_with_reward
    
    def get_all_decisions(self) -> List[ToolSelectionDecision]:
        """Get all collected decisions."""
        all_decisions = []
        for traj in self.trajectories_with_rewards:
            all_decisions.extend(traj.decisions)
        return all_decisions


class PPOFineTuner:
    """Main class for on-policy PPO fine-tuning of tool selection."""
    
    def __init__(self, scorer: TaskScorer, device: str = "cpu",
                 step_penalty: float = -0.02, tool_fail_penalty: float = -0.05,
                 feature_mode: str = "structured", model_type: str = "deep"):
        """
        Initialize PPO fine-tuner (on-policy).
        
        Args:
            scorer: Task scorer for evaluating answers
            device: Device to use ("cpu" or "cuda")
            step_penalty: Small negative reward per step (incentivizes efficiency)
            tool_fail_penalty: Additional penalty when a tool call fails
            feature_mode: "bow" (512-dim bag-of-words) or
                          "structured" (512 BoW + 32 handcrafted = 544-dim)
            model_type: "simple" (2-layer MLP) or "deep" (residual MLP with LayerNorm)
        """
        self.scorer = scorer
        self.device = device
        self.step_penalty = step_penalty
        self.tool_fail_penalty = tool_fail_penalty
        self.feature_mode = feature_mode
        self.available_tools = ["no_tool", "web_search", "web_fetch", "extract", "calculator"]
        self.collector = DecisionCollector(
            scorer,
            step_penalty=step_penalty,
            tool_fail_penalty=tool_fail_penalty,
        )
        
        # Feature dimension depends on mode
        feature_dim = 512 if feature_mode == "bow" else 544
        
        # Initialize model
        if model_type == "deep":
            self.model = DeepToolSelector(input_dim=feature_dim, num_tools=5)
        else:
            self.model = SimpleToolSelector(input_dim=feature_dim, num_tools=5)
        
        self.trainer = PPOTrainer(self.model, device=device, feature_mode=feature_mode)
        
        self.training_history = []
    
    def collect_trajectories(self, trajectories: List, tasks: Dict[str, str]):
        """
        Collect trajectories from agents and annotate with rewards.
        
        Args:
            trajectories: List of AgentTrajectory objects
            tasks: Dict mapping task_id -> task_description
        """
        for traj in trajectories:
            task_desc = tasks.get(traj.task_id, "")
            self.collector.collect_from_trajectory(traj, task_desc, traj.final_answer)
        
        print(f"Collected {len(trajectories)} trajectories")
        
        # Stats
        decisions = self.collector.get_all_decisions()
        correct_trajs = sum(1 for t in self.collector.trajectories_with_rewards if t.correct)
        print(f"  {len(decisions)} decisions, {correct_trajs} correct trajectories")
    
    def fine_tune(self, num_epochs: int = 3, batch_size: int = 8,
                  ppo_epochs: int = 3) -> Dict[str, Any]:
        """
        Fine-tune model on collected decisions using proper GAE advantages.
        
        Processes trajectories individually to compute per-trajectory GAE,
        then does PPO clipped updates with mini-batches.
        
        Args:
            num_epochs: Number of outer training epochs
            batch_size: Mini-batch size
            ppo_epochs: PPO update epochs per batch
        
        Returns:
            Training metrics
        """
        trajs = self.collector.trajectories_with_rewards
        
        if not trajs:
            print("No trajectories collected. Run collect_trajectories first.")
            return {}
        
        # Pre-compute features, values, and GAE for all trajectories
        all_features = []
        all_actions = []
        all_old_log_probs = []
        all_advantages = []
        all_returns = []
        
        for traj in trajs:
            if not traj.decisions:
                continue
            
            traj_features = []
            traj_actions = []
            traj_rewards = []
            traj_values = []
            traj_log_probs = []
            
            for d in traj.decisions:
                features = self.trainer.extract_features(d.context)
                action_idx = self.available_tools.index(d.chosen_tool)
                
                with torch.no_grad():
                    logits, value = self.model(features.unsqueeze(0))
                    log_probs = F.log_softmax(logits, dim=-1)
                    old_lp = log_probs[0, action_idx]
                
                traj_features.append(features)
                traj_actions.append(action_idx)
                traj_rewards.append(d.reward)
                traj_values.append(value.item())
                traj_log_probs.append(old_lp.item())
            
            # Compute GAE for this trajectory (sequential credit assignment)
            advantages, returns = self.trainer.compute_gae(traj_rewards, traj_values)
            
            all_features.extend(traj_features)
            all_actions.extend(traj_actions)
            all_old_log_probs.extend(traj_log_probs)
            all_advantages.extend(advantages)
            all_returns.extend(returns)
        
        if not all_features:
            print("No valid decisions to train on.")
            return {}
        
        # Convert to tensors
        features_t = torch.stack(all_features)
        actions_t = torch.LongTensor(all_actions)
        old_lp_t = torch.FloatTensor(all_old_log_probs)
        adv_t = torch.FloatTensor(all_advantages)
        ret_t = torch.FloatTensor(all_returns)
        
        # Normalize advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        
        total_decisions = len(all_features)
        print(f"\nFine-tuning on {total_decisions} decisions from {len(trajs)} trajectories "
              f"({num_epochs} epochs, GAE gamma={self.trainer.gamma}, lambda={self.trainer.gae_lambda})...")
        
        metrics_history = []
        
        for epoch in range(num_epochs):
            indices = torch.randperm(total_decisions)
            epoch_metrics = defaultdict(float)
            num_batches = 0
            
            for start in range(0, total_decisions, batch_size):
                batch_idx = indices[start:start + batch_size]
                
                batch = {
                    "context_features": features_t[batch_idx],
                    "action_indices": actions_t[batch_idx],
                }
                
                step_metrics = self.trainer.train_step(
                    batch,
                    old_lp_t[batch_idx],
                    adv_t[batch_idx],
                    ret_t[batch_idx],
                    num_epochs=ppo_epochs,
                )
                
                for k, v in step_metrics.items():
                    epoch_metrics[k] += v
                num_batches += 1
            
            for k in epoch_metrics:
                epoch_metrics[k] /= max(1, num_batches)
            
            metrics_history.append(dict(epoch_metrics))
            print(f"  Epoch {epoch+1}/{num_epochs}: policy_loss={epoch_metrics.get('policy_loss', 0):.4f} "
                  f"value_loss={epoch_metrics.get('value_loss', 0):.4f} "
                  f"entropy={epoch_metrics.get('entropy', 0):.4f}")
        
        self.training_history.extend(metrics_history)
        return {"epochs": num_epochs, "history": metrics_history}
    
    def select_tool(self, context: str) -> Tuple[str, int, float, float, torch.Tensor]:
        """
        Use current policy to select a tool (for on-policy data collection).
        
        Args:
            context: Text description of current task + history
        
        Returns:
            (tool_name, action_idx, log_prob, value_estimate, features)
        """
        features = self.trainer.extract_features(context)
        
        with torch.no_grad():
            logits, value = self.model(features.unsqueeze(0))
        
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        tool_name = self.available_tools[action.item()]
        return tool_name, action.item(), log_prob.item(), value.item(), features
    
    def on_policy_train(self, examples: Dict[str, Dict], num_iterations: int = 5,
                        num_agents: int = 1, max_steps: int = 4,
                        model: str = "qwen3:8b", ppo_epochs: int = 3,
                        batch_size: int = 4) -> List[Dict]:
        """
        True on-policy PPO training loop.
        
        At each iteration:
        1. Collect FRESH trajectories using the CURRENT policy
        2. Compute per-step rewards (step penalty + final correctness)
        3. Compute GAE advantages for proper credit assignment
        4. PPO clipped update
        
        This is on-policy because we re-collect trajectories with the
        updated policy at each iteration, ensuring the training data
        matches the current policy distribution.
        
        Args:
            examples: Dict of question_id -> {"question": str, "answer": str, ...}
            num_iterations: Number of on-policy iterations
            num_agents: Number of agents per question
            max_steps: Max tool calls per agent
            model: LLM model name for Ollama
            ppo_epochs: PPO update epochs per iteration
            batch_size: Mini-batch size
        
        Returns:
            List of per-iteration metrics
        """
        from multi_agent_baseline import LocalLLMAgent
        
        all_iter_metrics = []
        
        for iteration in range(num_iterations):
            print(f"\n{'='*60}")
            print(f"On-Policy Iteration {iteration+1}/{num_iterations}")
            print(f"{'='*60}")
            
            # Reset collector for fresh on-policy data
            self.collector = DecisionCollector(
                self.scorer,
                step_penalty=self.step_penalty,
                tool_fail_penalty=self.tool_fail_penalty,
            )
            
            num_correct = 0
            num_total = 0
            
            for q_id, example in examples.items():
                question = example["question"]
                
                for agent_id in range(num_agents):
                    agent = LocalLLMAgent(
                        agent_id=agent_id,
                        model=model,
                        tavily_api_key=os.getenv("TAVILY_API_KEY"),
                    )
                    
                    traj = agent.solve_with_policy(
                        q_id, question,
                        tool_policy=self,
                        max_steps=max_steps,
                    )
                    
                    traj_wr = self.collector.collect_from_trajectory(
                        traj, question, traj.final_answer
                    )
                    
                    num_total += 1
                    if traj_wr.correct:
                        num_correct += 1
                    
                    answer_str = (traj.final_answer or "N/A")[:40]
                    print(f"  [{q_id}] Agent {agent_id}: {answer_str}... "
                          f"({'CORRECT' if traj_wr.correct else 'WRONG'}, "
                          f"{traj.total_tool_calls} tools)")
            
            iteration_accuracy = num_correct / max(1, num_total)
            print(f"\n  Collection accuracy: {iteration_accuracy:.2%}")
            
            # PPO update on fresh on-policy data
            train_metrics = self.fine_tune(
                num_epochs=5,
                batch_size=batch_size,
                ppo_epochs=ppo_epochs,
            )
            
            iter_metrics = {
                "iteration": iteration + 1,
                "accuracy": iteration_accuracy,
                "correct": num_correct,
                "total": num_total,
                "num_decisions": len(self.collector.get_all_decisions()),
                "training": train_metrics,
            }
            all_iter_metrics.append(iter_metrics)
            
            print(f"  Summary: accuracy={iteration_accuracy:.2%}, "
                  f"decisions={iter_metrics['num_decisions']}")
        
        return all_iter_metrics
    
    def save_model(self, path: str):
        """Save fine-tuned model."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load fine-tuned model."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")
    
    def save_trajectories(self, path: str):
        """Save collected trajectories to JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        
        trajectories_data = []
        for traj in self.collector.trajectories_with_rewards:
            traj_dict = {
                "task_id": traj.task_id,
                "agent_id": traj.agent_id,
                "task_description": traj.task_description,
                "final_answer": traj.final_answer,
                "correct": traj.correct,
                "final_reward": traj.final_reward,
                "decisions": [
                    {
                        "step_id": d.step_id,
                        "chosen_tool": d.chosen_tool,
                        "reward": d.reward,
                        "task_description": d.task_description,
                    }
                    for d in traj.decisions
                ]
            }
            trajectories_data.append(traj_dict)
        
        with open(path, 'w') as f:
            json.dump(trajectories_data, f, indent=2)
        
        print(f"Trajectories saved to {path} ({len(trajectories_data)} trajectories)")
    
    def save_training_results(self, path: str, baseline_metrics: Dict = None):
        """Save training metrics and analysis results."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        
        decisions = self.collector.get_all_decisions()
        
        # Compute statistics
        correct_count = sum(1 for d in decisions if d.reward > 0.8)
        incorrect_count = sum(1 for d in decisions if d.reward < -0.1)
        tool_stats = defaultdict(int)
        for d in decisions:
            tool_stats[d.chosen_tool] += 1
        
        results = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_trajectories": len(self.collector.trajectories_with_rewards),
                "total_decisions": len(decisions),
                "successful_trajectories": sum(1 for t in self.collector.trajectories_with_rewards if t.correct),
                "failed_trajectories": sum(1 for t in self.collector.trajectories_with_rewards if not t.correct),
            },
            "baseline_metrics": baseline_metrics or {},
            "training_history": self.training_history,
            "decision_statistics": {
                "high_reward_decisions": correct_count,
                "low_reward_decisions": incorrect_count,
                "tool_usage": dict(tool_stats),
            },
            "decisions_sample": [
                {
                    "task": d.task_description[:50],
                    "tool": d.chosen_tool,
                    "reward": d.reward,
                }
                for d in decisions[:10]
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Training results saved to {path}")
    
    def load_trajectories(self, path: str) -> List[TrajectoryWithReward]:
        """Load trajectories from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        trajectories = []
        for traj_dict in data:
            decisions = [
                ToolSelectionDecision(
                    task_id=traj_dict["task_id"],
                    agent_id=traj_dict["agent_id"],
                    step_id=d["step_id"],
                    task_description=d["task_description"],
                    context="",
                    chosen_tool=d["chosen_tool"],
                    reward=d["reward"],
                )
                for d in traj_dict["decisions"]
            ]
            
            traj = TrajectoryWithReward(
                task_id=traj_dict["task_id"],
                agent_id=traj_dict["agent_id"],
                task_description=traj_dict["task_description"],
                decisions=decisions,
                final_answer=traj_dict["final_answer"],
                correct=traj_dict["correct"],
                final_reward=traj_dict["final_reward"],
            )
            trajectories.append(traj)
        
        print(f"Loaded {len(trajectories)} trajectories from {path}")
        return trajectories


if __name__ == "__main__":
    print("PPO Fine-tuner module loaded. Use with multi_agent_baseline.py")
