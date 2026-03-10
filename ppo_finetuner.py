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
    mask: Optional[List[int]] = None  # indices that were masked at this step
    question: Optional[str] = None
    paragraphs: Optional[List[Tuple[str, List[str]]]] = None


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
    """Dual-path policy for paragraph retrieval selection.

    Architecture:
      Path A — Per-paragraph scoring ("learned greedy"):
        Takes per_para_feats (N x D) and produces a relevance score for each
        paragraph.  Uses an MLP to combine multiple per-paragraph signals
        including IDF-weighted overlap, best-sentence match, bridge similarity,
        and co-occurrence features that go beyond simple word overlap.
      Path B — Context pathway:
        Takes global context features (question emb + progress + bridge)
        and produces a context vector that modulates the scores.
      Combined → 11 action logits (read_0..read_9 + answer).

    The model starts at greedy-level performance but can learn to surpass
    it by leveraging the richer per-paragraph features.
    """

    def __init__(self, input_dim: int = 426, hidden_dim: int = 128,
                 num_actions: int = NUM_ACTIONS,
                 num_paragraphs: int = NUM_PARAGRAPHS,
                 per_para_dim: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.num_paragraphs = num_paragraphs
        self.per_para_dim = per_para_dim

        # Per-paragraph scoring: MLP (10→8→1)
        # Input per paragraph: para_sim(1) + q_overlap(1) + bridge(1) +
        #   is_read(1) + idf_overlap(1) + best_sent(1) + para_len(1) +
        #   unique_ratio(1) + cooccurrence(1) + rank_feat(1) = 10
        self.para_scorer = nn.Sequential(
            nn.Linear(per_para_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

        # Context pathway: processes global features
        # global_dim = emb_dim + 3 progress + 5 global_structured
        global_dim = input_dim - num_paragraphs * 3 - 60  # subtract para-specific dims + new para feats
        self.ctx_proj = nn.Linear(global_dim, hidden_dim)
        self.ctx_ln = nn.LayerNorm(hidden_dim)

        self.ctx_res1 = nn.Linear(hidden_dim, hidden_dim)
        self.ctx_res2 = nn.Linear(hidden_dim, hidden_dim)
        self.ctx_ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.15)

        # Context → per-paragraph modulation
        self.ctx_to_para = nn.Linear(hidden_dim, num_paragraphs)

        # Answer head: should we stop reading?
        self.answer_head = nn.Linear(hidden_dim, 1)

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

        # Learnable weight between direct para score and context modulation
        # sigmoid(0.5)≈0.62 — gives context pathway meaningful weight
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Initialise para_scorer: first layer amplifies word overlap (dim 0)
        with torch.no_grad():
            # First layer: emphasize para_sim (idx 0) and idf_overlap (idx 4)
            self.para_scorer[0].weight.data.zero_()
            self.para_scorer[0].bias.data.zero_()
            self.para_scorer[0].weight.data[0, 0] = 5.0   # para_sim
            self.para_scorer[0].weight.data[1, 1] = 3.0   # q_overlap
            self.para_scorer[0].weight.data[2, 4] = 4.0   # idf_overlap
            self.para_scorer[0].weight.data[3, 5] = 3.0   # best_sent
            # Second layer: combine positively
            self.para_scorer[2].weight.data.fill_(1.0)
            self.para_scorer[2].bias.data.zero_()
            # Context modulation near zero initially
            self.ctx_to_para.weight.data *= 0.01
            self.ctx_to_para.bias.data.zero_()
            # Answer head starts negative (prefer reading over stopping early)
            self.answer_head.bias.data.fill_(-2.0)

    def forward(self, x: torch.Tensor,
                para_feats: torch.Tensor = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, input_dim) full feature vector
            para_feats: (B, N, per_para_dim) per-paragraph features.
                        If None, extracted from x using legacy layout.
        Returns:
            logits: (B, num_actions)
            value:  (B,)
        """
        B = x.shape[0]

        if para_feats is None:
            # Extract per-paragraph features from the flat vector
            # Layout: [emb | para_sims(10) | extra(32) | new_para(60)]
            # New layout adds 60 dims: idf_overlap(10) + best_sent(10) +
            #   para_len(10) + unique_ratio(10) + cooccurrence(10) + rank_feat(10)
            emb_dim = self.input_dim - self.num_paragraphs - 32 - 60
            para_sims = x[:, emb_dim:emb_dim + self.num_paragraphs]  # (B, 10)
            extra_start = emb_dim + self.num_paragraphs
            q_overlap = x[:, extra_start + 3:extra_start + 3 + self.num_paragraphs]  # (B, 10)
            bridge = x[:, extra_start + 19:extra_start + 19 + self.num_paragraphs]  # (B, 10)

            is_read = torch.zeros(B, self.num_paragraphs, device=x.device)

            # New per-paragraph features (after extra[32])
            new_start = extra_start + 32
            idf_overlap = x[:, new_start:new_start + 10]           # (B, 10)
            best_sent = x[:, new_start + 10:new_start + 20]        # (B, 10)
            para_len = x[:, new_start + 20:new_start + 30]         # (B, 10)
            unique_ratio = x[:, new_start + 30:new_start + 40]     # (B, 10)
            cooccurrence = x[:, new_start + 40:new_start + 50]     # (B, 10)
            rank_feat = x[:, new_start + 50:new_start + 60]        # (B, 10)

            # (B, 10, 10)
            para_feats = torch.stack([
                para_sims, q_overlap, bridge, is_read,
                idf_overlap, best_sent, para_len, unique_ratio,
                cooccurrence, rank_feat,
            ], dim=-1)

            # Global features: emb + progress[0:3] + global_extra[13,14:18,29:32]
            progress = x[:, extra_start:extra_start + 3]
            global_extra = torch.cat([
                x[:, extra_start + 13:extra_start + 14],  # q↔read overlap
                x[:, extra_start + 14:extra_start + 18],  # step flags
                x[:, extra_start + 18:extra_start + 19],  # read length
                x[:, extra_start + 29:extra_start + 32],  # bridge agg + coverage
            ], dim=-1)  # (B, 9)
            global_feats = torch.cat([
                x[:, :emb_dim],  # embedding
                progress,
                global_extra,
            ], dim=-1)
        else:
            # Direct per-paragraph features provided
            emb_dim = self.input_dim - self.num_paragraphs - 32 - 60
            extra_start = emb_dim + self.num_paragraphs
            progress = x[:, extra_start:extra_start + 3]
            global_extra = torch.cat([
                x[:, extra_start + 13:extra_start + 14],
                x[:, extra_start + 14:extra_start + 18],
                x[:, extra_start + 18:extra_start + 19],
                x[:, extra_start + 29:extra_start + 32],
            ], dim=-1)
            global_feats = torch.cat([
                x[:, :emb_dim], progress, global_extra,
            ], dim=-1)

        # Path A: per-paragraph scores
        para_scores = self.para_scorer(para_feats).squeeze(-1)  # (B, 10)

        # Path B: context processing
        h = F.relu(self.ctx_ln(self.ctx_proj(global_feats)))
        res = h
        h = F.relu(self.ctx_res1(h))
        h = self.ctx_res2(h)
        h = F.relu(self.ctx_ln2(h + res))
        h = self.dropout(h)

        ctx_modulation = self.ctx_to_para(h)  # (B, 10)

        # Combine: alpha * direct_score + (1 - alpha) * context_modulation
        a = torch.sigmoid(self.alpha)
        read_logits = a * para_scores + (1 - a) * ctx_modulation  # (B, 10)

        # Answer logit from context
        answer_logit = self.answer_head(h)  # (B, 1)

        logits = torch.cat([read_logits, answer_logit], dim=-1)  # (B, 11)
        value = self.value_head(h).squeeze(-1)  # (B,)

        return logits, value


# ======================================================================
#  Learned Reward Model (potential-based shaping)
# ======================================================================

class RewardModel(nn.Module):
    """Learned potential function Φ(s) for reward shaping.

    Predicts expected final recall from the current state features.
    Used for potential-based reward shaping (Ng et al. 1999):
        R_shaped(s,a,s') = R_env(s,a,s') + γ·Φ(s') − Φ(s)

    Theoretical guarantee: potential-based shaping preserves the
    optimal policy while providing denser learning signal.
    The model learns which intermediate states lead to high recall,
    giving PPO forward-looking gradient information that the binary
    gold/distractor reward cannot provide.
    """

    def __init__(self, input_dim: int = 614, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # output in [0, 1] to match recall range
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict potential (expected recall) from state features.
        Args:  x: (B, input_dim) state features
        Returns: (B,) predicted recall in [0, 1]
        """
        return self.net(x).squeeze(-1)


# ======================================================================
#  PPO Trainer
# ======================================================================

class PPOTrainer:
    """PPO optimiser for the retrieval selector."""

    def __init__(self, model: nn.Module, lr: float = 3e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_ratio: float = 0.1, entropy_coeff: float = 0.02,
                 target_kl: float = 0.015, device: str = "cpu",
                 blind: bool = False, kl_coeff: float = 0.2):
        self.model = model
        self.lr = lr
        self.optimizer = optim.Adam(model.parameters(), lr=lr,
                                    weight_decay=0.01)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = 0.2
        self.entropy_coeff = entropy_coeff
        self.kl_coeff = kl_coeff
        self.ref_model: Optional[RetrievalSelector] = None
        self.blind = blind

        # Sentence-transformer for semantic embeddings
        self._st_model = None
        self._st_cache: Dict[str, np.ndarray] = {}
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._st_model.eval()
            print("  Loaded sentence-transformer: all-MiniLM-L6-v2 (384-dim)")
        except ImportError:
            print("  WARNING: sentence-transformers not installed, falling back to BoW hash")
        self.target_kl = 0.05
        self.device = device
        self.model.to(device)

    def snapshot_reference(self):
        """Save a frozen copy of the current model as the BC reference for KL penalty."""
        import copy
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        print("  [KL-ref] Saved BC reference model for KL penalty")

    def set_lr(self, new_lr: float):
        """Update optimizer learning rate (for linear decay)."""
        for pg in self.optimizer.param_groups:
            pg["lr"] = new_lr

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
                   num_epochs: int = 3,
                   action_masks: Optional[torch.Tensor] = None) -> Dict[str, float]:
        feats = batch["features"].to(self.device)
        actions = batch["actions"].to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        if old_values is not None:
            old_values = old_values.to(self.device)
        if action_masks is not None:
            action_masks = action_masks.to(self.device)

        metrics: Dict[str, float] = defaultdict(float)
        n_epochs_run = 0
        for _ in range(num_epochs):
            logits, values = self.model(feats)
            if action_masks is not None:
                logits = logits + action_masks
            logits = logits.clamp(min=-30, max=30)
            probs = F.softmax(logits, dim=-1)
            probs = probs.clamp(min=1e-8)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            log_probs = probs.log()

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

            # KL penalty: keep PPO policy close to BC reference
            kl_loss = torch.tensor(0.0, device=feats.device)
            if self.ref_model is not None and self.kl_coeff > 0:
                with torch.no_grad():
                    ref_logits, _ = self.ref_model(feats)
                    if action_masks is not None:
                        ref_logits = ref_logits + action_masks
                    ref_logits = ref_logits.clamp(min=-30, max=30)
                    ref_probs = F.softmax(ref_logits, dim=-1).clamp(min=1e-8)
                    ref_probs = ref_probs / ref_probs.sum(dim=-1, keepdim=True)
                kl_loss = (probs * (probs.log() - ref_probs.log())).sum(dim=-1).mean()
                loss = loss + self.kl_coeff * kl_loss

            if torch.isnan(loss) or torch.isinf(loss):
                break

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            n_epochs_run += 1
            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"] += value_loss.item()
            metrics["entropy"] += entropy.item()
            metrics["kl_from_bc"] += kl_loss.item()

            with torch.no_grad():
                approx_kl = (old_log_probs - action_lp).mean().item()
            if self.target_kl and approx_kl > self.target_kl:
                break

        for k in metrics:
            metrics[k] /= max(1, n_epochs_run)
        return dict(metrics)

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text with sentence-transformer (cached)."""
        if text in self._st_cache:
            return self._st_cache[text]
        with torch.no_grad():
            emb = self._st_model.encode(text, show_progress_bar=False)
        self._st_cache[text] = emb
        return emb

    def _batch_encode(self, texts: List[str]) -> List[np.ndarray]:
        """Batch-encode texts, using cache for hits and batching misses."""
        results = [None] * len(texts)
        miss_indices = []
        miss_texts = []
        for i, t in enumerate(texts):
            if t in self._st_cache:
                results[i] = self._st_cache[t]
            else:
                miss_indices.append(i)
                miss_texts.append(t)
        if miss_texts:
            with torch.no_grad():
                embs = self._st_model.encode(miss_texts, show_progress_bar=False, batch_size=64)
            for j, idx in enumerate(miss_indices):
                self._st_cache[miss_texts[j]] = embs[j]
                results[idx] = embs[j]
        return results

    # ------ feature extraction ------
    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (
            np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def extract_features(self, context: str,
                         question: str = None,
                         paragraphs: List[Tuple[str, List[str]]] = None,
                         ) -> torch.Tensor:
        """Feature vector: emb + 10 para_sim + 32 structured + 60 new_para.

        New per-paragraph features (beyond greedy's word overlap):
          idf_overlap(10)  - IDF-weighted word overlap (rare words count more)
          best_sent(10)    - best single-sentence match score
          para_len(10)     - normalized paragraph length
          unique_ratio(10) - fraction of unique (non-stopword) words
          cooccurrence(10) - max word overlap between this para and any other
          rank_feat(10)    - greedy rank position (normalized)
        """
        # --- parse context string once ---
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
        read_words = (set(clean_read.lower().split()) - _STOP_WORDS
                      if clean_read.strip() else set())
        n_read = context.count("[READ]")

        # --- 384-dim question embedding + 10-dim paragraph similarity ---
        para_embs: List[Optional[np.ndarray]] = [None] * NUM_PARAGRAPHS
        if self._st_model is not None and question:
            q_emb = self._encode_text(question)
            emb = torch.FloatTensor(q_emb)

            para_sims = torch.zeros(NUM_PARAGRAPHS)
            if paragraphs:
                # Batch-encode all paragraphs at once for speed
                para_texts = []
                for pi, (title, sents) in enumerate(paragraphs[:NUM_PARAGRAPHS]):
                    if self.blind:
                        para_texts.append(" ".join(sents[:3]))
                    else:
                        para_texts.append(title + " " + " ".join(sents[:3]))
                p_embs = self._batch_encode(para_texts)
                for pi, p_emb in enumerate(p_embs):
                    para_embs[pi] = p_emb
                    para_sims[pi] = self._cosine(q_emb, p_emb)
        elif self._st_model is not None:
            q_emb = self._encode_text(context)
            emb = torch.FloatTensor(q_emb)
            para_sims = torch.zeros(NUM_PARAGRAPHS)
        else:
            q_emb = None
            # Hash question ONLY (not entire context) for a clean question representation
            q_text = question if question else (task_match.group(1) if task_match else context)
            emb = self._bow_hash(q_text, dim=512)
            # Direct word overlap (exactly what Greedy uses) — NOT noisy BoW cosine
            para_sims = torch.zeros(NUM_PARAGRAPHS)
            if paragraphs and q_words:
                for pi, (title, sents) in enumerate(
                        paragraphs[:NUM_PARAGRAPHS]):
                    if self.blind:
                        text = " ".join(sents)
                    else:
                        text = title + " " + " ".join(sents)
                    c_words = set(text.lower().split()) - _STOP_WORDS
                    if c_words:
                        # Overlap count normalised by question length — same signal as Greedy
                        para_sims[pi] = len(q_words & c_words) / max(1, len(q_words))

        # --- 32-dim structured features ---
        extra = torch.zeros(32)

        # [0-2] basic progress
        extra[0] = n_steps / 6.0
        extra[1] = n_read / NUM_PARAGRAPHS
        extra[2] = 1.0 if has_read else 0.0

        if self.blind:
            # ---- BLIND MODE: content-based features ----

            # [3-12] per-paragraph content↔question WORD overlap
            if paragraphs and q_words:
                for i, (_, sents) in enumerate(
                        paragraphs[:NUM_PARAGRAPHS]):
                    c_words = (set(" ".join(sents[:3]).lower().split())
                               - _STOP_WORDS)
                    if c_words:
                        extra[3 + i] = len(q_words & c_words) / len(q_words)

            # [13] question↔read-content word overlap
            if q_words and read_words:
                extra[13] = len(q_words & read_words) / len(q_words)

            # [14-17] step position flags
            for j in range(4):
                extra[14 + j] = float(n_steps >= j + 1)

            # [18] read content length
            if read_text:
                extra[18] = min(1.0, len(read_text) / 500.0)

            # [19-28] read_content↔paragraph_content similarity (bridge)
            # This is the key sequential signal: after reading paragraph A,
            # which unread paragraphs become more relevant?
            n_bridge = 0
            max_bridge = 0.0
            if has_read and clean_read.strip():
                if self._st_model is not None:
                    read_emb = self._encode_text(clean_read[:500])
                    for i in range(NUM_PARAGRAPHS):
                        is_already_read = (
                            i < len(raw_titles) and "[READ]" in raw_titles[i])
                        if para_embs[i] is not None:
                            sim = self._cosine(read_emb, para_embs[i])
                        else:
                            sim = 0.0
                        extra[19 + i] = sim
                        if not is_already_read and sim > 0.3:
                            n_bridge += 1
                            max_bridge = max(max_bridge, sim)
                elif paragraphs:
                    # Word overlap bridge: read_content↔paragraph_content
                    for i in range(min(NUM_PARAGRAPHS,
                                       len(paragraphs) if paragraphs else 0)):
                        is_already_read = (
                            i < len(raw_titles) and "[READ]" in raw_titles[i])
                        _, sents_i = paragraphs[i]
                        c_words_i = set(" ".join(sents_i).lower().split()) - _STOP_WORDS
                        if read_words and c_words_i:
                            sim = len(read_words & c_words_i) / max(1, len(c_words_i))
                        else:
                            sim = 0.0
                        extra[19 + i] = sim
                        if not is_already_read and sim > 0.1:
                            n_bridge += 1
                            max_bridge = max(max_bridge, sim)

            # [29] fraction of unread paras with high bridge similarity
            n_unread = max(1, NUM_PARAGRAPHS - n_read)
            extra[29] = n_bridge / n_unread

            # [30] max bridge similarity among unread
            extra[30] = max_bridge

            # [31] question-word coverage from all read content
            if q_words and read_words:
                extra[31] = len(q_words & read_words) / len(q_words)

        else:
            # ---- NORMAL MODE: title-based features (original) ----

            # [3-12] per-paragraph title↔question word overlap
            for i, t in enumerate(raw_titles[:NUM_PARAGRAPHS]):
                clean = t.replace("[READ] ", "").replace("[READ]", "").strip()
                t_words = set(clean.lower().split()) - _STOP_WORDS
                if q_words and t_words:
                    extra[3 + i] = len(q_words & t_words) / len(q_words)

            # [13] question↔read-content word overlap
            if q_words and read_words:
                extra[13] = len(q_words & read_words) / len(q_words)

            # [14-17] step position flags
            for j in range(4):
                extra[14 + j] = float(n_steps >= j + 1)

            # [18] read content length
            if read_text:
                extra[18] = min(1.0, len(read_text) / 500.0)

            # [19-28] read-content↔title overlap (bridge)
            n_bridge = 0
            max_bridge = 0.0
            for i, t in enumerate(raw_titles[:NUM_PARAGRAPHS]):
                is_already_read = "[READ]" in t
                clean = (t.replace("[READ] ", "")
                          .replace("[READ]", "").strip())
                t_words = set(clean.lower().split()) - _STOP_WORDS
                overlap = (len(read_words & t_words) / len(t_words)
                           if t_words and read_words else 0.0)
                extra[19 + i] = overlap
                if not is_already_read and overlap > 0:
                    n_bridge += 1
                    max_bridge = max(max_bridge, overlap)

            n_unread = max(1, NUM_PARAGRAPHS - n_read)
            extra[29] = n_bridge / n_unread
            extra[30] = max_bridge

            # [31] question-word coverage from all read info
            if q_words and (read_words or n_read > 0):
                all_info = set(read_words)
                for i, t in enumerate(raw_titles[:NUM_PARAGRAPHS]):
                    if "[READ]" in t:
                        clean = (t.replace("[READ] ", "")
                                  .replace("[READ]", "").strip())
                        all_info |= (set(clean.lower().split())
                                     - _STOP_WORDS)
                extra[31] = len(q_words & all_info) / len(q_words)

        feat = torch.cat([emb, para_sims, extra])

        # --- 60-dim new per-paragraph features ---
        new_para = torch.zeros(60)
        if paragraphs and q_words:
            n_paras = min(len(paragraphs), NUM_PARAGRAPHS)

            # Build IDF weights: log(N / df) for each question word
            # Words appearing in many paragraphs are less discriminative
            word_df: Dict[str, int] = defaultdict(int)
            para_word_sets: List[Set[str]] = []
            for pi in range(n_paras):
                _, sents_p = paragraphs[pi]
                ws = set(" ".join(sents_p).lower().split()) - _STOP_WORDS
                para_word_sets.append(ws)
                for w in ws:
                    word_df[w] += 1
            # Pad if fewer than NUM_PARAGRAPHS
            while len(para_word_sets) < NUM_PARAGRAPHS:
                para_word_sets.append(set())

            # [0-9] IDF-weighted overlap
            for pi in range(n_paras):
                idf_score = 0.0
                for w in q_words & para_word_sets[pi]:
                    df = word_df.get(w, 1)
                    idf_score += np.log(n_paras / df + 1)
                # Normalize by total possible IDF score
                max_idf = sum(np.log(n_paras / word_df.get(w, 1) + 1)
                              for w in q_words) if q_words else 1.0
                new_para[pi] = idf_score / max(1e-8, max_idf)

            # [10-19] Best single-sentence match
            for pi in range(n_paras):
                _, sents_p = paragraphs[pi]
                best_s = 0.0
                for sent in sents_p[:5]:  # check first 5 sentences
                    s_words = set(sent.lower().split()) - _STOP_WORDS
                    if s_words and q_words:
                        ov = len(q_words & s_words) / len(q_words)
                        best_s = max(best_s, ov)
                new_para[10 + pi] = best_s

            # [20-29] Paragraph length (normalized)
            max_len = max(len(" ".join(paragraphs[pi][1]).split())
                          for pi in range(n_paras)) if n_paras > 0 else 1
            for pi in range(n_paras):
                plen = len(" ".join(paragraphs[pi][1]).split())
                new_para[20 + pi] = plen / max(1, max_len)

            # [30-39] Unique word ratio (vocabulary richness)
            for pi in range(n_paras):
                all_words = " ".join(paragraphs[pi][1]).lower().split()
                n_total_w = len(all_words)
                n_unique = len(set(all_words) - _STOP_WORDS)
                new_para[30 + pi] = n_unique / max(1, n_total_w)

            # [40-49] Co-occurrence: max overlap with any OTHER paragraph
            for pi in range(n_paras):
                max_co = 0.0
                for pj in range(n_paras):
                    if pi == pj:
                        continue
                    if para_word_sets[pi] and para_word_sets[pj]:
                        co = len(para_word_sets[pi] & para_word_sets[pj])
                        co_norm = co / max(1, min(len(para_word_sets[pi]),
                                                   len(para_word_sets[pj])))
                        max_co = max(max_co, co_norm)
                new_para[40 + pi] = max_co

            # [50-59] Greedy rank position (normalized)
            overlap_scores = []
            for pi in range(n_paras):
                ov_count = len(q_words & para_word_sets[pi])
                overlap_scores.append((ov_count, pi))
            overlap_scores.sort(reverse=True)
            for rank, (_, pi) in enumerate(overlap_scores):
                new_para[50 + pi] = 1.0 - rank / max(1, n_paras - 1)

        feat = torch.cat([feat, new_para])
        feat = torch.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)
        return feat

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

    Priority ordering (P1 >> P2 >> P3):
      P1 — Read correct context:      +1.0 per gold paragraph read.
      P2 — Avoid redundant reads:      -0.05 per distractor + -0.03 step cost.
      P3 — Read in reasoning order:    +0.1 bonus when gold is read in order.

    Answer action: STOP_SCALE * recall + COMPLETION_BONUS * (recall==1).
    Linear recall provides gradient for partial progress.
    Low penalties encourage exploration over premature stopping.
    """

    REWARD_SUPPORTING = 1.0   # P1: gold read (high incentive to find)
    REWARD_DISTRACTOR = -0.2  # P2: distractor penalty (makes random reads negative EV with step)
    REWARD_ORDER_BONUS = 0.1  # P3: order bonus
    REWARD_BRIDGE_BONUS = 0.15 # P3: bonus for multi-hop bridge reading
    STEP_PENALTY = -0.08      # P2: per-step cost (penalise unnecessary reads)
    STOP_SCALE = 1.0          # answer reward base scale (scales F1)
    COMPLETION_BONUS = 1.5    # bonus for finding ALL golds

    def __init__(self):
        self.trajectories: List[TrajectoryWithReward] = []

    def collect(self, traj: AgentTrajectory, question: str,
                paragraphs: List[Tuple[str, List[str]]],
                supporting_titles: Set[str],
                supporting_indices_ordered: Optional[List[int]] = None,
                ) -> TrajectoryWithReward:
        """Convert an AgentTrajectory into reward-annotated decisions.

        If supporting_indices_ordered is provided (reasoning order of gold
        paragraph indices), adds a small bonus when the policy reads a gold
        paragraph in that order.
        """
        total_gold = max(1, len(supporting_titles))
        gold_recall = traj.num_supporting_read / total_gold
        is_correct = gold_recall >= 0.5

        decisions: List[RetrievalDecision] = []
        read_set: Set[int] = set()
        num_gold_read_so_far = 0
        n = min(len(paragraphs), NUM_PARAGRAPHS)

        # Build set of gold paragraph indices for F1 computation
        gold_titles_indices: Set[int] = set()
        for i in range(n):
            if paragraphs[i][0] in supporting_titles:
                gold_titles_indices.add(i)

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
                # F1-based answer reward: balances precision and recall
                n_reads = len(read_set)
                n_supp_read = len(read_set & gold_titles_indices)
                precision = n_supp_read / max(1, n_reads)
                recall = gold_recall
                f1 = 2 * precision * recall / max(1e-8, precision + recall) if (precision + recall) > 0 else 0.0
                reward = (self.STOP_SCALE * f1
                          + self.COMPLETION_BONUS * float(gold_recall >= 1.0 - 1e-6))
                action_idx = NUM_PARAGRAPHS  # answer index
            else:
                pidx = step.paragraph_idx
                action_idx = pidx
                if step.is_supporting:
                    # Gold read: F1 improves significantly
                    reward = self.REWARD_SUPPORTING
                    # Bonus for reading the next expected gold in order
                    if (supporting_indices_ordered
                            and num_gold_read_so_far < len(supporting_indices_ordered)
                            and pidx == supporting_indices_ordered[num_gold_read_so_far]):
                        reward += self.REWARD_ORDER_BONUS
                    num_gold_read_so_far += 1
                else:
                    reward = self.REWARD_DISTRACTOR
                reward += self.STEP_PENALTY  # encourage early stop

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
                mask=list(read_set) if read_set else None,
                question=question,
                paragraphs=paragraphs,
            ))

        twr = TrajectoryWithReward(
            task_id=traj.task_id,
            agent_id=traj.agent_id,
            task_description=question,
            decisions=decisions,
            final_answer=traj.final_answer,
            correct=is_correct,
            final_reward=gold_recall,
            num_supporting_read=traj.num_supporting_read,
            total_reads=traj.total_reads,
        )
        self.trajectories.append(twr)
        return twr


# ======================================================================
#  Base Fine-Tuner (shared functionality)
# ======================================================================

class BaseFineTuner:
    """Base class for all fine-tuning approaches (PPO, DPO, etc.)."""

    def __init__(self, scorer: TaskScorer = None, device: str = "cpu",
                 blind: bool = False, lr: float = 1e-5,
                 entropy_coeff: float = 0.01, kl_coeff: float = 0.2):
        self.scorer = scorer
        self.device = device
        self.blind = blind
        self.num_actions = NUM_ACTIONS
        self.action_names = [f"read_{i}" for i in range(NUM_PARAGRAPHS)] + ["answer"]

        # Build trainer first to detect sentence-transformer availability
        # Then set input_dim accordingly: 384+42+60=486 (st) or 512+42+60=614 (bow)
        _tmp_model = RetrievalSelector(input_dim=486, num_actions=self.num_actions)
        self.trainer = PPOTrainer(_tmp_model, lr=lr, entropy_coeff=entropy_coeff,
                                  device=device, blind=blind, kl_coeff=kl_coeff)
        has_st = self.trainer._st_model is not None
        self.input_dim = 486 if has_st else 614
        if not has_st:
            self.model = RetrievalSelector(input_dim=614, num_actions=self.num_actions)
            self.trainer = PPOTrainer(self.model, lr=lr, entropy_coeff=entropy_coeff,
                                      device=device, blind=blind, kl_coeff=kl_coeff)
        else:
            self.model = _tmp_model
            self.trainer.model = self.model

    # ---- action selection (called by RetrievalAgent.solve_with_policy) ----
    def select_action(self, context: str, read_set: Set[int] = None,
                      question: str = None,
                      paragraphs: List[Tuple[str, List[str]]] = None,
                      deterministic: bool = False) -> int:
        """Select next action (read_idx or answer) given current context.
        Args:
            context: state repr (task + titles + what's been read + step count)
            read_set: set of already-read paragraph indices
            question, paragraphs: for feature extraction
            deterministic: if True, use argmax; else sample from policy
        Returns:
            action index (0-9 for read_0..read_9, 10 for answer)
        """
        read_set = read_set or set()
        features = self.trainer.extract_features(
            context, question=question, paragraphs=paragraphs)
        with torch.no_grad():
            logits, _ = self.model(features.unsqueeze(0))
        logits = logits[0].clone()
        # Mask already-read paragraphs
        for idx in read_set:
            logits[idx] = -1e9
        if deterministic:
            return logits.argmax().item()
        else:
            probs = F.softmax(logits, dim=0)
            return torch.multinomial(probs, 1).item()

    # ---- evaluation ----
    def eval_retrieval(self, examples: Dict[str, Dict],
                       max_steps: int, label: str = "eval") -> Dict:
        """Fast eval: run policy on examples WITHOUT LLM, return retrieval metrics only.

        Uses training=True mode (no LLM answer generation) to evaluate how
        well the policy retrieves supporting paragraphs.  ~100x faster than
        eval_policy because no LLM inference is needed.

        Model is set to eval mode (argmax action selection, no dropout).
        """
        self.model.eval()
        total = 0
        total_reads = 0
        total_supp = 0
        total_gold = 0
        n_correct = 0  # recall >= 0.5
        for q_id, ex in examples.items():
            agent = RetrievalAgent(agent_id=0, model="qwen3:8b")
            traj = agent.solve_with_policy(
                q_id, ex["question"], ex["paragraphs"],
                ex["supporting_titles"], policy=self, max_steps=max_steps,
                training=True,
            )
            total += 1
            total_reads += traj.total_reads
            total_supp += traj.num_supporting_read
            n_gold = len(ex["supporting_titles"])
            total_gold += n_gold
            if n_gold > 0 and traj.num_supporting_read / n_gold >= 0.5:
                n_correct += 1
        recall = total_supp / max(1, total_gold)
        prec = total_supp / max(1, total_reads)
        f1 = 2 * prec * recall / max(1e-9, prec + recall)
        self.model.train()
        return {
            "strategy": label,
            "recall": recall,
            "precision": prec,
            "f1": f1,
            "retrieval_acc": n_correct / max(1, total),
            "avg_reads": total_reads / max(1, total),
            "avg_supporting_found": total_supp / max(1, total),
            "total": total,
        }

    def eval_policy(self, examples: Dict[str, Dict], scorer: TaskScorer,
                    max_steps: int, label: str = "model") -> Tuple[Dict, Dict[str, AgentTrajectory]]:
        """Run policy on examples with LLM (training=False), return (metrics_dict, trajs_dict)."""
        self.model.eval()
        trajs: Dict[str, AgentTrajectory] = {}
        correct = 0
        total = 0
        total_reads = 0
        total_supp = 0
        total_gold = 0
        for q_id, ex in examples.items():
            agent = RetrievalAgent(agent_id=0, model="qwen3:8b")
            traj = agent.solve_with_policy(
                q_id, ex["question"], ex["paragraphs"],
                ex["supporting_titles"], policy=self, max_steps=max_steps,
                training=False,
            )
            trajs[q_id] = traj
            score = scorer.score_answer(q_id, traj.final_answer or "")
            ok = score > 0.8
            total += 1
            if ok:
                correct += 1
            total_reads += traj.total_reads
            total_supp += traj.num_supporting_read
            total_gold += len(ex["supporting_titles"])
        acc = correct / max(1, total)
        avg_r = total_reads / max(1, total)
        avg_s = total_supp / max(1, total)
        prec = total_supp / max(1, total_reads)
        rec = total_supp / max(1, total_gold)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        metrics = {
            "strategy": label,
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "avg_reads": avg_r,
            "avg_supporting_found": avg_s,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }
        self.model.train()
        return metrics, trajs

    # ---- model persistence ----
    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=False))

    # ---- behavior cloning (expert warm start, used for both BC and SFT) ----
    def behavior_clone(self, examples: Dict[str, Dict],
                       max_steps: int = 5,
                       bc_epochs: int = 5,
                       batch_size: int = 16,
                       lr: float = 1e-3,
                       dev_examples: Optional[Dict[str, Dict]] = None,
                       patience: int = 3,
                       strategy: str = "greedy",
                       adaptive_k: bool = False) -> List[Dict]:
        """Train policy to imitate expert trajectories (no LLM).

        When strategy="oracle", trains on gold read order (SFT mode).
        When strategy="greedy", trains on greedy expert (BC mode).
        When adaptive_k=True, max_reads=min(num_gold, max_steps) per question.

        Returns list of per-epoch metrics dicts."""
        agent = RetrievalAgent(agent_id=0, model="qwen3:8b")
        pairs: List[Tuple[torch.Tensor, int, Optional[Set[int]]]] = []

        print(f"\n  [BC] Collecting {strategy} expert trajectories"
              + (" (adaptive K)" if adaptive_k else f" (K={max_steps})") + "...")
        for q_id, ex in examples.items():
            k = max_steps
            if adaptive_k:
                k = min(len(ex["supporting_titles"]), max_steps)
                k = max(k, 1)  # at least 1 read
            traj = agent.solve(
                q_id, ex["question"], ex["paragraphs"],
                ex["supporting_titles"], strategy=strategy,
                max_reads=k, training=True,
            )
            n = min(len(ex["paragraphs"]), NUM_PARAGRAPHS)
            read_set: Set[int] = set()

            for step in traj.steps:
                read_paras = [(ex["paragraphs"][i][0], ex["paragraphs"][i][1])
                              for i in sorted(read_set)]
                context = (
                    f"Task: {ex['question']}\n"
                    f"Titles: {' | '.join(('[READ] ' if i in read_set else '') + ex['paragraphs'][i][0] for i in range(n))}\n"
                    f"Read: {' '.join('[' + t + '] ' + ' '.join(s) for t, s in read_paras) if read_paras else 'Nothing yet'}\n"
                    f"Step: {len(read_set)}"
                )
                feat = self.trainer.extract_features(
                    context, question=ex["question"],
                    paragraphs=ex["paragraphs"][:NUM_PARAGRAPHS])
                expert_idx = step.paragraph_idx if step.paragraph_idx >= 0 else NUM_PARAGRAPHS
                pairs.append((feat, expert_idx, set(read_set)))

                if step.paragraph_idx >= 0 and step.paragraph_idx not in read_set:
                    read_set.add(step.paragraph_idx)

        if not pairs:
            print("  [BC] No expert pairs collected, skipping.")
            return []

        # Build dev pairs if dev_examples given
        dev_pairs: List[Tuple[torch.Tensor, int, Optional[Set[int]]]] = []
        if dev_examples:
            for q_id, ex in dev_examples.items():
                k = max_steps
                if adaptive_k:
                    k = min(len(ex["supporting_titles"]), max_steps)
                    k = max(k, 1)
                traj = agent.solve(
                    q_id, ex["question"], ex["paragraphs"],
                    ex["supporting_titles"], strategy=strategy,
                    max_reads=k, training=True,
                )
                n = min(len(ex["paragraphs"]), NUM_PARAGRAPHS)
                read_set_d: Set[int] = set()
                for step in traj.steps:
                    read_paras = [(ex["paragraphs"][i][0], ex["paragraphs"][i][1])
                                  for i in sorted(read_set_d)]
                    context = (
                        f"Task: {ex['question']}\n"
                        f"Titles: {' | '.join(('[READ] ' if i in read_set_d else '') + ex['paragraphs'][i][0] for i in range(n))}\n"
                        f"Read: {' '.join('[' + t + '] ' + ' '.join(s) for t, s in read_paras) if read_paras else 'Nothing yet'}\n"
                        f"Step: {len(read_set_d)}"
                    )
                    feat = self.trainer.extract_features(
                        context, question=ex["question"],
                        paragraphs=ex["paragraphs"][:NUM_PARAGRAPHS])
                    expert_idx = step.paragraph_idx if step.paragraph_idx >= 0 else NUM_PARAGRAPHS
                    dev_pairs.append((feat, expert_idx, set(read_set_d)))
                    if step.paragraph_idx >= 0 and step.paragraph_idx not in read_set_d:
                        read_set_d.add(step.paragraph_idx)

        print(f"  [BC] {len(pairs)} train pairs"
              + (f", {len(dev_pairs)} dev pairs" if dev_pairs else "")
              + f", training for up to {bc_epochs} epochs...")
        # Train full model — ctx pathway needed for answer_head to learn stopping
        opt = optim.Adam(self.model.parameters(), lr=lr)
        N = len(pairs)
        indices = list(range(N))

        bc_history: List[Dict] = []
        best_dev_loss = float('inf')
        best_state = None
        no_improve = 0

        for ep in range(bc_epochs):
            np.random.shuffle(indices)
            total_loss = 0.0
            nb = 0
            for s in range(0, N, batch_size):
                bi = indices[s:s + batch_size]
                feats = torch.stack([pairs[i][0] for i in bi])
                acts = torch.LongTensor([pairs[i][1] for i in bi])

                logits_raw, _ = self.model(feats)
                logits = logits_raw.clone()
                for j, mask in enumerate([pairs[i][2] for i in bi]):
                    if mask:
                        for idx in mask:
                            logits[j, idx] = -1e9

                loss = F.cross_entropy(logits, acts)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item()
                nb += 1

            train_loss = total_loss / max(1, nb)
            epoch_info = {"epoch": ep + 1, "train_loss": train_loss}

            # Dev loss
            if dev_pairs:
                self.model.eval()
                dev_loss_sum = 0.0
                dev_nb = 0
                with torch.no_grad():
                    for s in range(0, len(dev_pairs), batch_size):
                        bi_d = list(range(s, min(s + batch_size, len(dev_pairs))))
                        feats_d = torch.stack([dev_pairs[i][0] for i in bi_d])
                        acts_d = torch.LongTensor([dev_pairs[i][1] for i in bi_d])
                        logits_d, _ = self.model(feats_d)
                        logits_d = logits_d.clone()
                        for j, mask in enumerate([dev_pairs[i][2] for i in bi_d]):
                            if mask:
                                for idx in mask:
                                    logits_d[j, idx] = -1e9
                        dev_loss_sum += F.cross_entropy(logits_d, acts_d).item()
                        dev_nb += 1
                self.model.train()
                dev_loss = dev_loss_sum / max(1, dev_nb)
                epoch_info["dev_loss"] = dev_loss
                print(f"    BC epoch {ep+1}/{bc_epochs}  train_loss={train_loss:.4f}  dev_loss={dev_loss:.4f}")

                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience and ep >= 2:
                    print(f"    [BC] Early stopping at epoch {ep+1} (dev patience={patience})")
                    bc_history.append(epoch_info)
                    break
            else:
                print(f"    BC epoch {ep+1}/{bc_epochs}  loss={train_loss:.4f}")

            bc_history.append(epoch_info)

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"  [BC] Restored best dev checkpoint (dev_loss={best_dev_loss:.4f})")
        print("  [BC] Done.\n")
        return bc_history


# ======================================================================
#  PPO Fine-Tuner  (on-policy)
# ======================================================================


class PPOFineTuner(BaseFineTuner):
    """On-policy PPO training for paragraph retrieval selection."""

    def __init__(self, scorer: TaskScorer = None, device: str = "cpu",
                 blind: bool = False, lr: float = 1e-5,
                 entropy_coeff: float = 0.01, kl_coeff: float = 0.2):
        super().__init__(scorer=scorer, device=device, blind=blind, 
                         lr=lr, entropy_coeff=entropy_coeff, kl_coeff=kl_coeff)

        self.collector = DecisionCollector()
        self.training_history: List[Dict] = []
        self.all_train_trajectories: List[Dict] = []

        # Learned reward model for potential-based shaping
        self.reward_model = RewardModel(input_dim=self.input_dim, hidden_dim=64).to(device)
        self.reward_model_optimizer = optim.Adam(
            self.reward_model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.shaping_coeff = 0.0  # starts at 0; activated after first training
        self._reward_model_trained = False

    # ---- action selection (called by RetrievalAgent.solve_with_policy) ----
    def select_action(self, context: str, read_set: Set[int] = None,
                      question: str = None,
                      paragraphs: List[Tuple[str, List[str]]] = None):
        """Policy selects next action given context string.

        Already-read paragraph indices in read_set are masked to -inf
        so the policy can never waste a step re-reading.

        When model is in eval mode (model.eval()), uses argmax (greedy
        decoding).  In train mode, samples for exploration.

        Returns (action_name, action_idx, log_prob, value, features).
        """
        features = self.trainer.extract_features(
            context, question=question, paragraphs=paragraphs)
        with torch.no_grad():
            logits, value = self.model(features.unsqueeze(0))
        if read_set:
            for idx in read_set:
                logits[0, idx] = -1e9
        if torch.isnan(logits).any():
            logits = torch.zeros_like(logits)
        logits = logits.clamp(min=-30, max=30)
        probs = F.softmax(logits, dim=-1)
        probs = probs.clamp(min=1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        dist = torch.distributions.Categorical(probs)
        if self.model.training:
            action = dist.sample()
        else:
            action = logits[0].argmax().unsqueeze(0)
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
        all_masks: List[Optional[List[int]]] = []

        for twr in trajs:
            if not twr.decisions:
                continue
            t_feats, t_acts, t_rewards, t_values, t_lps = [], [], [], [], []
            t_masks: List[Optional[List[int]]] = []
            for d in twr.decisions:
                feat = self.trainer.extract_features(
                    d.context, question=d.question, paragraphs=d.paragraphs)
                with torch.no_grad():
                    logits, val = self.model(feat.unsqueeze(0))
                    if d.mask:
                        for mi in d.mask:
                            logits[0, mi] = -1e9
                    logits = logits.clamp(min=-30, max=30)
                    lp_probs = F.softmax(logits, dim=-1).clamp(min=1e-8)
                    lp_probs = lp_probs / lp_probs.sum(dim=-1, keepdim=True)
                    lp = lp_probs.log()[0, d.action_idx]
                t_feats.append(feat)
                t_acts.append(d.action_idx)
                t_rewards.append(d.reward)
                t_values.append(val.item())
                t_lps.append(lp.item())
                t_masks.append(d.mask)

            # Potential-based reward shaping: R' = R + γΦ(s') − Φ(s)
            if self._reward_model_trained and self.shaping_coeff > 0:
                with torch.no_grad():
                    phi = self.reward_model(torch.stack(t_feats)).tolist()
                gamma = self.trainer.gamma
                for t in range(len(t_rewards)):
                    phi_s = phi[t]
                    phi_s_next = phi[t + 1] if t + 1 < len(phi) else 0.0
                    t_rewards[t] += self.shaping_coeff * (gamma * phi_s_next - phi_s)

            advs, rets = self.trainer.compute_gae(t_rewards, t_values)
            all_features.extend(t_feats)
            all_actions.extend(t_acts)
            all_old_lp.extend(t_lps)
            all_old_values.extend(t_values)
            all_advantages.extend(advs)
            all_returns.extend(rets)
            all_masks.extend(t_masks)

        if not all_features:
            return {}

        feats_t = torch.stack(all_features)
        acts_t = torch.LongTensor(all_actions)
        old_lp_t = torch.FloatTensor(all_old_lp)
        old_val_t = torch.FloatTensor(all_old_values)
        adv_t = torch.FloatTensor(all_advantages)
        ret_t = torch.FloatTensor(all_returns)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        mask_t = torch.zeros(len(all_features), self.num_actions)
        for i, m_indices in enumerate(all_masks):
            if m_indices:
                for mi in m_indices:
                    mask_t[i, mi] = -1e9

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
                    action_masks=mask_t[bi],
                )
                for k, v in m.items():
                    epoch_m[k] += v
                nb += 1
            for k in epoch_m:
                epoch_m[k] /= max(1, nb)
            history.append(dict(epoch_m))
            self.training_history.append(dict(epoch_m))
        return {"epochs": num_epochs, "history": history}

    # ---- reward model training ----
    def train_reward_model(self, rm_epochs: int = 10,
                           batch_size: int = 32) -> Dict[str, float]:
        """Train the learned potential function Φ(s) on collected trajectories.

        For each trajectory, every state gets labeled with the trajectory's
        final recall.  The reward model learns: state → expected recall.
        This is then used for potential-based shaping in the next PPO iteration.
        """
        trajs = self.collector.trajectories
        if not trajs:
            return {"rm_loss": 0.0}

        # Build (state_features, final_recall) dataset from latest rollouts
        feats_list: List[torch.Tensor] = []
        targets_list: List[float] = []
        for twr in trajs:
            final_recall = twr.final_reward  # gold_recall ∈ [0, 1]
            for d in twr.decisions:
                feat = self.trainer.extract_features(
                    d.context, question=d.question, paragraphs=d.paragraphs)
                feats_list.append(feat)
                targets_list.append(final_recall)

        if not feats_list:
            return {"rm_loss": 0.0}

        feats_t = torch.stack(feats_list).to(self.device)
        targets_t = torch.FloatTensor(targets_list).to(self.device)

        self.reward_model.train()
        N = len(feats_list)
        total_loss = 0.0
        n_batches = 0
        for _ in range(rm_epochs):
            idx = torch.randperm(N)
            for s in range(0, N, batch_size):
                bi = idx[s:s + batch_size]
                pred = self.reward_model(feats_t[bi])
                loss = F.mse_loss(pred, targets_t[bi])
                self.reward_model_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.reward_model.parameters(), 1.0)
                self.reward_model_optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        self._reward_model_trained = True
        self.reward_model.eval()
        return {"rm_loss": avg_loss}

    # ---- behavior cloning (expert warm start) ----
    def behavior_clone(self, examples: Dict[str, Dict],
                       max_steps: int = 5,
                       bc_epochs: int = 5,
                       batch_size: int = 16,
                       lr: float = 1e-3,
                       dev_examples: Optional[Dict[str, Dict]] = None,
                       patience: int = 3,
                       strategy: str = "greedy",
                       adaptive_k: bool = False) -> List[Dict]:
        """Train policy to imitate expert trajectories (no LLM).

        When adaptive_k=True, max_reads=min(num_gold, max_steps) per question.
        This teaches greedy ranking + oracle-informed stopping.

        Returns list of per-epoch metrics dicts."""
        agent = RetrievalAgent(agent_id=0, model="qwen3:8b")
        pairs: List[Tuple[torch.Tensor, int, Optional[Set[int]]]] = []

        print(f"\n  [BC] Collecting {strategy} expert trajectories"
              + (" (adaptive K)" if adaptive_k else f" (K={max_steps})") + "...")
        for q_id, ex in examples.items():
            k = max_steps
            if adaptive_k:
                k = min(len(ex["supporting_titles"]), max_steps)
                k = max(k, 1)  # at least 1 read
            traj = agent.solve(
                q_id, ex["question"], ex["paragraphs"],
                ex["supporting_titles"], strategy=strategy,
                max_reads=k, training=True,
            )
            n = min(len(ex["paragraphs"]), NUM_PARAGRAPHS)
            read_set: Set[int] = set()

            for step in traj.steps:
                read_paras = [(ex["paragraphs"][i][0], ex["paragraphs"][i][1])
                              for i in sorted(read_set)]
                context = (
                    f"Task: {ex['question']}\n"
                    f"Titles: {' | '.join(('[READ] ' if i in read_set else '') + ex['paragraphs'][i][0] for i in range(n))}\n"
                    f"Read: {' '.join('[' + t + '] ' + ' '.join(s) for t, s in read_paras) if read_paras else 'Nothing yet'}\n"
                    f"Step: {len(read_set)}"
                )
                feat = self.trainer.extract_features(
                    context, question=ex["question"],
                    paragraphs=ex["paragraphs"][:NUM_PARAGRAPHS])
                expert_idx = step.paragraph_idx if step.paragraph_idx >= 0 else NUM_PARAGRAPHS
                pairs.append((feat, expert_idx, set(read_set)))

                if step.paragraph_idx >= 0 and step.paragraph_idx not in read_set:
                    read_set.add(step.paragraph_idx)

        if not pairs:
            print("  [BC] No expert pairs collected, skipping.")
            return []

        # Build dev pairs if dev_examples given
        dev_pairs: List[Tuple[torch.Tensor, int, Optional[Set[int]]]] = []
        if dev_examples:
            for q_id, ex in dev_examples.items():
                k = max_steps
                if adaptive_k:
                    k = min(len(ex["supporting_titles"]), max_steps)
                    k = max(k, 1)
                traj = agent.solve(
                    q_id, ex["question"], ex["paragraphs"],
                    ex["supporting_titles"], strategy=strategy,
                    max_reads=k, training=True,
                )
                n = min(len(ex["paragraphs"]), NUM_PARAGRAPHS)
                read_set_d: Set[int] = set()
                for step in traj.steps:
                    read_paras = [(ex["paragraphs"][i][0], ex["paragraphs"][i][1])
                                  for i in sorted(read_set_d)]
                    context = (
                        f"Task: {ex['question']}\n"
                        f"Titles: {' | '.join(('[READ] ' if i in read_set_d else '') + ex['paragraphs'][i][0] for i in range(n))}\n"
                        f"Read: {' '.join('[' + t + '] ' + ' '.join(s) for t, s in read_paras) if read_paras else 'Nothing yet'}\n"
                        f"Step: {len(read_set_d)}"
                    )
                    feat = self.trainer.extract_features(
                        context, question=ex["question"],
                        paragraphs=ex["paragraphs"][:NUM_PARAGRAPHS])
                    expert_idx = step.paragraph_idx if step.paragraph_idx >= 0 else NUM_PARAGRAPHS
                    dev_pairs.append((feat, expert_idx, set(read_set_d)))
                    if step.paragraph_idx >= 0 and step.paragraph_idx not in read_set_d:
                        read_set_d.add(step.paragraph_idx)

        print(f"  [BC] {len(pairs)} train pairs"
              + (f", {len(dev_pairs)} dev pairs" if dev_pairs else "")
              + f", training for up to {bc_epochs} epochs...")
        # Train full model — ctx pathway needed for answer_head to learn stopping
        opt = optim.Adam(self.model.parameters(), lr=lr)
        N = len(pairs)
        indices = list(range(N))

        bc_history: List[Dict] = []
        best_dev_loss = float('inf')
        best_state = None
        no_improve = 0

        for ep in range(bc_epochs):
            np.random.shuffle(indices)
            total_loss = 0.0
            nb = 0
            for s in range(0, N, batch_size):
                bi = indices[s:s + batch_size]
                feats = torch.stack([pairs[i][0] for i in bi])
                acts = torch.LongTensor([pairs[i][1] for i in bi])

                logits_raw, _ = self.model(feats)
                logits = logits_raw.clone()
                for j, mask in enumerate([pairs[i][2] for i in bi]):
                    if mask:
                        for idx in mask:
                            logits[j, idx] = -1e9

                loss = F.cross_entropy(logits, acts)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item()
                nb += 1

            train_loss = total_loss / max(1, nb)
            epoch_info = {"epoch": ep + 1, "train_loss": train_loss}

            # Dev loss
            if dev_pairs:
                self.model.eval()
                dev_loss_sum = 0.0
                dev_nb = 0
                with torch.no_grad():
                    for s in range(0, len(dev_pairs), batch_size):
                        bi_d = list(range(s, min(s + batch_size, len(dev_pairs))))
                        feats_d = torch.stack([dev_pairs[i][0] for i in bi_d])
                        acts_d = torch.LongTensor([dev_pairs[i][1] for i in bi_d])
                        logits_d, _ = self.model(feats_d)
                        logits_d = logits_d.clone()
                        for j, mask in enumerate([dev_pairs[i][2] for i in bi_d]):
                            if mask:
                                for idx in mask:
                                    logits_d[j, idx] = -1e9
                        dev_loss_sum += F.cross_entropy(logits_d, acts_d).item()
                        dev_nb += 1
                self.model.train()
                dev_loss = dev_loss_sum / max(1, dev_nb)
                epoch_info["dev_loss"] = dev_loss
                print(f"    BC epoch {ep+1}/{bc_epochs}  train_loss={train_loss:.4f}  dev_loss={dev_loss:.4f}")

                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience and ep >= 2:
                    print(f"    [BC] Early stopping at epoch {ep+1} (dev patience={patience})")
                    bc_history.append(epoch_info)
                    break
            else:
                print(f"    BC epoch {ep+1}/{bc_epochs}  loss={train_loss:.4f}")

            bc_history.append(epoch_info)

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"  [BC] Restored best dev checkpoint (dev_loss={best_dev_loss:.4f})")
        print("  [BC] Done.\n")
        return bc_history

    # ---- behavior cloning from oracle (ground-truth read order) ----
    def behavior_clone_oracle(self, examples: Dict[str, Dict],
                              max_steps: int = 5,
                              bc_epochs: int = 5,
                              batch_size: int = 16,
                              lr: float = 1e-3,
                              dev_examples: Optional[Dict[str, Dict]] = None,
                              patience: int = 3) -> List[Dict]:
        """Train policy to imitate oracle trajectories: read gold paragraphs in order, then answer.
        No LLM; (state, action) from ground-truth supporting_indices_ordered.
        Returns list of per-epoch metrics dicts."""
        pairs: List[Tuple[torch.Tensor, int, Optional[Set[int]]]] = []

        print("\n  [BC-Oracle] Building (state, action) from ground-truth read order...")
        n_para = NUM_PARAGRAPHS

        # Pre-warm ST cache: batch-encode all questions + paragraphs at once
        if self.trainer._st_model is not None:
            all_texts = []
            for ex in examples.values():
                all_texts.append(ex["question"])
                for title, sents in ex["paragraphs"][:n_para]:
                    if self.blind:
                        all_texts.append(" ".join(sents[:3]))
                    else:
                        all_texts.append(title + " " + " ".join(sents[:3]))
            if dev_examples:
                for ex in dev_examples.values():
                    all_texts.append(ex["question"])
                    for title, sents in ex["paragraphs"][:n_para]:
                        if self.blind:
                            all_texts.append(" ".join(sents[:3]))
                        else:
                            all_texts.append(title + " " + " ".join(sents[:3]))
            # Deduplicate before encoding
            unique_texts = list(set(t for t in all_texts if t not in self.trainer._st_cache))
            if unique_texts:
                print(f"  [BC-Oracle] Batch-encoding {len(unique_texts)} unique texts...")
                self.trainer._batch_encode(unique_texts)
                print(f"  [BC-Oracle] Cache warmed ({len(self.trainer._st_cache)} entries)")

        for q_id, ex in examples.items():
            paragraphs = ex["paragraphs"][:n_para]
            supp_titles = ex["supporting_titles"]
            # Ordered gold indices (reasoning order if available)
            ordered = ex.get("supporting_indices_ordered")
            if ordered is not None:
                gold_indices = [i for i in ordered if i < len(paragraphs)]
            else:
                gold_indices = sorted(
                    i for i in range(len(paragraphs))
                    if paragraphs[i][0] in supp_titles
                )
            if not gold_indices:
                continue
            # NOTE: some datasets can produce duplicate gold indices in the
            # provided "ordered" list; duplicates would make the expert action
            # illegal (already masked) and explode CE loss. Deduplicate in-order.
            seen: Set[int] = set()
            dedup_gold: List[int] = []
            for gi in gold_indices:
                if gi not in seen:
                    dedup_gold.append(gi)
                    seen.add(gi)
            read_order = dedup_gold[:max_steps]
            read_set: Set[int] = set()

            for step_idx, para_idx in enumerate(read_order):
                if para_idx in read_set:
                    # Shouldn't happen after dedup, but keep robust.
                    continue
                context = (
                    f"Task: {ex['question']}\n"
                    f"Titles: {' | '.join(('[READ] ' if i in read_set else '') + paragraphs[i][0] for i in range(len(paragraphs)))}\n"
                    f"Read: {' '.join('[' + paragraphs[i][0] + '] ' + ' '.join(paragraphs[i][1][:3]) for i in sorted(read_set)) if read_set else 'Nothing yet'}\n"
                    f"Step: {len(read_set)}"
                )
                feat = self.trainer.extract_features(
                    context, question=ex["question"],
                    paragraphs=ex["paragraphs"][:n_para])
                pairs.append((feat, para_idx, set(read_set)))
                read_set.add(para_idx)

            # Final step: answer (stop)
            context = (
                f"Task: {ex['question']}\n"
                f"Titles: {' | '.join(('[READ] ' if i in read_set else '') + paragraphs[i][0] for i in range(len(paragraphs)))}\n"
                f"Read: {' '.join('[' + paragraphs[i][0] + '] ' + ' '.join(paragraphs[i][1][:3]) for i in sorted(read_set))}\n"
                f"Step: {len(read_set)}"
            )
            feat = self.trainer.extract_features(
                context, question=ex["question"],
                paragraphs=ex["paragraphs"][:n_para])
            pairs.append((feat, NUM_PARAGRAPHS, set(read_set)))

        if not pairs:
            print("  [BC-Oracle] No pairs collected, skipping.")
            return []

        # Build dev pairs from dev_examples (oracle order)
        dev_pairs: List[Tuple[torch.Tensor, int, Optional[Set[int]]]] = []
        if dev_examples:
            for q_id, ex in dev_examples.items():
                paragraphs_d = ex["paragraphs"][:n_para]
                supp_titles_d = ex["supporting_titles"]
                ordered_d = ex.get("supporting_indices_ordered")
                if ordered_d is not None:
                    gold_d = [i for i in ordered_d if i < len(paragraphs_d)]
                else:
                    gold_d = sorted(i for i in range(len(paragraphs_d))
                                    if paragraphs_d[i][0] in supp_titles_d)
                if not gold_d:
                    continue
                seen_d: Set[int] = set()
                dedup_d: List[int] = []
                for gi in gold_d:
                    if gi not in seen_d:
                        dedup_d.append(gi)
                        seen_d.add(gi)
                read_order_d = dedup_d[:max_steps]
                read_set_d: Set[int] = set()
                for para_idx in read_order_d:
                    if para_idx in read_set_d:
                        continue
                    context = (
                        f"Task: {ex['question']}\n"
                        f"Titles: {' | '.join(('[READ] ' if i in read_set_d else '') + paragraphs_d[i][0] for i in range(len(paragraphs_d)))}\n"
                        f"Read: {' '.join('[' + paragraphs_d[i][0] + '] ' + ' '.join(paragraphs_d[i][1][:3]) for i in sorted(read_set_d)) if read_set_d else 'Nothing yet'}\n"
                        f"Step: {len(read_set_d)}"
                    )
                    feat = self.trainer.extract_features(
                        context, question=ex["question"],
                        paragraphs=ex["paragraphs"][:n_para])
                    dev_pairs.append((feat, para_idx, set(read_set_d)))
                    read_set_d.add(para_idx)
                # answer step
                context = (
                    f"Task: {ex['question']}\n"
                    f"Titles: {' | '.join(('[READ] ' if i in read_set_d else '') + paragraphs_d[i][0] for i in range(len(paragraphs_d)))}\n"
                    f"Read: {' '.join('[' + paragraphs_d[i][0] + '] ' + ' '.join(paragraphs_d[i][1][:3]) for i in sorted(read_set_d))}\n"
                    f"Step: {len(read_set_d)}"
                )
                feat = self.trainer.extract_features(
                    context, question=ex["question"],
                    paragraphs=ex["paragraphs"][:n_para])
                dev_pairs.append((feat, NUM_PARAGRAPHS, set(read_set_d)))

        # Sanity check: verify no expert action falls inside its mask
        n_violations = 0
        for feat, act, mask in pairs:
            if act in mask:
                n_violations += 1
        if n_violations:
            print(f"  [BC-Oracle] WARNING: {n_violations}/{len(pairs)} pairs "
                  f"have expert action inside mask!")

        print(f"  [BC-Oracle] {len(pairs)} train pairs"
              + (f", {len(dev_pairs)} dev pairs" if dev_pairs else "")
              + f", training for up to {bc_epochs} epochs...")
        opt = optim.Adam(self.model.parameters(), lr=lr)
        N = len(pairs)
        indices = list(range(N))
        bc_history: List[Dict] = []
        best_dev_loss = float('inf')
        best_state = None
        no_improve = 0
        for ep in range(bc_epochs):
            np.random.shuffle(indices)
            total_loss = 0.0
            nb = 0
            for s in range(0, N, batch_size):
                bi = indices[s:s + batch_size]
                feats = torch.stack([pairs[i][0] for i in bi])
                acts = torch.LongTensor([pairs[i][1] for i in bi])
                logits_raw, _ = self.model(feats)
                logits = logits_raw.clone()
                for j, mask in enumerate([pairs[i][2] for i in bi]):
                    if mask:
                        for idx in mask:
                            logits[j, idx] = -1e9
                loss = F.cross_entropy(logits, acts)
                if ep == 0 and nb == 0:
                    with torch.no_grad():
                        print(f"    [diag] feats: shape={feats.shape} "
                              f"min={feats.min():.3f} max={feats.max():.3f} "
                              f"nan={torch.isnan(feats).sum().item()}")
                        print(f"    [diag] logits_raw: min={logits_raw.min():.3f} "
                              f"max={logits_raw.max():.3f} "
                              f"nan={torch.isnan(logits_raw).sum().item()}")
                        print(f"    [diag] acts: {acts.tolist()}")
                        print(f"    [diag] loss (batch 0): {loss.item():.4f}")
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item()
                nb += 1
            train_loss = total_loss / max(1, nb)
            epoch_info = {"epoch": ep + 1, "train_loss": train_loss}

            # Dev loss
            if dev_pairs:
                self.model.eval()
                dev_loss_sum = 0.0
                dev_nb = 0
                with torch.no_grad():
                    for s in range(0, len(dev_pairs), batch_size):
                        bi_d = list(range(s, min(s + batch_size, len(dev_pairs))))
                        feats_d = torch.stack([dev_pairs[i][0] for i in bi_d])
                        acts_d = torch.LongTensor([dev_pairs[i][1] for i in bi_d])
                        logits_d, _ = self.model(feats_d)
                        logits_d = logits_d.clone()
                        for j, mask in enumerate([dev_pairs[i][2] for i in bi_d]):
                            if mask:
                                for idx in mask:
                                    logits_d[j, idx] = -1e9
                        dev_loss_sum += F.cross_entropy(logits_d, acts_d).item()
                        dev_nb += 1
                self.model.train()
                dev_loss = dev_loss_sum / max(1, dev_nb)
                epoch_info["dev_loss"] = dev_loss
                print(f"    BC-Oracle epoch {ep+1}/{bc_epochs}  train_loss={train_loss:.4f}  dev_loss={dev_loss:.4f}")

                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience and ep >= 2:
                    print(f"    [BC-Oracle] Early stopping at epoch {ep+1} (dev patience={patience})")
                    bc_history.append(epoch_info)
                    break
            else:
                print(f"    BC-Oracle epoch {ep+1}/{bc_epochs}  loss={train_loss:.4f}")

            bc_history.append(epoch_info)

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"  [BC-Oracle] Restored best dev checkpoint (dev_loss={best_dev_loss:.4f})")
        print("  [BC-Oracle] Done.\n")
        return bc_history

    def eval_retrieval(self, examples: Dict[str, Dict],
                       max_steps: int, label: str = "eval") -> Dict:
        """Fast eval: run policy on examples WITHOUT LLM, return retrieval metrics only.

        Uses training=True mode (no LLM answer generation) to evaluate how
        well the policy retrieves supporting paragraphs.  ~100x faster than
        eval_policy because no LLM inference is needed.

        Model is set to eval mode (argmax action selection, no dropout).
        """
        self.model.eval()
        total = 0
        total_reads = 0
        total_supp = 0
        total_gold = 0
        n_correct = 0  # recall >= 0.5
        for q_id, ex in examples.items():
            agent = RetrievalAgent(agent_id=0, model="qwen3:8b")
            traj = agent.solve_with_policy(
                q_id, ex["question"], ex["paragraphs"],
                ex["supporting_titles"], policy=self, max_steps=max_steps,
                training=True,
            )
            total += 1
            total_reads += traj.total_reads
            total_supp += traj.num_supporting_read
            n_gold = len(ex["supporting_titles"])
            total_gold += n_gold
            if n_gold > 0 and traj.num_supporting_read / n_gold >= 0.5:
                n_correct += 1
        recall = total_supp / max(1, total_gold)
        prec = total_supp / max(1, total_reads)
        f1 = 2 * prec * recall / max(1e-9, prec + recall)
        self.model.train()
        return {
            "strategy": label,
            "recall": recall,
            "precision": prec,
            "f1": f1,
            "retrieval_acc": n_correct / max(1, total),
            "avg_reads": total_reads / max(1, total),
            "avg_supporting_found": total_supp / max(1, total),
            "total": total,
        }

    def eval_policy(self, examples: Dict[str, Dict], scorer: TaskScorer,
                    max_steps: int, label: str = "BC-only") -> Tuple[Dict, Dict[str, AgentTrajectory]]:
        """Run policy on examples with LLM (training=False), return (metrics_dict, trajs_dict)."""
        self.model.eval()
        trajs: Dict[str, AgentTrajectory] = {}
        correct = 0
        total = 0
        total_reads = 0
        total_supp = 0
        total_gold = 0
        for q_id, ex in examples.items():
            agent = RetrievalAgent(agent_id=0, model="qwen3:8b")
            traj = agent.solve_with_policy(
                q_id, ex["question"], ex["paragraphs"],
                ex["supporting_titles"], policy=self, max_steps=max_steps,
                training=False,
            )
            trajs[q_id] = traj
            score = scorer.score_answer(q_id, traj.final_answer or "")
            ok = score > 0.8
            total += 1
            if ok:
                correct += 1
            total_reads += traj.total_reads
            total_supp += traj.num_supporting_read
            total_gold += len(ex["supporting_titles"])
        acc = correct / max(1, total)
        avg_r = total_reads / max(1, total)
        avg_s = total_supp / max(1, total)
        prec = total_supp / max(1, total_reads)
        rec = total_supp / max(1, total_gold)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        metrics = {
            "strategy": label,
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "avg_reads": avg_r,
            "avg_supporting_found": avg_s,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }
        self.model.train()
        return metrics, trajs

    # ---- on-policy training loop ----
    def on_policy_train(self, examples: Dict[str, Dict],
                        num_iterations: int = 8,
                        max_steps: int = 3,
                        ppo_epochs: int = 3,
                        batch_size: int = 16,
                        checkpoint_dir: str = "checkpoints",
                        resume_from: str = None,
                        patience: int = 3,
                        eval_examples: Optional[Dict[str, Dict]] = None,
                        bc_epochs: int = 3,
                        bc_expert: str = "greedy_st",
                        scorer: Optional[TaskScorer] = None,
                        bc_fraction: float = 1.0,
                        bc_max_reads: int = 3,
                        rollout_fraction: float = 0.25,
                        ) -> Tuple[List[Dict], Optional[Tuple[Dict, Dict[str, AgentTrajectory]]]]:
        """
        PPO training with BC warm start from Greedy-ST and early stopping.

        BC clones Greedy-ST(bc_max_reads) — the strongest non-RL baseline.
        A frozen copy is kept as reference for KL penalty during PPO.

        Each iteration sub-samples rollout_fraction of training examples
        (different random subset each time) to reduce overfitting.

        Early stopping: uses *eval retrieval F1* (lightweight, no LLM) if
        eval_examples given, else train rollout F1.

        Returns (all_metrics, bc_only_result, best_iter, best_metric).
        """
        all_metrics: List[Dict] = []
        start_iter = 0
        bc_only_result: Optional[Tuple[Dict, Dict[str, AgentTrajectory]]] = None

        if resume_from and os.path.isfile(resume_from):
            ckpt = self.load_checkpoint(resume_from)
            start_iter = ckpt["iteration"]
            all_metrics = ckpt.get("all_metrics", [])
            print(f"  Resuming training from iteration {start_iter + 1}")
        else:
            bc_examples = examples

            # BC warm start: clone greedy (best non-RL baseline)
            if bc_expert == "oracle":
                self.behavior_clone_oracle(
                    bc_examples, max_steps=bc_max_reads, bc_epochs=bc_epochs,
                    batch_size=batch_size, lr=1e-3,
                )
            else:
                self.behavior_clone(
                    bc_examples, max_steps=bc_max_reads, bc_epochs=bc_epochs,
                    batch_size=batch_size, lr=1e-3,
                    strategy=bc_expert,
                )
            # Snapshot BC policy as reference for KL penalty during PPO
            self.trainer.snapshot_reference()

            # BC-only baseline: eval current policy (no PPO yet) when scorer + eval_examples provided
            if eval_examples and scorer and len(eval_examples) > 0:
                print("\n  [BC-only baseline] Evaluating policy after BC (before PPO)...")
                bc_metrics, bc_trajs = self.eval_policy(
                    eval_examples, scorer, max_steps, label="BC-only"
                )
                bc_only_result = (bc_metrics, bc_trajs)
                print(f"  BC-only: acc={bc_metrics['accuracy']:.1%}  reads={bc_metrics['avg_reads']:.1f}  "
                      f"supp={bc_metrics['avg_supporting_found']:.1f}  R={bc_metrics['recall']:.1%}")

        os.makedirs(checkpoint_dir, exist_ok=True)

        base_lr = self.trainer.lr
        min_lr = base_lr * 0.3  # slower LR decay (floor at 30% of base)
        base_entropy = self.trainer.entropy_coeff

        best_metric = -1.0
        best_iter = start_iter
        no_improve = 0

        # Freeze para_scorer during PPO: preserve BC's learned paragraph
        # ranking.  PPO optimises ctx pathway (bridge detection, stopping,
        # alpha blending) which is where sequential decision-making lives.
        for p in self.model.para_scorer.parameters():
            p.requires_grad_(False)
        print("  [PPO] Frozen para_scorer (preserving BC ranking)")

        for it in range(start_iter, num_iterations):
            frac = 1.0 - it / max(1, num_iterations - 1)
            cur_lr = min_lr + (base_lr - min_lr) * frac
            self.trainer.set_lr(cur_lr)
            # Entropy schedule: warm up then decay
            warmup_frac = min(1.0, (it + 1) / max(1, num_iterations * 0.3))
            decay_frac = 1.0 - max(0, it - num_iterations * 0.3) / max(1, num_iterations * 0.7)
            self.trainer.entropy_coeff = base_entropy * min(warmup_frac, decay_frac)

            print(f"\n{'='*60}")
            print(f"On-Policy Iteration {it+1}/{num_iterations}  (lr={cur_lr:.2e}  ent={self.trainer.entropy_coeff:.4f})")
            print(f"{'='*60}")

            # Fresh collector (no LLM scorer needed — pure retrieval reward)
            self.collector = DecisionCollector()

            n_correct = 0
            n_total = 0
            n_supp = 0
            n_reads = 0
            n_gold = 0

            # Sub-sample a random subset per iteration
            all_qids = list(examples.keys())
            n_rollout = max(batch_size, int(len(all_qids) * max(0.5, rollout_fraction)))
            iter_qids = list(np.random.choice(all_qids, size=min(n_rollout, len(all_qids)), replace=False))

            for q_id in iter_qids:
                ex = examples[q_id]
                question = ex["question"]
                paragraphs = ex["paragraphs"]
                supp_titles = ex["supporting_titles"]

                agent = RetrievalAgent(agent_id=0, model="qwen3:8b")
                traj = agent.solve_with_policy(
                    q_id, question, paragraphs, supp_titles,
                    policy=self, max_steps=max_steps,
                    training=True,
                )
                supp_indices_ordered = ex.get("supporting_indices_ordered")
                twr = self.collector.collect(
                    traj, question, paragraphs, supp_titles,
                    supporting_indices_ordered=supp_indices_ordered,
                )

                n_total += 1
                if twr.correct:
                    n_correct += 1
                n_supp += twr.num_supporting_read
                n_reads += twr.total_reads
                n_gold += len(supp_titles)

                tag = "✓" if twr.correct else "✗"
                gold_n = len(supp_titles)
                print(f"  {tag} [{q_id}] reads={twr.total_reads} "
                      f"supp={twr.num_supporting_read}/{gold_n} "
                      f"recall={twr.final_reward:.0%}")

            acc = n_correct / max(1, n_total)
            avg_supp = n_supp / max(1, n_total)
            avg_reads = n_reads / max(1, n_total)
            prec = n_supp / max(1, n_reads)
            recall = n_supp / max(1, n_gold)
            print(f"\n  acc={acc:.1%}  reads={avg_reads:.1f}  supp={avg_supp:.1f}  "
                  f"P={prec:.1%}  R={recall:.1%}")

            train_m = self._ppo_update(
                num_epochs=4, batch_size=batch_size, ppo_epochs=1,
            )

            # Train reward model on this iteration's trajectories,
            # then enable shaping starting from iteration 3
            rm_info = self.train_reward_model(rm_epochs=10, batch_size=32)
            if not self._reward_model_trained or it < 2:
                self.shaping_coeff = 0.0
            else:
                # Ramp up shaping coefficient over iterations (start at iter 3)
                self.shaping_coeff = min(0.2, 0.1 * (it - 1))

            # Adaptive KL: keep KL from BC in [0.1, 0.4] sweet spot
            last_hist = (train_m.get("history", [{}]) or [{}])[-1]
            current_kl = last_hist.get("kl_from_bc", 0.0)
            if current_kl > 0.4:
                self.trainer.kl_coeff = min(0.3, self.trainer.kl_coeff * 2.0)
            elif current_kl < 0.1:
                self.trainer.kl_coeff = max(0.005, self.trainer.kl_coeff * 0.8)
            print(f"  Reward model: loss={rm_info['rm_loss']:.4f}  "
                  f"shaping_coeff={self.shaping_coeff:.2f}  "
                  f"kl={current_kl:.4f}  kl_coeff={self.trainer.kl_coeff:.4f}")

            # Per-iteration reward/return (for report)
            returns = [sum(d.reward for d in twr.decisions)
                      for twr in self.collector.trajectories]
            mean_return = sum(returns) / len(returns) if returns else 0.0
            total_steps = sum(len(twr.decisions) for twr in self.collector.trajectories)
            total_reward = sum(sum(d.reward for d in twr.decisions)
                              for twr in self.collector.trajectories)
            avg_step_reward = total_reward / total_steps if total_steps else 0.0

            all_metrics.append({
                "iteration": it + 1,
                "accuracy": acc,
                "correct": n_correct,
                "total": n_total,
                "avg_reads": avg_reads,
                "avg_supporting_found": avg_supp,
                "precision": prec,
                "recall": recall,
                "num_decisions": len(self.collector.trajectories),
                "mean_return": mean_return,
                "avg_step_reward": avg_step_reward,
                "training": train_m,
                "reward_model": rm_info,
                "shaping_coeff": self.shaping_coeff,
                "kl_coeff": self.trainer.kl_coeff,
            })
            # Persist full per-iteration trajectories for JSON logging
            iter_trajs: List[Dict] = []
            for twr in self.collector.trajectories:
                iter_trajs.append({
                    "task_id": twr.task_id,
                    "task_description": twr.task_description,
                    "final_answer": twr.final_answer,
                    "correct": twr.correct,
                    "final_reward": twr.final_reward,
                    "num_supporting_read": twr.num_supporting_read,
                    "total_reads": twr.total_reads,
                    "decisions": [
                        {
                            "step": d.step_id,
                            "action": d.action_name,
                            "reward": round(d.reward, 3),
                        }
                        for d in twr.decisions
                    ],
                })
            self.all_train_trajectories.append({
                "iteration": it + 1,
                "summary": all_metrics[-1],
                "trajectories": iter_trajs,
            })

            ckpt_path = os.path.join(
                checkpoint_dir, f"ckpt_iter_{it+1:03d}.pt")
            self.save_checkpoint(ckpt_path, it + 1, all_metrics)

            # ---- early stopping: lightweight retrieval-only eval on eval set (no LLM) ----
            f1_train = 2 * prec * recall / max(1e-9, prec + recall)
            stop_metric = f1_train
            if eval_examples and len(eval_examples) > 0:
                eval_ret = self.eval_retrieval(
                    eval_examples, max_steps, label=f"PPO-iter{it+1} (eval)"
                )
                stop_metric = eval_ret["f1"]
                print(f"  eval: R={eval_ret['recall']:.1%} P={eval_ret['precision']:.1%} "
                      f"F1={eval_ret['f1']:.1%} reads={eval_ret['avg_reads']:.1f} "
                      f"(early-stop on F1)")
                all_metrics[-1]["eval_retrieval"] = eval_ret

            if stop_metric > best_metric:
                best_metric = stop_metric
                best_iter = it + 1
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience and it >= 2:
                best_ckpt = os.path.join(
                    checkpoint_dir, f"ckpt_iter_{best_iter:03d}.pt")
                signal_name = "eval F1" if eval_examples else "train F1"
                print(f"\n  Early stopping: {signal_name} "
                      f"did not improve for {patience} iterations.")
                print(f"  Reloading best checkpoint (iter {best_iter}, "
                      f"metric={best_metric:.1%}): {best_ckpt}")
                if os.path.isfile(best_ckpt):
                    self.load_checkpoint(best_ckpt)
                break

        # If loop completed without early stop, use best checkpoint (not last iter)
        last_iter = num_iterations
        if best_iter != last_iter:
            best_ckpt = os.path.join(
                checkpoint_dir, f"ckpt_iter_{best_iter:03d}.pt")
            if os.path.isfile(best_ckpt):
                print(f"\n  Using best checkpoint (iter {best_iter}, metric={best_metric:.1%})")
                self.load_checkpoint(best_ckpt)

        return all_metrics, bc_only_result, best_iter, best_metric

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

    # ---- checkpointing ----
    def save_checkpoint(self, path: str, iteration: int,
                        all_metrics: List[Dict] = None):
        """Save full training state so training can be resumed."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "iteration": iteration,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.trainer.optimizer.state_dict(),
            "reward_model_state_dict": self.reward_model.state_dict(),
            "reward_model_trained": self._reward_model_trained,
            "shaping_coeff": self.shaping_coeff,
            "training_history": self.training_history,
            "all_train_trajectories": self.all_train_trajectories,
            "all_metrics": all_metrics or [],
        }, path)
        print(f"  Checkpoint saved: {path}  (iteration {iteration})")

    def load_checkpoint(self, path: str) -> Dict:
        """Load training state from checkpoint. Returns checkpoint dict."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "reward_model_state_dict" in ckpt:
            self.reward_model.load_state_dict(ckpt["reward_model_state_dict"])
            self._reward_model_trained = ckpt.get("reward_model_trained", False)
            self.shaping_coeff = ckpt.get("shaping_coeff", 0.0)
        self.training_history = ckpt.get("training_history", [])
        self.all_train_trajectories = ckpt.get("all_train_trajectories", [])
        print(f"  Resumed from checkpoint: {path}  (iteration {ckpt['iteration']})")
        return ckpt

    # ---- persistence ----
    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=False))

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
            "train_trajectories": self.all_train_trajectories,
        }
        with open(path, "w") as f:
            json.dump(results, f, indent=2)


# ======================================================================
#  DPO Fine-Tuner  (Direct Preference Optimization)
# ======================================================================

class DPOFineTuner(BaseFineTuner):
    """Direct Preference Optimization (SFT → DPO pipeline)."""

    def __init__(self, scorer: TaskScorer = None, device: str = "cpu",
                 blind: bool = False, lr: float = 1e-5,
                 entropy_coeff: float = 0.01, kl_coeff: float = 0.2):
        super().__init__(scorer=scorer, device=device, blind=blind, 
                         lr=lr, entropy_coeff=entropy_coeff, kl_coeff=kl_coeff)

    # ---- action selection (called by RetrievalAgent.solve_with_policy) ----
    def select_action(self, context: str, read_set: Set[int] = None,
                      question: str = None,
                      paragraphs: List[Tuple[str, List[str]]] = None):
        """Policy selects next action given context string (compatible with PPO interface).

        Already-read paragraph indices in read_set are masked to -inf
        so the policy can never waste a step re-reading.

        When model is in eval mode (model.eval()), uses argmax (greedy
        decoding).  In train mode, samples for exploration.

        Returns (action_name, action_idx, log_prob, value, features).
        Note: value is dummy (0.0) for DPO since no value head is trained.
        """
        features = self.trainer.extract_features(
            context, question=question, paragraphs=paragraphs)
        with torch.no_grad():
            logits, _ = self.model(features.unsqueeze(0))
        if read_set:
            for idx in read_set:
                logits[0, idx] = -1e9
        if torch.isnan(logits).any():
            logits = torch.zeros_like(logits)
        logits = logits.clamp(min=-30, max=30)
        probs = F.softmax(logits, dim=-1)
        probs = probs.clamp(min=1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        dist = torch.distributions.Categorical(probs)
        if self.model.training:
            action = dist.sample()
        else:
            action = logits[0].argmax().unsqueeze(0)
        lp = dist.log_prob(action)
        idx = action.item()
        name = self.action_names[idx] if idx < len(self.action_names) else "answer"
        return name, idx, lp.item(), 0.0, features  # value=0.0 (dummy for DPO)

    # ---- direct preference optimization (DPO) ----
    def collect_preference_pairs(self, examples: Dict[str, Dict],
                                  max_steps: int = 5) -> List[Tuple[torch.Tensor, torch.Tensor, int, int, Optional[Set[int]], Optional[Set[int]]]]:
        """Collect preference pairs: oracle trajectories (winning) vs model's own generated trajectories (losing).
        
        Classic DPO setup:
          - Winning trajectory: oracle read order (reading gold paras in sequence) — ground truth
          - Losing trajectory: model's own trajectory (using current learned policy)
        
        The model learns to prefer its own oracle trajectories over its current suboptimal policy.
        
        Returns: list of (feat_oracle, feat_policy, act_oracle, act_policy, mask_oracle, mask_policy) tuples
        """
        agent = RetrievalAgent(agent_id=0, model="qwen3:8b")
        pairs = []
        
        print(f"\n  [DPO] Collecting preference pairs: oracle (win) vs model policy (lose)...")
        
        for q_id, ex in examples.items():
            # Win trajectory: oracle (ground truth)
            traj_oracle = agent.solve(
                q_id, ex["question"], ex["paragraphs"],
                ex["supporting_titles"], strategy="oracle",
                max_reads=max_steps, training=True,
            )
            
            # Lose trajectory: model's own generated trajectory
            traj_policy = agent.solve_with_policy(
                q_id, ex["question"], ex["paragraphs"],
                ex["supporting_titles"], policy=self,
                max_steps=max_steps, training=True,
            )
            
            n = min(len(ex["paragraphs"]), NUM_PARAGRAPHS)
            
            # For each step in the trajectories, create a preference pair
            max_steps_seq = max(len(traj_oracle.steps), len(traj_policy.steps))
            
            for step_idx in range(max_steps_seq):
                read_set_oracle: Set[int] = set()
                read_set_policy: Set[int] = set()
                
                # Collect actions up to this step
                for s in range(step_idx):
                    if s < len(traj_oracle.steps):
                        step_o = traj_oracle.steps[s]
                        if step_o.paragraph_idx >= 0:
                            read_set_oracle.add(step_o.paragraph_idx)
                    if s < len(traj_policy.steps):
                        step_p = traj_policy.steps[s]
                        if step_p.paragraph_idx >= 0:
                            read_set_policy.add(step_p.paragraph_idx)
                
                # Build context for oracle trajectory
                read_paras_oracle = [(ex["paragraphs"][i][0], ex["paragraphs"][i][1])
                                     for i in sorted(read_set_oracle)]
                context_oracle = (
                    f"Task: {ex['question']}\n"
                    f"Titles: {' | '.join(('[READ] ' if i in read_set_oracle else '') + ex['paragraphs'][i][0] for i in range(n))}\n"
                    f"Read: {' '.join('[' + t + '] ' + ' '.join(s) for t, s in read_paras_oracle) if read_paras_oracle else 'Nothing yet'}\n"
                    f"Step: {len(read_set_oracle)}"
                )
                
                # Build context for policy trajectory
                read_paras_policy = [(ex["paragraphs"][i][0], ex["paragraphs"][i][1])
                                     for i in sorted(read_set_policy)]
                context_policy = (
                    f"Task: {ex['question']}\n"
                    f"Titles: {' | '.join(('[READ] ' if i in read_set_policy else '') + ex['paragraphs'][i][0] for i in range(n))}\n"
                    f"Read: {' '.join('[' + t + '] ' + ' '.join(s) for t, s in read_paras_policy) if read_paras_policy else 'Nothing yet'}\n"
                    f"Step: {len(read_set_policy)}"
                )
                
                # Extract features
                feat_oracle = self.trainer.extract_features(
                    context_oracle, question=ex["question"],
                    paragraphs=ex["paragraphs"][:NUM_PARAGRAPHS])
                feat_policy = self.trainer.extract_features(
                    context_policy, question=ex["question"],
                    paragraphs=ex["paragraphs"][:NUM_PARAGRAPHS])
                
                # Get actions
                act_oracle = traj_oracle.steps[step_idx].paragraph_idx if step_idx < len(traj_oracle.steps) else NUM_PARAGRAPHS
                act_policy = traj_policy.steps[step_idx].paragraph_idx if step_idx < len(traj_policy.steps) else NUM_PARAGRAPHS
                
                if act_oracle < 0:
                    act_oracle = NUM_PARAGRAPHS
                if act_policy < 0:
                    act_policy = NUM_PARAGRAPHS
                
                pairs.append((feat_oracle, feat_policy, act_oracle, act_policy, set(read_set_oracle), set(read_set_policy)))
        
        print(f"  [DPO] Collected {len(pairs)} preference pairs (oracle > policy)\n")
        return pairs

    def train_dpo(self, pairs: List[Tuple[torch.Tensor, torch.Tensor, int, int, Optional[Set[int]], Optional[Set[int]]]],
                  epochs: int = 5,
                  batch_size: int = 16,
                  lr: float = 1e-4,
                  beta: float = 0.5,
                  dev_pairs: Optional[List] = None,
                  patience: int = 3) -> List[Dict]:
        """Train policy using Direct Preference Optimization (DPO).
        
        DPO directly optimizes:
            log σ(β * (log π(win_action|win_state) - log π(lose_action|lose_state)))
        
        This encourages the model to assign higher probability to actions in
        winning trajectories and lower probability to actions in losing trajectories.
        """
        if not pairs:
            print("  [DPO] No preference pairs, skipping training.")
            return []
        
        print(f"\n  [DPO] Training on {len(pairs)} preference pairs for up to {epochs} epochs...")
        opt = optim.Adam(self.model.parameters(), lr=lr)
        
        dpo_history: List[Dict] = []
        best_dev_loss = float('inf')
        best_state = None
        no_improve = 0
        
        N = len(pairs)
        indices = list(range(N))
        
        for ep in range(epochs):
            np.random.shuffle(indices)
            total_loss = 0.0
            nb = 0
            
            for s in range(0, N, batch_size):
                bi = indices[s:s + batch_size]
                
                # Extract batch
                feats_win = torch.stack([pairs[i][0] for i in bi])
                feats_lose = torch.stack([pairs[i][1] for i in bi])
                acts_win = torch.LongTensor([pairs[i][2] for i in bi])
                acts_lose = torch.LongTensor([pairs[i][3] for i in bi])
                masks_win = [pairs[i][4] for i in bi]
                masks_lose = [pairs[i][5] for i in bi]
                
                # Get logits
                logits_win, _ = self.model(feats_win)
                logits_lose, _ = self.model(feats_lose)
                
                # Apply masks (mark unavailable actions as -inf)
                for j, mask in enumerate(masks_win):
                    if mask:
                        for idx in mask:
                            logits_win[j, idx] = -1e9
                for j, mask in enumerate(masks_lose):
                    if mask:
                        for idx in mask:
                            logits_lose[j, idx] = -1e9
                
                # Compute log probabilities
                log_probs_win = F.log_softmax(logits_win, dim=-1)
                log_probs_lose = F.log_softmax(logits_lose, dim=-1)
                
                # Get log probs for chosen actions
                log_prob_chosen_win = log_probs_win.gather(1, acts_win.unsqueeze(1)).squeeze(1)
                log_prob_chosen_lose = log_probs_lose.gather(1, acts_lose.unsqueeze(1)).squeeze(1)
                
                # DPO loss: log σ(β * (log_win - log_lose))
                # where σ is sigmoid
                ratio = beta * (log_prob_chosen_win - log_prob_chosen_lose)
                dpo_loss = -F.logsigmoid(ratio).mean()
                
                opt.zero_grad()
                dpo_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                
                total_loss += dpo_loss.item()
                nb += 1
            
            train_loss = total_loss / max(1, nb)
            epoch_info = {"epoch": ep + 1, "train_loss": train_loss}
            
            # Dev loss
            if dev_pairs:
                self.model.eval()
                dev_loss_sum = 0.0
                dev_nb = 0
                with torch.no_grad():
                    for s in range(0, len(dev_pairs), batch_size):
                        bi_d = list(range(s, min(s + batch_size, len(dev_pairs))))
                        feats_win_d = torch.stack([dev_pairs[i][0] for i in bi_d])
                        feats_lose_d = torch.stack([dev_pairs[i][1] for i in bi_d])
                        acts_win_d = torch.LongTensor([dev_pairs[i][2] for i in bi_d])
                        acts_lose_d = torch.LongTensor([dev_pairs[i][3] for i in bi_d])
                        masks_win_d = [dev_pairs[i][4] for i in bi_d]
                        masks_lose_d = [dev_pairs[i][5] for i in bi_d]
                        
                        logits_win_d, _ = self.model(feats_win_d)
                        logits_lose_d, _ = self.model(feats_lose_d)
                        
                        for j, mask in enumerate(masks_win_d):
                            if mask:
                                for idx in mask:
                                    logits_win_d[j, idx] = -1e9
                        for j, mask in enumerate(masks_lose_d):
                            if mask:
                                for idx in mask:
                                    logits_lose_d[j, idx] = -1e9
                        
                        log_probs_win_d = F.log_softmax(logits_win_d, dim=-1)
                        log_probs_lose_d = F.log_softmax(logits_lose_d, dim=-1)
                        
                        log_prob_chosen_win_d = log_probs_win_d.gather(1, acts_win_d.unsqueeze(1)).squeeze(1)
                        log_prob_chosen_lose_d = log_probs_lose_d.gather(1, acts_lose_d.unsqueeze(1)).squeeze(1)
                        
                        ratio_d = beta * (log_prob_chosen_win_d - log_prob_chosen_lose_d)
                        dpo_loss_d = -F.logsigmoid(ratio_d).mean()
                        
                        dev_loss_sum += dpo_loss_d.item()
                        dev_nb += 1
                
                self.model.train()
                dev_loss = dev_loss_sum / max(1, dev_nb)
                epoch_info["dev_loss"] = dev_loss
                print(f"    DPO epoch {ep+1}/{epochs}  train_loss={train_loss:.4f}  dev_loss={dev_loss:.4f}")
                
                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience and ep >= 2:
                    print(f"    [DPO] Early stopping at epoch {ep+1} (dev patience={patience})")
                    dpo_history.append(epoch_info)
                    break
            else:
                print(f"    DPO epoch {ep+1}/{epochs}  loss={train_loss:.4f}")
            
            dpo_history.append(epoch_info)
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"  [DPO] Restored best dev checkpoint (dev_loss={best_dev_loss:.4f})")
        print("  [DPO] Done.\n")
        return dpo_history

