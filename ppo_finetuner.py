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
    """Scores task outcomes based on correctness."""
    
    def __init__(self):
        """Initialize scorer with task patterns."""
        self.task_patterns = {}
    
    def register_ground_truth(self, task_id: str, correct_answer: str):
        """Register ground truth for a task."""
        self.task_patterns[task_id] = correct_answer
    
    def score_answer(self, task_id: str, answer: str) -> float:
        """Score an answer (0-1). Simple string matching for demo."""
        if task_id not in self.task_patterns:
            return 0.0  # Unknown task
        
        correct = self.task_patterns[task_id]
        
        # Exact match
        if answer.lower().strip() == correct.lower().strip():
            return 1.0
        
        # Partial match (contains key parts)
        correct_words = set(correct.lower().split())
        answer_words = set(answer.lower().split())
        
        if not correct_words:
            return 0.0
        
        overlap = len(correct_words & answer_words) / len(correct_words)
        return min(1.0, overlap + 0.1)  # Allow some partial credit


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


class PPOTrainer:
    """PPO-style trainer for tool selection."""
    
    def __init__(self, model: nn.Module, learning_rate: float = 1e-4, 
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2, entropy_coeff: float = 0.01,
                 device: str = "cpu"):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.device = device
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
        """Extract simple features from context text (CPU-friendly)."""
        # Simple bag-of-words with hashing
        words = context.lower().split()
        feature_vec = np.zeros(512)
        
        for word in words:
            idx = hash(word) % 512
            feature_vec[idx] += 1.0
        
        # Normalize
        if feature_vec.sum() > 0:
            feature_vec /= feature_vec.sum()
        
        return torch.FloatTensor(feature_vec)


class DecisionCollector:
    """Collects tool selection decisions from agent trajectories."""
    
    def __init__(self, scorer: TaskScorer, success_reward: float = 1.0, 
                 failure_penalty: float = -0.5):
        """
        Initialize collector with reward/penalty configuration.
        
        Args:
            scorer: Task scorer for evaluating answers
            success_reward: Reward for decisions in successful trajectories (default 1.0)
            failure_penalty: Penalty for decisions in failed trajectories (default -0.5)
        """
        self.scorer = scorer
        self.success_reward = success_reward
        self.failure_penalty = failure_penalty
        self.trajectories_with_rewards: List[TrajectoryWithReward] = []
    
    def collect_from_trajectory(self, trajectory, task_description: str, 
                               final_answer: Optional[str]) -> TrajectoryWithReward:
        """
        Convert an agent trajectory to reward-annotated decisions.
        
        Successful trajectories (answer correct) get success_reward.
        Failed trajectories (answer incorrect) get failure_penalty.
        """
        # Score the final answer
        answer_score = self.scorer.score_answer(trajectory.task_id, final_answer or "")
        
        # Determine reward: success or penalty
        if answer_score > 0.8:
            # Successful trajectory
            trajectory_reward = self.success_reward
            is_correct = True
        else:
            # Failed trajectory
            trajectory_reward = self.failure_penalty
            is_correct = False
        
        decisions = []
        
        for step in trajectory.steps:
            # Reconstruct context from previous steps
            context = f"Previous steps:\n"
            for prev_step in trajectory.steps[:step.step_id]:
                context += f"  - {prev_step.tool_called}: {prev_step.tool_result.content[:50]}\n"
            
            decision = ToolSelectionDecision(
                task_id=trajectory.task_id,
                agent_id=trajectory.agent_id,
                step_id=step.step_id,
                task_description=task_description,
                context=context,
                chosen_tool=step.tool_called,
                reward=trajectory_reward,  # Success or penalty
            )
            decisions.append(decision)
        
        traj_with_reward = TrajectoryWithReward(
            task_id=trajectory.task_id,
            agent_id=trajectory.agent_id,
            task_description=task_description,
            decisions=decisions,
            final_answer=final_answer,
            correct=is_correct,
            final_reward=trajectory_reward,
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
    """Main class for PPO fine-tuning of tool selection."""
    
    def __init__(self, scorer: TaskScorer, device: str = "cpu",
                 success_reward: float = 1.0, failure_penalty: float = -0.5):
        """
        Initialize PPO fine-tuner.
        
        Args:
            scorer: Task scorer for evaluating answers
            device: Device to use ("cpu" or "cuda")
            success_reward: Reward for decisions in successful trajectories
            failure_penalty: Penalty for decisions in failed trajectories
        """
        self.scorer = scorer
        self.device = device
        self.collector = DecisionCollector(
            scorer,
            success_reward=success_reward,
            failure_penalty=failure_penalty
        )
        
        # Initialize model and trainer
        self.model = SimpleToolSelector(num_tools=5)
        self.trainer = PPOTrainer(self.model, device=device)
        
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
        correct_count = sum(1 for d in decisions if d.reward > 0.8)
        print(f"  {len(decisions)} decisions, {correct_count} from correct trajectories")
    
    def fine_tune(self, num_epochs: int = 3, batch_size: int = 8, 
                  learning_rate: float = 1e-4) -> Dict[str, Any]:
        """
        Fine-tune model on collected decisions.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        
        Returns:
            Training metrics
        """
        decisions = self.collector.get_all_decisions()
        
        if not decisions:
            print("No decisions collected. Run collect_trajectories first.")
            return {}
        
        print(f"\nFine-tuning on {len(decisions)} decisions ({num_epochs} epochs)...")
        
        metrics_history = []
        
        for epoch in range(num_epochs):
            epoch_metrics = defaultdict(float)
            
            # Mini-batch training
            for i in range(0, len(decisions), batch_size):
                batch_decisions = decisions[i:i+batch_size]
                
                # Extract features and actions
                context_features = torch.stack([
                    self.trainer.extract_features(d.context) for d in batch_decisions
                ])
                action_indices = torch.LongTensor([
                    ["no_tool", "web_search", "web_fetch", "extract", "calculator"].index(d.chosen_tool)
                    for d in batch_decisions
                ])
                rewards = torch.FloatTensor([d.reward for d in batch_decisions])
                
                # Forward pass to get log probs
                with torch.no_grad():
                    logits, values = self.model(context_features)
                    log_probs = F.log_softmax(logits, dim=-1)
                    old_log_probs = log_probs.gather(1, action_indices.unsqueeze(-1)).squeeze(-1)
                
                # Compute advantages (simplified: use rewards directly)
                advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
                returns = rewards
                
                # Training step
                batch = {
                    "context_features": context_features,
                    "action_indices": action_indices,
                }
                
                step_metrics = self.trainer.train_step(
                    batch, old_log_probs, advantages, returns, num_epochs=1
                )
                
                for key, val in step_metrics.items():
                    epoch_metrics[key] += val
            
            # Average over batches
            num_batches = max(1, len(decisions) // batch_size)
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches
            
            metrics_history.append(dict(epoch_metrics))
            print(f"  Epoch {epoch+1}/{num_epochs}: {dict(epoch_metrics)}")
        
        self.training_history = metrics_history
        return {"epochs": num_epochs, "history": metrics_history}
    
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
