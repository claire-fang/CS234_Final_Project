# CS234 Final Project: PPO-Guided Paragraph Retrieval for Multi-Hop QA

## Overview

This project trains a lightweight RL policy (PPO) to select which **context paragraphs** to read before answering multi-hop questions from [HotpotQA](https://hotpotqa.github.io/) (distractor setting).

Each HotpotQA question comes with **10 paragraphs** (2 gold supporting + 8 distractors). The agent must learn to read the right paragraphs and skip distractors, then generate an answer using only the selected context.

## MDP Formulation

| Element | Definition |
|---|---|
| **State** $s_t$ | 544-dim vector: 512-dim BoW hash of (question + read history) + 32 structured features (step count, per-paragraph title-question overlap, read flags, etc.) |
| **Action** $a_t$ | One of 11 discrete actions: `read_0` ... `read_9` (read a paragraph) or `answer` (stop and answer) |
| **Transition** | Read the selected paragraph, append to context, re-extract features |
| **Reward** $r_t$ | **Dense**: +0.3 (supporting paragraph), -0.1 (distractor), -0.2 (re-read), +1.0/0.0 (correct/wrong final answer) |
| **Episode** | One question, up to 10 steps |

## Pipeline

### 1. Data Loading

Load 40 questions from HotpotQA (train split, distractor setting). Split into 30 training + 10 evaluation. Each question includes 10 context paragraphs and ground-truth supporting fact titles.

### 2. Baselines

Five retrieval strategies are evaluated, all using the same LLM (Qwen3:8b) for answer generation:

| Strategy | Description |
|---|---|
| **Oracle** | Read only the 2 gold supporting paragraphs (upper bound) |
| **No Context** | Answer with no paragraphs (lower bound) |
| **Random(k=3)** | Randomly select 3 paragraphs |
| **Greedy(k=3)** | Select 3 paragraphs with highest title-question word overlap |
| **All Context** | Read all 10 paragraphs |

### 3. PPO Training

A small residual MLP (`RetrievalSelector`, ~70K params) is trained via PPO to select paragraphs sequentially:

```
Input (544) -> LayerNorm -> Linear(544->128) + ReLU + Residual
            -> Linear(128->128) + ReLU
            |-- Policy head -> 11 action logits
            |-- Value head  -> V(s)
```

Training runs **8 on-policy iterations**: each iteration collects trajectories on the 30 training questions using the current policy, computes GAE advantages (gamma=0.99, lambda=0.95), and updates via clipped PPO (epsilon=0.2).

The dense reward signal gives the policy immediate feedback on each paragraph choice, unlike sparse final-answer-only rewards.

### 4. Evaluation & Report

The trained policy is evaluated on the 10 held-out questions. A detailed report compares all strategies by accuracy, average paragraphs read, and per-question results.

## Project Structure

```
hotpot_pipeline.py       # Main entry: data loading, baselines, PPO, evaluation, reporting
multi_agent_baseline.py  # RetrievalAgent: paragraph selection + LLM answer generation
ppo_finetuner.py         # RetrievalSelector (MLP), PPOTrainer, DecisionCollector, TaskScorer
run_modal.py             # Deploy on Modal cloud (A10G GPU, auto Ollama setup)
requirements.txt         # Dependencies
results/                 # Outputs (model, metrics, trajectories, report)
checkpoints/
```

## Quick Start

### Run on Modal (Recommended)

```bash
pip install modal
modal setup
modal run run_modal.py
```

### Run Locally

```bash
pip install torch numpy requests datasets

# Start Ollama with required models
ollama serve &
ollama pull qwen3:8b
ollama pull qwen3:14b

python hotpot_pipeline.py
```
