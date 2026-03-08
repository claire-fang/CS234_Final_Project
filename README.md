# PPO for Adaptive Paragraph Retrieval in Multi-Hop QA

**CS234 Final Project** — Stanford University

## Overview

This project applies **Proximal Policy Optimization (PPO)** to the problem of *adaptive paragraph retrieval* for multi-hop question answering. Given a question and a pool of 10 candidate paragraphs, a small learned policy decides — one step at a time — which paragraph to read next and **when to stop**. After the policy finishes reading, a large language model answers the question using only the selected paragraphs.

The core claim: **PPO can achieve competitive accuracy while reading fewer irrelevant documents**, because it learns to match its reading budget to question complexity — stopping early for simple questions and reading more for hard ones.

---

## Repository Structure

```
hotpot_pipeline.py        # Main pipeline: data loading, pre-filter, baselines, PPO training, reporting
ppo_finetuner.py          # Policy network, PPO trainer, reward, feature extraction
multi_agent_baseline.py   # RetrievalAgent: fixed-strategy baselines + PPO execution
run_modal.py              # Modal cloud deployment (GPU execution)
results/                  # Output: report.txt, model weights, JSON logs
```

---

## Step 1: Environment Setup

### 1.1 Install Python Dependencies

```bash
pip install torch numpy requests datasets transformers
```

### 1.2 Install and Start Ollama

1. Install Ollama: https://ollama.com  
2. Pull the model:
   ```bash
   ollama pull qwen3:8b
   ```
3. Ensure Ollama is running on `http://localhost:11434` (default).

---

## Step 2: Dataset

### 2.1 Supported Datasets

The pipeline supports **two datasets**:

| Dataset | HuggingFace ID | Description |
|---------|----------------|-------------|
| **2WikiMultiHopQA** | `framolfese/2WikiMultihopQA` | Multi-hop QA with chain reasoning (A → B) |
| **HotpotQA** | `hotpot_qa` (distractor) | 10 paragraphs: 2 gold + 8 distractors |

### 2.2 Default: 2WikiMultiHopQA

- **Type filter**: By default loads only `compositional` questions — these require sequential reading (e.g., "What nationality is the director of Film X?" → read Film X → discover director → read director paragraph).
- **Fallback**: If fewer than half of `max_examples` are found with the type filter, the loader retries without the filter (loads all types).

### 2.3 Paragraph Pool Format

Each question is paired with exactly **10 candidate paragraphs**: `(title, [sentences])`. Paragraphs are padded or truncated to 10. For 2Wiki, supporting facts come from `supporting_facts`; for HotpotQA, from `supporting_facts.title`.

### 2.4 Data Scale (per run mode)

| Mode | N_CANDIDATES | N_TARGET (after pre-filter) | N_EVAL | N_ITER (PPO) |
|------|--------------|-----------------------------|--------|--------------|
| Normal | 200 | 40 | 10 | 3 |
| `--small` | 30 | 10 | 3 | 2 |

---

## Step 3: Pipeline Execution Flow

The main pipeline (`hotpot_pipeline.py`) runs **6 sequential steps**:

### Step [1/6] Load Candidate Data

- **2Wiki mode** (default): `load_2wiki_data(split="train", max_examples=N_CANDIDATES, type_filter="compositional")`
- **Hotpot mode** (`--hotpot`): `load_hotpot_data(split="train", max_examples=N_CANDIDATES)`

### Step [2/6] Pre-filter (No-Context Check)

- For each candidate question, run the LLM with **no context**.
- **Discard** questions the LLM answers correctly (retrieval is unnecessary).
- **Keep** questions until `target_count` (N_TARGET) is reached.
- Ensures every remaining question genuinely needs retrieval.

### Step [3/6] Run Baselines

Evaluate fixed-strategy baselines on the **eval set** (last N_EVAL questions):

| Strategy | Label | Reads | Description |
|----------|-------|-------|-------------|
| `oracle` | Oracle | gold only | Upper bound: reads exactly the gold supporting paragraphs |
| `no_context` | No Context | 0 | Lower bound: LLM answers from memory only |
| `random` | Random (K) | K | Read K randomly chosen paragraphs (K = K_BUDGET = 3) |

### Step [4/6] PPO Training

- Train the policy on the **train set** (all examples except the last N_EVAL).
- Each iteration: collect fresh trajectories → assign dense rewards → PPO update.
- Budget: `K_BUDGET = 3` max reads per question.

### Step [5/6] PPO Evaluation

- Evaluate the trained PPO policy on the eval set with the same budget (3 max steps).

### Step [6/6] Static Top-K Ablation

- **Non-sequential baseline**: same PPO model, but all reads decided at once from the **initial state** (step 0, nothing read).
- Ranks paragraphs by action logits from step 0; takes top-K.
- Shows that sequential state updates (bridge-entity features) are necessary.

---

## Step 4: Models

### 4.1 Large Language Model (LLM)

**Model:** `qwen3:8b`, served via [Ollama](https://ollama.com) at `http://localhost:11434`.

**Roles:**

1. **Answer generation** — given question + selected paragraphs, generate a short factual answer.
2. **Judge** — when exact/containment matching fails, the LLM decides if prediction matches ground truth (CORRECT / INCORRECT). Results are cached.

The LLM is **never fine-tuned**; used only at inference.

### 4.2 Small Policy Network (RetrievalSelector)

- **Architecture**: Residual MLP, ~50K parameters.
- **Input**: 544-dim features (512 BoW hash + 32 structured).
- **Output**: 11 action logits + 1 value estimate.

#### State (544 dims)

- **512-dim BoW hash**: All words in context hashed into 512 bins, L1-normalized.
- **32-dim structured features**:
  - Step count, #[READ] fraction, has-read flag
  - Per-paragraph title–question overlap (dims 3–12)
  - Read-content–question overlap (dim 13)
  - Step position flags (dims 14–17)
  - Read content length (dim 18)
  - **Bridge-entity features** (dims 19–31): overlap between read content and each paragraph title — creates sequential dependency (reading A reveals which paragraphs link to A).

#### Action Space (11 actions)

| Action | Index | Meaning |
|--------|-------|---------|
| `read_0` … `read_9` | 0–9 | Read paragraph i |
| `answer` | 10 | Stop and answer now |

---

## Step 5: Training (PPO)

### 5.1 Reward Function (Dense, Per-Step)

| Event | Reward |
|-------|--------|
| Read supporting paragraph (first time) | **+0.3** |
| Read distractor paragraph (first time) | **−0.1** |
| Re-read already-read paragraph | **−0.2** |
| Answer correctly | **+1.0** |
| Answer incorrectly | **0.0** |

### 5.2 Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| `lr` | 3e-4 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_ratio` | 0.2 |
| `entropy_coeff` | 0.05 |
| `target_kl` | 0.02 |
| `max_grad_norm` | 0.5 |
| `batch_size` | 16 |
| `num_epochs` (per iteration) | 2 |
| `ppo_epochs` (per mini-batch) | 3 |
| `K_BUDGET` (max_steps) | 3 |

### 5.3 Training Loop

- **Iteration**: collect trajectories on all train examples → PPO update.
- **Epochs**: 2 sweeps over collected data per iteration.
- **PPO epochs**: up to 3 per mini-batch, early-stopped by KL.

---

## Step 6: Comparison Methods

All methods evaluated on the same eval set. Metrics:

- **Accuracy**: fraction correct (cascade scorer)
- **Avg reads**: average paragraphs read per question
- **Precision**: supporting_read / total_reads
- **Recall**: supporting_read / total_gold
- **F1**: harmonic mean of precision and recall

| Method | Reads | Strategy |
|--------|-------|----------|
| Oracle | gold only | Upper bound |
| No Context | 0 | Lower bound |
| Random (3) | 3 (fixed) | 3 random paragraphs |
| Static Top-3 | 3 (fixed) | Top-3 by initial-state logits (non-sequential) |
| PPO (ours) | 1–3 (adaptive) | Learned sequential policy |

---

## Step 7: Running the Pipeline

### 7.1 Local Run

```bash
# Full run (2Wiki, ~40 train, 10 eval)
python hotpot_pipeline.py

# Quick test (reduced scale)
python hotpot_pipeline.py --small

# Use HotpotQA instead of 2Wiki
python hotpot_pipeline.py --hotpot
```

### 7.2 Cloud Run (Modal GPU)

```bash
pip install modal
modal setup
modal run run_modal.py
```

Optional flags: `--small`, `--dataset 2wiki` or `--dataset hotpot`.

### 7.3 Outputs

| File | Contents |
|------|----------|
| `report.txt` | Full report: summary table, per-question detail, PPO training curve |
| `hotpot_tool_selector.pt` | PPO policy weights |
| `trajectories.json` | Per-question trajectories from last PPO iteration |
| `training_results.json` | Training metrics + baseline summary |
| `comparison.json` | Full metrics for all methods |

---

## Step 8: Scorer

Answer correctness uses a three-stage cascade:

1. **Exact match**: normalize (lowercase, remove punctuation, articles) → if prediction = ground truth → 1.0.
2. **Containment**: if ground truth is substring of prediction → 1.0.
3. **LLM judge**: call `qwen3:8b` with CORRECT/INCORRECT prompt; cache result.

A question is correct if score > 0.8.

---

## Quick Reference: Pipeline Steps Summary

| Step | Action |
|------|--------|
| 1 | Load data (2Wiki or Hotpot) |
| 2 | Pre-filter: drop questions LLM can answer without context |
| 3 | Run baselines: Oracle, No Context, Random(3) |
| 4 | PPO train on train set |
| 5 | PPO eval on eval set |
| 6 | Static Top-3 ablation (same model, non-sequential) |
| — | Save report, model, trajectories, comparison JSON |
