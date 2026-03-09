# PPO for Adaptive Paragraph Retrieval (Blind Mode)

**CS234 Final Project** — Stanford University

## Overview

Multi-hop QA requires selecting which paragraphs to read before asking an LLM for an answer. This project trains a **small PPO policy** (~200K params) to decide **which paragraphs to read** and **when to stop**. We run in **blind mode**: paragraph titles are anonymized to `Para_0` … `Para_9`, so all methods rely on **paragraph content only** (no title leakage). That makes the comparison fair and highlights whether the RL policy adds value beyond a simple similarity ranker.

**Why blind?** With real titles, a greedy baseline that ranks by title–question word overlap can look strong because titles leak information. In blind mode, **Greedy (BoW)** degrades to content word overlap, **Greedy-ST** ranks by sentence-transformer cosine similarity (same signal as PPO’s main features), and **PPO** gets the same similarity signal plus **sequential state** (what’s been read, bridge similarity to unread paragraphs) and **adaptive stopping**. If PPO beats Greedy-ST, the gain comes from the learned policy, not from a better feature.

**Core claim:** PPO learns an adaptive reading budget and ordering from content-based rewards (including order bonus when supporting facts are ordered), and can outperform fixed-budget baselines (Random, Greedy BoW, Greedy-ST) in blind mode.

---

## Repository Structure

```
hotpot_pipeline.py        # Pipeline: data, prefilter, split, baselines, PPO, report (blind-only doc here)
ppo_finetuner.py          # Policy, PPO trainer, reward (order-aware), features (blind: content-only)
multi_agent_baseline.py   # RetrievalAgent: baselines + PPO rollout; Greedy-ST strategy
run_modal.py              # Modal deployment (--blind, --resume)
results/                  # report.txt, model weights, JSON logs
checkpoints_blind/        # Blind run: split.json, baselines.json, ckpt_iter_*.pt
```

---

## Step 1: Environment

```bash
pip install torch numpy requests datasets transformers sentence-transformers
```

Install [Ollama](https://ollama.com), then:

```bash
ollama pull qwen3:8b
```

Ollama must be running at `http://localhost:11434`.

---

## Step 2: Data (Blind Mode)

- **Dataset:** 2WikiMultiHopQA (`framolfese/2WikiMultihopQA`), types `compositional` and `bridge_comparison` (mixed gold counts: 2 or 4).
- **Scale (full):** 400 candidates → prefilter → 120 kept → train/eval split → **then titles anonymized** to `Para_0`…`Para_9`. Train ≈100, Eval 20. `--small`: 30 candidates, 10 kept, 3 eval.
- **Order:** Supporting facts are **ordered** in the dataset (reasoning order). We store `supporting_indices_ordered` and use it in the reward (order bonus when the policy reads gold in that order).
- **Prefilter:** Shuffle candidates; use LLM no-context check to keep ~60% of target as “hard”; fill the rest from the remaining pool so eval has a mix of gold=2 and gold=4.
- **Split:** Random train/eval after prefilter. **Saved to `checkpoints_blind/split.json`** so resume does not re-split or re-prefilter.

---

## Step 3: Pipeline (Blind Only)

All commands below assume **blind mode** (`--blind`).

### [1/6] Load data

Load 2Wiki (or Hotpot with `--hotpot`), shuffle, prefilter, split, then **anonymize titles** and write `checkpoints_blind/split.json`. On **resume**, this step is skipped and the saved split is loaded.

### [2/6] (Skipped on resume)

Prefilter is done inside step 1 when not resuming.

### [3/6] Baselines (Blind Only)

Run on the **eval set** (same 20 questions, blind titles). Results are written to `checkpoints_blind/baselines.json` and reused on resume.  
In this final version we keep a **small, focused set** of baselines:

| Strategy      | Reads | Description |
|---------------|-------|-------------|
| Oracle        | gold only | Upper bound on retrieval (reads all supporting paragraphs only) |
| No Context    | 0     | Lower bound (answer with zero paragraphs) |
| Random (2–4)  | 2 / 3 / 4 | K random paragraphs under the same budget as PPO |
| BC-only (oracle BC) | adaptive ≤5 | Behavior-cloned from **oracle trajectories** (gold read order), no PPO updates |
| PPO (ours)    | adaptive ≤5 | PPO fine-tuned policy, initialized from BC-only. Reported metrics come directly from PPO on the eval set (no fallback to BC metrics). |

### [4/6] BC + PPO (training)

**BC warm start (oracle-based):**

- For runs with `--bc-oracle` (our main setting), BC clones the **oracle retrieval policy**:
  - Oracle trajectories: read all gold paragraphs in dataset order, then answer.
  - BC trains on all train examples with `bc_max_reads = 3` and `bc_epochs = 20`.
  - After BC, we save a checkpoint `ckpt_iter_000.pt` (pure BC) and also run a **BC-only eval** on the eval set (`BC-only` row in the table).

**PPO fine-tuning (blind, from BC):**

- **Input (blind):** 384-dim question embedding + 10-dim **content-only** paragraph–question cosine similarity + 32-dim structured (step count, read count, content word overlap, **read-content ↔ paragraph-content ST similarity** as a bridge signal).
- **Reward (dense, no LLM):**
  - Gold read: **+0.3** per supporting paragraph.
  - Distractor read: **−0.1**.
  - Order bonus: **+0.1** when reading gold in the dataset’s supporting-fact order.
  - Answer (stop): **recall of gold paragraphs** (0.0–1.0), scaled by `STOP_SCALE = 1.0`. This is the main “when to stop” signal.
- **Budget:** `K_BUDGET = 5`. Action masking prevents re-reading.
- **BC reference & KL:** After BC we snapshot a frozen copy of the BC policy and add a **KL penalty** during PPO to keep the policy close to BC (prevents catastrophic forgetting).
- **Optimiser:** Small MLP (~30K params, hidden 64, strong dropout), Adam with weight decay, very small LR (e.g. `1e-5`) with linear decay, clip ratio 0.1, low entropy (0.01).
- **Early stopping:** Patience 3 on **eval retrieval F1** (no LLM), and we always keep / reload the best checkpoint (can be BC-only, iter 0, if PPO does not improve).

### [5/6] PPO evaluation

Eval set, same budget; LLM used only here for answer generation and scoring.

### [6] Report

- Summary table (all strategies), per-category breakdown (e.g. compositional 2-gold vs bridge_comparison 4-gold), per-question detail, **PPO training curve** (accuracy, reads, precision, recall), **reward/return per iteration**, and loss curve when available.

---

## Step 4: How to Run (Blind Only)

### Local

```bash
# Full blind run (2Wiki, 120 kept, 20 eval, PPO up to 10 iters)
python hotpot_pipeline.py --blind

# Quick test
python hotpot_pipeline.py --blind --small
```

### Modal (GPU)

```bash
pip install modal
modal setup
modal run run_modal.py --blind
```

### Resume (no re-split, no re-baselines)

After the first run, **data split** and **baseline results** are in `checkpoints_blind/`. To continue PPO from the latest checkpoint:

**Local:**

```bash
python hotpot_pipeline.py --blind --resume
```

(Or point to a specific checkpoint: `--resume=checkpoints_blind/ckpt_iter_003.pt`.)

**Modal:**

```bash
modal run run_modal.py --blind --resume
```

Modal restores `checkpoints_blind/` from the volume `cs234_checkpoints_blind`, then the pipeline loads `split.json` and `baselines.json` and resumes PPO from the latest `ckpt_iter_*.pt`. Data and baselines are **not** re-run.

---

## Step 5: Outputs

| Output | Description |
|--------|-------------|
| `results/report.txt` | Full report (summary, per-category, per-question, PPO curve, reward/return, loss) |
| `results/hotpot_tool_selector.pt` | Final PPO weights |
| `results/trajectories.json` | PPO trajectories (eval) |
| `results/training_results.json` | Training metrics and per-iteration trajectories |
| `results/comparison.json` | All methods’ metrics |
| `checkpoints_blind/split.json` | Train/eval split (reused on resume) |
| `checkpoints_blind/baselines.json` | Baseline metrics + trajectories (reused on resume) |
| `checkpoints_blind/ckpt_iter_*.pt` | PPO checkpoints (used by --resume) |

---

## Quick Reference (Blind Mode, Final Version)

| Step | What happens |
|------|----------------------|
| 1 | Load 2Wiki (mixed types) → prefilter → split → anonymize titles → save `checkpoints_blind/split.json` |
| 3 | Run baselines (Oracle, No Context, Random 2/3/4) → save `checkpoints_blind/baselines.json` |
| 4 | BC from oracle trajectories (if `--bc-oracle`) → snapshot BC as iter 0 → PPO train with KL-to-BC on train set → save `checkpoints_blind/ckpt_iter_*.pt` |
| 5 | PPO eval on 20 questions (LLM for answers) |
| -- | Write report (with reward/return curve, loss, and a genuine PPO vs BC-only comparison — no metric fallback) |
| **Resume** | Load split + baselines from `checkpoints_blind/`, resume PPO from latest checkpoint; no re-data, no re-baselines |
