# CS234 Final Project: PPO-Guided Paragraph Retrieval for Multi-Hop QA

## What This Project Does

We train a **small MLP** (~70K parameters) with **PPO** to decide which paragraphs to feed to a **large LLM** (Qwen3 8B) for answering multi-hop questions.

The two models **never directly communicate**. Python code acts as the middleman:

```
                              Python Control Loop
                     ┌─────────────────────────────────┐
                     │                                 │
  ┌──────────┐       │   544-dim       action          │
  │ Small MLP │ ◄────┼── feature  ──►  "read_3"       │
  │  (70K)    │      │   vector        or "answer"     │
  └──────────┘       │                    │            │
                     │          ┌─────────┘            │
                     │          ▼                      │
                     │   if read_3:                    │
                     │     append paragraph 3 to list  │
                     │                                 │
                     │   if answer:                    │
                     │     concat all read paragraphs  │
                     │     into a prompt               │
                     │          │                      │
                     │          ▼                      │
  ┌──────────┐       │   "Question: ...               │
  │ Qwen 8B  │ ◄────┼──  Paragraphs: ...             │
  │  (LLM)   │      │    Answer:"                     │
  └──────────┘       │          │                      │
                     │          ▼                      │
                     │   "Colombia"  ← final answer    │
                     └─────────────────────────────────┘
```

**Key point**: The LLM is only called **once per question**, at the very end, after the small model has finished selecting paragraphs. During paragraph selection, the LLM is not involved at all.

---

## Why HotpotQA?

[HotpotQA](https://hotpotqa.github.io/) (distractor setting) gives each question **10 candidate paragraphs**:
- **2 gold supporting facts** — contain the info needed to answer
- **8 distractors** — look relevant but are useless

Example:
```
Question: "Which country is the birthplace of the author of
           One Hundred Years of Solitude?"
Answer:    "Colombia"

10 paragraphs:
  [0] García Márquez  → "Colombian novelist who wrote..."      ★ gold
  [1] Mathematics     → "Mathematics is a broad field..."      · distractor
  [2] Colombia        → "Colombia is a country in South..."    ★ gold
  [3] Physics         → "Physics studies natural laws..."      · distractor
  ... (6 more distractors)
```

The dataset **labels** which paragraphs are gold. We use these labels to give **per-step reward** during RL training.

---

## How One Question Is Solved (Step by Step)

### What the small model sees

The small model does **NOT** read paragraph text. It sees a **544-dim numeric vector** containing:
- BoW hash of the question (512 dims)
- Word overlap between each paragraph **title** and the question (10 numbers)
- Which paragraphs have been read already (10 binary flags)
- Current step number
- BoW hash of already-read content mixed with question

### The loop

```
Step 0:
  State → [question BoW, title overlaps, nothing read yet, step=0]
  Small model outputs 11 logits → samples action "read_0"
  → Python appends paragraph 0 ("García Márquez") to read_list
  → Reward: +0.3 (it's a gold supporting fact!)

Step 1:
  State → [question BoW + read content BoW, title overlaps, para 0 marked read, step=1]
  Small model outputs 11 logits → samples action "read_2"
  → Python appends paragraph 2 ("Colombia") to read_list
  → Reward: +0.3 (another gold supporting fact!)

Step 2:
  State → [updated BoW, title overlaps, paras 0,2 marked read, step=2]
  Small model outputs 11 logits → samples action "answer"
  → Python takes read_list = [para_0, para_2] and builds a prompt:
      "Question: ... Paragraphs: [García Márquez] ... [Colombia] ... Answer:"
  → Sends prompt to Qwen 8B (LLM), which replies "Colombia"
  → Reward: +1.0 (correct!) or 0.0 (wrong)
```

The 11 possible actions are: `read_0, read_1, ..., read_9, answer`.
The small model chooses one per step. When it chooses `answer`, the loop stops and the LLM is called.

### If it reaches max_steps (5) without choosing "answer"

Python automatically calls the LLM with whatever paragraphs have been read so far. No crash, just a forced answer.

---

## Reward Scheme (Dense)

| Event | Reward |
|---|---|
| Read a **gold supporting** paragraph | **+0.3** |
| Read a **distractor** paragraph | **-0.1** |
| Re-read an already-read paragraph | **-0.2** |
| Final answer **correct** | **+1.0** |
| Final answer **wrong** | **0.0** |

Every step gets a reward, not just the final answer. This is the "dense reward" that makes PPO training feasible.

---

## Pre-filter: Why We Need It

Qwen 8B has strong parametric knowledge — it can answer many HotpotQA questions **without any paragraphs** (we measured ~90% accuracy with no context).

If the LLM already knows the answer, paragraph selection is meaningless and PPO can't learn anything useful.

**Solution**: Before training, we run every candidate question with `no_context`. If the LLM gets it right, we **throw that question away**. We only keep questions the LLM **cannot answer without paragraphs**. This guarantees that reading the right paragraphs actually matters.

---

## Baselines

We compare 5 fixed strategies (all use the same LLM for final answer generation):

| Strategy | How It Selects Paragraphs |
|---|---|
| **Oracle** | Directly use the 2 gold paragraphs (cheating upper bound) |
| **No Context** | Don't read anything, LLM answers from memory (lower bound) |
| **Random(3)** | Pick 3 paragraphs at random |
| **Greedy(3)** | Pick 3 paragraphs whose titles have the most word overlap with the question |
| **All Context** | Read all 10 paragraphs |

PPO should beat Random (proves it learned), and ideally approach Greedy or Oracle.

---

## PPO Training

The small model (`RetrievalSelector`) is a residual MLP:

```
Input (544) → LayerNorm → Linear(544→128) + ReLU + Residual
            → Linear(128→128) + ReLU
            ├── Policy head  → 11 logits (probability of each action)
            └── Value head   → V(s) (estimated future reward)
```

**On-policy training** (8 iterations in full mode):
1. Use current policy to solve all 30 training questions (collecting trajectories)
2. Compute per-step advantages via GAE (γ=0.99, λ=0.95)
3. Run clipped PPO update (ε=0.2) on the collected data
4. Repeat with updated policy

---

## Why RL Instead of Supervised Learning?

Supervised learning needs labeled action sequences: "step 1 → read para 3, step 2 → read para 7, step 3 → answer". HotpotQA doesn't have this. It only tells you **which paragraphs are gold**, not **what order to read them in** or **when to stop**.

RL only needs reward signals, which we have (gold/distractor labels + answer correctness). It naturally learns:
- Which paragraphs to prioritize
- When to stop reading and answer
- How to recover if an early choice was bad

---

## Project Structure

```
hotpot_pipeline.py       # Main: data loading, pre-filter, baselines, PPO training, eval, report
multi_agent_baseline.py  # RetrievalAgent: the control loop (small model ↔ LLM coordination)
ppo_finetuner.py         # RetrievalSelector (MLP), PPOTrainer, DecisionCollector, TaskScorer
run_modal.py             # Deploy on Modal cloud (A10G GPU + Ollama)
```

## Quick Start

```bash
# On Modal (recommended):
pip install modal && modal setup
modal run run_modal.py            # full run (~40 min)
modal run run_modal.py --small    # quick test (~8 min)

# Locally:
pip install torch numpy requests datasets
ollama serve & ollama pull qwen3:8b && ollama pull qwen3:14b
python hotpot_pipeline.py                # full
python hotpot_pipeline.py --small        # quick test
```
