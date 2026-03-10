# Cross-Dataset Transfer: Zero-Shot Evaluation on HotpotQA

## Research Question

> Does the RL retrieval policy learn general multi-hop reasoning skills that transfer across datasets, or does it overfit to 2WikiMultiHopQA-specific patterns?

## Motivation

The PPO policy (F1=66.4%, p=0.0325 over BC) was trained exclusively on 2WikiMultiHopQA. If it performs well on HotpotQA without any retraining, that's strong evidence the policy learned **generalizable multi-hop retrieval skills** — not just dataset-specific artifacts.

This is the highest-value, lowest-cost experiment: no training required, just evaluation of existing saved models on new data.

## Dataset Comparison

| Aspect | 2WikiMultiHopQA (train) | HotpotQA (eval) |
|---|---|---|
| Gold paragraphs | 2 or 4 (variable) | Always 2 |
| Question types | compositional, bridge_comparison | bridge, comparison |
| Reasoning structure | Multi-hop chains (2-hop or 4-hop) | Two-hop (A→B) |
| Paragraph pool | 10 (gold + distractors) | 10 (2 gold + 8 distractors) |
| Feature space | 614-dim (512 BoW hash + 10 word overlap + 32 structured + 60 per-para) | 614-dim (identical) |

Both datasets produce the same example format: `{question, paragraphs, supporting_titles, supporting_indices_ordered}`. The feature extractor (BoW hashing, word overlap, IDF weighting, etc.) is entirely dataset-agnostic. The saved models use `input_dim=614` (BoW fallback mode).

## Experiment Design

### What We Evaluate

| Method | Source | What It Tests |
|---|---|---|
| **PPO (zero-shot)** | `checkpoints_blind/ppo_best.pt` | Does the RL-tuned policy generalize? |
| **BC (zero-shot)** | `checkpoints_blind/bc_model.pt` | Does the BC policy generalize? Isolates whether PPO's RL fine-tuning helped or hurt transfer. |
| **Greedy (K=1..7)** | No model needed | Baseline: how far does word overlap get you on HotpotQA? |

### What We Do NOT Run

- **Random baseline**: trivially computable (expected F1 = 2/10 = 20%), no need to waste eval time
- **LLM evaluation**: the question is about **retrieval quality**, not LLM answering ability
- **Any training**: this is strictly zero-shot transfer

## Implementation Plan

### New file: `eval_transfer.py`

A single self-contained script. **No existing files are modified.**

### Steps

1. **Load HotpotQA data** — `load_hotpot_data("train", max_examples=1500)` from `hotpot_pipeline.py` (use 1500 to match 2Wiki eval scale of 1499)
2. **Anonymize titles** — `_anonymize_titles()` for blind mode (same as 2Wiki training)
3. **Load saved models** from `checkpoints_blind/`:
   - `ppo_best.pt` → `PPOFineTuner(blind=True, lr=3e-5, entropy_coeff=0.02, kl_coeff=0.03)`
   - `bc_model.pt` → separate `PPOFineTuner` with identical config
4. **Evaluate retrieval** — `eval_policy_retrieval()` (per-gold-count breakdown, per-question F1 list)
5. **Run greedy baselines** — `eval_baseline_retrieval()` for K=1..7
6. **Significance tests** — paired permutation test (10,000 permutations, one-sided), same as `train_only.py`
7. **Load 2Wiki reference results** from `checkpoints_blind/train_metrics.json` for side-by-side table
8. **Save results** to `results_transfer/`:
   - `report.txt` — formatted text report
   - `transfer_metrics.json` — full metrics
   - `transfer_comparison.csv` — CSV table

### Configuration (matching existing codebase exactly)

```
Model config:     blind=True, lr=3e-5, entropy_coeff=0.02, kl_coeff=0.03
Budget:           K_BUDGET = 5
Seed:             42
Eval size:        1500 HotpotQA questions (comparable to 2Wiki's 1499)
Monkey-patch:     RetrievalAgent._verify_connection = lambda self: None
```

### Expected Output

```
                    HotpotQA (zero-shot)          2Wiki (reference)
Strategy         Prec   Recall   F1   Reads    Prec   Recall   F1   Reads
─────────────────────────────────────────────────────────────────────────
Greedy (3)        ...    ...    ...    ...      41.4%  51.0%  45.7%  3.0
BC (zero-shot)    ...    ...    ...    ~2.x     66.5%  65.3%  65.9%  2.4
PPO (zero-shot)   ...    ...    ...    ~2.x     67.2%  65.5%  66.4%  2.4
```

## Interpreting Results

| Outcome | Interpretation |
|---|---|
| PPO zero-shot ≥ Greedy on HotpotQA | Policy learned transferable retrieval heuristics beyond simple word overlap |
| PPO zero-shot ≈ Random on HotpotQA | Policy overfit to 2Wiki; learned behavior doesn't generalize |
| PPO > BC on HotpotQA | RL fine-tuning learned more robust features than pure imitation |
| BC > PPO on HotpotQA | PPO overfit to 2Wiki's reward structure; BC is more general |

All four outcomes are valid, publishable findings for a CS234 project.

## Success Criteria

- PPO zero-shot achieves **>60% of its 2Wiki F1** on HotpotQA → partial transfer
- OR transfer fails badly (<50% of 2Wiki F1) → the policy is dataset-specific (motivates domain adaptation)
- The adaptive read-count behavior (reading ~2 paragraphs for gold=2 questions) is observed on HotpotQA
