"""
Command 2: Load trained BC + PPO, prefilter, run single LLM-based eval.

Models evaluated (for the paper):
    oracle, no_context,
    random 2/3/4, greedy 2/3/4,
    BC-only, PPO (based on BC)

Requires Ollama running with qwen3:8b.

Usage:
    python eval_llm.py                   # full eval (blind, 2wiki)
    python eval_llm.py --small           # quick test (fewer questions)
    python eval_llm.py --no-blind        # non-blind mode
    python eval_llm.py --no-prefilter    # skip the no-context prefilter
"""

import ast
import json
import os
import sys
from pathlib import Path
from collections import Counter

from hotpot_pipeline import (
    setup_scorer, filter_by_no_context, run_baseline,
    _compute_retrieval_metrics, generate_report,
    _serialize_baselines, _box, _json_default,
)
from multi_agent_baseline import NUM_PARAGRAPHS, RetrievalAgent, AgentTrajectory
from ppo_finetuner import PPOFineTuner, TaskScorer


def _fix_supporting_titles(examples):
    """Normalise supporting_titles from JSON (list/str → set)."""
    for ex in examples.values():
        st = ex["supporting_titles"]
        if isinstance(st, str):
            try:
                st = ast.literal_eval(st)
            except (ValueError, SyntaxError):
                st = set()
        ex["supporting_titles"] = set(st) if not isinstance(st, set) else st


def main(small=False, blind=True, prefilter=True):
    _box("Command 2: LLM Evaluation")

    K_BUDGET = 5
    ckpt_dir = "checkpoints_blind" if blind else "checkpoints"
    output_dir = "results"
    Path(output_dir).mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # [1/4] Load saved split from Command 1
    # ------------------------------------------------------------------
    split_path = os.path.join(ckpt_dir, "split.json")
    if not os.path.isfile(split_path):
        raise FileNotFoundError(
            f"{split_path} not found. Run train_only.py first (Command 1).")

    print(f"\n[1/4] Loading data split from {split_path}...")
    with open(split_path) as f:
        saved = json.load(f)

    eval_examples = saved["eval"]
    _fix_supporting_titles(eval_examples)

    gc = Counter(len(ex["supporting_titles"]) for ex in eval_examples.values())
    print(f"  Eval pool: {len(eval_examples)} examples")
    print(f"  Gold distribution: "
          + ", ".join(f"{k}-gold:{v}" for k, v in sorted(gc.items())))

    # ------------------------------------------------------------------
    # [2/4] Pre-filter (remove questions LLM can answer without context)
    # ------------------------------------------------------------------
    if prefilter:
        N_TARGET = 50 if small else min(100, len(eval_examples))
        print(f"\n[2/4] Pre-filtering (keeping ≤{N_TARGET} hard questions)...")
        filtered = filter_by_no_context(eval_examples, target_count=N_TARGET)
    else:
        print("\n[2/4] Skipping prefilter (--no-prefilter)...")
        filtered = dict(eval_examples)
        if small:
            ids = list(filtered.keys())[:20]
            filtered = {k: filtered[k] for k in ids}

    print(f"  Final eval set: {len(filtered)} questions")
    gc_f = Counter(len(ex["supporting_titles"]) for ex in filtered.values())
    print(f"  Gold distribution: "
          + ", ".join(f"{k}-gold:{v}" for k, v in sorted(gc_f.items())))

    scorer = setup_scorer(filtered)

    # ------------------------------------------------------------------
    # [3/4] Run all strategies with LLM
    # ------------------------------------------------------------------
    print(f"\n[3/4] Running LLM evaluation ({len(filtered)} questions)...\n")
    baseline_results = []

    # Oracle
    m, t = run_baseline(filtered, scorer, "oracle", label="Oracle")
    baseline_results.append((m, t))

    # No Context
    m, t = run_baseline(filtered, scorer, "no_context", label="No Context")
    baseline_results.append((m, t))

    # Random 1-7
    for k in range(1, 8):
        m, t = run_baseline(filtered, scorer, "random",
                            max_reads=k, label=f"Random ({k})")
        baseline_results.append((m, t))

    # Greedy (content-only BoW) 1-7
    for k in range(1, 8):
        m, t = run_baseline(filtered, scorer, "greedy",
                            max_reads=k, label=f"Greedy ({k})")
        baseline_results.append((m, t))

    # BC-only  (load BC model weights)
    bc_path = os.path.join(ckpt_dir, "bc_model.pt")
    if not os.path.isfile(bc_path):
        raise FileNotFoundError(
            f"{bc_path} not found. Run train_only.py first (Command 1).")
    print(f"\n  Loading BC model from {bc_path}...")
    bc_tuner = PPOFineTuner(scorer, device="cpu", blind=blind)
    bc_tuner.load_model(bc_path)
    bc_m, bc_t = bc_tuner.eval_policy(
        filtered, scorer, K_BUDGET, label="BC-only")
    baseline_results.append((bc_m, bc_t))
    print(f"  BC-only: acc={bc_m['accuracy']:.1%}  "
          f"reads={bc_m['avg_reads']:.1f}  R={bc_m['recall']:.1%}")

    # PPO (best, based on BC)  —  load PPO model weights
    ppo_path = os.path.join(ckpt_dir, "ppo_best.pt")
    if not os.path.isfile(ppo_path):
        raise FileNotFoundError(
            f"{ppo_path} not found. Run train_only.py first (Command 1).")
    print(f"\n  Loading PPO model from {ppo_path}...")
    ppo_tuner = PPOFineTuner(scorer, device="cpu", blind=blind)
    ppo_tuner.load_model(ppo_path)
    ppo_m, ppo_t = ppo_tuner.eval_policy(
        filtered, scorer, K_BUDGET, label="PPO (ours)")
    ppo_result = (ppo_m, ppo_t)
    print(f"  PPO: acc={ppo_m['accuracy']:.1%}  "
          f"reads={ppo_m['avg_reads']:.1f}  R={ppo_m['recall']:.1%}")

    # ------------------------------------------------------------------
    # [4/4] Report
    # ------------------------------------------------------------------
    print(f"\n[4/4] Generating report...")

    # Load training curve from Command 1 (if available)
    metrics_path = os.path.join(ckpt_dir, "train_metrics.json")
    iter_metrics = []
    best_iter, best_metric = None, None
    if os.path.isfile(metrics_path):
        with open(metrics_path) as f:
            tm = json.load(f)
        iter_metrics = tm.get("ppo_training_curve", [])
        best_iter = tm.get("best_iter")
        best_metric = tm.get("best_metric")

    # Summary table
    _box("FINAL REPORT")
    all_metrics = [m for m, _ in baseline_results] + [ppo_m]

    print(f"\n  {'Strategy':<20} {'Acc':>7} {'Reads':>6} {'Supp':>5} "
          f"{'Prec':>6} {'Recall':>7} {'F1':>6}")
    print(f"  {'─'*20} {'─'*7} {'─'*6} {'─'*5} {'─'*6} {'─'*7} {'─'*6}")
    for m in all_metrics:
        print(f"  {m['strategy']:<20} {m.get('accuracy',0):>6.1%} "
              f"{m['avg_reads']:>6.1f} {m['avg_supporting_found']:>5.1f} "
              f"{m.get('precision', 0):>5.0%} "
              f"{m.get('recall', 0):>6.0%} "
              f"{m.get('f1', 0):>5.0%}")

    # Detailed report file
    ds_label = "2WikiMultiHopQA"  # inferred from the training data
    report_lines = generate_report(
        baseline_results, ppo_result, filtered, scorer,
        iter_metrics, ds_label=ds_label, budget=K_BUDGET,
        best_iter=best_iter, best_metric=best_metric,
    )
    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    # JSON comparison
    comparison = {
        "baselines": [m for m, _ in baseline_results],
        "ppo": ppo_m,
        "training_curve": iter_metrics,
    }
    comp_path = os.path.join(output_dir, "comparison.json")
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2, default=_json_default)

    # Baselines cache
    bl_path = os.path.join(ckpt_dir, "baselines.json")
    with open(bl_path, "w") as f:
        json.dump(_serialize_baselines(baseline_results), f,
                  indent=2, default=_json_default)

    print(f"\n  Results saved:")
    print(f"    {report_path:<40} Detailed report")
    print(f"    {comp_path:<40} Comparison JSON")
    print(f"    {bl_path:<40} Baseline cache")
    print()


if __name__ == "__main__":
    main(
        small="--small" in sys.argv,
        blind="--no-blind" not in sys.argv,   # default: blind=True
        prefilter="--no-prefilter" not in sys.argv,
    )
