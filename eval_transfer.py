"""
Zero-Shot Cross-Dataset Transfer: Evaluate 2Wiki-trained models on HotpotQA.

Evaluates retrieval metrics only (no LLM, no training).
Loads saved BC and PPO models from checkpoints_blind/ and runs them
on HotpotQA data to test whether learned policies generalize.

Usage:
    python eval_transfer.py              # full run (1500 HotpotQA questions)
    python eval_transfer.py --small      # quick test (100 questions)
"""

import json
import os
import csv
import sys
import random
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------------------------
# Monkey-patch: allow RetrievalAgent to work without an Ollama connection.
# Must happen BEFORE any other module imports RetrievalAgent.
# ---------------------------------------------------------------------------
import multi_agent_baseline
multi_agent_baseline.RetrievalAgent._verify_connection = lambda self: None

from hotpot_pipeline import load_hotpot_data, _anonymize_titles, _json_default
from train_only import eval_policy_retrieval, eval_baseline_retrieval
from ppo_finetuner import PPOFineTuner


# ======================================================================
#  Significance test (copied from train_only.py — defined locally there)
# ======================================================================

def paired_permutation_test(f1s_a, f1s_b, n_perm=10000, seed=42):
    """One-sided paired permutation test.
    H0: mean(a) <= mean(b).  H1: mean(a) > mean(b).
    Under H0, the sign of each paired difference is random.
    Returns p-value."""
    rng = np.random.RandomState(seed)
    a, b = np.array(f1s_a, dtype=float), np.array(f1s_b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d = a - b
    obs_diff = np.mean(d)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=n)
        perm_diff = np.mean(d * signs)
        if perm_diff >= obs_diff:
            count += 1
    return count / n_perm


# ======================================================================
#  Main
# ======================================================================

def main(small=False):
    print("=" * 70)
    print("  Zero-Shot Cross-Dataset Transfer: 2Wiki → HotpotQA")
    print("=" * 70)

    # Reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    K_BUDGET = 5
    ckpt_dir = "checkpoints_blind"
    out_dir = "results_transfer"
    Path(out_dir).mkdir(exist_ok=True)

    if small:
        N_EVAL = 100
        print("*** SMALL MODE: 100 questions for quick testing ***")
    else:
        N_EVAL = 1500

    # ------------------------------------------------------------------
    # [1/5] Load HotpotQA data
    # ------------------------------------------------------------------
    print(f"\n[1/5] Loading HotpotQA data ({N_EVAL} examples)...")
    hotpot_examples = load_hotpot_data("train", max_examples=N_EVAL)
    print(f"  Loaded {len(hotpot_examples)} HotpotQA examples")

    # Verify gold count distribution
    gc = Counter(len(ex["supporting_titles"]) for ex in hotpot_examples.values())
    print(f"  Gold distribution: "
          + ", ".join(f"{k}-gold:{v}" for k, v in sorted(gc.items())))

    # Anonymize titles (blind mode, same as 2Wiki training)
    print("  Anonymising titles (blind mode)...")
    hotpot_examples = _anonymize_titles(hotpot_examples)

    # ------------------------------------------------------------------
    # [2/5] Load trained models
    # ------------------------------------------------------------------
    print(f"\n[2/5] Loading trained models from {ckpt_dir}/...")

    ppo_path = os.path.join(ckpt_dir, "ppo_best.pt")
    bc_path = os.path.join(ckpt_dir, "bc_model.pt")

    if not os.path.isfile(ppo_path):
        print(f"  ERROR: {ppo_path} not found. Run train_only.py first.")
        sys.exit(1)
    if not os.path.isfile(bc_path):
        print(f"  ERROR: {bc_path} not found. Run train_only.py first.")
        sys.exit(1)

    # Detect saved model's input_dim from checkpoint weights.
    # ctx_proj.weight shape is [hidden, global_dim] where
    # global_dim = input_dim - num_paragraphs*3 - 60, so
    # input_dim = global_dim + 90.
    sd = torch.load(ppo_path, map_location="cpu", weights_only=False)
    saved_input_dim = sd["ctx_proj.weight"].shape[1] + 90  # 524+90=614 or 396+90=486
    print(f"  Detected saved model input_dim={saved_input_dim}")

    # If saved model uses BoW features (614-dim), we must hide
    # sentence-transformers so PPOFineTuner builds a matching model
    # and extracts features the same way as during training.
    _st_hidden = False
    if saved_input_dim == 614:
        # Block sentence-transformers import so PPOTrainer falls back
        # to BoW features (matching how the model was trained).
        # PPOTrainer.__init__ does:
        #   from sentence_transformers import SentenceTransformer
        # inside try/except ImportError. We inject an empty module so
        # the `from ... import SentenceTransformer` raises ImportError
        # (attribute not found), triggering the BoW fallback.
        import types
        _real_st = sys.modules.get("sentence_transformers")
        _fake = types.ModuleType("sentence_transformers")
        _fake.__path__ = []  # make it look like a package
        sys.modules["sentence_transformers"] = _fake
        _st_hidden = True
        print("  Note: saved model uses BoW features (614-dim), "
              "disabling sentence-transformers for compatibility")

    # PPO model
    ppo_tuner = PPOFineTuner(
        scorer=None, device="cpu", blind=True,
        lr=3e-5, entropy_coeff=0.02, kl_coeff=0.03,
    )
    ppo_tuner.load_model(ppo_path)
    print(f"  Loaded PPO model ← {ppo_path} (input_dim={ppo_tuner.input_dim})")

    # BC model (separate instance)
    bc_tuner = PPOFineTuner(
        scorer=None, device="cpu", blind=True,
        lr=3e-5, entropy_coeff=0.02, kl_coeff=0.03,
    )
    bc_tuner.load_model(bc_path)
    print(f"  Loaded BC model  ← {bc_path} (input_dim={bc_tuner.input_dim})")

    # Restore sentence-transformers if we hid it
    if _st_hidden:
        if _real_st is not None:
            sys.modules["sentence_transformers"] = _real_st
        else:
            del sys.modules["sentence_transformers"]

    # ------------------------------------------------------------------
    # [3/5] Evaluate on HotpotQA
    # ------------------------------------------------------------------
    print(f"\n[3/5] Evaluating retrieval on HotpotQA ({len(hotpot_examples)} questions)...")

    # Greedy baselines
    print("\n  Greedy baselines:")
    greedy_results = []
    for k in range(1, 8):
        label = f"Greedy ({k})"
        m = eval_baseline_retrieval(hotpot_examples, "greedy", k, label)
        greedy_results.append(m)
        print(f"    {label:<15} P={m['precision']:.1%}  "
              f"R={m['recall']:.1%}  F1={m['f1']:.1%}  "
              f"reads={m.get('avg_reads', k):.1f}")

    # BC policy
    print("\n  BC (zero-shot):")
    bc_ret = eval_policy_retrieval(bc_tuner, hotpot_examples, K_BUDGET, "BC (zero-shot)")
    print(f"    P={bc_ret['precision']:.1%}  R={bc_ret['recall']:.1%}  "
          f"F1={bc_ret['f1']:.1%}  reads={bc_ret['avg_reads']:.2f}")

    # PPO policy
    print("\n  PPO (zero-shot):")
    ppo_ret = eval_policy_retrieval(ppo_tuner, hotpot_examples, K_BUDGET, "PPO (zero-shot)")
    print(f"    P={ppo_ret['precision']:.1%}  R={ppo_ret['recall']:.1%}  "
          f"F1={ppo_ret['f1']:.1%}  reads={ppo_ret['avg_reads']:.2f}")

    all_ret = greedy_results + [bc_ret, ppo_ret]

    # ------------------------------------------------------------------
    # [4/5] Significance tests
    # ------------------------------------------------------------------
    print(f"\n[4/5] Significance tests (paired permutation, one-sided, n=10000)...")

    best_greedy = max(greedy_results, key=lambda x: x["f1"])
    sig_results = {}

    # PPO vs best greedy
    if "per_q_f1" in ppo_ret and "per_q_f1" in best_greedy:
        p_val = paired_permutation_test(ppo_ret["per_q_f1"], best_greedy["per_q_f1"])
        sig_results["PPO vs best Greedy"] = {
            "ppo_f1": ppo_ret["f1"], "greedy_f1": best_greedy["f1"],
            "greedy_label": best_greedy["strategy"],
            "p_value": p_val, "significant": p_val < 0.05,
        }

    # PPO vs BC
    if "per_q_f1" in ppo_ret and "per_q_f1" in bc_ret:
        p_val = paired_permutation_test(ppo_ret["per_q_f1"], bc_ret["per_q_f1"])
        sig_results["PPO vs BC"] = {
            "ppo_f1": ppo_ret["f1"], "bc_f1": bc_ret["f1"],
            "p_value": p_val, "significant": p_val < 0.05,
        }

    # BC vs best greedy
    if "per_q_f1" in bc_ret and "per_q_f1" in best_greedy:
        p_val = paired_permutation_test(bc_ret["per_q_f1"], best_greedy["per_q_f1"])
        sig_results["BC vs best Greedy"] = {
            "bc_f1": bc_ret["f1"], "greedy_f1": best_greedy["f1"],
            "greedy_label": best_greedy["strategy"],
            "p_value": p_val, "significant": p_val < 0.05,
        }

    for name, res in sig_results.items():
        star = ("***" if res["p_value"] < 0.001
                else ("**" if res["p_value"] < 0.01
                      else ("*" if res["p_value"] < 0.05 else "n.s.")))
        print(f"  {name}:  p={res['p_value']:.4f}  {star}")

    # ------------------------------------------------------------------
    # [5/5] Load 2Wiki reference & generate report
    # ------------------------------------------------------------------
    print(f"\n[5/5] Generating report...")

    # Load 2Wiki reference results
    wiki_metrics_path = os.path.join(ckpt_dir, "train_metrics.json")
    wiki_ref = {}
    if os.path.isfile(wiki_metrics_path):
        with open(wiki_metrics_path) as f:
            wiki_data = json.load(f)
        wiki_ref = {
            "bc": wiki_data.get("bc_retrieval", {}),
            "ppo": wiki_data.get("ppo_retrieval", {}),
            "best_greedy": None,
        }
        # Find best greedy from 2Wiki baselines
        baselines = wiki_data.get("baselines_retrieval", [])
        greedy_baselines = [m for m in baselines if "Greedy" in m.get("strategy", "")]
        if greedy_baselines:
            wiki_ref["best_greedy"] = max(greedy_baselines, key=lambda x: x.get("f1", 0))

    # Gold-count breakdown keys
    gold_counts = set()
    for m in all_ret:
        for k in m:
            if k.startswith("gold_"):
                gold_counts.add(int(k.split("_")[1]))

    # --- Text report ---
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("  Zero-Shot Cross-Dataset Transfer: 2Wiki → HotpotQA")
    report_lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append("Configuration:")
    report_lines.append(f"  Models trained on: 2WikiMultiHopQA (blind mode)")
    report_lines.append(f"  Evaluated on: HotpotQA ({len(hotpot_examples)} questions)")
    report_lines.append(f"  Budget: K={K_BUDGET}, seed=42")
    report_lines.append(f"  Model config: blind=True, lr=3e-5, entropy=0.02, kl=0.03")
    report_lines.append("")

    # HotpotQA results table
    report_lines.append("  HotpotQA Results (zero-shot transfer)")
    report_lines.append(f"  {'Strategy':<20} {'Prec':>6} {'Recall':>7} {'F1':>6} {'Reads':>6}")
    report_lines.append(f"  {'─'*20} {'─'*6} {'─'*7} {'─'*6} {'─'*6}")
    for m in all_ret:
        report_lines.append(
            f"  {m['strategy']:<20} {m['precision']:>5.1%}  "
            f"{m['recall']:>6.1%}  {m['f1']:>5.1%}  "
            f"{m.get('avg_reads', 0):>5.1f}")

    # Per-gold-count breakdown
    for gc_val in sorted(gold_counts):
        key = f"gold_{gc_val}"
        report_lines.append(f"\n  --- Gold={gc_val} subset ---")
        report_lines.append(f"  {'Strategy':<20} {'Prec':>6} {'Recall':>7} {'F1':>6} {'Reads':>6} {'N':>5}")
        report_lines.append(f"  {'─'*20} {'─'*6} {'─'*7} {'─'*6} {'─'*6} {'─'*5}")
        for m in all_ret:
            sub = m.get(key)
            if sub:
                report_lines.append(
                    f"  {m['strategy']:<20} {sub['precision']:>5.1%}  "
                    f"{sub['recall']:>6.1%}  {sub['f1']:>5.1%}  "
                    f"{sub.get('avg_reads', 0):>5.1f} {sub.get('total', ''):>5}")

    # Side-by-side comparison with 2Wiki
    if wiki_ref:
        report_lines.append("")
        report_lines.append("")
        report_lines.append("  Side-by-Side: HotpotQA (transfer) vs 2Wiki (in-domain)")
        report_lines.append(f"  {'Method':<20} {'HotpotQA F1':>12} {'2Wiki F1':>10} {'Delta':>8}")
        report_lines.append(f"  {'─'*20} {'─'*12} {'─'*10} {'─'*8}")

        comparisons = [
            ("Best Greedy", best_greedy, wiki_ref.get("best_greedy")),
            ("BC", bc_ret, wiki_ref.get("bc")),
            ("PPO", ppo_ret, wiki_ref.get("ppo")),
        ]
        for label, hotpot_m, wiki_m in comparisons:
            h_f1 = hotpot_m.get("f1", 0) if hotpot_m else 0
            w_f1 = wiki_m.get("f1", 0) if wiki_m else 0
            delta = h_f1 - w_f1
            sign = "+" if delta >= 0 else ""
            wiki_str = f"{w_f1:>9.1%}" if wiki_m else "      N/A"
            report_lines.append(
                f"  {label:<20} {h_f1:>11.1%} {wiki_str} {sign}{delta:>6.1%}")

    # Significance tests
    report_lines.append("")
    report_lines.append("")
    report_lines.append("  Significance Tests (paired permutation, one-sided, n=10000)")
    report_lines.append(f"  {'─'*60}")
    for name, res in sig_results.items():
        star = ("***" if res["p_value"] < 0.001
                else ("**" if res["p_value"] < 0.01
                      else ("*" if res["p_value"] < 0.05 else "n.s.")))
        report_lines.append(f"  {name}:  p={res['p_value']:.4f}  {star}")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # Save report
    report_path = os.path.join(out_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write(report_text + "\n")
    print(f"\n  Saved report → {report_path}")

    # --- JSON metrics ---
    def _strip_per_q(d):
        out = {}
        for k, v in d.items():
            if k == "per_q_f1":
                continue
            if isinstance(v, dict):
                out[k] = _strip_per_q(v)
            else:
                out[k] = v
        return out

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "source_dataset": "2wiki",
            "target_dataset": "hotpot",
            "n_eval": len(hotpot_examples),
            "budget": K_BUDGET,
            "blind": True,
            "seed": 42,
            "model_config": "blind=True, lr=3e-5, entropy=0.02, kl=0.03",
        },
        "hotpot_results": {
            "greedy_baselines": [_strip_per_q(m) for m in greedy_results],
            "bc_zero_shot": _strip_per_q(bc_ret),
            "ppo_zero_shot": _strip_per_q(ppo_ret),
        },
        "wiki_reference": {
            "bc": wiki_ref.get("bc", {}),
            "ppo": wiki_ref.get("ppo", {}),
            "best_greedy": wiki_ref.get("best_greedy", {}),
        },
        "significance_tests": sig_results,
    }

    metrics_path = os.path.join(out_dir, "transfer_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=_json_default)
    print(f"  Saved metrics → {metrics_path}")

    # --- CSV table ---
    csv_path = os.path.join(out_dir, "transfer_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "subset", "precision", "recall", "f1", "avg_reads", "n"])
        for m in all_ret:
            w.writerow([m["strategy"], "all", f"{m['precision']:.4f}",
                        f"{m['recall']:.4f}", f"{m['f1']:.4f}",
                        f"{m.get('avg_reads', 0):.2f}", m.get("total", "")])
            for gc_val in sorted(gold_counts):
                sub = m.get(f"gold_{gc_val}")
                if sub:
                    w.writerow([m["strategy"], f"gold={gc_val}",
                                f"{sub['precision']:.4f}", f"{sub['recall']:.4f}",
                                f"{sub['f1']:.4f}", f"{sub.get('avg_reads', 0):.2f}",
                                sub.get("total", "")])
    print(f"  Saved CSV → {csv_path}")

    print("\n  Done.")


if __name__ == "__main__":
    main(small="--small" in sys.argv)
