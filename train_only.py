"""
Command 1: Train BC + PPO with retrieval-only metrics (no LLM).

Disjoint train sets: 1000 examples for BC, 1000 new examples for PPO.
Ensures gold=2 and gold=4 data exist in all splits.
No LLM calls — only retrieval precision / recall / F1 metrics.

Usage:
    python train_only.py                  # full run (blind, 2wiki)
    python train_only.py --small          # quick test with tiny data
    python train_only.py --hotpot         # use HotpotQA instead of 2Wiki
    python train_only.py --no-blind       # keep real paragraph titles
"""

import json
import os
import random
import sys
import csv
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Monkey-patch: allow RetrievalAgent to work without an Ollama connection.
# Must happen BEFORE any other module imports RetrievalAgent.
# ---------------------------------------------------------------------------
import multi_agent_baseline  # noqa: E402
multi_agent_baseline.RetrievalAgent._verify_connection = lambda self: None

from hotpot_pipeline import (
    load_2wiki_data, load_hotpot_data, _anonymize_titles,
    _compute_retrieval_metrics, _box, _json_default,
)
from multi_agent_baseline import NUM_PARAGRAPHS, RetrievalAgent
from ppo_finetuner import PPOFineTuner, DecisionCollector


# ======================================================================
#  Helpers
# ======================================================================

def _per_q_f1(supp_read, total_reads, n_gold):
    """Compute per-question F1."""
    if n_gold == 0:
        return 1.0 if total_reads == 0 else 0.0
    rec = supp_read / n_gold
    prec = supp_read / max(1, total_reads)
    return 2 * prec * rec / max(1e-9, prec + rec)


def eval_baseline_retrieval(examples, strategy, max_reads, label):
    """Compute retrieval metrics for a fixed strategy (no LLM), with per-gold breakdown.
    Returns metrics dict with 'per_q_f1' list for significance tests."""
    agent = RetrievalAgent(agent_id=0, model="qwen3:8b")
    total = total_reads = total_supp = total_gold = 0
    per_q_f1s = []
    by_gc = defaultdict(lambda: {"total": 0, "reads": 0, "supp": 0, "gold": 0, "f1s": []})
    for qid, ex in examples.items():
        traj = agent.solve(
            qid, ex["question"], ex["paragraphs"],
            ex["supporting_titles"], strategy=strategy,
            max_reads=max_reads, training=True,
        )
        total += 1
        total_reads += traj.total_reads
        total_supp += traj.num_supporting_read
        gc = len(ex["supporting_titles"])
        total_gold += gc
        qf1 = _per_q_f1(traj.num_supporting_read, traj.total_reads, gc)
        per_q_f1s.append(qf1)
        by_gc[gc]["total"] += 1
        by_gc[gc]["reads"] += traj.total_reads
        by_gc[gc]["supp"] += traj.num_supporting_read
        by_gc[gc]["gold"] += gc
        by_gc[gc]["f1s"].append(qf1)
    m = _compute_retrieval_metrics(total, total_reads, total_supp, total_gold)
    m["strategy"] = label
    m["total"] = total
    m["per_q_f1"] = per_q_f1s
    for gc, acc in sorted(by_gc.items()):
        sub = _compute_retrieval_metrics(acc["total"], acc["reads"], acc["supp"], acc["gold"])
        sub["total"] = acc["total"]
        sub["per_q_f1"] = acc["f1s"]
        m[f"gold_{gc}"] = sub
    return m


def eval_policy_retrieval(fine_tuner, examples, max_steps, label):
    """Eval a policy model with per-gold-count breakdown (deterministic argmax).
    Returns metrics dict with 'per_q_f1' list for significance tests."""
    fine_tuner.model.eval()
    total = total_reads = total_supp = total_gold = 0
    n_correct = 0
    per_q_f1s = []
    by_gc = defaultdict(lambda: {"total": 0, "reads": 0, "supp": 0, "gold": 0, "f1s": []})
    for q_id, ex in examples.items():
        agent = RetrievalAgent(agent_id=0, model="qwen3:8b")
        traj = agent.solve_with_policy(
            q_id, ex["question"], ex["paragraphs"],
            ex["supporting_titles"], policy=fine_tuner, max_steps=max_steps,
            training=True,
        )
        total += 1
        total_reads += traj.total_reads
        total_supp += traj.num_supporting_read
        n_gold = len(ex["supporting_titles"])
        total_gold += n_gold
        if n_gold > 0 and traj.num_supporting_read / n_gold >= 0.5:
            n_correct += 1
        qf1 = _per_q_f1(traj.num_supporting_read, traj.total_reads, n_gold)
        per_q_f1s.append(qf1)
        by_gc[n_gold]["total"] += 1
        by_gc[n_gold]["reads"] += traj.total_reads
        by_gc[n_gold]["supp"] += traj.num_supporting_read
        by_gc[n_gold]["gold"] += n_gold
        by_gc[n_gold]["f1s"].append(qf1)

    recall = total_supp / max(1, total_gold)
    prec = total_supp / max(1, total_reads)
    f1 = 2 * prec * recall / max(1e-9, prec + recall)
    m = {
        "strategy": label,
        "recall": recall,
        "precision": prec,
        "f1": f1,
        "retrieval_acc": n_correct / max(1, total),
        "avg_reads": total_reads / max(1, total),
        "avg_supporting_found": total_supp / max(1, total),
        "total": total,
    }
    m["per_q_f1"] = per_q_f1s
    for gc, acc in sorted(by_gc.items()):
        sub = _compute_retrieval_metrics(acc["total"], acc["reads"], acc["supp"], acc["gold"])
        sub["total"] = acc["total"]
        sub["per_q_f1"] = acc["f1s"]
        m[f"gold_{gc}"] = sub
    fine_tuner.model.train()
    return m


def stratified_split(raw, n_bc, n_bc_dev, n_ppo, n_eval):
    """Split data into disjoint BC-train / BC-dev / PPO / eval ensuring gold-count variety."""
    by_gold = defaultdict(list)
    for qid, ex in raw.items():
        ng = len(ex["supporting_titles"])
        by_gold[ng].append(qid)

    total_needed = n_bc + n_bc_dev + n_ppo + n_eval
    bc_ids, bc_dev_ids, ppo_ids, eval_ids = [], [], [], []

    for ng in sorted(by_gold.keys()):
        ids = by_gold[ng]
        random.shuffle(ids)
        n = len(ids)
        alloc_bc = max(1, int(n * n_bc / total_needed))
        alloc_dev = max(1, int(n * n_bc_dev / total_needed))
        alloc_ppo = max(1, int(n * n_ppo / total_needed))
        alloc_eval = max(1, int(n * n_eval / total_needed))
        c = 0
        bc_ids.extend(ids[c:c + alloc_bc]); c += alloc_bc
        bc_dev_ids.extend(ids[c:c + alloc_dev]); c += alloc_dev
        ppo_ids.extend(ids[c:c + alloc_ppo]); c += alloc_ppo
        eval_ids.extend(ids[c:c + alloc_eval]); c += alloc_eval

    # Top-up if proportional allocation fell short
    used = set(bc_ids + bc_dev_ids + ppo_ids + eval_ids)
    remaining = [q for q in raw if q not in used]
    random.shuffle(remaining)
    while len(bc_ids) < n_bc and remaining:
        bc_ids.append(remaining.pop())
    while len(bc_dev_ids) < n_bc_dev and remaining:
        bc_dev_ids.append(remaining.pop())
    while len(ppo_ids) < n_ppo and remaining:
        ppo_ids.append(remaining.pop())
    while len(eval_ids) < n_eval and remaining:
        eval_ids.append(remaining.pop())

    return (
        {q: raw[q] for q in bc_ids},
        {q: raw[q] for q in bc_dev_ids},
        {q: raw[q] for q in ppo_ids},
        {q: raw[q] for q in eval_ids},
    )


# ======================================================================
#  Main
# ======================================================================

def main(small=False, dataset="2wiki", blind=True):
    ds_label = "2WikiMultiHopQA" if dataset == "2wiki" else "HotpotQA"
    mode_tag = " [BLIND]" if blind else ""
    _box(f"Command 1: Train BC + PPO ({ds_label}){mode_tag}")

    if small:
        N_BC, N_BC_DEV, N_PPO, N_EVAL, N_ITER = 150, 30, 250, 50, 8
        print("*** SMALL MODE: reduced scale for quick testing ***")
    else:
        N_BC, N_BC_DEV, N_PPO, N_EVAL, N_ITER = 1500, 300, 3000, 1000, 15

    BC_PATIENCE = 5   # early stop for BC
    PPO_PATIENCE = 6  # early stop for PPO

    K_BUDGET = 5
    ckpt_dir = "checkpoints_blind" if blind else "checkpoints"
    Path(ckpt_dir).mkdir(exist_ok=True)

    # Reproducibility
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # ------------------------------------------------------------------
    # [1/5] Load data
    # ------------------------------------------------------------------
    total_needed = N_BC + N_BC_DEV + N_PPO + N_EVAL
    max_load = int(total_needed * 1.5)
    print(f"\n[1/5] Loading data ({max_load} candidates)...")
    if dataset == "2wiki":
        raw = load_2wiki_data("train", max_examples=max_load, type_filter=None)
    else:
        raw = load_hotpot_data("train", max_examples=max_load)

    gc_raw = Counter(len(ex["supporting_titles"]) for ex in raw.values())
    print(f"  Raw gold distribution: "
          + ", ".join(f"{k}-gold:{v}" for k, v in sorted(gc_raw.items())))

    # ------------------------------------------------------------------
    # [2/5] Stratified split: BC-train / BC-dev / PPO / eval  (disjoint)
    # ------------------------------------------------------------------
    print(f"\n[2/5] Splitting: BC={N_BC}+dev={N_BC_DEV}, PPO={N_PPO}, Eval={N_EVAL} (disjoint)...")
    bc_examples, bc_dev_examples, ppo_examples, eval_examples = stratified_split(
        raw, N_BC, N_BC_DEV, N_PPO, N_EVAL)

    for name, ex_dict in [("BC-train", bc_examples),
                          ("BC-dev", bc_dev_examples),
                          ("PPO", ppo_examples),
                          ("Eval", eval_examples)]:
        gc = Counter(len(ex["supporting_titles"]) for ex in ex_dict.values())
        print(f"  {name:>9}: {len(ex_dict):>5} examples  gold: "
              + ", ".join(f"{k}:{v}" for k, v in sorted(gc.items())))

    # Verify disjoint
    all_sets = {
        "BC": set(bc_examples.keys()),
        "BC-dev": set(bc_dev_examples.keys()),
        "PPO": set(ppo_examples.keys()),
        "Eval": set(eval_examples.keys()),
    }
    for name_a, sa in all_sets.items():
        for name_b, sb in all_sets.items():
            if name_a < name_b:
                assert sa.isdisjoint(sb), f"{name_a} and {name_b} overlap!"
    print("  ✓ All splits are disjoint")

    # Verify gold=2 and gold=4 exist (at least somewhere)
    all_golds = set()
    for d in [bc_examples, bc_dev_examples, ppo_examples, eval_examples]:
        for ex in d.values():
            all_golds.add(len(ex["supporting_titles"]))
    if 2 in all_golds and 4 in all_golds:
        print("  ✓ Both gold=2 and gold=4 present")
    else:
        print(f"  ⚠ Gold counts present: {sorted(all_golds)} (wanted 2 and 4)")

    if blind:
        print("  Anonymising titles (blind mode)...")
        bc_examples = _anonymize_titles(bc_examples)
        bc_dev_examples = _anonymize_titles(bc_dev_examples)
        ppo_examples = _anonymize_titles(ppo_examples)
        eval_examples = _anonymize_titles(eval_examples)

    # Save split
    split_path = os.path.join(ckpt_dir, "split.json")
    with open(split_path, "w") as f:
        json.dump({
            "bc": bc_examples,
            "bc_dev": bc_dev_examples,
            "ppo": ppo_examples,
            "eval": eval_examples,
        }, f, indent=2, default=_json_default)
    print(f"  Saved split → {split_path}")

    # ------------------------------------------------------------------
    # [3/5] BC (oracle) on BC set  —  no LLM
    # ------------------------------------------------------------------
    print(f"\n[3/5] Behavioral cloning (oracle) on BC set "
          f"({len(bc_examples)} train + {len(bc_dev_examples)} dev, "
          f"up to 20 epochs, patience={BC_PATIENCE})...")
    fine_tuner = PPOFineTuner(
        scorer=None, device="cpu", blind=blind,
        lr=3e-5, entropy_coeff=0.02, kl_coeff=0.03,
    )
    bc_history = fine_tuner.behavior_clone_oracle(
        bc_examples, max_steps=K_BUDGET, bc_epochs=20,
        batch_size=16, lr=1e-3,
        dev_examples=bc_dev_examples, patience=BC_PATIENCE,
    )

    bc_model_path = os.path.join(ckpt_dir, "bc_model.pt")
    fine_tuner.save_model(bc_model_path)
    fine_tuner.trainer.snapshot_reference()
    print(f"  Saved BC model → {bc_model_path}")

    # BC retrieval eval (no LLM)
    bc_ret = eval_policy_retrieval(fine_tuner, eval_examples, K_BUDGET, "BC-only")
    print(f"  BC-only retrieval:  P={bc_ret['precision']:.1%}  "
          f"R={bc_ret['recall']:.1%}  F1={bc_ret['f1']:.1%}")

    # Baseline retrieval comparison (no LLM)
    print("\n  Baseline retrieval metrics (eval set, no LLM):")
    baselines_ret = []
    for strategy in ["random", "greedy"]:
        for k in range(1, 8):
            label = f"{strategy.capitalize()} ({k})"
            m = eval_baseline_retrieval(eval_examples, strategy, k, label)
            baselines_ret.append(m)
            print(f"    {label:<15} P={m['precision']:.1%}  "
                  f"R={m['recall']:.1%}  F1={m['f1']:.1%}")

    # ------------------------------------------------------------------
    # [4/5] PPO training on PPO set  —  no LLM
    # ------------------------------------------------------------------
    print(f"\n[4/5] PPO training on PPO set "
          f"({len(ppo_examples)} examples, {N_ITER} iters, budget={K_BUDGET})...")

    # Save iter-0 checkpoint (= current BC model) so on_policy_train can
    # "resume" from it and skip the internal BC phase.
    ckpt0_path = os.path.join(ckpt_dir, "ckpt_iter_000.pt")
    fine_tuner.save_checkpoint(ckpt0_path, 0, [])

    iter_metrics, _, best_iter, best_metric = fine_tuner.on_policy_train(
        ppo_examples,
        num_iterations=N_ITER,
        max_steps=K_BUDGET,
        ppo_epochs=3,
        batch_size=16,
        checkpoint_dir=ckpt_dir,
        resume_from=ckpt0_path,
        patience=PPO_PATIENCE,
        eval_examples=eval_examples,
        bc_epochs=20,       # unused (resume skips BC)
        bc_expert="oracle",  # unused (resume skips BC)
        scorer=None,         # no LLM
    )

    ppo_model_path = os.path.join(ckpt_dir, "ppo_best.pt")
    fine_tuner.save_model(ppo_model_path)
    print(f"\n  Saved PPO best model → {ppo_model_path}  (best iter={best_iter})")

    # ------------------------------------------------------------------
    # [5/5] Final retrieval comparison + significance tests + plots
    # ------------------------------------------------------------------
    print(f"\n[5/5] Final retrieval comparison (eval set, no LLM):")

    ppo_ret = eval_policy_retrieval(fine_tuner, eval_examples, K_BUDGET, "PPO (ours)")

    all_ret = baselines_ret + [bc_ret, ppo_ret]

    # Gold-count breakdown keys
    gold_counts = set()
    for m in all_ret:
        for k in m:
            if k.startswith("gold_"):
                gold_counts.add(int(k.split("_")[1]))

    # Find best greedy by F1
    best_greedy = max((m for m in baselines_ret if "Greedy" in m["strategy"]),
                      key=lambda x: x["f1"])

    # --- Bootstrap paired significance test ---
    def paired_permutation_test(f1s_a, f1s_b, n_perm=10000, seed=42):
        """One-sided paired permutation test.
        H0: mean(a) <= mean(b).  H1: mean(a) > mean(b).
        Under H0, the sign of each paired difference is random.
        Returns p-value."""
        rng = np.random.RandomState(seed)
        a, b = np.array(f1s_a, dtype=float), np.array(f1s_b, dtype=float)
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        d = a - b  # paired differences
        obs_diff = np.mean(d)
        count = 0
        for _ in range(n_perm):
            signs = rng.choice([-1, 1], size=n)
            perm_diff = np.mean(d * signs)
            if perm_diff >= obs_diff:
                count += 1
        return count / n_perm

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
        sig_results["PPO vs BC-only"] = {
            "ppo_f1": ppo_ret["f1"], "bc_f1": bc_ret["f1"],
            "p_value": p_val, "significant": p_val < 0.05,
        }
    # Per gold-count significance
    for gc in sorted(gold_counts):
        key = f"gold_{gc}"
        ppo_sub = ppo_ret.get(key, {})
        gr_sub = best_greedy.get(key, {})
        if "per_q_f1" in ppo_sub and "per_q_f1" in gr_sub:
            p_val = paired_permutation_test(ppo_sub["per_q_f1"], gr_sub["per_q_f1"])
            sig_results[f"PPO vs {best_greedy['strategy']} (gold={gc})"] = {
                "ppo_f1": ppo_sub["f1"], "greedy_f1": gr_sub["f1"],
                "p_value": p_val, "significant": p_val < 0.05,
            }

    # --- Print tables to terminal ---
    def print_table(header, rows, gold_counts, all_ret):
        lines = []
        lines.append(f"\n  {header}")
        lines.append(f"  {'Strategy':<15} {'Prec':>6} {'Recall':>7} {'F1':>6} {'Reads':>6}")
        lines.append(f"  {'─'*15} {'─'*6} {'─'*7} {'─'*6} {'─'*6}")
        for m in all_ret:
            lines.append(f"  {m['strategy']:<15} {m['precision']:>5.1%}  "
                         f"{m['recall']:>6.1%}  {m['f1']:>5.1%}  "
                         f"{m.get('avg_reads', 0):>5.1f}")
        for gc in sorted(gold_counts):
            key = f"gold_{gc}"
            lines.append(f"\n  --- Gold={gc} subset ---")
            lines.append(f"  {'Strategy':<15} {'Prec':>6} {'Recall':>7} {'F1':>6} {'Reads':>6}")
            lines.append(f"  {'─'*15} {'─'*6} {'─'*7} {'─'*6} {'─'*6}")
            for m in all_ret:
                sub = m.get(key)
                if sub:
                    lines.append(f"  {m['strategy']:<15} {sub['precision']:>5.1%}  "
                                 f"{sub['recall']:>6.1%}  {sub['f1']:>5.1%}  "
                                 f"{sub.get('avg_reads', 0):>5.1f}")
        return lines

    table_lines = print_table("Final Retrieval Comparison", [], gold_counts, all_ret)

    # Significance test lines
    sig_lines = ["\n  === Significance Tests (paired permutation, one-sided, n=10000) ==="]
    for name, res in sig_results.items():
        star = "***" if res["p_value"] < 0.001 else ("**" if res["p_value"] < 0.01 else ("*" if res["p_value"] < 0.05 else "n.s."))
        sig_lines.append(f"  {name}:  p={res['p_value']:.4f}  {star}")

    for line in table_lines + sig_lines:
        print(line)

    # --- Save text report ---
    report_path = os.path.join(ckpt_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"  Retrieval Comparison Report  ({ds_label}, {'blind' if blind else 'normal'})\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  BC train={len(bc_examples)}, BC dev={len(bc_dev_examples)}\n")
        f.write(f"  PPO train={len(ppo_examples)}, Eval={len(eval_examples)}\n")
        f.write(f"  PPO iters={N_ITER}, budget={K_BUDGET}, best_iter={best_iter}\n")
        f.write(f"  BC: oracle (gold read order), patience={BC_PATIENCE}\n")
        f.write(f"  PPO: lr=3e-5, entropy=0.02, kl=0.03(adaptive), patience={PPO_PATIENCE}\n")
        f.write(f"  Reward: SUPPORTING={DecisionCollector.REWARD_SUPPORTING}, "
                f"DISTRACTOR={DecisionCollector.REWARD_DISTRACTOR}, "
                f"STEP={DecisionCollector.STEP_PENALTY}, "
                f"BRIDGE={DecisionCollector.REWARD_BRIDGE_BONUS}\n")
        f.write(f"  Answer: {DecisionCollector.STOP_SCALE}*recall + "
                f"{DecisionCollector.COMPLETION_BONUS}*(recall==1)\n")
        f.write(f"  Learned reward shaping: potential-based Φ(s), ramp coeff 0→0.3\n\n")

        for line in table_lines:
            f.write(line.rstrip() + "\n")
        f.write("\n")
        for line in sig_lines:
            f.write(line.rstrip() + "\n")

        f.write("\n\n" + "=" * 70 + "\n")
        f.write("  PPO Training Curve\n")
        f.write("=" * 70 + "\n")
        f.write(f"  {'Iter':>4} {'Train F1':>9} {'Eval F1':>8} {'Reads':>6} {'Return':>7} {'P-Loss':>7} {'Entropy':>8} {'RM-Loss':>8} {'Shape':>6}\n")
        f.write(f"  {'─'*4} {'─'*9} {'─'*8} {'─'*6} {'─'*7} {'─'*7} {'─'*8} {'─'*8} {'─'*6}\n")
        for it in iter_metrics:
            tp = it.get("precision", 0)
            tr_ = it.get("recall", 0)
            tf1 = 2*tp*tr_ / max(1e-9, tp+tr_)
            ev = it.get("eval_retrieval", {})
            trn = it.get("training", {})
            hist = trn.get("history", [{}])
            last_h = hist[-1] if hist else {}
            rm = it.get("reward_model", {})
            f.write(f"  {it['iteration']:>4} {tf1:>8.1%} {ev.get('f1', 0):>7.1%} "
                    f"{it.get('avg_reads', 0):>5.1f} {it.get('mean_return', 0):>6.3f} "
                    f"{last_h.get('policy_loss', 0):>6.4f} {last_h.get('entropy', 0):>7.4f} "
                    f"{rm.get('rm_loss', 0):>7.4f} {it.get('shaping_coeff', 0):>5.2f}\n")
        f.write(f"\n  Best iteration: {best_iter} (eval F1={best_metric:.1%})\n")

        f.write("\n\n" + "=" * 70 + "\n")
        f.write("  BC Loss Curve\n")
        f.write("=" * 70 + "\n")
        for ep in bc_history:
            dev_s = f"  dev={ep['dev_loss']:.4f}" if "dev_loss" in ep else ""
            f.write(f"  epoch {ep['epoch']:>2}  train={ep['train_loss']:.4f}{dev_s}\n")

    print(f"\n  Saved report → {report_path}")

    # --- Save JSON metrics ---
    metrics_path = os.path.join(ckpt_dir, "train_metrics.json")
    reward_config = {
        "REWARD_SUPPORTING": DecisionCollector.REWARD_SUPPORTING,
        "REWARD_DISTRACTOR": DecisionCollector.REWARD_DISTRACTOR,
        "REWARD_ORDER_BONUS": DecisionCollector.REWARD_ORDER_BONUS,
        "REWARD_BRIDGE_BONUS": DecisionCollector.REWARD_BRIDGE_BONUS,
        "STEP_PENALTY": DecisionCollector.STEP_PENALTY,
        "STOP_SCALE": DecisionCollector.STOP_SCALE,
        "COMPLETION_BONUS": DecisionCollector.COMPLETION_BONUS,
        "answer_formula": "STOP_SCALE * recall^2 + COMPLETION_BONUS * (recall==1)",
    }
    # Strip per_q_f1 from JSON to keep file small
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
    with open(metrics_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {
                "n_bc": len(bc_examples), "n_bc_dev": len(bc_dev_examples),
                "n_ppo": len(ppo_examples),
                "n_eval": len(eval_examples), "n_iter": N_ITER,
                "budget": K_BUDGET, "blind": blind, "dataset": dataset,
                "bc_patience": BC_PATIENCE, "ppo_patience": PPO_PATIENCE,
                "entropy_coeff": 0.02, "kl_coeff": "0.03(adaptive)", "lr": 3e-5,
            },
            "reward_config": reward_config,
            "bc_loss_history": bc_history,
            "bc_retrieval": _strip_per_q(bc_ret),
            "baselines_retrieval": [_strip_per_q(m) for m in baselines_ret],
            "ppo_retrieval": _strip_per_q(ppo_ret),
            "ppo_training_curve": iter_metrics,
            "best_iter": best_iter,
            "best_metric": best_metric,
            "significance_tests": sig_results,
        }, f, indent=2, default=_json_default)
    print(f"  Saved metrics → {metrics_path}")

    # --- Save per-iteration PPO trajectories (for detailed analysis) ---
    traj_path = os.path.join(ckpt_dir, "ppo_trajectories.json")
    with open(traj_path, "w") as f:
        json.dump(fine_tuner.all_train_trajectories, f, indent=1, default=_json_default)
    print(f"  Saved PPO trajectories → {traj_path}")

    # --- Save CSV tables for easy paper plotting ---
    # 1) BC loss curve
    bc_csv = os.path.join(ckpt_dir, "bc_loss_curve.csv")
    with open(bc_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "dev_loss"])
        for ep in bc_history:
            w.writerow([ep["epoch"], f"{ep['train_loss']:.6f}",
                        f"{ep.get('dev_loss', ''):.6f}" if "dev_loss" in ep else ""])
    print(f"  Saved BC loss curve → {bc_csv}")

    # 2) PPO training curve
    ppo_csv = os.path.join(ckpt_dir, "ppo_training_curve.csv")
    with open(ppo_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "train_precision", "train_recall", "train_f1",
                     "train_avg_reads", "mean_return", "avg_step_reward",
                     "eval_precision", "eval_recall", "eval_f1", "eval_avg_reads",
                     "policy_loss", "value_loss", "entropy", "kl_from_bc", "kl_coeff"])
        for it in iter_metrics:
            train_p = it.get("precision", 0)
            train_r = it.get("recall", 0)
            train_f1 = 2*train_p*train_r / max(1e-9, train_p+train_r)
            ev = it.get("eval_retrieval", {})
            tr = it.get("training", {})
            hist = tr.get("history", [{}])
            last_h = hist[-1] if hist else {}
            w.writerow([
                it["iteration"],
                f"{train_p:.6f}", f"{train_r:.6f}", f"{train_f1:.6f}",
                f"{it.get('avg_reads', 0):.3f}",
                f"{it.get('mean_return', 0):.6f}",
                f"{it.get('avg_step_reward', 0):.6f}",
                f"{ev.get('precision', '')}", f"{ev.get('recall', '')}",
                f"{ev.get('f1', '')}", f"{ev.get('avg_reads', '')}",
                f"{last_h.get('policy_loss', '')}", f"{last_h.get('value_loss', '')}",
                f"{last_h.get('entropy', '')}", f"{last_h.get('kl_from_bc', '')}",
                f"{it.get('kl_coeff', '')}",
            ])
    print(f"  Saved PPO training curve → {ppo_csv}")

    # 3) Final comparison table (with gold-count breakdown)
    table_csv = os.path.join(ckpt_dir, "retrieval_comparison.csv")
    with open(table_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "subset", "precision", "recall", "f1", "avg_reads", "n"])
        for m in all_ret:
            w.writerow([m["strategy"], "all", f"{m['precision']:.4f}",
                        f"{m['recall']:.4f}", f"{m['f1']:.4f}",
                        f"{m.get('avg_reads', 0):.2f}", m.get("total", "")])
            for gc in sorted(gold_counts):
                sub = m.get(f"gold_{gc}")
                if sub:
                    w.writerow([m["strategy"], f"gold={gc}",
                                f"{sub['precision']:.4f}", f"{sub['recall']:.4f}",
                                f"{sub['f1']:.4f}", f"{sub.get('avg_reads', 0):.2f}",
                                sub.get("total", "")])
    print(f"  Saved comparison table → {table_csv}")

    # --- Visualizations ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        random_pts = [(m['recall'], m['precision'], m['strategy'])
                      for m in baselines_ret if 'Random' in m['strategy']]
        greedy_pts = [(m['recall'], m['precision'], m['strategy'])
                      for m in baselines_ret if 'Greedy' in m['strategy']]

        # ===== Plot 1: Overall P-R tradeoff =====
        fig, ax = plt.subplots(figsize=(8, 6))
        rr = [p[0] for p in random_pts]
        rp = [p[1] for p in random_pts]
        ax.plot(rr, rp, 'o-', color='#999999', label='Random(K)',
                markersize=5, linewidth=1.5)
        for r, p, lbl in random_pts:
            k = lbl.split('(')[1].rstrip(')')
            ax.annotate(f'K={k}', (r, p), textcoords="offset points",
                        xytext=(5, 5), fontsize=7, color='#999999')
        gr = [p[0] for p in greedy_pts]
        gp = [p[1] for p in greedy_pts]
        ax.plot(gr, gp, 's-', color='#2196F3', label='Greedy(K)',
                markersize=5, linewidth=1.5)
        for r, p, lbl in greedy_pts:
            k = lbl.split('(')[1].rstrip(')')
            ax.annotate(f'K={k}', (r, p), textcoords="offset points",
                        xytext=(5, 5), fontsize=7, color='#2196F3')
        for f1_val in [0.2, 0.3, 0.4, 0.5, 0.6]:
            r_range = np.linspace(0.01, 0.99, 200)
            p_iso = f1_val * r_range / (2 * r_range - f1_val)
            mask = (p_iso > 0) & (p_iso <= 1)
            ax.plot(r_range[mask], p_iso[mask], '--', color='#E0E0E0',
                    linewidth=0.8, alpha=0.7)
            idx = np.argmin(np.abs(r_range - f1_val))
            if mask[idx]:
                ax.annotate(f'F1={f1_val}', (r_range[idx], p_iso[idx]),
                            fontsize=6, color='#BDBDBD')
        ax.scatter(bc_ret['recall'], bc_ret['precision'], marker='D',
                   s=120, color='#FF9800', zorder=5, edgecolors='black',
                   linewidths=0.8, label=f"BC-only (F1={bc_ret['f1']:.1%})")
        ax.scatter(ppo_ret['recall'], ppo_ret['precision'], marker='*',
                   s=250, color='#F44336', zorder=5, edgecolors='black',
                   linewidths=0.8, label=f"PPO (F1={ppo_ret['f1']:.1%})")
        ax.annotate('', xy=(ppo_ret['recall'], ppo_ret['precision']),
                    xytext=(bc_ret['recall'], bc_ret['precision']),
                    arrowprops=dict(arrowstyle='->', color='#4CAF50',
                                   lw=1.5, ls='--'))
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Tradeoff: Paragraph Retrieval\n'
                     '(eval set, no LLM)', fontsize=13)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        plot_path = os.path.join(ckpt_dir, "precision_recall.png")
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved PR plot → {plot_path}")

        # ===== Plot 2: Per-gold-count P-R subplots =====
        n_gc = len(gold_counts)
        if n_gc > 0:
            fig2, axes = plt.subplots(1, n_gc, figsize=(7 * n_gc, 6), squeeze=False)
            for col, gc in enumerate(sorted(gold_counts)):
                ax2 = axes[0, col]
                key = f"gold_{gc}"
                # Greedy curve for this gold count
                gc_greedy_r, gc_greedy_p = [], []
                for m in baselines_ret:
                    if "Greedy" in m["strategy"]:
                        sub = m.get(key)
                        if sub:
                            gc_greedy_r.append(sub["recall"])
                            gc_greedy_p.append(sub["precision"])
                if gc_greedy_r:
                    ax2.plot(gc_greedy_r, gc_greedy_p, 's-', color='#2196F3',
                             label='Greedy(K)', markersize=5, linewidth=1.5)
                    for i, m in enumerate(m_ for m_ in baselines_ret if "Greedy" in m_["strategy"]):
                        sub = m.get(key)
                        if sub:
                            k = m["strategy"].split('(')[1].rstrip(')')
                            ax2.annotate(f'K={k}', (sub["recall"], sub["precision"]),
                                         textcoords="offset points", xytext=(5, 5),
                                         fontsize=7, color='#2196F3')
                # Random curve
                gc_rand_r, gc_rand_p = [], []
                for m in baselines_ret:
                    if "Random" in m["strategy"]:
                        sub = m.get(key)
                        if sub:
                            gc_rand_r.append(sub["recall"])
                            gc_rand_p.append(sub["precision"])
                if gc_rand_r:
                    ax2.plot(gc_rand_r, gc_rand_p, 'o-', color='#999999',
                             label='Random(K)', markersize=4, linewidth=1)
                # F1 iso-curves
                for f1_val in [0.2, 0.3, 0.4, 0.5, 0.6]:
                    r_range = np.linspace(0.01, 0.99, 200)
                    p_iso = f1_val * r_range / (2 * r_range - f1_val)
                    mask_iso = (p_iso > 0) & (p_iso <= 1)
                    ax2.plot(r_range[mask_iso], p_iso[mask_iso], '--', color='#E0E0E0',
                             linewidth=0.6, alpha=0.5)
                # BC and PPO points
                bc_sub = bc_ret.get(key)
                ppo_sub = ppo_ret.get(key)
                if bc_sub:
                    ax2.scatter(bc_sub['recall'], bc_sub['precision'], marker='D',
                                s=100, color='#FF9800', zorder=5, edgecolors='black',
                                linewidths=0.8, label=f"BC (F1={bc_sub['f1']:.1%})")
                if ppo_sub:
                    ax2.scatter(ppo_sub['recall'], ppo_sub['precision'], marker='*',
                                s=200, color='#F44336', zorder=5, edgecolors='black',
                                linewidths=0.8, label=f"PPO (F1={ppo_sub['f1']:.1%})")
                if bc_sub and ppo_sub:
                    ax2.annotate('', xy=(ppo_sub['recall'], ppo_sub['precision']),
                                 xytext=(bc_sub['recall'], bc_sub['precision']),
                                 arrowprops=dict(arrowstyle='->', color='#4CAF50',
                                                 lw=1.5, ls='--'))
                # Significance annotation
                sig_key = f"PPO vs {best_greedy['strategy']} (gold={gc})"
                if sig_key in sig_results:
                    p_val = sig_results[sig_key]["p_value"]
                    star = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))
                    ax2.text(0.02, 0.02, f"PPO vs Greedy: p={p_val:.3f} {star}",
                             transform=ax2.transAxes, fontsize=8,
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                ax2.set_xlabel('Recall', fontsize=11)
                ax2.set_ylabel('Precision', fontsize=11)
                ax2.set_title(f'Gold={gc} Subset (n={ppo_sub.get("total", "?")})',
                              fontsize=12)
                ax2.legend(loc='upper right', fontsize=8)
                ax2.grid(True, alpha=0.3)
                ax2.set_xlim(0, 1.0)
                ax2.set_ylim(0, 1.0)

            fig2.suptitle('Per-Gold-Count Precision-Recall Tradeoff', fontsize=14, y=1.02)
            fig2.tight_layout()
            pgc_path = os.path.join(ckpt_dir, "precision_recall_by_gold.png")
            fig2.savefig(pgc_path, dpi=150, bbox_inches='tight')
            plt.close(fig2)
            print(f"  Saved per-gold PR plot → {pgc_path}")

        # ===== Plot 3: PPO training curve =====
        if iter_metrics:
            iters = [it["iteration"] for it in iter_metrics]
            eval_f1s = [it.get("eval_retrieval", {}).get("f1", 0) for it in iter_metrics]
            train_f1s = []
            for it in iter_metrics:
                tp = it.get("precision", 0)
                tr_ = it.get("recall", 0)
                train_f1s.append(2*tp*tr_ / max(1e-9, tp+tr_))
            returns = [it.get("mean_return", 0) for it in iter_metrics]

            fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))

            ax3a.plot(iters, train_f1s, 'o-', color='#2196F3', label='Train F1',
                      markersize=4, linewidth=1.5)
            ax3a.plot(iters, eval_f1s, 's-', color='#F44336', label='Eval F1',
                      markersize=4, linewidth=1.5)
            ax3a.axhline(y=best_greedy["f1"], color='#4CAF50', linestyle='--',
                         linewidth=1, label=f'{best_greedy["strategy"]} F1')
            ax3a.axhline(y=bc_ret["f1"], color='#FF9800', linestyle=':',
                         linewidth=1, label='BC-only F1')
            ax3a.set_xlabel('PPO Iteration')
            ax3a.set_ylabel('F1 Score')
            ax3a.set_title('F1 Over PPO Iterations')
            ax3a.legend(fontsize=8)
            ax3a.grid(True, alpha=0.3)

            ax3b.plot(iters, returns, 'D-', color='#9C27B0', markersize=4, linewidth=1.5)
            ax3b.set_xlabel('PPO Iteration')
            ax3b.set_ylabel('Mean Return')
            ax3b.set_title('Mean Return Over PPO Iterations')
            ax3b.grid(True, alpha=0.3)

            fig3.tight_layout()
            curve_path = os.path.join(ckpt_dir, "ppo_training_curve.png")
            fig3.savefig(curve_path, dpi=150, bbox_inches='tight')
            plt.close(fig3)
            print(f"  Saved training curve plot → {curve_path}")

        # ===== Plot 4: F1 bar chart comparison =====
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        # Select key strategies for the bar chart
        key_strategies = ["Random (2)", "Greedy (2)", "Greedy (3)", "Greedy (5)", "BC-only", "PPO (ours)"]
        bar_data = [m for m in all_ret if m["strategy"] in key_strategies]
        if not bar_data:
            bar_data = all_ret[-4:]  # fallback to last 4
        bar_names = [m["strategy"] for m in bar_data]
        bar_f1_all = [m["f1"] for m in bar_data]
        x = np.arange(len(bar_names))
        width = 0.25

        bars_all = ax4.bar(x, bar_f1_all, width, label='All', color='#2196F3', alpha=0.8)
        for gc_i, gc in enumerate(sorted(gold_counts)):
            key = f"gold_{gc}"
            bar_f1_gc = [m.get(key, {}).get("f1", 0) for m in bar_data]
            offset = width * (gc_i + 1)
            colors = ['#FF9800', '#4CAF50', '#9C27B0', '#F44336']
            ax4.bar(x + offset, bar_f1_gc, width,
                    label=f'Gold={gc}', color=colors[gc_i % len(colors)], alpha=0.8)

        ax4.set_xlabel('Strategy')
        ax4.set_ylabel('F1 Score')
        ax4.set_title('F1 Comparison by Gold Count')
        ax4.set_xticks(x + width * len(gold_counts) / 2)
        ax4.set_xticklabels(bar_names, rotation=15, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        fig4.tight_layout()
        bar_path = os.path.join(ckpt_dir, "f1_comparison_bar.png")
        fig4.savefig(bar_path, dpi=150, bbox_inches='tight')
        plt.close(fig4)
        print(f"  Saved F1 bar chart → {bar_path}")

    except ImportError:
        print("  (matplotlib not installed — skipping plots; pip install matplotlib)")

    print("\n  Done. Run eval_llm.py next (with Ollama running) for LLM evaluation.")


if __name__ == "__main__":
    main(
        small="--small" in sys.argv,
        dataset="hotpot" if "--hotpot" in sys.argv else "2wiki",
        blind="--no-blind" not in sys.argv,  # default: blind=True
    )
