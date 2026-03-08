"""
Multi-Hop QA Pipeline: PPO for Paragraph Retrieval Selection.

Supports two datasets:
  - HotpotQA (distractor): 10 paragraphs (2 gold + 8 distractors)
  - 2WikiMultiHopQA:       10 paragraphs, chain-type (bridge) reasoning
                           that *requires* sequential reading (A → B)

The RL agent learns *which paragraphs to read* before answering,
receiving dense per-step reward for finding supporting facts.

Baselines:
  oracle      – read only gold paragraphs (upper bound)
  all_context – read all 10 paragraphs
  no_context  – answer with no context (lower bound)
  random      – read 3 random paragraphs
  greedy      – read 3 paragraphs with highest question-text overlap
  PPO         – learned retrieval policy (our method)
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict

import time

from multi_agent_baseline import (
    NUM_PARAGRAPHS, AgentStep, AgentTrajectory, RetrievalAgent,
)
from ppo_finetuner import TaskScorer, PPOFineTuner


# ======================================================================
#  Data loading
# ======================================================================

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: `datasets` not found. Using mock data.")


def load_hotpot_data(split: str = "train", max_examples: int = 40) -> Dict[str, Dict]:
    """Load HotpotQA (distractor) with parsed paragraph pools."""
    if not HAS_DATASETS:
        return _load_mock(max_examples)

    print(f"Loading HotpotQA ({split}, max {max_examples})...")
    dataset = load_dataset("hotpot_qa", "distractor", split=split)

    examples: Dict[str, Dict] = {}
    for i, item in enumerate(dataset):
        if i >= max_examples:
            break
        q_id = f"hotpot_{split}_{i}"

        # Parse context → list of (title, [sentences])
        titles = item["context"]["title"]
        sentences = item["context"]["sentences"]
        paragraphs = list(zip(titles, sentences))

        # Pad / truncate to NUM_PARAGRAPHS
        while len(paragraphs) < NUM_PARAGRAPHS:
            paragraphs.append(("(empty)", [""]))
        paragraphs = paragraphs[:NUM_PARAGRAPHS]

        supp_titles = set(item["supporting_facts"]["title"])

        examples[q_id] = {
            "question": item["question"],
            "answer": item["answer"],
            "paragraphs": paragraphs,
            "supporting_titles": supp_titles,
            "type": item.get("type", "bridge"),
            "level": item.get("level", "hard"),
        }

    print(f"✓ Loaded {len(examples)} HotpotQA examples")
    return examples


def load_2wiki_data(split: str = "train", max_examples: int = 40,
                    type_filter: str = "bridge") -> Dict[str, Dict]:
    """Load 2WikiMultiHopQA with the same paragraph-pool format as HotpotQA.

    Filters for bridge-type (chain reasoning) by default — these questions
    structurally require sequential reading (find entity A, then use A to
    locate B), which is the strongest justification for RL.
    """
    if not HAS_DATASETS:
        return _load_mock(max_examples)

    print(f"Loading 2WikiMultiHopQA ({split}, max {max_examples}, "
          f"type={type_filter or 'all'})...")
    try:
        dataset = load_dataset("framolfese/2WikiMultihopQA", split=split)
    except Exception as e:
        print(f"  Could not load framolfese/2WikiMultihopQA: {e}")
        print("  Falling back to HotpotQA...")
        return load_hotpot_data(split, max_examples)

    examples: Dict[str, Dict] = {}
    for i, item in enumerate(dataset):
        if len(examples) >= max_examples:
            break

        q_type = item.get("type", "")
        if type_filter and type_filter not in q_type:
            continue

        q_id = f"2wiki_{split}_{i}"

        try:
            ctx = item["context"]
            if isinstance(ctx, dict):
                titles = ctx["title"]
                sentences = ctx["sentences"]
                paragraphs = list(zip(titles, sentences))
            else:
                paragraphs = [(c[0], c[1]) for c in ctx]
        except Exception:
            continue

        while len(paragraphs) < NUM_PARAGRAPHS:
            paragraphs.append(("(empty)", [""]))
        paragraphs = paragraphs[:NUM_PARAGRAPHS]

        try:
            sf = item["supporting_facts"]
            if isinstance(sf, dict):
                supp_titles = set(sf["title"])
            else:
                supp_titles = set(s[0] if isinstance(s, (list, tuple)) else s
                                  for s in sf)
        except Exception:
            supp_titles = set()

        examples[q_id] = {
            "question": item["question"],
            "answer": item["answer"],
            "paragraphs": paragraphs,
            "supporting_titles": supp_titles,
            "type": q_type,
            "level": item.get("level", "hard"),
        }

    if len(examples) < max_examples // 2 and type_filter:
        print(f"  Only {len(examples)} '{type_filter}' examples found, "
              f"retrying without type filter...")
        return load_2wiki_data(split, max_examples, type_filter=None)

    print(f"✓ Loaded {len(examples)} 2WikiMultiHopQA examples "
          f"(type={type_filter or 'all'})")
    return examples


def filter_by_no_context(examples: Dict[str, Dict],
                        target_count: int = 40) -> Dict[str, Dict]:
    """Keep only questions the LLM cannot answer without context.

    Runs no_context on each candidate; drops questions answered correctly.
    This ensures retrieval actually matters for the remaining questions.
    """
    print(f"\n  Pre-filtering {len(examples)} questions (no-context check)...")
    scorer = setup_scorer(examples)
    agent = RetrievalAgent(agent_id=0, model="qwen3:8b")

    failed: Dict[str, Dict] = {}
    passed = 0
    for q_id, ex in examples.items():
        traj = agent.solve(
            q_id, ex["question"], ex["paragraphs"],
            ex["supporting_titles"], strategy="no_context",
        )
        score = scorer.score_answer(q_id, traj.final_answer or "")
        if score > 0.8:
            passed += 1
            tag = "SKIP (LLM knows)"
        else:
            failed[q_id] = ex
            tag = "KEEP"
        print(f"    {tag}: {ex['question'][:60]}")
        if len(failed) >= target_count:
            break

    print(f"  Pre-filter done: {passed} skipped (LLM knew), "
          f"{len(failed)} kept (need context)")
    if len(failed) < 10:
        print(f"  WARNING: only {len(failed)} hard questions found. "
              f"Consider increasing the candidate pool.")
    return failed


def _load_mock(n: int = 20) -> Dict[str, Dict]:
    """Minimal mock data for local testing."""
    print(f"Loading {n} mock examples...")
    _distractor_titles = [
        "Mathematics", "Physics", "Chemistry", "Biology",
        "History", "Geography", "Music", "Art",
    ]
    base = [
        {
            "question": "Which country is the birthplace of the author of 'One Hundred Years of Solitude'?",
            "answer": "Colombia",
            "supporting": [
                ("Gabriel García Márquez",
                 ["Gabriel García Márquez was a Colombian novelist, writer, and journalist."]),
                ("Colombia",
                 ["Colombia is a country in South America. It was the birthplace of many famous authors."]),
            ],
        },
        {
            "question": "What is the capital of the country where the Battle of Hastings took place?",
            "answer": "London",
            "supporting": [
                ("Battle of Hastings",
                 ["The Battle of Hastings was fought on 14 October 1066 in England."]),
                ("England",
                 ["England is a country in the United Kingdom. Its capital is London."]),
            ],
        },
        {
            "question": "Who directed the film that won Best Picture in 2020?",
            "answer": "Bong Joon-ho",
            "supporting": [
                ("Parasite (film)",
                 ["Parasite is a 2019 South Korean film directed by Bong Joon-ho.",
                  "It won the Academy Award for Best Picture in 2020."]),
                ("Bong Joon-ho",
                 ["Bong Joon-ho is a South Korean filmmaker."]),
            ],
        },
        {
            "question": "What river flows through the capital of France?",
            "answer": "Seine",
            "supporting": [
                ("Paris",
                 ["Paris is the capital city of France.",
                  "The Seine river flows through Paris."]),
                ("Seine",
                 ["The Seine is a 777-kilometre-long river in northern France."]),
            ],
        },
        {
            "question": "In what year was the discoverer of Pluto born?",
            "answer": "1906",
            "supporting": [
                ("Pluto",
                 ["Pluto was discovered in 1930 by Clyde Tombaugh."]),
                ("Clyde Tombaugh",
                 ["Clyde William Tombaugh was born on February 4, 1906."]),
            ],
        },
        {
            "question": "What is the nationality of the lead singer of the band that performed 'Paranoid Android'?",
            "answer": "English",
            "supporting": [
                ("Radiohead",
                 ["Radiohead is an English rock band formed in 1985.",
                  "They released 'Paranoid Android' in 1997."]),
                ("Thom Yorke",
                 ["Thomas Edward Yorke is an English musician and the lead singer of Radiohead."]),
            ],
        },
        {
            "question": "Which philosopher wrote about the ideal state in 'The Republic'?",
            "answer": "Plato",
            "supporting": [
                ("The Republic (Plato)",
                 ["The Republic is a Socratic dialogue by Plato.",
                  "It discusses the meaning of justice."]),
                ("Plato",
                 ["Plato was an ancient Greek philosopher."]),
            ],
        },
        {
            "question": "Who is the author of the book that inspired 'The Shawshank Redemption'?",
            "answer": "Stephen King",
            "supporting": [
                ("The Shawshank Redemption",
                 ["The Shawshank Redemption is a 1994 American film based on the novella by Stephen King."]),
                ("Stephen King",
                 ["Stephen Edwin King is an American author of horror and suspense."]),
            ],
        },
        {
            "question": "What is the currency of the country where the Pyramids of Giza are?",
            "answer": "Egyptian pound",
            "supporting": [
                ("Giza pyramid complex",
                 ["The Giza pyramid complex is located on the outskirts of Cairo, Egypt."]),
                ("Egypt",
                 ["Egypt uses the Egyptian pound as its official currency."]),
            ],
        },
        {
            "question": "Who was president of the United States when the first moon landing occurred?",
            "answer": "Richard Nixon",
            "supporting": [
                ("Apollo 11",
                 ["Apollo 11 was the spaceflight that first landed humans on the Moon on July 20, 1969."]),
                ("Richard Nixon",
                 ["Richard Milhous Nixon served as the 37th president of the United States from 1969 to 1974."]),
            ],
        },
        {
            "question": "What is the official language of the country where the Taj Mahal is located?",
            "answer": "Hindi",
            "supporting": [
                ("Taj Mahal",
                 ["The Taj Mahal is an ivory-white marble mausoleum in Agra, India."]),
                ("India",
                 ["India uses Hindi and English as its official languages."]),
            ],
        },
        {
            "question": "Which artist painted the ceiling of the Sistine Chapel?",
            "answer": "Michelangelo",
            "supporting": [
                ("Sistine Chapel ceiling",
                 ["The Sistine Chapel ceiling was painted by Michelangelo between 1508 and 1512."]),
                ("Michelangelo",
                 ["Michelangelo di Lodovico Buonarroti Simoni was an Italian sculptor and painter."]),
            ],
        },
        {
            "question": "In what country is the company that produces the iPhone headquartered?",
            "answer": "United States",
            "supporting": [
                ("iPhone",
                 ["The iPhone is a line of smartphones designed and marketed by Apple Inc."]),
                ("Apple Inc.",
                 ["Apple Inc. is an American multinational technology company headquartered in Cupertino, California."]),
            ],
        },
        {
            "question": "What is the elevation of the highest peak in the Himalayas?",
            "answer": "8848 metres",
            "supporting": [
                ("Himalayas",
                 ["The Himalayas is a mountain range in Asia. Mount Everest is its highest peak."]),
                ("Mount Everest",
                 ["Mount Everest has an elevation of 8,848 metres above sea level."]),
            ],
        },
        {
            "question": "What type of government does the country that launched Sputnik have today?",
            "answer": "Federal republic",
            "supporting": [
                ("Sputnik",
                 ["Sputnik was the first artificial Earth satellite, launched by the Soviet Union in 1957."]),
                ("Russia",
                 ["Russia, as successor to the Soviet Union, has a federal semi-presidential republic."]),
            ],
        },
        {
            "question": "Which novel by Jane Austen features the character Elizabeth Bennet?",
            "answer": "Pride and Prejudice",
            "supporting": [
                ("Elizabeth Bennet",
                 ["Elizabeth Bennet is the protagonist of Jane Austen's novel Pride and Prejudice."]),
                ("Pride and Prejudice",
                 ["Pride and Prejudice is a novel by Jane Austen, first published in 1813."]),
            ],
        },
        {
            "question": "Which composer wrote the symphony that features 'Ode to Joy'?",
            "answer": "Ludwig van Beethoven",
            "supporting": [
                ("Symphony No. 9 (Beethoven)",
                 ["The Symphony No. 9, composed by Ludwig van Beethoven, includes 'Ode to Joy'."]),
                ("Ludwig van Beethoven",
                 ["Ludwig van Beethoven was a German composer and pianist."]),
            ],
        },
        {
            "question": "What is the name of the currency used in the country where sushi originated?",
            "answer": "Yen",
            "supporting": [
                ("Sushi",
                 ["Sushi is a Japanese dish of prepared vinegared rice."]),
                ("Japan",
                 ["Japan uses the Japanese yen as its official currency."]),
            ],
        },
        {
            "question": "In which decade was the inventor of the World Wide Web born?",
            "answer": "1950s",
            "supporting": [
                ("World Wide Web",
                 ["The World Wide Web was invented by Tim Berners-Lee in 1989."]),
                ("Tim Berners-Lee",
                 ["Sir Timothy John Berners-Lee was born on 8 June 1955."]),
            ],
        },
        {
            "question": "What is the main ingredient in the pasta sauce 'Cacio e Pepe'?",
            "answer": "Pecorino cheese",
            "supporting": [
                ("Cacio e pepe",
                 ["Cacio e pepe is a Roman pasta dish.",
                  "The main ingredients are Pecorino Romano cheese and black pepper."]),
                ("Pecorino Romano",
                 ["Pecorino Romano is a hard Italian cheese made from sheep's milk."]),
            ],
        },
    ]

    examples: Dict[str, Dict] = {}
    for i, item in enumerate(base[:n]):
        q_id = f"hotpot_mock_{i}"
        # Build paragraph pool: 2 supporting + 8 distractors
        paragraphs = list(item["supporting"])
        supp_titles = {t for t, _ in item["supporting"]}
        used = set(supp_titles)
        for dt in _distractor_titles:
            if len(paragraphs) >= NUM_PARAGRAPHS:
                break
            if dt not in used:
                paragraphs.append((dt, [f"{dt} is a broad field of study."]))
                used.add(dt)
        # Shuffle so supporting facts aren't always first
        random.shuffle(paragraphs)
        while len(paragraphs) < NUM_PARAGRAPHS:
            paragraphs.append(("(empty)", [""]))

        examples[q_id] = {
            "question": item["question"],
            "answer": item["answer"],
            "paragraphs": paragraphs,
            "supporting_titles": supp_titles,
            "type": "bridge",
            "level": "medium",
        }
    print(f"✓ Loaded {len(examples)} mock examples")
    return examples


# ======================================================================
#  Evaluation helpers
# ======================================================================

def setup_scorer(examples: Dict[str, Dict]) -> TaskScorer:
    scorer = TaskScorer()
    for q_id, ex in examples.items():
        scorer.register_ground_truth(q_id, ex["answer"])
    return scorer


def run_baseline(examples: Dict[str, Dict], scorer: TaskScorer,
                 strategy: str, max_reads: int = 3,
                 label: str = "") -> Tuple[Dict, Dict[str, AgentTrajectory]]:
    """Run a fixed-strategy baseline on all examples.

    Returns (metrics_dict, {q_id: trajectory}).
    """
    label = label or strategy
    print(f"\n--- {label} ---")

    trajs: Dict[str, AgentTrajectory] = {}
    correct = 0
    total = 0
    total_reads = 0
    total_supp = 0

    for q_id, ex in examples.items():
        agent = RetrievalAgent(agent_id=0, model="qwen3:8b")
        traj = agent.solve(
            q_id, ex["question"], ex["paragraphs"],
            ex["supporting_titles"], strategy=strategy,
            max_reads=max_reads,
        )
        trajs[q_id] = traj
        score = scorer.score_answer(q_id, traj.final_answer or "")
        ok = score > 0.8
        total += 1
        if ok:
            correct += 1
        total_reads += traj.total_reads
        total_supp += traj.num_supporting_read

        tag = "✓" if ok else "✗"
        ans = (traj.final_answer or "N/A")[:40]
        print(f"  {tag} {ex['question'][:50]}...  → {ans}")

    acc = correct / max(1, total)
    avg_r = total_reads / max(1, total)
    avg_s = total_supp / max(1, total)
    print(f"  {label}: accuracy={acc:.1%}  avg_reads={avg_r:.1f}  "
          f"avg_supp={avg_s:.1f}")

    metrics = {
        "strategy": label,
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "avg_reads": avg_r,
        "avg_supporting_found": avg_s,
    }
    return metrics, trajs


def run_ppo_eval(examples: Dict[str, Dict], fine_tuner: PPOFineTuner,
                 scorer: TaskScorer,
                 max_steps: int = 5) -> Tuple[Dict, Dict[str, AgentTrajectory]]:
    """Evaluate the PPO-trained policy on examples."""
    print("\n--- PPO (ours) ---")

    trajs: Dict[str, AgentTrajectory] = {}
    correct = 0
    total = 0
    total_reads = 0
    total_supp = 0

    for q_id, ex in examples.items():
        agent = RetrievalAgent(agent_id=0, model="qwen3:8b")
        traj = agent.solve_with_policy(
            q_id, ex["question"], ex["paragraphs"],
            ex["supporting_titles"], policy=fine_tuner,
            max_steps=max_steps,
        )
        trajs[q_id] = traj
        score = scorer.score_answer(q_id, traj.final_answer or "")
        ok = score > 0.8
        total += 1
        if ok:
            correct += 1
        total_reads += traj.total_reads
        total_supp += traj.num_supporting_read

        tag = "✓" if ok else "✗"
        ans = (traj.final_answer or "N/A")[:40]
        print(f"  {tag} {ex['question'][:50]}...  → {ans}")

    acc = correct / max(1, total)
    avg_r = total_reads / max(1, total)
    avg_s = total_supp / max(1, total)
    print(f"  PPO: accuracy={acc:.1%}  avg_reads={avg_r:.1f}  "
          f"avg_supp={avg_s:.1f}")

    return {
        "strategy": "PPO (ours)",
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "avg_reads": avg_r,
        "avg_supporting_found": avg_s,
    }, trajs


def run_static_eval(examples: Dict[str, Dict], fine_tuner: PPOFineTuner,
                    scorer: TaskScorer,
                    k: int = 3) -> Tuple[Dict, Dict[str, AgentTrajectory]]:
    """Non-sequential ablation: same PPO model, but all reads decided at once.

    Picks the top-k paragraphs by the model's action logits from the
    initial state (step 0, nothing read).  No sequential state updates,
    so bridge-entity features are always zero — this isolates the value
    of sequential information gain.
    """
    label = f"Static Top-{k}"
    print(f"\n--- {label} (non-sequential ablation) ---")

    trajs: Dict[str, AgentTrajectory] = {}
    correct = 0
    total = 0
    total_reads = 0
    total_supp = 0

    for q_id, ex in examples.items():
        question = ex["question"]
        paragraphs = ex["paragraphs"]
        supp_titles = ex["supporting_titles"]
        n = min(len(paragraphs), NUM_PARAGRAPHS)

        titles_str = " | ".join(p[0] for p in paragraphs[:n])
        initial_context = (
            f"Task: {question}\n"
            f"Titles: {titles_str}\n"
            f"Read: Nothing yet\n"
            f"Step: 0"
        )

        ranked = fine_tuner.rank_paragraphs_static(initial_context, n)
        topk = ranked[:k]

        traj = AgentTrajectory(agent_id=0, task_id=q_id)
        for step_id, idx in enumerate(topk):
            title = paragraphs[idx][0]
            traj.steps.append(AgentStep(
                agent_id=0, step_id=step_id,
                action=f"read_{idx}", paragraph_idx=idx,
                paragraph_title=title,
                is_supporting=(title in supp_titles),
                already_read=False, timestamp=time.time(),
            ))

        read_paras = [(paragraphs[i][0], paragraphs[i][1]) for i in topk]
        agent = RetrievalAgent(agent_id=0, model="qwen3:8b")
        answer = agent._generate_answer(question, read_paras)
        traj.final_answer = answer

        traj.steps.append(AgentStep(
            agent_id=0, step_id=len(topk),
            action="answer", paragraph_idx=-1,
            paragraph_title="", is_supporting=False,
            already_read=False, timestamp=time.time(),
        ))

        traj.paragraphs_read = topk
        traj.num_supporting_read = sum(
            1 for i in topk if paragraphs[i][0] in supp_titles)
        traj.total_reads = len(topk)

        trajs[q_id] = traj
        score = scorer.score_answer(q_id, answer or "")
        ok = score > 0.8
        total += 1
        if ok:
            correct += 1
        total_reads += traj.total_reads
        total_supp += traj.num_supporting_read

        tag = "✓" if ok else "✗"
        ans = (answer or "N/A")[:40]
        print(f"  {tag} {ex['question'][:50]}...  → {ans}")

    acc = correct / max(1, total)
    avg_r = total_reads / max(1, total)
    avg_s = total_supp / max(1, total)
    print(f"  {label}: accuracy={acc:.1%}  avg_reads={avg_r:.1f}  "
          f"avg_supp={avg_s:.1f}")

    return {
        "strategy": label,
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "avg_reads": avg_r,
        "avg_supporting_found": avg_s,
    }, trajs


# ======================================================================
#  Reporting
# ======================================================================

def _box(title: str, w: int = 70):
    print(f"\n┌{'─' * (w-2)}┐")
    print(f"│{title:^{w-2}}│")
    print(f"└{'─' * (w-2)}┘")


def generate_report(baseline_results: List[Tuple[Dict, Dict[str, AgentTrajectory]]],
                    ppo_result: Tuple[Dict, Dict[str, AgentTrajectory]],
                    eval_examples: Dict[str, Dict],
                    scorer: TaskScorer,
                    iter_metrics: List[Dict]) -> List[str]:
    """Generate a detailed text report (returned as list of lines)."""
    L: List[str] = []
    ppo_metrics, ppo_trajs = ppo_result
    all_metrics = [m for m, _ in baseline_results] + [ppo_metrics]

    L.append("HOTPOT QA: PPO Paragraph Retrieval Selection")
    L.append("=" * 65)
    L.append("")

    # ---- Summary table ----
    L.append("┌──────────────────────────────────────────────────────────────────┐")
    L.append("│                         RESULTS SUMMARY                         │")
    L.append("└──────────────────────────────────────────────────────────────────┘")
    L.append("")
    L.append(f"  {'Strategy':<20} {'Accuracy':>10} {'Avg Reads':>10} {'Avg Supp':>10}")
    L.append(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10}")
    for m in all_metrics:
        L.append(f"  {m['strategy']:<20} {m['accuracy']:>9.1%} {m['avg_reads']:>10.1f} "
                 f"{m['avg_supporting_found']:>10.1f}")
    L.append("")

    # ---- Per-question detail ----
    L.append("┌──────────────────────────────────────────────────────────────────┐")
    L.append("│                       PER-QUESTION DETAIL                       │")
    L.append("└──────────────────────────────────────────────────────────────────┘")
    L.append("")

    q_ids = list(eval_examples.keys())
    header_parts = [f"{'#':<3}", f"{'Question':<30}", f"{'GT':<14}"]
    for m in all_metrics:
        short = m["strategy"][:6]
        header_parts.append(f"{short:>6}")
    L.append("  " + " ".join(header_parts))
    L.append("  " + "─" * (3 + 30 + 14 + 7 * len(all_metrics)))

    all_trajs_list = [t for _, t in baseline_results] + [ppo_trajs]

    for idx, q_id in enumerate(q_ids):
        ex = eval_examples[q_id]
        q_short = ex["question"][:28] + ".." if len(ex["question"]) > 28 else ex["question"]
        gt_short = ex["answer"][:12] + ".." if len(ex["answer"]) > 12 else ex["answer"]

        parts = [f"{idx+1:<3}", f"{q_short:<30}", f"{gt_short:<14}"]
        for trajs_dict in all_trajs_list:
            if q_id in trajs_dict:
                traj = trajs_dict[q_id]
                sc = scorer.score_answer(q_id, traj.final_answer or "")
                mark = "✓" if sc > 0.8 else "✗"
                parts.append(f"{mark:>6}")
            else:
                parts.append(f"{'?':>6}")
        L.append("  " + " ".join(parts))

        # Show PPO retrieval detail
        if q_id in ppo_trajs:
            pt = ppo_trajs[q_id]
            reads = [s.paragraph_title[:15] for s in pt.steps if s.action != "answer"]
            supp_marks = ["★" if s.is_supporting else "·" for s in pt.steps if s.action != "answer"]
            read_str = ", ".join(f"{r}({m})" for r, m in zip(reads, supp_marks))
            L.append(f"      PPO reads: {read_str if read_str else '(none)'}")
            L.append(f"      PPO answer: {(pt.final_answer or 'N/A')[:50]}")
    L.append("")

    # ---- Training curve ----
    if iter_metrics:
        L.append("┌──────────────────────────────────────────────────────────────────┐")
        L.append("│                       PPO TRAINING CURVE                        │")
        L.append("└──────────────────────────────────────────────────────────────────┘")
        L.append("")
        max_acc = max(m["accuracy"] for m in iter_metrics) or 0.01
        L.append(f"  {'Iter':>4}  {'Acc':>7}  {'Reads':>5}  {'Supp':>5}  {'':40}")
        L.append(f"  {'─'*4}  {'─'*7}  {'─'*5}  {'─'*5}  {'─'*40}")
        for m in iter_metrics:
            bar_len = int(m["accuracy"] / max_acc * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            L.append(f"  {m['iteration']:>4}  {m['accuracy']:>6.1%}  "
                     f"{m.get('avg_reads', 0):>5.1f}  "
                     f"{m.get('avg_supporting_found', 0):>5.1f}  "
                     f"{bar} {m['accuracy']:.1%}")
        L.append("")

        has_loss = any("training" in m and m["training"] for m in iter_metrics)
        if has_loss:
            L.append(f"  {'Iter':>4}  {'Policy Loss':>11}  {'Value Loss':>10}  {'Entropy':>8}")
            L.append(f"  {'─'*4}  {'─'*11}  {'─'*10}  {'─'*8}")
            for m in iter_metrics:
                t = m.get("training", {})
                if t:
                    hist = t.get("history", [])
                    if hist:
                        last = hist[-1]
                    else:
                        last = t
                    L.append(f"  {m['iteration']:>4}  "
                             f"{last.get('policy_loss', 0):>11.4f}  "
                             f"{last.get('value_loss', 0):>10.4f}  "
                             f"{last.get('entropy', 0):>8.4f}")
        L.append("")

    return L


# ======================================================================
#  Main pipeline
# ======================================================================

def main(small: bool = False, dataset: str = "2wiki"):
    ds_label = "2WikiMultiHopQA" if dataset == "2wiki" else "HotpotQA"
    _box(f"{ds_label}: PPO Paragraph Retrieval Selection")

    output_dir = "results"
    Path(output_dir).mkdir(exist_ok=True)

    if small:
        N_CANDIDATES, N_TARGET, N_EVAL, N_ITER = 30, 10, 3, 2
        print("*** SMALL MODE: reduced scale for quick testing ***")
    else:
        N_CANDIDATES, N_TARGET, N_EVAL, N_ITER = 200, 40, 20, 8

    # ---- Load data ----
    print(f"\n[1/7] Loading candidate data ({ds_label})...")
    if dataset == "2wiki":
        candidates = load_2wiki_data(split="train", max_examples=N_CANDIDATES,
                                     type_filter="bridge")
    else:
        candidates = load_hotpot_data(split="train", max_examples=N_CANDIDATES)

    # ---- Pre-filter: keep only questions LLM can't answer without context ----
    print("\n[2/7] Pre-filtering (removing questions LLM already knows)...")
    examples = filter_by_no_context(candidates, target_count=N_TARGET)

    ids = list(examples.keys())
    split_idx = max(1, len(ids) - N_EVAL)
    train_ids = ids[:split_idx]
    eval_ids = ids[split_idx:]
    train_examples = {k: examples[k] for k in train_ids}
    eval_examples = {k: examples[k] for k in eval_ids}
    print(f"  Train: {len(train_examples)},  Eval: {len(eval_examples)}")

    scorer = setup_scorer(examples)

    try:
        # ---- Baselines ----
        print("\n[3/7] Running baselines...")
        baseline_results: List[Tuple[Dict, Dict[str, AgentTrajectory]]] = []

        for strategy, label, reads in [
            ("oracle",      "Oracle",      3),
            ("no_context",  "No Context",  0),
            ("random",      "Random (3)",  3),
            ("greedy",      "Greedy (3)",  3),
        ]:
            m, t = run_baseline(eval_examples, scorer, strategy,
                                max_reads=reads, label=label)
            baseline_results.append((m, t))

        # ---- PPO Training ----
        print("\n[4/7] PPO training on train set...")
        fine_tuner = PPOFineTuner(scorer, device="cpu")
        iter_metrics = fine_tuner.on_policy_train(
            train_examples,
            num_iterations=N_ITER,
            max_steps=5,
            ppo_epochs=3,
            batch_size=16,
        )

        # ---- PPO Evaluation ----
        print("\n[5/7] Evaluating PPO on eval set...")
        ppo_result = run_ppo_eval(eval_examples, fine_tuner, scorer,
                                  max_steps=5)

        # ---- Static Top-K ablation (non-sequential, same model) ----
        print("\n[6/7] Static Top-K ablation (non-sequential)...")
        static_result = run_static_eval(eval_examples, fine_tuner, scorer,
                                        k=3)
        baseline_results.append(static_result)

        # ---- Report ----
        print("\n[7/7] Generating report...")

        _box("FINAL REPORT")
        all_metrics = [m for m, _ in baseline_results] + [ppo_result[0]]
        print(f"\n  {'Strategy':<20} {'Accuracy':>10} {'Avg Reads':>10} "
              f"{'Avg Supp':>10}")
        print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10}")
        for m in all_metrics:
            print(f"  {m['strategy']:<20} {m['accuracy']:>9.1%} "
                  f"{m['avg_reads']:>10.1f} "
                  f"{m['avg_supporting_found']:>10.1f}")

        # Save files
        fine_tuner.save_model(f"{output_dir}/hotpot_tool_selector.pt")
        fine_tuner.save_trajectories(f"{output_dir}/trajectories.json")
        fine_tuner.save_training_results(
            f"{output_dir}/training_results.json",
            baseline_metrics={m["strategy"]: m for m, _ in baseline_results},
        )

        comparison = {
            "baselines": [m for m, _ in baseline_results],
            "ppo": ppo_result[0],
            "training_curve": iter_metrics,
        }
        with open(f"{output_dir}/comparison.json", "w") as f:
            json.dump(comparison, f, indent=2, default=str)

        report_lines = generate_report(
            baseline_results, ppo_result,
            eval_examples, scorer, iter_metrics,
        )
        with open(f"{output_dir}/report.txt", "w") as f:
            f.write("\n".join(report_lines))

        print(f"\n  Results saved to {output_dir}/:")
        print(f"    report.txt               Detailed report")
        print(f"    hotpot_tool_selector.pt   PPO model weights")
        print(f"    trajectories.json         PPO trajectories")
        print(f"    training_results.json     Training metrics")
        print(f"    comparison.json           Full comparison data")
        print()

    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    main(
        small="--small" in sys.argv,
        dataset="hotpot" if "--hotpot" in sys.argv else "2wiki",
    )
