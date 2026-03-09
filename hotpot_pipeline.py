"""
Multi-Hop QA Pipeline: PPO for Paragraph Retrieval Selection.

Supports two datasets:
  - HotpotQA (distractor): 10 paragraphs (2 gold + 8 distractors)
  - 2WikiMultiHopQA:       10 paragraphs, chain-type compositional reasoning
                           that *requires* sequential reading (A → B)

The RL agent learns *which paragraphs to read* before answering,
receiving dense per-step reward for finding supporting facts.

All methods share the same read budget (K_BUDGET = 5).

Strategies:
  oracle      – read only gold paragraphs (upper bound, ignores budget)
  no_context  – answer with no context (lower bound)
  random      – read K random paragraphs
  static      – read top-K by initial-state logits (non-sequential ablation)
  PPO         – learned sequential retrieval policy (our method)
"""

import ast
import os
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict

import time


def _json_default(obj):
    """JSON serializer that handles Python sets (convert to sorted list)."""
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return str(obj)

from multi_agent_baseline import (
    NUM_PARAGRAPHS, AgentStep, AgentTrajectory, RetrievalAgent,
)
from ppo_finetuner import TaskScorer, PPOFineTuner


# ------------------------------------------------------------------
#  Baseline result serialisation helpers
# ------------------------------------------------------------------

def _serialize_baselines(
    baseline_results: List[Tuple[Dict, Dict[str, AgentTrajectory]]],
) -> List[Dict]:
    """Convert baseline (metrics, trajectories) pairs to JSON-safe dicts."""
    out = []
    for m, trajs in baseline_results:
        trajs_ser = {}
        for q_id, traj in trajs.items():
            trajs_ser[q_id] = asdict(traj)
        out.append({"metrics": m, "trajectories": trajs_ser})
    return out


def _deserialize_baselines(
    data: List[Dict],
) -> List[Tuple[Dict, Dict[str, AgentTrajectory]]]:
    """Reconstruct baseline results from JSON dicts."""
    results = []
    for entry in data:
        m = entry["metrics"]
        trajs = {}
        for q_id, td in entry.get("trajectories", {}).items():
            steps = [AgentStep(**s) for s in td.pop("steps", [])]
            trajs[q_id] = AgentTrajectory(**td, steps=steps)
        results.append((m, trajs))
    return results


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

        sf_titles = item["supporting_facts"]["title"]
        supp_titles = set(sf_titles)
        supporting_indices_ordered = []
        for title in sf_titles:
            for i, (t, _) in enumerate(paragraphs):
                if t == title:
                    supporting_indices_ordered.append(i)
                    break

        examples[q_id] = {
            "question": item["question"],
            "answer": item["answer"],
            "paragraphs": paragraphs,
            "supporting_titles": supp_titles,
            "supporting_indices_ordered": supporting_indices_ordered,
            "type": item.get("type", "bridge"),
            "level": item.get("level", "hard"),
        }

    print(f"✓ Loaded {len(examples)} HotpotQA examples")
    return examples


def load_2wiki_data(split: str = "train", max_examples: int = 40,
                    type_filter = "compositional") -> Dict[str, Dict]:
    """Load 2WikiMultiHopQA with the same paragraph-pool format as HotpotQA.

    type_filter can be a single string (e.g. "compositional") or a list
    of strings (e.g. ["compositional", "bridge_comparison"]) to load
    questions with varying gold-paragraph counts and reasoning chains.
    Pass None or '' to load all types.
    """
    if not HAS_DATASETS:
        return _load_mock(max_examples)

    if isinstance(type_filter, str):
        type_set = {type_filter} if type_filter else set()
    elif type_filter:
        type_set = set(type_filter)
    else:
        type_set = set()

    type_label = "+".join(sorted(type_set)) if type_set else "all"
    print(f"Loading 2WikiMultiHopQA ({split}, max {max_examples}, "
          f"type={type_label})...")
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
        if type_set and q_type not in type_set:
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
                titles_ordered = list(sf["title"])
                supp_titles = set(titles_ordered)
            else:
                titles_ordered = [s[0] if isinstance(s, (list, tuple)) else s
                                  for s in sf]
                supp_titles = set(titles_ordered)
        except Exception:
            titles_ordered = []
            supp_titles = set()

        # Paragraph indices in reasoning order (for order-aware reward)
        supporting_indices_ordered = []
        for title in titles_ordered:
            for i, (t, _) in enumerate(paragraphs):
                if t == title:
                    supporting_indices_ordered.append(i)
                    break

        examples[q_id] = {
            "question": item["question"],
            "answer": item["answer"],
            "paragraphs": paragraphs,
            "supporting_titles": supp_titles,
            "supporting_indices_ordered": supporting_indices_ordered,
            "type": q_type,
            "level": item.get("level", "hard"),
        }

    if len(examples) < max_examples // 2 and type_set:
        print(f"  Only {len(examples)} '{type_label}' examples found, "
              f"retrying without type filter...")
        return load_2wiki_data(split, max_examples, type_filter=None)

    gold_counts = [len(ex["supporting_titles"]) for ex in examples.values()]
    from collections import Counter
    gc_dist = Counter(gold_counts)
    gc_str = ", ".join(f"{k}-gold:{v}" for k, v in sorted(gc_dist.items()))
    avg_gold = sum(gold_counts) / max(1, len(gold_counts))
    print(f"✓ Loaded {len(examples)} 2WikiMultiHopQA examples "
          f"(type={type_filter or 'all'})")
    print(f"  Gold paragraph distribution: {gc_str}  (avg={avg_gold:.1f})")
    return examples


_PREFILTER_CACHE = "results/prefilter_cache.json"


def filter_by_no_context(examples: Dict[str, Dict],
                        target_count: int = 40) -> Dict[str, Dict]:
    """Keep only questions the LLM cannot answer without context.

    Runs no_context on each candidate; drops questions answered correctly.
    Results are cached to disk so repeated runs skip the LLM calls.
    """
    # --- try loading from cache ---
    cache: Dict[str, bool] = {}
    if os.path.isfile(_PREFILTER_CACHE):
        try:
            with open(_PREFILTER_CACHE) as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    all_cached = all(q_id in cache for q_id in examples)
    if all_cached and cache:
        print(f"\n  Pre-filter: loading cached results "
              f"({_PREFILTER_CACHE}, {len(cache)} entries)...")
        failed: Dict[str, Dict] = {}
        passed = 0
        for q_id, ex in examples.items():
            if cache.get(q_id, False):
                passed += 1
            else:
                failed[q_id] = ex
            if len(failed) >= target_count:
                break
        print(f"  Pre-filter (cached): {passed} skipped, "
              f"{len(failed)} kept")
        return failed

    # --- run LLM no-context check ---
    print(f"\n  Pre-filtering {len(examples)} questions (no-context check)...")
    scorer = setup_scorer(examples)
    agent = RetrievalAgent(agent_id=0, model="qwen3:8b")

    failed = {}
    passed = 0
    for q_id, ex in examples.items():
        if q_id in cache:
            llm_knows = cache[q_id]
        else:
            traj = agent.solve(
                q_id, ex["question"], ex["paragraphs"],
                ex["supporting_titles"], strategy="no_context",
            )
            score = scorer.score_answer(q_id, traj.final_answer or "")
            llm_knows = score > 0.8
            cache[q_id] = llm_knows

        if llm_knows:
            passed += 1
            tag = "SKIP (LLM knows)"
        else:
            failed[q_id] = ex
            tag = "KEEP"
        print(f"    {tag}: {ex['question'][:60]}")
        if len(failed) >= target_count:
            break

    # --- persist cache ---
    try:
        os.makedirs(os.path.dirname(_PREFILTER_CACHE) or ".", exist_ok=True)
        with open(_PREFILTER_CACHE, "w") as f:
            json.dump(cache, f)
        print(f"  Pre-filter cache saved ({len(cache)} entries)")
    except Exception as e:
        print(f"  Warning: could not save pre-filter cache: {e}")

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

        supporting_indices_ordered = []
        for (t, _) in item["supporting"]:
            for i, (tt, _) in enumerate(paragraphs):
                if tt == t:
                    supporting_indices_ordered.append(i)
                    break

        examples[q_id] = {
            "question": item["question"],
            "answer": item["answer"],
            "paragraphs": paragraphs,
            "supporting_titles": supp_titles,
            "supporting_indices_ordered": supporting_indices_ordered,
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


def _anonymize_titles(examples: Dict[str, Dict]) -> Dict[str, Dict]:
    """Replace real paragraph titles with 'Para_0' .. 'Para_9'.

    This removes the title-based information leak that benefits greedy
    baselines, forcing all methods to rely on paragraph *content* only.
    """
    out = {}
    for q_id, ex in examples.items():
        title_map = {}
        new_paras = []
        for i, (title, sents) in enumerate(ex["paragraphs"]):
            blind = f"Para_{i}"
            title_map[title] = blind
            new_paras.append((blind, sents))
        orig_supp = ex["supporting_titles"]
        if isinstance(orig_supp, list):
            orig_supp = set(orig_supp)
        new_supp = {title_map.get(t, t) for t in orig_supp}
        out[q_id] = {**ex, "paragraphs": new_paras,
                     "supporting_titles": new_supp}
    return out


def _compute_retrieval_metrics(total: int, total_reads: int,
                               total_supp: int, total_gold: int) -> Dict:
    avg_r = total_reads / max(1, total)
    avg_s = total_supp / max(1, total)
    precision = total_supp / max(1, total_reads)
    recall = total_supp / max(1, total_gold)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "avg_reads": avg_r,
        "avg_supporting_found": avg_s,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


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
    total_gold = 0

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
        total_gold += len(ex["supporting_titles"])

        tag = "✓" if ok else "✗"
        ans = (traj.final_answer or "N/A")[:40]
        print(f"  {tag} {ex['question'][:50]}...  → {ans}")

    acc = correct / max(1, total)
    rm = _compute_retrieval_metrics(total, total_reads, total_supp, total_gold)
    print(f"  {label}: acc={acc:.1%}  reads={rm['avg_reads']:.1f}  "
          f"supp={rm['avg_supporting_found']:.1f}  "
          f"P={rm['precision']:.1%}  R={rm['recall']:.1%}")

    metrics = {"strategy": label, "accuracy": acc,
               "correct": correct, "total": total, **rm}
    return metrics, trajs


def run_ppo_eval(examples: Dict[str, Dict], fine_tuner: PPOFineTuner,
                 scorer: TaskScorer,
                 max_steps: int = 3) -> Tuple[Dict, Dict[str, AgentTrajectory]]:
    """Evaluate the PPO-trained policy on examples."""
    print("\n--- PPO (ours) ---")

    trajs: Dict[str, AgentTrajectory] = {}
    correct = 0
    total = 0
    total_reads = 0
    total_supp = 0
    total_gold = 0

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
        total_gold += len(ex["supporting_titles"])

        tag = "✓" if ok else "✗"
        ans = (traj.final_answer or "N/A")[:40]
        print(f"  {tag} {ex['question'][:50]}...  → {ans}")

    acc = correct / max(1, total)
    rm = _compute_retrieval_metrics(total, total_reads, total_supp, total_gold)
    print(f"  PPO: acc={acc:.1%}  reads={rm['avg_reads']:.1f}  "
          f"supp={rm['avg_supporting_found']:.1f}  "
          f"P={rm['precision']:.1%}  R={rm['recall']:.1%}")

    return {"strategy": "PPO (ours)", "accuracy": acc,
            "correct": correct, "total": total, **rm}, trajs


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
    total_gold = 0

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
        total_gold += len(supp_titles)

        tag = "✓" if ok else "✗"
        ans = (answer or "N/A")[:40]
        print(f"  {tag} {ex['question'][:50]}...  → {ans}")

    acc = correct / max(1, total)
    rm = _compute_retrieval_metrics(total, total_reads, total_supp, total_gold)
    print(f"  {label}: acc={acc:.1%}  reads={rm['avg_reads']:.1f}  "
          f"supp={rm['avg_supporting_found']:.1f}  "
          f"P={rm['precision']:.1%}  R={rm['recall']:.1%}")

    return {"strategy": label, "accuracy": acc,
            "correct": correct, "total": total, **rm}, trajs


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
                    iter_metrics: List[Dict],
                    ds_label: str = "2WikiMultiHopQA",
                    budget: int = 3,
                    best_iter: int = None,
                    best_metric: float = None) -> List[str]:
    """Generate a detailed text report (returned as list of lines)."""
    L: List[str] = []
    ppo_metrics, ppo_trajs = ppo_result
    all_metrics = [m for m, _ in baseline_results] + [ppo_metrics]

    gold_counts = [len(ex["supporting_titles"]) for ex in eval_examples.values()]
    from collections import Counter
    gc_dist = Counter(gold_counts)
    avg_gold = sum(gold_counts) / max(1, len(gold_counts))

    L.append(f"{ds_label}: PPO Paragraph Retrieval Selection")
    L.append("=" * 65)
    L.append("")
    L.append(f"  Eval questions: {len(eval_examples)}")
    L.append(f"  Gold paragraphs per question: "
             + ", ".join(f"{k}({v})" for k, v in sorted(gc_dist.items()))
             + f"  avg={avg_gold:.1f}")
    L.append(f"  Read budget (all methods): {budget} max")
    if best_iter is not None and best_metric is not None:
        L.append(f"  PPO: early-stop best checkpoint at iteration {best_iter} "
                 f"(early-stop metric {best_metric:.1%})")
    L.append("")

    # ---- Summary table ----
    L.append("┌─────────────────────────────────────────────────────────────────────────────────────┐")
    L.append("│                                  RESULTS SUMMARY                                  │")
    L.append("└─────────────────────────────────────────────────────────────────────────────────────┘")
    L.append("")
    L.append(f"  {'Strategy':<20} {'Accuracy':>8} {'Reads':>6} {'Supp':>5} "
             f"{'Prec':>6} {'Recall':>7} {'F1':>6}")
    L.append(f"  {'─'*20} {'─'*8} {'─'*6} {'─'*5} {'─'*6} {'─'*7} {'─'*6}")
    for m in all_metrics:
        L.append(f"  {m['strategy']:<20} {m['accuracy']:>7.1%} "
                 f"{m['avg_reads']:>6.1f} {m['avg_supporting_found']:>5.1f} "
                 f"{m.get('precision', 0):>5.0%} "
                 f"{m.get('recall', 0):>6.0%} "
                 f"{m.get('f1', 0):>5.0%}")
    L.append("")
    L.append("  Precision = supporting_read / total_reads  (read efficiency)")
    L.append("  Recall    = supporting_read / total_gold   (coverage of gold)")
    L.append("  F1        = harmonic mean of Precision and Recall")
    L.append("")

    # ---- Per-category breakdown ----
    all_trajs_list = [t for _, t in baseline_results] + [ppo_trajs]
    categories: Dict[str, List[str]] = defaultdict(list)
    for q_id, ex in eval_examples.items():
        q_type = ex.get("type", "unknown")
        n_gold = len(ex.get("supporting_titles", []))
        cat = f"{q_type} ({n_gold}-gold)"
        categories[cat].append(q_id)

    if len(categories) > 1:
        L.append("┌─────────────────────────────────────────────────────────────────────────────────────┐")
        L.append("│                              PER-CATEGORY BREAKDOWN                                │")
        L.append("└─────────────────────────────────────────────────────────────────────────────────────┘")
        L.append("")
        for cat in sorted(categories.keys()):
            cat_ids = categories[cat]
            L.append(f"  ▸ {cat}  (n={len(cat_ids)})")
            L.append(f"    {'Strategy':<20} {'Accuracy':>8} {'Reads':>6} {'Supp':>5} "
                     f"{'Prec':>6} {'Recall':>7} {'F1':>6}")
            L.append(f"    {'─'*20} {'─'*8} {'─'*6} {'─'*5} {'─'*6} {'─'*7} {'─'*6}")
            for mi, m in enumerate(all_metrics):
                trajs_dict = all_trajs_list[mi]
                cat_correct = 0
                cat_total = 0
                cat_reads = 0
                cat_supp = 0
                cat_gold = 0
                for q_id in cat_ids:
                    ex = eval_examples[q_id]
                    cat_total += 1
                    cat_gold += len(ex["supporting_titles"])
                    if q_id in trajs_dict:
                        traj = trajs_dict[q_id]
                        sc = scorer.score_answer(q_id, traj.final_answer or "")
                        if sc > 0.8:
                            cat_correct += 1
                        read_steps = [s for s in traj.steps if s.action != "answer"]
                        cat_reads += len(read_steps)
                        cat_supp += sum(1 for s in read_steps if s.is_supporting)
                cat_acc = cat_correct / max(1, cat_total)
                cr = _compute_retrieval_metrics(cat_total, cat_reads, cat_supp, cat_gold)
                L.append(f"    {m['strategy']:<20} {cat_acc:>7.1%} "
                         f"{cr['avg_reads']:>6.1f} {cr['avg_supporting_found']:>5.1f} "
                         f"{cr.get('precision', 0):>5.0%} "
                         f"{cr.get('recall', 0):>6.0%} "
                         f"{cr.get('f1', 0):>5.0%}")
            L.append("")
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

        # Show retrieval detail for each strategy
        for mi, trajs_dict in enumerate(all_trajs_list):
            if q_id in trajs_dict:
                traj = trajs_dict[q_id]
                strat_name = all_metrics[mi]["strategy"]
                read_steps = [s for s in traj.steps if s.action != "answer"]
                reads = [s.paragraph_title[:15] for s in read_steps]
                supp_marks = ["★" if s.is_supporting else "·" for s in read_steps]
                read_str = ", ".join(f"{r}({mk})" for r, mk in zip(reads, supp_marks))
                n_supp = sum(1 for s in read_steps if s.is_supporting)
                n_total = len(read_steps)
                eff = f"{n_supp}/{n_total}" if n_total else "0/0"
                L.append(f"      {strat_name:<16} [{eff}] {read_str if read_str else '(none)'}")
    L.append("")

    # ---- Training curve ----
    if iter_metrics:
        L.append("┌──────────────────────────────────────────────────────────────────┐")
        L.append("│                       PPO TRAINING CURVE                        │")
        L.append("└──────────────────────────────────────────────────────────────────┘")
        L.append("")
        max_acc = max(m["accuracy"] for m in iter_metrics) or 0.01
        L.append(f"  {'Iter':>4}  {'Acc':>7}  {'Reads':>5}  {'Supp':>5}  "
                 f"{'Prec':>6}  {'Recall':>7}  {'':30}")
        L.append(f"  {'─'*4}  {'─'*7}  {'─'*5}  {'─'*5}  "
                 f"{'─'*6}  {'─'*7}  {'─'*30}")
        for m in iter_metrics:
            bar_len = int(m["accuracy"] / max_acc * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            L.append(f"  {m['iteration']:>4}  {m['accuracy']:>6.1%}  "
                     f"{m.get('avg_reads', 0):>5.1f}  "
                     f"{m.get('avg_supporting_found', 0):>5.1f}  "
                     f"{m.get('precision', 0):>5.0%}  "
                     f"{m.get('recall', 0):>6.0%}  "
                     f"{bar}")
        L.append("")

        # Reward / Return per iteration
        if any("mean_return" in m for m in iter_metrics):
            L.append("  Reward & return (rollout):")
            L.append(f"  {'Iter':>4}  {'Mean Return':>12}  {'Avg Step Reward':>16}")
            L.append(f"  {'─'*4}  {'─'*12}  {'─'*16}")
            for m in iter_metrics:
                L.append(f"  {m['iteration']:>4}  "
                         f"{m.get('mean_return', 0):>12.3f}  "
                         f"{m.get('avg_step_reward', 0):>16.4f}")
            L.append("")

        has_loss = any("training" in m and m["training"] for m in iter_metrics)
        if has_loss:
            L.append(f"  {'Iter':>4}  {'Policy Loss':>11}  {'Value Loss':>10}  {'Entropy':>8}  {'KL(BC)':>8}")
            L.append(f"  {'─'*4}  {'─'*11}  {'─'*10}  {'─'*8}  {'─'*8}")
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
                             f"{last.get('entropy', 0):>8.4f}  "
                             f"{last.get('kl_from_bc', 0):>8.4f}")
        L.append("")

    return L


# ======================================================================
#  Main pipeline
# ======================================================================

def main(small: bool = False, dataset: str = "2wiki",
         resume: str = None, blind: bool = False,
         ppo_only: bool = False,
         bc_oracle: bool = False,
         run_name: str = ""):
    """Run the full pipeline.

    Args:
        small: reduced data scale for quick testing.
        dataset: "2wiki" or "hotpot".
        resume: path to a checkpoint file (e.g. "checkpoints/ckpt_iter_003.pt")
                to resume PPO training from.
        blind: if True, anonymise paragraph titles to "Para_0" .. "Para_9",
               removing title-based information leaks that benefit greedy.
        ppo_only: if True, skip prefilter and baselines, go directly to BC+PPO.
        bc_oracle: if True, load split only (no prefilter/baseline), BC from ground-truth oracle, PPO 3 iters; save to report_oracle_bc.txt (no overwrite).
        run_name: if set (e.g. "oracle_bc"), all outputs go to results_<run_name>/ and checkpoints_blind_<run_name>/ (new folder, from-scratch run with prefilter+baselines then oracle BC+PPO when combined with --bc-oracle).
    """
    mode_tag = " [BLIND TITLES]" if blind else ""
    ds_label = "2WikiMultiHopQA" if dataset == "2wiki" else "HotpotQA"
    _box(f"{ds_label}: PPO Paragraph Retrieval Selection{mode_tag}")

    if run_name:
        output_dir = f"results_{run_name}"
        ckpt_dir = f"checkpoints_blind_{run_name}" if blind else f"checkpoints_{run_name}"
        print(f"\n*** RUN NAME: {run_name} → all outputs in {output_dir}/ and {ckpt_dir}/ ***")
    else:
        output_dir = "results"
        ckpt_dir = "checkpoints_blind" if blind else "checkpoints"
    Path(output_dir).mkdir(exist_ok=True)
    Path(ckpt_dir).mkdir(exist_ok=True)

    K_BUDGET = 5
    split_path = os.path.join(ckpt_dir, "split.json")
    baselines_path = os.path.join(ckpt_dir, "baselines.json")

    # Baselines (blind mode only in this project):
    # - Oracle: gold paragraphs only (upper bound on retrieval)
    # - No Context: answer with no paragraphs (lower bound)
    # - Random (2/3/4): K random paragraphs
    BASELINE_SPECS = [
        ("oracle",      "Oracle",        0),
        ("no_context",  "No Context",    0),
        ("random",      "Random (2)",    2),
        ("random",      "Random (3)",    3),
        ("random",      "Random (4)",    4),
    ]

    if small:
        N_CANDIDATES, N_TARGET, N_EVAL, N_ITER = 30, 10, 3, 2
        print("*** SMALL MODE: reduced scale for quick testing ***")
    else:
        N_CANDIDATES, N_TARGET, N_EVAL, N_ITER = 400, 120, 20, 10

    # ------------------------------------------------------------------
    # Resume fast-path: reuse saved split + baselines from previous run
    # ------------------------------------------------------------------
    if ppo_only:
        print("\n*** PPO-ONLY MODE: skipping prefilter and baselines ***")
    if bc_oracle and not run_name:
        print("\n*** BC-ORACLE MODE: load split only, BC from ground truth, PPO 3 iters, save to *_oracle_bc.* ***")
        if not os.path.isfile(split_path):
            raise FileNotFoundError(
                f"--bc-oracle requires existing split: {split_path} not found. Run once without --bc-oracle to create it."
            )

    # With run_name: always run from scratch (prefilter + baselines), then oracle BC + PPO into new folder
    if (resume or (bc_oracle and not run_name)) and os.path.isfile(split_path):
        print(f"\n[1/6] Loading saved data split from {split_path}...")
        with open(split_path) as f:
            saved = json.load(f)
        train_examples = saved["train"]
        eval_examples = saved["eval"]
        for d in (train_examples, eval_examples):
            for ex in d.values():
                st = ex["supporting_titles"]
                if isinstance(st, str):
                    # Legacy bugfix: old splits saved sets via str(), producing
                    # "{'Para_1', 'Para_3'}".  Parse it back properly.
                    try:
                        st = ast.literal_eval(st)
                    except (ValueError, SyntaxError):
                        st = set()
                ex["supporting_titles"] = set(st) if not isinstance(st, set) else st
        examples = {**train_examples, **eval_examples}
        from collections import Counter
        print(f"  Train: {len(train_examples)},  Eval: {len(eval_examples)}")
        post_gc = Counter(len(ex["supporting_titles"]) for ex in examples.values())
        print(f"  Gold distribution: "
              + ", ".join(f"{k}-gold:{v}" for k, v in sorted(post_gc.items())))

        scorer = setup_scorer(examples)

        # Search for baselines.json in current ckpt_dir and known alternate dirs
        baseline_results: List[Tuple[Dict, Dict[str, AgentTrajectory]]] = []
        bl_search_paths = [baselines_path]
        # Also look in oracle_bc variant directory (shares same eval split)
        alt_ckpt = ckpt_dir + "_oracle_bc"
        bl_search_paths.append(os.path.join(alt_ckpt, "baselines.json"))

        bl_found = None
        for bp in bl_search_paths:
            if os.path.isfile(bp):
                bl_found = bp
                break

        if bl_found:
            print(f"\n[2/6] Loading saved baseline results from {bl_found}...")
            with open(bl_found) as f:
                saved_bl = json.load(f)
            baseline_results = _deserialize_baselines(saved_bl)
            for m, _ in baseline_results:
                print(f"  {m['strategy']:<20} acc={m['accuracy']:.1%} "
                      f"reads={m['avg_reads']:.1f}")
            # Copy to local baselines_path if not already there
            if bl_found != baselines_path:
                with open(baselines_path, "w") as f:
                    json.dump(saved_bl, f, indent=2, default=_json_default)
                print(f"  (Copied to {baselines_path})")
        elif ppo_only:
            print("\n[2/6] No baselines.json found, skipping baselines (ppo_only mode)...")
        else:
            print("\n[3/6] Running baselines...")
            for strategy, label, reads in BASELINE_SPECS:
                m, t = run_baseline(eval_examples, scorer, strategy,
                                    max_reads=reads, label=label)
                baseline_results.append((m, t))
            with open(baselines_path, "w") as f:
                json.dump(_serialize_baselines(baseline_results), f,
                          indent=2, default=_json_default)

    else:
        # ------------------------------------------------------------------
        # Normal path: load data, prefilter, split, run baselines
        # ------------------------------------------------------------------
        print(f"\n[1/6] Loading candidate data ({ds_label})...")
        if dataset == "2wiki":
            candidates = load_2wiki_data(
                split="train", max_examples=N_CANDIDATES,
                type_filter=["compositional", "bridge_comparison"])
        else:
            candidates = load_hotpot_data(split="train",
                                          max_examples=N_CANDIDATES)

        cand_ids = list(candidates.keys())
        random.shuffle(cand_ids)
        candidates = {k: candidates[k] for k in cand_ids}

        gold_counts = [len(ex["supporting_titles"])
                       for ex in candidates.values()]
        from collections import Counter
        gc = Counter(gold_counts)
        print(f"  Candidate gold distribution: "
              + ", ".join(f"{k}-gold:{v}" for k, v in sorted(gc.items())))

        # ---- Pre-filter (skipped in ppo_only) ----
        if ppo_only:
            print("\n[2/6] Skipping prefilter (ppo_only mode)...")
            examples = dict(candidates)
            # Use up to N_TARGET examples
            ids = list(examples.keys())
            if len(ids) > N_TARGET:
                random.shuffle(ids)
                examples = {k: examples[k] for k in ids[:N_TARGET]}
        else:
            print("\n[2/6] Pre-filtering (removing questions LLM already knows)...")
            n_target = N_TARGET
            n_pref = max(1, int(0.6 * n_target))
            pref_examples = filter_by_no_context(candidates, target_count=n_pref)

            remaining_ids = [k for k in cand_ids if k not in pref_examples]
            examples: Dict[str, Dict] = dict(pref_examples)
            for k in remaining_ids:
                if len(examples) >= n_target:
                    break
                examples[k] = candidates[k]

        post_gc = Counter(len(ex["supporting_titles"])
                          for ex in examples.values())
        print(f"  Post-filter gold distribution: "
              + ", ".join(f"{k}-gold:{v}" for k, v in sorted(post_gc.items())))

        # ---- Split & persist ----
        ids = list(examples.keys())
        random.shuffle(ids)
        n_total = len(ids)
        n_eval = min(N_EVAL, n_total - 1)
        eval_ids = ids[:n_eval]
        train_ids = ids[n_eval:]
        train_examples = {k: examples[k] for k in train_ids}
        eval_examples = {k: examples[k] for k in eval_ids}

        if blind:
            print("  Anonymising titles (--blind mode)...")
            train_examples = _anonymize_titles(train_examples)
            eval_examples = _anonymize_titles(eval_examples)

        print(f"  Train: {len(train_examples)},  Eval: {len(eval_examples)}")

        with open(split_path, "w") as f:
            json.dump({"train": train_examples, "eval": eval_examples},
                      f, indent=2, default=_json_default)
        print(f"  Saved split to {split_path}")

        scorer = setup_scorer(examples)

        # ---- Baselines (skipped in ppo_only) ----
        baseline_results: List[Tuple[Dict, Dict[str, AgentTrajectory]]] = []
        if ppo_only:
            print("\n[3/6] Skipping baselines (ppo_only mode)...")
        else:
            print("\n[3/6] Running baselines...")
            for strategy, label, reads in BASELINE_SPECS:
                m, t = run_baseline(eval_examples, scorer, strategy,
                                    max_reads=reads, label=label)
                baseline_results.append((m, t))

            with open(baselines_path, "w") as f:
                json.dump(_serialize_baselines(baseline_results), f,
                          indent=2, default=_json_default)
            print(f"  Saved baselines to {baselines_path}")

    # BC expert:
    # - bc_oracle / run_name: behavior-clone from oracle (gold read order)
    # - otherwise: behavior-clone from Greedy-ST (strong non-RL baseline)
    n_ppo_iters = N_ITER
    bc_expert = "oracle" if (bc_oracle or run_name) else "greedy_st"
    out_suffix = "_oracle_bc" if (bc_oracle and not run_name) else ""

    try:
        # ---- PPO Training ----
        if resume and not bc_oracle:
            print(f"\n[4/6] PPO training RESUMED from {resume} "
                  f"(budget={K_BUDGET})...")
        else:
            print(f"\n[4/6] PPO training on train set (budget={K_BUDGET}, iters={n_ppo_iters}, "
                  f"bc=greedy_st(3), bc_ep=20, lr=1e-5, ent=0.01, kl=0.2)...")
        fine_tuner = PPOFineTuner(scorer, device="cpu", blind=blind,
                                  lr=1e-5, entropy_coeff=0.01, kl_coeff=0.2)
        iter_metrics, bc_only_result, best_iter, best_metric = fine_tuner.on_policy_train(
            train_examples,
            num_iterations=n_ppo_iters,
            max_steps=K_BUDGET,
            ppo_epochs=3,
            batch_size=16,
            checkpoint_dir=ckpt_dir,
            resume_from=resume if not bc_oracle else None,
            eval_examples=eval_examples,
            bc_epochs=20,
            bc_expert=bc_expert,
            scorer=scorer,
            bc_fraction=1.0,
            bc_max_reads=3,
        )
        if bc_only_result is not None:
            bc_metrics, bc_trajs = bc_only_result
            baseline_results.append((bc_metrics, bc_trajs))
            print(f"\n  Added BC-only baseline: acc={bc_metrics['accuracy']:.1%}")

        # ---- PPO Evaluation ----
        print(f"\n[5/6] Evaluating PPO on eval set (budget={K_BUDGET})...")
        ppo_result = run_ppo_eval(eval_examples, fine_tuner, scorer,
                                  max_steps=K_BUDGET)
        ppo_metrics, ppo_trajs = ppo_result

        # Label PPO row with best-iteration information (no metric fallback).
        ppo_metrics["strategy"] = f"PPO (best @ iter {best_iter})"
        ppo_metrics["best_iter"] = best_iter
        ppo_metrics["best_metric"] = best_metric
        ppo_result = (ppo_metrics, ppo_trajs)

        # ---- Report: ensure baselines are included when file exists ----
        if not baseline_results:
            for try_path in [baselines_path, os.path.join(ckpt_dir + "_oracle_bc", "baselines.json")]:
                if os.path.isfile(try_path):
                    with open(try_path) as f:
                        baseline_results = _deserialize_baselines(json.load(f))
                    print(f"\n  Loaded baselines from {try_path} for report.")
                    break

        print("\n  Generating report...")
        _box("FINAL REPORT")
        all_metrics = [m for m, _ in baseline_results] + [ppo_result[0]]
        print(f"\n  {'Strategy':<20} {'Acc':>7} {'Reads':>6} {'Supp':>5} "
              f"{'Prec':>6} {'Recall':>7} {'F1':>6}")
        print(f"  {'─'*20} {'─'*7} {'─'*6} {'─'*5} {'─'*6} {'─'*7} {'─'*6}")
        for m in all_metrics:
            print(f"  {m['strategy']:<20} {m['accuracy']:>6.1%} "
                  f"{m['avg_reads']:>6.1f} {m['avg_supporting_found']:>5.1f} "
                  f"{m.get('precision', 0):>5.0%} "
                  f"{m.get('recall', 0):>6.0%} "
                  f"{m.get('f1', 0):>5.0%}")

        # Save files (oracle_bc uses separate names so existing results are not overwritten)
        fine_tuner.save_model(f"{output_dir}/hotpot_tool_selector{out_suffix}.pt")
        fine_tuner.save_trajectories(f"{output_dir}/trajectories{out_suffix}.json")
        fine_tuner.save_training_results(
            f"{output_dir}/training_results{out_suffix}.json",
            baseline_metrics={m["strategy"]: m for m, _ in baseline_results},
        )

        comparison = {
            "baselines": [m for m, _ in baseline_results],
            "ppo": ppo_result[0],
            "training_curve": iter_metrics,
        }
        with open(f"{output_dir}/comparison{out_suffix}.json", "w") as f:
            json.dump(comparison, f, indent=2, default=_json_default)

        report_lines = generate_report(
            baseline_results, ppo_result,
            eval_examples, scorer, iter_metrics,
            ds_label=ds_label, budget=K_BUDGET,
            best_iter=best_iter, best_metric=best_metric,
        )
        report_path = f"{output_dir}/report{out_suffix}.txt"
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))

        print(f"\n  Results saved to {output_dir}/:")
        print(f"    {os.path.basename(report_path):<32} Detailed report")
        print(f"    hotpot_tool_selector{out_suffix}.pt   PPO model weights")
        print(f"    trajectories{out_suffix}.json         PPO trajectories")
        print(f"    training_results{out_suffix}.json     Training metrics")
        print(f"    comparison{out_suffix}.json          Full comparison data")
        print()

    except KeyboardInterrupt:
        print("\n\nInterrupted — saving emergency checkpoint...")
        try:
            fine_tuner.save_checkpoint(
                f"{ckpt_dir}/ckpt_interrupted.pt", -1, iter_metrics)
            print(f"  Saved {ckpt_dir}/ckpt_interrupted.pt  "
                  f"(resume with --resume={ckpt_dir}/ckpt_interrupted.pt)")
        except Exception:
            pass
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    _resume_path = None
    for _a in sys.argv:
        if _a.startswith("--resume="):
            _resume_path = _a.split("=", 1)[1]
        elif _a == "--resume":
            _idx = sys.argv.index("--resume")
            if _idx + 1 < len(sys.argv):
                _resume_path = sys.argv[_idx + 1]
    _run_name = ""
    for _a in sys.argv:
        if _a.startswith("--run-name="):
            _run_name = _a.split("=", 1)[1].strip()
            break
        if _a == "--run-name" and "--run-name" in sys.argv:
            _idx = sys.argv.index("--run-name")
            if _idx + 1 < len(sys.argv):
                _run_name = sys.argv[_idx + 1].strip()
            break
    main(
        small="--small" in sys.argv,
        dataset="hotpot" if "--hotpot" in sys.argv else "2wiki",
        resume=_resume_path,
        blind="--blind" in sys.argv,
        ppo_only="--ppo-only" in sys.argv,
        bc_oracle="--bc-oracle" in sys.argv,
        run_name=_run_name,
    )