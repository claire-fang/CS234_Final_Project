"""
HotpotQA Pipeline: Train and evaluate on real QA dataset.

Uses HotpotQA (multi-hop reasoning questions) to test whether PPO fine-tuning
for tool selection actually improves agent performance on realistic tasks.

Dataset: https://hotpotqa.github.io/
- Questions requiring multi-hop reasoning
- ~113k questions in full dataset
- We use ~20 train + ~20 eval examples for quick iteration
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import requests

# Try to load HotpotQA from huggingface
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: datasets library not found. Install with: pip install datasets")

from multi_agent_baseline import MultiAgentBaseline, AgentTrajectory, LocalLLMAgent
from ppo_finetuner import TaskScorer, PPOFineTuner


def load_hotpot_data(split: str = "train", max_examples: int = 20) -> Dict[str, Dict]:
    """
    Load HotpotQA examples from huggingface.
    
    Args:
        split: "train" or "validation"
        max_examples: Number of examples to load
    
    Returns:
        Dict mapping question_id -> {"question": str, "answer": str, "supporting_facts": ...}
    """
    if not HAS_DATASETS:
        print("Loading from local mock data instead...")
        return load_hotpot_mock(max_examples)
    
    print(f"Loading HotpotQA ({split} split, max {max_examples} examples)...")
    
    dataset = load_dataset("hotpot_qa", "distractor", split=split)
    
    examples = {}
    
    for i, item in enumerate(dataset):
        if i >= max_examples:
            break
        
        q_id = f"hotpot_{split}_{i}"
        
        # Extract answer (can be span or yes/no)
        answer = item.get("answer", "")
        
        examples[q_id] = {
            "question": item.get("question", ""),
            "answer": answer,
            "supporting_facts": item.get("supporting_facts", []),
            "type": item.get("type", "comparison"),  # "comparison" or "bridge"
            "level": item.get("level", "hard"),  # "easy", "medium", "hard"
            "context": item.get("context", []),  # List of [title, sentences]
            "raw_item": item,
        }
    
    print(f"✓ Loaded {len(examples)} HotpotQA examples")
    return examples


def load_hotpot_mock(num_examples: int = 20) -> Dict[str, Dict]:
    """
    Load mock HotpotQA data (for testing without datasets library).
    
    These are multi-hop reasoning questions that require multiple
    search steps to answer. The RL problem is learning the optimal
    tool-call *sequence* and *when to stop searching*.
    """
    print(f"Loading mock HotpotQA data ({num_examples} examples)...")
    
    mock_data = [
        {
            "question": "Which country is the birthplace of the author of \"One Hundred Years of Solitude\"?",
            "answer": "Colombia",
            "type": "bridge",
            "level": "medium",
        },
        {
            "question": "What is the birth date of the lead singer of the band that performed \"Paranoid Android\"?",
            "answer": "October 7, 1968",
            "type": "bridge",
            "level": "hard",
        },
        {
            "question": "In what year was the discoverer of Pluto born?",
            "answer": "1906",
            "type": "bridge",
            "level": "medium",
        },
        {
            "question": "What is the capital of the country where the Battle of Hastings took place?",
            "answer": "London",
            "type": "bridge",
            "level": "easy",
        },
        {
            "question": "Who is the author of the book that inspired the movie \"The Shawshank Redemption\"?",
            "answer": "Stephen King",
            "type": "bridge",
            "level": "easy",
        },
        {
            "question": "What is the elevation of the highest peak in the Himalayas in meters?",
            "answer": "8848",
            "type": "bridge",
            "level": "medium",
        },
        {
            "question": "In which city is the university where Albert Einstein taught located?",
            "answer": "Berlin",
            "type": "bridge",
            "level": "hard",
        },
        {
            "question": "What is the population of the country where the Library of Congress is located?",
            "answer": "331 million",
            "type": "bridge",
            "level": "hard",
        },
        {
            "question": "Which Greek philosopher wrote about the ideal state in \"The Republic\"?",
            "answer": "Plato",
            "type": "comparison",
            "level": "easy",
        },
        {
            "question": "What year did the war that inspired George Orwell's \"1984\" end?",
            "answer": "1945",
            "type": "bridge",
            "level": "medium",
        },
        {
            "question": "Who is the playwright of the work often considered the greatest in English literature?",
            "answer": "William Shakespeare",
            "type": "comparison",
            "level": "easy",
        },
        {
            "question": "In what country is the company that produces the iPhone headquartered?",
            "answer": "United States",
            "type": "bridge",
            "level": "easy",
        },
        {
            "question": "What is the scientific name of the animal that is the symbol of World Wildlife Fund?",
            "answer": "Ailuropoda melanoleuca",
            "type": "bridge",
            "level": "hard",
        },
        {
            "question": "Which composer wrote the \"Symphony No. 9\" that features \"Ode to Joy\"?",
            "answer": "Ludwig van Beethoven",
            "type": "comparison",
            "level": "easy",
        },
        {
            "question": "What is the capital of the country where the Great Wall is located?",
            "answer": "Beijing",
            "type": "bridge",
            "level": "easy",
        },
        {
            "question": "Which novel by Jane Austen features the character Elizabeth Bennet?",
            "answer": "Pride and Prejudice",
            "type": "comparison",
            "level": "easy",
        },
        {
            "question": "In what year did the country where the Colosseum is located become a republic?",
            "answer": "1946",
            "type": "bridge",
            "level": "hard",
        },
        {
            "question": "What is the name of the currency used in the country where sushi originated?",
            "answer": "Yen",
            "type": "bridge",
            "level": "easy",
        },
        {
            "question": "Who directed the film that won the Academy Award for Best Picture in 2020?",
            "answer": "Bong Joon-ho",
            "type": "bridge",
            "level": "hard",
        },
        {
            "question": "What is the main ingredient in the pasta sauce \"Cacio e Pepe\" from the country whose capital is Rome?",
            "answer": "Pecorino cheese",
            "type": "bridge",
            "level": "hard",
        },
        {
            "question": "What river flows through the capital of France?",
            "answer": "Seine",
            "type": "bridge",
            "level": "easy",
        },
        {
            "question": "Who was president of the United States when the first moon landing occurred?",
            "answer": "Richard Nixon",
            "type": "bridge",
            "level": "medium",
        },
        {
            "question": "What language is primarily spoken in the country where the Taj Mahal is located?",
            "answer": "Hindi",
            "type": "bridge",
            "level": "easy",
        },
        {
            "question": "Which artist painted the ceiling of the Sistine Chapel in Vatican City?",
            "answer": "Michelangelo",
            "type": "bridge",
            "level": "easy",
        },
        {
            "question": "What is the official language of the country that hosted the 2016 Summer Olympics?",
            "answer": "Portuguese",
            "type": "bridge",
            "level": "medium",
        },
        {
            "question": "What is the currency of the country where the Pyramids of Giza are located?",
            "answer": "Egyptian pound",
            "type": "bridge",
            "level": "easy",
        },
        {
            "question": "Which river is the longest in the continent where Mount Kilimanjaro is located?",
            "answer": "Nile",
            "type": "bridge",
            "level": "medium",
        },
        {
            "question": "What instrument did the composer of 'The Four Seasons' primarily play?",
            "answer": "Violin",
            "type": "bridge",
            "level": "medium",
        },
        {
            "question": "In which decade was the inventor of the World Wide Web born?",
            "answer": "1950s",
            "type": "bridge",
            "level": "hard",
        },
        {
            "question": "What type of government does the country that launched Sputnik have today?",
            "answer": "Federal republic",
            "type": "bridge",
            "level": "hard",
        },
    ]
    
    examples = {}
    for i, data in enumerate(mock_data[:num_examples]):
        q_id = f"hotpot_mock_{i}"
        examples[q_id] = {
            "question": data["question"],
            "answer": data["answer"],
            "type": data["type"],
            "level": data["level"],
            "supporting_facts": [],
            "context": [],
            "raw_item": None,
        }
    
    print(f"✓ Loaded {len(examples)} mock HotpotQA examples")
    return examples


def print_header(title: str, char: str = "="):
    """Print formatted header."""
    print(f"\n{char * 70}")
    print(f"{title:^70}")
    print(f"{char * 70}\n")


def collect_baseline_trajectories_hotpot(
    examples: Dict[str, Dict],
    num_agents: int = 1,
    max_steps: int = 4,
) -> Tuple[Dict[str, List[AgentTrajectory]], Dict[str, str]]:
    """
    Collect trajectories on HotpotQA examples.
    
    Returns:
        (Dict mapping q_id -> trajectories, Dict mapping q_id -> question)
    """
    print_header("PHASE 1: Collect Baseline Trajectories on HotpotQA")
    
    baseline = MultiAgentBaseline(
        num_agents=num_agents,
        model="qwen3:8b",
        tavily_api_key=os.getenv("TAVILY_API_KEY")
    )
    
    all_trajectories = {}
    question_texts = {}
    
    for i, (q_id, example) in enumerate(examples.items()):
        question = example["question"]
        question_texts[q_id] = question
        
        print(f"\n[{i+1}/{len(examples)}] {q_id}")
        print(f"  Q: {question[:70]}...")
        
        result = baseline.solve_task(
            q_id,
            question,
            max_steps_per_agent=max_steps
        )
        
        all_trajectories[q_id] = result.agent_trajectories
        
        # Show answers
        for agent_id, traj in enumerate(result.agent_trajectories):
            answer = traj.final_answer or "No answer"
            print(f"  Agent {agent_id}: {answer[:60]}...")
    
    return all_trajectories, question_texts


def analyze_trajectory_patterns(trajectories: Dict[str, List[AgentTrajectory]],
                                examples: Dict[str, Dict],
                                scorer: TaskScorer,
                                label: str = "") -> Dict:
    """
    Analyze multi-step trajectory patterns.
    
    For multi-hop QA, the key RL decisions are:
    - How many search steps before answering?
    - Does the policy learn to avoid useless tools (calculator, extract)?
    - Do shorter trajectories correlate with correct answers?
    """
    all_tools = ["no_tool", "web_search", "web_fetch", "extract", "calculator"]
    tool_counts = defaultdict(int)
    step_counts = []
    correct_steps = []
    wrong_steps = []
    ends_with_no_tool = 0
    total_trajs = 0
    step_distribution = defaultdict(int)  # step_count -> num trajectories
    tool_success = defaultdict(lambda: [0, 0])  # tool -> [success, total]

    for q_id, trajs in trajectories.items():
        for traj in trajs:
            total_trajs += 1
            n_steps = len(traj.steps)
            step_counts.append(n_steps)
            step_distribution[n_steps] += 1

            for step in traj.steps:
                tool_counts[step.tool_called] += 1
                tool_success[step.tool_called][1] += 1
                if step.tool_result and step.tool_result.ok:
                    tool_success[step.tool_called][0] += 1

            if traj.steps and traj.steps[-1].tool_called == "no_tool":
                ends_with_no_tool += 1

            if traj.final_answer:
                score = scorer.score_answer(q_id, traj.final_answer)
                if score > 0.8:
                    correct_steps.append(n_steps)
                else:
                    wrong_steps.append(n_steps)

    avg_steps = sum(step_counts) / max(1, len(step_counts))
    min_steps = min(step_counts) if step_counts else 0
    max_steps_val = max(step_counts) if step_counts else 0
    avg_correct = sum(correct_steps) / max(1, len(correct_steps)) if correct_steps else 0
    avg_wrong = sum(wrong_steps) / max(1, len(wrong_steps)) if wrong_steps else 0
    stop_rate = ends_with_no_tool / max(1, total_trajs)
    total_tool_calls = sum(tool_counts.values())

    return {
        "avg_steps": avg_steps,
        "min_steps": min_steps,
        "max_steps": max_steps_val,
        "avg_steps_correct": avg_correct,
        "avg_steps_wrong": avg_wrong,
        "stop_rate": stop_rate,
        "tool_counts": dict(tool_counts),
        "tool_success": {k: {"ok": v[0], "total": v[1]} for k, v in tool_success.items()},
        "step_distribution": dict(step_distribution),
        "total_trajectories": total_trajs,
        "total_tool_calls": total_tool_calls,
    }


def setup_scorer_hotpot(examples: Dict[str, Dict],
                       judge_model: str = "qwen3:8b") -> TaskScorer:
    """Setup scorer with HotpotQA ground truth answers."""
    scorer = TaskScorer(judge_model=judge_model)
    
    for q_id, example in examples.items():
        scorer.register_ground_truth(q_id, example["answer"])
    
    return scorer


def analyze_baseline_performance(all_trajectories: Dict[str, List[AgentTrajectory]],
                                examples: Dict[str, Dict],
                                scorer: TaskScorer,
                                label: str = "Baseline") -> Dict:
    """
    Analyze performance on HotpotQA.
    
    Returns:
        Performance metrics
    """
    print_header(f"Analyze {label} Performance")
    
    metrics = {
        "total_agents": 0,
        "correct": 0,
        "total": 0,
        "accuracy": 0.0,
        "tool_calls_avg": 0.0,
        "trajectories": {},
    }
    
    all_tool_calls = []
    
    print(f"{'Question':<30} {'Answer':<30} {'Score':<8} {'Tools':<8}")
    print("-" * 80)
    
    for q_id, trajectories in all_trajectories.items():
        example = examples[q_id]
        question = example["question"]
        
        for traj in trajectories:
            metrics["total_agents"] += 1
            
            if traj.final_answer:
                score = scorer.score_answer(q_id, traj.final_answer)
                metrics["total"] += 1
                
                if score > 0.8:
                    metrics["correct"] += 1
                
                q_short = question[:28] + "..." if len(question) > 28 else question
                a_short = traj.final_answer[:28] + "..." if len(traj.final_answer) > 28 else traj.final_answer
                
                print(f"{q_short:<30} {a_short:<30} {score:<8.2f} {traj.total_tool_calls:<8}")
                
                all_tool_calls.append(traj.total_tool_calls)
    
    metrics["accuracy"] = metrics["correct"] / max(1, metrics["total"])
    metrics["tool_calls_avg"] = sum(all_tool_calls) / len(all_tool_calls) if all_tool_calls else 0
    
    print(f"\n{'='*80}")
    print(f"{label} Performance:")
    print(f"  Accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
    print(f"  Avg tool calls per trajectory: {metrics['tool_calls_avg']:.1f}")
    
    return metrics


def on_policy_train_hotpot(train_examples: Dict[str, Dict],
                          scorer: TaskScorer,
                          num_iterations: int = 5,
                          num_agents: int = 1,
                          max_steps: int = 5,
                          output_dir: str = "results",
                          feature_mode: str = "structured",
                          model_type: str = "deep") -> Tuple[List[Dict], PPOFineTuner]:
    """
    On-policy PPO training on HotpotQA.
    
    True on-policy: at each iteration, agents collect fresh trajectories
    using the current policy, then PPO updates the policy.
    
    Args:
        train_examples: Training examples with ground truth
        scorer: Task scorer
        num_iterations: Number of on-policy iterations
        num_agents: Number of agents per question
        max_steps: Max tool calls per agent
        output_dir: Directory to save results
        feature_mode: "bow" or "structured"
        model_type: "simple" or "deep"
    
    Returns:
        (list of per-iteration metrics, fine_tuner object)
    """
    print_header("On-Policy PPO Training")
    
    fine_tuner = PPOFineTuner(scorer, device="cpu",
                              feature_mode=feature_mode, model_type=model_type)
    
    # On-policy training loop
    iter_metrics = fine_tuner.on_policy_train(
        train_examples,
        num_iterations=num_iterations,
        num_agents=num_agents,
        max_steps=max_steps,
    )
    
    # Save model and trajectories
    Path(output_dir).mkdir(exist_ok=True)
    fine_tuner.save_model(f"{output_dir}/hotpot_tool_selector.pt")
    fine_tuner.save_trajectories(f"{output_dir}/trajectories.json")
    
    return iter_metrics, fine_tuner


def collect_ppo_trajectories(examples: Dict[str, Dict],
                             fine_tuner: PPOFineTuner,
                             num_agents: int = 1,
                             max_steps: int = 4) -> Tuple[Dict[str, List[AgentTrajectory]], Dict[str, str]]:
    """
    Collect trajectories using the PPO-trained tool selection policy.
    
    Evaluates the trained policy by running agents that use the policy
    network to select tools, comparing against the baseline (LLM alone).
    
    Args:
        examples: Evaluation examples
        fine_tuner: Trained PPOFineTuner with policy
        num_agents: Number of agents per question
        max_steps: Max steps per agent
    
    Returns:
        (Dict mapping q_id -> trajectories, Dict mapping q_id -> question)
    """
    print_header("Collect PPO-Guided Trajectories")
    
    all_trajectories = {}
    question_texts = {}
    
    for i, (q_id, example) in enumerate(examples.items()):
        question = example["question"]
        question_texts[q_id] = question
        trajs = []
        
        print(f"\n[{i+1}/{len(examples)}] {q_id}")
        print(f"  Q: {question[:70]}...")
        
        for agent_id in range(num_agents):
            agent = LocalLLMAgent(
                agent_id=agent_id,
                model="qwen3:8b",
                tavily_api_key=os.getenv("TAVILY_API_KEY"),
            )
            traj = agent.solve_with_policy(
                q_id, question,
                tool_policy=fine_tuner,
                max_steps=max_steps,
            )
            trajs.append(traj)
            answer = traj.final_answer or "No answer"
            print(f"  Agent {agent_id}: {answer[:60]}...")
        
        all_trajectories[q_id] = trajs
    
    return all_trajectories, question_texts


def _print_box(title: str, width: int = 70):
    """Print a boxed title."""
    print(f"\n┌{'─' * (width - 2)}┐")
    print(f"│{title:^{width - 2}}│")
    print(f"└{'─' * (width - 2)}┘")


def _print_comparison_table(baseline_patterns: Dict, ppo_patterns: Dict):
    """Print a comprehensive side-by-side comparison table."""
    all_tools = ["no_tool", "web_search", "web_fetch", "extract", "calculator"]
    W = 72

    _print_box("SIDE-BY-SIDE COMPARISON", W)

    # Header
    print(f"\n  {'Metric':<36} {'Baseline':>14} {'PPO':>14}")
    print(f"  {'─' * 36} {'─' * 14} {'─' * 14}")

    # Step statistics
    print(f"  {'Avg Steps / Trajectory':<36} {baseline_patterns['avg_steps']:>14.1f} {ppo_patterns['avg_steps']:>14.1f}")
    print(f"  {'Min Steps':<36} {baseline_patterns['min_steps']:>14} {ppo_patterns['min_steps']:>14}")
    print(f"  {'Max Steps':<36} {baseline_patterns['max_steps']:>14} {ppo_patterns['max_steps']:>14}")
    print(f"  {'Avg Steps (Correct Answers)':<36} {baseline_patterns['avg_steps_correct']:>14.1f} {ppo_patterns['avg_steps_correct']:>14.1f}")
    print(f"  {'Avg Steps (Wrong Answers)':<36} {baseline_patterns['avg_steps_wrong']:>14.1f} {ppo_patterns['avg_steps_wrong']:>14.1f}")
    print(f"  {'Explicit Stop Rate (no_tool)':<36} {baseline_patterns['stop_rate']:>13.0%} {ppo_patterns['stop_rate']:>13.0%}")
    print(f"  {'Total Trajectories':<36} {baseline_patterns['total_trajectories']:>14} {ppo_patterns['total_trajectories']:>14}")
    print(f"  {'Total Tool Calls':<36} {baseline_patterns['total_tool_calls']:>14} {ppo_patterns['total_tool_calls']:>14}")

    # Tool usage breakdown
    print(f"\n  {'─' * 36} {'─' * 14} {'─' * 14}")
    print(f"  {'Tool Usage Counts':<36} {'Baseline':>14} {'PPO':>14}")
    print(f"  {'─' * 36} {'─' * 14} {'─' * 14}")

    b_total = max(1, baseline_patterns['total_tool_calls'])
    p_total = max(1, ppo_patterns['total_tool_calls'])

    for tool in all_tools:
        b_count = baseline_patterns['tool_counts'].get(tool, 0)
        p_count = ppo_patterns['tool_counts'].get(tool, 0)
        b_pct = b_count / b_total * 100
        p_pct = p_count / p_total * 100
        b_str = f"{b_count} ({b_pct:4.0f}%)"
        p_str = f"{p_count} ({p_pct:4.0f}%)"
        print(f"    {tool:<34} {b_str:>14} {p_str:>14}")

    # Step distribution
    all_step_keys = sorted(set(
        list(baseline_patterns['step_distribution'].keys()) +
        list(ppo_patterns['step_distribution'].keys())
    ))
    if all_step_keys:
        print(f"\n  {'─' * 36} {'─' * 14} {'─' * 14}")
        print(f"  {'Step Length Distribution':<36} {'Baseline':>14} {'PPO':>14}")
        print(f"  {'─' * 36} {'─' * 14} {'─' * 14}")
        for k in all_step_keys:
            b_n = baseline_patterns['step_distribution'].get(k, 0)
            p_n = ppo_patterns['step_distribution'].get(k, 0)
            b_bar = "█" * b_n
            p_bar = "█" * p_n
            print(f"    {k} step(s)  {b_bar:<10} {b_n:>4}    {p_bar:<10} {p_n:>4}")


def _print_per_question_table(baseline_trajs: Dict, ppo_trajs: Dict,
                              examples: Dict, scorer: TaskScorer):
    """Print per-question detail comparison."""
    _print_box("PER-QUESTION DETAIL", 72)

    print(f"\n  {'#':<3} {'Question':<32} {'GT Answer':<14} {'BL':>4} {'PPO':>4} {'BL Steps':>9} {'PPO Steps':>10}")
    print(f"  {'─' * 3} {'─' * 32} {'─' * 14} {'─' * 4} {'─' * 4} {'─' * 9} {'─' * 10}")

    q_ids = list(examples.keys())
    for idx, q_id in enumerate(q_ids):
        if q_id not in baseline_trajs or q_id not in ppo_trajs:
            continue
        ex = examples[q_id]
        q_short = ex["question"][:30] + ".." if len(ex["question"]) > 30 else ex["question"]
        gt_short = ex["answer"][:12] + ".." if len(ex["answer"]) > 12 else ex["answer"]

        # Best trajectory per question (take first agent)
        b_traj = baseline_trajs[q_id][0] if baseline_trajs[q_id] else None
        p_traj = ppo_trajs[q_id][0] if ppo_trajs[q_id] else None

        b_score = scorer.score_answer(q_id, b_traj.final_answer) if b_traj and b_traj.final_answer else 0
        p_score = scorer.score_answer(q_id, p_traj.final_answer) if p_traj and p_traj.final_answer else 0

        b_mark = "✓" if b_score > 0.8 else "✗"
        p_mark = "✓" if p_score > 0.8 else "✗"
        b_steps = len(b_traj.steps) if b_traj else 0
        p_steps = len(p_traj.steps) if p_traj else 0

        # Tool sequence
        b_tools = " → ".join(s.tool_called.replace("web_search", "srch").replace("no_tool", "stop")
                              .replace("web_fetch", "fetch").replace("extract", "extr").replace("calculator", "calc")
                              for s in (b_traj.steps if b_traj else []))
        p_tools = " → ".join(s.tool_called.replace("web_search", "srch").replace("no_tool", "stop")
                              .replace("web_fetch", "fetch").replace("extract", "extr").replace("calculator", "calc")
                              for s in (p_traj.steps if p_traj else []))

        print(f"  {idx+1:<3} {q_short:<32} {gt_short:<14} {b_mark:>4} {p_mark:>4} {b_steps:>9} {p_steps:>10}")
        print(f"  {'':3} BL: {b_tools}")
        print(f"  {'':3} PPO: {p_tools}")


def _print_training_curve(iter_metrics: List[Dict]):
    """Print ASCII training curve."""
    if not iter_metrics:
        return

    _print_box("PPO TRAINING CURVE", 72)

    max_acc = max(m["accuracy"] for m in iter_metrics)
    bar_width = 40

    print(f"\n  {'Iter':>4}  {'Accuracy':>8}  {'Decisions':>9}  {'':40}")
    print(f"  {'─' * 4}  {'─' * 8}  {'─' * 9}  {'─' * 40}")

    for m in iter_metrics:
        acc = m["accuracy"]
        n_dec = m["num_decisions"]
        bar_len = int(acc / max(0.01, max_acc) * bar_width) if max_acc > 0 else 0
        bar = "█" * bar_len + "░" * (bar_width - bar_len)
        print(f"  {m['iteration']:>4}  {acc:>7.1%}  {n_dec:>9}  {bar} {acc:.1%}")

    # Training loss if available
    has_loss = any("training" in m and m["training"] for m in iter_metrics)
    if has_loss:
        print(f"\n  {'Iter':>4}  {'Policy Loss':>11}  {'Value Loss':>10}  {'Entropy':>8}")
        print(f"  {'─' * 4}  {'─' * 11}  {'─' * 10}  {'─' * 8}")
        for m in iter_metrics:
            t = m.get("training", {})
            if t:
                history = t.get("history", [])
                if history:
                    last = history[-1]
                    pl = last.get("policy_loss", 0)
                    vl = last.get("value_loss", 0)
                    ent = last.get("entropy", 0)
                else:
                    pl = t.get("policy_loss", 0)
                    vl = t.get("value_loss", 0)
                    ent = t.get("entropy", 0)
                print(f"  {m['iteration']:>4}  {pl:>11.4f}  {vl:>10.4f}  {ent:>8.4f}")


def _generate_detailed_report(baseline_perf, ppo_perf, improvement, tools_diff,
                              baseline_patterns, ppo_patterns,
                              baseline_trajs, ppo_trajs, eval_examples, scorer,
                              iter_metrics):
    """Generate detailed text report that mirrors the terminal output."""
    all_tools = ["no_tool", "web_search", "web_fetch", "extract", "calculator"]
    L = []  # report lines

    L.append("HOTPOT QA PIPELINE: Baseline (LLM) vs LLM + PPO Tool Selector")
    L.append("=" * 65)
    L.append("")

    # --- Final report table ---
    L.append("┌──────────────────────────────────────────────────────────────────────┐")
    L.append("│                             FINAL REPORT                             │")
    L.append("└──────────────────────────────────────────────────────────────────────┘")
    L.append("")
    L.append(f"  Metric                             Baseline (LLM)      LLM + PPO")
    L.append(f"  ─────────────────────────────────── ──────────────── ──────────────")
    L.append(f"  Accuracy                           {baseline_perf['accuracy']:>13.1%}  {ppo_perf['accuracy']:>13.1%}")
    L.append(f"  Correct / Total                    {baseline_perf['correct']:>5}/{baseline_perf['total']:<8} {ppo_perf['correct']:>5}/{ppo_perf['total']:<8}")
    L.append(f"  Avg Tool Calls / Trajectory        {baseline_perf['tool_calls_avg']:>14.1f}  {ppo_perf['tool_calls_avg']:>13.1f}")
    L.append(f"  Accuracy Improvement                                      {improvement:>+.1%}")
    L.append(f"  Tool Calls Δ                                              {tools_diff:>+.1f}")
    L.append("")

    # --- Side-by-side comparison ---
    L.append("┌──────────────────────────────────────────────────────────────────────┐")
    L.append("│                       SIDE-BY-SIDE COMPARISON                        │")
    L.append("└──────────────────────────────────────────────────────────────────────┘")
    L.append("")
    L.append(f"  {'Metric':<36} {'Baseline':>14} {'PPO':>14}")
    L.append(f"  {'─' * 36} {'─' * 14} {'─' * 14}")
    L.append(f"  {'Avg Steps / Trajectory':<36} {baseline_patterns['avg_steps']:>14.1f} {ppo_patterns['avg_steps']:>14.1f}")
    L.append(f"  {'Min Steps':<36} {baseline_patterns['min_steps']:>14} {ppo_patterns['min_steps']:>14}")
    L.append(f"  {'Max Steps':<36} {baseline_patterns['max_steps']:>14} {ppo_patterns['max_steps']:>14}")
    L.append(f"  {'Avg Steps (Correct Answers)':<36} {baseline_patterns['avg_steps_correct']:>14.1f} {ppo_patterns['avg_steps_correct']:>14.1f}")
    L.append(f"  {'Avg Steps (Wrong Answers)':<36} {baseline_patterns['avg_steps_wrong']:>14.1f} {ppo_patterns['avg_steps_wrong']:>14.1f}")
    L.append(f"  {'Explicit Stop Rate (no_tool)':<36} {baseline_patterns['stop_rate']:>13.0%} {ppo_patterns['stop_rate']:>13.0%}")
    L.append(f"  {'Total Trajectories':<36} {baseline_patterns['total_trajectories']:>14} {ppo_patterns['total_trajectories']:>14}")
    L.append(f"  {'Total Tool Calls':<36} {baseline_patterns['total_tool_calls']:>14} {ppo_patterns['total_tool_calls']:>14}")
    L.append("")

    # Tool usage
    b_total = max(1, baseline_patterns['total_tool_calls'])
    p_total = max(1, ppo_patterns['total_tool_calls'])
    L.append(f"  {'Tool Usage Counts':<36} {'Baseline':>14} {'PPO':>14}")
    L.append(f"  {'─' * 36} {'─' * 14} {'─' * 14}")
    for tool in all_tools:
        b_count = baseline_patterns['tool_counts'].get(tool, 0)
        p_count = ppo_patterns['tool_counts'].get(tool, 0)
        b_pct = b_count / b_total * 100
        p_pct = p_count / p_total * 100
        L.append(f"    {tool:<34} {b_count} ({b_pct:4.0f}%)        {p_count} ({p_pct:4.0f}%)")
    L.append("")

    # Step distribution
    all_step_keys = sorted(set(
        list(baseline_patterns['step_distribution'].keys()) +
        list(ppo_patterns['step_distribution'].keys())
    ))
    if all_step_keys:
        L.append(f"  {'Step Length Distribution':<36} {'Baseline':>14} {'PPO':>14}")
        L.append(f"  {'─' * 36} {'─' * 14} {'─' * 14}")
        for k in all_step_keys:
            b_n = baseline_patterns['step_distribution'].get(k, 0)
            p_n = ppo_patterns['step_distribution'].get(k, 0)
            b_bar = "█" * b_n
            p_bar = "█" * p_n
            L.append(f"    {k} step(s)  {b_bar:<10} {b_n:>4}    {p_bar:<10} {p_n:>4}")
    L.append("")

    # --- Per-question detail ---
    L.append("┌──────────────────────────────────────────────────────────────────────┐")
    L.append("│                         PER-QUESTION DETAIL                          │")
    L.append("└──────────────────────────────────────────────────────────────────────┘")
    L.append("")
    L.append(f"  {'#':<3} {'Question':<32} {'GT Answer':<14} {'BL':>4} {'PPO':>4} {'BL Steps':>9} {'PPO Steps':>10}")
    L.append(f"  {'─' * 3} {'─' * 32} {'─' * 14} {'─' * 4} {'─' * 4} {'─' * 9} {'─' * 10}")

    q_ids = list(eval_examples.keys())
    for idx, q_id in enumerate(q_ids):
        if q_id not in baseline_trajs or q_id not in ppo_trajs:
            continue
        ex = eval_examples[q_id]
        q_short = ex["question"][:30] + ".." if len(ex["question"]) > 30 else ex["question"]
        gt_short = ex["answer"][:12] + ".." if len(ex["answer"]) > 12 else ex["answer"]

        b_traj = baseline_trajs[q_id][0] if baseline_trajs[q_id] else None
        p_traj = ppo_trajs[q_id][0] if ppo_trajs[q_id] else None

        b_score = scorer.score_answer(q_id, b_traj.final_answer) if b_traj and b_traj.final_answer else 0
        p_score = scorer.score_answer(q_id, p_traj.final_answer) if p_traj and p_traj.final_answer else 0

        b_mark = "✓" if b_score > 0.8 else "✗"
        p_mark = "✓" if p_score > 0.8 else "✗"
        b_steps = len(b_traj.steps) if b_traj else 0
        p_steps = len(p_traj.steps) if p_traj else 0

        def _abbrev(name):
            return name.replace("web_search", "srch").replace("no_tool", "stop") \
                       .replace("web_fetch", "fetch").replace("extract", "extr") \
                       .replace("calculator", "calc")
        b_tools = " → ".join(_abbrev(s.tool_called) for s in (b_traj.steps if b_traj else []))
        p_tools = " → ".join(_abbrev(s.tool_called) for s in (p_traj.steps if p_traj else []))

        b_answer = (b_traj.final_answer or "N/A")[:30]
        p_answer = (p_traj.final_answer or "N/A")[:30]

        L.append(f"  {idx+1:<3} {q_short:<32} {gt_short:<14} {b_mark:>4} {p_mark:>4} {b_steps:>9} {p_steps:>10}")
        L.append(f"      BL answer: {b_answer}")
        L.append(f"      PPO answer: {p_answer}")
        L.append(f"      BL tools: {b_tools}")
        L.append(f"      PPO tools: {p_tools}")
    L.append("")

    # --- Training curve ---
    L.append("┌──────────────────────────────────────────────────────────────────────┐")
    L.append("│                          PPO TRAINING CURVE                          │")
    L.append("└──────────────────────────────────────────────────────────────────────┘")
    L.append("")

    if iter_metrics:
        max_acc = max(m["accuracy"] for m in iter_metrics)
        bar_width = 40
        L.append(f"  {'Iter':>4}  {'Accuracy':>8}  {'Decisions':>9}")
        L.append(f"  {'─' * 4}  {'─' * 8}  {'─' * 9}  {'─' * 40}")
        for m in iter_metrics:
            acc = m["accuracy"]
            n_dec = m["num_decisions"]
            bar_len = int(acc / max(0.01, max_acc) * bar_width) if max_acc > 0 else 0
            bar = "█" * bar_len + "░" * (bar_width - bar_len)
            L.append(f"  {m['iteration']:>4}  {acc:>7.1%}  {n_dec:>9}  {bar} {acc:.1%}")
        L.append("")

        has_loss = any("training" in m and m["training"] for m in iter_metrics)
        if has_loss:
            L.append(f"  {'Iter':>4}  {'Policy Loss':>11}  {'Value Loss':>10}  {'Entropy':>8}")
            L.append(f"  {'─' * 4}  {'─' * 11}  {'─' * 10}  {'─' * 8}")
            for m in iter_metrics:
                t = m.get("training", {})
                if t:
                    history = t.get("history", [])
                    if history:
                        last = history[-1]
                        pl = last.get("policy_loss", 0)
                        vl = last.get("value_loss", 0)
                        ent = last.get("entropy", 0)
                    else:
                        pl = vl = ent = 0
                    L.append(f"  {m['iteration']:>4}  {pl:>11.4f}  {vl:>10.4f}  {ent:>8.4f}")
    L.append("")

    return L


def main():
    """Run complete HotpotQA pipeline: LLM (baseline) vs LLM + PPO tool selector."""
    
    _print_box("HOTPOT QA PIPELINE: LLM vs LLM + PPO Tool Selector")
    
    # Setup results directory
    output_dir = "results"
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load data
    print_header("Loading Data")
    examples = load_hotpot_data(split="train", max_examples=25)
    
    # Split into train and eval sets
    example_ids = list(examples.keys())
    split_idx = max(1, len(example_ids) - 8)
    train_ids = example_ids[:split_idx]
    eval_ids = example_ids[split_idx:]
    
    train_examples = {k: examples[k] for k in train_ids}
    eval_examples = {k: examples[k] for k in eval_ids}
    print(f"  Train: {len(train_examples)} examples")
    print(f"  Eval:  {len(eval_examples)} examples")
    print(f"  Total: {len(examples)} examples")
    
    try:
        # Phase 1: Baseline evaluation (LLM alone, no policy guidance)
        baseline_trajs, baseline_questions = collect_baseline_trajectories_hotpot(
            eval_examples, num_agents=1, max_steps=4
        )
        scorer = setup_scorer_hotpot(examples)
        baseline_perf = analyze_baseline_performance(
            baseline_trajs, eval_examples, scorer, label="Baseline (LLM Alone)"
        )
        
        # Phase 2: On-policy PPO training on train set
        iter_metrics, fine_tuner = on_policy_train_hotpot(
            train_examples, scorer,
            num_iterations=5,
            num_agents=1,
            max_steps=4,
            output_dir=output_dir,
            feature_mode="structured",
            model_type="deep",
        )
        
        # Phase 3: Evaluate PPO-guided agent on eval set
        ppo_trajs, ppo_questions = collect_ppo_trajectories(
            eval_examples, fine_tuner, num_agents=1, max_steps=4
        )
        ppo_perf = analyze_baseline_performance(
            ppo_trajs, eval_examples, scorer, label="LLM + PPO Tool Selector"
        )
        
        # Save results
        fine_tuner.save_training_results(
            f"{output_dir}/training_results.json",
            baseline_metrics=baseline_perf
        )
        
        # ═══════════════════════════════════════════════════════════
        #  FINAL REPORT
        # ═══════════════════════════════════════════════════════════

        _print_box("FINAL REPORT", 72)

        # --- Accuracy & Efficiency ---
        improvement = ppo_perf['accuracy'] - baseline_perf['accuracy']
        tools_diff = ppo_perf['tool_calls_avg'] - baseline_perf['tool_calls_avg']

        print(f"\n  ┌──────────────────────────────────────────────────────────────────┐")
        print(f"  │  {'Metric':<34} {'Baseline (LLM)':>14} {'LLM + PPO':>14}  │")
        print(f"  ├──────────────────────────────────────────────────────────────────┤")
        print(f"  │  {'Accuracy':<34} {baseline_perf['accuracy']:>13.1%} {ppo_perf['accuracy']:>14.1%}  │")
        print(f"  │  {'Correct / Total':<34} {baseline_perf['correct']:>5}/{baseline_perf['total']:<8} {ppo_perf['correct']:>5}/{ppo_perf['total']:<8}  │")
        print(f"  │  {'Avg Tool Calls / Trajectory':<34} {baseline_perf['tool_calls_avg']:>14.1f} {ppo_perf['tool_calls_avg']:>14.1f}  │")
        print(f"  ├──────────────────────────────────────────────────────────────────┤")
        print(f"  │  {'Accuracy Improvement':<34} {improvement:>+30.1%}  │")
        print(f"  │  {'Tool Calls Δ':<34} {tools_diff:>+30.1f}  │")
        print(f"  └──────────────────────────────────────────────────────────────────┘")

        # --- Trajectory pattern analysis ---
        baseline_patterns = analyze_trajectory_patterns(
            baseline_trajs, eval_examples, scorer, label="Baseline"
        )
        ppo_patterns = analyze_trajectory_patterns(
            ppo_trajs, eval_examples, scorer, label="PPO"
        )

        _print_comparison_table(baseline_patterns, ppo_patterns)

        # --- Per-question detail ---
        _print_per_question_table(baseline_trajs, ppo_trajs, eval_examples, scorer)

        # --- Training curve ---
        _print_training_curve(iter_metrics)

        # --- Save all results ---
        comparison_data = {
            "baseline": baseline_perf,
            "ppo": ppo_perf,
            "improvement": improvement,
            "training_curve": iter_metrics,
            "baseline_trajectory_patterns": baseline_patterns,
            "ppo_trajectory_patterns": ppo_patterns,
        }
        with open(f"{output_dir}/comparison.json", 'w') as f:
            json.dump(comparison_data, f, indent=2, default=str)

        # Save detailed text report (mirrors terminal output)
        report_lines = _generate_detailed_report(
            baseline_perf, ppo_perf, improvement, tools_diff,
            baseline_patterns, ppo_patterns,
            baseline_trajs, ppo_trajs, eval_examples, scorer,
            iter_metrics,
        )
        with open(f"{output_dir}/report.txt", 'w') as f:
            f.write("\n".join(report_lines))
        print(f"  Report saved to {output_dir}/report.txt")

        # Final summary
        print(f"\n  Results saved to {output_dir}/:")
        print(f"    hotpot_tool_selector.pt    PPO-trained policy model")
        print(f"    trajectories.json          On-policy trajectories")
        print(f"    training_results.json      Training metrics")
        print(f"    comparison.json            Full comparison data")
        print()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
