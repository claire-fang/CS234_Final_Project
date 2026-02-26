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

from multi_agent_baseline import MultiAgentBaseline, AgentTrajectory
from ppo_finetuner import TaskScorer, PPOFineTuner
from tool_verifier import SimpleToolVerifier


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
    
    Returns realistic multi-hop reasoning questions.
    """
    print(f"Loading mock HotpotQA data ({num_examples} examples)...")
    
    # Real-style HotpotQA questions
    mock_data = [
        {
            "question": "Which country is the birthplace of the author of \"One Hundred Years of Solitude\"?",
            "answer": "Colombia",
            "type": "bridge",
            "level": "medium"
        },
        {
            "question": "What is the birth date of the lead singer of the band that performed \"Paranoid Android\"?",
            "answer": "October 7, 1997",  # Thom Yorke, Radiohead
            "type": "bridge",
            "level": "hard"
        },
        {
            "question": "In what year was the discoverer of Pluto born?",
            "answer": "1906",  # Clyde Tombaugh
            "type": "bridge",
            "level": "medium"
        },
        {
            "question": "What is the capital of the country where the Battle of Hastings took place?",
            "answer": "London",
            "type": "bridge",
            "level": "easy"
        },
        {
            "question": "Who is the author of the book that inspired the movie \"The Shawshank Redemption\"?",
            "answer": "Stephen King",
            "type": "bridge",
            "level": "easy"
        },
        {
            "question": "What is the height of the mountain where Mount Everest is located?",
            "answer": "8848 meters",
            "type": "bridge",
            "level": "medium"
        },
        {
            "question": "In which city is the university where Albert Einstein taught located?",
            "answer": "Berlin",
            "type": "bridge",
            "level": "hard"
        },
        {
            "question": "What is the population of the country where the largest library in the world is located?",
            "answer": "USA",
            "type": "bridge",
            "level": "hard"
        },
        {
            "question": "Which Greek philosopher wrote about the ideal state in \"The Republic\"?",
            "answer": "Plato",
            "type": "bridge",
            "level": "easy"
        },
        {
            "question": "What year did the war that inspired George Orwell's \"1984\" end?",
            "answer": "1945",
            "type": "bridge",
            "level": "medium"
        },
        {
            "question": "Who is the playwright of the work that is considered the greatest in English literature?",
            "answer": "William Shakespeare",
            "type": "bridge",
            "level": "easy"
        },
        {
            "question": "In what country is the company that produces the iPhone located?",
            "answer": "United States",
            "type": "bridge",
            "level": "easy"
        },
        {
            "question": "What is the scientific name of the animal that is the symbol of World Wildlife Fund?",
            "answer": "Ailuropoda melanoleuca",
            "type": "bridge",
            "level": "hard"
        },
        {
            "question": "Which composer wrote the \"Symphony No. 9\" that features \"Ode to Joy\"?",
            "answer": "Ludwig van Beethoven",
            "type": "bridge",
            "level": "easy"
        },
        {
            "question": "What is the capital of the country where the Great Wall is located?",
            "answer": "Beijing",
            "type": "bridge",
            "level": "easy"
        },
        {
            "question": "Which novel by Jane Austen features the character Elizabeth Bennet?",
            "answer": "Pride and Prejudice",
            "type": "bridge",
            "level": "easy"
        },
        {
            "question": "In what year did the country where the Colosseum is located become a republic?",
            "answer": "1946",
            "type": "bridge",
            "level": "hard"
        },
        {
            "question": "What is the name of the currency used in the country where sushi originated?",
            "answer": "Yen",
            "type": "bridge",
            "level": "easy"
        },
        {
            "question": "Who directed the film that won the Academy Award for Best Picture in 2020?",
            "answer": "Bong Joon-ho",  # Parasite
            "type": "bridge",
            "level": "hard"
        },
        {
            "question": "What is the main ingredient in the pasta sauce a la \"Cacio e Pepe\" from the country that has Rome as capital?",
            "answer": "Pecorino cheese",
            "type": "bridge",
            "level": "hard"
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
    num_agents: int = 2,
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


def verify_tool_usage(all_trajectories: Dict[str, List[AgentTrajectory]],
                     questions: Dict[str, str]) -> Dict[str, Dict]:
    """
    Verify tool usage across trajectories.
    
    Returns:
        Dict with verification statistics
    """
    print_header("PHASE 2: Verify Tool Usage")
    
    verifier = SimpleToolVerifier()
    
    stats = {
        "total_tool_calls": 0,
        "no_tool_calls": 0,
        "calculator_calls": 0,
        "search_calls": 0,
        "fetch_calls": 0,
        "extract_calls": 0,
        "verified_correct": 0,
        "verified_incorrect": 0,
        "tool_breakdown": defaultdict(int),
        "verification_breakdown": defaultdict(int),
    }
    
    for q_id, trajectories in all_trajectories.items():
        question = questions.get(q_id, "")
        
        for traj in trajectories:
            for step in traj.steps:
                tool = step.tool_called
                stats["total_tool_calls"] += 1
                stats["tool_breakdown"][tool] += 1
                
                # Verify
                result = verifier.verify(question, tool)
                
                if result.is_valid:
                    stats["verified_correct"] += 1
                    stats["verification_breakdown"][f"{tool}_valid"] += 1
                else:
                    stats["verified_incorrect"] += 1
                    stats["verification_breakdown"][f"{tool}_invalid"] += 1
    
    print(f"Total tool calls: {stats['total_tool_calls']}")
    print(f"Verified correct: {stats['verified_correct']} ({100*stats['verified_correct']/max(1, stats['total_tool_calls']):.1f}%)")
    print(f"Verified incorrect: {stats['verified_incorrect']} ({100*stats['verified_incorrect']/max(1, stats['total_tool_calls']):.1f}%)")
    
    print(f"\nTool usage breakdown:")
    for tool, count in sorted(stats["tool_breakdown"].items(), key=lambda x: -x[1]):
        print(f"  {tool}: {count}")
    
    return stats


def setup_scorer_hotpot(examples: Dict[str, Dict]) -> TaskScorer:
    """Setup scorer with HotpotQA ground truth answers."""
    scorer = TaskScorer()
    
    for q_id, example in examples.items():
        scorer.register_ground_truth(q_id, example["answer"])
    
    return scorer


def analyze_baseline_performance(all_trajectories: Dict[str, List[AgentTrajectory]],
                                examples: Dict[str, Dict],
                                scorer: TaskScorer) -> Dict:
    """
    Analyze baseline performance on HotpotQA.
    
    Returns:
        Performance metrics
    """
    print_header("PHASE 3: Analyze Baseline Performance")
    
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
    print(f"Baseline Performance:")
    print(f"  Accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
    print(f"  Avg tool calls per trajectory: {metrics['tool_calls_avg']:.1f}")
    
    return metrics


def fine_tune_on_hotpot(all_trajectories: Dict[str, List[AgentTrajectory]],
                        examples: Dict[str, Dict],
                        scorer: TaskScorer,
                        num_epochs: int = 5,
                        output_dir: str = "results") -> Tuple[Dict, PPOFineTuner]:
    """
    Fine-tune PPO on HotpotQA trajectories.
    
    Args:
        all_trajectories: Dict of question_id -> list of trajectories
        examples: HotpotQA examples with ground truth
        scorer: Task scorer
        num_epochs: Number of PPO training epochs
        output_dir: Directory to save results
    
    Returns:
        (training_metrics, fine_tuner_object)
    """
    print_header("PHASE 4: Fine-tune with PPO on HotpotQA")
    
    # Flatten trajectories
    all_trajs = []
    for trajs in all_trajectories.values():
        all_trajs.extend(trajs)
    
    question_texts = {q_id: ex["question"] for q_id, ex in examples.items()}
    
    fine_tuner = PPOFineTuner(scorer, device="cpu")
    fine_tuner.collect_trajectories(all_trajs, question_texts)
    
    print(f"Collected {len(all_trajs)} trajectories")
    
    decisions = fine_tuner.collector.get_all_decisions()
    print(f"Total decisions: {len(decisions)}")
    
    # Show reward distribution
    high_reward = sum(1 for d in decisions if d.reward > 0.8)
    low_reward = sum(1 for d in decisions if d.reward < -0.1)
    print(f"  High reward (success): {high_reward}/{len(decisions)}")
    print(f"  Low reward (failure penalty): {low_reward}/{len(decisions)}")
    
    # Train
    metrics = fine_tuner.fine_tune(num_epochs=num_epochs, batch_size=4)
    
    # Save model and trajectories
    Path(output_dir).mkdir(exist_ok=True)
    fine_tuner.save_model(f"{output_dir}/hotpot_tool_selector.pt")
    fine_tuner.save_trajectories(f"{output_dir}/trajectories.json")
    
    return metrics, fine_tuner


def evaluate_fine_tuned_model(fine_tuner, all_trajectories: Dict[str, List[AgentTrajectory]],
                              examples: Dict[str, Dict],
                              scorer: TaskScorer) -> Dict:
    """
    Evaluate the fine-tuned model.
    
    (In practice, would run agents with fine-tuned model on held-out test set)
    """
    print_header("PHASE 5: Evaluation Results")
    
    # For now, analyze training data performance as proxy
    decisions = fine_tuner.collector.get_all_decisions()
    
    print(f"Training decisions statistics:")
    print(f"  Total: {len(decisions)}")
    print(f"  High reward (>0.8): {sum(1 for d in decisions if d.reward > 0.8)}")
    print(f"  Low reward (<0.5): {sum(1 for d in decisions if d.reward < 0.5)}")
    
    # Tool usage in high-reward trajectories
    high_reward_tools = {}
    for d in decisions:
        if d.reward > 0.8:
            tool = d.chosen_tool
            high_reward_tools[tool] = high_reward_tools.get(tool, 0) + 1
    
    print(f"\nTools in successful trajectories:")
    for tool, count in sorted(high_reward_tools.items(), key=lambda x: -x[1]):
        print(f"  {tool}: {count}")
    
    return {
        "decisions": len(decisions),
        "high_reward": sum(1 for d in decisions if d.reward > 0.8),
        "high_reward_tools": high_reward_tools,
    }


def main():
    """Run complete HotpotQA pipeline."""
    
    print("\n" + "=" * 70)
    print("HOTPOT QA PIPELINE: Train & Evaluate PPO Fine-tuning".center(70))
    print("=" * 70)
    
    # Setup results directory
    output_dir = "results"
    Path(output_dir).mkdir(exist_ok=True)
    
    # Phase 1: Load data
    print_header("Loading Data")
    examples = load_hotpot_data(split="train", max_examples=3)
    
    try:
        # Phase 2: Collect trajectories
        all_trajectories, question_texts = collect_baseline_trajectories_hotpot(
            examples,
            num_agents=2,
            max_steps=4
        )
        
        # Phase 3: Verify tool usage
        # verify_stats = verify_tool_usage(all_trajectories, question_texts)
        
        # Phase 4: Analyze baseline
        scorer = setup_scorer_hotpot(examples)
        baseline_perf = analyze_baseline_performance(all_trajectories, examples, scorer)
        
        # Phase 5: Fine-tune (now returns both metrics and fine_tuner)
        ft_metrics, fine_tuner = fine_tune_on_hotpot(
            all_trajectories,
            examples,
            scorer,
            num_epochs=5,
            output_dir=output_dir
        )
        
        # Phase 6: Evaluate
        eval_results = evaluate_fine_tuned_model(fine_tuner, all_trajectories, examples, scorer)
        
        # Save training results and summary
        fine_tuner.save_training_results(
            f"{output_dir}/training_results.json",
            baseline_metrics=baseline_perf
        )
        
        # Summary
        print_header("SUMMARY", char="*")
        print(f"\nBaseline Accuracy: {baseline_perf['accuracy']:.2%}")
        print(f"Avg Tool Calls: {baseline_perf['tool_calls_avg']:.1f}")
        print(f"\nFine-tuning Completed:")
        print(f"  Epochs: {ft_metrics.get('epochs', 'N/A')}")
        print(f"  Final policy loss: {ft_metrics['history'][-1].get('policy_loss', 'N/A'):.4f}")
        
        print(f"\nResults saved to: {output_dir}/")
        print(f"  - hotpot_tool_selector.pt (fine-tuned model)")
        print(f"  - trajectories.json (all collected trajectories)")
        print(f"  - training_results.json (metrics and analysis)")
        print(f"\nNext steps:")
        print(f"  1. Evaluate on held-out test set")
        print(f"  2. Train with more examples (100+)")
        print(f"  3. Compare tool selection before/after fine-tuning")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
