"""
Integration script: Multi-Agent Baseline + PPO Fine-tuning.

Workflow:
1. Run baseline to collect trajectories
2. Score tasks and annotate with rewards
3. Fine-tune model on tool selection with PPO
4. Evaluate improvements
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Import from baseline and fine-tuner
from multi_agent_baseline import MultiAgentBaseline, AgentTrajectory
from ppo_finetuner import TaskScorer, PPOFineTuner, DecisionCollector


def collect_baseline_trajectories(
    num_agents: int = 2,
    model: str = "qwen3:8b",
    tavily_api_key: str = None,
) -> Tuple[List[AgentTrajectory], Dict[str, str]]:
    """
    Collect trajectories from baseline agents without fine-tuning.
    
    Returns:
        (all_trajectories, task_descriptions)
    """
    print("="*70)
    print("PHASE 1: Collecting baseline trajectories")
    print("="*70)
    
    baseline = MultiAgentBaseline(
        num_agents=num_agents,
        model=model,
        tavily_api_key=tavily_api_key
    )
    
    # Define tasks with ground truth answers
    tasks = [
        ("q1", "What is the capital of France?", "Paris"),
        ("q2", "What is 15 * 7?", "105"),
        ("q3", "What is 200 / 8?", "25"),
        ("q4", "What is 50 + 30 - 10?", "70"),
        ("q5", "What is 2 ** 10?", "1024"),
    ]
    
    task_descriptions = {task_id: task for task_id, task, _ in tasks}
    task_answers = {task_id: answer for task_id, _, answer in tasks}
    
    all_trajectories = []
    
    for task_id, task, expected_answer in tasks:
        result = baseline.solve_task(task_id, task, max_steps_per_agent=3)
        for traj in result.agent_trajectories:
            all_trajectories.append(traj)
    
    return all_trajectories, task_descriptions, task_answers


def setup_scorer(task_answers: Dict[str, str]) -> TaskScorer:
    """Setup task scorer with ground truth answers."""
    scorer = TaskScorer()
    for task_id, answer in task_answers.items():
        scorer.register_ground_truth(task_id, answer)
    return scorer


def analyze_baseline_performance(trajectories: List[AgentTrajectory], 
                                 scorer: TaskScorer) -> Dict:
    """Analyze baseline performance before fine-tuning."""
    print("\n" + "="*70)
    print("PHASE 2: Analyzing baseline performance")
    print("="*70)
    
    correct = 0
    total = 0
    tool_usage = {}
    
    for traj in trajectories:
        if traj.final_answer:
            score = scorer.score_answer(traj.task_id, traj.final_answer)
            if score > 0.8:
                correct += 1
            total += 1
            
            # Track tool usage
            for step in traj.steps:
                tool = step.tool_called
                tool_usage[tool] = tool_usage.get(tool, 0) + 1
    
    accuracy = correct / max(1, total)
    
    print(f"\nBaseline Performance:")
    print(f"  Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"  Total trajectories: {len(trajectories)}")
    print(f"  Total tool calls: {sum(tool_usage.values())}")
    print(f"  Tool usage breakdown:")
    for tool, count in sorted(tool_usage.items(), key=lambda x: -x[1]):
        print(f"    {tool}: {count}")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "tool_usage": tool_usage,
    }


def fine_tune_tool_selection(trajectories: List[AgentTrajectory],
                            task_descriptions: Dict[str, str],
                            scorer: TaskScorer,
                            output_dir: str = "checkpoints") -> Dict:
    """
    Fine-tune model on tool selection using PPO.
    
    Args:
        trajectories: Collected baseline trajectories
        task_descriptions: Mapping of task_id -> task_description
        scorer: Task scorer with ground truth
        output_dir: Directory to save fine-tuned model
    
    Returns:
        Fine-tuning metrics
    """
    print("\n" + "="*70)
    print("PHASE 3: Fine-tuning with PPO")
    print("="*70)
    
    fine_tuner = PPOFineTuner(scorer)
    
    # Collect decisions from trajectories
    fine_tuner.collect_trajectories(trajectories, task_descriptions)
    
    # Fine-tune
    ft_metrics = fine_tuner.fine_tune(num_epochs=3, batch_size=8, learning_rate=1e-4)
    
    # Save model
    Path(output_dir).mkdir(exist_ok=True)
    model_path = f"{output_dir}/tool_selector_ppo.pt"
    fine_tuner.save_model(model_path)
    
    return ft_metrics


def create_demo_script():
    """Create a minimal demo script for local testing."""
    demo_code = '''"""
Minimal demo: Run baseline + PPO fine-tuning on CPU.

Prerequisites:
  pip install torch requests

Run with:
  python demo_ppo.py
"""

import os
os.environ["TAVILY_API_KEY"] = "your-api-key-here"  # Optional

from integration import (
    collect_baseline_trajectories,
    setup_scorer,
    analyze_baseline_performance,
    fine_tune_tool_selection,
)

def main():
    print("\\n" + "="*70)
    print("Multi-Agent Baseline + PPO Fine-tuning Demo")
    print("="*70)
    
    # Phase 1: Collect trajectories
    trajectories, task_desc, task_answers = collect_baseline_trajectories(
        num_agents=2,
        model="qwen3:8b",  # Make sure Ollama is running
    )
    
    # Phase 2: Setup scorer and analyze
    scorer = setup_scorer(task_answers)
    baseline_perf = analyze_baseline_performance(trajectories, scorer)
    
    # Phase 3: Fine-tune
    ft_metrics = fine_tune_tool_selection(
        trajectories, 
        task_desc, 
        scorer,
        output_dir="checkpoints"
    )
    
    print("\\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"Baseline accuracy: {baseline_perf['accuracy']:.2%}")
    print(f"Fine-tuning completed: {ft_metrics}")
    print("\\nModel saved to: checkpoints/tool_selector_ppo.pt")

if __name__ == "__main__":
    main()
'''
    
    with open("demo_ppo.py", "w") as f:
        f.write(demo_code)
    
    print("\nCreated demo_ppo.py")


def main():
    """Full pipeline: baseline → analysis → fine-tuning."""
    
    # Phase 1: Collect trajectories
    trajectories, task_desc, task_answers = collect_baseline_trajectories(
        num_agents=2,
        model="qwen3:8b",
    )
    
    # Phase 2: Setup scorer and analyze baseline
    scorer = setup_scorer(task_answers)
    baseline_perf = analyze_baseline_performance(trajectories, scorer)
    
    # Phase 3: Fine-tune with PPO
    ft_metrics = fine_tune_tool_selection(
        trajectories,
        task_desc,
        scorer
    )
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nBaseline Performance:")
    print(f"  Accuracy: {baseline_perf['accuracy']:.2%}")
    print(f"  Correct: {baseline_perf['correct']}/{baseline_perf['total']}")
    
    print(f"\nFine-tuning Metrics:")
    print(f"  Epochs: {ft_metrics.get('epochs', 'N/A')}")
    print(f"  Final policy loss: {ft_metrics['history'][-1].get('policy_loss', 'N/A'):.4f}")
    print(f"  Final value loss: {ft_metrics['history'][-1].get('value_loss', 'N/A'):.4f}")
    
    print(f"\nModel saved to: checkpoints/tool_selector_ppo.pt")
    print("\nNext steps:")
    print("  1. Use the fine-tuned model in a new agent for improved inference")
    print("  2. Collect more trajectories and iterate")
    print("  3. Test on held-out tasks")


if __name__ == "__main__":
    main()
