"""
End-to-End Example: Baseline → Verification → PPO Fine-tuning

This script demonstrates the complete workflow:
1. Run baseline agents
2. Verify tool selections
3. Collect trajectories with rewards
4. Fine-tune with PPO
5. Evaluate improvements
"""

import os
from pathlib import Path
from typing import Dict, List

# Import all components
from multi_agent_baseline import MultiAgentBaseline, AgentTrajectory, MultiAgentResult
from ppo_finetuner import TaskScorer, PPOFineTuner, DecisionCollector
from tool_verifier import SimpleToolVerifier, LearnableToolVerifier, VerifierEnsemble


def setup_tasks() -> Dict[str, str]:
    """Define tasks with ground truth answers."""
    return {
        "math_1": {
            "task": "What is 25 * 4?",
            "answer": "100",
            "required_tools": ["calculator"]
        },
        "math_2": {
            "task": "What is 144 / 12?",
            "answer": "12",
            "required_tools": ["calculator"]
        },
        "math_3": {
            "task": "What is 2 + 3 * 5?",
            "answer": "17",
            "required_tools": ["calculator"]
        },
        "search_1": {
            "task": "Who is the current president of the United States?",
            "answer": "Joe Biden",
            "required_tools": ["web_search"]
        },
        "search_2": {
            "task": "What year did the Titanic sink?",
            "answer": "1912",
            "required_tools": ["web_search"]
        },
    }


def print_header(title: str, char: str = "="):
    """Print formatted header."""
    print(f"\n{char * 70}")
    print(f"{title:^70}")
    print(f"{char * 70}\n")


def phase_1_collect_baseline(tasks: Dict[str, str], num_agents: int = 2) -> Dict[str, MultiAgentResult]:
    """
    Phase 1: Collect baseline trajectories.
    
    Returns:
        Dict mapping task_id -> MultiAgentResult
    """
    print_header("PHASE 1: Collect Baseline Trajectories")
    
    baseline = MultiAgentBaseline(
        num_agents=num_agents,
        model="qwen3:8b",  # Make sure this is available locally
        tavily_api_key=os.getenv("TAVILY_API_KEY")
    )
    
    results = {}
    for task_id, task_info in tasks.items():
        print(f"\n→ Running task: {task_id}")
        print(f"  Query: {task_info['task']}")
        
        result = baseline.solve_task(
            task_id,
            task_info['task'],
            max_steps_per_agent=3
        )
        results[task_id] = result
    
    return results


def phase_2_verify_tools(baseline_results: Dict[str, MultiAgentResult],
                        tasks: Dict[str, str]) -> None:
    """
    Phase 2: Verify tool selections from baseline runs.
    
    Analyzes which tools agents used and checks if they were appropriate.
    """
    print_header("PHASE 2: Verify Tool Selections")
    
    verifier = SimpleToolVerifier()
    
    total_tool_calls = 0
    correct_tools = 0
    
    for task_id, result in baseline_results.items():
        task_info = tasks[task_id]
        required_tools = task_info["required_tools"]
        
        print(f"\nTask: {task_id}")
        print(f"  Query: {task_info['task']}")
        print(f"  Expected tools: {required_tools}")
        
        for i, traj in enumerate(result.agent_trajectories):
            print(f"\n  Agent {traj.agent_id}:")
            
            for step in traj.steps:
                total_tool_calls += 1
                tool = step.tool_called
                
                # Verify tool
                verification = verifier.verify(
                    task_info['task'],
                    tool,
                    step.tool_args
                )
                
                if verification.is_valid:
                    correct_tools += 1
                    status = "✓"
                else:
                    status = "✗"
                
                print(f"    {status} {tool:15} (conf={verification.confidence:.2f})")
    
    print(f"\n\nVerification Summary:")
    print(f"  Total tool calls: {total_tool_calls}")
    print(f"  Correct selections: {correct_tools}/{total_tool_calls} "
          f"({100*correct_tools/total_tool_calls:.1f}%)")


def phase_3_score_trajectories(baseline_results: Dict[str, MultiAgentResult],
                               tasks: Dict[str, str]) -> TaskScorer:
    """
    Phase 3: Score final answers and prepare rewards.
    
    Returns:
        TaskScorer with registered ground truth answers
    """
    print_header("PHASE 3: Score Final Answers")
    
    scorer = TaskScorer()
    
    # Register ground truth
    for task_id, task_info in tasks.items():
        scorer.register_ground_truth(task_id, task_info['answer'])
    
    # Score results
    print("Task Scores:")
    print(f"{'Task':<15} {'Agent':<8} {'Answer':<30} {'Score':<8}")
    print("-" * 65)
    
    for task_id, result in baseline_results.items():
        for traj in result.agent_trajectories:
            answer = traj.final_answer or "No answer"
            score = scorer.score_answer(task_id, answer)
            
            # Truncate long answers
            answer_display = (answer[:28] + "...") if len(answer) > 28 else answer
            
            print(f"{task_id:<15} {traj.agent_id:<8} {answer_display:<30} {score:.2f}")
    
    return scorer


def phase_4_collect_decisions(baseline_results: Dict[str, MultiAgentResult],
                              tasks: Dict[str, str],
                              scorer: TaskScorer) -> List:
    """
    Phase 4: Collect tool selection decisions for training.
    
    Returns:
        List of ToolSelectionDecision objects
    """
    print_header("PHASE 4: Collect Training Decisions")
    
    collector = DecisionCollector(scorer)
    
    task_descriptions = {tid: tinfo['task'] for tid, tinfo in tasks.items()}
    
    # Flatten trajectories
    all_trajectories = []
    for result in baseline_results.values():
        all_trajectories.extend(result.agent_trajectories)
    
    for traj in all_trajectories:
        collector.collect_from_trajectory(
            traj,
            task_descriptions[traj.task_id],
            traj.final_answer
        )
    
    decisions = collector.get_all_decisions()
    
    print(f"Collected {len(all_trajectories)} trajectories")
    print(f"Total decisions: {len(decisions)}")
    
    # Analyze rewards
    rewards = [d.reward for d in decisions]
    import statistics
    print(f"Reward statistics:")
    print(f"  Mean: {statistics.mean(rewards):.3f}")
    print(f"  Min: {min(rewards):.3f}")
    print(f"  Max: {max(rewards):.3f}")
    print(f"  Decisions with high reward (>0.8): {sum(1 for r in rewards if r > 0.8)}")
    
    return decisions, all_trajectories, collector


def phase_5_ppo_finetuning(trajectories: List[AgentTrajectory],
                           tasks: Dict[str, str],
                           scorer: TaskScorer,
                           output_dir: str = "checkpoints") -> Dict:
    """
    Phase 5: Fine-tune tool selection with PPO.
    
    Returns:
        Training metrics
    """
    print_header("PHASE 5: PPO Fine-tuning")
    
    fine_tuner = PPOFineTuner(scorer, device="cpu")
    
    # Collect trajectories
    task_descriptions = {tid: tinfo['task'] for tid, tinfo in tasks.items()}
    fine_tuner.collect_trajectories(trajectories, task_descriptions)
    
    # Train
    print("\nStarting PPO training...")
    metrics = fine_tuner.fine_tune(
        num_epochs=3,
        batch_size=4,
        learning_rate=1e-4
    )
    
    # Save
    Path(output_dir).mkdir(exist_ok=True)
    model_path = f"{output_dir}/tool_selector_ppo.pt"
    fine_tuner.save_model(model_path)
    
    print(f"\n✓ Model saved to {model_path}")
    
    return metrics


def phase_6_analysis(metrics: Dict) -> None:
    """
    Phase 6: Analyze training results.
    """
    print_header("PHASE 6: Training Analysis")
    
    print("PPO Training Curves:")
    print(f"{'Epoch':<8} {'Policy Loss':<15} {'Value Loss':<15} {'Entropy':<15}")
    print("-" * 55)
    
    for i, epoch_metrics in enumerate(metrics['history']):
        print(f"{i+1:<8} {epoch_metrics.get('policy_loss', 0):<15.4f} "
              f"{epoch_metrics.get('value_loss', 0):<15.4f} "
              f"{epoch_metrics.get('entropy', 0):<15.4f}")
    
    print("\nTraining completed successfully!")
    print("Next steps:")
    print("  1. Integrate fine-tuned model into agents")
    print("  2. Collect more evaluation data")
    print("  3. Measure improvement in task success rate")


def main():
    """Run complete end-to-end pipeline."""
    
    print("\n" + "=" * 70)
    print("MULTI-AGENT + PPO FINE-TUNING PIPELINE".center(70))
    print("=" * 70)
    
    # Setup
    print("\nInitializing...")
    tasks = setup_tasks()
    print(f"✓ Loaded {len(tasks)} tasks")
    
    try:
        # Phase 1: Baseline
        baseline_results = phase_1_collect_baseline(tasks, num_agents=2)
        
        # Phase 2: Verify
        phase_2_verify_tools(baseline_results, tasks)
        
        # Phase 3: Score
        scorer = phase_3_score_trajectories(baseline_results, tasks)
        
        # Phase 4: Collect decisions
        decisions, trajectories, collector = phase_4_collect_decisions(
            baseline_results, tasks, scorer
        )
        
        # Phase 5: PPO fine-tuning
        metrics = phase_5_ppo_finetuning(trajectories, tasks, scorer)
        
        # Phase 6: Analysis
        phase_6_analysis(metrics)
        
        print_header("COMPLETE", char="*")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
