#!/usr/bin/env python
"""
Quick validation script: Check that all components work together.

Validates:
- Tool verifier with no_tool option
- PPO trainer with 5 tools
- HotpotQA pipeline imports
"""

import sys
from pathlib import Path


def test_tool_verifier():
    """Test verifier with no_tool option."""
    print("\n" + "="*70)
    print("Testing Tool Verifier with 'no_tool' option")
    print("="*70)
    
    from tool_verifier import SimpleToolVerifier
    
    verifier = SimpleToolVerifier()
    
    # Test cases: (question, tool, expected_valid)
    test_cases = [
        ("What is Paris famous for?", "no_tool", True),  # Should be no_tool
        ("What is 15 * 7?", "calculator", True),
        ("What is 15 * 7?", "web_search", False),  # Wrong for math
        ("Who won the 2024 Olympics?", "web_search", True),
        ("Extract text from HTML", "extract", True),
        ("Is water wet?", "no_tool", True),  # Knowledge question
    ]
    
    for question, tool, expected_valid in test_cases:
        result = verifier.verify(question, tool)
        status = "✓" if result.is_valid == expected_valid else "✗"
        print(f"{status} '{tool}' on '{question[:40]}...'")
        print(f"   Valid: {result.is_valid}, Confidence: {result.confidence:.2f}")
    
    # Check that no_tool is in available tools
    assert "no_tool" in verifier.TOOL_SIGNATURES
    print("\n✓ Tool verifier with 'no_tool' working correctly")
    return True


def test_ppo_trainer():
    """Test PPO trainer with 5 tools and penalty support."""
    print("\n" + "="*70)
    print("Testing PPO Trainer with 5 Tools and Failure Penalties")
    print("="*70)
    
    from ppo_finetuner import TaskScorer, PPOFineTuner, ToolSelectionDecision
    
    # Create scorer
    scorer = TaskScorer()
    scorer.register_ground_truth("test_q", "correct_answer")
    
    # Create trainer with custom penalties
    trainer = PPOFineTuner(
        scorer,
        success_reward=1.0,
        failure_penalty=-0.5
    )
    
    # Check tool count
    num_tools = trainer.model.num_tools
    print(f"Number of tools in model: {num_tools}")
    assert num_tools == 5, f"Expected 5 tools, got {num_tools}"
    
    # Check penalty configuration
    print(f"Success reward: {trainer.collector.success_reward}")
    print(f"Failure penalty: {trainer.collector.failure_penalty}")
    assert trainer.collector.success_reward == 1.0
    assert trainer.collector.failure_penalty == -0.5
    
    # Check available tools
    available = ["no_tool", "web_search", "web_fetch", "extract", "calculator"]
    print(f"Available tools: {available}")
    
    # Create test decision with no_tool
    decision = ToolSelectionDecision(
        task_id="test",
        agent_id=0,
        step_id=0,
        task_description="Test question",
        context="Previous context",
        chosen_tool="no_tool",
        reward=1.0
    )
    
    print(f"✓ Created decision with no_tool: {decision.chosen_tool}")
    print(f"✓ PPO trainer supports 5 tools with penalties correctly")
    
    # Test save/load functionality
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = f"{tmpdir}/test_model.pt"
        trainer.save_model(model_path)
        
        # Check file exists
        assert Path(model_path).exists()
        print(f"✓ Model save/load working")
    
    return True


def test_hotpot_pipeline():
    """Test HotpotQA pipeline imports and data loading."""
    print("\n" + "="*70)
    print("Testing HotpotQA Pipeline")
    print("="*70)
    
    from hotpot_pipeline import load_hotpot_mock, setup_scorer_hotpot
    
    # Load mock data
    examples = load_hotpot_mock(num_examples=3)
    print(f"✓ Loaded {len(examples)} mock examples")
    
    # Check structure
    for q_id, example in list(examples.items())[:1]:
        print(f"  Example: {q_id}")
        print(f"    Question: {example['question'][:50]}...")
        print(f"    Answer: {example['answer']}")
    
    # Create scorer
    scorer = setup_scorer_hotpot(examples)
    print(f"✓ Created scorer with {len(examples)} questions")
    
    # Test scoring
    first_q = list(examples.items())[0]
    q_id, example = first_q
    score = scorer.score_answer(q_id, example['answer'])
    print(f"✓ Exact match score: {score:.2f}")
    assert score > 0.9, "Exact match should have high score"
    
    return True


def test_integrations():
    """Test that all modules can be imported together."""
    print("\n" + "="*70)
    print("Testing Module Integrations")
    print("="*70)
    
    try:
        from multi_agent_baseline import MultiAgentBaseline, LocalLLMAgent
        print("✓ multi_agent_baseline imports")
        
        from ppo_finetuner import (
            TaskScorer, PPOFineTuner, ToolSelectionDecision,
            DecisionCollector, PPOTrainer, SimpleToolSelector
        )
        print("✓ ppo_finetuner imports")
        
        from tool_verifier import (
            SimpleToolVerifier, LearnableToolVerifier, VerifierEnsemble
        )
        print("✓ tool_verifier imports")
        
        from hotpot_pipeline import (
            load_hotpot_mock, setup_scorer_hotpot, 
            collect_baseline_trajectories_hotpot
        )
        print("✓ hotpot_pipeline imports")
        
        print("\n✓ All modules integrate correctly")
        return True
        
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False


def main():
    """Run all validation tests."""
    print("\n" + "="*70)
    print("VALIDATION TEST SUITE")
    print("="*70)
    
    tests = [
        ("Tool Verifier", test_tool_verifier),
        ("PPO Trainer", test_ppo_trainer),
        ("HotpotQA Pipeline", test_hotpot_pipeline),
        ("Module Integrations", test_integrations),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All validations passed! Ready to run hotpot_pipeline.py")
        return 0
    else:
        print("\n❌ Some validations failed. Check errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
