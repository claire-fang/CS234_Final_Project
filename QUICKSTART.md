# Quick Start Guide: PPO Fine-tuning for Tool Selection

## 5-Minute Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Ollama (if not running)

```bash
# In one terminal
ollama serve

# In another terminal, pull model
ollama pull qwen3:8b
```

### 3. Run the End-to-End Example

```bash
python example_e2e.py
```

This runs all phases:
- Phase 1: Collect baseline trajectories
- Phase 2: Verify tool selections  
- Phase 3: Score final answers
- Phase 4: Collect training decisions
- Phase 5: PPO fine-tuning
- Phase 6: Analyze results

---

## Key Files & What They Do

| File | Purpose |
|------|---------|
| `multi_agent_baseline.py` | Base system (agents, tools, trajectory collection) |
| `tool_verifier.py` | **Verifier that outputs yes/no for tool selection** |
| `ppo_finetuner.py` | **PPO trainer for tool selection** |
| `integration.py` | Full pipeline + analytics |
| `example_e2e.py` | Complete end-to-end example |
| `README.md` | Full documentation |

---

## Core Workflow

### Option A: Simple (Recommended for first run)

```python
from example_e2e import main
main()
```

### Option B: Step-by-step

```python
# 1. Run baseline
from multi_agent_baseline import MultiAgentBaseline
baseline = MultiAgentBaseline(num_agents=2)
result = baseline.solve_task("q1", "What is 15 * 7?")

# 2. Verify tool selections
from tool_verifier import SimpleToolVerifier
verifier = SimpleToolVerifier()
for step in result.agent_trajectories[0].steps:
    check = verifier.verify("What is 15 * 7?", step.tool_called)
    print(f"{step.tool_called}: {check.is_valid}")

# 3. Score and fine-tune
from ppo_finetuner import TaskScorer, PPOFineTuner
scorer = TaskScorer()
scorer.register_ground_truth("q1", "105")

fine_tuner = PPOFineTuner(scorer)
fine_tuner.collect_trajectories(
    result.agent_trajectories,
    {"q1": "What is 15 * 7?"}
)
fine_tuner.fine_tune(num_epochs=3)
fine_tuner.save_model("model.pt")
```

### Option C: Custom tasks

```python
tasks = {
    "custom_1": {
        "task": "Your question here",
        "answer": "Expected answer",
        "required_tools": ["calculator"]  # optional
    }
}

# Add to example_e2e.py setup_tasks() or use directly with PPOFineTuner
```

---

## The Verifier: Yes/No Signal

The verifier is the key innovation. It outputs:

```python
from tool_verifier import SimpleToolVerifier

verifier = SimpleToolVerifier()
result = verifier.verify(
    task="What is 15 * 7?",
    tool="calculator"
)

print(result.is_valid)      # True/False signal
print(result.confidence)    # 0.0 - 1.0
print(result.reason)        # Why this verdict
```

**Built-in verifiers:**

| Verifier | Best For |
|----------|----------|
| `SimpleToolVerifier` | Rule-based, static (no training) |
| `LearnableToolVerifier` | Learns from trajectory outcomes |
| `VerifierEnsemble` | Combines multiple verifiers |

Example with learning:

```python
from tool_verifier import LearnableToolVerifier

verifier = LearnableToolVerifier()

# Learn from successful trajectory
verifier.train_on_trajectory(
    task="What is 15 * 7?",
    trajectory_tools=["calculator"],
    task_success=True
)

# Confidence increases for similar tasks
result = verifier.verify("What is 20 * 5?", "calculator")
print(result.confidence)  # Higher than before
```

---

## The PPO Trainer: Why/How

**Why PPO?**
- Simple to implement
- No separate reward model needed
- Works well with small models on CPU
- Stable training

**How it works:**

1. Collect tool selection decisions
2. Score trajectories by task correctness
3. Reward decisions in successful trajectories
4. Use PPO to optimize policy

**Key idea:** All decisions in a successful trajectory get the same reward signal.

Configuration:

```python
fine_tuner.fine_tune(
    num_epochs=3,          # More epochs = longer training
    batch_size=8,          # Smaller batch = cheaper but noisier
    learning_rate=1e-4     # Learning rate
)
```

---

## Troubleshooting

### Ollama Connection Error

```bash
# Make sure Ollama is running
ollama serve

# Check it's accessible
curl http://localhost:11434/api/tags
```

### No Tavily API Key

```bash
# Optional - web search won't work without it
export TAVILY_API_KEY="your-key-here"
# Or just skip web search tasks
```

### CPU Too Slow

- Reduce `num_agents` (use 1-2 instead of 3)
- Reduce `max_steps_per_agent` (use 2-3 instead of 5)
- Use simpler model (if available)
- Reduce batch size in fine-tuning

### Model Not Improving

Likely causes:
1. Tasks too hard for base model
2. Scorer not working correctly (test manually)
3. Not enough data (collect more trajectories)
4. Training too short (increase num_epochs)

Solutions:
```python
# Test scorer
scorer = TaskScorer()
scorer.register_ground_truth("q1", "105")
print(scorer.score_answer("q1", "105"))      # Should be 1.0
print(scorer.score_answer("q1", "wrong"))    # Should be 0.0

# Increase data
fine_tuner.fine_tune(num_epochs=10)  # More training

# Check trajectories were scored
decisions = fine_tuner.collector.get_all_decisions()
print(f"Decisions with high reward: {sum(1 for d in decisions if d.reward > 0.8)}")
```

---

## What's Happening Behind the Scenes

```
Agent 1               Agent 2               Agent 3
  |                    |                      |
  v                    v                      v
[calculator]      [web_search]          [calculator]
[web_fetch]       [extract]             [calculator]
  |                    |                      |
  v                    v                      v
"105"             "No answer"              "105"
  |                    |                      |
  v                    v                      v
Score: 1.0         Score: 0.0             Score: 1.0
  |                    |                      |
  +----------+----------+----------+
             |
             v
    PPO Trainer sees:
    - "calculator" is rewarded (1.0)
    - "web_search" is rewarded (0.0)
    - Pattern: calculator works for math
    
    Updates policy:
    P(calculator | math_task) ↑
    P(web_search | math_task) ↓
```

---

## Next Steps After Quick Start

1. **Test on your own tasks:** Modify setup_tasks() in example_e2e.py
2. **Combine multiple tools:** Try tasks requiring search + extract
3. **Measure improvement:** Run agents before/after fine-tuning on held-out test set
4. **Iterate:** Collect more data, fine-tune again, repeat
5. **Production:** Integrate fine-tuned model into real agent system

---

## File Sizes & Performance

On Mac M1/M2 (CPU only):

| Operation | Time | Memory |
|-----------|------|--------|
| Collect 1 trajectory | 10-30s | <50MB |
| PPO epoch (batch_size=8) | 1-2s | ~100MB |
| Full pipeline (5 tasks) | ~5 min | <200MB |

---

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `TAVILY_API_KEY` | Web search API | `export TAVILY_API_KEY="abc123"` |
| `OLLAMA_BASE_URL` | Ollama endpoint | (default: localhost:11434) |

---

## Common Patterns

### Pattern 1: Train on fixed set, evaluate on new

```python
# Train on 5 tasks
train_tasks = ["q1", "q2", "q3", "q4", "q5"]
# Results saved to checkpoints/tool_selector_ppo.pt

# Evaluate on new tasks (use fine-tuned model)
eval_tasks = ["q6", "q7", "q8"]
```

### Pattern 2: Iterative improvement

```python
for iteration in range(3):
    # Collect
    trajectories = collect_baseline_trajectories(tasks)
    
    # Fine-tune
    fine_tuner = PPOFineTuner(scorer)
    fine_tuner.collect_trajectories(trajectories, tasks)
    fine_tuner.fine_tune(num_epochs=5)
    
    # Evaluate
    # (in next iteration, agents should use fine-tuned model)
```

### Pattern 3: Add custom verifier

```python
from tool_verifier import SimpleToolVerifier

class CustomVerifier(SimpleToolVerifier):
    def verify(self, task, tool, context=""):
        # Your custom logic
        return result

# Use in fine-tuner
fine_tuner = PPOFineTuner(scorer)
fine_tuner.verifier = CustomVerifier()
```

---

## Learning Resources

- **PPO Paper:** https://arxiv.org/abs/1707.06347
- **GAE:** https://arxiv.org/abs/1506.02438
- **Tool Use in LLMs:** https://arxiv.org/abs/2210.03629
- **Ollama:** https://ollama.ai

---

**Happy fine-tuning! 🚀**

For questions, check README.md or modify example_e2e.py to debug.
