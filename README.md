# Multi-Agent Baseline with PPO Fine-tuning for Tool Selection

A lightweight, CPU-friendly system for fine-tuning multi-agent tool selection using PPO with task correctness as the reward signal.

## Overview

**Problem**: How to improve tool selection in multi-agent systems without labeled data?

**Solution**: 
1. Run baseline agents to collect trajectories
2. Score trajectories by task correctness (final answer quality)
3. Use PPO to optimize tool selection decisions based on task outcomes
4. Train on CPU with minimal compute requirements

## Architecture

```
┌─────────────────────────────────────┐
│  Multi-Agent Baseline               │
│  (multi_agent_baseline.py)          │
│                                     │
│  - Multiple LLM agents              │
│  - Tool execution (search, calc...) │
│  - Trajectory collection            │
└──────────────┬──────────────────────┘
               │ trajectories
               ▼
┌─────────────────────────────────────┐
│  Task Scorer                        │
│  (ppo_finetuner.py)                 │
│                                     │
│  - Score final answers              │
│  - Compute rewards (0-1)            │
│  - Annotate decisions               │
└──────────────┬──────────────────────┘
               │ decisions + rewards
               ▼
┌─────────────────────────────────────┐
│  PPO Fine-tuner                     │
│  (ppo_finetuner.py)                 │
│                                     │
│  - Simple neural network            │
│  - PPO training loop                │
│  - Optimize tool selection policy   │
└──────────────┬──────────────────────┘
               │
               ▼
        Fine-tuned Model
```

## Components

### 1. `multi_agent_baseline.py` (Existing)
- **LocalLLMAgent**: Single agent with tool-use capability
- **MultiAgentBaseline**: Runs multiple agents on same task
- Tools: web_search, web_fetch, extract, calculator

### 2. `ppo_finetuner.py` (New)

#### Key Classes:

**TaskScorer**
- Scores final answers (0-1 scale)
- Supports exact match, partial credit
- Registers ground truth answers

**ToolSelectionDecision**
- Records a single tool selection point
- Includes task context, chosen tool, reward

**DecisionCollector**
- Converts agent trajectories to training data
- Annotates decisions with task reward
- Gathers all decisions for training

**SimpleToolSelector** (Neural Network)
- Small 2-layer network (CPU-friendly)
- Input: context features (bag-of-words)
- Output: logits over 4 tools + value estimate
- ~50K parameters

**PPOTrainer**
- Implements PPO algorithm
- Generalized Advantage Estimation (GAE)
- All operations on CPU
- Gradient clipping and entropy regularization

**PPOFineTuner** (Main Interface)
- Orchestrates: collection → training → saving
- Easy API for full pipeline

### 3. `integration.py` (New)
- Full pipeline script
- Demo showing: baseline → analysis → fine-tuning
- Analytics and performance tracking

## Workflow

### Step 1: Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Start Ollama (if not running)
# Make sure your model is available:
ollama pull qwen3:8b
ollama serve
```

### Step 2: Run Baseline

```python
from integration import collect_baseline_trajectories, setup_scorer, analyze_baseline_performance

# Collect trajectories
trajectories, task_desc, task_answers = collect_baseline_trajectories(
    num_agents=2,
    model="qwen3:8b",
    tavily_api_key="your-key"  # Optional
)

# Score and analyze
scorer = setup_scorer(task_answers)
perf = analyze_baseline_performance(trajectories, scorer)
print(f"Baseline accuracy: {perf['accuracy']:.2%}")
```

### Step 3: Fine-tune with PPO

```python
from integration import fine_tune_tool_selection

metrics = fine_tune_tool_selection(
    trajectories,
    task_desc,
    scorer,
    output_dir="checkpoints"
)
```

### Step 4: Use Fine-tuned Model in Agent

```python
from ppo_finetuner import PPOFineTuner
from multi_agent_baseline import LocalLLMAgent

# Load fine-tuned model
fine_tuner = PPOFineTuner(scorer)
fine_tuner.load_model("checkpoints/tool_selector_ppo.pt")

# Use model.forward() to guide tool selection in agents
# (Integration patch needed - see next section)
```

## Key Design Decisions

### 1. Simple Neural Network (Not LLM)
- **Why**: PPO on LLMs expensive; focusing only on tool decision
- **Input**: Context as bag-of-words (hash-based, CPU-friendly)
- **Output**: Tools + value estimate

### 2. Reward = Task Correctness
- **Signal**: 1.0 if final answer correct, else 0-1 by match quality
- **Advantage**: No labeled tool correctness needed; only need ground truth for tasks
- **Trade-off**: Sparse reward, but learnable

### 3. All Decisions in Trajectory Get Same Reward
- **Rationale**: Trajectory-level reward (episodic)
- **Future**: Could use step-level intrinsic rewards (tool success)

### 4. CPU-Only Training
- No GPU required
- Small model (~50K params)
- Fast iterations
- Uses PyTorch CPU ops + efficient GAE

## Usage Examples

### Full Pipeline Script
```bash
python integration.py
```

### Minimal Demo
```bash
python demo_ppo.py
```

### Custom Tasks
```python
from ppo_finetuner import TaskScorer, PPOFineTuner
from multi_agent_baseline import MultiAgentBaseline

scorer = TaskScorer()
scorer.register_ground_truth("math1", "42")
scorer.register_ground_truth("capitals", "Paris")

# Collect data for these tasks
baseline = MultiAgentBaseline(num_agents=3)
trajectories = [
    baseline.solve_task("math1", "What is 6*7?", max_steps_per_agent=2).agent_trajectories,
    baseline.solve_task("capitals", "Capital of France?", max_steps_per_agent=2).agent_trajectories,
]

# Flatten
all_trajs = [t for batch in trajectories for t in batch]

# Fine-tune
fine_tuner = PPOFineTuner(scorer)
fine_tuner.collect_trajectories(all_trajs, {"math1": "...", "capitals": "..."})
fine_tuner.fine_tune(num_epochs=5, batch_size=4)
fine_tuner.save_model("my_model.pt")
```

## Configuration

### PPO Hyperparameters
```python
PPOTrainer(
    gamma=0.99,          # Discount factor
    gae_lambda=0.95,     # GAE smoothing
    clip_ratio=0.2,      # PPO clip range
    entropy_coeff=0.01,  # Entropy bonus
    learning_rate=1e-4,  # Adam LR
)
```

### Training Parameters
```python
fine_tuner.fine_tune(
    num_epochs=3,        # Training epochs
    batch_size=8,        # Batch size
    learning_rate=1e-4,  # Override optimizer LR
)
```

### Task Scorer
```python
scorer = TaskScorer()
score = scorer.score_answer("q1", "predicted_answer")  # Returns 0-1

# Exact matching (default)
# Partial credit: word overlap ratio

# To customize, subclass TaskScorer and override score_answer()
```

## Performance Expectations

On CPU (M1/M2):
- **Data collection**: ~1-2 min per task (depends on LLM speed)
- **Fine-tuning**: ~5-30 sec per epoch (batch_size=8)
- **Inference**: <10ms per decision

Example Results:
```
Baseline accuracy: 60% (3/5 tasks)
After PPO fine-tuning: Expected +10-20% improvement
Total time: ~10 mins for full pipeline
```

## Next Steps & Extensions

### 1. **Integrate Fine-tuned Model into Agent**
Currently, fine-tuned model is separate. To use it:
- Modify `LocalLLMAgent._execute_tool()` to query model.forward()
- Use logits as confidence scores for tool selection
- Or: Use model to re-rank LLM's tool suggestions

### 2. **Step-Level Rewards**
- Reward successful tools immediately
- Penalize failed tools
- More informative signal

### 3. **Curriculum Learning**
- Start with simple tasks
- Gradually increase difficulty
- Better convergence

### 4. **Multi-Task Learning**
- Train on multiple task types (math, search, reasoning)
- Learn generalizable tool selection

### 5. **Online Learning**
- Fine-tune incrementally as new trajectories collected
- No need to rerun baseline each iteration

### 6. **Evaluation**
- Hold-out test set of tasks
- Compare: baseline agent vs fine-tuned agent
- Track tool usage patterns

## Troubleshooting

### "Failed to connect to Ollama"
```bash
# Make sure Ollama is running
ollama serve

# In another terminal
ollama pull qwen3:8b
```

### "TAVILY_API_KEY not set"
```bash
export TAVILY_API_KEY="your-key"
# Or pass tavily_api_key parameter to MultiAgentBaseline()
```

### Slow PPO training
- Reduce batch size
- Reduce num_epochs
- Use fewer tasks for initial experiments

### Model not improving
- Check if tasks are solvable by baseline first
- Verify scorer is correct (test on manual answers)
- Try more trajectories (collect more data)
- Adjust entropy_coeff (more exploration if stuck)

## Files Overview

```
.
├── multi_agent_baseline.py   # Existing: baseline agents + tools
├── ppo_finetuner.py         # New: PPO trainer + scorer
├── integration.py           # New: pipeline + analytics
├── demo_ppo.py              # New: minimal demo script
├── requirements.txt         # Dependencies
└── checkpoints/             # Saved models
    └── tool_selector_ppo.pt
```

## References

- PPO: Proximal Policy Optimization - https://arxiv.org/abs/1707.06347
- GAE: Generalized Advantage Estimation - https://arxiv.org/abs/1506.02438
- Tool Use in LLMs: ReAct - https://arxiv.org/abs/2210.03629

## License

MIT
