# HotpotQA Evaluation Setup - Complete Guide

You now have a complete end-to-end system to train and evaluate your PPO fine-tuning approach on real multi-hop reasoning questions from HotpotQA.

## What Was Added

### 1. **Tool Verifier with "No Tool" Option** ✅
- File: [tool_verifier.py](tool_verifier.py)
- New tool: `"no_tool"` - signals when questions may not need external tools
- The verifier outputs `yes/no` signals for any (task, tool) pair
- Three implementations:
  - `SimpleToolVerifier`: Rule-based (immediate use)
  - `LearnableToolVerifier`: Learns from outcomes  
  - `VerifierEnsemble`: Combinesverifiers

Example:
```python
from tool_verifier import SimpleToolVerifier

verifier = SimpleToolVerifier()

# Check if tool is appropriate
result = verifier.verify(
    task="What is the capital of France?",
    tool="no_tool"  # NEW: can ask if tool needed
)
# result.is_valid: bool
# result.confidence: float (0-1)
# result.reason: str
```

**Available tools:**
- `"no_tool"` - Answer from reasoning/knowledge (NEW!)
- `"web_search"` - Find information
- `"web_fetch"` - Get URL content  
- `"extract"` - Parse HTML
- `"calculator"` - Do math

### 2. **HotpotQA Pipeline** ✅
- File: [hotpot_pipeline.py](hotpot_pipeline.py)
- Loads ~20 real multi-hop reasoning questions
- Automatically converts failed imports to mock data
- 6-phase pipeline: load → collect → verify → analyze → train → evaluate

Run it:
```bash
python hotpot_pipeline.py
```

### 3. **Updated PPO Trainer** ✅
- File: [ppo_finetuner.py](ppo_finetuner.py)
- Now supports 5 tools (was 4)
- Fully compatible with tool selection decisions
- Can learn when to use (or not use) tools

### 4. **Validation Suite** ✅
- File: [validate.py](validate.py)
- Tests all components work together
- Reports any issues

Run it:
```bash
python validate.py  # Should show "4/4 tests passed"
```

## Quick Start (3 steps)

### Step 1: Install (if needed)
```bash
pip install torch numpy requests
# Optional for real HotpotQA data:
pip install datasets
```

### Step 2: Start Ollama
```bash
ollama serve
```

### Step 3: Run the Pipeline
```bash
python hotpot_pipeline.py
```

**Expected output:**
```
HOTPOT QA PIPELINE: Train & Evaluate PPO Fine-tuning
Loading Data
✓ Loaded 20 mock HotpotQA examples

PHASE 1: Collect Baseline Trajectories...
PHASE 2: Verify Tool Usage...
PHASE 3: Analyze Baseline...
PHASE 4: Fine-tune with PPO...

✓ Model saved to checkpoints/hotpot_tool_selector.pt
```

**Time:** ~5-15 minutes depending on LLM speed

## Key Changes

### Tool Verifier Now Supports "No Tool"

Before:
```python
TOOLS = ["web_search", "web_fetch", "extract", "calculator"]
# Can't express: "this question doesn't need tools"
```

After:
```python
TOOLS = ["no_tool", "web_search", "web_fetch", "extract", "calculator"]
# Can now decide: answer from knowledge vs. tool needed
```

### PPO Trainer Now Handles 5 Tools

Before:
```python
num_tools = 4
agent_action ∈ {calculator, web_search, web_fetch, extract}
```

After:
```python
num_tools = 5
agent_action ∈ {no_tool, calculator, web_search, web_fetch, extract}
```

### Real Data Available

Before:
```python
tasks = {
    "math_1": {"task": "15 * 7", "answer": "105"},
    "search_1": {"task": "Who won 2024 Olympics?", "answer": "USA"},
}
```

After:
```python
# Real HotpotQA: "Which country is birthplace of author of X?"
examples = load_hotpot_data(split="train", max_examples=20)
# Auto-handles missing datasets library with mock data
```

## Pipeline Details

### Phase 1: Load Data
- Loads HotpotQA examples
- Falls back to mock data if `datasets` library missing
- Default: 20 examples

```python
examples = load_hotpot_data(split="train", max_examples=20)
```

### Phase 2: Collect Trajectories
- Runs baseline agents (2 agents per question)
- Records all tool selections
- Captures final answers

Output:
```
[1/20] hotpot_mock_0
  Q: Which country is the birthplace of the author of...
  Agent 0: Colombia
  Agent 1: Colombia
```

### Phase 3: Verify Tool Usage
- Checks if tools used were appropriate
- Doesn't affect training (just diagnostic)
- Shows tool selection quality before fine-tuning

Output:
```
Verified correct: 12/45 (26.7%)
Verified incorrect: 33/45 (73.3%)
Tool usage breakdown:
  web_search: 30
  no_tool: 10
  calculator: 5
```

### Phase 4: Analyze Baseline Performance
- Scores final answers (0-1)
- Measures accuracy without fine-tuning
- Shows average tool usage

Output:
```
Baseline Performance:
  Accuracy: 45.00% (9/20)
  Avg tool calls per trajectory: 2.1
```

### Phase 5: Fine-tune with PPO
- Collects all tool decisions
- Rewards decisions in successful trajectories
- Trains neural network with PPO algorithm
- Saves model to `checkpoints/hotpot_tool_selector.pt`

Output:
```
Fine-tuning on 40 decisions (5 epochs)...
  Epoch 1/5: policy_loss=0.6234 value_loss=0.4521 entropy=1.2341
  Epoch 2/5: policy_loss=0.5123 value_loss=0.3456 entropy=1.1234
  Epoch 3/5: policy_loss=0.4521 value_loss=0.2789 entropy=1.0456
  Epoch 4/5: policy_loss=0.4123 value_loss=0.2345 entropy=1.0123
  Epoch 5/5: policy_loss=0.3789 value_loss=0.2012 entropy=0.9876
✓ Model saved
```

### Phase 6: Evaluate
- Analyzes what model learned
- Shows tool patterns in successful trajectories
- Provides insights for next iteration

Output:
```
Training decisions statistics:
  Total: 45
  High reward (>0.8): 12
  Low reward (<0.5): 20

Tools in successful trajectories:
  web_search: 8
  no_tool: 3
  calculator: 1
```

## Configuration Options

### Change Number of Examples
```python
# In hotpot_pipeline.py, line ~110
examples = load_hotpot_data(split="train", max_examples=50)
```

### Use Real HotpotQA (not mock)
```bash
pip install datasets
# Then hotpot_pipeline.py will automatically load real data
```

### Train More
```python
# More epochs = longer training but potentially better results
ft_metrics = fine_tune_on_hotpot(
    all_trajectories,
    examples,
    scorer,
    num_epochs=10,  # Default is 5
)
```

### Change Baseline Agents
```python
# More agents per question
all_trajectories, question_texts = collect_baseline_trajectories_hotpot(
    examples,
    num_agents=3,  # Default is 2
    max_steps=4,
)
```

## Expected Results

### Baseline
- Accuracy: 30-50% (depends on LLM quality)
- Tool verification: Many tools marked "wrong" initially
- Average tool calls: 1.5-3 per trajectory

### After Fine-tuning
- No immediate accuracy improvement (model is young)
- Tool patterns should emerge (e.g., web_search used more for factual Q's)
- Model learns which tools correlate with success

**Note:** To truly measure improvement, need to integrate fine-tuned model back into agents and re-evaluate on test set.

## Understanding "No Tool" Option

### When to predict "no_tool"
Examples:
- "Is water wet?" → Can answer from knowledge
- "Who wrote Romeo and Juliet?" → Well-known fact
- "What is the capital of France?" → Common knowledge

### When _not_ to predict "no_tool"
Examples:
- "Who won the 2024 Olympics medal in X?" → Needs current info (web_search)
- "What is 15 * 7?" → Needs calculation (calculator)
- Current definitions/prices → Needs search

### How it works in PPO training

The model learns:
```
If answer_correct & tool_used_was_no_tool → Reward ✓
If answer_correct & tool_used_was_web_search → Reward ✓
If answer_wrong & tool_used_was_calculator → No reward ✗

Pattern: Learn which tools → right for right questions
```

## What Could Go Wrong

### "Accuracy is very low (0-20%)"

Likely causes:
1. LLM can't answer the questions
2. Scorer too strict
3. Question format doesn't match expected answer

Solution:
```python
# Test scorer manually
scorer = setup_scorer_hotpot(examples)
example = list(examples.items())[0]
q_id, data = example
score = scorer.score_answer(q_id, data['answer'])
print(f"Exact match score: {score}")  # Should be 1.0
```

### "Verifier says tools wrong, but answers correct"

This is **not a problem**!

Reason:
- Verifier output is heuristic feedback for debugging
- **Actual signal is task success** (whether answer is right)
- PPO learns utility of tools from rewards, not verifier

### "No improvement in tool usage after fine-tuning"

Possible reasons:
1. Baseline already optimal
2. Not enough training data
3. Reward signal too noisy
4. Need larger model

Solutions:
```python
# Collect more data (50+ examples)
examples = load_hotpot_data(split="train", max_examples=100)

# Train longer
fine_tune_on_hotpot(..., num_epochs=20)

# Inspect what model learned
decisions = trainer.collector.get_all_decisions()
high_reward = [d for d in decisions if d.reward > 0.8]
print(f"Tool patterns in successful trajectories:")
for d in high_reward[:5]:
    print(f"  {d.task_description[:40]}... → {d.chosen_tool}")
```

## Next Steps

1. **Run the basic pipeline**
   ```bash
   python hotpot_pipeline.py
   ```

2. **Inspect results**
   - Check baseline accuracy
   - Look at tool usage patterns
   - Review what verifier says

3. **Scale up (optional)**
   ```python
   # More data for better training
   examples = load_hotpot_data(split="train", max_examples=100)
   ```

4. **Integrate into agents**
   - Modify agent to use fine-tuned model for tool selection
   - A/B test: agent with fine-tuning vs without

5. **Iterate**
   - Collect more data
   - Improve verifier rules
   - Refine reward signal

## Files Reference

| File | Purpose |
|------|---------|
| `hotpot_pipeline.py` | Main entry point - run this! |
| `tool_verifier.py` | Verifier with "no_tool" support |
| `ppo_finetuner.py` | PPO trainer (5 tools) |
| `multi_agent_baseline.py` | Baseline agents |
| `validate.py` | Validation tests |
| `HOTPOT_README.md` | Detailed documentation |
| `requirements.txt` | Dependencies |

## Key Takeaway

You now have a complete system to:

✅ **Collect data** on real reasoning questions  
✅ **Evaluate** current tool usage  
✅ **Verify** if tools are appropriate  
✅ **Train** with PPO on actual task success  
✅ **Test** your specific idea: "Can fine-tuning improve tool selection?"

The verifier now supports **"no tool needed"** decisions, making the system more realistic.

---

**Ready?** Run: `python hotpot_pipeline.py` 🚀
