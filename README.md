# PPO-Guided Tool Selection for LLM Agents

> **CS 234 Final Project — Stanford University**

## 1. Project Overview

LLM agents need external tools (web search, etc.) to answer complex multi-hop questions, but they don't inherently know **which tool to call**, **how many times**, or **when to stop searching and just answer**. This project trains a small policy network with **PPO (Proximal Policy Optimization)** to make these tool-selection decisions optimally.

The core experiment:

| Setup | Description |
|-------|-------------|
| **Baseline** | A single LLM agent decides which tool to use on its own at each step |
| **LLM + PPO** | A PPO-trained policy network selects the tool; the same LLM generates tool arguments and the final answer |

We evaluate on **HotpotQA** (multi-hop reasoning QA), where questions require combining multiple facts — e.g., *"Which country is the birthplace of the author of One Hundred Years of Solitude?"* requires (1) searching for the author → Gabriel García Márquez, (2) searching for his birthplace → Colombia, (3) knowing to stop and answer.

---

## 2. Step-by-Step Pipeline

The pipeline (`hotpot_pipeline.py`) runs three phases sequentially:

### Phase 1: Baseline Evaluation (LLM Alone)

**Model**: Qwen 3 8B (`qwen3:8b`) served locally via [Ollama](https://ollama.com/) at `http://localhost:11434`.

The LLM agent (`LocalLLMAgent` in `multi_agent_baseline.py`) solves each evaluation question independently:

1. The LLM receives the question + a system prompt listing available tools.
2. It autonomously decides to either call a tool (outputting a JSON tool-call block) or give a final answer (outputting `ANSWER: ...`).
3. Tool results are appended to the conversation history, and the LLM repeats up to `max_steps=5`.
4. The agent's final answer is scored against the ground truth.

**Tools available** (the LLM picks from these on its own in baseline mode):

| Tool | Input | What It Does |
|------|-------|--------------|
| `web_search` | `{"query": "..."}` | Calls [Tavily API](https://tavily.com/) → returns search result titles + snippets |
| `web_fetch` | `{"url": "..."}` | HTTP GET → returns first 1000 chars of page content |
| `extract` | `{"html": "..."}` | Strips HTML tags → returns clean text |
| `calculator` | `{"expr": "2+3*5"}` | Safe `eval()` on arithmetic expressions (digits and operators only) |
| `no_tool` | — | LLM answers directly from accumulated context |

**Answer scoring** uses a 3-level cascade (`TaskScorer` in `ppo_finetuner.py`):

1. **Exact match** — normalize both strings (lowercase, remove punctuation/articles), check equality → 1.0
2. **Containment** — ground truth is a substring of the answer → 1.0
3. **LLM-as-judge** — prompt a larger model (`qwen3:14b`) with "Does the prediction match the ground truth? Reply CORRECT or INCORRECT" → 1.0 or 0.0

This handles semantic equivalence (e.g., "USA" vs. "United States") that simple string matching would miss.

### Phase 2: On-Policy PPO Training

The policy network is trained on-policy over 5 iterations on the training set.

#### 2a. Policy Network Architecture

**Default model**: `DeepToolSelector` — a residual MLP with ~70K parameters, running on CPU.

```
Input (544-dim)
    │
    ▼
Linear(544 → 128) + LayerNorm + ReLU          ← input projection
    │
    ├─── Linear(128 → 128) + ReLU ──┐
    │    Linear(128 → 128)           │         ← residual block 1
    └────────────── + ──────────────┘
    │ LayerNorm + ReLU
    │
    ├─── Linear(128 → 128) + ReLU ──┐
    │    Linear(128 → 128)           │         ← residual block 2
    └────────────── + ──────────────┘
    │ LayerNorm + ReLU
    │
    ├── Linear(128 → 5)  → tool logits        ← policy head (π)
    └── Linear(128 → 1)  → state value        ← value head  (V)
```

- **Policy head**: outputs logits over the 5 tools; sampled via `Categorical` distribution
- **Value head**: outputs scalar $V(s)$, the estimated return from state $s$
- Also available: `SimpleToolSelector` — 2-layer MLP (~50K params), no residuals/LayerNorm

#### 2b. State Representation (544-dim)

The policy input is a 544-dimensional feature vector extracted from the current context string by `PPOTrainer.extract_features()`:

**Bag-of-words hash (512 dims)**:
- Tokenize context as whitespace-split words, lowercase
- For each word: `feature_vec[hash(word) % 512] += 1`
- L1-normalize: `feature_vec /= feature_vec.sum()`

**Handcrafted features (32 dims)**, addressing information that BoW cannot capture:

| Index | Feature | Computation |
|-------|---------|-------------|
| 0 | Step count | `len(re.findall(r'Step \d+:', context)) / 8` |
| 1–5 | Per-tool mention count | `context.count(tool_name) / (n_steps + 1)` for each of the 5 tools |
| 6 | Question-history word overlap | `len(question_words ∩ history_words) / len(question_words)` (stop words removed) |
| 7 | Has history | `0.0` if "No steps yet", else `1.0` |
| 8 | History text length | `min(1.0, len(history_text) / 500)` |
| 9–12 | Step position indicators | Binary: `n_steps >= 1`, `>= 2`, `>= 3`, `>= 4` |
| 13–14 | Error/failure signals | `context.count('error') / (n_steps + 1)`, same for 'failed' |
| 15–31 | Reserved | Zero-padded |

Final input: `torch.cat([bow_512, handcrafted_32])` → 544-dim vector.

#### 2c. On-Policy Training Loop

Each of the 5 iterations (`PPOFineTuner.on_policy_train()`):

1. **Collect fresh trajectories** using the **current** policy:
   - For each training question, create a `LocalLLMAgent` and call `solve_with_policy()`
   - At each step, the policy network selects a tool; the LLM generates tool arguments and executes the tool
   - Collect the full trajectory (states, actions, rewards, log probs)

2. **Compute per-step rewards** (`DecisionCollector`):

   | Signal | Value | When Applied |
   |--------|-------|--------------|
   | Step penalty | $-0.02$ | Every step |
   | Tool failure penalty | $-0.05$ | When `tool_result.ok == False` |
   | Answer correctness | $0.0$ – $1.0$ | **Last step only** |

   Example 3-step trajectory: `rewards = [-0.02, -0.02, 0.98]`

3. **Compute GAE advantages** per-trajectory (`PPOTrainer.compute_gae()`):

   GAE propagates the final correctness signal backward through the trajectory, solving the credit assignment problem.

   For each trajectory with rewards $r_t$ and value estimates $V(s_t)$:

   $$\delta_t = r_t + \gamma \cdot V(s_{t+1}) - V(s_t)$$

   $$\hat{A}_t = \sum_{l=0}^{T-t} (\gamma \lambda)^l \, \delta_{t+l}$$

   Computed backward:

   ```
   advantage = 0
   for t = T-1, T-2, ..., 0:
       δ_t = r_t + γ · V(s_{t+1}) − V(s_t)        # γ = 0.99
       advantage = δ_t + γ · λ · advantage           # λ = 0.95
       A_t = advantage
       R_t = A_t + V(s_t)                            # return target
   ```

   Advantages are then **normalized** across all decisions: $\hat{A} \leftarrow \frac{\hat{A} - \mu}{\sigma + \epsilon}$

4. **PPO clipped update** (`PPOTrainer.train_step()`):

   Mini-batches of size 4 are sampled from the collected decisions. For each batch, 3 PPO epochs:

   $$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_\text{old}}(a_t | s_t)} = \exp(\log\pi_\theta - \log\pi_{\theta_\text{old}})$$

   $$L^{\text{CLIP}} = -\mathbb{E}\left[\min\left(r_t \hat{A}_t, \; \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$$

   $$L^{\text{VF}} = \mathbb{E}\left[(V_\theta(s_t) - R_t)^2\right]$$

   $$L = L^{\text{CLIP}} + 0.5 \cdot L^{\text{VF}} - 0.01 \cdot H[\pi_\theta]$$

   where $\epsilon = 0.2$ (clip ratio), and $H[\pi_\theta] = -\sum_a \pi(a|s) \log \pi(a|s)$ is the entropy bonus.

   Gradients are clipped to max norm 0.5: `torch.nn.utils.clip_grad_norm_(params, 0.5)`

5. **Discard trajectories and repeat** — the collector is reset at each iteration, ensuring the training data always matches the current policy. This is what makes it truly on-policy.

**All hyperparameters**:

| Parameter | Value | Where Set |
|-----------|-------|-----------|
| Optimizer | Adam | `PPOTrainer.__init__` |
| Learning rate | $1 \times 10^{-4}$ | `PPOTrainer.__init__` |
| Discount factor $\gamma$ | 0.99 | `PPOTrainer.__init__` |
| GAE $\lambda$ | 0.95 | `PPOTrainer.__init__` |
| Clip ratio $\epsilon$ | 0.2 | `PPOTrainer.__init__` |
| Entropy coefficient | 0.01 | `PPOTrainer.__init__` |
| Value loss coefficient | 0.5 | `PPOTrainer.train_step` |
| Gradient clip max norm | 0.5 | `PPOTrainer.train_step` |
| Step penalty | $-0.02$ | `PPOFineTuner.__init__` |
| Tool failure penalty | $-0.05$ | `PPOFineTuner.__init__` |
| On-policy iterations | 5 | `main()` |
| PPO epochs per batch | 3 | `on_policy_train` |
| Mini-batch size | 4 | `on_policy_train` |
| Max steps per question | 5 | `main()` |
| Feature mode | structured (544-dim) | `main()` |
| Policy network | DeepToolSelector | `main()` |
| Hidden dim | 128 | `DeepToolSelector.__init__` |

### Phase 3: PPO Evaluation

After training, the PPO-trained policy is evaluated on the **same held-out eval set** as the baseline:

1. For each eval question, create a new `LocalLLMAgent` and call `solve_with_policy()` with the trained `PPOFineTuner` as the policy.
2. At each step, `PPOFineTuner.select_tool(context)` runs:
   - Extract 544-dim features from context
   - Forward pass through DeepToolSelector → logits, value
   - Sample action from `Categorical(softmax(logits))`
   - Return `(tool_name, action_idx, log_prob, value, features)`
3. The LLM generates tool arguments for the selected tool (or gives a final answer if `no_tool` was selected).
4. Score answers against ground truth with the same 3-level scorer.
5. Print side-by-side comparison: accuracy, avg tool calls, per-question detail, tool usage distribution, training loss curves.

---

## 3. Concrete Example: One Question End to End

**Question**: *"Which country is the birthplace of the author of One Hundred Years of Solitude?"*
**Ground truth**: Colombia

### Step 0: Feature Extraction

```
context = "Task: Which country is the birthplace of the author of...\nHistory: No steps yet"

# BoW hash (512-dim): hash each word into a 512-dim vector, L1-normalize
# Handcrafted (32-dim): step_count=0, all tool counts=0, overlap=0, has_history=0, ...

→ features: 544-dim tensor
```

### Step 1: Policy → `web_search`

```
features (544) → DeepToolSelector →
    policy head → logits = [-0.2, 1.8, 0.1, -0.5, -1.0]
    value  head → V(s₀) = 0.30

    softmax → probs = [0.05, 0.60, 0.07, 0.04, 0.02]
                       no_tool search fetch extract calc

    sampled: web_search (prob 0.60)
    log_prob = log(0.60) = -0.51
```

LLM generates: `{"query": "author of One Hundred Years of Solitude"}`
Tavily returns: `"Gabriel García Márquez wrote One Hundred Years of Solitude..."`

**Reward**: $r_1 = -0.02$ (step penalty; tool succeeded → no extra penalty)

### Step 2: Policy → `web_search` (again)

Context now includes Step 1 results. Policy network sees the updated 544-dim features.

```
logits = [0.5, 1.2, 0.0, -0.3, -0.8]  →  sampled: web_search
V(s₁) = 0.50
```

LLM generates: `{"query": "Gabriel García Márquez birthplace country"}`
Tavily returns: `"García Márquez was born in Aracataca, Colombia"`

**Reward**: $r_2 = -0.02$

### Step 3: Policy → `no_tool`

Context now includes both search results.

```
logits = [2.1, 0.3, -0.1, -0.5, -0.8]  →  sampled: no_tool
V(s₂) = 0.40
```

LLM outputs final answer: **"Colombia"**

**Reward**: $r_3 = -0.02 + 1.0 = 0.98$ (step penalty + answer correct)

### GAE Computation

```
rewards = [-0.02, -0.02,  0.98]
values  = [ 0.30,  0.50,  0.40]

t=2: δ₂ = 0.98 + 0.99×0 − 0.40 = 0.58           A₂ = 0.58
t=1: δ₁ = −0.02 + 0.99×0.40 − 0.50 = −0.124      A₁ = −0.124 + 0.99×0.95×0.58 = 0.42
t=0: δ₀ = −0.02 + 0.99×0.50 − 0.30 = 0.175       A₀ = 0.175 + 0.99×0.95×0.42 = 0.57
```

All three steps receive **positive** advantages (0.57, 0.42, 0.58), even though steps 1–2 had negative immediate rewards. GAE correctly credits the early `web_search` decisions for contributing to the final correct answer.

### PPO Update

```
For each step (s, a, A, R, log_π_old):
    log_π_new = log π_θ(a|s)                       # forward pass with current params
    ratio = exp(log_π_new − log_π_old)
    surr1 = ratio × A
    surr2 = clip(ratio, 0.8, 1.2) × A
    L_clip = −min(surr1, surr2)

    L_value = (V_θ(s) − R)²
    L = L_clip + 0.5 × L_value − 0.01 × entropy

    optimizer.step()
```

After this update, the policy is slightly more likely to choose `web_search` for factual questions and `no_tool` once sufficient information has been gathered.

---

## 4. Project Structure

```
├── hotpot_pipeline.py       # Main entry — orchestrates Baseline vs PPO comparison
├── multi_agent_baseline.py  # LocalLLMAgent: LLM + tool execution + policy-guided mode
├── ppo_finetuner.py         # DeepToolSelector, PPOTrainer, GAE, on-policy loop, TaskScorer
├── run_modal.py             # Deploy on Modal cloud (A10G GPU, auto Ollama setup)
├── requirements.txt         # Dependencies
├── results/                 # Training outputs
│   ├── hotpot_tool_selector.pt   # Trained policy model (.pt)
│   ├── training_results.json     # Per-iteration training metrics
│   ├── trajectories.json         # Collected on-policy trajectories
│   └── comparison.json           # Full Baseline vs PPO comparison data
└── checkpoints/
```

---

## 5. Quick Start

### Option A: Run Locally

```bash
# 1. Install dependencies
pip install torch numpy requests
pip install datasets          # for real HotpotQA data (optional; falls back to 30 mock questions)

# 2. Start Ollama and pull models
ollama serve &
ollama pull qwen3:8b          # inference model
ollama pull qwen3:14b         # judge model (for answer scoring)

# 3. (Optional) Set Tavily API key for real web search
export TAVILY_API_KEY="tvly-xxxxx"

# 4. Run the pipeline
python hotpot_pipeline.py
```

### Option B: Run on Modal (Cloud GPU)

```bash
# 1. Install and authenticate Modal
pip install modal
modal setup                   # one-time: GitHub login

# 2. Run (auto-provisions A10G GPU, installs Ollama, pulls models)
modal run run_modal.py --tavily-key "tvly-xxxxx"
```

### What Happens When You Run

1. **Load data**: 50 HotpotQA questions (40 train / 10 eval), or 30 mock questions without `datasets`
2. **Phase 1 — Baseline**: Evaluate LLM alone on 10 eval questions
3. **Phase 2 — Training**: 5 on-policy PPO iterations on 40 training questions
4. **Phase 3 — PPO eval**: Evaluate LLM + trained policy on the same 10 eval questions
5. **Final report**: Side-by-side accuracy, tool usage, per-question detail, training curves

```
  ┌──────────────────────────────────────────────────────────────────┐
  │  Metric                              Baseline (LLM)   LLM + PPO │
  ├──────────────────────────────────────────────────────────────────┤
  │  Accuracy                                  40.0%         60.0%  │
  │  Avg Tool Calls / Trajectory                 2.8           1.9  │
  ├──────────────────────────────────────────────────────────────────┤
  │  Accuracy Improvement                             +20.0%        │
  └──────────────────────────────────────────────────────────────────┘
```

---

## 6. Current Limitations

1. **Feature representation** — 544-dim (512 BoW hash + 32 handcrafted). Loses word order and deep semantics; a sentence transformer (e.g., `all-MiniLM-L6-v2`) would be a natural upgrade.
2. **Small sample size** — 50 HotpotQA questions (40 train / 10 eval) or 30 mock questions. Results have high variance.
3. **Action space underutilization** — HotpotQA is factual lookup, so only `web_search` and `no_tool` are genuinely useful; `calculator`, `web_fetch`, `extract` serve as distractors the policy must learn to avoid.
4. **Single LLM backbone** — Both tool argument generation and answer scoring depend on the same Qwen family; a stronger backbone (e.g., GPT-4) would likely improve both baseline and PPO.
5. **No sequential modeling** — The policy network is a feedforward MLP; it cannot model temporal dependencies across steps (an LSTM or Transformer policy head would help).

---

## References

- PPO: Schulman et al., "Proximal Policy Optimization Algorithms" — https://arxiv.org/abs/1707.06347
- GAE: Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" — https://arxiv.org/abs/1506.02438
- ReAct: Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" — https://arxiv.org/abs/2210.03629
- HotpotQA: Yang et al., "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering" — https://arxiv.org/abs/1809.09600

