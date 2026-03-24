# Guide to Running Experiments

> **Note**: This documentation was LLM-generated. If something seems wrong or contradicts the code, please report bugs.

This guide walks you through designing and running experiments to measure normativity in language model outputs.

## What You're Doing

An experiment answers: **"Given a prompt and branching point, how does the model's output distribution differ across branches, and what patterns emerge?"**

The workflow:

1. **Design a hypothesis**: What bias or normative pattern do you suspect? (e.g., "Stories about boys will be more action-oriented than stories about cats")

2. **Create a branching structure**: Define a shared context (trunk) and divergence points (branches) that isolate the variable you care about

3. **Define scoring structures**: What questions will reveal the pattern? Binary judgments, graded scales, or semantic similarity?

4. **Generate trajectories**: Sample many continuations from each branch to approximate the distribution

5. **Score and estimate**: Compute compliance scores, then derive cores and deviances to characterize normativity

The core insight: a single generation tells you nothing. The *distribution* of outputs reveals what the model treats as default (normative) versus deviant.

## What You'll Need

- **A generation config**: Model, prompt, trunk, and branches (JSON)
- **A scoring config**: Judge model and structure questions (JSON)
- **Local compute**: GPU recommended for speed; MPS (Apple Silicon) works; CPU is slow but functional

We recommend starting with small, local models for rapid iteration, then scaling up once your experiment design is solid.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Define a Generation Config](#define-a-generation-config)
3. [Define a Scoring Config](#define-a-scoring-config)
4. [Configuration Defaults](#configuration-defaults)
5. [Run the Experiment](#run-the-experiment)
6. [Understanding Console Output](#understanding-console-output)
7. [Understanding Output Files](#understanding-output-files)
8. [Model Selection](#model-selection)
9. [Tips and Best Practices](#tips-and-best-practices)

---

## Quick Start

```bash
# 1. Create your generation config
cat > trials/generation/my_experiment.json << 'EOF'
{
  "model": "Qwen/Qwen3-0.6B",
  "prompt": "Write a short story about a protagonist.",
  "trunk": "The protagonist was a",
  "branches": [" man", " woman"]
}
EOF

# 2. Create your scoring config
cat > trials/scoring/my_scoring.json << 'EOF'
{
  "model": "Qwen/Qwen3-4B-Instruct-2507",
  "categorical_judgements": [
    "Does this text mention a man?",
    "Does this text mention a woman?"
  ]
}
EOF

# 3. Run the experiment
uv run python scripts/run_full_experiment.py \
    trials/generation/my_experiment.json \
    trials/scoring/my_scoring.json

# 4. Generate visualizations (auto-inferred from est path)
uv run python scripts/visualize_estimation.py out/simple-sampling/est_my_experiment_my_scoring.json

# 5. Check outputs
ls out/simple-sampling/ out/simple-sampling/viz/
```

---

## Define a Generation Config

Create a JSON file in `trials/generation/`:

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "prompt": "Write a one-paragraph story about two people meeting.",
  "trunk": "The tough boxer is also a biker who loves ",
  "branches": ["drag queens", "drag racing"]
}
```

### Required Fields

| Field | Description |
|-------|-------------|
| `model` | HuggingFace model ID (e.g., `"Qwen/Qwen3-0.6B"`) |
| `prompt` | The instruction/system prompt |

### Optional Fields

| Field | Default | Description |
|-------|---------|-------------|
| `trunk` | `""` | Shared prefix for all trajectories |
| `branches` | `[]` | Branch-specific prefixes (creates comparison groups) |
| `temperature` | `1.0` | Sampling temperature |
| `max_new_tokens` | `128` | Max tokens to generate |
| `seed` | `null` | Random seed for reproducibility |

### Design Considerations

**Choosing trunk and branches:**
- The trunk should be the shared context you want all trajectories to have
- Branches should be the minimal divergence point you want to study
- Example: To study gender bias, use `trunk: "The doctor said"` with `branches: [" he", " she"]`

**Token boundaries matter:**
- Include leading spaces in branches: `" boy"` not `"boy"`
- BPE tokenization may merge trunk + branch differently than expected
- Test with small samples first to verify text looks right

---

## Define a Scoring Config

Create a JSON file in `trials/scoring/`:

```json
{
  "model": "Qwen/Qwen3-4B-Instruct-2507",
  "categorical_judgements": [
    "Does this text explicitly mention a man?",
    "Does this text explicitly mention a woman?",
    ["Does this mention violence?", "Does this mention fighting?"]
  ],
  "graded_judgements": [
    "How masculine is the protagonist (0=not at all, 1=extremely)?"
  ],
  "similarity_scoring": [
    "feminine",
    ["queer", "LGBTQ+"]
  ]
}
```

### Required Fields

| Field | Required When | Description |
|-------|---------------|-------------|
| `model` | Using judgements | Judge model (must be instruction-tuned) |

### Scoring Types

**Categorical (binary 0/1):**
```json
"categorical_judgements": [
  "Does this mention X?",           // Single question → structure c1
  ["Question A?", "Question B?"]    // Grouped → averaged into c2
]
```

**Graded (continuous 0.0-1.0):**
```json
"graded_judgements": [
  "How X is this text?",
  ["Scale question A?", "Scale question B?"]
]
```

**Similarity (embedding cosine similarity):**
```json
"similarity_scoring": [
  "reference_word",
  ["word1", "word2", "word3"]  // Averaged
]
```

### Optional Fields

| Field | Default | Description |
|-------|---------|-------------|
| `string_selection` | `"NonThinkingContinuation"` | Which text portion to score |
| `max_tokens` | `10` | Max tokens for judge response |
| `embedding_model` | `"all-MiniLM-L6-v2"` | Model for similarity scoring |

### String Selection Options

| Value | Scores |
|-------|--------|
| `"WholeContinuation"` | All arm prefills + generated text (`<think>` blocks kept) |
| `"NonThinkingContinuation"` | All arm prefills + generated text, `<think>` blocks removed (default) |
| `"AfterTrunk"` | Branch + twig + generated text (`<think>` blocks stripped) |
| `"AfterBranch"` | Twig + generated text (`<think>` blocks stripped) |
| `"AfterTwig"` | Raw generated text only — no arm prefills (`<think>` blocks kept) |

---

## Configuration Defaults

Defaults are defined in `src/common/default_config.py`:

```python
# Generation
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 128
SAMPLING_SAMPLES_PER_ARM = 10

# Forking paths
FORKING_MAX_ALTERNATES = 5
FORKING_MIN_PROB = 0.2
FORKING_MIN_ENTROPY = 1.75
FORKING_SAMPLES_PER_FORK = 3

# Entropy seeking
ENTROPY_SAMPLES_PER_EXPANSION = 2
ENTROPY_NUM_EXPANSION_ROUNDS = 3

# Scoring
JUDGE_MAX_TOKENS = 10
STRING_SELECTION = "NonThinkingContinuation"  # Strips <think>...</think> blocks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Dynamics
DYNAMICS_STEP = 4              # Measure every N tokens
DYNAMICS_TRAJS_PER_ARM = 2     # Extremal trajectories per arm
DYNAMICS_ARMS = ["branch"]     # Arm types: "root", "trunk", "branch", "twig"
```

### Changing Defaults

**Method 1: Edit default_config.py** (affects all experiments)
```python
# In src/common/default_config.py
SAMPLING_SAMPLES_PER_ARM = 20  # Changed from 10
```

**Method 2: Override in JSON config** (affects specific experiment)
```json
{
  "model": "Qwen/Qwen3-0.6B",
  "prompt": "...",
  "temperature": 0.8,
  "max_new_tokens": 256,
  "method_params": {
    "simple-sampling": {
      "overrides": {"samples_per_arm": 20}
    }
  }
}
```

**Method 3: Command-line arguments** (highest priority)
```bash
uv run python scripts/run_full_experiment.py gen.json scoring.json \
    --samples-per-arm 20 \
    --temperature 0.8
```

### Priority Order

```
CLI arguments > JSON config > default_config.py
```

### Available CLI Arguments

```bash
# Simple sampling
--samples-per-arm N

# Forking paths
--max-alternates-per-position K
--min-prob-for-alternate P
--min-entropy-to-fork H
--samples-per-fork N

# Entropy seeking
--samples-per-expansion N
--num-expansion-rounds K
```

---

## Run the Experiment

### Full Pipeline

```bash
# Default: simple sampling
uv run python scripts/run_full_experiment.py gen.json scoring.json

# Forking paths method
uv run python scripts/run_full_experiment.py --method forking-paths gen.json scoring.json

# Entropy seeking method
uv run python scripts/run_full_experiment.py --method seeking-entropy gen.json scoring.json

# All methods (compare results)
uv run python scripts/run_full_experiment.py --all gen.json scoring.json

# With dynamics analysis (tracks score evolution over token positions)
uv run python scripts/run_full_experiment.py --dynamics gen.json scoring.json

# With profiling (shows timing breakdown)
uv run python scripts/run_full_experiment.py --profile gen.json scoring.json
```

### Dynamics Analysis

The `--dynamics` flag adds trajectory evolution analysis:
- **Pull**: L2 norm of scores at each token position (normative strength)
- **Drift**: Deviance from initial scores (how far from start)
- **Potential**: Deviance from final scores (how far to end state)

Dynamics analyzes extremal trajectories (highest/lowest inverse perplexity per arm) and outputs to:
```
out/<method>/<gen_name>/<scoring_name>/
    dynamics.json                    # Raw data
    viz/dynamics/
        all/                         # Individual trajectory plots
            traj_0_trunk.png
            traj_1_branch_1.png
        dynamics_branch_1.png        # Aggregate per arm (3-column layout)
        dynamics_branch_2.png
```

### Individual Steps

```bash
# Step 1: Generate trajectories (default: simple-sampling)
uv run python scripts/generate_trajectories.py gen.json --samples-per-arm 10

# Or use a specific method:
uv run python scripts/generate_trajectories.py gen.json --method forking-paths

# Step 2: Score trajectories
uv run python scripts/score_trajectories.py scoring.json out/simple-sampling/gen_example.json

# Step 3: Estimate normativity
uv run python scripts/estimate_normativity.py out/simple-sampling/score_example_example.json

# Step 4: Generate visualizations
# gen and scoring paths are auto-inferred from the estimation path
uv run python scripts/visualize_estimation.py out/simple-sampling/est_example_example.json

# Supply paths explicitly if auto-inference fails (e.g. files were moved)
uv run python scripts/visualize_estimation.py out/simple-sampling/est_example_example.json \
    --scoring out/simple-sampling/score_example_example.json \
    --generation out/simple-sampling/gen_example.json
```

---

## Understanding Console Output

The pipeline prints progress and results to the console.

### Stage 1: Generation

```
══════════════════════════════════════════════════════════════════
  STAGE [1/3] GENERATE (simple-sampling)
══════════════════════════════════════════════════════════════════

  Loading model: Qwen/Qwen3-0.6B
  Device: mps

  Generating trajectories...
  Branch: trunk (10 samples)
  Branch: branch_1 (10 samples)
  Branch: branch_2 (10 samples)

────────────────────────────────────────────────────────────────
  GENERATION SUMMARY
────────────────────────────────────────────────────────────────

  Settings:
    Model: Qwen/Qwen3-0.6B
    Method: sampling
    Temperature: 1.0
    Max tokens: 128

  trunk (10 trajectories, 80% finished):
    [0] Once upon a time, there was a little boy who...
    [1] Once upon a time, there was a magic forest...

  branch_1 (10 trajectories, 90% finished):
    [10] Once upon a time, there was a boy named Jack...
```

**Key things to check:**
- **Device**: Confirms GPU/MPS/CPU usage
- **Finished %**: Trajectories that hit EOS (vs. max_tokens cutoff)
- **Sample text**: Verify generations look reasonable

### Stage 2: Scoring

```
══════════════════════════════════════════════════════════════════
  STAGE [2/3] SCORE
══════════════════════════════════════════════════════════════════

  Loading judge: Qwen/Qwen3-4B-Instruct-2507

  Scoring 30 trajectories...
  [c1] Does this story talk about a person?
  [c2] Does this story talk about an animal?

────────────────────────────────────────────────────────────────
  SCORING SUMMARY
────────────────────────────────────────────────────────────────

  CATEGORICAL JUDGMENTS (% answering YES) - BY GROUP

  === TRUNK (10 trajectories) ===
    [c1] Does this story talk about a person?: 60.0%
    [c2] Does this story talk about an animal?: 40.0%

  === BRANCH_1 (10 trajectories) ===
    [c1] Does this story talk about a person?: 100.0%
    [c2] Does this story talk about an animal?: 10.0%
```

**Key things to check:**
- **Per-structure rates**: Shows % yes for each question per branch
- **Branch differences**: Compare rates across branches to see divergence

### Stage 3: Estimation

```
══════════════════════════════════════════════════════════════════
  STAGE [3/3] ESTIMATE
══════════════════════════════════════════════════════════════════

────────────────────────────────────────────────────────────────
  STRUCTURES
────────────────────────────────────────────────────────────────
  c1: Does this story talk about a person?
  c2: Does this story talk about an animal?

────────────────────────────────────────────────────────────────
  COMPLIANCE RATES BY BRANCH
────────────────────────────────────────────────────────────────
  Branch          N      c1      c2
  ─────────────────────────────────
  trunk          10   60.0%   40.0%
  branch_1       10  100.0%   10.0%
  branch_2       10   30.0%   90.0%

────────────────────────────────────────────────────────────────
  CORES BY BRANCH
────────────────────────────────────────────────────────────────

  [0] trunk (10 trajectories)

    weighting                 c1      c2    E[∂]    Var[∂]
    ────────────────────────────────────────────────────────
    prob-weighted          0.600   0.400   0.2500   0.0625
    inv-ppl-weighted       0.550   0.450   0.2200   0.0484
```

**Key things to check:**
- **Core values**: Expected compliance for each structure (0-1)
- **E[∂] (expected deviance)**: Low = homogenized, high = diverse
- **Var[∂]**: How consistent are the deviations

### Final Summary

After all stages complete, a summary shows the experiment setup and results:

```
  SETUP
  ────────────────────────────────────────────────────────────────────────────
  Prompt:   Write a story in LESS than 12 words about EITHER a boy OR a cat...
  Trunk:    "Once upon a time, there was a"
  Branches: " boy", " cat"

  Models:
    gen:   Qwen/Qwen3-0.6B
    judge: Qwen/Qwen3-4B-Instruct-2507
    embed: all-MiniLM-L6-v2

  Structures:
    c1: [group of 2]
        • Does this story talk about a person?
        • Does this story talk about a boy?
    c2: [group of 2]
        • Does this story talk about an animal?
        • Does this story talk about a cat?
    c3: Does this story talk about someone happy?
    g1: [group of 2]
        • How masculine is the protagonist?
        • How non-feminine is the protagonist?
    g2: How sad is the protagonist?
    s1: sim(gay, queer)
    s2: sim("feline")


  RESULTS
  ────────────────────────────────────────────────────────────────────────────
                      c1     c2     c3     g1     g2     s1     s2
    ─────────────────────────────────────────────────────────────────
  TRUNK (30 trajectories)
    prob    E[∂]=0.2601
       core          0.98   0.98   0.08   0.62   0.00   0.61   0.73
       E[θ]         -0.25  -0.24   0.42  -0.09   0.00  -0.04  -0.06
    inv_ppl E[∂]=0.7607
       core          0.81   0.79   0.51   0.57   0.00   0.58   0.68
       E[θ]         -0.08  -0.05   0.01   0.04   0.00  -0.01   0.01

  BRANCH_1 (10 trajectories)
    prob    E[∂]=0.1373
       core          1.00   0.95   0.03   0.73   0.00   0.59   0.72
       E[θ]          0.00  -0.35   0.57  -0.08   0.00  -0.03  -0.08
    inv_ppl E[∂]=0.6921
       core          1.00   0.69   0.56   0.68   0.00   0.56   0.65
       E[θ]          0.00  -0.09   0.04  -0.03   0.00   0.00  -0.01

  BRANCH_2 (10 trajectories)
    prob    E[∂]=0.5166
       core          0.27   1.00   0.87   0.14   0.00   0.63   0.75
       E[θ]          0.08   0.00  -0.47   0.26   0.00  -0.05  -0.02
    inv_ppl E[∂]=0.7102
       core          0.42   1.00   0.43   0.41   0.00   0.58   0.73
       E[θ]         -0.07   0.00   0.01  -0.01   0.00   0.00  -0.01
```

**Reading the results table:**
- **Column headers** (c1, c2, g1, s1...): Structure labels matching the SETUP section
- **prob**: Probability-weighted (by continuation likelihood)
- **invp**: Inverse-perplexity weighted (favors confident outputs)
- **E[∂]**: Expected deviance from core (0=homogeneous, 1=diverse)
- **core**: Expected compliance vector (what the model treats as "normal")
- **E[θ]**: Expected orientation relative to trunk core (= core_branch - core_trunk; 0 for trunk)

**Example interpretation:**
- BRANCH_2 (cat stories) has `c1=0.27` meaning only 27% mention a person
- BRANCH_1 (boy stories) has `c1=1.00` meaning 100% mention a person
- TRUNK shows the blended distribution before branching

---

## Understanding Output Files

All outputs go to `out/<method>/`.

### Generation Output: `out/<method>/gen_<config>.json`

```json
{
  "config": {
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "...",
    "trunk": "...",
    "branches": ["trunk", " boy", " cat"],
    "temperature": 1.0,
    "max_new_tokens": 128
  },
  "method": "simple-sampling",
  "num_trajectories": 30,
  "tree": {
    "trajs": [
      {
        "idx": 0,
        "token_ids": [1, 2, 3, ...],
        "logprobs": [-0.5, -1.2, ...],
        "continuation_text": "boy named Jack who lived...",
        "group_idx": [1]
      }
    ],
    "trunk_length": 15,
    "prompt_length": 10
  },
  "eos_token": "<|im_end|>"
}
```

**Key fields:**
- `trajs[].continuation_text`: The generated text
- `trajs[].group_idx`: Which branch (0=trunk, 1=first branch, etc.)
- `trajs[].logprobs`: Log-probability of each token

### Scoring Output: `out/<method>/score_<gen>_<scoring>.json`

```json
{
  "generation_file": "out/simple-sampling/gen_example.json",
  "judge_model": "Qwen/Qwen3-4B-Instruct-2507",
  "categorical_judgements": ["Does this...?", ["Q1?", "Q2?"]],
  "branches": ["trunk", "branch_1", "branch_2"],
  "results": [
    {
      "traj_idx": 0,
      "branch": "branch_1",
      "branch_idx": 1,
      "scores": [1, 1, 0],
      "graded_scores": [0.7],
      "similarity_scores": [0.5],
      "conditional_logprobs": {"trunk": -12.5, "branch_1": -8.2},
      "n_continuation_tokens": 45
    }
  ]
}
```

**Key fields:**
- `results[].scores`: Binary judgments (0/1/null)
- `results[].graded_scores`: Continuous judgments (0.0-1.0)
- `results[].conditional_logprobs`: Log p(continuation | each prefix)

### Estimation Output: `out/<method>/est_<gen>_<scoring>.json`

```json
{
  "groups": [
    {
      "group_idx": 0,
      "name": "trunk",
      "core": [0.6, 0.4],
      "core_inv_ppl": [0.55, 0.45],
      "deviance_avg": 0.25,
      "deviance_var": 0.06,
      "trajectories": [
        {"traj_idx": 0, "orientation": [0.4, -0.4], "deviance": 0.57}
      ],
      "core_variants": [
        {"name": "standard", "q": 1.0, "r": 1.0, "core": [0.6, 0.4], "deviance_avg": 0.25},
        {"name": "mode", "q": 1.0, "r": "inf", "core": [1.0, 0.0], "deviance_avg": 0.45}
      ]
    }
  ],
  "structure_info": [
    {"label": "c1", "description": "Does this...?", "is_grouped": false}
  ],
  "branch_rates": [
    {"branch": "trunk", "trajectory_count": 10, "structure_rates": {"c1": 0.6}}
  ]
}
```

**Key fields:**
- `groups[].core`: Probability-weighted expected compliance
- `groups[].deviance_avg`: E[∂] - expected deviance from core
- `groups[].core_variants`: Different (q,r) parameterizations
- `branch_rates`: Per-structure compliance rates by branch

### Summary Output: `out/summary_est_<...>.json`

Human-readable summary with labeled structures and rates.

---

## Model Selection

### Generation Models

**Recommendation: Use local models for generation.**

| Model | Size | Notes |
|-------|------|-------|
| `Qwen/Qwen3-0.6B` | 0.6B | Fast, good for experimentation |
| `Qwen/Qwen3-1.7B` | 1.7B | Better quality, still fast |
| `Qwen/Qwen3-4B` | 4B | Good balance of speed/quality |
| `meta-llama/Llama-3.2-1B` | 1B | Alternative family |

### Base vs. Instruct Models

**Base models** (e.g., `Qwen/Qwen3-0.6B`):
- Continue text naturally without instruction-following
- May show more raw distributional patterns
- Less constrained, more diverse outputs
- Good for studying inherent model biases

**Instruct models** (e.g., `Qwen/Qwen3-0.6B-Instruct`):
- Follow instructions more reliably
- More coherent, task-focused outputs
- May have alignment-induced biases
- Better for controlled experiments

**Recommendation for generation:**
- Start with base models to see raw patterns
- Use instruct models when you need the model to follow a specific format
- The `-Base` suffix indicates base models (e.g., `Qwen/Qwen3-0.6B-Base`)

### Judge Models

**Recommendation: Use instruction-tuned models for judging.**

| Model | Size | Notes |
|-------|------|-------|
| `Qwen/Qwen3-4B-Instruct-2507` | 4B | Good balance |
| `Qwen/Qwen3-8B-Instruct` | 8B | More reliable |
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | Alternative |

**Judge model should be:**
- Instruction-tuned (to follow 0/1 format)
- Large enough for reliable judgments
- Different from generation model (to avoid self-bias)

### API Models

For API models, set environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

Then use:
```json
{
  "model": "gpt-4o-mini",
  "prompt": "..."
}
```

**Note:** API models are slower and cost money. Use local models for iteration.

---

## Tips and Best Practices

### Start Small

```bash
# Test with 2 samples first
uv run python scripts/run_full_experiment.py gen.json scoring.json --samples-per-arm 2

# Then scale up
uv run python scripts/run_full_experiment.py gen.json scoring.json --samples-per-arm 50
```

### Verify Generations

Before running scoring, check that generations look right:
1. Open `out/<method>/gen_*.json`
2. Check `continuation_text` fields
3. Verify branches diverge as expected

### Check Judge Reliability

Judge models can be unreliable. To verify:
1. Look at `raw_judgments` in scoring output
2. Check for parsing failures (scores = `null`)
3. Try different judge models if results seem wrong

### Iterate on Questions

Question phrasing matters. Compare:
- "Does this mention women?" (ambiguous)
- "Does this text explicitly mention the word 'woman' or 'women'?" (specific)

### Use Grouped Structures

When multiple questions capture the same concept:
```json
"categorical_judgements": [
  ["Does this mention a person?", "Does this mention a human?", "Does this mention someone?"]
]
```

This averages them into one structure, reducing noise.

### Compare Weighting Schemes

- **prob-weighted core**: What the model actually outputs (frequency)
- **inv-ppl-weighted core**: What the model is confident about (quality)

Large differences suggest the model produces frequent low-confidence outputs.

### Reproducibility

Set a seed for reproducible results:
```json
{
  "model": "...",
  "prompt": "...",
  "seed": 42
}
```

### Save Experiment Configs

Keep generation and scoring configs together:
```
trials/
├── generation/
│   ├── gender_bias_v1.json
│   └── gender_bias_v2.json
└── scoring/
    ├── gender_structures.json
    └── identity_structures.json
```
