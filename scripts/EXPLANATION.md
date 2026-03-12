# Script Documentation

> **Note**: This documentation was AI-generated and may contain errors. If something seems off, check the code or open an issue.


Detailed documentation for each script, their arguments, and how they work together.

## Pipeline Overview

The experiment pipeline has three stages:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  GENERATE   │ ──▶ │    SCORE    │ ──▶ │  ESTIMATE   │
│             │     │             │     │             │
│ Trajectories│     │ Compliance  │     │ Normativity │
│ from model  │     │ scores      │     │ metrics     │
└─────────────┘     └─────────────┘     └─────────────┘
```

**Input**: Generation config (model, prompt, branches) + Scoring config (judge, structures)
**Output**: Estimation results with cores, orientations, and deviances per branch

---

## run_full_experiment.py

**Purpose**: Orchestrate the complete three-stage pipeline.

### Usage

```bash
# Default: simple-sampling method
uv run python scripts/run_full_experiment.py \
    trials/generation/<gen_config>.json \
    trials/scoring/<scoring_config>.json

# Specific method
uv run python scripts/run_full_experiment.py --method forking-paths \
    trials/generation/<gen_config>.json \
    trials/scoring/<scoring_config>.json

# All methods with comparison
uv run python scripts/run_full_experiment.py --all \
    trials/generation/<gen_config>.json \
    trials/scoring/<scoring_config>.json

# Compute drift and horizon dynamics
uv run python scripts/run_full_experiment.py --dynamics \
    trials/generation/<gen_config>.json \
    trials/scoring/<scoring_config>.json
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `generation_config` | positional | Path to generation config JSON |
| `scoring_config` | positional | Path to scoring config JSON |
| `--method` | optional | Generation method: `simple-sampling`, `forking-paths`, or `seeking-entropy` (default: `simple-sampling`) |
| `--all` | flag | Run all methods and compare results |
| `--dynamics` | flag | Compute drift and horizon dynamics for trajectories (requires scoring model) |

### Method-Specific Parameters

All method parameters can be passed directly:

```bash
# Simple sampling
uv run python scripts/run_full_experiment.py gen.json scoring.json \
    --samples-per-arm 20

# Forking paths
uv run python scripts/run_full_experiment.py --method forking-paths gen.json scoring.json \
    --max-alternates-per-position 5 \
    --min-prob-for-alternate 0.1 \
    --min-entropy-to-fork 1.5 \
    --samples-per-fork 2

# Entropy seeking
uv run python scripts/run_full_experiment.py --method seeking-entropy gen.json scoring.json \
    --samples-per-expansion 3 \
    --num-expansion-rounds 5
```

### How It Works

1. **Parse arguments**: Determine method(s) and collect parameter overrides
2. **For each method**:
   - Load generation config and apply CLI overrides
   - Set random seed for reproducibility
   - **Step 1 (Generate)**: Run generation pipeline, save to `out/<method>/gen_<config>.json`
   - **Step 2 (Score)**: Load generation output, run scoring pipeline, save to `out/<method>/score_*.json`
   - **Step 3 (Estimate)**: Load scoring output, compute normativity metrics, save to `out/<method>/est_*.json`
3. **If `--all`**: Display comparison table across methods

### Output Files

```
out/
├── simple-sampling/
│   ├── gen_<config>.json
│   ├── score_<gen>_<scoring>.json
│   ├── est_<gen>_<scoring>.json
│   ├── summary_gen_*.txt
│   ├── summary_score_*.txt
│   └── summary_est_*.txt
├── forking-paths/
│   └── ...
└── seeking-entropy/
    └── ...
```

---

## generate_trajectories.py

**Purpose**: Unified script for generating trajectories using any registered method.

### Available Methods

1. **simple-sampling** (default): Temperature sampling with N samples per arm
2. **forking-paths**: Systematically explore one-step deviations from the greedy path
3. **seeking-entropy**: Expand the trajectory tree at high-uncertainty positions

### Usage

```bash
# Simple sampling (default)
uv run python scripts/generate_trajectories.py trials/generation/<config>.json
uv run python scripts/generate_trajectories.py trials/generation/<config>.json --samples-per-arm 20

# Forking paths
uv run python scripts/generate_trajectories.py trials/generation/<config>.json --method forking-paths
uv run python scripts/generate_trajectories.py trials/generation/<config>.json --method forking-paths \
    --max-alternates-per-position 5 \
    --min-prob-for-alternate 0.1 \
    --min-entropy-to-fork 1.5 \
    --samples-per-fork 2

# Entropy seeking
uv run python scripts/generate_trajectories.py trials/generation/<config>.json --method seeking-entropy
uv run python scripts/generate_trajectories.py trials/generation/<config>.json --method seeking-entropy \
    --samples-per-expansion 3 \
    --num-expansion-rounds 5
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `config` | positional | required | Path to generation config JSON |
| `--method` | choice | simple-sampling | Generation method: `simple-sampling`, `forking-paths`, or `seeking-entropy` |

### Method-Specific Arguments

#### Simple Sampling

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--samples-per-arm` | int | 10 | Number of trajectories per arm |

#### Forking Paths

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--max-alternates-per-position` | int | 5 | Max alternate tokens per position |
| `--min-prob-for-alternate` | float | 0.2 | Minimum probability for alternate token |
| `--min-entropy-to-fork` | float | 1.75 | Minimum entropy (nats) to consider forking |
| `--samples-per-fork` | int | 3 | Continuations to sample per fork point |

#### Entropy Seeking

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--samples-per-expansion` | int | 2 | Trajectories per expansion round |
| `--num-expansion-rounds` | int | 3 | Number of expansion rounds |

### How Each Method Works

#### Simple Sampling

For each arm (trunk + branches):
1. Construct prompt with arm-specific prefill
2. Sample N trajectories using temperature sampling
3. Record token IDs, log-probabilities, and decoded text

```
For each arm in [trunk, branch_1, branch_2, ...]:
    prefill = trunk (for trunk) or trunk + branch_text (for branches)

    for i in 1..samples_per_arm:
        trajectory = model.generate(prompt + prefill, temperature=T)
        save(trajectory.token_ids, trajectory.logprobs)
```

**Use case**: Simple, unbiased samples from the distribution.

#### Forking Paths

For each arm:
1. Generate the greedy path (always pick highest-probability token)
2. At each position where entropy exceeds threshold:
   - Identify alternate tokens meeting probability threshold
   - For each alternate, spawn continuations from that deviation
3. Return all trajectories (greedy path + fork continuations)

```
greedy_path = generate_greedy(prompt + prefill)

for position in greedy_path:
    if entropy[position] >= min_entropy:
        alternates = get_top_k_tokens(position, k=max_alternates)
        alternates = [t for t in alternates if prob[t] >= min_prob]

        for alt_token in alternates:
            prefix = greedy_path[:position] + [alt_token]
            for i in 1..samples_per_fork:
                continuation = model.generate(prefix, temperature=T)
                save(continuation)
```

**Use case**: Reveals what the model "almost" said. Good for understanding local sensitivity.

#### Entropy Seeking

For each arm:
1. Initialize tree with N sampled trajectories
2. Compute entropy at all positions via forward pass
3. For K rounds:
   - Find (path, position) with highest unused entropy
   - Sample N new continuations from that fork point
   - Mark position as used, compute entropy for new paths
4. Return all trajectories

```
# Initialize
for i in 1..samples_per_expansion:
    traj = model.generate(prompt + prefill, temperature=T)
    tree.add(traj, entropies=compute_entropies(traj))

# Expand
for round in 1..num_expansion_rounds:
    (best_path, best_pos) = find_highest_unused_entropy(tree)

    prefix = best_path[:best_pos]
    for i in 1..samples_per_expansion:
        continuation = model.generate(prefix, temperature=T)
        tree.add(continuation, parent=best_path, fork_pos=best_pos)

    mark_used(best_path, best_pos)
```

**Use case**: Focuses exploration on uncertain decision points.

### Output

- `out/<method>/gen_<config>.json`: Full trajectory data
- `out/<method>/summary_gen_<config>.txt`: Human-readable summary

---

## score_trajectories.py

**Purpose**: Score generated trajectories against structures.

### Usage

```bash
uv run python scripts/score_trajectories.py \
    trials/scoring/<scoring_config>.json \
    out/<method>/gen_<config>.json
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `scoring_config` | positional | Path to scoring config JSON |
| `generation_output` | positional | Path to generation output JSON |

### How It Works

1. **Load inputs**: Scoring config and generation output
2. **Initialize scorers**: Load judge model, embedding model as needed
3. **For each trajectory**:
   - Select text portion based on `string_selection` setting
   - **Categorical judgments**: Ask judge 0/1 questions
   - **Graded judgments**: Ask judge 0.0-1.0 scale questions
   - **Similarity scoring**: Compute embedding similarity to reference words
   - **Count occurrences**: Count word/phrase occurrences
4. **Compute conditional log-probs**: p(continuation | each prefix)
5. **Save results**

### Scoring Types

| Type | Config Key | Output |
|------|------------|--------|
| Binary yes/no | `categorical_judgements` | 0 or 1 per question |
| Graded scale | `graded_judgements` | 0.0-1.0 per question |
| Embedding similarity | `similarity_scoring` | Cosine similarity |
| Word counting | `count_occurrences` | Normalized occurrence count |

### Output

- `out/<method>/score_<gen>_<scoring>.json`: Full scoring data
- `out/<method>/summary_score_*.txt`: Human-readable summary

---

## visualize_estimation.py

**Purpose**: Generate visualization plots from an existing estimation output JSON.

### Usage

```bash
uv run python scripts/visualize_estimation.py out/<method>/est_<name>.json
uv run python scripts/visualize_estimation.py out/<method>/est_<name>.json --output-dir custom/path
# Explicit paths if auto-inference fails:
uv run python scripts/visualize_estimation.py out/<method>/est_<name>.json \
    --scoring out/<method>/score_<name>.json \
    --generation out/<method>/gen_<name>.json
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `estimation` | positional | Path to estimation output JSON (`out/<method>/est_*.json`) |
| `--scoring` | optional | Path to scoring JSON (`out/<method>/score_*.json`). Auto-inferred from estimation path. |
| `--generation` | optional | Path to generation JSON (`out/<method>/gen_*.json`). Auto-inferred from `generation_file` field in scoring output. |
| `--output-dir` | optional | Output directory for plots (default: `out/<method>/viz`) |

### How It Works

1. Infers the generation method name from the parent folder (e.g. `out/simple-sampling/est_...` → `simple-sampling`)
2. Auto-infers the scoring path (`score_<name>.json` in same folder) and generation path (from the `generation_file` field in the score JSON)
3. Calls `visualize_result()` to generate all plots
4. Saves plots to `out/<method>/viz/`

### Which plots require which files?

| Plot | Requires |
|------|----------|
| Core, deviance, orientation, dynamics subplots | estimation JSON only |
| `dynamics.png` | scoring JSON (`--scoring`) |
| `tree_word.png`, `tree_phrase.png` | generation JSON (`--generation`) |

Both are auto-inferred from the estimation filename; warnings are printed if they can't be found.

### Output

See [src/viz/README.md](../src/viz/README.md) for the full list of generated plot files.

---

## estimate_normativity.py

**Purpose**: Compute normativity metrics from scoring results.

### Usage

```bash
uv run python scripts/estimate_normativity.py out/<method>/score_<name>.json
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `scores` | positional | Path to scoring output JSON |

### How It Works

1. **Load scoring data**: Trajectories with compliance vectors
2. **Group by arm**: Separate trunk vs branch trajectories
3. **For each group**:
   - Compute **core** (expected compliance) with probability weighting
   - Compute **orientation** for each trajectory: compliance - core
   - Compute **deviance** for each trajectory: ||orientation||
   - Compute **E[deviance]** and **Var[deviance]**
4. **Compute core variants**: Different (q,r) parameterizations
5. **Save results**

### Key Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Core | E[compliance] | What the model treats as "normal" |
| Orientation | compliance - core | How a trajectory deviates from normal |
| Deviance | \|\|orientation\|\| | Magnitude of deviation |
| E[deviance] | E[\|\|orientation\|\|] | Average non-normativity |

### Output

- `out/<method>/est_<gen>_<scoring>.json`: Full estimation data
- `out/<method>/summary_est_*.txt`: Human-readable summary

---

## Configuration Files

### Generation Config

Located in `trials/generation/`:

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "prompt": "Write a story about a protagonist.",
  "trunk": "The protagonist was a",
  "branches": [" man", " woman"],
  "temperature": 1.0,
  "max_new_tokens": 128,
  "seed": 42
}
```

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `model` | yes | - | HuggingFace model ID |
| `prompt` | yes | - | Instruction/system prompt |
| `trunk` | no | `""` | Shared prefix for all trajectories |
| `branches` | no | `[]` | Branch-specific prefixes |
| `temperature` | no | 1.0 | Sampling temperature |
| `max_new_tokens` | no | 128 | Max tokens to generate |
| `seed` | no | null | Random seed |

### Scoring Config

Located in `trials/scoring/`:

```json
{
  "model": "Qwen/Qwen3.5-0.8B",
  "categorical_judgements": [
    ["Does this mention a person?", "Does this mention a human?"],
    "Does this mention an animal?"
  ],
  "graded_judgements": [
    "How masculine is the protagonist (0=not at all, 1=extremely)?"
  ],
  "similarity_scoring": [
    ["masculine", "manly"],
    "feminine"
  ],
  "count_occurrences": [
    ["boy", "man"],
    "cat"
  ],
  "string_selection": "WholeContinuation"
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `model` | for judgments | Judge model (instruction-tuned) |
| `categorical_judgements` | no | Binary 0/1 questions |
| `graded_judgements` | no | Scale 0.0-1.0 questions |
| `similarity_scoring` | no | Reference words for embedding similarity |
| `count_occurrences` | no | Words/phrases to count |
| `embedding_model` | no | Model for similarity (default: `all-MiniLM-L6-v2`) |
| `string_selection` | no | Which text to score (default: `WholeContinuation`) |

### String Selection Options

| Value | Scores |
|-------|--------|
| `WholeTrajectory` | Prompt + generated text |
| `WholeContinuation` | Generated text only |
| `AfterTrunk` | Generated text minus trunk |
| `AfterBranch` | Generated text minus trunk and branch |

---

## Default Values

All defaults are defined in `src/common/default_config.py`:

```python
# Generation
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 128

# Simple sampling
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
STRING_SELECTION = "WholeContinuation"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
```

### Priority Order

```
CLI arguments > JSON config > default_config.py
```

---

## How Scripts Work Together

### Data Flow

```
Generation Config ─────▶ generate_trajectories.py ──▶ Generation Output
                                                           │
                                                           ▼
Scoring Config    ─────▶ score_trajectories.py ──────────▶ Scoring Output
                                                           │
                                                           ▼
                         estimate_normativity.py ────────▶ Estimation Output
```

### File Dependencies

```
trials/generation/example.json
trials/scoring/example.json
        │
        ▼
out/simple-sampling/gen_example.json
        │
        ▼
out/simple-sampling/score_example_example.json
        │
        ▼
out/simple-sampling/est_example_example.json
```

### Running Individual Steps

```bash
# Step 1: Generate (default: simple-sampling)
uv run python scripts/generate_trajectories.py trials/generation/example.json

# Or use a specific method:
uv run python scripts/generate_trajectories.py trials/generation/example.json --method forking-paths

# Step 2: Score (requires generation output)
uv run python scripts/score_trajectories.py \
    trials/scoring/example.json \
    out/simple-sampling/gen_example.json

# Step 3: Estimate (requires scoring output)
uv run python scripts/estimate_normativity.py \
    out/simple-sampling/score_example_example.json
```

### Using run_full_experiment.py

Equivalent to running all three steps:

```bash
uv run python scripts/run_full_experiment.py \
    trials/generation/example.json \
    trials/scoring/example.json
```
