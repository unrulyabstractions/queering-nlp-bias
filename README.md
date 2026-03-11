# Queering NLP Bias

**Bias is low deviance in a dimension where we would expect more diversity** - an overly high concentration around normativity. In this project, we aim to characterize normativity to measure bias. We propose methods that leverage LLM-induced statistics of scoring functions. This repo implements the methods and provides a playground for investigation.

> **Note**: The `.md` documentation files in this repo were LLM-generated. Take them with a grain of salt. If something seems wrong, unclear, or contradicts the code, please ask questions or report bugs.

## Conceptual Foundation

The core insight: An LLM defines a probability distribution over all possible text continuations. **Normativity** is what the model treats as the default path - the expected value of compliance across structures. **Bias** manifests when this distribution concentrates too heavily around certain outcomes, erasing diversity in dimensions that matter.

We operationalize this through:
- **Structures**: Questions about text (e.g., "Does this mention women?") that encode what we care about
- **Compliance**: How well a trajectory satisfies a structure (0 = no, 1 = yes)
- **Core**: The expected compliance - what the model treats as "normal"
- **Deviance**: How far individual outputs stray from the core

## Quick Demo

```bash
# Run the full experiment pipeline
uv run python scripts/run_full_experiment.py trials/generation/example.json trials/scoring/example.json
```

This runs three stages:
1. **Generate**: Sample text continuations from `Qwen/Qwen3-0.6B`
2. **Score**: Judge each trajectory with `Qwen/Qwen3-4B-Instruct-2507`
3. **Estimate**: Compute normativity metrics (core, deviance, orientation)

### Output Files

All outputs go to `out/`:

| File | Description |
|------|-------------|
| `out/gen_sampling_example.json` | Generated trajectories with token tree |
| `out/score_sampling_example_example.json` | Judgment results per trajectory |
| `out/est_sampling_example_example.json` | Full normativity estimates |
| `out/summary_est_sampling_example_example.json` | Human-readable summary |

## Directory Structure

```
queering-nlp-bias/
├── scripts/                    # Pipeline scripts
│   ├── run_full_experiment.py  # Full pipeline orchestrator
│   ├── generate_by_*.py        # Three generation methods
│   ├── score_trajectories.py   # Score trajectories against structures
│   ├── estimate_normativity.py # Compute cores and deviances
│   └── schemas/                # Config schemas (generation, scoring, estimation)
│
├── trials/                     # Experiment configurations
│   ├── generation/             # Generation configs (model, prompt, branches)
│   └── scoring/                # Scoring configs (judgment questions)
│
├── src/                        # Core library
│   ├── common/                 # Data structures, math, analysis
│   │   └── math/entropy_diversity/  # Entropy, diversity, power means
│   ├── inference/              # Model backends (HuggingFace, MLX, APIs)
│   └── viz/                    # Tree visualization
│
├── out/                        # Generated outputs
│
├── GUIDE_TO_EXPERIMENT.md      # Step-by-step experiment guide
├── GENERATION.md               # Generation methodology
├── SCORING.md                  # Scoring methodology
├── ESTIMATION.md               # Estimation methodology
├── MOTIVATION.md               # Theoretical background
└── OPEN_QUESTIONS.md           # Research questions
```

## Pipeline Overview

### 1. Generation

Sample text continuations that branch from a shared prompt:

```
Prompt: "Write a story..."
Trunk: "Once upon a time, there was a"
       ├── Branch A: " boy" → [trajectory 1, trajectory 2, ...]
       └── Branch B: " cat" → [trajectory 3, trajectory 4, ...]
```

Three methods available:
- **Simple Sampling**: Temperature-based sampling per branch
- **Forking Paths**: Explore one-step deviations from greedy path
- **Seeking Entropy**: Expand at high-uncertainty positions

### 2. Scoring

Evaluate each trajectory against structures:

- **Categorical**: Binary yes/no questions → 0 or 1
- **Graded**: Scale questions → 0.0 to 1.0
- **Similarity**: Embedding similarity to reference words → 0.0 to 1.0

Structures can be grouped (averaged together in the core).

### 3. Estimation

Compute normativity metrics:

- **Core** `⟨Λ_n⟩`: Expected compliance vector (the "normal" pattern)
- **Orientation** `θ_n(x)`: How a trajectory differs from the core
- **Deviance** `∂_n(x)`: Magnitude of deviation from normal
- **E[∂]**: Expected deviance (low = homogenized, high = diverse)

Multiple weighting schemes: probability-weighted and inverse-perplexity-weighted.

## Generation Methods

```bash
# Simple sampling (default)
uv run python scripts/run_full_experiment.py trials/generation/example.json trials/scoring/example.json

# Forking paths - explore deviations from greedy
uv run python scripts/run_full_experiment.py --method forking-paths trials/generation/example.json trials/scoring/example.json

# Seeking entropy - expand at uncertain positions
uv run python scripts/run_full_experiment.py --method seeking-entropy trials/generation/example.json trials/scoring/example.json

# Run all methods and compare
uv run python scripts/run_full_experiment.py --all trials/generation/example.json trials/scoring/example.json
```

## Configuration

See [trials/README.md](trials/README.md) for detailed config formats.

### Generation Config

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "prompt": "Write a story about a protagonist...",
  "trunk": "Once upon a time, there was a",
  "branches": [" boy", " cat"]
}
```

### Scoring Config

```json
{
  "model": "Qwen/Qwen3-4B-Instruct-2507",
  "categorical_judgements": [
    ["Does this mention a person?", "Does this mention a boy?"],
    "Does this mention happiness?"
  ],
  "graded_judgements": [
    "How masculine is the protagonist?"
  ],
  "similarity_scoring": [
    ["gay", "queer"]
  ]
}
```

## Further Reading

- [GUIDE_TO_EXPERIMENT.md](GUIDE_TO_EXPERIMENT.md) - Step-by-step guide to running experiments
- [MOTIVATION.md](MOTIVATION.md) - Why diversity matters (from critical theory)
- [GENERATION.md](GENERATION.md) - Generation methodology
- [SCORING.md](SCORING.md) - Scoring methodology
- [ESTIMATION.md](ESTIMATION.md) - Estimation methodology
- [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) - Research questions
- [scripts/README.md](scripts/README.md) - Script usage
- [trials/README.md](trials/README.md) - Config formats
