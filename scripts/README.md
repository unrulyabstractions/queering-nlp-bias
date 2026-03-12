# Scripts

> **Note**: This documentation was AI-generated and may contain errors. If something seems off, check the code or open an issue.


Pipeline scripts for trajectory generation, scoring, and normativity estimation.

For conceptual background, see the methodology docs:
- [GENERATION.md](../GENERATION.md) - trajectory sampling methods
- [SCORING.md](../SCORING.md) - structure compliance scoring
- [ESTIMATION.md](../ESTIMATION.md) - normativity metrics

For detailed script documentation, see [EXPLANATION.md](./EXPLANATION.md).

## Quick Start

```bash
# Run the full three-stage pipeline (generate -> score -> estimate)
uv run python scripts/run_full_experiment.py \
    trials/generation/example.json \
    trials/scoring/example.json

# Use a specific generation method
uv run python scripts/run_full_experiment.py --method forking-paths \
    trials/generation/example.json \
    trials/scoring/example.json

# Run all methods and compare results
uv run python scripts/run_full_experiment.py --all \
    trials/generation/example.json \
    trials/scoring/example.json

# Compute drift and horizon dynamics for trajectories
uv run python scripts/run_full_experiment.py --dynamics \
    trials/generation/example.json \
    trials/scoring/example.json
```

## Available Scripts

| Script | Purpose |
|--------|---------|
| `run_full_experiment.py` | Full pipeline orchestrator (generate -> score -> estimate) |
| `generate_trajectories.py` | Unified trajectory generation (all methods via --method) |
| `score_trajectories.py` | Score trajectories against structures |
| `estimate_normativity.py` | Compute normativity metrics from scores |
| `visualize_estimation.py` | Generate `out/<method>/viz/` plots from estimation output JSON |

## Generation Methods

All generation methods are available through the unified `generate_trajectories.py` script:

```bash
# Simple sampling (default)
uv run python scripts/generate_trajectories.py trials/generation/example.json
uv run python scripts/generate_trajectories.py trials/generation/example.json --samples-per-arm 20

# Forking paths
uv run python scripts/generate_trajectories.py trials/generation/example.json --method forking-paths
uv run python scripts/generate_trajectories.py trials/generation/example.json --method forking-paths \
    --max-alternates-per-position 5 \
    --min-prob-for-alternate 0.1 \
    --min-entropy-to-fork 1.5 \
    --samples-per-fork 2

# Entropy seeking
uv run python scripts/generate_trajectories.py trials/generation/example.json --method seeking-entropy
uv run python scripts/generate_trajectories.py trials/generation/example.json --method seeking-entropy \
    --samples-per-expansion 3 \
    --num-expansion-rounds 5
```

### Method-Specific Parameters

#### Simple Sampling (default)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--samples-per-arm` | 10 | Trajectories per arm (trunk + each branch) |

#### Forking Paths

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-alternates-per-position` | 5 | Max alternate tokens per position |
| `--min-prob-for-alternate` | 0.2 | Minimum probability for alternate token |
| `--min-entropy-to-fork` | 1.75 | Minimum entropy (nats) to consider forking |
| `--samples-per-fork` | 3 | Continuations per fork point |

#### Entropy Seeking

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--samples-per-expansion` | 2 | Trajectories per expansion round |
| `--num-expansion-rounds` | 3 | Number of tree expansion rounds |

## Scoring and Estimation

```bash
# Score trajectories
uv run python scripts/score_trajectories.py \
    trials/scoring/example.json \
    out/simple-sampling/gen_example.json

# Estimate normativity
uv run python scripts/estimate_normativity.py \
    out/simple-sampling/score_example_example.json

# Generate visualizations from estimation output
uv run python scripts/visualize_estimation.py \
    out/simple-sampling/est_example_example.json
```

## Output Files

All outputs are saved to `out/<method>/`:

| Pattern | Contents |
|---------|----------|
| `<method>/gen_<config>.json` | Generated trajectories |
| `<method>/score_<gen>_<scoring>.json` | Scoring results |
| `<method>/est_<gen>_<scoring>.json` | Estimation results |
| `<method>/summary_*` | Human-readable summaries |

## Directory Structure

```
scripts/
├── run_full_experiment.py        # Full pipeline orchestrator
├── generate_trajectories.py      # Unified generation (--method flag)
├── score_trajectories.py
├── estimate_normativity.py
├── visualize_estimation.py
├── EXPLANATION.md                # Detailed script documentation
└── schemas/                      # Script utilities
    └── script_utils.py           # Argument parsing, logging
```

## Schema Classes

Schema classes are located in `src/`:

```python
from src.generation import GenerationConfig, GenerationOutput
from src.scoring import ScoringConfig, ScoringOutput
from src.estimation import ScoringData, EstimationOutput
```
