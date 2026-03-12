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
```

## Available Scripts

| Script | Purpose |
|--------|---------|
| `run_full_experiment.py` | Full pipeline orchestrator (generate -> score -> estimate) |
| `generate_by_simple_sampling.py` | Temperature sampling generation |
| `generate_by_forking_paths.py` | Systematic one-step deviation exploration |
| `generate_by_seeking_entropy.py` | Entropy-guided tree expansion |
| `score_trajectories.py` | Score trajectories against structures |
| `estimate_normativity.py` | Compute normativity metrics from scores |
| `visualize_estimation.py` | Generate `out/viz/` plots from estimation output JSON |

## Generation Methods

### Simple Sampling (default)

```bash
uv run python scripts/generate_by_simple_sampling.py trials/generation/example.json
uv run python scripts/generate_by_simple_sampling.py trials/generation/example.json --samples-per-arm 20
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--samples-per-arm` | 10 | Trajectories per arm (trunk + each branch) |

### Forking Paths

```bash
uv run python scripts/generate_by_forking_paths.py trials/generation/example.json
uv run python scripts/generate_by_forking_paths.py trials/generation/example.json \
    --max-alternates-per-position 5 \
    --min-prob-for-alternate 0.1 \
    --min-entropy-to-fork 1.5 \
    --samples-per-fork 2
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-alternates-per-position` | 5 | Max alternate tokens per position |
| `--min-prob-for-alternate` | 0.2 | Minimum probability for alternate token |
| `--min-entropy-to-fork` | 1.75 | Minimum entropy (nats) to consider forking |
| `--samples-per-fork` | 3 | Continuations per fork point |

### Entropy Seeking

```bash
uv run python scripts/generate_by_seeking_entropy.py trials/generation/example.json
uv run python scripts/generate_by_seeking_entropy.py trials/generation/example.json \
    --samples-per-expansion 3 \
    --num-expansion-rounds 5
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--samples-per-expansion` | 2 | Trajectories per expansion round |
| `--num-expansion-rounds` | 3 | Number of tree expansion rounds |

## Scoring and Estimation

```bash
# Score trajectories
uv run python scripts/score_trajectories.py \
    trials/scoring/example.json \
    out/gen_simple-sampling_example.json

# Estimate normativity
uv run python scripts/estimate_normativity.py \
    out/score_simple-sampling_example_example.json

# Generate visualizations from estimation output
uv run python scripts/visualize_estimation.py \
    out/est_simple-sampling_example_example.json
```

## Output Files

All outputs are saved to `out/`:

| Pattern | Contents |
|---------|----------|
| `gen_<method>_<config>.json` | Generated trajectories |
| `score_<method>_<gen>_<scoring>.json` | Scoring results |
| `est_<method>_<gen>_<scoring>.json` | Estimation results |
| `summary_*` | Human-readable summaries |

## Directory Structure

```
scripts/
├── run_full_experiment.py        # Full pipeline orchestrator
├── generate_by_simple_sampling.py
├── generate_by_forking_paths.py
├── generate_by_seeking_entropy.py
├── score_trajectories.py
├── estimate_normativity.py
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
