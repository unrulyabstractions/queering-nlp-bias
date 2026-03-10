# Scripts

Pipeline scripts for trajectory generation, scoring, and estimation.

See the methodology docs for conceptual background:
- [GENERATION.md](../GENERATION.md)
- [SCORING.md](../SCORING.md)
- [ESTIMATION.md](../ESTIMATION.md)

## Full Pipeline

```bash
# Run all three stages with simple sampling (default)
uv run python scripts/run_full_experiment.py trials/generation/example.json trials/scoring/example.json

# With forking paths method
uv run python scripts/run_full_experiment.py --forking-paths trials/generation/example.json trials/scoring/example.json

# With entropy-seeking method
uv run python scripts/run_full_experiment.py --seeking-entropy trials/generation/example.json trials/scoring/example.json

# Run all methods and compare
uv run python scripts/run_full_experiment.py --all trials/generation/example.json trials/scoring/example.json
```

## Individual Scripts

### Generation

All generation scripts output to `out/gen_<method>_<config>.json`.

#### Simple Sampling

```bash
uv run python scripts/generate_by_simple_sampling.py trials/generation/example.json
uv run python scripts/generate_by_simple_sampling.py trials/generation/example.json --samples-per-branch 10
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--samples-per-branch` | 2 | Trajectories per branch |

#### Forking Paths

```bash
uv run python scripts/generate_by_forking_paths.py trials/generation/example.json
uv run python scripts/generate_by_forking_paths.py trials/generation/example.json \
    --max-alternates-per-position 5 \
    --min-prob-for-alternate 0.01 \
    --min-entropy-to-fork 1.0 \
    --samples-per-fork 2
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-alternates-per-position` | 3 | Max alternate tokens per position |
| `--min-prob-for-alternate` | 0.05 | Minimum probability for alternate |
| `--min-entropy-to-fork` | 0.0 | Minimum entropy to consider forking |
| `--samples-per-fork` | 1 | Continuations per fork point |

#### Seeking Entropy

```bash
uv run python scripts/generate_by_seeking_entropy.py trials/generation/example.json
uv run python scripts/generate_by_seeking_entropy.py trials/generation/example.json \
    --samples-per-expansion 3 \
    --num-expansion-rounds 4
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--samples-per-expansion` | 2 | Trajectories per expansion |
| `--num-expansion-rounds` | 3 | Number of expansion rounds |

### Scoring

```bash
uv run python scripts/score_trajectories.py trials/scoring/example.json out/gen_sampling_example.json
```

Output: `out/score_<method>_<gen>_<scoring>.json`

### Estimation

```bash
uv run python scripts/estimate_normativity.py out/score_sampling_example_example.json
```

Output: `out/est_<method>_<gen>_<scoring>.json` and `out/summary_est_<...>.json`

## run_full_experiment.py Parameters

All method-specific parameters can be passed to the full pipeline:

```bash
# Simple sampling with 10 samples
uv run python scripts/run_full_experiment.py gen.json scoring.json --samples-per-branch 10

# Forking paths with custom parameters
uv run python scripts/run_full_experiment.py --forking-paths gen.json scoring.json \
    --max-alternates-per-position 5 \
    --min-prob-for-alternate 0.01

# Entropy seeking with custom parameters
uv run python scripts/run_full_experiment.py --seeking-entropy gen.json scoring.json \
    --samples-per-expansion 3 \
    --num-expansion-rounds 5
```

## Directory Structure

```
scripts/
├── run_full_experiment.py        # Full pipeline orchestrator
├── generate_by_simple_sampling.py
├── generate_by_forking_paths.py
├── generate_by_seeking_entropy.py
├── score_trajectories.py
├── estimate_normativity.py
└── schemas/                      # Config and output schemas
    ├── generation.py             # GenerationConfig, GenerationOutput
    ├── scoring.py                # ScoringConfig, JudgmentOutput
    ├── estimation.py             # EstimationOutput, GroupEstimate
    ├── default_config.py         # Default parameter values
    ├── script_utils.py           # Logging utilities
    └── log_utils.py              # Output formatting
```

## Schema Classes

### GenerationConfig

```python
from schemas import GenerationConfig

config = GenerationConfig.load("trials/generation/example.json")
print(config.model)      # "Qwen/Qwen3-0.6B"
print(config.branches)   # [" boy", " cat"]
```

### ScoringConfig

```python
from schemas import ScoringConfig

config = ScoringConfig.load("trials/scoring/example.json")
print(config.categorical_judgements)  # [["Does this...", ...], ...]
```

### EstimationOutput

```python
from schemas import EstimationOutput

# Access groups and their cores
for group in output.groups:
    print(f"{group.name}: {group.core}")
    print(f"  E[deviance]: {group.deviance_avg}")
```
