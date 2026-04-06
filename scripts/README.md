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

# Compute drift and potential dynamics for trajectories
uv run python scripts/run_full_experiment.py --dynamics \
    trials/generation/example.json \
    trials/scoring/example.json

# Profile performance (shows timing breakdown)
uv run python scripts/run_full_experiment.py --profile \
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
| `score_estimate_visualize.py` | Run score → estimate → visualize on an existing `generation.json` |
| `import_csv_generations.py` | Convert a two-column CSV into `generation.json` format (no model needed) |
| `import_more_of_the_same.py` | Merge specified/associated CSVs from the More-of-the-Same dataset into pipeline-ready CSV |

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

# Score + estimate + visualize from an existing generation.json in one command
uv run python scripts/score_estimate_visualize.py \
    out/my-method/generation.json \
    trials/scoring/example.json
```

## Data Import (CSV → Pipeline)

Use these scripts to bring external or pre-generated text into the pipeline without running inference.

### `import_csv_generations.py` — Generic CSV importer

Converts any two-column `(text, label)` CSV into the `generation.json` format expected by
`score_estimate_visualize.py` and other downstream stages.

```bash
# Minimal — outputs to out/<csv_stem>/generation.json
uv run python scripts/import_csv_generations.py data.csv

# Specify output path
uv run python scripts/import_csv_generations.py data.csv --output out/myexp/generation.json

# Custom column indices, no header row, custom metadata
uv run python scripts/import_csv_generations.py data.csv \
    --text-col 0 --label-col 1 --no-header \
    --model gpt-4o --prompt "Describe the patient."
```

**CSV format** (default: column 0 = text, column 1 = arm label, first row skipped as header):

```
text,label
"He fixed the car quickly.",man
"She fixed the car quickly.",woman
```

**Outputs**: `out/<csv_stem>/generation.json` + a `generation_cfg.json` stub so visualizations load metadata.

| Argument | Default | Description |
|----------|---------|-------------|
| `csv` | required | Path to input CSV |
| `--output` / `-o` | `out/<stem>/generation.json` | Output path |
| `--text-col` | `0` | Zero-based column index for generated text |
| `--label-col` | `1` | Zero-based column index for arm/condition label |
| `--no-header` | off | Pass if CSV has no header row |
| `--model` | `external` | Model name recorded in metadata |
| `--prompt` | `""` | Prompt text recorded in metadata |

### `import_more_of_the_same.py` — More-of-the-Same dataset importer

Merges the *specified* and *associated* CSVs produced by the [More-of-the-Same](https://github.com/jennm/more-of-the-same) dataset pipeline into a single two-column `(text, label)` CSV that `import_csv_generations.py` can then ingest.

- **Specified CSV** must have `text`, `gender`, and `occupation` columns. Labels are emitted as `"<gender> (specified)"`.
- **Associated CSV** must have `text`, `inferred_gender`, and `occupation` columns. Labels are emitted as `"<gender> (associated)"`.

Sampling is applied independently to each CSV: first per `(occupation, gender)` group, then per occupation.

```bash
# Basic merge — outputs to trials/csv_import_data/more-of-the-same-<pg>-pg-<po>-po[-random].csv
uv run python scripts/import_more_of_the_same.py specified.csv associated.csv

# Custom output path
uv run python scripts/import_more_of_the_same.py specified.csv associated.csv \
    --output out/mots/merged.csv

# Controlled sampling
uv run python scripts/import_more_of_the_same.py specified.csv associated.csv \
    --samples-per-occupation 5 --samples-per-gender 2

# Deterministic (first-N) instead of random sampling
uv run python scripts/import_more_of_the_same.py specified.csv associated.csv \
    --no-random-sample

# Skip non-binary rows
uv run python scripts/import_more_of_the_same.py specified.csv associated.csv \
    --exclude-non-binary

# No sampling — keep all occupation/gender pairs
uv run python scripts/import_more_of_the_same.py specified.csv associated.csv \
    --all-pairs
```

| Argument | Default | Description |
|----------|---------|-------------|
| `specified_csv` | required | Path to the specified CSV (`gender` column) |
| `associated_csv` | required | Path to the associated CSV (`inferred_gender` column) |
| `--output` / `-o` | auto | Output CSV path |
| `--samples-per-occupation` | `3` | Max rows per occupation after gender sampling |
| `--samples-per-gender` | `3` | Max rows per (occupation, gender) group |
| `--random-sample` / `--no-random-sample` | random | Whether to sample randomly or take first N |
| `--exclude-non-binary` | off | Exclude rows where gender marker is `N` |
| `--all-pairs` | off | Keep all pairs with no sampling |

**Typical workflow**:

```bash
# 1. Import and merge the More-of-the-Same CSVs
uv run python scripts/import_more_of_the_same.py \
    data/mots/specified.csv data/mots/associated.csv \
    --output trials/csv_import_data/mots-merged.csv

# 2. Convert the merged CSV into generation.json format
uv run python scripts/import_csv_generations.py \
    trials/csv_import_data/mots-merged.csv \
    --output out/mots/generation.json

# 3. Score, estimate, and visualize
uv run python scripts/score_estimate_visualize.py \
    out/mots/generation.json \
    trials/scoring/example.json
```

## Output Files

All outputs are saved to `out/<method>/`:

| Pattern | Contents |
|---------|----------|
| `<method>/gen_<config>.json` | Generated trajectories |
| `<method>/score_<gen>_<scoring>.json` | Scoring results |
| `<method>/est_<gen>_<scoring>.json` | Estimation results |
| `<method>/summary_*` | Human-readable summaries |
| `<method>/dynamics.json` | Dynamics data (with `--dynamics`) |
| `<method>/viz/dynamics/` | Dynamics plots (with `--dynamics`) |

## Directory Structure

```
scripts/
├── run_full_experiment.py        # Full pipeline orchestrator
├── generate_trajectories.py      # Unified generation (--method flag)
├── score_trajectories.py
├── estimate_normativity.py
├── visualize_estimation.py
├── score_estimate_visualize.py   # Score + estimate + visualize existing generation.json
├── import_csv_generations.py     # Convert two-column CSV → generation.json
├── import_more_of_the_same.py    # Merge More-of-the-Same specified/associated CSVs
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
