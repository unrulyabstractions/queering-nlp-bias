# Scripts

Executable scripts for the trajectory generation and analysis pipeline.

## Full Pipeline

```bash
# Default: simple temperature sampling
python scripts/run_full_experiment.py trials/generation/test.json trials/scoring/animal.json

# Forking paths: probe one-step deviations from greedy path
python scripts/run_full_experiment.py --forking-paths trials/generation/test.json trials/scoring/animal.json

# Seeking entropy: expand at high-uncertainty positions
python scripts/run_full_experiment.py --seeking-entropy trials/generation/test.json trials/scoring/animal.json
```

## Individual Scripts

### Generation Methods

All generation scripts output to `out/gen_<config>.json`.

#### Simple Sampling

```bash
python scripts/generate_by_simple_sampling.py trials/generation/test.json
python scripts/generate_by_simple_sampling.py trials/generation/test.json --samples-per-branch 5
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--samples-per-branch` | 2 | Trajectories to sample per branch |

#### Forking Paths

```bash
python scripts/generate_by_forking_paths.py trials/generation/test.json
python scripts/generate_by_forking_paths.py trials/generation/test.json \
    --max-alternates-per-position 5 \
    --min-prob-for-alternate 0.01 \
    --min-entropy-to-fork 1.0 \
    --samples-per-fork 2
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-alternates-per-position` | 3 | Max alternate tokens to consider per position |
| `--min-prob-for-alternate` | 0.05 | Minimum probability for an alternate token |
| `--min-entropy-to-fork` | 0.0 | Minimum entropy at position to consider forking |
| `--samples-per-fork` | 1 | Continuations to sample per fork point |

#### Seeking Entropy

```bash
python scripts/generate_by_seeking_entropy.py trials/generation/test.json
python scripts/generate_by_seeking_entropy.py trials/generation/test.json \
    --samples-per-expansion 3 \
    --num-expansion-rounds 4
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--samples-per-expansion` | 2 | Trajectories to sample per expansion |
| `--num-expansion-rounds` | 3 | Number of expansion rounds |

### Scoring

```bash
python scripts/score_trajectories.py trials/scoring/animal.json out/gen_sampling_test.json
```

Output: `out/score_<gen>_<scoring>.json`

### Estimation

```bash
python scripts/estimate_normativity.py out/score_sampling_test_animal.json
```

Output: `out/est_<gen>_<scoring>.json`

## run_full_experiment.py Parameters

All method-specific parameters can be passed to `run_full_experiment.py`:

```bash
# Simple sampling
python scripts/run_full_experiment.py gen.json scoring.json --samples-per-branch 10

# Forking paths
python scripts/run_full_experiment.py --forking-paths gen.json scoring.json \
    --max-alternates-per-position 5 \
    --min-prob-for-alternate 0.01 \
    --samples-per-fork 2

# Seeking entropy
python scripts/run_full_experiment.py --seeking-entropy gen.json scoring.json \
    --samples-per-expansion 3 \
    --num-expansion-rounds 5
```
