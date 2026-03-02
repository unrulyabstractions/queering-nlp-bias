# queering-nlp-bias

Measuring and analyzing normativity in language model text generation through trajectory sampling and structural analysis.

## Quick Start

```bash
# Run full experiment pipeline
python scripts/run_full_experiment.py trials/generation/test.json trials/scoring/animal.json

# With different generation methods
python scripts/run_full_experiment.py --forking-paths trials/generation/test.json trials/scoring/animal.json
python scripts/run_full_experiment.py --seeking-entropy trials/generation/test.json trials/scoring/animal.json
```

## Project Structure

```
.
├── scripts/           # Executable scripts (see scripts/README.md)
├── trials/            # Config files for experiments (see trials/README.md)
├── out/               # Generated outputs (gen_*, judge_*, est_*)
└── src/               # Core library code
```

## Pipeline Overview

1. **Generate** - Sample text continuations from a language model
2. **Judge** - Evaluate each trajectory against categorical judgments
3. **Estimate** - Compute normativity metrics (core, orientation, deviance)

## Output Files

All outputs go to `out/`:

| File Pattern | Description |
|--------------|-------------|
| `gen_<config>.json` | Generated trajectories with token trees |
| `judge_<gen>_<scoring>.json` | Judgment results for each trajectory |
| `est_<gen>_<scoring>.json` | Normativity estimates and summaries |

## See Also

- [scripts/README.md](scripts/README.md) - Script usage and parameters
- [trials/README.md](trials/README.md) - Config file formats and options
