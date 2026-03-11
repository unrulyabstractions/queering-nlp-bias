# Scoring Package

> **Note**: This documentation was AI-generated and may contain errors. If something seems off, check the code or open an issue.


Score generated trajectories using configurable scoring methods.

## Quick Links

| File | Purpose |
|------|---------|
| `scoring_config.py` | `ScoringConfig` - configuration for scoring runs |
| `scoring_pipeline.py` | `run_scoring_pipeline()` - main entry point |
| `scoring_method_registry.py` | Method registration and discovery |
| `scoring_data.py` | `TrajectoryData` - input data structures |
| `scoring_output.py` | `ScoringResult`, `ScoringOutput` - output structures |
| `methods/` | Scoring method implementations |

## Available Methods

| Method | Config Key | Label | Requires |
|--------|------------|-------|----------|
| `categorical` | `categorical_judgements` | c1, c2... | LLM |
| `graded` | `graded_judgements` | g1, g2... | LLM |
| `similarity` | `similarity_scoring` | s1, s2... | Embedder |
| `count-occurrences` | `count_occurrences` | o1, o2... | None |

## Usage

```python
from src.scoring import ScoringConfig, run_scoring_pipeline, GenerationOutputData

# Load data
gen_data = GenerationOutputData.load("out/gen_example.json")
config = ScoringConfig.load("trials/scoring/example.json")

# Run scoring
result = run_scoring_pipeline(
    config=config,
    trajectories=gen_data.trajectories,
    branches=gen_data.branches,
    arm_texts=gen_data.arm_texts,
)

# Save and summarize
result.output.save("out/score_example.json")
result.output.summarize()
```

See [EXPLANATION.md](EXPLANATION.md) for detailed specification.
