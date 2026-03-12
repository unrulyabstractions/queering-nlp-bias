# Scoring Package

Score generated trajectories using configurable scoring methods.

## Quick Links

| File | Purpose |
|------|---------|
| `scoring_config.py` | Configuration for scoring runs |
| `scoring_pipeline.py` | Core pipeline logic and text processing |
| `scoring_method_registry.py` | Method registration and discovery |
| `scoring_data.py` | Input data structures (trajectories, generation output) |
| `scoring_output.py` | Output structures and path computation |
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
gen_data = GenerationOutputData.load("out/simple-sampling/example/generation.json")
config = ScoringConfig.load("trials/scoring/example.json")

# Run scoring
result = run_scoring_pipeline(
    config=config,
    trajectories=gen_data.trajectories,
    arm_names=gen_data.arm_names,
    arm_texts=gen_data.arm_texts,
)

# Save output to out/<method>/<gen_name>/<scoring_name>/scoring.json
output_path = result.output.compute_output_path(gen_path, scoring_path)
result.output.save(output_path)
result.output.summarize()
```

See [EXPLANATION.md](EXPLANATION.md) for detailed specification and [ADDING_METHOD.md](ADDING_METHOD.md) for adding new scoring methods.
