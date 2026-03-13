# CLAUDE.md - src/scoring/

This module scores generated trajectories using configurable methods (LLM judgments, similarity, word counts).

## Architecture

```python
# Simple: one object, simple methods
scorer = Scorer(config)        # or Scorer.load("config.json")
scores = scorer.score(text)    # -> list[float] (structure scores)

# Pipeline for many trajectories
result = run_scoring_pipeline(config, trajectories, arm_names, arm_texts)
```

`Scorer` mirrors `ModelRunner`: load once, call many times.

## Registry Pattern

**Adding a new scoring method requires ONE FILE in `methods/`.**

```python
# methods/my_scoring_method.py
from dataclasses import dataclass
from typing import ClassVar
from ..scoring_method_registry import ScoringMethodParams, register_method

@dataclass
class MyParams(ScoringMethodParams):
    name: ClassVar[str] = "my-method"           # Registry key
    config_key: ClassVar[str] = "my_method_items"  # JSON field name
    label_prefix: ClassVar[str] = "m"           # Labels: m1, m2...
    requires_runner: ClassVar[bool] = False     # Needs LLM?
    requires_embedder: ClassVar[bool] = False   # Needs embeddings?

    threshold: float = 0.5  # Method-specific param

@register_method(MyParams)
def score_my_method(text, items, params, runner=None, embedder=None, log_fn=None):
    scores = []
    raw_responses = []
    for item in items:
        # Score each item
        scores.append(compute_score(text, item, params.threshold))
        raw_responses.append("")
    return scores, raw_responses
```

The method is automatically discovered - no other changes needed.

## Key Files

| File | Purpose |
|------|---------|
| `scoring_pipeline.py` | `run_scoring_pipeline()` entry point |
| `scoring_method_registry.py` | `@register_method`, `get_method()`, `score_with_bundling()` |
| `scoring_config.py` | Config loading with method params |
| `scoring_data.py` | `TrajectoryData`, `GenerationOutputData` input types |
| `scoring_output.py` | `ScoringOutput`, `ScoringResult` output types |

## Available Methods

| Method | Config Key | Label | Requires |
|--------|------------|-------|----------|
| `categorical` | `categorical_judgements` | c1, c2... | LLM |
| `graded` | `graded_judgements` | g1, g2... | LLM |
| `similarity` | `similarity_scoring` | s1, s2... | Embedder |
| `count-occurrences` | `count_occurrences` | o1, o2... | None |

## Bundled Items

Config items can be single strings or lists (bundles):

```json
{
  "categorical_judgements": [
    "Simple question?",
    ["Bundled Q1?", "Bundled Q2?"]  // Averaged into one structure score
  ]
}
```

Use `score_with_bundling()` from registry to handle this automatically.

## String Selection

The `string_selection` config determines which text portion to score:
- `WholeContinuation` - full generated text (default)
- `NonThinkingContinuation` - strips `<think>...</think>` blocks
- `AfterTrunk` / `AfterBranch` / `AfterTwig` - after specific arm prefix

## Common Pitfalls

1. **Methods must return `(scores, raw_responses)` tuple** - both lists
2. **Bundle support is optional** - use `score_with_bundling()` helper
3. **Config key must match JSON field name** - e.g., `my_method_items` not `myMethodItems`
4. **All params need defaults** - for `get_default_params()` to work

## Output Path Convention

```
out/<gen-method>/<gen-name>/<scoring-name>/scoring.json
out/<gen-method>/<gen-name>/<scoring-name>/summary_scoring.txt
```

## See Also

- [EXPLANATION.md](./EXPLANATION.md) - detailed algorithm specification
- [ADDING_METHOD.md](./ADDING_METHOD.md) - step-by-step guide
- [methods/README.md](./methods/README.md) - existing method implementations
- [Root CLAUDE.md](../../CLAUDE.md) - global project rules
