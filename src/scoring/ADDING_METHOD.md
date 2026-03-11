# Adding a New Scoring Method

> **Note**: This documentation was AI-generated and may contain errors. If something seems off, check the code or open an issue.


Adding a new scoring method requires **ONE FILE** - no changes to config, pipeline, or output needed.

## Quick Summary

Create one file with:
1. A params dataclass with `name`, `config_key`, `label_prefix` ClassVars
2. A function decorated with `@register_method`

That's it. The method is automatically:
- Discovered by the config (reads from `config_key` in JSON)
- Run by the pipeline
- Stored in results
- Displayed in summary

## Minimal Example

```python
"""My custom scoring method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from src.common.callback_types import LogFn
from src.inference import ModelRunner
from src.inference.embedding_runner import EmbeddingRunner

from ..scoring_method_registry import ScoringMethodParams, register_method


@dataclass
class MyParams(ScoringMethodParams):
    """Parameters for my method."""

    my_param: int = 10

    # REQUIRED: Registry metadata
    name: ClassVar[str] = "my-method"           # Method name
    config_key: ClassVar[str] = "my_method"     # JSON field name
    label_prefix: ClassVar[str] = "m"           # Structure label (m1, m2, ...)

    # OPTIONAL: Resource requirements
    requires_runner: ClassVar[bool] = False     # Needs LLM?
    requires_embedder: ClassVar[bool] = False   # Needs embeddings?


@register_method(MyParams)
def score_my_method(
    text: str,
    items: list[str | list[str]],  # Your method's data from config
    params: MyParams,
    runner: ModelRunner | None = None,
    embedder: EmbeddingRunner | None = None,
    log_fn: LogFn | None = None,
) -> tuple[list[float], list[str]]:
    """Score text using my algorithm."""
    scores = []

    for struct_idx, item in enumerate(items):
        if isinstance(item, list):
            # Bundled items
            if log_fn:
                log_fn(f"[{params.label_prefix}{struct_idx + 1}] Bundled ({len(item)} items)")
            for sub_item in item:
                score = compute_score(text, sub_item, params)
                scores.append(score)
                if log_fn:
                    log_fn(f"     • {sub_item} -> {score:.3f}")
        else:
            score = compute_score(text, item, params)
            scores.append(score)
            if log_fn:
                log_fn(f"[{params.label_prefix}{struct_idx + 1}] {item} -> {score:.3f}")

    return scores, [""] * len(scores)
```

Save as `src/scoring/methods/my_method.py` and it's registered.

## JSON Config

Your method's data is read from `config_key`:

```json
{
    "model": "Qwen/Qwen3.5-0.8B",
    "my_method": ["target1", ["bundled1", "bundled2"]]
}
```

## That's It!

No other files need modification:
- ✅ Config automatically reads `my_method` field
- ✅ Pipeline automatically runs your method
- ✅ Results automatically stored as `method_scores["my-method"]`
- ✅ Summary automatically displays your scores

## Testing

```bash
# Check registration
uv run python -c "from src.scoring import list_methods; print(list_methods())"

# Run with your method
uv run python scripts/score_trajectories.py trials/scoring/my_config.json out/gen_*.json
```

## Key Points

1. **config_key** = JSON field name (e.g., `"my_method"` -> reads `config["my_method"]`)
2. **label_prefix** = single letter for structure labels (m1, m2, m3...)
3. **items** parameter = your method's data from config (already extracted)
4. **requires_runner/requires_embedder** = declare what resources you need

## Examples

Look at existing methods:
- `count_occurrences_method.py` - No LLM, no embeddings (simplest)
- `similarity_method.py` - Uses embeddings
- `categorical_method.py` - Uses LLM
