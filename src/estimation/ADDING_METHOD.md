# Adding a New Weighting Method

> **Note**: This documentation was AI-generated and may contain errors. If something seems off, check the code or open an issue.


Adding a weighting method to estimation requires **ONE FILE**.

## Steps

1. Create `src/estimation/methods/my_weighting_method.py`
2. Define params class with `name` and `description` ClassVars
3. Implement weighting function that returns normalized weights
4. Done - method is auto-discovered

## Template

```python
"""My weighting method for estimation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from ..weighting_method_registry import WeightingMethodParams, register_method

# Set to False to disable this weighting method
ENABLED = True


@dataclass
class MyWeightingParams(WeightingMethodParams):
    """Parameters for my weighting method."""

    # Add any configurable parameters here
    # my_param: float = 1.0

    # REQUIRED ClassVars
    name: ClassVar[str] = "my-weighting"  # Registry key
    description: ClassVar[str] = "my-weighted"  # Display label


def compute_my_weights(
    log_probs: Sequence[float],
    n_tokens: Sequence[int],
    params: MyWeightingParams,
) -> list[float]:
    """Compute weights for trajectories.

    Args:
        log_probs: Log probabilities for each trajectory
        n_tokens: Number of tokens per trajectory
        params: Method parameters

    Returns:
        Normalized weights summing to 1.0
    """
    # Your weighting logic here
    n = len(log_probs)
    if n == 0:
        return []

    # Example: uniform weights
    return [1.0 / n] * n


if ENABLED:
    compute_my_weights = register_method(MyWeightingParams)(compute_my_weights)
```

## Disabling a Method

To disable a weighting method, set `ENABLED = False` at the top of its file.
The method will not be registered and won't appear in results.

## What Happens Automatically

Once you create the file:

1. **Auto-registration**: The `@register_method` decorator registers your method
2. **Auto-discovery**: The `methods/__init__.py` auto-imports all method files
3. **Pipeline integration**: `compute_arm_estimate()` iterates over all registered methods
4. **Output inclusion**: Results appear in `ArmEstimate.estimates[method_name]`
5. **Display integration**: Logging functions iterate over all methods

## Existing Methods

| Method | File | Description |
|--------|------|-------------|
| `prob` | `prob_weighting_method.py` | Standard probability weighting |
| `inv-ppl` | `inv_ppl_weighting_method.py` | Inverse perplexity (per-token confidence) |
| `uniform` | `uniform_weighting_method.py` | Equal weights (baseline) |

## Testing Your Method

```python
from src.estimation.weighting_method_registry import get_method, list_methods

# Verify registration
assert "my-weighting" in list_methods()

# Test the function
weight_fn = get_method("my-weighting")
weights = weight_fn([-1.0, -2.0, -3.0], [10, 15, 20], MyWeightingParams())
assert abs(sum(weights) - 1.0) < 1e-6  # Must sum to 1
```
