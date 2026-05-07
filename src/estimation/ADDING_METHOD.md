# Adding a New Estimation Method

Adding a method to estimation requires **ONE FILE**.

Two kinds of method are supported:

* **Weighting methods**: produce weights; the core is derived via
  `generalized_system_core(scores, weights, q=1, r=1)`. Use this when your
  method is fundamentally a way of weighting trajectories.
* **Direct-core methods**: produce the core directly (selection, mode,
  etc.). Provide a `core_fn=` argument to `register_method`. Weights are
  still required (used for spread metrics — typically uniform).

## Steps

1. Create `src/estimation/methods/my_method.py`
2. Define params class with `name` and `description` ClassVars
3. Implement the function(s) appropriate for the kind of method
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

## Direct-Core Method Template

If your method computes the core itself (selection, mode, etc.), pass
`core_fn=` to `register_method`. The `core_fn` receives the full
`list[TrajectoryScoringData]` for the arm and returns the core vector
directly. It may raise `MethodNotApplicableError` to signal that the
method can't be applied (e.g., greedy with no greedy trajectory) — the
pipeline silently excludes it.

```python
from collections.abc import Sequence

from ..estimation_structure import TrajectoryScoringData
from ..weighting_method_registry import (
    MethodNotApplicableError,
    WeightingMethodParams,
    register_method,
)


@dataclass
class MyDirectParams(WeightingMethodParams):
    name: ClassVar[str] = "my-direct"
    description: ClassVar[str] = "my-direct-core"


def _uniform_weights(log_probs, n_tokens, params):
    n = len(log_probs)
    return [1.0 / n] * n if n else []


def _my_core(
    trajs: Sequence[TrajectoryScoringData],
    params: MyDirectParams,
) -> list[float]:
    if not trajs:
        raise MethodNotApplicableError("no trajectories")
    # ... compute core from trajs ...
    return core


if ENABLED:
    _uniform_weights = register_method(MyDirectParams, core_fn=_my_core)(
        _uniform_weights
    )
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

| Method | File | Kind | Description |
|--------|------|------|-------------|
| `prob` | `prob_weighting_method.py` | weighting | Standard probability weighting |
| `log-prob` | `log_prob_weighting_method.py` | weighting | Log-probability weighting |
| `inv-ppl` | `inv_ppl_weighting_method.py` | weighting | Inverse perplexity (per-token confidence) |
| `log-inv-ppl` | `log_inv_ppl_weighting_method.py` | weighting | Log inverse perplexity |
| `uniform` | `uniform_weighting_method.py` | weighting | Equal weights (baseline) |
| `greedy` | `greedy_core_method.py` | direct-core | Compliance of greedy-decoded trajectory |
| `max-prob` | `max_prob_core_method.py` | direct-core | Compliance of max-log-prob trajectory |
| `max-inv-ppl` | `max_inv_ppl_core_method.py` | direct-core | Compliance of max-inv-perplexity trajectory |
| `mode` | `mode_core_method.py` | direct-core | Per-structure logit-KDE mode |

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
