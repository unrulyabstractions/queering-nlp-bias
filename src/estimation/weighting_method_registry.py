"""Core estimation method registry.

Each method produces a *core* (expected structure compliance vector) plus
*weights* used for spread metrics (deviance, orientation).

Two flavors of methods are supported:

1. Weighting methods (classic): only `weight_fn` is provided. The core is
   derived from `generalized_system_core(scores, weights, q=1, r=1)`.

    @dataclass
    class MyWeightingParams(WeightingMethodParams):
        name: ClassVar[str] = "my-weighting"
        description: ClassVar[str] = "My custom weighting scheme"

    @register_method(MyWeightingParams)
    def compute_my_weights(log_probs, n_tokens, params):
        return [1.0 / len(log_probs)] * len(log_probs)

2. Direct-core methods (selection / mode / etc.): both `weight_fn` and
   `core_fn` are provided. `core_fn` overrides the derived core.

    @register_method(MyParams, core_fn=my_core_fn)
    def my_weights(log_probs, n_tokens, params):
        return [1.0 / len(log_probs)] * len(log_probs)

A direct-core method may raise `MethodNotApplicableError` when its
prerequisites aren't met for a given arm (e.g. `greedy` with no greedy
trajectory). The pipeline catches and silently excludes that method for
that arm.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, ClassVar

from src.common.params_schema import ParamsSchema


class MethodNotApplicableError(Exception):
    """Raised when a method cannot be applied to the data at hand.

    The estimation pipeline catches this and silently excludes the method
    from the arm's `estimates` dict.
    """

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETER BASE CLASS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class WeightingMethodParams(ParamsSchema):
    """Base class for weighting method parameters.

    Subclasses MUST define as ClassVar:
    - name: str - method name for registry lookup
    - description: str - human-readable description for display

    All fields should have defaults so get_default_params() works.
    """

    name: ClassVar[str]
    description: ClassVar[str]


# Type alias for weighting functions
# Args: (log_probs, n_tokens, params)
# Returns: normalized weights summing to 1.0
WeightFn = Callable[
    [Sequence[float], Sequence[int], WeightingMethodParams],
    list[float],
]

# Type alias for direct-core estimation functions.
# Args: (trajectories, params) — trajectories are TrajectoryScoringData objects.
# Returns: a core vector (one value per structure)
# Raises: MethodNotApplicableError if the method can't be applied
# Typed as `Sequence[Any]` here to avoid a circular import with
# estimation_structure; in practice this is `list[TrajectoryScoringData]`.
CoreFn = Callable[
    [Sequence, WeightingMethodParams],
    list[float],
]


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

# Registry stores (params_class, weight_fn, core_fn_or_None) tuples
_REGISTRY: dict[
    str,
    tuple[type[WeightingMethodParams], WeightFn, CoreFn | None],
] = {}


def register_method(
    params_class: type[WeightingMethodParams],
    core_fn: CoreFn | None = None,
):
    """Decorator to register a core estimation method.

    Args:
        params_class: ParamsSchema subclass with `name` and `description` ClassVars.
        core_fn: Optional direct-core function. If provided, the pipeline uses
            its return value as the core instead of deriving it from weights.

    Usage:
        @register_method(MyParams)
        def compute_my_weights(log_probs, n_tokens, params):
            return normalized_weights

        @register_method(MyParams, core_fn=my_core_fn)
        def compute_uniform_weights(log_probs, n_tokens, params):
            return uniform_weights
    """

    def decorator(fn: WeightFn) -> WeightFn:
        _REGISTRY[params_class.name] = (params_class, fn, core_fn)
        return fn

    return decorator


def get_method(name: str) -> WeightFn:
    """Get the weight function for a method by name."""
    if name not in _REGISTRY:
        valid = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown estimation method: {name}. Valid: {valid}")
    return _REGISTRY[name][1]


def get_core_fn(name: str) -> CoreFn | None:
    """Get the optional direct-core function for a method by name."""
    if name not in _REGISTRY:
        valid = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown estimation method: {name}. Valid: {valid}")
    return _REGISTRY[name][2]


def get_default_params(name: str) -> WeightingMethodParams:
    """Get default params for a method (all defaults applied)."""
    if name not in _REGISTRY:
        valid = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown estimation method: {name}. Valid: {valid}")
    return _REGISTRY[name][0]()


def get_params_class(name: str) -> type[WeightingMethodParams]:
    """Get the params class for a method by name."""
    if name not in _REGISTRY:
        valid = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown estimation method: {name}. Valid: {valid}")
    return _REGISTRY[name][0]


def list_methods() -> list[str]:
    """List all registered method names."""
    return sorted(_REGISTRY.keys())


def iter_methods() -> list[tuple[str, type[WeightingMethodParams], WeightFn]]:
    """Iterate over all registered methods.

    Returns methods with the default method first, then remaining methods
    in sorted order.

    Returns:
        List of (name, params_class, weight_fn) tuples. The optional
        core_fn is not included here; use `get_core_fn(name)` to retrieve it.
    """
    from src.common.default_config import DEFAULT_WEIGHTING_METHOD

    result: list[tuple[str, type[WeightingMethodParams], WeightFn]] = []
    if DEFAULT_WEIGHTING_METHOD in _REGISTRY:
        pc, fn, _ = _REGISTRY[DEFAULT_WEIGHTING_METHOD]
        result.append((DEFAULT_WEIGHTING_METHOD, pc, fn))

    for name, (pc, fn, _) in sorted(_REGISTRY.items()):
        if name != DEFAULT_WEIGHTING_METHOD:
            result.append((name, pc, fn))

    return result


def get_method_description(name: str) -> str:
    """Get the description for an estimation method."""
    if name not in _REGISTRY:
        valid = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown estimation method: {name}. Valid: {valid}")
    return _REGISTRY[name][0].description
