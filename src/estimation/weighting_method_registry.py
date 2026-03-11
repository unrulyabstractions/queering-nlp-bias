"""Weighting method registry for estimation.

Adding a new weighting method is ONE FILE:

    @dataclass
    class MyWeightingParams(WeightingMethodParams):
        name: ClassVar[str] = "my-weighting"
        description: ClassVar[str] = "My custom weighting scheme"

    @register_method(MyWeightingParams)
    def compute_my_weights(log_probs, n_tokens, params):
        # Return normalized weights summing to 1.0
        return [1.0 / len(log_probs)] * len(log_probs)

The method is automatically:
- Discovered by the estimation pipeline
- Applied to all arms
- Has its results stored in WeightedEstimate dict
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, ClassVar

from src.common.params_schema import ParamsSchema


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


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

# Registry stores (params_class, weight_fn) pairs
_REGISTRY: dict[str, tuple[type[WeightingMethodParams], WeightFn]] = {}


def register_method(params_class: type[WeightingMethodParams]):
    """Decorator to register a weighting method.

    Usage:
        @dataclass
        class MyParams(WeightingMethodParams):
            name: ClassVar[str] = "my-weighting"
            description: ClassVar[str] = "My weighting scheme"

        @register_method(MyParams)
        def compute_my_weights(log_probs, n_tokens, params):
            return normalized_weights
    """

    def decorator(fn: WeightFn) -> WeightFn:
        _REGISTRY[params_class.name] = (params_class, fn)
        return fn

    return decorator


def get_method(name: str) -> WeightFn:
    """Get a weighting function by name."""
    if name not in _REGISTRY:
        valid = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown weighting method: {name}. Valid: {valid}")
    return _REGISTRY[name][1]


def get_default_params(name: str) -> WeightingMethodParams:
    """Get default params for a method (all defaults applied)."""
    if name not in _REGISTRY:
        valid = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown weighting method: {name}. Valid: {valid}")
    return _REGISTRY[name][0]()


def get_params_class(name: str) -> type[WeightingMethodParams]:
    """Get the params class for a method by name."""
    if name not in _REGISTRY:
        valid = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown weighting method: {name}. Valid: {valid}")
    return _REGISTRY[name][0]


def list_methods() -> list[str]:
    """List all registered method names."""
    return sorted(_REGISTRY.keys())


def iter_methods() -> list[tuple[str, type[WeightingMethodParams], WeightFn]]:
    """Iterate over all registered methods.

    Returns methods with the default weighting method first,
    then remaining methods in sorted order.

    Returns:
        List of (name, params_class, weight_fn) tuples
    """
    from src.common.default_config import DEFAULT_WEIGHTING_METHOD

    # Put default method first, then sort the rest
    result = []
    if DEFAULT_WEIGHTING_METHOD in _REGISTRY:
        pc, fn = _REGISTRY[DEFAULT_WEIGHTING_METHOD]
        result.append((DEFAULT_WEIGHTING_METHOD, pc, fn))

    for name, (pc, fn) in sorted(_REGISTRY.items()):
        if name != DEFAULT_WEIGHTING_METHOD:
            result.append((name, pc, fn))

    return result


def get_method_description(name: str) -> str:
    """Get the description for a weighting method."""
    if name not in _REGISTRY:
        valid = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown weighting method: {name}. Valid: {valid}")
    return _REGISTRY[name][0].description
