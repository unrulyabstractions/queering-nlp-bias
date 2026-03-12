"""Generation method registry with auto-discovery.

Adding a new generation method is ONE FILE:

    @dataclass
    class MyParams(GenerationMethodParams):
        my_param: int = 10
        name: ClassVar[str] = "my-method"

    @register_method(MyParams)
    def generate_my_method(runner, config, params, log_fn=None):
        ...

That's it. The method is now available via:
- get_method("my-method")
- get_default_params("my-method")
- run_full_experiment.py --method my-method
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, ClassVar

from src.common.callback_types import LogFn
from src.common.experiment_types import ArmGenerationResult
from src.common.params_schema import ParamsSchema
from src.inference import ModelRunner

if TYPE_CHECKING:
    from .generation_config import GenerationConfig

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETER BASE CLASS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class GenerationMethodParams(ParamsSchema):
    """Base class for generation method parameters.

    Subclasses MUST define as ClassVar:
    - name: str - method name for registry lookup and output files

    All fields should have defaults so get_default_params() works.
    """

    # Subclasses override this as ClassVar
    name: ClassVar[str]


# Type alias for generate functions
GenerateFn = Callable[
    [ModelRunner, "GenerationConfig", GenerationMethodParams, LogFn | None],
    ArmGenerationResult,
]

# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

# Registry stores (params_class, generate_fn) pairs
_REGISTRY: dict[str, tuple[type[GenerationMethodParams], GenerateFn]] = {}


def register_method(params_class: type[GenerationMethodParams]):
    """Decorator to register a generation method.

    Usage:
        @register_method(MyParams)
        def generate_my_method(runner, config, params, log_fn=None):
            ...
    """

    def decorator(generate_fn: GenerateFn) -> GenerateFn:
        _REGISTRY[params_class.name] = (params_class, generate_fn)
        return generate_fn

    return decorator


def get_method(name: str) -> GenerateFn:
    """Get a generation function by name."""
    if name not in _REGISTRY:
        valid = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown method: {name}. Valid: {valid}")
    return _REGISTRY[name][1]


def get_default_params(name: str) -> GenerationMethodParams:
    """Get default params for a method (all defaults applied)."""
    if name not in _REGISTRY:
        valid = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown method: {name}. Valid: {valid}")
    return _REGISTRY[name][0]()


def get_params_class(name: str) -> type[GenerationMethodParams]:
    """Get the params class for a method by name."""
    if name not in _REGISTRY:
        valid = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown method: {name}. Valid: {valid}")
    return _REGISTRY[name][0]


def list_methods() -> list[str]:
    """List all registered method names."""
    return sorted(_REGISTRY.keys())


def get_output_name(name: str) -> str:
    """Get output file name for a method (identical to method name).

    Validates that the method exists in the registry.
    """
    if name not in _REGISTRY:
        valid = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown method: {name}. Valid: {valid}")
    return name


# Alias - method names and output names are identical
list_output_names = list_methods


def get_method_name_from_output(output_name: str) -> str:
    """Get method name from output name (they're identical).

    Validates that the name exists in the registry.
    """
    if output_name not in _REGISTRY:
        valid = ", ".join(list_methods())
        raise ValueError(f"Unknown output name: {output_name}. Valid: {valid}")
    return output_name


def params_from_dict(method: str, data: dict) -> GenerationMethodParams:
    """Create a params instance from a dict using the correct class.

    Args:
        method: The method name (e.g., "simple-sampling")
        data: Raw dict of parameter values

    Returns:
        Properly typed params instance
    """
    params_class = get_params_class(method)
    return params_class.from_dict(data)
