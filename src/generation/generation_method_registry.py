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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, ClassVar, Generic, TypeVar

from src.common.callback_types import LogFn
from src.common.params_schema import ParamsSchema
from src.inference import ModelRunner

from .generation_types import ArmGenerationResult

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


P = TypeVar("P", bound=GenerationMethodParams)

# Type alias for generate functions
GenerateFn = Callable[
    [ModelRunner, "GenerationConfig", GenerationMethodParams, LogFn | None],
    ArmGenerationResult,
]

# ══════════════════════════════════════════════════════════════════════════════
# METHOD BASE CLASS (for backwards compatibility)
# ══════════════════════════════════════════════════════════════════════════════


class GenerationMethod(ABC, Generic[P]):
    """Abstract base class for generation method implementations.

    You can either:
    1. Subclass this and use @register_method(ParamsClass) on the class
    2. Just use @register_method(ParamsClass) on a generate function directly

    Option 2 is simpler and recommended for new methods.
    """

    # Set by @register_method decorator
    params_class: ClassVar[type[GenerationMethodParams]]

    @property
    def name(self) -> str:
        """Method name from params class."""
        return self.params_class.name

    @property
    def output_name(self) -> str:
        """Output file name (same as name)."""
        return self.params_class.name

    @abstractmethod
    def generate(
        self,
        runner: ModelRunner,
        config: GenerationConfig,
        params: P,
        log_fn: LogFn | None = None,
    ) -> ArmGenerationResult:
        """Execute this generation method."""
        ...


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

# Registry stores (params_class, generate_fn) pairs
_REGISTRY: dict[str, tuple[type[GenerationMethodParams], GenerateFn]] = {}


def register_method(params_class: type[GenerationMethodParams]):
    """Decorator to register a generation method.

    Can decorate either a function or a class:

    Function (recommended):
        @register_method(MyParams)
        def generate_my_method(runner, config, params, log_fn=None):
            ...

    Class (for backwards compatibility):
        @register_method(MyParams)
        class MyMethod(GenerationMethod[MyParams]):
            def generate(self, runner, config, params, log_fn=None):
                ...
    """

    def decorator(fn_or_cls):
        if isinstance(fn_or_cls, type) and issubclass(fn_or_cls, GenerationMethod):
            # Class-based registration (backwards compatibility)
            fn_or_cls.params_class = params_class
            # Create a function that instantiates and calls the method
            method_instance = fn_or_cls()

            def generate_fn(runner, config, params, log_fn=None):
                return method_instance.generate(runner, config, params, log_fn)

            _REGISTRY[params_class.name] = (params_class, generate_fn)
        else:
            # Function-based registration (simpler, recommended)
            _REGISTRY[params_class.name] = (params_class, fn_or_cls)

        return fn_or_cls

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


def get_output_name(name: str) -> str:
    """Get output file name for a method (same as method name)."""
    if name not in _REGISTRY:
        valid = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown method: {name}. Valid: {valid}")
    return _REGISTRY[name][0].name


def list_methods() -> list[str]:
    """List all registered method names."""
    return sorted(_REGISTRY.keys())


def list_output_names() -> list[str]:
    """List all registered output names (same as method names)."""
    return sorted(_REGISTRY.keys())


def get_method_name_from_output(output_name: str) -> str:
    """Get method name from output name (they're the same now)."""
    if output_name not in _REGISTRY:
        valid = ", ".join(list_output_names())
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
