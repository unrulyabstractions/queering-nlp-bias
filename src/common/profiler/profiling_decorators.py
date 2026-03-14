"""Profiling decorators for experiment steps."""

from __future__ import annotations

import functools
from typing import Callable, TypeVar, Union, overload

from .profiling_timer import P
from ..device_utils import get_memory_usage, log_memory
from src.common.logging import log

F = TypeVar("F", bound=Callable)


def _get_accel_mem(mem: dict) -> float:
    """Get accelerator memory (MLX > MPS > CUDA)."""
    return (
        mem.get("mlx_alloc_gb", 0)
        or mem.get("mps_alloc_gb", 0)
        or mem.get("cuda_alloc_gb", 0)
    )


def track_memory(func: F) -> F:
    """Decorator to log memory usage before and after a function.

    Usage:
        @track_memory
        def cleanup(self):
            # cleanup logic
            pass

    Logs:
        - Memory before function call
        - Memory after function call
        - Delta (freed/allocated)
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get function context (for logging)
        func_name = func.__name__
        # Try to get self.model_name or similar for context
        context = ""
        if args and hasattr(args[0], "model_name"):
            context = f" ({args[0].model_name})"

        log(f"\n[Memory] {func_name}{context}...")

        # Memory before
        mem_before = get_memory_usage()
        ram_before = mem_before.get("ram_gb", 0)
        accel_before = _get_accel_mem(mem_before)

        # Run function
        result = func(*args, **kwargs)

        # Memory after
        mem_after = get_memory_usage()
        ram_after = mem_after.get("ram_gb", 0)
        accel_after = _get_accel_mem(mem_after)

        # Log delta
        ram_delta = ram_after - ram_before
        accel_delta = accel_after - accel_before
        log(f"  RAM: {ram_after:.2f}GB (Δ{ram_delta:+.2f}GB)")
        if accel_before > 0 or accel_after > 0:
            log(f"  Accel: {accel_after:.2f}GB (Δ{accel_delta:+.2f}GB)")

        return result

    return wrapper  # type: ignore


@overload
def profile(func: F) -> F: ...


@overload
def profile(identifier: str) -> Callable[[F], F]: ...


def profile(
    func_or_identifier: Union[F, str, None] = None, verbose: bool = False
) -> Union[F, Callable[[F], F]]:
    """Decorator to profile functions.

    Can be used with or without arguments:
        @profile
        def my_func(): ...

        @profile("custom_name")
        def my_func(): ...

    If no identifier is provided, uses the function's name.
    """

    def make_wrapper(func: F, name: str, use_verbose: bool) -> F:
        profile_name = name.lower().replace(" ", "_")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Print step header
            if use_verbose:
                log(f"\n{'=' * 60}")
                log(f"PROFILE: {name}")
                log("=" * 60)

            # Run with profiling
            with P(profile_name):
                result = func(*args, **kwargs)

            # Log memory
            log_memory(f"after_{profile_name}", use_verbose)

            return result

        return wrapper  # type: ignore

    # Called as @profile (no parens) - func_or_identifier is the function
    if callable(func_or_identifier):
        return make_wrapper(func_or_identifier, func_or_identifier.__name__, verbose)

    # Called as @profile() or @profile("name") - func_or_identifier is str or None
    identifier = func_or_identifier or ""

    def decorator(func: F) -> F:
        name = identifier if identifier else func.__name__
        return make_wrapper(func, name, verbose)

    return decorator
