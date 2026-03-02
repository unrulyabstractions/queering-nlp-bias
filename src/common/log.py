"""Simple logging utilities for clean, readable code."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from functools import wraps
from typing import Callable

# Simple print with flush
def log(msg: str = "", end: str = "\n", gap: int = 0) -> None:
    """Print with immediate flush.

    Args:
        msg: Message to print
        end: Line ending (default newline)
        gap: Number of blank lines to print before the message
    """
    for _ in range(gap):
        print(flush=True)
    print(msg, end=end, flush=True)


def log_section(title: str) -> None:
    """Print a section header."""
    log(f"\n{title}")


def log_params(**kwargs) -> None:
    """Print parameters as indented key-value pairs."""
    for key, value in kwargs.items():
        log(f"  {key}: {value}")


def log_progress(current: int, total: int, prefix: str = "") -> None:
    """Print progress indicator (overwrites line)."""
    log(f"{prefix}{current}/{total}", end="\r")


def log_done(msg: str = "") -> None:
    """Print completion message (clears progress line)."""
    log(msg)


@contextmanager
def log_step(name: str):
    """Context manager for logging a step with completion."""
    log(f"{name}...", end="")
    sys.stdout.flush()
    try:
        yield
        log(" done")
    except Exception:
        log(" failed")
        raise


def logged(name: str | None = None) -> Callable:
    """Decorator to log function entry with parameters."""
    def decorator(fn: Callable) -> Callable:
        fn_name = name or fn.__name__

        @wraps(fn)
        def wrapper(*args, **kwargs):
            if kwargs:
                params = ", ".join(f"{k}={v}" for k, v in kwargs.items() if not k.startswith("_"))
                log(f"{fn_name}({params})")
            else:
                log(f"{fn_name}")
            return fn(*args, **kwargs)

        return wrapper
    return decorator
