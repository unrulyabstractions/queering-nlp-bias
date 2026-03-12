"""Function decorators for logging.

Provides decorators to automatically log function calls.
"""

from __future__ import annotations

from functools import wraps
from typing import Callable

from .log_primitives import log


def logged(name: str | None = None) -> Callable:
    """Decorator to log function entry with parameters.

    Args:
        name: Custom name for the function (defaults to function.__name__)

    Returns:
        Decorated function that logs calls
    """

    def decorator(fn: Callable) -> Callable:
        fn_name = name or fn.__name__

        @wraps(fn)
        def wrapper(*args, **kwargs):
            if kwargs:
                params = ", ".join(
                    f"{k}={v}" for k, v in kwargs.items() if not k.startswith("_")
                )
                log(f"{fn_name}({params})")
            else:
                log(f"{fn_name}")
            return fn(*args, **kwargs)

        return wrapper

    return decorator
