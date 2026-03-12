"""Schema utilities for loading data from JSON.

This module provides utilities for safely converting JSON values,
particularly handling special float values (inf, nan) that JSON
cannot represent natively.
"""

from __future__ import annotations

from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float, handling JSON infinity/NaN strings.

    JSON cannot represent infinity or NaN natively, so they get serialized
    as strings like "Inf", "Infinity", "-Inf", "NaN". This function handles
    those cases when loading data back from JSON.

    Args:
        value: Value to convert (int, float, string, or None)
        default: Default value if conversion fails

    Returns:
        Float value
    """
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        lower = value.lower()
        if lower in ("inf", "infinity"):
            return float("inf")
        if lower in ("-inf", "-infinity"):
            return float("-inf")
        if lower == "nan":
            return float("nan")
        try:
            return float(value)
        except ValueError:
            return default
    return default
