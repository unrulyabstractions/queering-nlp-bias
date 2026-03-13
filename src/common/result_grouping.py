"""Shared utilities for grouping results by arm.

Provides a generic way to group result dictionaries by their arm field.
"""

from __future__ import annotations

from typing import Any


def group_results_by_arm(
    results: list[dict[str, Any]],
    arm_key: str = "arm",
    default_arm: str = "trunk",
) -> dict[str, list[dict[str, Any]]]:
    """Group result dictionaries by arm name.

    Args:
        results: List of result dictionaries
        arm_key: Key to extract arm name from each result
        default_arm: Default arm name if key is missing

    Returns:
        Dictionary mapping arm names to lists of results
    """
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        arm = result.get(arm_key, default_arm)
        by_arm.setdefault(arm, []).append(result)
    return by_arm
