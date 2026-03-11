"""Display name utilities for arms and structures.

This module provides utilities for generating human-readable display names
for various elements in the trajectory analysis system.
"""

from __future__ import annotations


def arm_display_name(arm_idx: int) -> str:
    """Convert arm index to display name.

    Arms are conditioning points in the generation tree:
    - Index 0 is the trunk (base trajectory)
    - Index 1+ are branches (conditioned on different prefixes)

    Args:
        arm_idx: Index of the arm (0 = trunk, 1+ = branches)

    Returns:
        Display name: "trunk" for index 0, "branch_N" for index N.

    Example:
        >>> arm_display_name(0)
        'trunk'
        >>> arm_display_name(1)
        'branch_1'
        >>> arm_display_name(3)
        'branch_3'
    """
    return "trunk" if arm_idx == 0 else f"branch_{arm_idx}"


def structure_label(idx: int, kind: str) -> str:
    """Generate a structure label with 1-based indexing.

    Used for labeling categorical (c) and graded (g) structures
    in scoring and estimation output.

    Args:
        idx: 0-based index of the structure
        kind: Kind prefix (e.g., "c" for categorical, "g" for graded)

    Returns:
        Label with 1-based index: "c1", "c2", "g1", etc.

    Example:
        >>> structure_label(0, "c")
        'c1'
        >>> structure_label(2, "g")
        'g3'
    """
    return f"{kind}{idx + 1}"
