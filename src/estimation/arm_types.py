"""Centralized arm type handling for estimation pipeline.

Defines arm classification, ordering, and display utilities to replace
hardcoded string checks throughout the codebase.

Arm Ordering:
    root (arm_idx=0) -> trunk (arm_idx=1) -> branches (arm_idx=2+) -> twigs (N+)

Reference Core:
    Trunk remains the reference for branch orientation metrics.
    Root gets its own metrics computed independently.
"""

from __future__ import annotations

from enum import Enum


class ArmKind(Enum):
    """Classification of arm types in the estimation pipeline."""

    ROOT = "root"
    TRUNK = "trunk"
    BRANCH = "branch"
    TWIG = "twig"  # Sub-variation of a branch


# Color scheme for arms - use distinct colors for visibility
ARM_COLORS = {
    "root": "#8E44AD",  # Purple (baseline before trunk)
    "trunk": "#2980B9",  # Blue (baseline)
}

# Colors for branches (indexed by branch number - 1)
BRANCH_COLORS = [
    "#4A90D9",  # blue
    "#E67E22",  # orange
    "#2ECC71",  # green
    "#9B59B6",  # purple
    "#E74C3C",  # red
    "#1ABC9C",  # teal
    "#F39C12",  # yellow
    "#8E44AD",  # violet
]

# Twig colors are lighter variants of branch colors
TWIG_COLORS = [
    "#7FB8E8",  # light blue
    "#F5A962",  # light orange
    "#7FD68B",  # light green
    "#C28BD6",  # light purple
    "#F28B8B",  # light red
    "#5ED4C8",  # light teal
    "#F7C64E",  # light yellow
    "#A968C4",  # light violet
]


def classify_arm(name: str) -> ArmKind:
    """Classify an arm name into its ArmKind.

    Args:
        name: Arm name (e.g., "root", "trunk", "branch_1", "twig_1_b1")

    Returns:
        The corresponding ArmKind enum value
    """
    if name == "root":
        return ArmKind.ROOT
    elif name == "trunk":
        return ArmKind.TRUNK
    elif name.startswith("twig_"):
        return ArmKind.TWIG
    elif name.startswith("branch_"):
        return ArmKind.BRANCH
    else:
        # Default to branch for unknown names
        return ArmKind.BRANCH


def is_twig(name: str) -> bool:
    """Check if an arm is a twig (sub-variation of a branch).

    Args:
        name: Arm name

    Returns:
        True if the arm is a twig
    """
    return classify_arm(name) == ArmKind.TWIG


def get_parent_branch(name: str) -> str | None:
    """Get the parent branch name for a twig.

    Args:
        name: Arm name (e.g., "twig_2_b1")

    Returns:
        Parent branch name (e.g., "branch_1") or None if not a twig
    """
    if not name.startswith("twig_"):
        return None
    # "twig_2_b1" -> extract branch number from "_bN" suffix
    try:
        branch_num = int(name.rsplit("_b", 1)[1])
        return f"branch_{branch_num}"
    except (ValueError, IndexError):
        return None


def get_twig_index(name: str) -> int | None:
    """Get the twig index from a twig arm name.

    Args:
        name: Arm name (e.g., "twig_2_b1")

    Returns:
        Twig index (1-based) or None if not a twig
    """
    if not name.startswith("twig_"):
        return None
    try:
        # "twig_2_b1" -> "2"
        parts = name.split("_")
        return int(parts[1])
    except (ValueError, IndexError):
        return None


def get_branch_index(name: str) -> int | None:
    """Get the branch index from a branch or twig arm name.

    Args:
        name: Arm name (e.g., "branch_1" or "twig_2_b1")

    Returns:
        Branch index (1-based) or None if not a branch/twig
    """
    if name.startswith("branch_"):
        try:
            parts = name.split("_")
            return int(parts[1])
        except (ValueError, IndexError):
            return None
    elif name.startswith("twig_"):
        # "twig_2_b1" -> extract branch number from "_bN" suffix
        try:
            return int(name.rsplit("_b", 1)[1])
        except (ValueError, IndexError):
            return None
    return None


def is_baseline_arm(name: str) -> bool:
    """Check if an arm is a baseline arm (root or trunk).

    Baseline arms are generated independently rather than as variations.

    Args:
        name: Arm name

    Returns:
        True if the arm is root or trunk
    """
    kind = classify_arm(name)
    return kind in (ArmKind.ROOT, ArmKind.TRUNK)


def is_reference_arm(name: str) -> bool:
    """Check if an arm is the reference arm for orientation metrics.

    The trunk arm serves as the reference for computing branch orientations
    and deviances. Root is not a reference - it's compared to trunk like branches.

    Args:
        name: Arm name

    Returns:
        True if the arm is trunk (the reference arm)
    """
    return classify_arm(name) == ArmKind.TRUNK


def get_arm_color(name: str) -> str:
    """Get the color for an arm.

    Args:
        name: Arm name (e.g., "root", "trunk", "branch_1", "branch_1_twig_2")

    Returns:
        Hex color string

    Raises:
        ValueError: If the arm name cannot be mapped to a color.
    """
    if name in ARM_COLORS:
        return ARM_COLORS[name]

    kind = classify_arm(name)

    # For twigs, use lighter color based on parent branch
    if kind == ArmKind.TWIG:
        branch_idx = get_branch_index(name)
        if branch_idx is not None:
            return TWIG_COLORS[(branch_idx - 1) % len(TWIG_COLORS)]
        raise ValueError(
            f"Cannot determine branch index for twig '{name}'. "
            "Expected format: twig_N_bM"
        )

    # For branches, extract index and use BRANCH_COLORS
    if kind == ArmKind.BRANCH:
        branch_idx = get_branch_index(name)
        if branch_idx is not None:
            return BRANCH_COLORS[(branch_idx - 1) % len(BRANCH_COLORS)]
        raise ValueError(
            f"Cannot determine branch index for branch '{name}'. "
            "Expected format: branch_N"
        )

    raise ValueError(
        f"Cannot determine color for arm '{name}' with kind {kind}. "
        f"Expected one of: root, trunk, branch_N, or twig_N_bM"
    )


def get_arm_sort_key(name: str) -> tuple[int, int, int, str]:
    """Get a sort key for ordering arms consistently.

    Ordering: root (0) -> trunk (1) -> branch_1, twig_1_b1, twig_2_b1, ...
              -> branch_2, twig_1_b2, ...

    Args:
        name: Arm name

    Returns:
        Tuple for sorting (category, branch_idx, twig_idx, name)
    """
    kind = classify_arm(name)
    if kind == ArmKind.ROOT:
        return (0, 0, 0, name)
    elif kind == ArmKind.TRUNK:
        return (1, 0, 0, name)
    elif kind == ArmKind.TWIG:
        branch_idx = get_branch_index(name) or 0
        twig_idx = get_twig_index(name) or 0
        return (2, branch_idx, twig_idx, name)
    else:  # BRANCH
        branch_idx = get_branch_index(name) or 0
        return (2, branch_idx, 0, name)  # Branches have twig_idx=0 to come first


def get_arm_name_from_index(arm_idx: int) -> str:
    """Convert an arm index to its canonical string name.

    Arm index convention (used during generation):
    - 0 = root (prompt only, no trunk)
    - 1 = trunk (prompt + trunk text)
    - 2+ = branch_N (where N = arm_idx - 1)

    Args:
        arm_idx: Integer index from trajectory generation

    Returns:
        Canonical arm name ("root", "trunk", "branch_1", etc.)
    """
    if arm_idx == 0:
        return "root"
    elif arm_idx == 1:
        return "trunk"
    else:
        return f"branch_{arm_idx - 1}"


def get_display_name(name: str) -> str:
    """Get a display-friendly name for an arm.

    Args:
        name: Arm name

    Returns:
        Display name (e.g., "ROOT", "TRUNK", "BRANCH_1", "ALL_ARMS")
    """
    return name.upper()


def get_short_display_name(name: str) -> str:
    """Get a short display name for table headers.

    Args:
        name: Arm name

    Returns:
        Short but clear display name (e.g., "root", "trunk", "br1", "tw1_b1")
    """
    if name == "root":
        return "root"
    elif name == "trunk":
        return "trunk"
    elif name.startswith("twig_"):
        # "twig_2_b1" -> "tw2_b1"
        twig_idx = get_twig_index(name)
        branch_idx = get_branch_index(name)
        return f"tw{twig_idx}_b{branch_idx}"
    elif name.startswith("branch_"):
        idx = name.replace("branch_", "")
        return f"br{idx}"
    return name


def sort_arm_names(names: list[str]) -> list[str]:
    """Sort arm names in canonical order.

    Args:
        names: List of arm names

    Returns:
        Sorted list with root first, trunk second, branches/twigs in order
    """
    return sorted(names, key=get_arm_sort_key)


def get_ordered_arms_for_plotting(arm_names: list[str]) -> list[str]:
    """Get arms in proper order for plotting.

    Args:
        arm_names: Available arm names

    Returns:
        Ordered list suitable for plotting
    """
    return sort_arm_names(arm_names)


def get_arm_ancestry(name: str) -> list[str]:
    """Get the conditioning ancestry for an arm.

    Returns list of arm names from root to this arm, representing
    the conditioning hierarchy path.

    Args:
        name: Arm name

    Returns:
        List of arm names from root to this arm

    Examples:
        "root" -> ["root"]
        "trunk" -> ["root", "trunk"]
        "branch_2" -> ["root", "trunk", "branch_2"]
        "twig_1_b2" -> ["root", "trunk", "branch_2", "twig_1_b2"]
    """
    kind = classify_arm(name)
    if kind == ArmKind.ROOT:
        return ["root"]
    elif kind == ArmKind.TRUNK:
        return ["root", "trunk"]
    elif kind == ArmKind.BRANCH:
        return ["root", "trunk", name]
    elif kind == ArmKind.TWIG:
        parent = get_parent_branch(name)
        if parent:
            return ["root", "trunk", parent, name]
        return ["root", "trunk", name]
    return [name]


def get_downstream_arms(reference_name: str, all_arm_names: list[str]) -> list[str]:
    """Get arms that are downstream from a reference arm in the conditioning hierarchy.

    Hierarchy: root -> trunk -> branches -> twigs

    Downstream means arms that are conditioned on MORE text than the reference:
    - root: trunk, branches, twigs are downstream
    - trunk: branches, twigs are downstream
    - branch_N: only twigs of that branch are downstream
    - twig: nothing downstream (terminal node)

    Args:
        reference_name: The reference arm name
        all_arm_names: All available arm names

    Returns:
        List of arm names that are downstream from the reference
    """
    ref_kind = classify_arm(reference_name)

    downstream = []
    for name in all_arm_names:
        if name == reference_name:
            continue

        kind = classify_arm(name)

        # root: everything downstream (trunk, branches, twigs)
        if ref_kind == ArmKind.ROOT:
            if kind in (ArmKind.TRUNK, ArmKind.BRANCH, ArmKind.TWIG):
                downstream.append(name)

        # trunk: branches and twigs are downstream
        elif ref_kind == ArmKind.TRUNK:
            if kind in (ArmKind.BRANCH, ArmKind.TWIG):
                downstream.append(name)

        # branch: only its own twigs are downstream
        elif ref_kind == ArmKind.BRANCH:
            if kind == ArmKind.TWIG:
                # Check if this twig belongs to this branch
                parent = get_parent_branch(name)
                if parent == reference_name:
                    downstream.append(name)

        # twig: nothing is downstream (terminal node)
        # elif ref_kind == ArmKind.TWIG: pass

    return downstream


def has_downstream_arms(reference_name: str, all_arm_names: list[str]) -> bool:
    """Check if an arm has any downstream arms.

    Args:
        reference_name: The arm to check
        all_arm_names: All available arm names

    Returns:
        True if there are downstream arms
    """
    return len(get_downstream_arms(reference_name, all_arm_names)) > 0
