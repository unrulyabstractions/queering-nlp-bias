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


# Color scheme for arms - ITERATION 10 (FINAL)
# Refined: Elegant, research-grade palette with clear hierarchy
#
# Design principles:
# 1. Root/trunk: Sophisticated muted neutrals (baseline tones)
# 2. Branches: Rich, saturated colors that are distinct but harmonious
# 3. Twigs: twig_1 stays close to parent (warm shift), twig_2 clearly lighter (cool shift)

ARM_COLORS = {
    "root": "#9590A8",  # Dusty lavender-gray (baseline)
    "trunk": "#5585A0",  # Steel blue (baseline)
}

# Branch colors - Elegant, distinct, well-spaced on color wheel
BRANCH_COLORS = [
    "#C85555",  # Muted coral (branch 1) - warm
    "#358872",  # Sage teal (branch 2) - cool
    "#7565B5",  # Soft iris (branch 3) - cool
    "#D58535",  # Amber (branch 4) - warm
    "#3888A5",  # Cerulean (branch 5) - cool
    "#A56585",  # Dusty rose (branch 6) - warm
    "#5A9545",  # Forest (branch 7) - cool
    "#C555A0",  # Orchid (branch 8) - warm
]

# Twig configs: (lightness, hue_shift, saturation_boost)
# twig_1: barely lighter, warm shift - stays close to parent
# twig_2: clearly lighter, cool shift - distinct but related
TWIG_CONFIGS = [
    (0.18, 25, 0.05),   # twig_1: subtle change, warm
    (0.45, -55, 0.15),  # twig_2: clear change, cool
    (0.22, 38, 0.07),   # twig_3
    (0.50, -70, 0.18),  # twig_4
    (0.25, 45, 0.10),   # twig_5
    (0.58, -85, 0.22),  # twig_6
]


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return (
        int(hex_color[:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB tuple to hex color."""
    return f"#{r:02x}{g:02x}{b:02x}"


def _rgb_to_hsl(r: int, g: int, b: int) -> tuple[float, float, float]:
    """Convert RGB (0-255) to HSL (h: 0-360, s: 0-1, l: 0-1)."""
    r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
    max_c = max(r_norm, g_norm, b_norm)
    min_c = min(r_norm, g_norm, b_norm)
    delta = max_c - min_c

    # Lightness
    lightness = (max_c + min_c) / 2.0

    if delta == 0:
        return (0.0, 0.0, lightness)

    # Saturation
    if lightness < 0.5:
        saturation = delta / (max_c + min_c)
    else:
        saturation = delta / (2.0 - max_c - min_c)

    # Hue
    if max_c == r_norm:
        hue = 60.0 * (((g_norm - b_norm) / delta) % 6)
    elif max_c == g_norm:
        hue = 60.0 * (((b_norm - r_norm) / delta) + 2)
    else:
        hue = 60.0 * (((r_norm - g_norm) / delta) + 4)

    return (hue, saturation, lightness)


def _hsl_to_rgb(h: float, s: float, l: float) -> tuple[int, int, int]:
    """Convert HSL (h: 0-360, s: 0-1, l: 0-1) to RGB (0-255)."""
    if s == 0:
        val = int(l * 255)
        return (val, val, val)

    def hue_to_rgb(p: float, q: float, t: float) -> float:
        if t < 0:
            t += 1
        if t > 1:
            t -= 1
        if t < 1 / 6:
            return p + (q - p) * 6 * t
        if t < 1 / 2:
            return q
        if t < 2 / 3:
            return p + (q - p) * (2 / 3 - t) * 6
        return p

    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q
    h_norm = h / 360.0

    r = hue_to_rgb(p, q, h_norm + 1 / 3)
    g = hue_to_rgb(p, q, h_norm)
    b = hue_to_rgb(p, q, h_norm - 1 / 3)

    return (int(r * 255), int(g * 255), int(b * 255))


def _shift_hue(hex_color: str, degrees: float) -> str:
    """Shift the hue of a color by given degrees."""
    r, g, b = _hex_to_rgb(hex_color)
    h, s, l = _rgb_to_hsl(r, g, b)
    h = (h + degrees) % 360
    r, g, b = _hsl_to_rgb(h, s, l)
    return _rgb_to_hex(r, g, b)


def _lighten_color(
    hex_color: str, factor: float = 0.5, saturation_boost: float = 0.0
) -> str:
    """Lighten a hex color by increasing lightness in HSL space.

    Args:
        hex_color: Color in hex format (e.g., "#4A90D9")
        factor: Blend factor (0 = original, 1 = white)
        saturation_boost: Amount to boost saturation (0 = none, counteracts washing out)

    Returns:
        Lightened hex color
    """
    r, g, b = _hex_to_rgb(hex_color)
    h, s, l = _rgb_to_hsl(r, g, b)
    # Increase lightness
    new_l = l + (1.0 - l) * factor
    # Preserve saturation better (don't desaturate as much) + optional boost
    new_s = s * (1.0 - factor * 0.15) + saturation_boost
    new_s = min(1.0, max(0.0, new_s))  # Clamp to [0, 1]
    r, g, b = _hsl_to_rgb(h, new_s, new_l)
    return _rgb_to_hex(r, g, b)


def classify_arm(name: str) -> ArmKind:
    """Classify an arm name into its ArmKind.

    Args:
        name: Arm name (e.g., "root", "trunk", "branch_1", "twig_b1_2")

    Returns:
        The corresponding ArmKind enum value
    """
    if name == "root":
        return ArmKind.ROOT
    elif name == "trunk":
        return ArmKind.TRUNK
    elif name.startswith("twig_b"):
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
        name: Arm name (e.g., "twig_b1_2")

    Returns:
        Parent branch name (e.g., "branch_1") or None if not a twig
    """
    if not name.startswith("twig_b"):
        return None
    # "twig_b1_2" -> extract branch number after "twig_b"
    try:
        # Split: "twig_b1_2" -> ["twig", "b1", "2"]
        parts = name.split("_")
        branch_num = int(parts[1][1:])  # "b1" -> 1
        return f"branch_{branch_num}"
    except (ValueError, IndexError):
        return None


def get_twig_index(name: str) -> int | None:
    """Get the twig index from a twig arm name.

    Args:
        name: Arm name (e.g., "twig_b1_2")

    Returns:
        Twig index (1-based) or None if not a twig
    """
    if not name.startswith("twig_b"):
        return None
    try:
        # "twig_b1_2" -> "2" (last part)
        parts = name.split("_")
        return int(parts[2])
    except (ValueError, IndexError):
        return None


def get_branch_index(name: str) -> int | None:
    """Get the branch index from a branch or twig arm name.

    Args:
        name: Arm name (e.g., "branch_1" or "twig_b1_2")

    Returns:
        Branch index (1-based) or None if not a branch/twig
    """
    if name.startswith("branch_"):
        try:
            parts = name.split("_")
            return int(parts[1])
        except (ValueError, IndexError):
            return None
    elif name.startswith("twig_b"):
        # "twig_b1_2" -> extract branch number from "bN"
        try:
            parts = name.split("_")
            return int(parts[1][1:])  # "b1" -> 1
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

    Root, trunk, and each branch have distinct pastel colors. Twigs inherit
    their parent branch's color but are lighter and hue-shifted for visual
    differentiation while maintaining family resemblance.

    Args:
        name: Arm name (e.g., "root", "trunk", "branch_1", "twig_b1_2")

    Returns:
        Hex color string

    Raises:
        ValueError: If the arm name cannot be mapped to a color.
    """
    if name in ARM_COLORS:
        return ARM_COLORS[name]

    kind = classify_arm(name)

    # For twigs: paired lightness + hue shift + saturation boost
    if kind == ArmKind.TWIG:
        branch_idx = get_branch_index(name)
        twig_idx = get_twig_index(name)
        if branch_idx is not None and twig_idx is not None:
            # Get parent branch color
            branch_color = BRANCH_COLORS[(branch_idx - 1) % len(BRANCH_COLORS)]
            # Get config for this twig (lightness, hue_shift, sat_boost)
            lighten_factor, hue_shift, sat_boost = TWIG_CONFIGS[
                (twig_idx - 1) % len(TWIG_CONFIGS)
            ]
            # Apply hue shift first, then lighten with saturation boost
            shifted_color = _shift_hue(branch_color, hue_shift)
            return _lighten_color(shifted_color, lighten_factor, sat_boost)
        raise ValueError(
            f"Cannot determine branch/twig index for twig '{name}'. "
            "Expected format: twig_bN_M"
        )

    # For branches, extract index and use BRANCH_COLORS.
    # Flat/CSV arms don't follow branch_N format, so fall back to a hash-based index.
    if kind == ArmKind.BRANCH:
        branch_idx = get_branch_index(name)
        if branch_idx is None:
            branch_idx = (hash(name) % len(BRANCH_COLORS)) + 1
        return BRANCH_COLORS[(branch_idx - 1) % len(BRANCH_COLORS)]

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
        Short but clear display name (e.g., "root", "trunk", "br1", "tw_b1_2")
    """
    if name == "root":
        return "root"
    elif name == "trunk":
        return "trunk"
    elif name.startswith("twig_b"):
        # "twig_b1_2" -> "tw_b1_2"
        return name.replace("twig_", "tw_")
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
        "twig_b2_1" -> ["root", "trunk", "branch_2", "twig_b2_1"]
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
