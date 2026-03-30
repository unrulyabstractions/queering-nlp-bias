"""Tree data structures for forking visualizations.

Builds tree representations of arm hierarchies:
- root -> trunk -> branches -> twigs

Pure data processing - no matplotlib code.
"""

from __future__ import annotations

import re
from typing import Any

from src.estimation.arm_types import (
    ArmKind,
    classify_arm,
    get_branch_index,
)


def build_arm_tree(
    arm_names: list[str],
    arm_n_traj: dict[str, int],
    arm_suffix_probs: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build a tree structure from arm names.

    Returns dict with:
        - name: arm name
        - n_traj: trajectory count
        - children: list of child arm dicts
        - p_given_parent: p(arm|parent) from model logprobs

    Raises:
        KeyError: if arm_suffix_probs is missing required arms
    """
    if arm_suffix_probs is None:
        raise ValueError("arm_suffix_probs is required, no silent defaults")

    # Build lookup
    arms_by_kind: dict[ArmKind, list[str]] = {
        ArmKind.ROOT: [],
        ArmKind.TRUNK: [],
        ArmKind.BRANCH: [],
        ArmKind.TWIG: [],
    }
    for name in arm_names:
        kind = classify_arm(name)
        arms_by_kind[kind].append(name)

    # Build tree
    def make_node(name: str) -> dict:
        if name not in arm_n_traj:
            raise KeyError(f"arm_n_traj missing '{name}', no silent defaults")
        if name not in arm_suffix_probs:
            raise KeyError(f"arm_suffix_probs missing '{name}', no silent defaults")
        return {
            "name": name,
            "n_traj": arm_n_traj[name],
            "p_given_parent": arm_suffix_probs[name],
            "children": [],
        }

    # Start with root if it exists
    root_name = arms_by_kind[ArmKind.ROOT][0] if arms_by_kind[ArmKind.ROOT] else None
    trunk_name = arms_by_kind[ArmKind.TRUNK][0] if arms_by_kind[ArmKind.TRUNK] else None

    def _make_branch_nodes() -> list[dict]:
        """Build branch nodes (each with their twig children)."""
        nodes = []
        for branch_name in sorted(arms_by_kind[ArmKind.BRANCH]):
            branch_node = make_node(branch_name)
            branch_idx = get_branch_index(branch_name)
            for twig_name in sorted(arms_by_kind[ArmKind.TWIG]):
                if get_branch_index(twig_name) == branch_idx:
                    branch_node["children"].append(make_node(twig_name))
            nodes.append(branch_node)
        return nodes

    # When there is a trunk, attach branches to it
    if trunk_name:
        trunk_node = make_node(trunk_name)
        trunk_node["children"] = _make_branch_nodes()
        if root_name:
            root_node = make_node(root_name)
            root_node["children"] = [trunk_node]
            return root_node
        return trunk_node

    # No trunk — branches attach directly to root
    if not root_name:
        raise ValueError("No root or trunk arm found, cannot build tree")
    root_node = make_node(root_name)
    root_node["children"] = _make_branch_nodes()
    return root_node


def compute_tree_layout(
    tree: dict,
    x: float = 0,
    y: float = 0,
    x_spacing: float = 1.0,
    y_spacing: float = 1.0,
) -> list[dict]:
    """Compute (x, y) positions for each node in tree layout.

    Returns list of {name, x, y, p_given_parent, n_traj, child_ys}
    """
    if not tree:
        return []

    positions: list[dict] = []

    def layout_node(node: dict, x: float, y: float, depth: int) -> float:
        """Layout a node and its children, return total height used."""
        children = node.get("children", [])

        if not children:
            # Leaf node
            positions.append({
                "name": node["name"],
                "x": x,
                "y": y,
                "p_given_parent": node["p_given_parent"],
                "n_traj": node["n_traj"],
            })
            return y_spacing

        # Layout children first to get total height
        # Reverse children so branch_1 is at TOP (lower index = higher y position)
        child_x = x + x_spacing
        child_y = y
        total_height = 0

        child_positions = []
        for child in reversed(children):
            child_height = layout_node(child, child_x, child_y, depth + 1)
            child_positions.append(child_y + child_height / 2 - y_spacing / 2)
            child_y += child_height
            total_height += child_height

        # Position this node in center of its children
        node_y = y + total_height / 2 - y_spacing / 2
        positions.append({
            "name": node["name"],
            "x": x,
            "y": node_y,
            "p_given_parent": node["p_given_parent"],
            "n_traj": node["n_traj"],
            "child_ys": child_positions,
        })

        return total_height

    layout_node(tree, x, y, 0)
    return positions


def get_arm_values(
    arm_name: str,
    arm_weighted_cores: dict[str, list[float]],
    n_structures: int,
) -> list[float]:
    """Get structure compliance values for an arm from weighted cores.

    Uses weighted cores from estimation (same values as core.png).

    Returns:
        List of compliance percentages (0-100).

    Raises:
        KeyError: if arm not found in arm_weighted_cores.
    """
    if arm_name not in arm_weighted_cores:
        raise KeyError(f"arm_weighted_cores missing '{arm_name}', no silent defaults")

    core = arm_weighted_cores[arm_name]
    if len(core) != n_structures:
        raise ValueError(
            f"Core length {len(core)} != n_structures {n_structures} for '{arm_name}'"
        )

    # Convert from 0.0-1.0 to percentage 0-100
    return [val * 100 for val in core]


def build_sibling_groups(
    positions: list[dict],
    arm_names: list[str],
) -> dict[str, list[str]]:
    """Build sibling groups for per-forking-point normalization.

    Siblings are arms that share the same parent (fork from the same point).
    """
    sibling_groups: dict[str, list[str]] = {}
    for pos in positions:
        arm_name = pos["name"]
        kind = classify_arm(arm_name)
        if kind == ArmKind.ROOT:
            parent = None
        elif kind == ArmKind.TRUNK:
            parent = "root"
        elif kind == ArmKind.BRANCH:
            parent = "trunk"
        elif kind == ArmKind.TWIG:
            branch_idx = get_branch_index(arm_name)
            parent = f"branch_{branch_idx}"
        else:
            parent = None

        if parent:
            if parent not in sibling_groups:
                sibling_groups[parent] = []
            sibling_groups[parent].append(arm_name)

    return sibling_groups


def compute_normalized_probs(
    positions: list[dict],
    sibling_groups: dict[str, list[str]],
) -> dict[str, float]:
    """Compute normalized probabilities within each sibling group.

    Returns dict mapping arm name to normalized probability (0.0 to 1.0).
    """
    pos_by_name = {pos["name"]: pos for pos in positions}
    arm_normalized_probs: dict[str, float] = {}

    for parent, siblings in sibling_groups.items():
        sibling_probs = [
            pos_by_name[s]["p_given_parent"]
            for s in siblings if s in pos_by_name
        ]
        if not sibling_probs:
            continue

        if len(sibling_probs) == 1:
            # Single child = 100% of flow
            for s in siblings:
                if s in pos_by_name:
                    arm_normalized_probs[s] = 1.0
        else:
            # Multiple siblings: normalize within group
            min_p = min(sibling_probs)
            max_p = max(sibling_probs)
            range_p = max_p - min_p if max_p > min_p else 1.0
            for s in siblings:
                if s in pos_by_name:
                    p = pos_by_name[s]["p_given_parent"]
                    arm_normalized_probs[s] = (p - min_p) / range_p

    # Root always gets 1.0
    arm_normalized_probs["root"] = 1.0

    return arm_normalized_probs


def build_parent_texts(
    arm_names: list[str],
    arm_texts: dict[str, str],
) -> dict[str, str | None]:
    """Build parent text lookup for display text diffing."""
    parent_texts: dict[str, str | None] = {}
    root_text = arm_texts.get("root")
    trunk_text = arm_texts.get("trunk")

    for name in arm_names:
        kind = classify_arm(name)
        if kind == ArmKind.ROOT:
            parent_texts[name] = None
        elif kind == ArmKind.TRUNK:
            parent_texts[name] = root_text
        elif kind == ArmKind.BRANCH:
            parent_texts[name] = trunk_text
        elif kind == ArmKind.TWIG:
            branch_idx = get_branch_index(name)
            parent_texts[name] = arm_texts.get(f"branch_{branch_idx}")
        else:
            parent_texts[name] = None

    return parent_texts


def filter_downstream_arms(reference_arm: str, all_arms: list[str]) -> list[str]:
    """Filter to only include reference arm and its downstream arms.

    For root: include all arms
    For trunk: include trunk + branches + twigs
    For branch_N: include branch_N + its twigs only
    """
    ref_kind = classify_arm(reference_arm)

    if ref_kind == ArmKind.ROOT:
        return all_arms
    elif ref_kind == ArmKind.TRUNK:
        # Include trunk, all branches, all twigs
        return [a for a in all_arms if classify_arm(a) != ArmKind.ROOT]
    elif ref_kind == ArmKind.BRANCH:
        # Include this branch + its twigs only
        branch_idx = get_branch_index(reference_arm)
        result = [reference_arm]
        for a in all_arms:
            if classify_arm(a) == ArmKind.TWIG:
                twig_branch_idx = get_branch_index(a)
                if twig_branch_idx == branch_idx:
                    result.append(a)
        return result
    else:
        # Twig has no downstream
        return [reference_arm]


def build_subtree(
    reference_arm: str,
    arm_names: list[str],
    arm_n_traj: dict[str, int],
    arm_suffix_probs: dict[str, float] | None = None,
) -> dict | None:
    """Build a subtree rooted at reference_arm.

    For branch_N: returns branch_N with its twigs as children
    For trunk: returns trunk with branches as children
    For root: returns full tree
    """
    if arm_suffix_probs is None:
        raise ValueError("arm_suffix_probs is required")

    ref_kind = classify_arm(reference_arm)

    def make_node(name: str) -> dict:
        if name not in arm_n_traj:
            raise KeyError(f"arm_n_traj missing '{name}'")
        if name not in arm_suffix_probs:
            raise KeyError(f"arm_suffix_probs missing '{name}'")
        return {
            "name": name,
            "n_traj": arm_n_traj[name],
            "p_given_parent": arm_suffix_probs[name],
            "children": [],
        }

    if ref_kind == ArmKind.ROOT:
        # Full tree - use existing function
        return build_arm_tree(arm_names, arm_n_traj, arm_suffix_probs)

    if ref_kind == ArmKind.TRUNK:
        # Trunk with branches as children
        trunk_node = make_node("trunk")
        branches = sorted([a for a in arm_names if classify_arm(a) == ArmKind.BRANCH])
        for branch_name in branches:
            branch_node = make_node(branch_name)
            # Add twigs for this branch
            branch_idx = get_branch_index(branch_name)
            for twig_name in sorted(arm_names):
                if classify_arm(twig_name) == ArmKind.TWIG:
                    if get_branch_index(twig_name) == branch_idx:
                        branch_node["children"].append(make_node(twig_name))
            trunk_node["children"].append(branch_node)
        return trunk_node

    if ref_kind == ArmKind.BRANCH:
        # Branch with its twigs as children
        branch_node = make_node(reference_arm)
        branch_idx = get_branch_index(reference_arm)
        for twig_name in sorted(arm_names):
            if classify_arm(twig_name) == ArmKind.TWIG:
                if get_branch_index(twig_name) == branch_idx:
                    branch_node["children"].append(make_node(twig_name))
        return branch_node

    # Twig - just itself
    return make_node(reference_arm)


def get_display_text(
    arm_text: str,
    parent_text: str | None,
    arm_name: str,
) -> str:
    """Get display text for an arm - shows differentiating part from parent.

    For root: shows "<think>...</think>" label
    For others: shows only the part that differs from parent
    If no difference, shows empty string (arm name is shown separately)
    """
    text = arm_text.strip()

    # Special case for root: show a nice label for the thinking block
    if arm_name == "root":
        if "<think>" in text:
            return "<think>...</think>"
        return text[:30] if text else ""

    # Remove <think>...</think> prefix for comparison
    text_clean = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)

    # If we have parent text, show only the difference
    if parent_text:
        parent_clean = re.sub(r"<think>.*?</think>\s*", "", parent_text, flags=re.DOTALL)
        if text_clean.startswith(parent_clean):
            diff = text_clean[len(parent_clean):].strip()
            # Return diff even if empty - arm name shown separately
            return diff

    # No parent or doesn't start with parent - show cleaned text
    return text_clean.strip()


def get_dynamic_sizes(
    n_structures: int,
    n_arms: int = 0,
) -> dict[str, Any]:
    """Get dynamic sizes for bars, fonts, and legend based on structure count and tree size.

    Args:
        n_structures: Number of structures (determines bar height)
        n_arms: Total number of arms in tree (for scaling up larger trees)
    """
    # Base sizes by structure count - LARGER baseline fonts
    if n_structures <= 6:
        base = {
            "bar_height": 0.36,
            "bar_width_scale": 3.2,
            "arm_label_fontsize": 24,
            "legend_fontsize": 12,
            "legend_line_height": 0.024,
        }
    elif n_structures <= 10:
        base = {
            "bar_height": 0.30,
            "bar_width_scale": 2.8,
            "arm_label_fontsize": 20,
            "legend_fontsize": 10,
            "legend_line_height": 0.020,
        }
    else:
        base = {
            "bar_height": 0.24,
            "bar_width_scale": 2.5,
            "arm_label_fontsize": 16,
            "legend_fontsize": 9,
            "legend_line_height": 0.016,
        }

    # Scale UP for larger trees (many branches/twigs)
    # More aggressive scaling for better readability
    if n_arms >= 8:
        scale_factor = 1.5  # 50% bigger for large trees
        base["bar_height"] *= scale_factor
        base["bar_width_scale"] *= scale_factor
        base["arm_label_fontsize"] = int(base["arm_label_fontsize"] * scale_factor)
    elif n_arms >= 4:
        scale_factor = 1.35  # 35% bigger for medium trees (lowered threshold)
        base["bar_height"] *= scale_factor
        base["bar_width_scale"] *= scale_factor
        base["arm_label_fontsize"] = int(base["arm_label_fontsize"] * scale_factor)

    return base


def compute_min_y_spacing(
    n_structures: int,
    bar_height: float,
    arm_label_fontsize: float,
) -> float:
    """Compute minimum y_spacing to prevent tree node collisions.

    Tree nodes consist of:
    - Box: height = n_structures * bar_height + 0.22
    - Label above: at y + n_structures * bar_height / 2 + 0.3 + text_height
    - Arm name below: at y - n_structures * bar_height / 2 - 0.4 - text_height

    For adjacent nodes at y and y + y_spacing to not collide:
    - Lower node's label top must be below upper node's name bottom
    - y_spacing >= label_offset + name_offset + text_heights

    Args:
        n_structures: Number of structures (determines box height)
        bar_height: Height of each bar (from get_dynamic_sizes)
        arm_label_fontsize: Font size for arm labels (affects text height)

    Returns:
        Minimum y_spacing to prevent node collisions
    """
    # Box extent from node center
    box_extent = n_structures * bar_height / 2

    # Label positioned above box
    label_offset = box_extent + 0.3

    # Arm name positioned below box
    name_offset = box_extent + 0.4

    # Text height estimates (proportional to fontsize)
    # At fontsize 36, text is roughly 0.5 units tall
    label_text_height = arm_label_fontsize * 0.014
    name_text_height = (arm_label_fontsize - 4) * 0.012

    # Minimum spacing = distance from lower node's label top to upper node's name bottom
    min_spacing = label_offset + label_text_height + name_offset + name_text_height

    # Add margin for visual clarity
    margin = 0.3

    return min_spacing + margin


def validate_tree_node_spacing(
    positions: list[dict],
    n_structures: int,
    bar_height: float,
    y_spacing: float,
    arm_label_fontsize: float = 24,
) -> None:
    """Assert that tree nodes don't collide vertically.

    Groups nodes by x-level and checks that adjacent nodes have sufficient spacing.

    Args:
        positions: List of node position dicts from compute_tree_layout
        n_structures: Number of structures
        bar_height: Height of each bar
        y_spacing: Spacing used in layout
        arm_label_fontsize: Font size for arm labels (for text height estimation)

    Raises:
        AssertionError: If any nodes at the same x-level would collide
    """
    # Group positions by x coordinate (same level in tree)
    levels: dict[float, list[dict]] = {}
    for pos in positions:
        x_key = round(pos["x"], 2)
        if x_key not in levels:
            levels[x_key] = []
        levels[x_key].append(pos)

    # Calculate node visual height using same formula as compute_min_y_spacing
    box_extent = n_structures * bar_height / 2
    label_offset = box_extent + 0.3
    name_offset = box_extent + 0.4

    # Text height estimates (proportional to fontsize) - same as compute_min_y_spacing
    label_text_height = arm_label_fontsize * 0.014
    name_text_height = (arm_label_fontsize - 4) * 0.012

    # Total extent above and below node center
    node_top_extent = label_offset + label_text_height
    node_bottom_extent = name_offset + name_text_height

    # Check each level for overlaps
    for x_level, level_positions in levels.items():
        if len(level_positions) < 2:
            continue

        sorted_by_y = sorted(level_positions, key=lambda p: p["y"])
        for i in range(len(sorted_by_y) - 1):
            lower_node = sorted_by_y[i]
            upper_node = sorted_by_y[i + 1]

            lower_top = lower_node["y"] + node_top_extent
            upper_bottom = upper_node["y"] - node_bottom_extent
            gap = upper_bottom - lower_top

            MIN_NODE_GAP = 0.1  # Minimum visual gap between nodes
            assert gap >= MIN_NODE_GAP, (
                f"Tree node collision at x={x_level:.1f}: "
                f"'{lower_node['name']}' (y_top={lower_top:.2f}) overlaps with "
                f"'{upper_node['name']}' (y_bottom={upper_bottom:.2f}), "
                f"gap={gap:.2f} < {MIN_NODE_GAP}. "
                f"y_spacing={y_spacing:.2f} is too small for n_structures={n_structures}, "
                f"bar_height={bar_height:.2f}"
            )
