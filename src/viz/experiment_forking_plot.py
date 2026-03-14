"""Structure forking visualization.

Shows structure compliance per arm in a tree layout:
trunk -> branches -> twigs

Each arm has its own bar chart showing all structures.
Arm label size is proportional to p(arm|parent).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from src.estimation.arm_types import (
    ArmKind,
    classify_arm,
    get_arm_color,
    get_branch_index,
)

from .viz_plot_utils import get_structure_color, save_figure


def _build_arm_tree(
    arm_names: list[str],
    arm_n_traj: dict[str, int],
    arm_suffix_probs: dict[str, float] | None = None,
) -> dict[str, dict]:
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

    if not trunk_name:
        raise ValueError("No trunk arm found, cannot build tree")

    # Build trunk node
    trunk_node = make_node(trunk_name)

    # Add branches to trunk
    for branch_name in sorted(arms_by_kind[ArmKind.BRANCH]):
        branch_node = make_node(branch_name)

        # Add twigs for this branch
        branch_idx = get_branch_index(branch_name)

        for twig_name in sorted(arms_by_kind[ArmKind.TWIG]):
            twig_branch_idx = get_branch_index(twig_name)
            if twig_branch_idx == branch_idx:
                twig_node = make_node(twig_name)
                branch_node["children"].append(twig_node)

        trunk_node["children"].append(branch_node)

    # If root exists, make it the top-level with trunk as child
    if root_name:
        root_node = make_node(root_name)
        root_node["children"] = [trunk_node]
        return root_node

    return trunk_node


def _compute_tree_layout(
    tree: dict,
    x: float = 0,
    y: float = 0,
    x_spacing: float = 1.0,
    y_spacing: float = 1.0,
) -> list[dict]:
    """Compute (x, y) positions for each node in tree layout.

    Returns list of {name, x, y, p_given_parent, children_positions}
    """
    if not tree:
        return []

    positions = []

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


def _get_arm_values(
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


def _get_display_text(
    arm_text: str,
    parent_text: str | None,
    arm_name: str,
) -> str:
    """Get display text for an arm - shows differentiating part from parent.

    For root: shows "<think>...</think>" label
    For others: shows only the part that differs from parent
    """
    import re

    text = arm_text.strip()

    # Special case for root: show a nice label for the thinking block
    if arm_name == "root":
        if "<think>" in text:
            return "<think>...</think>"
        return text[:30] if text else "root"

    # Remove <think>...</think> prefix for comparison
    text_clean = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)

    # If we have parent text, show only the difference
    if parent_text:
        parent_clean = re.sub(r"<think>.*?</think>\s*", "", parent_text, flags=re.DOTALL)
        if text_clean.startswith(parent_clean):
            diff = text_clean[len(parent_clean):].strip()
            if diff:
                return diff

    return text_clean.strip() if text_clean.strip() else text[:40]


def plot_structure_forking(
    structure_info: list[dict[str, Any]],
    arm_n_traj: dict[str, int],
    arm_texts: dict[str, str],
    output_path: Path,
    metadata: dict[str, str] | None = None,
    arm_suffix_probs: dict[str, float] | None = None,
    arm_weighted_cores: dict[str, list[float]] | None = None,
) -> Path | None:
    """Create tree-shaped structure visualization.

    Shows trunk -> branches -> twigs layout where each arm has a bar chart
    of structure compliance. Arm label size is proportional to p(arm|parent).

    Args:
        structure_info: List of structure info dicts
        arm_n_traj: Dict mapping arm name to trajectory count
        arm_texts: Dict mapping arm name to conditioning text
        output_path: Where to save the plot
        metadata: Optional dict with 'prompt', 'model', 'judge' keys
        arm_suffix_probs: P(arm_suffix | parent_prefix) from model logprobs
        arm_weighted_cores: Dict mapping arm name to weighted core values (0.0-1.0)

    Returns:
        Path to saved file, or None if insufficient data

    Raises:
        ValueError: if arm_weighted_cores is not provided
    """
    if arm_weighted_cores is None:
        raise ValueError("arm_weighted_cores is required, no silent defaults")

    if not structure_info or not arm_weighted_cores:
        return None

    arm_names = list(arm_weighted_cores.keys())

    # Build tree and layout
    tree = _build_arm_tree(arm_names, arm_n_traj, arm_suffix_probs)
    if not tree:
        return None

    # Dynamic spacing based on number of arms
    n_arms = len(arm_names)
    n_branches = sum(1 for a in arm_names if a.startswith("branch"))
    n_twigs = sum(1 for a in arm_names if a.startswith("twig"))

    # Tighter spacing for more arms, looser for fewer
    x_spacing = 3.8  # Horizontal spacing between levels
    y_spacing = max(3.5, 4.0 + 0.3 * max(0, n_twigs - 4))  # More vertical spacing for twigs

    positions = _compute_tree_layout(tree, x=0, y=0, x_spacing=x_spacing, y_spacing=y_spacing)
    if not positions:
        return None

    # Calculate figure size - TIGHT, minimal white space
    max_x = max(p["x"] for p in positions) + 3.2
    max_y = max(p["y"] for p in positions) + 2.0
    min_y = min(p["y"] for p in positions) - 1.8

    # Width proportional to tree depth, height proportional to vertical spread
    fig_width = max(22, max_x * 3.5)
    fig_height = max(14, (max_y - min_y) * 2.8)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(-1.5, max_x + 0.5)
    ax.set_ylim(min_y - 0.5, max_y + 1.0)
    ax.axis("off")

    # Structure labels for legend
    structure_labels = [s["label"] for s in structure_info]
    n_structures = len(structure_labels)

    # Build parent lookup for text diffing
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

    # Build sibling groups for per-forking-point normalization
    # Siblings are arms that share the same parent (fork from the same point)
    sibling_groups: dict[str, list[str]] = {}  # parent_name -> list of child names
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

    # Build arm_name -> normalized prob (normalized within sibling group)
    arm_normalized_probs: dict[str, float] = {}
    pos_by_name = {pos["name"]: pos for pos in positions}

    for parent, siblings in sibling_groups.items():
        sibling_probs = [pos_by_name[s]["p_given_parent"] for s in siblings if s in pos_by_name]
        if sibling_probs:
            # Single child = 100% of flow at this forking point
            if len(sibling_probs) == 1:
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
                        if range_p <= 0:
                            raise ValueError(f"Invalid range_p={range_p} for siblings, no silent defaults")
                        arm_normalized_probs[s] = (p - min_p) / range_p

    # Root always gets 1.0
    arm_normalized_probs["root"] = 1.0

    def normalize_prob(arm_name: str) -> float:
        """Get normalized probability for an arm (normalized within its sibling group)."""
        if arm_name not in arm_normalized_probs:
            raise KeyError(f"arm_normalized_probs missing '{arm_name}', no silent defaults")
        return arm_normalized_probs[arm_name]

    # Build lookup from position y to child info for line thickness
    # Use rounded keys to handle floating point precision issues
    pos_by_y: dict[float, dict] = {round(pos["y"], 6): pos for pos in positions}

    def find_pos_by_y(target_y: float) -> dict | None:
        """Find position by y with floating point tolerance."""
        rounded = round(target_y, 6)
        if rounded in pos_by_y:
            return pos_by_y[rounded]
        # Fallback: search with tolerance
        for y, pos in pos_by_y.items():
            if abs(y - target_y) < 0.001:
                return pos
        return None

    # Draw connecting lines first (so they're behind bars)
    for pos in positions:
        if "child_ys" in pos:
            parent_x = pos["x"] + 2.6  # End of parent's bar area
            parent_y = pos["y"]
            for child_y in pos["child_ys"]:
                # Find child's probability for line thickness
                child_pos = find_pos_by_y(child_y)
                if child_pos:
                    child_name = child_pos["name"]
                    norm_prob = normalize_prob(child_name)
                    line_width = 1.0 + norm_prob * 10.0  # Range: 1.0 to 11.0 (dramatic)
                    gray_val = int(180 - norm_prob * 140)  # Range: 180 to 40 (more contrast)
                    line_color = f"#{gray_val:02x}{gray_val:02x}{gray_val:02x}"
                else:
                    raise ValueError(f"Child not found at y={child_y}, no silent defaults")

                # Draw curved connector - use dynamic x_spacing
                child_x = pos["x"] + x_spacing
                mid_x = (parent_x + child_x) / 2
                ax.plot(
                    [parent_x, mid_x, mid_x, child_x - 0.3],
                    [parent_y, parent_y, child_y, child_y],
                    color=line_color,
                    linewidth=line_width,
                    solid_capstyle="round",
                    zorder=1,
                )

    # Draw each arm's bar chart - HUGE subplots
    bar_height = 0.28
    bar_width_scale = 2.5  # Scale for bar widths

    for pos in positions:
        arm_name = pos["name"]
        x = pos["x"]
        y = pos["y"]
        p_given_parent = pos["p_given_parent"]
        norm_prob = normalize_prob(arm_name)

        # Get values for this arm (uses weighted cores, same as core.png)
        values = _get_arm_values(arm_name, arm_weighted_cores, n_structures)

        # Fixed font size - same for all arms
        font_size = 24

        # Get differentiating text for this arm - SHOW ALL TEXT (no truncation)
        raw_text = arm_texts.get(arm_name, arm_name)
        parent_text = parent_texts.get(arm_name)
        display_text = _get_display_text(raw_text, parent_text, arm_name)

        # Draw arm label above bars - FULL TEXT, same size for all
        ax.text(
            x - 0.05,
            y + n_structures * bar_height / 2 + 0.4,
            display_text,
            fontsize=font_size,
            fontweight="bold",
            color="#000",
            ha="left",
            va="bottom",
            zorder=10,
        )

        # Draw arm name below the box for ALL arms
        ax.text(
            x + bar_width_scale / 2,
            y - n_structures * bar_height / 2 - 0.6,
            arm_name,
            fontsize=18,
            color="#444",
            ha="center",
            va="top",
            style="italic",
            fontweight="semibold",
            zorder=10,
        )

        # Draw horizontal bars for each structure - BIGGER
        for i, (val, label) in enumerate(zip(values, structure_labels)):
            bar_y = (
                y + (n_structures - 1 - i) * bar_height - n_structures * bar_height / 2
            )
            bar_w = val / 100 * bar_width_scale

            # Bar
            color = get_structure_color(i)
            ax.barh(
                bar_y,
                bar_w,
                height=bar_height * 0.88,
                left=x,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                zorder=5,
            )

        # Draw background box - border thickness based on probability
        box_height = n_structures * bar_height + 0.22
        # Higher prob = thicker, darker border
        border_width = 1.0 + norm_prob * 4.0  # Range: 1.0 to 5.0
        border_gray = int(170 - norm_prob * 120)  # Range: 170 to 50
        border_color = f"#{border_gray:02x}{border_gray:02x}{border_gray:02x}"
        box = FancyBboxPatch(
            (x - 0.1, y - box_height / 2 - 0.1),
            bar_width_scale + 0.25,
            box_height + 0.2,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor="#f5f5f5",
            edgecolor=border_color,
            linewidth=border_width,
            zorder=2,
        )
        ax.add_patch(box)

        # Add 50% reference line OVER the bars - VERY VISIBLE
        ref_y_min = y - box_height / 2
        ref_y_max = y + box_height / 2
        ax.plot(
            [x + 0.5 * bar_width_scale, x + 0.5 * bar_width_scale],
            [ref_y_min, ref_y_max],
            color="#333",
            linestyle="--",
            linewidth=2.5,
            zorder=10,  # OVER the bars (bars are zorder=5)
        )

    # Add legend for structures - MUCH BIGGER, moved away from corner
    legend_x = 0.04
    legend_y = 0.92

    line_height = 0.038  # More spacing
    for i, struct in enumerate(structure_info):
        desc = struct.get("description", "")

        color = get_structure_color(i)
        y_pos = legend_y - i * line_height

        # Color swatch - BIGGER
        ax.add_patch(
            plt.Rectangle(
                (legend_x, y_pos - 0.012),
                0.022,
                0.022,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="white",
                linewidth=2.0,
                clip_on=False,
            )
        )
        # Description text - BIGGER
        ax.text(
            legend_x + 0.03,
            y_pos,
            desc,
            transform=ax.transAxes,
            fontsize=24,
            fontweight="bold",
            va="center",
            color="#111",
        )

    # Add model and judge info - BOTTOM LEFT (3x bigger as requested)
    if metadata and metadata.get("judge"):
        fig.text(
            0.015, 0.055,
            "Judge LLM:",
            fontsize=24,
            fontfamily='monospace',
            verticalalignment="bottom",
            horizontalalignment="left",
            color="#666",
        )
        fig.text(
            0.095, 0.050,
            metadata['judge'],
            fontsize=36,
            fontweight='bold',
            verticalalignment="bottom",
            horizontalalignment="left",
            color="#222",
        )
    if metadata and metadata.get("model"):
        fig.text(
            0.015, 0.015,
            "Gen Model:",
            fontsize=24,
            fontfamily='monospace',
            verticalalignment="bottom",
            horizontalalignment="left",
            color="#666",
        )
        fig.text(
            0.095, 0.010,
            metadata['model'],
            fontsize=36,
            fontweight='bold',
            verticalalignment="bottom",
            horizontalalignment="left",
            color="#222",
        )

    # No title - removed as requested

    plt.tight_layout()
    save_figure(fig, output_path)
    return output_path
