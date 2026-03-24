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
from matplotlib.patches import FancyBboxPatch

from .forking_plot_renderers import (
    desaturate_color,
    draw_connecting_lines,
    draw_metadata,
    draw_reference_lines_core,
    draw_reference_lines_orientation,
    draw_wrapped_arm_label,
    populate_tree_content_tracker,
    render_legend,
)
from .forking_tree_builder import (
    build_arm_tree,
    build_parent_texts,
    build_sibling_groups,
    build_subtree,
    compute_min_y_spacing,
    compute_normalized_probs,
    compute_tree_layout,
    filter_downstream_arms,
    get_arm_values,
    get_display_text,
    get_dynamic_sizes,
    validate_tree_node_spacing,
)
from .legend_layout_engine import optimize_legend_placement
from .viz_bounding_box import BoundingBox
from .viz_plot_utils import get_structure_color, save_figure


def plot_structure_forking(
    structure_info: list[dict[str, Any]],
    arm_n_traj: dict[str, int],
    arm_texts: dict[str, str],
    output_path: Path,
    metadata: dict[str, str] | None = None,
    arm_suffix_probs: dict[str, float] | None = None,
    arm_weighted_cores: dict[str, list[float]] | None = None,
    weighting_method: str | None = None,
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
        weighting_method: Weighting method name to display (e.g., "prob", "inv-ppl")

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
    tree = build_arm_tree(arm_names, arm_n_traj, arm_suffix_probs)
    if not tree:
        return None

    n_twigs = sum(1 for a in arm_names if a.startswith("twig"))
    n_structures = len(structure_info)
    n_arms = len(arm_names)

    # Get dynamic sizes FIRST - needed for spacing calculation
    sizes = get_dynamic_sizes(n_structures, n_arms=n_arms)
    bar_height = sizes["bar_height"]
    bar_width_scale = sizes["bar_width_scale"]
    arm_label_fontsize = sizes["arm_label_fontsize"]

    # Dynamic spacing based on actual bar dimensions
    x_spacing = bar_width_scale + 1.2  # Ensure enough gap for connecting lines

    # Compute minimum y_spacing based on actual scaled bar_height to prevent collisions
    min_y_spacing = compute_min_y_spacing(n_structures, bar_height, arm_label_fontsize)

    # Additional spacing for many twigs
    twig_factor = 0.2 * max(0, n_twigs - 4)
    y_spacing = max(min_y_spacing, min_y_spacing + twig_factor)

    positions = compute_tree_layout(
        tree, x=0, y=0, x_spacing=x_spacing, y_spacing=y_spacing
    )

    # Validate tree node spacing (will assert if nodes collide)
    if positions:
        validate_tree_node_spacing(
            positions, n_structures, bar_height, y_spacing, arm_label_fontsize
        )
    if not positions:
        return None

    # Calculate tree bounds (before offset)
    raw_max_x = max(p["x"] for p in positions) + 3.0
    tree_max_y = max(p["y"] for p in positions)
    tree_min_y = min(p["y"] for p in positions)

    # Calculate tree content bounds (tight around bars and labels)
    tree_content_top = tree_max_y + n_structures * bar_height / 2 + 0.5
    tree_content_bottom = tree_min_y - n_structures * bar_height / 2 - 0.5
    tree_width = raw_max_x + bar_width_scale + 0.3

    # Populate TreeContentTracker for legend optimization (pre-offset)
    tree_content = populate_tree_content_tracker(
        positions,
        arm_texts,
        n_structures,
        bar_height,
        bar_width_scale,
        arm_label_fontsize,
        x_spacing,
    )

    # Define figure bounds for optimizer
    legend_height_estimate = n_structures * 0.25 + 1.0
    figure_bounds = BoundingBox(
        x_min=-0.1,
        y_min=tree_content_bottom,
        x_max=tree_width,
        y_max=tree_content_top + legend_height_estimate,
    )

    # Extract descriptions and optimize legend placement
    descriptions = [s.get("description", f"S{i}") for i, s in enumerate(structure_info)]
    legend_layout, legend_top_y, _ = optimize_legend_placement(
        descriptions,
        tree_content,
        figure_bounds,
        target_quadrant="top_left",
    )

    # DYNAMIC TREE OFFSET: Shift tree right based on legend width
    tree_x_offset = _compute_tree_offset(legend_layout)

    # Apply offset to all positions
    for pos in positions:
        pos["x"] += tree_x_offset

    # Recalculate tree bounds with offset
    max_x = max(p["x"] for p in positions) + 3.0
    tree_width = max_x + bar_width_scale + 0.3

    # Rebuild TreeContentTracker with offset positions
    tree_content = populate_tree_content_tracker(
        positions,
        arm_texts,
        n_structures,
        bar_height,
        bar_width_scale,
        arm_label_fontsize,
        x_spacing,
    )

    # Total plot bounds
    plot_min_y = tree_content_bottom
    content_bounds = tree_content.get_content_bounds()
    tree_top = content_bounds.y_max if content_bounds else tree_content_top
    plot_max_y = max(legend_top_y + 0.2, tree_top + 0.3)

    # Figure sizing
    content_width = tree_width
    scale = 1.5
    fig_width = max(12, content_width * scale)
    fig_height = (plot_max_y - plot_min_y) * scale

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(-0.1, content_width)
    ax.set_ylim(plot_min_y, plot_max_y)
    ax.axis("off")

    structure_labels = [s["label"] for s in structure_info]

    # Build parent lookup for text diffing
    parent_texts = build_parent_texts(arm_names, arm_texts)

    # Build sibling groups and compute normalized probabilities
    sibling_groups = build_sibling_groups(positions, arm_names)
    arm_normalized_probs = compute_normalized_probs(positions, sibling_groups)

    # Draw probability-proportional connecting lines
    draw_connecting_lines(
        ax, positions, arm_normalized_probs, bar_width_scale, x_spacing
    )

    # Draw each arm's bars
    for pos in positions:
        arm_name = pos["name"]
        x = pos["x"]
        y = pos["y"]
        norm_prob = arm_normalized_probs.get(arm_name, 0.5)

        # Get values for this arm (uses weighted cores, same as core.png)
        values = get_arm_values(arm_name, arm_weighted_cores, n_structures)

        # Get differentiating text for this arm
        raw_text = arm_texts.get(arm_name, arm_name)
        parent_text = parent_texts.get(arm_name)
        display_text = get_display_text(raw_text, parent_text, arm_name)

        # Draw arm label above bars - wrap if too wide
        max_label_width = bar_width_scale + 1.5  # Allow some overhang
        draw_wrapped_arm_label(
            ax,
            x - 0.05,
            y + n_structures * bar_height / 2 + 0.3,
            display_text,
            fontsize=arm_label_fontsize,
            max_width=max_label_width,
        )

        # Draw arm name below the box
        arm_name_fontsize = max(12, arm_label_fontsize - 4)
        ax.text(
            x + bar_width_scale / 2,
            y - n_structures * bar_height / 2 - 0.4,
            arm_name,
            fontsize=arm_name_fontsize,
            color="#444",
            ha="center",
            va="top",
            style="italic",
            fontweight="semibold",
            zorder=10,
        )

        # Draw horizontal bars for each structure
        for i, (val, label) in enumerate(zip(values, structure_labels)):
            bar_y = (
                y + (n_structures - 1 - i) * bar_height - n_structures * bar_height / 2
            )
            bar_w = val / 100 * bar_width_scale

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
        border_width = 1.0 + norm_prob * 4.0
        border_gray = int(170 - norm_prob * 120)
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

        # Draw reference lines at 0%, 50%, 100%
        draw_reference_lines_core(ax, x, y, box_height, bar_width_scale)

    # Draw legend and metadata
    render_legend(ax, legend_layout, legend_top_y)

    # Add weighting method label in top-left corner (use figure coords since axis is off)
    if weighting_method:
        fig.text(
            0.01,
            0.99,
            f"[{weighting_method}]",
            fontsize=11,
            fontweight="bold",
            fontfamily="monospace",
            color="#555",
            ha="left",
            va="top",
            zorder=200,
        )

    draw_metadata(fig, metadata, n_structures)

    save_figure(fig, output_path)
    return output_path


def plot_orientation_tree(
    structure_info: list[dict[str, Any]],
    arm_n_traj: dict[str, int],
    arm_texts: dict[str, str],
    output_path: Path,
    reference_arm: str,
    arm_orientations: dict[str, list[float]],
    metadata: dict[str, str] | None = None,
    arm_suffix_probs: dict[str, float] | None = None,
) -> Path | None:
    """Create tree-shaped orientation visualization.

    Like plot_structure_forking but shows orientation vectors relative to
    a reference arm instead of core values.

    Args:
        structure_info: List of structure info dicts
        arm_n_traj: Dict mapping arm name to trajectory count
        arm_texts: Dict mapping arm name to conditioning text
        output_path: Where to save the plot
        reference_arm: Name of reference arm (e.g., "trunk", "branch_1")
        arm_orientations: Dict mapping arm name to orientation vector (-1 to 1)
        metadata: Optional dict with 'prompt', 'model', 'judge' keys
        arm_suffix_probs: P(arm_suffix | parent_prefix) from model logprobs

    Returns:
        Path to saved file, or None if insufficient data
    """
    if not structure_info or not arm_orientations:
        return None

    all_arm_names = list(arm_orientations.keys())

    # Filter to only show relevant subtree (reference arm + downstream)
    arm_names = filter_downstream_arms(reference_arm, all_arm_names)
    if len(arm_names) < 2:
        return None

    # Filter the data dicts to only include relevant arms
    filtered_n_traj = {k: v for k, v in arm_n_traj.items() if k in arm_names}
    filtered_suffix_probs = (
        {k: v for k, v in arm_suffix_probs.items() if k in arm_names}
        if arm_suffix_probs
        else None
    )
    filtered_orientations = {
        k: v for k, v in arm_orientations.items() if k in arm_names
    }
    filtered_arm_texts = {k: v for k, v in arm_texts.items() if k in arm_names}

    # Build tree and layout
    tree = build_subtree(
        reference_arm, arm_names, filtered_n_traj, filtered_suffix_probs
    )
    if not tree:
        return None

    n_twigs = sum(1 for a in arm_names if a.startswith("twig"))
    n_structures = len(structure_info)
    n_arms = len(arm_names)

    # Get dynamic sizes FIRST - needed for spacing calculation
    sizes = get_dynamic_sizes(n_structures, n_arms=n_arms)
    bar_height = sizes["bar_height"]
    bar_width_scale = sizes["bar_width_scale"]
    arm_label_fontsize = sizes["arm_label_fontsize"]

    # Dynamic spacing based on actual bar dimensions
    x_spacing = bar_width_scale + 1.2  # Ensure enough gap for connecting lines

    # Compute minimum y_spacing based on actual scaled bar_height to prevent collisions
    min_y_spacing = compute_min_y_spacing(n_structures, bar_height, arm_label_fontsize)

    # Additional spacing for many twigs (orientation plots need slightly more)
    twig_factor = 0.3 * max(0, n_twigs - 4)
    y_spacing = max(min_y_spacing, min_y_spacing + twig_factor)

    positions = compute_tree_layout(
        tree, x=0, y=0, x_spacing=x_spacing, y_spacing=y_spacing
    )
    if not positions:
        return None

    # Validate tree node spacing (will assert if nodes collide)
    validate_tree_node_spacing(
        positions, n_structures, bar_height, y_spacing, arm_label_fontsize
    )

    # Calculate tree bounds (before offset)
    raw_max_x = max(p["x"] for p in positions) + 3.2
    tree_max_y = max(p["y"] for p in positions)
    tree_min_y = min(p["y"] for p in positions)

    # Calculate tree content bounds
    tree_content_top = tree_max_y + n_structures * bar_height / 2 + 0.5
    tree_content_bottom = tree_min_y - n_structures * bar_height / 2 - 0.5
    tree_width = raw_max_x + bar_width_scale + 0.3

    # Populate TreeContentTracker for legend optimization (pre-offset)
    tree_content = populate_tree_content_tracker(
        positions,
        filtered_arm_texts,
        n_structures,
        bar_height,
        bar_width_scale,
        arm_label_fontsize,
        x_spacing,
    )

    # Add title to tracker (orientation plots have a title between tree and legend)
    title_height = 0.32
    tree_content.set_title(
        x=0,
        y=tree_content_top,
        width=tree_width * 0.6,
        height=title_height,
    )

    # Define figure bounds for optimizer
    legend_max_height = 8.0
    figure_bounds = BoundingBox(
        x_min=-0.1,
        y_min=tree_content_bottom,
        x_max=tree_width,
        y_max=tree_content_top + title_height + legend_max_height,
    )

    # Extract descriptions and optimize legend placement
    descriptions = [s.get("description", f"S{i}") for i, s in enumerate(structure_info)]
    legend_layout, legend_top_y, _ = optimize_legend_placement(
        descriptions,
        tree_content,
        figure_bounds,
        target_quadrant="top_left",
    )

    # DYNAMIC TREE OFFSET: Shift tree right based on legend width
    tree_x_offset = _compute_tree_offset(legend_layout)

    # Apply offset to all positions
    for pos in positions:
        pos["x"] += tree_x_offset

    # Recalculate tree bounds with offset
    max_x = max(p["x"] for p in positions) + 3.2
    tree_width = max_x + bar_width_scale + 0.3

    # Rebuild TreeContentTracker with offset positions
    tree_content = populate_tree_content_tracker(
        positions,
        filtered_arm_texts,
        n_structures,
        bar_height,
        bar_width_scale,
        arm_label_fontsize,
        x_spacing,
    )

    # Re-add title with offset
    tree_content.set_title(
        x=tree_x_offset,
        y=tree_content_top,
        width=tree_width * 0.6,
        height=title_height,
    )

    # Total plot bounds
    plot_min_y = tree_content_bottom
    content_bounds = tree_content.get_content_bounds()
    tree_top = content_bounds.y_max if content_bounds else tree_content_top
    plot_max_y = max(legend_top_y + 0.2, tree_top + 0.3)

    # Figure sizing
    content_width = tree_width
    scale = 1.5
    fig_width = max(12, content_width * scale)
    fig_height = (plot_max_y - plot_min_y) * scale

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(-0.1, content_width)
    ax.set_ylim(plot_min_y, plot_max_y)
    ax.axis("off")

    structure_labels = [s["label"] for s in structure_info]

    # Build parent lookup for text diffing
    parent_texts = build_parent_texts(arm_names, arm_texts)

    # Build sibling groups and normalized probabilities for line drawing
    sibling_groups = build_sibling_groups(positions, arm_names)
    arm_normalized_probs = compute_normalized_probs(positions, sibling_groups)

    # Draw probability-proportional connecting lines
    draw_connecting_lines(
        ax, positions, arm_normalized_probs, bar_width_scale, x_spacing
    )

    # Draw each arm's orientation bars (centered at 0)
    for pos in positions:
        arm_name = pos["name"]
        x = pos["x"]
        y = pos["y"]

        if arm_name not in filtered_orientations:
            continue
        orientation = filtered_orientations[arm_name]

        # Get display text
        raw_text = filtered_arm_texts.get(arm_name, arm_name)
        parent_text = parent_texts.get(arm_name)
        display_text = get_display_text(raw_text, parent_text, arm_name)

        # Draw arm label above bars - wrap if too wide
        max_label_width = bar_width_scale + 1.5  # Allow some overhang
        draw_wrapped_arm_label(
            ax,
            x - 0.05,
            y + n_structures * bar_height / 2 + 0.3,
            display_text,
            fontsize=arm_label_fontsize,
            max_width=max_label_width,
        )

        # Draw arm name below the box
        arm_name_fontsize = max(12, arm_label_fontsize - 4)
        ax.text(
            x + bar_width_scale / 2,
            y - n_structures * bar_height / 2 - 0.4,
            arm_name,
            fontsize=arm_name_fontsize,
            color="#444",
            ha="center",
            va="top",
            style="italic",
            fontweight="semibold",
            zorder=10,
        )

        # Draw horizontal bars for each structure (centered, diverging)
        center_x = x + bar_width_scale / 2
        for i, (val, label) in enumerate(zip(orientation, structure_labels)):
            bar_y = (
                y + (n_structures - 1 - i) * bar_height - n_structures * bar_height / 2
            )
            bar_w = val * (bar_width_scale / 2)

            # Color: normal for positive, desaturated for negative
            if val >= 0:
                color = get_structure_color(i)
            else:
                base_color = get_structure_color(i)
                color = desaturate_color(base_color, 0.5)

            # Draw bar from center
            if bar_w >= 0:
                ax.barh(
                    bar_y,
                    bar_w,
                    height=bar_height * 0.88,
                    left=center_x,
                    color=color,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=5,
                )
            else:
                ax.barh(
                    bar_y,
                    abs(bar_w),
                    height=bar_height * 0.88,
                    left=center_x + bar_w,
                    color=color,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=5,
                    hatch="//",
                    alpha=0.7,
                )

        # Draw background box
        box_height = n_structures * bar_height + 0.22
        box = FancyBboxPatch(
            (x - 0.1, y - box_height / 2 - 0.1),
            bar_width_scale + 0.25,
            box_height + 0.2,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor="#f5f5f5",
            edgecolor="#999",
            linewidth=1.5,
            zorder=2,
        )
        ax.add_patch(box)

        # Draw reference lines at -100%, -50%, 0%, 50%, 100%
        draw_reference_lines_orientation(ax, center_x, y, box_height, bar_width_scale)

    # Draw legend
    render_legend(ax, legend_layout, legend_top_y)

    # Add title (below legend, above tree) - ALL CAPS, ROBOTO MONO style
    ax.text(
        content_width / 2,
        tree_content_top + 0.4,  # Higher up
        f"ORIENTATION RELATIVE TO {reference_arm.upper()}",
        fontsize=14,
        fontweight="bold",
        fontfamily="monospace",  # Roboto Mono fallback
        ha="center",
        va="bottom",
        color="#333",
        zorder=100,
    )

    draw_metadata(fig, metadata, n_structures)

    save_figure(fig, output_path)
    return output_path


def _compute_tree_offset(legend_layout: dict[str, Any]) -> float:
    """Compute tree X offset based on legend width.

    Places tree to the right of the legend for balanced layout.
    """
    legend_items = legend_layout.get("items", [])
    if not legend_items:
        return 0

    legend_width = legend_layout.get("total_width", 0)
    if legend_width == 0:
        char_width = legend_layout.get("char_width", 0.055)
        legend_width = max(
            item["text_x"] + len(item["description"]) * char_width * 0.7
            for item in legend_items
        )

    return legend_width + 0.2  # Tight gap between legend and tree
