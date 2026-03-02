"""Tree plotting and layout."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.common.log import log
from .trees import TreeNode, build_tree

COLORS = ["#FFB3BA", "#BAFFC9", "#BAE1FF", "#FFFFBA", "#FFDFba", "#E0BBE4"]

# Text part types for styled rendering
TEXT_PART_TEMPLATE = "template"  # Chat template prefix/suffix - gray, smaller
TEXT_PART_PROMPT = "prompt"  # User's prompt - normal
TEXT_PART_CONTINUATION = "continuation"  # Model continuation - gray


def visualize_experiment(
    result_dir: Path,
    max_viz_samples_per_branch_point: int | None = None,
) -> None:
    """Visualize a single experiment's results.

    Expects folder structure: result_dir/{model}/{branch}/gen.json + est.json
    Outputs viz to: result_dir/{model}/{branch}/viz/
    """
    # Walk all gen.json files in the nested structure
    for gen_file in sorted(result_dir.glob("**/gen.json")):
        branch_dir = gen_file.parent
        rel = branch_dir.relative_to(result_dir)
        log(f"  {rel}/")

        with open(gen_file) as f:
            data = json.load(f)

        trajectories = data["trajectories"]
        prompt = data.get("prompt_text", "<root>")
        formatted_prompt = data.get("formatted_prompt", prompt)
        model = data.get("model", "unknown")
        model_short = model.rsplit("/", 1)[-1]
        branch_name = branch_dir.name

        if not trajectories:
            log("    No trajectories")
            continue

        # Filter to top N trajectories by probability per branch point
        if max_viz_samples_per_branch_point is not None:
            trajectories = sorted(
                trajectories, key=lambda t: t["probability"], reverse=True
            )[:max_viz_samples_per_branch_point]
            log(f"    Filtered to top {len(trajectories)} by probability")

        scores, structures = _load_scores(branch_dir, trajectories)
        greedy_traj = next((t for t in trajectories if t.get("is_greedy")), None)
        full_prefix = formatted_prompt
        greedy_parts = None
        if greedy_traj:
            greedy_text = greedy_traj["text"]
            greedy_parts = _build_text_parts(prompt, formatted_prompt, greedy_text)

        viz_dir = branch_dir / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)

        for mode in ["word", "phrase", "token"]:
            tree = build_tree(trajectories, scores, full_prefix, mode)
            if tree:
                plot_tree(
                    tree,
                    f"{mode.title()} Tree — {model_short} — {branch_name}",
                    viz_dir / f"{mode}_tree.png",
                    structures,
                    scores,
                    greedy_parts=greedy_parts,
                )
            elif mode == "token":
                log("    Skipping token tree (no token data)")


def _load_scores(
    branch_dir: Path, trajectories: list[dict]
) -> tuple[dict[str, list[float]], list[str]]:
    """Load structure scores from estimation file."""
    scores, structures = {}, []
    est_file = branch_dir / "est.json"

    if est_file.exists():
        with open(est_file) as f:
            est = json.load(f)
        if est.get("systems"):
            sys_data = est["systems"][0]
            structures = [s["structure"] for s in sys_data["structures"]]
            num_scores = len(sys_data["structures"][0]["scores"]) if structures else 0
            for i, t in enumerate(trajectories):
                if i < num_scores:
                    scores[t["text"]] = [s["scores"][i] for s in sys_data["structures"]]
        log(f"    {len(structures)} structures, {len(scores)}/{len(trajectories)} scored")

    return scores, structures


def _build_text_parts(
    prompt: str, formatted_prompt: str, continuation: str
) -> list[tuple[str, str]]:
    """
    Build list of (text, part_type) tuples for styled rendering.

    Splits formatted_prompt into template prefix, user prompt, template suffix,
    then adds continuation.
    """
    parts = []

    # Find where the user prompt appears in the formatted prompt
    prompt_start = formatted_prompt.find(prompt)

    if prompt_start == -1:
        # Prompt not found in formatted - just return the whole thing as template
        parts.append((formatted_prompt, TEXT_PART_TEMPLATE))
        parts.append((continuation, TEXT_PART_CONTINUATION))
        return parts

    # Template prefix (before user prompt)
    if prompt_start > 0:
        prefix = formatted_prompt[:prompt_start]
        parts.append((prefix, TEXT_PART_TEMPLATE))

    # User prompt
    parts.append((prompt, TEXT_PART_PROMPT))

    # Template suffix (after user prompt)
    prompt_end = prompt_start + len(prompt)
    if prompt_end < len(formatted_prompt):
        suffix = formatted_prompt[prompt_end:]
        parts.append((suffix, TEXT_PART_TEMPLATE))

    # Model continuation
    parts.append((continuation, TEXT_PART_CONTINUATION))

    return parts


def plot_tree(
    root: TreeNode,
    title: str,
    path: Path,
    structures: list[str],
    scores: dict[str, list[float]],
    greedy_parts: list[tuple[str, str]] | None = None,
) -> None:
    """Render tree to PNG."""
    # Use dynamic layout based on tree structure
    has_scores = bool(structures and scores)
    layout_info = _layout_tree(root, has_scores=has_scores)
    edges, nodes = _collect(root)

    if not nodes:
        return

    all_x = [pos[0] for _, pos in nodes]
    all_y = [pos[1] for _, pos in nodes]

    # Dynamic figure dimensions based on actual tree extent
    x_range = max(all_x) - min(all_x) if all_x else 1
    y_range = max(all_y) - min(all_y) if all_y else 1

    # Scale figure size to tree dimensions
    tree_width = max(10, x_range * 1.2 + 4)
    tree_height = max(4, y_range * 0.9 + 2)

    # Add space for header (title + styled text)
    header_height = 1.2 if greedy_parts else 0.6
    # Legend on the right side
    legend_width = 3.0 if structures else 0
    total_width = tree_width + legend_width
    total_height = tree_height + header_height

    fig = plt.figure(figsize=(total_width, total_height))

    # Create grid: header spans top, tree + legend on bottom row
    if structures:
        gs = fig.add_gridspec(
            2,
            2,
            height_ratios=[header_height, tree_height],
            width_ratios=[tree_width, legend_width],
            hspace=0.05,
            wspace=0.02,
        )
        ax_header = fig.add_subplot(gs[0, :])
        ax_tree = fig.add_subplot(gs[1, 0])
        ax_legend = fig.add_subplot(gs[1, 1])
    else:
        gs = fig.add_gridspec(
            2, 1, height_ratios=[header_height, tree_height], hspace=0.05
        )
        ax_header = fig.add_subplot(gs[0])
        ax_tree = fig.add_subplot(gs[1])
        ax_legend = None

    # Draw header (title + styled text)
    ax_header.axis("off")
    ax_header.text(
        0.5,
        0.7,
        title,
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="center",
        transform=ax_header.transAxes,
    )
    if greedy_parts:
        _draw_styled_text(ax_header, greedy_parts, y_start=0.4)

    # Draw tree
    ax_tree.axis("off")
    _draw_edges(ax_tree, edges)
    _draw_nodes(ax_tree, nodes, scores, structures, layout_info["x_sp"])

    # Calculate axis limits with padding for labels
    x_range = max(all_x) - min(all_x) if len(set(all_x)) > 1 else 1
    y_range = max(all_y) - min(all_y) if len(set(all_y)) > 1 else 1

    # Extra right margin for leaf labels (positioned to right of nodes)
    max_label_chars = layout_info["metrics"]["max_label_len"]
    right_margin = min(max_label_chars, 20) * 0.12 + 1.0

    ax_tree.set_xlim(min(all_x) - 0.5, max(all_x) + right_margin)
    ax_tree.set_ylim(min(all_y) - 0.6, max(all_y) + 0.5)

    # Draw legend in dedicated area
    if structures and ax_legend:
        _draw_legend(ax_legend, structures)

    try:
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        log(f"  Saved: {path}")
    except ValueError as e:
        log(f"  Skipped: {path.name} ({e})")
    plt.close()


def _draw_styled_text(
    ax,
    parts: list[tuple[str, str]],
    y_start: float = 0.5,
) -> None:
    """Draw styled text on two lines: prompt (grey) and continuation (bold)."""
    # Escape newlines and dollar signs (matplotlib interprets $ as LaTeX)
    parts = [
        (text.replace("\n", "↵").replace("$", "\\$"), part_type)
        for text, part_type in parts
    ]

    # Separate prompt (template + prompt) from continuation
    prompt_text = "".join(
        text for text, ptype in parts if ptype != TEXT_PART_CONTINUATION
    )
    continuation_text = "".join(
        text for text, ptype in parts if ptype == TEXT_PART_CONTINUATION
    )

    # Scale font sizes based on text length
    def calc_fontsize(text: str, base: float, base_chars: int) -> float:
        return max(5, min(base, base * base_chars / max(len(text), 1)))

    prompt_fontsize = calc_fontsize(prompt_text, 9, 120)
    continuation_fontsize = calc_fontsize(continuation_text, 11, 80)

    # Draw prompt line (grey, lighter)
    if prompt_text:
        ax.text(
            0.5,
            y_start,
            prompt_text,
            fontfamily="monospace",
            fontsize=prompt_fontsize,
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#999999",
            alpha=0.7,
        )

    # Draw continuation line (bold, darker)
    if continuation_text:
        ax.text(
            0.5,
            y_start - 0.4,
            continuation_text,
            fontfamily="monospace",
            fontsize=continuation_fontsize,
            fontweight="bold",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#333333",
        )


def _compute_tree_metrics(node: TreeNode) -> dict:
    """Compute tree metrics for dynamic spacing."""

    def _recurse(n, depth):
        if n.is_leaf():
            return {
                "max_depth": depth,
                "total_leaves": 1,
                "max_label_len": len(n.label),
                "nodes_at_depth": {depth: 1},
            }

        metrics = {
            "max_depth": depth,
            "total_leaves": 0,
            "max_label_len": len(n.label),
            "nodes_at_depth": {depth: 1},
        }

        for child in n.children.values():
            child_metrics = _recurse(child, depth + 1)
            metrics["max_depth"] = max(metrics["max_depth"], child_metrics["max_depth"])
            metrics["total_leaves"] += child_metrics["total_leaves"]
            metrics["max_label_len"] = max(
                metrics["max_label_len"], child_metrics["max_label_len"]
            )
            for d, count in child_metrics["nodes_at_depth"].items():
                metrics["nodes_at_depth"][d] = (
                    metrics["nodes_at_depth"].get(d, 0) + count
                )

        return metrics

    return _recurse(node, 0)


def _node_vertical_footprint(node: TreeNode, has_scores: bool) -> float:
    """Calculate vertical space needed for a node (node + label + scores)."""
    # Base height for node circle + label
    height = 1.0
    # Extra space for scores on leaf nodes
    if node.is_leaf() and has_scores:
        height += 0.3
    # Extra space for greedy star
    if node.is_greedy:
        height += 0.2
    return height


def _layout_recursive(
    node: TreeNode,
    depth: int,
    y_start: float,
    x_sp: float,
    min_y_gap: float,
    has_scores: bool,
) -> float:
    """
    Layout tree ensuring no vertical overlap.

    Returns the y-coordinate just below this subtree (for sibling placement).
    """
    node.x = depth * x_sp

    if node.is_leaf():
        footprint = _node_vertical_footprint(node, has_scores)
        node.y = y_start + footprint / 2
        return y_start + footprint + min_y_gap

    # Layout children from bottom to top, accumulating their y positions
    children = sorted(node.children.values(), key=lambda c: c.prob)
    curr_y = y_start
    child_positions = []

    for child in children:
        next_y = _layout_recursive(
            child, depth + 1, curr_y, x_sp, min_y_gap, has_scores
        )
        child_positions.append(child.y)
        curr_y = next_y

    # Parent centered among children
    node.y = (child_positions[0] + child_positions[-1]) / 2
    return curr_y


def _layout_tree(root: TreeNode, has_scores: bool = False) -> dict:
    """
    Layout tree with guaranteed non-overlapping placement.

    Uses content-aware spacing: each node reserves vertical space for its
    label and optional scores. Horizontal spacing adapts to label length.
    """
    metrics = _compute_tree_metrics(root)
    max_label_len = metrics["max_label_len"]
    total_leaves = metrics["total_leaves"]
    max_depth = metrics["max_depth"]

    # Horizontal spacing based on label length (chars -> plot units)
    # ~8 chars per unit at fontsize 8 monospace
    label_width = min(max_label_len, 18) * 0.12
    x_sp = max(3.0, label_width + 1.5)

    # Cap total width to avoid enormous images (e.g. token trees with 100+ levels)
    max_width_units = 300
    if max_depth > 0 and x_sp * max_depth > max_width_units:
        x_sp = max_width_units / max_depth

    # Minimum vertical gap between nodes
    # Larger trees need proportionally more gap to stay readable
    base_gap = 0.4
    density_factor = 1.0 + (total_leaves / 30) * 0.3
    min_y_gap = base_gap * min(density_factor, 2.0)

    _layout_recursive(root, 0, 0.0, x_sp, min_y_gap, has_scores)

    return {"x_sp": x_sp, "min_y_gap": min_y_gap, "metrics": metrics}


def _collect(node: TreeNode) -> tuple[list, list]:
    """Collect edges and nodes for plotting."""
    edges, nodes = [], [(node, (node.x, node.y))]

    if node.children:
        total = sum(c.prob for c in node.children.values())
        for child in node.children.values():
            edges.append(((node.x, node.y), (child.x, child.y), child, total))
            ce, cn = _collect(child)
            edges.extend(ce)
            nodes.extend(cn)

    return edges, nodes


def _draw_edges(ax, edges: list) -> list:
    """
    Draw tree edges with probability labels.

    Uses bezier curves to avoid crossing node labels. Returns list of
    edge label text objects for optional collision adjustment.
    """
    edge_texts = []

    for (x1, y1), (x2, y2), child, total in edges:
        rel = child.prob / total if total else 0.5

        ax.plot(
            [x1, x2],
            [y1, y2],
            color="#888",
            linewidth=0.5 + rel * 3,
            alpha=0.7,
            zorder=1,
        )

        if child.has_true_prob:
            # Position label on the edge, offset slightly to avoid node overlap
            # Place at 40% along edge (closer to parent) to avoid child label
            label_x = x1 + (x2 - x1) * 0.35
            label_y = y1 + (y2 - y1) * 0.35

            txt = ax.annotate(
                _fmt_prob(child.prob),
                (label_x, label_y),
                fontsize=6,
                ha="center",
                va="center",
                color="#555",
                bbox={
                    "boxstyle": "round,pad=0.08",
                    "facecolor": "white",
                    "alpha": 0.9,
                    "edgecolor": "none",
                },
                zorder=3,
            )
            edge_texts.append(txt)

    return edge_texts


def _draw_nodes(
    ax, nodes: list, scores: dict, structures: list[str], x_sp: float
) -> None:
    """
    Draw tree nodes with labels and structure annotations.

    Labels are positioned to avoid edge crossings:
    - Leaf nodes: label to the RIGHT (no outgoing edges)
    - Internal nodes: label BELOW (edges go right, label doesn't interfere)

    Nodes are colored by their dominant structure (core). Leaf nodes show
    the structure name below the label. Greedy nodes get a star marker.
    """
    node_radius = 0.15
    n_structs = len(structures)

    for node, (x, y) in nodes:
        # Color by dominant structure (uses core for branching nodes)
        color = _node_color(node, scores, n_structs)
        node_size = 50 + node.count * 25

        ax.scatter(
            [x],
            [y],
            s=node_size,
            c=[color],
            alpha=0.9,
            edgecolors="black",
            linewidths=0.8,
            zorder=2,
        )

        # Show full label with proportional font size
        label = node.label.replace("\n", "↵").replace("$", "\\$")
        label_len = len(label)

        # For very long labels, wrap into multiple lines and left-align
        if label_len > 60:
            wrap_width = 40
            lines = [label[i : i + wrap_width] for i in range(0, label_len, wrap_width)]
            label = "\n".join(lines)
            label_fontsize = max(4, min(6, 6 * 80 / label_len))
            h_align = "left"
            label_x_pos = x
        else:
            # Scale font: base 9 for ~15 chars, scale down for longer
            label_fontsize = max(5, min(9, 9 * 15 / max(label_len, 1)))
            h_align = "center"
            label_x_pos = x

        # Label BELOW node
        ax.annotate(
            label,
            (label_x_pos, y - node_radius - 0.1),
            fontsize=label_fontsize,
            ha=h_align,
            va="top",
            fontfamily="monospace",
            zorder=4,
            linespacing=1.2,
        )

        label_x = x + node_radius + 0.1
        if node.is_leaf():
            # Leaf: label to the RIGHT of node (no outgoing edges to cross)
            # Greedy star above the node
            if node.is_greedy:
                ax.annotate(
                    "★",
                    (x, y + 0.3),
                    fontsize=10,
                    ha="center",
                    va="bottom",
                    color="#DAA520",
                    zorder=4,
                )
            # Structure name below label
            if node.scores and structures:
                _draw_structure_tag(ax, label_x, y, node.scores, structures)


def _draw_structure_tag(
    ax, label_x: float, y: float, node_scores: list[float], structures: list[str]
) -> None:
    """Draw structure scores as colored values in a horizontal row."""
    if not node_scores or not structures:
        return

    # Position to the right of the node
    tag_x = label_x + 0.8
    col_width = 0.55

    # Draw each score in its structure's color, laid out horizontally
    for i, score in enumerate(node_scores):
        if i >= len(structures):
            break
        color = _darken(COLORS[i % len(COLORS)])

        ax.annotate(
            f"{score:.2f}",
            (tag_x + i * col_width, y),
            fontsize=6,
            ha="left",
            va="center",
            fontfamily="monospace",
            color=color,
            fontweight="bold",
            zorder=4,
        )


def _draw_legend(ax, structures: list[str]) -> None:
    """Draw structure legend vertically on the right side with full labels."""
    ax.axis("off")

    n = len(structures)
    if n == 0:
        return

    # Vertical layout: one item per row, top-aligned
    item_height = 1.0 / max(n + 1, 2)
    start_y = 1.0 - item_height

    for i, structure in enumerate(structures):
        x = 0.05
        y = start_y - i * item_height
        color = COLORS[i % len(COLORS)]

        # Draw colored box
        ax.add_patch(
            plt.Rectangle(
                (x, y - 0.015),
                0.04,
                0.03,
                fc=color,
                ec="black",
                lw=0.5,
                transform=ax.transAxes,
                clip_on=False,
            )
        )

        # Draw full label
        ax.text(
            x + 0.06,
            y,
            structure,
            fontsize=7,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )


def _node_color(node: TreeNode, scores: dict, n_structs: int) -> str:
    """Get node color based on dominant structure."""
    if node.is_leaf() and node.scores:
        return COLORS[int(np.argmax(node.scores)) % len(COLORS)]

    if node.is_branching() and scores and n_structs:
        core = _compute_core(node, scores, n_structs)
        if core:
            return COLORS[int(np.argmax(core)) % len(COLORS)]

    return "#DDDDDD"


def _compute_core(node: TreeNode, scores: dict, n: int) -> list[float] | None:
    """Compute probability-weighted core at a node."""
    if not node.traj_probs:
        return None

    total, weighted = 0.0, [0.0] * n
    for text, prob in node.traj_probs:
        if text in scores:
            total += prob
            for i, s in enumerate(scores[text]):
                weighted[i] += prob * s

    return [w / total for w in weighted] if total else None


def _fmt_prob(p: float) -> str:
    return f"{p:.1e}" if p < 0.01 else f"{p:.2f}"


def _darken(hex_color: str) -> str:
    rgb = bytes.fromhex(hex_color[1:])
    return f"#{int(rgb[0] * 0.6):02x}{int(rgb[1] * 0.6):02x}{int(rgb[2] * 0.6):02x}"
