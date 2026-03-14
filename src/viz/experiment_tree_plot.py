"""Tree visualization for generation trajectories.

Visualizes the trajectory tree from generation output, showing
branching structure, probabilities, and structure scores.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath

from src.common.continuation_text import get_continuation_text

from .viz_plot_utils import STRUCTURE_COLORS, lighten_color

# Suppress matplotlib glyph warnings (emojis, special unicode)
warnings.filterwarnings("ignore", message="Glyph .* missing from current font")

if TYPE_CHECKING:
    from src.estimation.estimation_experiment_types import EstimationResult

# Color palette - softer, more modern colors
PALETTE = {
    "bg": "#FAFAFA",
    "edge": "#A0A0A0",
    "edge_highlight": "#5080C0",
    "node_default": "#E8E8E8",
    "node_border": "#666666",
    "text": "#333333",
    "text_light": "#666666",
    "greedy_star": "#E6A800",
    "grid": "#EEEEEE",
}


@dataclass
class VizTreeNode:
    """Node in the visualization tree."""

    label: str
    prob: float = 1.0
    log_prob: float = 0.0
    cumulative_log_prob: float = 0.0  # Chained log probability from root
    children: dict[str, VizTreeNode] = field(default_factory=dict)
    count: int = 1
    is_leaf: bool = False
    is_greedy: bool = False
    scores: list[float] | None = None
    aggregated_scores: list[float] | None = None  # Aggregated from children
    traj_idx: int | None = None
    x: float = 0.0
    y: float = 0.0

    def is_branching(self) -> bool:
        """Return True if this node has multiple children."""
        return len(self.children) > 1


def sanitize_label(text: str, max_len: int = 25) -> str:
    """Clean up label text for display.

    Removes/replaces problematic characters and truncates.
    """
    # Replace newlines
    text = text.replace("\n", " ").replace("\r", "")

    # Remove or replace emojis and special unicode
    # Keep basic ASCII + common punctuation
    cleaned = []
    for char in text:
        if ord(char) < 128 or char in "àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ":
            cleaned.append(char)
        elif ord(char) > 0x1F600:  # Emoji range
            cleaned.append("·")  # Replace emoji with dot
        else:
            cleaned.append(char)
    text = "".join(cleaned)

    # Truncate
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"

    return text.strip()


def load_generation_data(gen_path: Path) -> dict | None:
    """Load generation output JSON."""
    if not gen_path.exists():
        return None
    with open(gen_path) as f:
        return json.load(f)


def load_structure_labels(estimation_path: Path) -> list[str]:
    """Load structure labels from estimation output."""
    if not estimation_path.exists():
        return []

    with open(estimation_path) as f:
        data = json.load(f)

    labels = []
    # Try both field names for compatibility
    structure_info = data.get("structures", data.get("structure_info", []))
    for info in structure_info:
        labels.append(info.get("label", f"s{info.get('idx', 0)}"))

    return labels


def load_trajectory_scores(estimation_path: Path) -> dict[int, list[float]]:
    """Load per-trajectory structure scores from estimation output."""
    if not estimation_path.exists():
        return {}

    with open(estimation_path) as f:
        data = json.load(f)

    scores_by_traj = {}
    for arm in data.get("arms", []):
        for traj in arm.get("trajectories", []):
            traj_idx = traj.get("traj_idx", 0)
            orientation = traj.get("orientation", [])
            if orientation:
                scores_by_traj[traj_idx] = orientation

    return scores_by_traj


def extract_trajectories_from_gen_data(gen_data: dict) -> list[dict]:
    """Extract trajectory list from generation data."""
    # Try tree.trajs first (forking-paths, seeking-entropy)
    tree = gen_data.get("tree", {})
    if "trajs" in tree:
        trajs = tree["trajs"]
        return [
            {
                "text": get_continuation_text(t),
                "traj_idx": t.get("traj_idx", i),
                "probability": 1.0,
                "log_probability": sum(t.get("logprobs", [0.0])),
            }
            for i, t in enumerate(trajs)
        ]

    # Try top-level trajectories (simple-sampling)
    if "trajectories" in gen_data:
        return gen_data["trajectories"]

    return []


def build_word_tree(
    trajectories: list[dict],
    scores_by_traj: dict[int, list[float]],
    prompt: str,
) -> VizTreeNode:
    """Build a tree from trajectories using word-level tokenization."""
    root = VizTreeNode(label="[root]", prob=1.0)

    for i, traj in enumerate(trajectories):
        text = traj.get("text", "")
        prob = traj.get("probability", 1.0)
        log_prob = traj.get("log_probability", 0.0)
        is_greedy = traj.get("is_greedy", False)
        traj_idx = traj.get("traj_idx", i)

        # Extract continuation
        continuation = text
        if prompt and text.startswith(prompt):
            continuation = text[len(prompt):]

        # Split into words
        words = re.findall(r"\S+", continuation)
        if not words:
            continue

        current = root
        for j, word in enumerate(words):
            if word not in current.children:
                current.children[word] = VizTreeNode(label=word, prob=0.0)

            child = current.children[word]
            child.count += 1
            child.prob = max(child.prob, prob)
            child.log_prob = max(child.log_prob, log_prob)

            if j == len(words) - 1:
                child.is_leaf = True
                child.is_greedy = is_greedy
                child.traj_idx = traj_idx
                child.scores = scores_by_traj.get(traj_idx)

            current = child

    return root


def collapse_chains(node: VizTreeNode) -> VizTreeNode:
    """Collapse single-child chains into phrase nodes."""
    labels = [node.label]
    current = node

    while len(current.children) == 1:
        child = next(iter(current.children.values()))
        labels.append(child.label)
        current = child

    collapsed = VizTreeNode(
        label=" ".join(labels) if len(labels) > 1 else node.label,
        prob=current.prob,
        log_prob=current.log_prob,
        count=current.count,
        is_leaf=current.is_leaf,
        is_greedy=current.is_greedy,
        scores=current.scores,
        traj_idx=current.traj_idx,
    )

    for child in current.children.values():
        c = collapse_chains(child)
        collapsed.children[c.label] = c

    return collapsed


def propagate_cumulative_log_probs(
    node: VizTreeNode,
    parent_cumulative: float = 0.0,
) -> None:
    """Propagate cumulative log probabilities down the tree."""
    node.cumulative_log_prob = parent_cumulative + node.log_prob

    for child in node.children.values():
        propagate_cumulative_log_probs(child, node.cumulative_log_prob)


def aggregate_scores_up(node: VizTreeNode) -> list[float] | None:
    """Aggregate structure scores from children up to parent nodes.

    For each parent, compute the weighted average of children's scores
    based on their probabilities. Returns the aggregated scores.
    """
    # If leaf node with scores, use those
    if node.is_leaf and node.scores:
        node.aggregated_scores = node.scores
        return node.scores

    # If no children, no scores
    if not node.children:
        return None

    # Aggregate from children
    child_scores_list = []
    child_weights = []

    for child in node.children.values():
        child_scores = aggregate_scores_up(child)
        if child_scores:
            child_scores_list.append(child_scores)
            child_weights.append(child.prob)

    if not child_scores_list:
        return None

    # Compute weighted average
    n_dims = len(child_scores_list[0])
    total_weight = sum(child_weights)
    if total_weight == 0:
        total_weight = 1.0

    aggregated = []
    for dim in range(n_dims):
        weighted_sum = sum(
            scores[dim] * weight
            for scores, weight in zip(child_scores_list, child_weights)
        )
        aggregated.append(weighted_sum / total_weight)

    node.aggregated_scores = aggregated
    return aggregated


def layout_tree(
    node: VizTreeNode,
    x: float = 0.0,
    y_counter: list[float] | None = None,
    x_spacing: float = 2.5,
    y_spacing: float = 1.2,
) -> None:
    """Layout tree nodes with proper spacing."""
    if y_counter is None:
        y_counter = [0.0]

    node.x = x * x_spacing

    if not node.children:
        node.y = y_counter[0] * y_spacing
        y_counter[0] += 1.0
    else:
        child_ys = []
        for child in sorted(node.children.values(), key=lambda c: -c.prob):
            layout_tree(child, x + 1.0, y_counter, x_spacing, y_spacing)
            child_ys.append(child.y)
        node.y = sum(child_ys) / len(child_ys)


def collect_edges_and_nodes(
    node: VizTreeNode,
) -> tuple[list[tuple], list[tuple[VizTreeNode, tuple[float, float]]]]:
    """Collect all edges and nodes for plotting."""
    edges = []
    nodes = [(node, (node.x, node.y))]

    if node.children:
        total_prob = sum(c.prob for c in node.children.values())
        for child in node.children.values():
            edges.append(((node.x, node.y), (child.x, child.y), child, total_prob))
            ce, cn = collect_edges_and_nodes(child)
            edges.extend(ce)
            nodes.extend(cn)

    return edges, nodes


def draw_curved_edge(ax, x1: float, y1: float, x2: float, y2: float,
                     linewidth: float, color: str, alpha: float = 0.7) -> None:
    """Draw a curved bezier edge between two points."""
    # Control points for smooth S-curve
    mid_x = (x1 + x2) / 2
    ctrl1 = (mid_x, y1)
    ctrl2 = (mid_x, y2)

    # Create bezier curve path
    verts = [(x1, y1), ctrl1, ctrl2, (x2, y2)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    path = MplPath(verts, codes)

    patch = mpatches.PathPatch(
        path,
        facecolor="none",
        edgecolor=color,
        linewidth=linewidth,
        alpha=alpha,
        capstyle="round",
    )
    ax.add_patch(patch)


def draw_node_box(ax, x: float, y: float, label: str, color: str,
                  is_leaf: bool = False, is_greedy: bool = False,
                  fontsize: float = 8) -> None:
    """Draw a node with rounded rectangle background."""
    # Calculate box dimensions based on label length
    label = sanitize_label(label, max_len=20 if is_leaf else 15)

    # Draw text with background box
    bbox_props = {
        "boxstyle": "round,pad=0.3,rounding_size=0.2",
        "facecolor": color,
        "edgecolor": PALETTE["node_border"],
        "linewidth": 1.5 if is_greedy else 0.8,
        "alpha": 0.95,
    }

    text = ax.text(
        x, y, label,
        fontsize=fontsize,
        fontfamily="monospace",
        ha="center",
        va="center",
        color=PALETTE["text"],
        bbox=bbox_props,
        zorder=10,
    )

    # Greedy star marker
    if is_greedy:
        ax.annotate(
            "★",
            (x, y),
            xytext=(0, 12),
            textcoords="offset points",
            fontsize=12,
            ha="center",
            va="bottom",
            color=PALETTE["greedy_star"],
            fontweight="bold",
            zorder=11,
        )

    return text


def plot_tree(
    root: VizTreeNode,
    title: str,
    output_path: Path,
    structure_labels: list[str] | None = None,
) -> Path | None:
    """Render tree to PNG with improved styling."""
    # Propagate cumulative log probs and aggregate scores
    propagate_cumulative_log_probs(root)
    aggregate_scores_up(root)

    # Layout the tree
    layout_tree(root)

    # Collect nodes and edges
    edges, nodes = collect_edges_and_nodes(root)

    if not nodes:
        return None

    # Compute bounds
    all_x = [pos[0] for _, pos in nodes]
    all_y = [pos[1] for _, pos in nodes]

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    # Figure dimensions
    x_range = x_max - x_min + 1
    y_range = y_max - y_min + 1

    fig_width = max(12, x_range * 1.8 + 4)
    fig_height = max(6, y_range * 0.8 + 3)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    # Title
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15, color=PALETTE["text"])

    # Draw subtle grid
    for y in np.arange(y_min - 1, y_max + 2, 1):
        ax.axhline(y=y, color=PALETTE["grid"], linewidth=0.5, zorder=0)

    # Compute min/max cumulative log probs for normalization
    all_cum_log_probs = [child.cumulative_log_prob for _, _, child, _ in edges]
    if all_cum_log_probs:
        min_clp = min(all_cum_log_probs)
        max_clp = max(all_cum_log_probs)
        clp_range = max_clp - min_clp if max_clp != min_clp else 1.0
    else:
        min_clp, max_clp, clp_range = 0.0, 0.0, 1.0

    # Draw edges first (underneath nodes)
    for (x1, y1), (x2, y2), child, total_prob in edges:
        # Normalize cumulative log prob to [0, 1] range (higher is better)
        norm_clp = (child.cumulative_log_prob - min_clp) / clp_range

        # Edge thickness based on cumulative log probability (chained)
        linewidth = 0.8 + norm_clp * 4.0

        # Color: highlight high-probability paths
        if norm_clp > 0.6:
            color = PALETTE["edge_highlight"]
            alpha = 0.85
        elif norm_clp > 0.3:
            color = PALETTE["edge"]
            alpha = 0.7
        else:
            color = PALETTE["edge"]
            alpha = 0.4

        draw_curved_edge(ax, x1, y1, x2, y2, linewidth, color, alpha)

        # Probability label on edge (only for branching nodes)
        if total_prob > 0 and len([e for e in edges if e[0] == (x1, y1)]) > 1:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2

            rel_prob = child.prob / total_prob
            prob_str = f"{rel_prob:.0%}" if rel_prob >= 0.01 else f"{rel_prob:.1e}"
            ax.annotate(
                prob_str,
                (mid_x, mid_y),
                fontsize=7,
                ha="center",
                va="center",
                color=PALETTE["text_light"],
                fontweight="medium",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "alpha": 0.9,
                    "edgecolor": "none",
                },
                zorder=5,
            )

    # Draw nodes
    for node, (x, y) in nodes:
        # Determine node color based on scores (aggregated for parents, direct for leaves)
        node_scores = node.aggregated_scores or node.scores
        if node_scores and structure_labels:
            # Color by dominant structure score
            abs_scores = [abs(s) for s in node_scores]
            dominant_idx = int(np.argmax(abs_scores))
            color = STRUCTURE_COLORS[dominant_idx % len(STRUCTURE_COLORS)]
            # Lighten more for parent nodes, less for leaves
            lighten_factor = 0.6 if node.is_leaf else 0.75
            color = lighten_color(color, lighten_factor)
        elif node.label == "[root]":
            color = "#D0D0D0"
        else:
            color = PALETTE["node_default"]

        # Font size based on whether it's a leaf
        fontsize = 8 if node.is_leaf else 7

        draw_node_box(
            ax, x, y, node.label, color,
            is_leaf=node.is_leaf,
            is_greedy=node.is_greedy,
            fontsize=fontsize,
        )

        # Trajectory count badge for non-leaf nodes with multiple trajectories
        if not node.is_leaf and node.count > 1:
            ax.annotate(
                f"×{node.count}",
                (x, y),
                xytext=(0, -14),
                textcoords="offset points",
                fontsize=6,
                ha="center",
                va="top",
                color=PALETTE["text_light"],
                fontweight="bold",
                zorder=11,
            )

    # Legend for structure colors
    if structure_labels:
        legend_handles = []
        for i, label in enumerate(structure_labels[:8]):
            color = STRUCTURE_COLORS[i % len(STRUCTURE_COLORS)]
            patch = mpatches.Patch(
                facecolor=lighten_color(color, 0.7),
                edgecolor=color,
                linewidth=1.5,
                label=label,
            )
            legend_handles.append(patch)

        ax.legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=8,
            framealpha=0.95,
            edgecolor=PALETTE["node_border"],
            fancybox=True,
        )

    # Axis styling
    margin_x = 1.5
    margin_y = 0.8
    ax.set_xlim(x_min - margin_x, x_max + margin_x + 2)
    ax.set_ylim(y_min - margin_y, y_max + margin_y)
    ax.axis("off")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.tight_layout()
        plt.savefig(
            output_path, dpi=150, bbox_inches="tight",
            facecolor=PALETTE["bg"], edgecolor="none"
        )
        plt.close()
        return output_path
    except Exception:
        plt.close()
        return None


def create_tree_plots(
    result: EstimationResult,
    output_dir: Path,
) -> list[Path]:
    """Create tree visualizations from generation output."""
    created_files = []

    # Load generation data
    gen_data = load_generation_data(result.paths.generation)
    if not gen_data:
        return created_files

    trajectories = extract_trajectories_from_gen_data(gen_data)
    if not trajectories:
        return created_files

    # Load structure labels and scores from estimation output
    structure_labels = load_structure_labels(result.paths.estimation)
    scores_by_traj = load_trajectory_scores(result.paths.estimation)

    # Get prompt for tree building
    config = gen_data.get("config", {})
    prompt = config.get("trunk", "")
    formatted_prompt = gen_data.get("formatted_prompt", prompt)

    # Build word tree
    tree = build_word_tree(trajectories, scores_by_traj, formatted_prompt)

    # Create word-level tree plot
    word_tree_path = output_dir / "tree_word.png"
    title = f"Trajectory Tree (Word-Level) — {result.method}"
    saved = plot_tree(tree, title, word_tree_path, structure_labels)
    if saved:
        created_files.append(saved)

    # Create collapsed phrase tree
    phrase_tree = collapse_chains(tree)
    phrase_tree_path = output_dir / "tree_phrase.png"
    title = f"Trajectory Tree (Phrase-Level) — {result.method}"
    saved = plot_tree(phrase_tree, title, phrase_tree_path, structure_labels)
    if saved:
        created_files.append(saved)

    return created_files
