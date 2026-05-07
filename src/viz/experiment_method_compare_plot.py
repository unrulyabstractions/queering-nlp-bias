"""Method-comparison plot: trunk core under several "normativity" lenses.

Compares four estimators of the trunk's system core, each interpretable
as a different way to characterize "normativity":

* `uniform` → Normativity as average
* `greedy`  → Normativity as greedy decoding
* `mode`    → Normativity as mode
* `median`  → Normativity as median

For the trunk arm, each method's core is drawn as a contiguous group of
structure-colored bars; the four method blocks sit side by side so the
shapes can be read off directly. The legend uses each structure's full
description rather than its short code.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from src.estimation.arm_types import get_arm_color

from .viz_plot_utils import (
    annotate_bar_values,
    get_structure_color,
    get_structure_color_by_label,
    save_figure,
    style_axis_clean,
)

if TYPE_CHECKING:
    from src.estimation.estimation_experiment_types import EstimationResult


# Methods compared, in display order, with the user-facing axis labels.
_METHODS_AND_LABELS: list[tuple[str, str]] = [
    ("uniform", "As Average"),
    ("greedy", "As Greedy Decoding"),
    ("mode", "As Mode"),
    ("median", "As Median"),
]


def _structure_descriptions(
    structure_info: list[dict[str, Any]] | None,
    fallback_labels: list[str],
) -> list[str]:
    """Resolve legend labels for each structure.

    Prefers a short label from the optional `judgement_legend.json` sidecar
    (loaded into `viz_plot_utils._JUDGEMENT_LEGEND` by the visualizer entry
    point). Falls back to the full description, then to the short code.
    """
    from .viz_plot_utils import get_structure_short_label

    if not structure_info:
        return list(fallback_labels)
    descs: list[str] = []
    for i, _ in enumerate(fallback_labels):
        info = structure_info[i] if i < len(structure_info) else None
        if isinstance(info, dict):
            full = (info.get("description") or "").strip()
            label = get_structure_short_label(full, fallback=fallback_labels[i])
            descs.append(label or fallback_labels[i])
        else:
            descs.append(fallback_labels[i])
    return descs


def plot_cores_by_method(
    result: "EstimationResult",
    weighting_methods: list[str],
    structure_labels: list[str],
    output_path: Path,
    *,
    arm_descriptions: dict[str, str] | None = None,
    arm_texts: dict[str, str] | None = None,
    **_unused: Any,
) -> Path | None:
    """Plot the trunk core under uniform / greedy / mode / max-inv-ppl.

    Returns None if the trunk is missing or none of the four target
    methods have a core for it.
    """
    trunk = next((a for a in result.arms if a.name == "trunk" and a.estimates), None)
    if trunk is None or not structure_labels:
        return None

    cores: list[tuple[str, str, list[float]]] = []
    for method, axis_label in _METHODS_AND_LABELS:
        if method not in weighting_methods:
            continue
        try:
            c = trunk.get_core(method)
        except KeyError:
            continue
        if c:
            cores.append((method, axis_label, c))
    if not cores:
        return None

    n_methods = len(cores)
    n_structures = len(structure_labels)

    legend_labels = _structure_descriptions(result.structure_info, structure_labels)
    # Full descriptions for sidecar color lookup (legend_labels are short).
    full_descriptions = [
        (info.get("description") or structure_labels[i])
        if isinstance(info, dict)
        else structure_labels[i]
        for i, info in enumerate(result.structure_info or [])
    ] or list(structure_labels)

    # 2x2 grid (top-up to 4 methods); falls back gracefully if fewer.
    n_cols = 2
    n_rows = (n_methods + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(11.0, 4.5 * n_rows),
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes.flatten()

    arm_color = get_arm_color("trunk")

    x = np.arange(n_structures)
    structure_handles: list[object] = [None] * n_structures  # type: ignore[list-item]

    for ax_idx, (method, axis_label, core) in enumerate(cores):
        ax = axes_flat[ax_idx]
        for s_idx in range(n_structures):
            bars = ax.bar(
                [x[s_idx]],
                [core[s_idx]],
                0.7,
                color=get_structure_color_by_label(
                    full_descriptions[s_idx] if s_idx < len(full_descriptions) else None,
                    fallback_idx=s_idx,
                ),
                edgecolor="black",
                linewidth=0.4,
                alpha=0.9,
                label=legend_labels[s_idx] if ax_idx == 0 else None,
            )
            if structure_handles[s_idx] is None:
                structure_handles[s_idx] = bars[0]
            annotate_bar_values(ax, bars, [core[s_idx]], fontsize=9)

        ax.axhline(y=0.5, color="#cccccc", linestyle="--", linewidth=0.8, zorder=0)
        ax.set_title(axis_label, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([""] * n_structures)
        ax.set_xlim(-0.6, n_structures - 0.4)
        ax.set_ylim(0, 1.15)
        if ax_idx % n_cols == 0:
            ax.set_ylabel("Structure Core Score", fontsize=10)
        style_axis_clean(ax, dense_grid=True, grid_axis="y")

    # Hide unused panes (when n_methods < n_rows * n_cols).
    for j in range(n_methods, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Trunk subtitle becomes the figure-level title (in trunk color).
    trunk_label = ""
    if arm_descriptions:
        trunk_label = (arm_descriptions.get("trunk") or "").strip()
    if not trunk_label and arm_texts:
        trunk_label = (arm_texts.get("trunk") or "").strip()
    if trunk_label:
        sub = trunk_label[:140] + ("…" if len(trunk_label) > 140 else "")
        fig.suptitle(
            f'Normativity at "{sub}" characterized',
            fontsize=13,
            fontweight="bold",
            color=arm_color,
            y=0.995,
        )

    fig.legend(
        structure_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=min(n_structures, 2),
        fontsize=10,
        frameon=False,
        title="Structure",
        title_fontsize=10,
    )

    return save_figure(fig, output_path, tight_layout_rect=[0, 0.10, 1, 0.96])
