"""KDE plots of per-trajectory compliance scores per arm.

For each arm, draws one figure with a subplot per structure showing:
- The logit-transformed KDE density on (0, 1)
- A rug at each sample's compliance score
- Vertical markers for the population mean and the logit-KDE mode

These plots make the score distribution visible behind each arm's core,
which is especially informative when scores are bimodal (a common case
for LLM compliance).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from src.common.math.logit_kde import logit_kde_evaluate, logit_kde_mode
from src.estimation.arm_types import get_arm_color, get_ordered_arms_for_plotting
from src.estimation.estimation_scoring_data import ScoringData

from .viz_plot_utils import (
    get_structure_color,
    save_figure,
    style_axis_clean,
)

if TYPE_CHECKING:
    from src.estimation.estimation_experiment_types import EstimationResult

_GRID = np.linspace(1e-6, 1.0 - 1e-6, 1001)


def plot_kde_per_arm(
    result: "EstimationResult",
    structure_labels: list[str],
    output_dir: Path,
) -> list[Path]:
    """Plot per-structure KDE for each arm.

    One figure per arm, with one subplot per structure.

    Args:
        result: EstimationResult; result.paths.judgment must point to scoring.json.
        structure_labels: Per-structure labels for subplot titles.
        output_dir: Directory where `<arm>.png` files are written.

    Returns:
        List of created file paths.
    """
    scoring_path = result.paths.judgment
    data = ScoringData.load(scoring_path)
    by_arm = data.group_by_arm()

    arm_names = [a.name for a in result.arms]
    ordered = get_ordered_arms_for_plotting(arm_names)

    n_structures = len(structure_labels)
    if n_structures == 0:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    for arm_name in ordered:
        trajs = by_arm.get(arm_name, [])
        if not trajs:
            continue

        per_struct = [
            [t.structure_scores[i] for t in trajs] for i in range(n_structures)
        ]
        saved = _plot_arm_kde(
            arm_name=arm_name,
            per_structure_scores=per_struct,
            structure_labels=structure_labels,
            output_path=output_dir / f"{arm_name}.png",
        )
        if saved:
            created.append(saved)

    return created


def _plot_arm_kde(
    arm_name: str,
    per_structure_scores: list[list[float]],
    structure_labels: list[str],
    output_path: Path,
) -> Path | None:
    """Draw one figure with a KDE subplot per structure for a single arm."""
    n = len(structure_labels)
    if n == 0:
        return None

    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.4 * n_cols, 2.7 * n_rows),
        sharex=True,
    )
    axes = np.atleast_1d(axes).reshape(-1)

    arm_color = get_arm_color(arm_name)

    for i, label in enumerate(structure_labels):
        ax = axes[i]
        scores = per_structure_scores[i]
        _draw_one(ax, scores, label, color=get_structure_color(i), arm_color=arm_color)

    # Hide unused axes (if any)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    n_traj = len(per_structure_scores[0]) if per_structure_scores else 0
    fig.suptitle(
        f"Compliance KDE — arm: {arm_name} (n = {n_traj})",
        fontsize=12,
        fontweight="bold",
        color=arm_color,
    )

    return save_figure(fig, output_path, tight_layout_rect=[0, 0, 1, 0.96])


def _draw_one(
    ax,
    scores: list[float],
    label: str,
    *,
    color: str,
    arm_color: str,
) -> None:
    """Draw one structure's compliance distribution.

    Compliance scores often pile up exactly at 0 or 1 (point masses from
    binary judges or saturated metrics). Smearing those with KDE produces
    boundary-blowup artifacts. We split them out:

      * Point masses at exact 0 and 1 are shown as twin bars on the right axis,
        labeled with the fraction of trajectories at that boundary.
      * The logit-KDE is fit and drawn on the strictly-interior values only.
      * Mean and logit-KDE mode (over all values) are shown as vertical lines.
    """
    arr = np.asarray(scores, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(label, fontsize=10)
        return

    n = arr.size
    at_zero = arr <= 0.0
    at_one = arr >= 1.0
    interior = arr[(~at_zero) & (~at_one)]
    p_zero = float(np.mean(at_zero))
    p_one = float(np.mean(at_one))
    p_interior = 1.0 - p_zero - p_one

    mean = float(np.mean(arr))
    mode_val = logit_kde_mode(arr)

    # Point masses on a secondary right axis (proportion 0..1)
    ax_pm = ax.twinx()
    ax_pm.set_ylim(0, 1.0)
    bar_w = 0.04
    ax_pm.bar(
        [0.0],
        [p_zero],
        width=bar_w,
        color="#888888",
        alpha=0.55,
        align="center",
        label="P(=0/1)",
    )
    ax_pm.bar(
        [1.0],
        [p_one],
        width=bar_w,
        color="#888888",
        alpha=0.55,
        align="center",
    )
    ax_pm.tick_params(axis="y", labelsize=7, colors="#888888")
    ax_pm.set_ylabel("P(=0/1)", fontsize=7, color="#888888")
    for spine in ("top",):
        ax_pm.spines[spine].set_visible(False)

    # KDE on interior values
    if interior.size >= 2 and np.unique(interior).size >= 2:
        fy = logit_kde_evaluate(interior, _GRID)
        # Scale density by the interior mass so the curve area equals
        # the *interior fraction* — visually comparable to the point masses.
        if np.any(fy > 0):
            fy = fy * p_interior
        ax.plot(_GRID, fy, color=color, linewidth=1.6)
        ax.fill_between(_GRID, 0, fy, color=color, alpha=0.18)
        ymax = float(np.max(fy))
    else:
        ymax = 0.0

    # Rug of all values (alpha low so dense regions don't saturate)
    ax.plot(
        arr,
        np.zeros_like(arr),
        marker="|",
        linestyle="",
        markersize=8,
        color=color,
        alpha=0.25,
    )

    ax.axvline(mean, color="#444444", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(mode_val, color=arm_color, linestyle="-", linewidth=1.2, alpha=0.9)

    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(0, ymax * 1.2 if ymax > 0 else 1.0)
    ax.set_title(
        f"{label}  μ={mean:.2f}  mode={mode_val:.2f}  "
        f"P0={p_zero:.2f} P1={p_one:.2f}  n={n}",
        fontsize=8,
    )
    ax.set_xlabel("compliance", fontsize=8)
    style_axis_clean(ax, grid_axis="y", grid_alpha=0.25)
