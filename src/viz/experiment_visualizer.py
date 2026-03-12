"""Main visualization entry point for experiment results.

Generates all visualizations for an EstimationResult organized as:

{output_dir}/{gen_method}/{est_method}/
  - core.png                 Core bar plot per arm
  - deviance.png             Deviance trunk->branch lines
  - excess_deviance.png      Excess deviance (over-compliance) trunk->branch
  - deficit_deviance.png     Deficit deviance (under-compliance) trunk->branch
  - mutual_deviance.png      Mutual deviance (symmetric, JS divergence) trunk->branch
  - core_diversity.png       Core diversity (Hill D_1) per arm
  - dynamics.png             Trajectory dynamics over positions
  - orientation.png          Orientation vectors per branch
  - generalized_cores.png    Generalized cores heatmap (all arms, q/r variants)
  - generalized_deviance.png E[∂] as q→∞ and r→∞ line plots

{output_dir}/{gen_method}/
  - estimation_comparison.png  Compare cores across weighting methods
  - trunk_vs_all.png           Trunk vs all_arms comparison
  - bundled_structures.png     Individual questions in bundled structures
  - tree_word.png              Token tree (word level)
  - tree_phrase.png            Token tree (phrase level)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.common.default_config import DEFAULT_WEIGHTING_METHOD
from src.common.logging import log

from .experiment_breakdown_plot import plot_bundled_structures
from .experiment_core_barplot import (
    plot_cores_barplot,
    plot_cores_comparison,
    plot_generation_comparison,
    plot_trunk_vs_all_arms,
)
from .experiment_deviance_plot import (
    plot_core_diversity_by_arm,
    plot_deficit_deviance_by_arm,
    plot_deviance_by_arm,
    plot_excess_deviance_by_arm,
    plot_mutual_deviance_by_arm,
    plot_orientation_by_branch,
)
from .experiment_dynamics_plot import plot_dynamics
from .experiment_tree_plot import create_tree_plots, load_structure_labels
from .experiment_variants_plot import plot_generalized_cores, plot_generalized_deviance

if TYPE_CHECKING:
    from src.estimation.estimation_experiment_types import EstimationResult


def visualize_result(
    result: EstimationResult,
    output_dir: Path | str | None = None,
) -> list[Path]:
    """Generate all visualizations for an experiment result."""
    if output_dir is None:
        output_dir = Path("out/viz")
    else:
        output_dir = Path(output_dir)

    gen_dir = output_dir / result.method
    gen_dir.mkdir(parents=True, exist_ok=True)
    created_files: list[Path] = []

    # Get structure labels
    structure_labels = load_structure_labels(result.paths.estimation)
    if not structure_labels and result.arms and result.arms[0].estimates:
        first_method = next(iter(result.arms[0].estimates.keys()))
        core = result.arms[0].get_core(first_method)
        structure_labels = [f"s{i + 1}" for i in range(len(core))]

    # Get weighting methods
    weighting_methods: list[str] = []
    for arm in result.arms:
        if arm.estimates:
            weighting_methods = list(arm.estimates.keys())
            break

    # Per-estimation-method plots in subfolders
    for method in weighting_methods:
        est_dir = gen_dir / method
        est_dir.mkdir(parents=True, exist_ok=True)

        created_files.extend(_create_est_method_plots(
            result, method, structure_labels, est_dir
        ))

    # Cross-method plots in gen_dir root
    created_files.extend(_create_cross_method_plots(
        result, weighting_methods, structure_labels, gen_dir
    ))

    if created_files:
        log(f"  [viz] {result.method}: {len(created_files)} plots -> {gen_dir}/")

    return created_files


def _create_est_method_plots(
    result: "EstimationResult",
    method: str,
    structure_labels: list[str],
    est_dir: Path,
) -> list[Path]:
    """Create plots for a single estimation method."""
    created: list[Path] = []

    # Core
    saved = plot_cores_barplot(result, method, structure_labels, est_dir / "core.png")
    if saved:
        created.append(saved)

    # Deviance
    saved = plot_deviance_by_arm(result, method, est_dir / "deviance.png")
    if saved:
        created.append(saved)

    # Excess deviance
    saved = plot_excess_deviance_by_arm(result, method, est_dir / "excess_deviance.png")
    if saved:
        created.append(saved)

    # Deficit deviance
    saved = plot_deficit_deviance_by_arm(result, method, est_dir / "deficit_deviance.png")
    if saved:
        created.append(saved)

    # Mutual deviance
    saved = plot_mutual_deviance_by_arm(result, method, est_dir / "mutual_deviance.png")
    if saved:
        created.append(saved)

    # Core diversity
    saved = plot_core_diversity_by_arm(result, method, est_dir / "core_diversity.png")
    if saved:
        created.append(saved)

    # Dynamics
    saved = plot_dynamics(result, method, structure_labels, est_dir / "dynamics.png")
    if saved:
        created.append(saved)

    # Orientation
    saved = plot_orientation_by_branch(result, method, structure_labels, est_dir / "orientation.png")
    if saved:
        created.append(saved)

    # Generalized cores heatmap (all arms in one figure)
    saved = plot_generalized_cores(
        result, method, structure_labels, est_dir / "generalized_cores.png"
    )
    if saved:
        created.append(saved)

    # Generalized deviance line plots
    saved = plot_generalized_deviance(result, method, est_dir / "generalized_deviance.png")
    if saved:
        created.append(saved)

    return created


def _create_cross_method_plots(
    result: "EstimationResult",
    weighting_methods: list[str],
    structure_labels: list[str],
    gen_dir: Path,
) -> list[Path]:
    """Create plots that span multiple estimation methods."""
    created: list[Path] = []

    # Estimation comparison
    saved = plot_cores_comparison(
        result, weighting_methods, structure_labels, gen_dir / "estimation_comparison.png"
    )
    if saved:
        created.append(saved)

    # Trunk vs all_arms
    saved = plot_trunk_vs_all_arms(
        result, weighting_methods, structure_labels, gen_dir / "trunk_vs_all.png"
    )
    if saved:
        created.append(saved)

    # Bundled structures breakdown (individual questions)
    saved = plot_bundled_structures(
        result.arm_scoring, result.structure_info, gen_dir / "bundled_structures.png"
    )
    if saved:
        created.append(saved)

    # Tree plots
    tree_files = create_tree_plots(result, gen_dir)
    created.extend(tree_files)

    return created


def visualize_generation_comparison(
    results: list["EstimationResult"],
    output_dir: Path | str | None = None,
) -> list[Path]:
    """Generate comparison visualization across generation methods.

    Creates a generation_comparison.png in the output_dir root that
    compares trunk cores across different generation methods.

    Args:
        results: List of EstimationResults from different generation methods
        output_dir: Output directory (default: out/viz)

    Returns:
        List of created file paths
    """
    if len(results) < 2:
        return []

    if output_dir is None:
        output_dir = Path("out/viz")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created_files: list[Path] = []

    # Get structure labels from first result
    structure_labels = load_structure_labels(results[0].paths.estimation)
    if not structure_labels and results[0].arms and results[0].arms[0].estimates:
        first_method = next(iter(results[0].arms[0].estimates.keys()))
        core = results[0].arms[0].get_core(first_method)
        structure_labels = [f"s{i + 1}" for i in range(len(core))]

    # Get weighting methods from first result
    weighting_methods: list[str] = []
    for arm in results[0].arms:
        if arm.estimates:
            weighting_methods = list(arm.estimates.keys())
            break

    # Create generation comparison plot
    saved = plot_generation_comparison(
        results, weighting_methods, structure_labels,
        output_dir / "generation_comparison.png",
        default_method=DEFAULT_WEIGHTING_METHOD,
    )
    if saved:
        created_files.append(saved)
        log(f"  [viz] Generation comparison: 1 plot -> {output_dir}/")

    return created_files
