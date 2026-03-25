"""Main visualization entry point for experiment results.

Generates all visualizations for an EstimationResult organized as:

{output_dir}/{gen_method}/{est_method}/
  - core.png                 Core bar plot per arm
  - deviance.png             Deviance trunk->branch lines
  - excess_deviance.png      Excess deviance (over-compliance) trunk->branch
  - deficit_deviance.png     Deficit deviance (under-compliance) trunk->branch
  - mutual_deviance.png      Mutual deviance (symmetric, JS divergence) trunk->branch
  - core_diversity.png       Core diversity (Hill D_1) per arm
  - orientation_{arm}.png    Orientation vectors relative to each arm
  - generalized_cores.png    Generalized cores heatmap (all arms, q/r variants)
  - generalized_deviance.png E[∂] as q→∞ and r→∞ line plots

{output_dir}/{gen_method}/
  - estimation_comparison.png  Compare cores across weighting methods
  - summary_breakdown.png      Structure breakdown (all questions by branch)
  - summary_core_evolution.png  Structure compliance in tree layout
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.common.default_config import DEFAULT_WEIGHTING_METHOD, STRING_SELECTION
from src.common.logging import log

from .experiment_breakdown_plot import plot_structure_breakdown
from .experiment_core_barplot import (
    plot_cores_barplot,
    plot_cores_comparison,
    plot_generation_comparison,
)
from .experiment_deviance_plot import (
    plot_core_diversity_by_arm,
    plot_deficit_deviance_by_arm,
    plot_deviance_by_arm,
    plot_excess_deviance_by_arm,
    plot_mutual_deviance_by_arm,
    plot_orientation_by_branch,
)
from .experiment_forking_plot import plot_orientation_tree, plot_structure_forking
from .experiment_tree_plot import load_structure_labels
from .experiment_variants_plot import plot_generalized_cores, plot_generalized_deviance

if TYPE_CHECKING:
    from src.estimation.estimation_experiment_types import EstimationResult


def visualize_result(
    result: EstimationResult,
    output_dir: Path | str | None = None,
    *,
    camera_ready: bool = False,
    summaries_only: bool = False,
) -> list[Path]:
    """Generate all visualizations for an experiment result.

    Output structure: out/<method>/<gen_name>/<scoring_name>/viz/...

    Args:
        result: The estimation result to visualize
        output_dir: Output directory (default: result path + /viz)
        camera_ready: If True, use high DPI (300) and enable all annotations
        summaries_only: If True, only generate summary plots (faster)
    """
    from .viz_plot_utils import set_camera_ready

    set_camera_ready(camera_ready)

    if output_dir is None:
        # Default: same folder as estimation.json, plus /viz
        output_dir = result.paths.estimation.parent / "viz"
    else:
        output_dir = Path(output_dir)

    gen_dir = output_dir
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

    # Load arm metadata for tree plots
    arm_descriptions, arm_texts, arm_n_traj, metadata, arm_suffix_probs = (
        _load_arm_metadata(result.paths)
    )

    # Per-estimation-method plots in subfolders (skip if summaries_only)
    if not summaries_only:
        for method in weighting_methods:
            est_dir = gen_dir / method
            est_dir.mkdir(parents=True, exist_ok=True)

            created_files.extend(
                _create_est_method_plots(
                    result,
                    method,
                    structure_labels,
                    est_dir,
                    arm_n_traj=arm_n_traj,
                    arm_texts=arm_texts,
                    arm_suffix_probs=arm_suffix_probs,
                    metadata=metadata,
                )
            )

    # Cross-method plots in gen_dir root
    created_files.extend(
        _create_cross_method_plots(result, weighting_methods, structure_labels, gen_dir)
    )

    if created_files:
        log(f"  [viz] {result.method}: {len(created_files)} plots -> {gen_dir}/")

    return created_files


def _create_est_method_plots(
    result: EstimationResult,
    method: str,
    structure_labels: list[str],
    est_dir: Path,
    arm_n_traj: dict[str, int] | None = None,
    arm_texts: dict[str, str] | None = None,
    arm_suffix_probs: dict[str, float] | None = None,
    metadata: dict[str, str] | None = None,
) -> list[Path]:
    """Create plots for a single estimation method."""
    created: list[Path] = []

    # Core
    saved = plot_cores_barplot(result, method, structure_labels, est_dir / "core.png")
    if saved:
        created.append(saved)

    # Structure core evolution (tree layout) per method
    if arm_n_traj and arm_texts:
        arm_weighted_cores = _compute_arm_weighted_cores(result, method)
        saved = plot_structure_forking(
            result.structure_info,
            arm_n_traj,
            arm_texts,
            est_dir / "core_evolution.png",
            metadata=metadata,
            arm_suffix_probs=arm_suffix_probs,
            arm_weighted_cores=arm_weighted_cores,
            weighting_method=method,
        )
        if saved:
            created.append(saved)

    # Deviance
    saved = plot_deviance_by_arm(result, method, est_dir / "deviance.png")
    if saved:
        created.append(saved)

    # Core diversity
    saved = plot_core_diversity_by_arm(result, method, est_dir / "core_diversity.png")
    if saved:
        created.append(saved)

    # Orientation (separate plot per reference arm) - goes to orientation/ subfolder
    orientation_files = plot_orientation_by_branch(
        result, method, structure_labels, est_dir
    )
    created.extend(orientation_files)

    # Orientation tree plots (like forking but with orientation)
    orientation_tree_files = _create_orientation_tree_plots(
        result, method, structure_labels, est_dir
    )
    created.extend(orientation_tree_files)

    # Compare statistics subfolder
    stats_dir = est_dir / "compare_statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Excess deviance
    saved = plot_excess_deviance_by_arm(
        result, method, stats_dir / "excess_deviance.png"
    )
    if saved:
        created.append(saved)

    # Deficit deviance
    saved = plot_deficit_deviance_by_arm(
        result, method, stats_dir / "deficit_deviance.png"
    )
    if saved:
        created.append(saved)

    # Mutual deviance
    saved = plot_mutual_deviance_by_arm(
        result, method, stats_dir / "mutual_deviance.png"
    )
    if saved:
        created.append(saved)

    # Generalized cores heatmap
    saved = plot_generalized_cores(
        result, method, structure_labels, stats_dir / "generalized_cores.png"
    )
    if saved:
        created.append(saved)

    # Generalized deviance line plots
    saved = plot_generalized_deviance(
        result, method, stats_dir / "generalized_deviance.png"
    )
    if saved:
        created.append(saved)

    return created


def _create_orientation_tree_plots(
    result: EstimationResult,
    method: str,
    structure_labels: list[str],
    est_dir: Path,
) -> list[Path]:
    """Create orientation tree plots for each reference arm.

    Like summary_core_evolution.png but shows orientation vectors instead of core values.
    Saved to orientation/evolution_{ref_arm}.png
    """

    created: list[Path] = []

    # Load arm metadata
    arm_descriptions, arm_texts, arm_n_traj, metadata, arm_suffix_probs = (
        _load_arm_metadata(result.paths)
    )

    # Get all arm names
    all_arm_names = [a.name for a in result.arms]

    # Find arms with downstream children (these will be reference arms)
    from .experiment_deviance_plot import has_downstream_arms

    reference_arms = [
        name for name in all_arm_names if has_downstream_arms(name, all_arm_names)
    ]

    # Create orientation subfolder
    orientation_dir = est_dir / "orientation"
    orientation_dir.mkdir(parents=True, exist_ok=True)

    for ref_name in reference_arms:
        # Build orientation dict for ALL arms (needed for tree structure)
        # Use zeros for arms without orientation relative to this ref
        arm_orientations: dict[str, list[float]] = {}
        n_structures = len(structure_labels)

        for arm in result.arms:
            # Determine reference type for orientation lookup
            ref_type = ref_name if ref_name in ("root", "trunk") else "parent"
            orientation = arm.get_orientation(ref_type, method)

            if orientation:
                arm_orientations[arm.name] = orientation
            else:
                # Use zeros for arms without orientation (e.g., the reference itself)
                arm_orientations[arm.name] = [0.0] * n_structures

        if not arm_orientations:
            continue

        # Create tree plot
        output_path = orientation_dir / f"evolution_{ref_name}.png"
        saved = plot_orientation_tree(
            result.structure_info,
            arm_n_traj,
            arm_texts,
            output_path,
            reference_arm=ref_name,
            arm_orientations=arm_orientations,
            metadata=metadata,
            arm_suffix_probs=arm_suffix_probs,
        )
        if saved:
            created.append(saved)

    return created


def _create_cross_method_plots(
    result: EstimationResult,
    weighting_methods: list[str],
    structure_labels: list[str],
    gen_dir: Path,
) -> list[Path]:
    """Create plots that span multiple estimation methods."""
    created: list[Path] = []

    # Load arm metadata from scoring.json and generation_cfg.json
    arm_descriptions, arm_texts, arm_n_traj, metadata, arm_suffix_probs = (
        _load_arm_metadata(result.paths)
    )

    # Estimation comparison - needs arm_descriptions for legend
    saved = plot_cores_comparison(
        result,
        weighting_methods,
        structure_labels,
        gen_dir / "estimation_comparison.png",
        arm_descriptions=arm_descriptions,
        metadata=metadata,
    )
    if saved:
        created.append(saved)

    # Structure breakdown - uses short config descriptions in legend
    saved = plot_structure_breakdown(
        result.arm_scoring,
        result.structure_info,
        gen_dir / "summary_breakdown.png",
        arm_descriptions=arm_descriptions,
        metadata=metadata,
    )
    if saved:
        created.append(saved)

    # Structure forking (tree layout) - shows differentiating arm texts
    # Use weighted core values from estimation (same as core.png), not raw arm_scoring
    arm_weighted_cores = _compute_arm_weighted_cores(result)

    saved = plot_structure_forking(
        result.structure_info,
        arm_n_traj,
        arm_texts,
        gen_dir / "summary_core_evolution.png",
        metadata=metadata,
        arm_suffix_probs=arm_suffix_probs,
        arm_weighted_cores=arm_weighted_cores,
        weighting_method=DEFAULT_WEIGHTING_METHOD,
    )
    if saved:
        created.append(saved)

    # Tree plots - disabled (too slow with many trajectories)
    # tree_files = create_tree_plots(result, gen_dir)
    # created.extend(tree_files)

    return created


def _compute_arm_weighted_cores(
    result: EstimationResult,
    method: str = DEFAULT_WEIGHTING_METHOD,
) -> dict[str, list[float]]:
    """Compute weighted core values for each arm using the estimation method.

    This ensures the forking plot shows the same values as core.png.

    Args:
        result: Estimation result with arm data
        method: Weighting method to use

    Returns:
        Dict mapping arm name to list of structure compliance values (0.0-1.0).

    Raises:
        KeyError: if method not found (no silent fallbacks)
    """
    arm_weighted_cores: dict[str, list[float]] = {}

    for arm in result.arms:
        core = arm.get_core(method)  # Will raise KeyError if method not found
        arm_weighted_cores[arm.name] = core

    return arm_weighted_cores


def _compute_arm_suffix_probs(
    paths: Any,
    arm_names: list[str],
) -> dict[str, float]:
    """Compute P(arm_suffix | parent_prefix) for each arm using generation data.

    IMPORTANT: Each arm's probability must be computed from a trajectory that
    actually went through that arm, since different branches have different tokens.

    Returns:
        Dict mapping arm name to probability of its suffix given parent.

    Raises:
        ValueError: if required data is missing (no silent defaults)
    """
    import json

    from src.common.token_trajectory import TokenTrajectory
    from src.estimation.arm_types import ArmKind, classify_arm, get_branch_index

    arm_suffix_probs: dict[str, float] = {}

    with open(paths.generation) as f:
        gen_data = json.load(f)

    tree = gen_data.get("tree", {})
    trajs_data = tree.get("trajs", [])
    if not trajs_data:
        raise ValueError("No trajectories in generation data")

    # Build arm index mapping (arm_name -> index in arm_token_lengths)
    config = gen_data.get("config", {})
    config_arms = config.get("arms", [])
    arm_name_to_idx = {arm.get("name", ""): i for i, arm in enumerate(config_arms)}

    # Load scoring.json to get arm assignments per trajectory
    with open(paths.judgment) as f:
        scoring_data = json.load(f)
    results = scoring_data.get("results", [])

    # Build mapping: arm_name -> list of trajectory indices that went through it
    arm_to_traj_indices: dict[str, list[int]] = {}
    for i, r in enumerate(results):
        arm = r.get("arm")
        if arm:
            if arm not in arm_to_traj_indices:
                arm_to_traj_indices[arm] = []
            arm_to_traj_indices[arm].append(i)

    # For root/trunk, any trajectory works since they share the same prefix
    # For branches/twigs, we need a trajectory that actually went through that arm
    def get_traj_for_arm(arm_name: str) -> TokenTrajectory:
        """Get a trajectory that went through the given arm."""
        kind = classify_arm(arm_name)

        if kind in (ArmKind.ROOT, ArmKind.TRUNK):
            # Any trajectory works for shared prefix
            return TokenTrajectory.from_dict(trajs_data[0])

        # For branches/twigs, find a trajectory that went through this arm
        # A twig trajectory also goes through its parent branch
        if arm_name in arm_to_traj_indices:
            idx = arm_to_traj_indices[arm_name][0]
            return TokenTrajectory.from_dict(trajs_data[idx])

        # For branches, any twig under this branch also works
        if kind == ArmKind.BRANCH:
            branch_idx = get_branch_index(arm_name)
            for twig_name in arm_to_traj_indices:
                if twig_name.startswith(f"twig_b{branch_idx}_"):
                    idx = arm_to_traj_indices[twig_name][0]
                    return TokenTrajectory.from_dict(trajs_data[idx])

        raise ValueError(f"No trajectory found for arm '{arm_name}'")

    # Compute P(suffix | parent) for each arm
    for arm_name in arm_names:
        arm_idx = arm_name_to_idx.get(arm_name)
        if arm_idx is None:
            raise KeyError(f"arm_name_to_idx missing '{arm_name}'")

        traj = get_traj_for_arm(arm_name)
        if not traj.arm_token_lengths:
            raise ValueError(f"Trajectory for '{arm_name}' has no arm_token_lengths")
        if arm_idx >= len(traj.arm_token_lengths):
            raise IndexError(f"arm_idx {arm_idx} >= len(arm_token_lengths)")

        kind = classify_arm(arm_name)
        arm_end = traj.arm_token_lengths[arm_idx]

        # Find parent's end position
        if kind == ArmKind.ROOT:
            # Root has no parent, prob = 1
            arm_suffix_probs[arm_name] = 1.0
            continue
        elif kind == ArmKind.TRUNK:
            parent_idx = arm_name_to_idx.get("root")
        elif kind == ArmKind.BRANCH:
            parent_idx = arm_name_to_idx.get("trunk")
        elif kind == ArmKind.TWIG:
            branch_idx = get_branch_index(arm_name)
            parent_idx = arm_name_to_idx.get(f"branch_{branch_idx}")
        else:
            raise ValueError(f"Unknown arm kind for '{arm_name}'")

        if parent_idx is None:
            raise KeyError(f"Parent index not found for '{arm_name}'")
        if parent_idx >= len(traj.arm_token_lengths):
            raise IndexError(f"parent_idx {parent_idx} >= len(arm_token_lengths)")

        parent_end = traj.arm_token_lengths[parent_idx]

        # If no suffix tokens (parent_end == arm_end), probability is 1.0
        if parent_end >= arm_end:
            arm_suffix_probs[arm_name] = 1.0
            continue

        # Compute conditional probability of suffix tokens
        prob = traj.get_conditional_prob(parent_end, arm_end)
        if prob is None:
            # No logprobs available for this range, default to 1.0
            arm_suffix_probs[arm_name] = 1.0
        else:
            arm_suffix_probs[arm_name] = prob

    return arm_suffix_probs


def _load_arm_metadata(
    paths: Any,
) -> tuple[
    dict[str, str], dict[str, str], dict[str, int], dict[str, str], dict[str, float]
]:
    """Load arm data from scoring.json and generation_cfg.json.

    Returns:
        Tuple of (arm_descriptions, arm_texts, arm_n_traj, metadata, arm_suffix_probs)
        - arm_descriptions: Short config descriptions for legend (from generation_cfg)
        - arm_texts: Full prefill texts for forking plot (from scoring)
        - arm_n_traj: Trajectory counts per arm
        - metadata: prompt, model, judge info
        - arm_suffix_probs: P(arm_suffix | parent_prefix) from model logprobs
    """
    import json

    arm_descriptions: dict[str, str] = {}
    arm_texts: dict[str, str] = {}
    arm_n_traj: dict[str, int] = {}
    metadata: dict[str, str] = {}
    arm_suffix_probs: dict[str, float] = {}

    # Load from scoring.json
    try:
        with open(paths.judgment) as f:
            scoring_data = json.load(f)
        arm_texts = scoring_data.get("arm_texts", {})

        # Get trajectory counts from results grouped by arm
        results = scoring_data.get("results", [])
        arm_counts: dict[str, int] = {}
        for r in results:
            arm = r.get("arm", "trunk")
            arm_counts[arm] = arm_counts.get(arm, 0) + 1
        arm_n_traj = arm_counts

        # Load judge from scoring metadata
        scoring_meta = scoring_data.get("metadata", {})
        metadata["judge"] = scoring_meta.get("judge_model", "")
    except (OSError, json.JSONDecodeError, KeyError):
        pass

    # Load string_selection from scoring_cfg.json (saved alongside scoring.json)
    # Fall back to the default value so it always shows in the visualization
    try:
        scoring_cfg_path = paths.judgment.parent / "scoring_cfg.json"
        with open(scoring_cfg_path) as f:
            scoring_cfg = json.load(f)
        metadata["string_selection"] = scoring_cfg.get("string_selection", STRING_SELECTION)
    except (OSError, json.JSONDecodeError, KeyError):
        metadata["string_selection"] = STRING_SELECTION

    # Load arm descriptions, model, and prompt from generation_cfg.json
    try:
        gen_cfg_path = paths.generation.parent / "generation_cfg.json"
        with open(gen_cfg_path) as f:
            gen_cfg = json.load(f)

        # Model and prompt
        metadata["model"] = gen_cfg.get("model", "")
        metadata["prompt"] = gen_cfg.get("prompt", "")

        # Build arm descriptions from config
        root_desc = gen_cfg.get("root", "")
        trunk_desc = gen_cfg.get("trunk", "")
        branches = gen_cfg.get("branches", [])
        twig_vars = gen_cfg.get("twig_variations", [])

        arm_descriptions["root"] = root_desc
        arm_descriptions["trunk"] = trunk_desc
        for i, branch_text in enumerate(branches, 1):
            arm_descriptions[f"branch_{i}"] = branch_text
            for j, twig_text in enumerate(twig_vars, 1):
                arm_descriptions[f"twig_b{i}_{j}"] = twig_text
    except (OSError, json.JSONDecodeError, KeyError):
        pass

    # Compute arm suffix probabilities from generation data
    all_arm_names = list(arm_texts.keys()) or list(arm_descriptions.keys())
    arm_suffix_probs = _compute_arm_suffix_probs(paths, all_arm_names)

    return arm_descriptions, arm_texts, arm_n_traj, metadata, arm_suffix_probs


def visualize_generation_comparison(
    results: list[EstimationResult],
    output_dir: Path | str | None = None,
) -> list[Path]:
    """Generate comparison visualization across generation methods.

    Creates a comparison plot that compares trunk cores across different
    generation methods. Output path includes gen_name and scoring_name
    to prevent overwrites from different experiment configurations.

    Args:
        results: List of EstimationResults from different generation methods
        output_dir: Output directory (default: out/generation_comparisons/{gen_name}/)

    Returns:
        List of created file paths
    """
    if len(results) < 2:
        return []

    # Extract gen_name and scoring_name from estimation path
    # Path structure: out/{method}/{gen_name}/{scoring_name}/estimation.json
    estimation_path = Path(results[0].paths.estimation)
    scoring_name = estimation_path.parent.name
    gen_name = estimation_path.parent.parent.name

    if output_dir is None:
        output_dir = Path("out/generation_comparisons") / gen_name
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"{scoring_name}.png"

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
        results,
        weighting_methods,
        structure_labels,
        output_dir / output_filename,
        default_method=DEFAULT_WEIGHTING_METHOD,
    )
    if saved:
        created_files.append(saved)
        log(f"  [viz] Generation comparison: {output_dir / output_filename}")

    return created_files
