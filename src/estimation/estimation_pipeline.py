"""Core estimation pipeline logic.

This module contains the core algorithms for normativity estimation,
independent of logging and script utilities.

Key function: compute_arm_estimate() - computes arm-level statistics from
trajectory data using all registered weighting methods.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common.math.entropy_diversity.structure_aware import (
    core_diversity,
    deviance,
    deviance_variance,
    expected_deficit_deviance,
    expected_deviance,
    expected_excess_deviance,
    expected_mutual_deviance,
    generalized_system_core,
    orientation,
)
from src.common.math.vector_utils import compute_orientation_vector

# Import methods to trigger registration
from . import methods as _methods  # noqa: F401
from .arm_types import ArmKind, classify_arm, get_parent_branch
from .estimation_core_types import NAMED_CORES, CoreVariant
from .estimation_output import EstimationOutput
from .estimation_scoring_data import ScoringData
from .estimation_structure import (
    ArmEstimate,
    TrajectoryEstimate,
    TrajectoryScoringData,
)
from .estimation_weighted_types import WeightedEstimate
from .weighting_method_registry import get_default_params, get_method, iter_methods

# ══════════════════════════════════════════════════════════════════════════════
# CORE VARIANTS COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════


def compute_core_variants(
    structure_scores_list: list[list[float]], weights: list[float]
) -> list[CoreVariant]:
    """Compute all named core variants with their deviances.

    Args:
        structure_scores_list: List of compliance vectors per trajectory
        weights: Weights for each trajectory (already normalized)

    Returns:
        List of CoreVariant with computed cores and deviance stats
    """
    variants = []
    for name, q, r, desc in NAMED_CORES:
        try:
            core = generalized_system_core(structure_scores_list, weights, q=q, r=r)
            dev_avg = expected_deviance(
                structure_scores_list, core, weights=weights, norm="l2"
            )
            dev_var = deviance_variance(
                structure_scores_list, core, weights=weights, norm="l2"
            )
            variants.append(
                CoreVariant(
                    name=name,
                    q=q,
                    r=r,
                    description=desc,
                    core=core,
                    deviance_avg=dev_avg,
                    deviance_var=dev_var,
                )
            )
        except (ValueError, ZeroDivisionError, OverflowError):
            # Some (q, r) combinations may fail numerically
            pass
    return variants


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHTED ESTIMATE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════


def compute_weighted_estimate(
    method_name: str,
    structure_scores_list: list[list[float]],
    log_probs: list[float],
    n_tokens: list[int],
    trunk_core: list[float] | None = None,
    root_core: list[float] | None = None,
    parent_core: list[float] | None = None,
) -> WeightedEstimate:
    """Compute estimation using a specific weighting method.

    Args:
        method_name: Name of the weighting method (e.g., "prob", "inv-ppl")
        structure_scores_list: List of compliance vectors per trajectory
        log_probs: Log probabilities for each trajectory
        n_tokens: Number of tokens per trajectory
        trunk_core: Optional trunk core to compute deviance against
        root_core: Optional root core to compute deviance against
        parent_core: Optional parent branch core (for twigs) to compute orientation

    Returns:
        WeightedEstimate with all statistics for this weighting method
    """
    # Get the weighting function and compute weights
    weight_fn = get_method(method_name)
    params = get_default_params(method_name)
    weights = weight_fn(log_probs, n_tokens, params)

    # Compute primary core (q=1, r=1)
    core = generalized_system_core(structure_scores_list, weights, q=1.0, r=1.0)

    # Compute deviance stats relative to this arm's core
    dev_avg = expected_deviance(structure_scores_list, core, weights=weights, norm="l2")
    dev_var = deviance_variance(structure_scores_list, core, weights=weights, norm="l2")

    # E[d|root] - deviance relative to root core
    dev_avg_root = 0.0
    if root_core is not None:
        dev_avg_root = expected_deviance(
            structure_scores_list, root_core, weights=weights, norm="l2"
        )

    # E[∂|trunk] - deviance relative to trunk core
    dev_avg_trunk = 0.0
    if trunk_core is not None:
        dev_avg_trunk = expected_deviance(
            structure_scores_list, trunk_core, weights=weights, norm="l2"
        )

    # E[∂⁺] - excess deviance (over-compliance)
    excess_dev = expected_excess_deviance(structure_scores_list, core, weights=weights)

    # E[∂⁻] - deficit deviance (under-compliance)
    deficit_dev = expected_deficit_deviance(structure_scores_list, core, weights=weights)

    # E[∂_M] - mutual deviance (symmetric, uses JS divergence)
    mutual_dev = expected_mutual_deviance(structure_scores_list, core, weights=weights)

    # Core diversity (effective number of structures)
    if not core:
        raise ValueError(
            f"Empty core computed for method '{method_name}'. "
            "Cannot compute core diversity."
        )
    core_div = core_diversity(core)

    # Compute all named core variants
    core_variants = compute_core_variants(structure_scores_list, weights)

    # Compute orientation vectors and norms relative to reference cores
    orientation_from_root, orientation_norm_from_root = compute_orientation_vector(
        core, root_core
    )
    orientation_from_trunk, orientation_norm_from_trunk = compute_orientation_vector(
        core, trunk_core
    )
    orientation_from_parent, orientation_norm_from_parent = compute_orientation_vector(
        core, parent_core
    )

    return WeightedEstimate(
        method_name=method_name,
        core=core,
        deviance_avg=dev_avg,
        deviance_var=dev_var,
        deviance_avg_root=dev_avg_root,
        deviance_avg_trunk=dev_avg_trunk,
        excess_deviance_avg=excess_dev,
        deficit_deviance_avg=deficit_dev,
        mutual_deviance_avg=mutual_dev,
        core_diversity=core_div,
        orientation_from_root=orientation_from_root,
        orientation_norm_from_root=orientation_norm_from_root,
        orientation_from_trunk=orientation_from_trunk,
        orientation_norm_from_trunk=orientation_norm_from_trunk,
        orientation_from_parent=orientation_from_parent,
        orientation_norm_from_parent=orientation_norm_from_parent,
        core_variants=core_variants,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE RESULT
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class PipelineResult:
    """Result of running the estimation pipeline.

    Contains the output data, arm estimates, and reference cores
    needed for downstream processing.
    """

    output: EstimationOutput
    arms: list[ArmEstimate]
    # Reference cores from trunk (for computing branch orientations)
    trunk_cores: dict[str, list[float]]  # {method_name: core}


# ══════════════════════════════════════════════════════════════════════════════
# ARM ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════


def compute_arm_estimate(
    arm_idx: int,
    name: str,
    trajectories: list[TrajectoryScoringData],
    trunk_cores: dict[str, list[float]] | None = None,
    root_cores: dict[str, list[float]] | None = None,
    parent_cores: dict[str, list[float]] | None = None,
) -> ArmEstimate:
    """Compute arm-level estimate from trajectory structure scores.

    This is the primary computation function for arm estimation. It:
    1. Iterates over all registered weighting methods
    2. Computes WeightedEstimate for each method
    3. Stores results in ArmEstimate.estimates dict

    Args:
        arm_idx: Index of this arm in processing order.
            If root present: root=0, trunk=1, all_arms=2, branches=3+
            If no root: trunk=0, all_arms=1, branches=2+
        name: Name of this arm (e.g., "root", "trunk", "branch_1", "all_arms")
        trajectories: Trajectories with structure_scores and conditional log probs
        trunk_cores: Optional dict of trunk cores by method name (for deviance metrics)
        root_cores: Optional dict of root cores by method name (for deviance metrics)
        parent_cores: Optional dict of parent branch cores by method name (for twig orientation)

    Returns:
        ArmEstimate with all computed statistics across all weighting methods
    """
    n_trajs = len(trajectories)

    # Fail loudly if arm has no trajectories - this indicates a pipeline bug
    if n_trajs == 0:
        raise ValueError(
            f"Cannot compute estimate for arm '{name}' with no trajectories. "
            "Check that trajectories were generated and scored for this arm."
        )

    # Extract data from trajectories
    structure_scores_list = [t.structure_scores for t in trajectories]

    # Extract log probs - for pooled arms (all_arms), use trunk conditioning
    # For regular arms, use the arm's own conditioning
    logprob_key = "trunk" if name == "all_arms" else name

    log_probs: list[float] = []
    for t in trajectories:
        if logprob_key not in t.conditional_logprobs:
            raise KeyError(
                f"Trajectory {t.traj_idx} missing conditional_logprobs for '{logprob_key}'. "
                f"Available arms: {list(t.conditional_logprobs.keys())}"
            )
        log_probs.append(t.conditional_logprobs[logprob_key])

    n_tokens = [t.n_continuation_tokens for t in trajectories]
    n_structures = len(structure_scores_list[0])

    # Validate consistent dimensions
    for i, c in enumerate(structure_scores_list[1:], start=1):
        if len(c) != n_structures:
            raise ValueError(
                f"Compliance {i} has {len(c)} dimensions, expected {n_structures}"
            )

    # Compute estimates for all registered weighting methods
    estimates: dict[str, WeightedEstimate] = {}
    for method_name, _, _ in iter_methods():
        trunk_core = trunk_cores.get(method_name) if trunk_cores else None
        root_core = root_cores.get(method_name) if root_cores else None
        parent_core = parent_cores.get(method_name) if parent_cores else None
        estimates[method_name] = compute_weighted_estimate(
            method_name=method_name,
            structure_scores_list=structure_scores_list,
            log_probs=log_probs,
            n_tokens=n_tokens,
            trunk_core=trunk_core,
            root_core=root_core,
            parent_core=parent_core,
        )

    # Calculate trajectory-level estimates (using prob-weighted core as reference)
    prob_estimate = estimates.get("prob")
    if prob_estimate is None:
        raise KeyError(
            f"Missing 'prob' weighting method in estimates for arm '{name}'. "
            f"Available methods: {list(estimates.keys())}"
        )
    if not prob_estimate.core:
        raise ValueError(
            f"Empty core for 'prob' weighting method in arm '{name}'. "
            "Cannot compute trajectory-level estimates."
        )
    core_for_traj = prob_estimate.core
    traj_estimates = [
        TrajectoryEstimate(
            traj_idx=t.traj_idx,
            orientation=list(orientation(t.structure_scores, core_for_traj)),
            deviance=float(deviance(t.structure_scores, core_for_traj, norm="l2")),
        )
        for t in trajectories
    ]

    return ArmEstimate(
        arm_idx=arm_idx,
        name=name,
        trajectories=traj_estimates,
        estimates=estimates,
    )


# ══════════════════════════════════════════════════════════════════════════════
# CORE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


def run_estimation_pipeline(
    data: ScoringData,
    judgment_file: str,
) -> PipelineResult:
    """Run full estimation pipeline on judgment data.

    Computes normativity estimates for all arms:
    1. Groups trajectories by arm
    2. Computes root estimate if present (prompt-only conditioning)
    3. Computes trunk estimate (reference cores for orientation)
    4. Computes branch estimates relative to trunk cores
    5. Computes twig estimates if present

    Arm ordering: root (0) -> trunk (1) -> branches (2+) -> twigs (N+)

    Args:
        data: Loaded judgment data with compliance scores
        judgment_file: Path to judgment file (for output metadata)

    Returns:
        PipelineResult with output and arm estimates
    """
    # Group trajectories by arm
    by_arm = data.group_by_arm()

    # Get arm names in config order
    arm_name_list = data.arm_names if data.arm_names else ["trunk"]

    arms: list[ArmEstimate] = []
    root_cores: dict[str, list[float]] = {}
    trunk_cores: dict[str, list[float]] = {}
    current_arm_idx = 0

    # Process root first (if present) - prompt-only conditioning
    root_trajs = by_arm.get("root", [])
    has_root = len(root_trajs) > 0
    if has_root:
        root_estimate = compute_arm_estimate(current_arm_idx, "root", root_trajs)
        arms.append(root_estimate)
        current_arm_idx += 1
        # Extract root cores
        for method_name, _, _ in iter_methods():
            est = root_estimate.estimates.get(method_name)
            if est:
                root_cores[method_name] = est.core

    # Process trunk (with root_cores if available for deviance metrics)
    trunk_arm_trajs = by_arm.get("trunk", [])
    trunk_arm_estimate = compute_arm_estimate(
        current_arm_idx, "trunk", trunk_arm_trajs,
        root_cores=root_cores if has_root else None,
    )
    arms.append(trunk_arm_estimate)
    current_arm_idx += 1

    # Extract trunk cores for each weighting method
    for method_name, _, _ in iter_methods():
        est = trunk_arm_estimate.estimates.get(method_name)
        if est:
            trunk_cores[method_name] = est.core

    # Track branch cores for twig parent references
    # {branch_name: {method_name: core}}
    branch_cores: dict[str, dict[str, list[float]]] = {}

    # First pass: process branches (need their cores before processing twigs)
    for name in arm_name_list:
        if name in ("trunk", "root"):
            continue
        if classify_arm(name) != ArmKind.BRANCH:
            continue

        trajs = by_arm.get(name, [])
        estimate = compute_arm_estimate(
            current_arm_idx, name, trajs,
            trunk_cores=trunk_cores,
            root_cores=root_cores if has_root else None,
        )
        arms.append(estimate)
        current_arm_idx += 1

        # Store branch cores for twig orientation
        branch_cores[name] = {
            method_name: est.core
            for method_name, est in estimate.estimates.items()
        }

    # Second pass: process twigs (with parent branch cores)
    for name in arm_name_list:
        if classify_arm(name) != ArmKind.TWIG:
            continue

        trajs = by_arm.get(name, [])

        # Look up parent branch cores for twig orientation
        parent_name = get_parent_branch(name)
        parent_cores = branch_cores.get(parent_name) if parent_name else None

        estimate = compute_arm_estimate(
            current_arm_idx, name, trajs,
            trunk_cores=trunk_cores,
            root_cores=root_cores if has_root else None,
            parent_cores=parent_cores,
        )
        arms.append(estimate)
        current_arm_idx += 1

    # Build output
    structure_info = data.get_structure_info()
    arm_scoring = data.compute_arm_scoring()
    continuations_by_arm = data.get_continuations_by_arm()

    output = EstimationOutput.create(
        judgment_file=judgment_file,
        scoring_data=data.scoring_data,
        arms=arms,
        texts=data.get_texts(),
        generation_file=data.generation_file,
        scoring_file=data.scoring_file,
        judge_model=data.judge_model,
        embedding_model=data.embedding_model,
        structure_info=structure_info,
        arm_scoring=arm_scoring,
        continuations_by_arm=continuations_by_arm,
    )

    return PipelineResult(
        output=output,
        arms=arms,
        trunk_cores=trunk_cores,
    )
