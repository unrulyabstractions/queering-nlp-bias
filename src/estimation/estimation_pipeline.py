"""Core estimation pipeline logic.

This module contains the core algorithms for normativity estimation,
independent of logging and script utilities.

Key function: compute_arm_estimate() - computes arm-level statistics from
trajectory data using all registered weighting methods.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.common.math.entropy_diversity.structure_aware import (
    deviance,
    deviance_variance,
    expected_deviance,
    expected_orientation,
    generalized_system_core,
    orientation,
)

# Import methods to trigger registration
from . import methods as _methods  # noqa: F401
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
    reference_core: list[float] | None = None,
) -> WeightedEstimate:
    """Compute estimation using a specific weighting method.

    Args:
        method_name: Name of the weighting method (e.g., "prob", "inv-ppl")
        structure_scores_list: List of compliance vectors per trajectory
        log_probs: Log probabilities for each trajectory
        n_tokens: Number of tokens per trajectory
        reference_core: Optional core to compute orientation against

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

    # Compute metrics relative to reference (trunk) core
    ref_core = reference_core if reference_core is not None else core

    # E[θ|T] - orientation relative to trunk
    orient_avg = expected_orientation(structure_scores_list, ref_core, weights=weights)

    # ||E[θ|T]|| - L2 norm of orientation (distance between cores)
    orient_norm = math.sqrt(sum(x * x for x in orient_avg)) if orient_avg else 0.0

    # E[d|T] - deviance relative to trunk
    dev_avg_trunk = expected_deviance(
        structure_scores_list, ref_core, weights=weights, norm="l2"
    )

    # E[Δd] = E[d|branch] - E[d|trunk]
    dev_delta = dev_avg - dev_avg_trunk

    # Compute all named core variants
    core_variants = compute_core_variants(structure_scores_list, weights)

    return WeightedEstimate(
        method_name=method_name,
        core=core,
        deviance_avg=dev_avg,
        deviance_var=dev_var,
        deviance_avg_trunk=dev_avg_trunk,
        deviance_delta=dev_delta,
        orientation_avg=orient_avg,
        orientation_norm=orient_norm,
        core_variants=core_variants,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE RESULT
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class EstimationResult:
    """Result of running the estimation pipeline."""

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
    reference_cores: dict[str, list[float]] | None = None,
) -> ArmEstimate:
    """Compute arm-level estimate from trajectory structure scores.

    This is the primary computation function for arm estimation. It:
    1. Iterates over all registered weighting methods
    2. Computes WeightedEstimate for each method
    3. Stores results in ArmEstimate.estimates dict

    Args:
        arm_idx: Index of this arm (0=trunk, 1+=branches)
        name: Name of this arm (e.g., "trunk", "branch_1")
        trajectories: Trajectories with structure_scores and conditional log probs
        reference_cores: Optional dict of reference cores by method name

    Returns:
        ArmEstimate with all computed statistics across all weighting methods
    """
    n_trajs = len(trajectories)

    # Handle empty arm: return neutral estimate
    if n_trajs == 0:
        estimates = {
            method_name: WeightedEstimate.empty(method_name)
            for method_name, _, _ in iter_methods()
        }
        return ArmEstimate(
            arm_idx=arm_idx,
            name=name,
            trajectories=[],
            estimates=estimates,
        )

    # Extract data from trajectories
    structure_scores_list = [t.structure_scores for t in trajectories]
    log_probs = [t.conditional_logprobs.get(name, 0.0) for t in trajectories]
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
        ref_core = (
            reference_cores.get(method_name) if reference_cores is not None else None
        )
        estimates[method_name] = compute_weighted_estimate(
            method_name=method_name,
            structure_scores_list=structure_scores_list,
            log_probs=log_probs,
            n_tokens=n_tokens,
            reference_core=ref_core,
        )

    # Calculate trajectory-level estimates (using prob-weighted core as reference)
    prob_core = estimates.get("prob")
    core_for_traj = prob_core.core if prob_core else []
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
) -> EstimationResult:
    """Run full estimation pipeline on judgment data.

    Computes normativity estimates for all arms (trunk + branches):
    1. Groups trajectories by arm
    2. Computes trunk-only estimate (reference cores)
    3. Computes trunk-all estimate (all trajectories)
    4. Computes branch estimates relative to trunk cores

    Args:
        data: Loaded judgment data with compliance scores
        judgment_file: Path to judgment file (for output metadata)

    Returns:
        EstimationResult with output and arm estimates
    """
    # Group trajectories by arm
    by_arm = data.group_by_arm()

    # Get branch names in config order
    branch_names = data.branches if data.branches else ["trunk"]

    arms: list[ArmEstimate] = []
    trunk_cores: dict[str, list[float]] = {}

    # Process trunk (arm-only trajectories) first as reference
    trunk_arm_trajs = by_arm.get("trunk", [])
    trunk_arm_estimate = compute_arm_estimate(0, "trunk", trunk_arm_trajs)
    arms.append(trunk_arm_estimate)

    # Extract trunk cores for each weighting method
    for method_name, _, _ in iter_methods():
        est = trunk_arm_estimate.estimates.get(method_name)
        if est:
            trunk_cores[method_name] = est.core

    # Process all_arms with all trajectories pooled
    all_trajs = [t for trajs in by_arm.values() for t in trajs]
    all_arms_estimate = compute_arm_estimate(0, "all_arms", all_trajs, trunk_cores)
    arms.append(all_arms_estimate)

    # Process branches
    for idx, name in enumerate(branch_names):
        if name == "trunk":
            continue

        trajs = by_arm.get(name, [])
        estimate = compute_arm_estimate(idx, name, trajs, trunk_cores)
        arms.append(estimate)

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

    return EstimationResult(
        output=output,
        arms=arms,
        trunk_cores=trunk_cores,
    )
