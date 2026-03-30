"""Estimation pipeline - compute normativity metrics from scored trajectories.

Usage:
    data = ScoringData.load("scoring.json")
    result = run_estimation_pipeline(data, "scoring.json")
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
from src.common.profiler import profile

from . import methods as _methods  # noqa: F401 - triggers registration
from .arm_types import ArmKind, classify_arm, get_parent_branch
from .estimation_core_types import NAMED_CORES, CoreVariant
from .estimation_output import EstimationOutput
from .estimation_scoring_data import ScoringData
from .estimation_structure import ArmEstimate, TrajectoryEstimate, TrajectoryScoringData
from .estimation_weighted_types import WeightedEstimate
from .weighting_method_registry import get_default_params, get_method, iter_methods


@dataclass
class EstimationPipelineResult:
    """Result of estimation pipeline."""

    output: EstimationOutput
    arms: list[ArmEstimate]
    trunk_cores: dict[str, list[float]]


# ══════════════════════════════════════════════════════════════════════════════
# CORE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════


def _compute_core_variants(
    scores: list[list[float]], weights: list[float]
) -> list[CoreVariant]:
    """Compute all named core variants (different q,r params)."""
    variants = []
    for name, q, r, desc in NAMED_CORES:
        try:
            core = generalized_system_core(scores, weights, q=q, r=r)
            variants.append(
                CoreVariant(
                    name=name,
                    q=q,
                    r=r,
                    description=desc,
                    core=core,
                    deviance_avg=expected_deviance(
                        scores, core, weights=weights, norm="l2"
                    ),
                    deviance_var=deviance_variance(
                        scores, core, weights=weights, norm="l2"
                    ),
                )
            )
        except (ValueError, ZeroDivisionError, OverflowError):
            pass
    return variants


def _compute_weighted_estimate(
    method_name: str,
    scores: list[list[float]],
    log_probs: list[float],
    n_tokens: list[int],
    ref_cores: dict[str, list[float] | None],
) -> WeightedEstimate:
    """Compute estimate for one weighting method."""
    weights = get_method(method_name)(
        log_probs, n_tokens, get_default_params(method_name)
    )
    core = generalized_system_core(scores, weights, q=1.0, r=1.0)

    if not core:
        raise ValueError(f"Empty core for method '{method_name}'")

    trunk_core = ref_cores.get("trunk")
    root_core = ref_cores.get("root")
    parent_core = ref_cores.get("parent")

    orient_root, norm_root = compute_orientation_vector(core, root_core)
    orient_trunk, norm_trunk = compute_orientation_vector(core, trunk_core)
    orient_parent, norm_parent = compute_orientation_vector(core, parent_core)

    return WeightedEstimate(
        method_name=method_name,
        core=core,
        deviance_avg=expected_deviance(scores, core, weights=weights, norm="l2"),
        deviance_var=deviance_variance(scores, core, weights=weights, norm="l2"),
        deviance_avg_root=expected_deviance(
            scores, root_core, weights=weights, norm="l2"
        )
        if root_core
        else 0.0,
        deviance_avg_trunk=expected_deviance(
            scores, trunk_core, weights=weights, norm="l2"
        )
        if trunk_core
        else 0.0,
        excess_deviance_avg=expected_excess_deviance(scores, core, weights=weights),
        deficit_deviance_avg=expected_deficit_deviance(scores, core, weights=weights),
        mutual_deviance_avg=expected_mutual_deviance(scores, core, weights=weights),
        core_diversity=core_diversity(core),
        orientation_from_root=orient_root,
        orientation_norm_from_root=norm_root,
        orientation_from_trunk=orient_trunk,
        orientation_norm_from_trunk=norm_trunk,
        orientation_from_parent=orient_parent,
        orientation_norm_from_parent=norm_parent,
        core_variants=_compute_core_variants(scores, weights),
    )


def _compute_arm_estimate(
    arm_idx: int,
    name: str,
    trajs: list[TrajectoryScoringData],
    ref_cores: dict[str, dict[str, list[float]]],
) -> ArmEstimate:
    """Compute estimate for one arm."""
    if not trajs:
        raise ValueError(f"No trajectories for arm '{name}'")

    scores = [t.structure_scores for t in trajs]
    log_probs = [t.conditional_logprobs[name] for t in trajs]
    n_tokens = [t.n_generated_tokens for t in trajs]

    # Compute estimates for all weighting methods
    estimates: dict[str, WeightedEstimate] = {}
    for method_name, _, _ in iter_methods():
        method_refs = {
            "trunk": ref_cores.get("trunk", {}).get(method_name),
            "root": ref_cores.get("root", {}).get(method_name),
            "parent": ref_cores.get("parent", {}).get(method_name),
        }
        estimates[method_name] = _compute_weighted_estimate(
            method_name, scores, log_probs, n_tokens, method_refs
        )

    # Per-trajectory estimates using prob-weighted core
    prob_core = estimates["prob"].core
    traj_estimates = [
        TrajectoryEstimate(
            traj_idx=t.traj_idx,
            orientation=list(orientation(t.structure_scores, prob_core)),
            deviance=float(deviance(t.structure_scores, prob_core, norm="l2")),
        )
        for t in trajs
    ]

    return ArmEstimate(
        arm_idx=arm_idx, name=name, trajectories=traj_estimates, estimates=estimates
    )


def _extract_cores(estimate: ArmEstimate) -> dict[str, list[float]]:
    """Extract cores from estimate for each weighting method."""
    return {name: est.core for name, est in estimate.estimates.items()}


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


@profile
def run_estimation_pipeline(
    data: ScoringData, judgment_file: str
) -> EstimationPipelineResult:
    """Run estimation pipeline on scored trajectories."""
    by_arm = data.group_by_arm()
    arm_names = data.arm_names or ["trunk"]

    arms: list[ArmEstimate] = []
    ref_cores: dict[str, dict[str, list[float]]] = {"root": {}, "trunk": {}}
    branch_cores: dict[str, dict[str, list[float]]] = {}
    idx = 0

    # Process root if present
    if by_arm.get("root"):
        est = _compute_arm_estimate(idx, "root", by_arm["root"], ref_cores)
        arms.append(est)
        ref_cores["root"] = _extract_cores(est)
        idx += 1

    # Process trunk if present in data
    if by_arm.get("trunk"):
        est = _compute_arm_estimate(idx, "trunk", by_arm["trunk"], ref_cores)
        arms.append(est)
        ref_cores["trunk"] = _extract_cores(est)
        idx += 1

    # Process branches
    for name in arm_names:
        if name in ("trunk", "root") or classify_arm(name) != ArmKind.BRANCH:
            continue
        est = _compute_arm_estimate(idx, name, by_arm.get(name, []), ref_cores)
        arms.append(est)
        branch_cores[name] = _extract_cores(est)
        idx += 1

    # Process twigs
    for name in arm_names:
        if classify_arm(name) != ArmKind.TWIG:
            continue
        parent = get_parent_branch(name)
        twig_refs = {**ref_cores, "parent": branch_cores.get(parent, {})}
        est = _compute_arm_estimate(idx, name, by_arm.get(name, []), twig_refs)
        arms.append(est)
        idx += 1

    output = EstimationOutput.create(
        judgment_file=judgment_file,
        generation_file=data.generation_file,
        scoring_file=data.scoring_file,
        judge_model=data.judge_model,
        embedding_model=data.embedding_model,
        structures=data.get_structure_info(),
        arms=arms,
        arm_scoring=data.compute_arm_scoring(),
    )

    return EstimationPipelineResult(
        output=output, arms=arms, trunk_cores=ref_cores["trunk"]
    )
