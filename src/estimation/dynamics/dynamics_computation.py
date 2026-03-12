"""Dynamics computation for drift and horizon analysis.

Drift y(k): deviance of PARTIAL text (at token k) relative to root core
  - Re-run scoring on partial text
  - Many points along the trajectory

Horizon z(arm): deviance of FULL text relative to arm's core
  - Use pre-computed full trajectory scores
  - One point per arm, plotted at arm's prefix token count

PIPE, NOT PARSE: All data flows through typed objects.
"""

from __future__ import annotations

import math

from src.common.callback_types import LogFn
from src.common.math.entropy_diversity.structure_aware import deviance, orientation
from src.estimation import ScoringData
from src.estimation.arm_types import get_arm_ancestry
from src.estimation.estimation_experiment_types import EstimationResult
from src.estimation.estimation_structure import TrajectoryScoringData
from src.inference import ModelRunner
from src.inference.embedding_runner import EmbeddingRunner
from src.scoring import ScoringConfig

from .dynamics_types import (
    DriftPoint,
    DynamicsResult,
    HorizonPoint,
    PullPoint,
    TrajectoryDynamics,
)
from .logging import (
    log_drift_point,
    log_dynamics_header,
    log_dynamics_result,
    log_horizon_point,
    log_pull_point,
    log_trajectory_start,
    log_trajectory_summary,
)


def _select_representative_trajectories(
    trajectories_by_arm: dict[str, list[TrajectoryScoringData]],
    arm_names: list[str],
    trajs_per_arm: int,
) -> list[TrajectoryScoringData]:
    """Select representative trajectories from each arm."""
    selected: list[TrajectoryScoringData] = []
    for arm_name in arm_names:
        arm_trajs = trajectories_by_arm.get(arm_name, [])
        selected.extend(arm_trajs[:trajs_per_arm])
    return selected


def _estimate_token_count(text: str) -> int:
    """Estimate token count from text (rough: ~4 chars per token)."""
    return max(1, len(text) // 4)


def _score_partial_text(
    text: str,
    config: ScoringConfig,
    runner: ModelRunner | None,
    embedder: EmbeddingRunner | None,
) -> list[float]:
    """Score a partial text string, returning structure scores."""
    from src.scoring.scoring_method_registry import get_method

    all_scores: list[float] = []

    for method_name in config.get_active_methods():
        items = config.get_method_items(method_name)
        if not items:
            continue

        params = config.get_scoring_params(method_name)
        score_fn = get_method(method_name)

        scores, _ = score_fn(text, items, params, runner, embedder, None)

        # Average bundled items to get one score per structure
        score_idx = 0
        for item in items:
            if isinstance(item, list):
                bundle_scores = scores[score_idx : score_idx + len(item)]
                avg = sum(bundle_scores) / len(bundle_scores) if bundle_scores else 0.0
                all_scores.append(avg)
                score_idx += len(item)
            else:
                all_scores.append(scores[score_idx] if score_idx < len(scores) else 0.0)
                score_idx += 1

    return all_scores


def _compute_measurement_positions(n_tokens: int, step: int = 2) -> list[int]:
    """Compute token positions for drift measurements.

    Args:
        n_tokens: Total tokens in continuation
        step: Measure every N tokens (default 2)
    """
    if n_tokens <= 0:
        return []

    positions = list(range(step, n_tokens + 1, step))

    # Always include final position
    if n_tokens not in positions:
        positions.append(n_tokens)

    return positions


def _compute_deviance(scores: list[float], core: list[float]) -> float:
    """Compute deviance of scores relative to a core."""
    if len(scores) != len(core):
        return 0.0
    theta = list(orientation(scores, core))
    return deviance(theta, [0.0] * len(theta), norm="l2")


def _compute_trajectory_dynamics(
    traj: TrajectoryScoringData,
    root_core: list[float],
    arm_cores: dict[str, list[float]],
    arm_prefix_tokens: dict[str, int],
    config: ScoringConfig,
    runner: ModelRunner | None,
    embedder: EmbeddingRunner | None,
    drift_step: int,
    log_fn: LogFn | None = None,
) -> TrajectoryDynamics:
    """Compute dynamics for a single trajectory."""
    positions = _compute_measurement_positions(traj.n_continuation_tokens, drift_step)

    log_trajectory_start(
        traj_idx=traj.traj_idx,
        arm_name=traj.branch,
        n_tokens=traj.n_continuation_tokens,
        n_drift_points=len(positions),
        n_horizon_points=len(arm_cores),
        log_fn=log_fn,
    )

    # === DRIFT: Re-score partial text at various k, compare to root core ===
    drift_points: list[DriftPoint] = []

    for k in positions:
        char_ratio = k / max(traj.n_continuation_tokens, 1)
        char_pos = int(len(traj.text) * char_ratio)
        partial_text = traj.text[:char_pos]

        if not partial_text.strip():
            continue

        partial_scores = _score_partial_text(partial_text, config, runner, embedder)

        if partial_scores:
            drift_dev = _compute_deviance(partial_scores, root_core)

            drift_points.append(
                DriftPoint(
                    token_position=k,
                    partial_scores=partial_scores,
                    deviance=drift_dev,
                )
            )

            log_drift_point(
                position=k,
                n_tokens=traj.n_continuation_tokens,
                scores=partial_scores,
                deviance=drift_dev,
                log_fn=log_fn,
            )

    # === HORIZON & PULL: Only for arms on this trajectory's path ===
    # A trajectory on branch_1 passes through: root -> trunk -> branch_1
    # A trajectory on twig_2_b1 passes through: root -> trunk -> branch_1 -> twig_2_b1
    full_scores = traj.structure_scores
    ancestry = get_arm_ancestry(traj.branch)

    horizon_points: list[HorizonPoint] = []
    pull_points: list[PullPoint] = []

    for arm_name in ancestry:
        arm_core = arm_cores.get(arm_name)
        if arm_core is None:
            continue

        prefix_tokens = arm_prefix_tokens.get(arm_name, 0)

        # Horizon: deviance of full trajectory from this arm's core
        horizon_dev = _compute_deviance(full_scores, arm_core)
        horizon_points.append(
            HorizonPoint(
                arm_name=arm_name,
                arm_prefix_tokens=prefix_tokens,
                deviance=horizon_dev,
            )
        )
        log_horizon_point(
            arm_name=arm_name,
            prefix_tokens=prefix_tokens,
            deviance=horizon_dev,
            is_own_arm=(arm_name == traj.branch),
            log_fn=log_fn,
        )

        # Pull: l2 norm of arm's core
        pull_val = math.sqrt(sum(v * v for v in arm_core))
        pull_points.append(
            PullPoint(
                arm_name=arm_name,
                arm_prefix_tokens=prefix_tokens,
                pull=pull_val,
            )
        )
        log_pull_point(
            arm_name=arm_name,
            prefix_tokens=prefix_tokens,
            pull=pull_val,
            log_fn=log_fn,
        )

    result = TrajectoryDynamics(
        traj_idx=traj.traj_idx,
        arm_name=traj.branch,
        full_text=traj.text,
        n_tokens=traj.n_continuation_tokens,
        full_scores=full_scores,
        drift_points=drift_points,
        horizon_points=horizon_points,
        pull_points=pull_points,
    )

    log_trajectory_summary(result, log_fn=log_fn)

    return result


def compute_dynamics(
    estimation_result: EstimationResult,
    scoring_config: ScoringConfig,
    runner: ModelRunner | None = None,
    embedder: EmbeddingRunner | None = None,
    trajs_per_arm: int = 1,
    drift_step: int = 2,
    log_fn: LogFn | None = None,
) -> DynamicsResult:
    """Compute dynamics for representative trajectories.

    Drift: re-score partial text at multiple k, compare to root core
    Horizon: use full trajectory scores, compare to each arm's core

    Args:
        estimation_result: Result from estimation pipeline
        scoring_config: Scoring configuration
        runner: Model runner for LLM-based scoring
        embedder: Embedding runner for embedding-based scoring
        trajs_per_arm: Number of trajectories per arm to analyze
        drift_step: Measure drift every N tokens (default 2)
        log_fn: Optional logging callback

    Returns:
        DynamicsResult with drift and horizon data
    """
    # Load scoring data
    scoring_data = ScoringData.load(estimation_result.paths.judgment)
    arm_names = scoring_data.arm_names if scoring_data.arm_names else ["root", "trunk"]
    arm_texts = scoring_data.arm_texts

    # Build arm cores and estimate prefix token counts
    arm_cores: dict[str, list[float]] = {}
    arm_prefix_tokens: dict[str, int] = {}

    for arm in estimation_result.arms:
        core = arm.get_core("prob")
        if core:
            arm_cores[arm.name] = core
            # Estimate token count from arm text
            arm_text = arm_texts.get(arm.name, "")
            arm_prefix_tokens[arm.name] = _estimate_token_count(arm_text)

    # Root has 0 prefix tokens
    arm_prefix_tokens["root"] = 0

    root_core = arm_cores.get("root", [])
    if not root_core:
        raise ValueError("No root core found")

    n_arms = len(arm_cores)

    # Get typed trajectories grouped by arm
    trajectories_by_arm = scoring_data.group_by_arm()

    # Select representative trajectories
    selected = _select_representative_trajectories(
        trajectories_by_arm, arm_names, trajs_per_arm
    )

    # Log header
    log_dynamics_header(
        n_trajectories=len(selected),
        n_arms=n_arms,
        drift_step=drift_step,
        log_fn=log_fn,
    )

    # Compute dynamics for each trajectory
    traj_dynamics: list[TrajectoryDynamics] = []
    for traj in selected:
        dynamics = _compute_trajectory_dynamics(
            traj=traj,
            root_core=root_core,
            arm_cores=arm_cores,
            arm_prefix_tokens=arm_prefix_tokens,
            config=scoring_config,
            runner=runner,
            embedder=embedder,
            drift_step=drift_step,
            log_fn=log_fn,
        )
        traj_dynamics.append(dynamics)

    result = DynamicsResult(
        root_core=root_core,
        arm_cores=arm_cores,
        arm_prefix_tokens=arm_prefix_tokens,
        trajectories=traj_dynamics,
    )

    log_dynamics_result(result, log_fn=log_fn)

    return result
