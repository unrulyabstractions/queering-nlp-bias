"""Dynamics computation - score partial text at each token position.

Usage:
    scorer = Scorer(config)
    result = compute_dynamics(trajectories, scorer, step=4)
"""

from __future__ import annotations

from src.common.callback_types import LogFn
from src.common.default_config import DYNAMICS_STEP
from src.common.math.vector_utils import l2_distance, l2_norm
from src.common.profiler import profile
from src.scoring import ScoringConfig
from src.scoring.scorer import Scorer

from .dynamics_types import DynamicsResult, PositionScores, TrajectoryDynamics


def _measurement_positions(n_tokens: int, step: int) -> list[int]:
    """Token positions to measure: step, 2*step, ..., n_tokens."""
    if n_tokens <= 0:
        return []
    positions = list(range(step, n_tokens + 1, step))
    if n_tokens not in positions:
        positions.append(n_tokens)
    return positions


def _compute_trajectory_dynamics(
    traj_idx: int,
    arm_name: str,
    text: str,
    n_tokens: int,
    scorer: Scorer,
    step: int,
    log_fn: LogFn | None,
) -> TrajectoryDynamics:
    """Compute dynamics for one trajectory."""
    positions = _measurement_positions(n_tokens, step)

    if log_fn:
        log_fn(f"  [{traj_idx}] {arm_name}: {n_tokens} tokens, {len(positions)} points")

    # Score at each position
    scored: list[tuple[int, list[float]]] = []
    for k in positions:
        ratio = k / max(n_tokens, 1)
        partial = text[: int(len(text) * ratio)]
        if not partial.strip():
            continue
        scores = scorer.score(partial)
        if scores:
            scored.append((k, scores))

    if not scored:
        return TrajectoryDynamics(
            traj_idx=traj_idx, arm_name=arm_name, text=text, n_tokens=n_tokens, positions=[]
        )

    initial, final = scored[0][1], scored[-1][1]
    result_positions: list[PositionScores] = []

    for k, scores in scored:
        pull, drift, horizon = l2_norm(scores), l2_distance(scores, initial), l2_distance(scores, final)
        result_positions.append(PositionScores(k=k, scores=scores, pull=pull, drift=drift, horizon=horizon))
        if log_fn:
            log_fn(f"      @{k:3d}: pull={pull:.3f} drift={drift:.3f} horizon={horizon:.3f}")

    return TrajectoryDynamics(
        traj_idx=traj_idx, arm_name=arm_name, text=text, n_tokens=n_tokens, positions=result_positions
    )


@profile
def compute_dynamics(
    trajectories: list[tuple[int, str, str, int]],
    scorer: Scorer | ScoringConfig,
    step: int = DYNAMICS_STEP,
    log_fn: LogFn | None = None,
) -> DynamicsResult:
    """Compute dynamics for trajectories.

    Args:
        trajectories: List of (traj_idx, arm_name, text, n_tokens)
        scorer: Scorer instance or ScoringConfig (Scorer created if config passed)
        step: Measure every N tokens
        log_fn: Logging callback

    Returns:
        DynamicsResult with pull/drift/horizon at each position
    """
    if isinstance(scorer, ScoringConfig):
        scorer = Scorer(scorer)

    if log_fn:
        log_fn(f"DYNAMICS: {len(trajectories)} trajectories, step={step}")

    results: list[TrajectoryDynamics] = []
    n_structures = 0

    for traj_idx, arm_name, text, n_tokens in trajectories:
        dyn = _compute_trajectory_dynamics(traj_idx, arm_name, text, n_tokens, scorer, step, log_fn)
        results.append(dyn)
        if dyn.positions:
            n_structures = len(dyn.positions[0].scores)

    return DynamicsResult(trajectories=results, n_structures=n_structures, step=step)
