"""Dynamics computation — track the system default and attunement along a trajectory.

At each measured position k we (1) score the prefix to get its realized system
attunement Λ_n(x_p) and (2) sample continuations to estimate the system default
⟨Λ_n⟩(x_p) (the barycenter). From these we compute the paper's deviance-based
pull/drift/potential (see dynamics_metrics). This samples the model at every
measured prefix, so cost scales with positions × samples_per_position — tune
DynamicsConfig accordingly.

Usage:
    result = compute_dynamics(trajectories, scorer, runner, DynamicsConfig())
"""

from __future__ import annotations

from src.common.callback_types import LogFn
from src.common.profiler import profile
from src.inference import ModelRunner
from src.scoring.scorer import Scorer

from .dynamics_metrics import drift, potential, pull
from .dynamics_sampling import estimate_system_default
from .dynamics_types import (
    DynamicsConfig,
    DynamicsResult,
    DynamicsTrajectory,
    PositionScores,
    TrajectoryDynamics,
)


def _measurement_positions(n_tokens: int, step: int) -> list[int]:
    """Token positions to measure: step, 2*step, ..., n_tokens."""
    if n_tokens <= 0:
        return []
    positions = list(range(step, n_tokens + 1, step))
    if n_tokens not in positions:
        positions.append(n_tokens)
    return positions


def _partial_text(text: str, k: int, n_tokens: int) -> str:
    """Prefix of `text` proportional to token position k (character-based approximation)."""
    ratio = k / max(n_tokens, 1)
    return text[: int(len(text) * ratio)]


def _measure_position(
    traj: DynamicsTrajectory,
    k: int,
    scorer: Scorer,
    runner: ModelRunner,
    config: DynamicsConfig,
) -> tuple[list[float], list[float]] | None:
    """Return (system_attunement, system_default) at position k, or None if the prefix is empty."""
    prefix_text = _partial_text(traj.text, k, traj.n_tokens)
    if not prefix_text.strip():
        return None
    system_attunement = scorer.score(prefix_text)
    if not system_attunement:
        return None
    system_default = estimate_system_default(
        runner,
        scorer,
        traj.prompt,
        traj.prefill,
        prefix_text,
        config.samples_per_position,
        config.continuation_max_tokens,
        config.temperature,
    )
    return system_attunement, system_default


def _compute_trajectory_dynamics(
    traj: DynamicsTrajectory,
    scorer: Scorer,
    runner: ModelRunner,
    config: DynamicsConfig,
    log_fn: LogFn | None,
) -> TrajectoryDynamics:
    """Compute dynamics for one trajectory."""
    positions = _measurement_positions(traj.n_tokens, config.step)
    if log_fn:
        log_fn(
            f"  [{traj.traj_idx}] {traj.arm_name}: {traj.n_tokens} tokens, "
            f"{len(positions)} points x {config.samples_per_position} samples"
        )

    # (k, system_attunement Λ_n(x_p), system_default ⟨Λ_n⟩(x_p)) per measured position
    measured: list[tuple[int, list[float], list[float]]] = []
    for k in positions:
        result = _measure_position(traj, k, scorer, runner, config)
        if result is not None:
            attunement, default = result
            measured.append((k, attunement, default))

    if not measured:
        return TrajectoryDynamics(
            traj_idx=traj.traj_idx,
            arm_name=traj.arm_name,
            text=traj.text,
            n_tokens=traj.n_tokens,
            positions=[],
        )

    initial_default = measured[0][2]  # ⟨Λ_n⟩(x_0): drift reference frame
    final_attunement = measured[-1][1]  # Λ_n(x_final): potential subject string

    result_positions: list[PositionScores] = []
    for k, attunement, default in measured:
        p, d, z = (
            pull(default),
            drift(attunement, initial_default),
            potential(final_attunement, default),
        )
        result_positions.append(
            PositionScores(
                k=k,
                system_attunement=attunement,
                system_default=default,
                pull=p,
                drift=d,
                potential=z,
            )
        )
        if log_fn:
            log_fn(f"      @{k:3d}: pull={p:.3f} drift={d:.3f} potential={z:.3f}")

    return TrajectoryDynamics(
        traj_idx=traj.traj_idx,
        arm_name=traj.arm_name,
        text=traj.text,
        n_tokens=traj.n_tokens,
        positions=result_positions,
    )


@profile
def compute_dynamics(
    trajectories: list[DynamicsTrajectory],
    scorer: Scorer,
    runner: ModelRunner,
    config: DynamicsConfig | None = None,
    log_fn: LogFn | None = None,
) -> DynamicsResult:
    """Compute paper-correct dynamics (pull/drift/potential) for trajectories.

    Args:
        trajectories: DynamicsTrajectory objects (each carries prompt + prefill so
            continuations can be sampled from any prefix).
        scorer: Scorer producing the system attunement Λ_n for a text.
        runner: ModelRunner used to sample continuations for the system default.
        config: DynamicsConfig (step, samples_per_position, continuation_max_tokens,
            temperature). Defaults are used when omitted.
        log_fn: Logging callback.

    Returns:
        DynamicsResult with pull/drift/potential (and the attunement + default) at
        each measured position.
    """
    config = config or DynamicsConfig()
    if log_fn:
        log_fn(
            f"DYNAMICS: {len(trajectories)} trajectories, step={config.step}, "
            f"samples/pos={config.samples_per_position}"
        )

    results: list[TrajectoryDynamics] = []
    n_structures = 0
    for traj in trajectories:
        dyn = _compute_trajectory_dynamics(traj, scorer, runner, config, log_fn)
        results.append(dyn)
        if dyn.positions:
            n_structures = len(dyn.positions[0].system_attunement)

    return DynamicsResult(trajectories=results, n_structures=n_structures, step=config.step)
