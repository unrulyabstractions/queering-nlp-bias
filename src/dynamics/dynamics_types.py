"""Data types for dynamics analysis.

This module tracks how a trajectory evolves token by token. At each measured
position k it tracks BOTH paper quantities, kept strictly distinct:

- the realized system attunement Λ_n(x_p) — score of the prefix text, and
- the system default ⟨Λ_n⟩(x_p) — barycenter estimated by sampling continuations.

From these it computes the paper's deviance-based metrics (Eqs. 8-9), all
dimension-normalized (||·|| = ||·||_2 / sqrt(dim)):

- Pull:      ||⟨Λ_n⟩(x_p)||                  magnitude of the system default
- Drift:     ||Λ_n(x_p) - ⟨Λ_n⟩(x_0)||       current attunement vs the initial default
- Potential: ||Λ_n(x_final) - ⟨Λ_n⟩(x_p)||   final attunement vs the current default
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common.base_schema import BaseSchema
from src.common.default_config import (
    DYNAMICS_CONTINUATION_MAX_TOKENS,
    DYNAMICS_SAMPLES_PER_POSITION,
    DYNAMICS_STEP,
    TEMPERATURE,
)


@dataclass
class DynamicsTrajectory(BaseSchema):
    """One trajectory to analyze, with the context needed to sample continuations.

    prompt + prefill condition the model exactly as during generation; `text` is
    the (string-selected) generated continuation whose prefixes we walk.
    """

    traj_idx: int
    arm_name: str
    prompt: str  # The prompt that conditioned generation (filled template in template mode)
    prefill: str  # Arm prefill (trunk/branch text) prepended before the generated text
    text: str  # The generated continuation (string-selected)
    n_tokens: int  # Number of generated tokens in `text`


@dataclass
class DynamicsConfig(BaseSchema):
    """Parameters controlling how the dynamics (and its sampling) are computed."""

    step: int = DYNAMICS_STEP  # Measure every N tokens
    samples_per_position: int = DYNAMICS_SAMPLES_PER_POSITION  # Continuations per prefix
    continuation_max_tokens: int = DYNAMICS_CONTINUATION_MAX_TOKENS  # Tokens per continuation
    temperature: float = TEMPERATURE  # Sampling temperature (1.0 → true Monte-Carlo of ⟨Λ_n⟩)


@dataclass
class PositionScores(BaseSchema):
    """System attunement, system default, and dynamics metrics at one token position."""

    k: int  # Token position
    system_attunement: list[float]  # Λ_n(x_p): structure scores of the prefix text
    system_default: list[float]  # ⟨Λ_n⟩(x_p): sampled barycenter at the prefix
    pull: float  # ||⟨Λ_n⟩(x_p)||: dimension-normalized magnitude of the system default
    drift: float  # ||Λ_n(x_p) - ⟨Λ_n⟩(x_0)||: deviance from the initial default
    potential: float  # ||Λ_n(x_final) - ⟨Λ_n⟩(x_p)||: deviance of the final outcome from the default


@dataclass
class TrajectoryDynamics(BaseSchema):
    """Dynamics for a single trajectory."""

    traj_idx: int
    arm_name: str
    text: str
    n_tokens: int
    positions: list[PositionScores]  # Scores at each measured position

    @property
    def pull_series(self) -> list[tuple[int, float]]:
        """(k, pull) pairs for plotting."""
        return [(p.k, p.pull) for p in self.positions]

    @property
    def drift_series(self) -> list[tuple[int, float]]:
        """(k, drift) pairs for plotting."""
        return [(p.k, p.drift) for p in self.positions]

    @property
    def potential_series(self) -> list[tuple[int, float]]:
        """(k, potential) pairs for plotting."""
        return [(p.k, p.potential) for p in self.positions]


@dataclass
class DynamicsResult(BaseSchema):
    """Result of dynamics computation."""

    trajectories: list[TrajectoryDynamics] = field(default_factory=list)
    n_structures: int = 0  # Number of structure dimensions
    step: int = DYNAMICS_STEP  # Token step between measurements
