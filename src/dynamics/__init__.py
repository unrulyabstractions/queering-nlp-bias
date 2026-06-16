"""Dynamics module: tracks how a trajectory's system attunement Λ_n(x_p) and
system default ⟨Λ_n⟩(x_p) (estimated by sampling continuations) evolve token by
token, reporting paper-correct pull, drift, and potential at each position."""

from src.common.auto_export import auto_export

__all__ = auto_export(__file__, __name__, globals())
