"""Paper-correct dynamics metrics (Eqs. 4, 8, 9).

Every metric is an orientation/deviance: an *attunement* of a string minus a
*system default* of a reference prefix — never attunement−attunement or
default−default. All are dimension-normalized (||·|| = ||·||_2 / sqrt(dim)).
"""

from __future__ import annotations

from src.common.math.entropy_diversity.structure_aware import normalized_deviance

System = list[float]


def normalized_norm(vector: System) -> float:
    """Dimension-normalized magnitude ||v||_Λ = ||v||_2 / sqrt(dim) (paper Eq. 4)."""
    if not vector:
        return 0.0
    return float(normalized_deviance(vector, [0.0] * len(vector)))


def pull(system_default: System) -> float:
    """||⟨Λ_n⟩(x_p)|| — strength of the normative attractor (magnitude of the default)."""
    return normalized_norm(system_default)


def drift(system_attunement: System, initial_system_default: System) -> float:
    """∂_n(x_p | x_0) = ||Λ_n(x_p) - ⟨Λ_n⟩(x_0)|| — deviance from the initial default."""
    if not system_attunement or not initial_system_default:
        return 0.0
    return float(normalized_deviance(system_attunement, initial_system_default))


def potential(final_system_attunement: System, system_default: System) -> float:
    """∂_n(x_final | x_p) = ||Λ_n(x_final) - ⟨Λ_n⟩(x_p)|| — deviance of the end from the default."""
    if not final_system_attunement or not system_default:
        return 0.0
    return float(normalized_deviance(final_system_attunement, system_default))
