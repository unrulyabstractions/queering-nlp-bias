"""Core constants and types for normativity estimation.

This module defines the fundamental constants and type definitions used
throughout the estimation pipeline, including the generalized core
parameterizations (q, r) and related helper functions.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common.base_schema import BaseSchema

# ══════════════════════════════════════════════════════════════════════════════
# GENERALIZED CORE PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# Named (q, r) parameterizations for generalized cores
# Reference: https://www.unrulyabstractions.com/pdfs/diversity.pdf
#
# r controls which trajectories get attention:
#   r=1: actual distribution, r=0: uniform, r=inf: mode, r=-inf: anti-mode (rarest)
# q controls how compliance values are aggregated:
#   q=1: arithmetic, q=0: geometric, q=-1: harmonic, q=inf: max, q=-inf: min

# Number of core variants to display (5 paper cases + 4 user-requested combos)
NUM_DISPLAYED_VARIANTS = 9

NAMED_CORES: list[tuple[str, float, float, str]] = [
    # First NUM_DISPLAYED_VARIANTS are shown: paper's 5 + user-requested (q,r) combinations
    ("standard", 1.0, 1.0, "<alpha> standard expected compliance"),
    ("uniform", 1.0, 0.0, "uniform avg over support"),
    ("mode", 1.0, float("inf"), "compliance of mode"),
    ("max", float("inf"), 1.0, "max compliance in support"),
    ("mode_min", float("-inf"), float("inf"), "min compliance among modes"),
    ("confident", 1.0, 2.0, "confident core (q=1, r=2)"),
    ("rms", 2.0, 1.0, "root-mean-square (q=2, r=1)"),
    ("rms_conf", 2.0, 2.0, "RMS confident (q=2, r=2)"),
    ("top_heavy", 1.0, 100.0, "heavily mode-biased (q=1, r=100)"),
    # Additional cases - varying r (which trajectories)
    ("antimode", 1.0, float("-inf"), "compliance of rarest (anti-mode)"),
    ("inverse", 1.0, -1.0, "inverse probability weighting"),
    # Additional cases - varying q (how to aggregate)
    ("geometric", 0.0, 1.0, "geometric mean (sensitive to exclusion)"),
    ("harmonic", -1.0, 1.0, "harmonic mean (penalizes low compliance)"),
    # Combinations for contrasting dominant vs. rare
    ("rare_max", float("inf"), float("-inf"), "max compliance among rarest"),
    ("actual_min", float("-inf"), 1.0, "min compliance under actual dist"),
    ("rare_min", float("-inf"), float("-inf"), "min compliance among rarest"),
    ("rare_geometric", 0.0, float("-inf"), "geometric mean in long tail"),
]


# ══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class CoreParams(BaseSchema):
    """Parameters (q, r) for a generalized core."""

    q: float
    r: float


@dataclass
class CoreVariant(BaseSchema):
    """A generalized core with specific (q, r) parameterization."""

    name: str  # e.g., "standard", "antimode"
    q: float  # power mean order
    r: float  # escort order
    description: str  # human-readable description
    core: list[float]  # computed core values <Lambda_n>_{q,r}
    deviance_avg: float  # E[d_n] relative to this core
    deviance_var: float  # Var[d_n] relative to this core


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
