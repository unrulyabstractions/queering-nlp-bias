"""Parameters for forking paths generation method.

Separated from method implementation to avoid circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from src.common.default_config import (
    FORKING_MAX_ALTERNATES,
    FORKING_MIN_ENTROPY,
    FORKING_MIN_PROB,
    FORKING_SAMPLES_PER_FORK,
)

from ..generation_method_registry import GenerationMethodParams


@dataclass
class ForkingParams(GenerationMethodParams):
    """Parameters for forking paths generation."""

    max_alternates: int = field(default_factory=lambda: FORKING_MAX_ALTERNATES)
    min_prob: float = field(default_factory=lambda: FORKING_MIN_PROB)
    min_entropy: float = field(default_factory=lambda: FORKING_MIN_ENTROPY)
    samples_per_fork: int = field(default_factory=lambda: FORKING_SAMPLES_PER_FORK)

    name: ClassVar[str] = "forking-paths"

    _cli_args: ClassVar[dict[str, str]] = {
        "max_alternates": "--max-alternates-per-position",
        "min_prob": "--min-prob-for-alternate",
        "min_entropy": "--min-entropy-to-fork",
        "samples_per_fork": "--samples-per-fork",
    }
