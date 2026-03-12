"""Node metrics for branching point analysis.

Provides metrics for analyzing branching nodes in token trees.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...base_schema import BaseSchema
from ..distributional_analysis_base import DistributionalAnalysis


@dataclass
class NodeMetrics(DistributionalAnalysis):
    """Metrics at a branching node's vocab distribution."""

    next_token_logprobs: list[float]  # logprobs of candidate tokens at this node
    vocab_entropy: (
        float  # H of full vocab dist at divergent pos — lower = more decisive
    )
    vocab_diversity: float  # D₁ = e^H — effective vocab size at decision point


@dataclass
class NodeAnalysis(BaseSchema):
    """Analysis at a branching node."""

    node_idx: int
    metrics: NodeMetrics
