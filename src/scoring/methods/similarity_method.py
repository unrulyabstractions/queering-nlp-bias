"""Similarity scoring method.

This module implements embedding-based similarity scoring of trajectories
using a sentence embedding model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from src.common.callback_types import LogFn
from src.common.default_config import EMBEDDING_MODEL
from src.inference import ModelRunner
from src.inference.embedding_runner import EmbeddingRunner

from ..scoring_method_registry import ScoringMethodParams, register_method, score_with_bundling

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class SimilarityParams(ScoringMethodParams):
    """Parameters for embedding similarity scoring."""

    embedding_model: str = field(default_factory=lambda: EMBEDDING_MODEL)

    # Registry metadata
    name: ClassVar[str] = "similarity"
    config_key: ClassVar[str] = "similarity_scoring"
    label_prefix: ClassVar[str] = "s"
    requires_runner: ClassVar[bool] = False
    requires_embedder: ClassVar[bool] = True

    _cli_args: ClassVar[dict[str, str]] = {
        "embedding_model": "--embedding-model",
    }


# ══════════════════════════════════════════════════════════════════════════════
# REGISTERED SCORING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════


@register_method(SimilarityParams)
def score_similarity(
    text: str,
    items: list[str | list[str]],
    params: SimilarityParams,
    runner: ModelRunner | None = None,
    embedder: EmbeddingRunner | None = None,
    log_fn: LogFn | None = None,
) -> tuple[list[float], list[str]]:
    """Score text on similarity references.

    Args:
        text: Text to compare
        items: Reference strings from config
        params: Method parameters
        runner: Not used
        embedder: Embedding runner for computing similarities
        log_fn: Optional logging callback

    Returns:
        Tuple of (scores, raw_responses)
    """
    if embedder is None:
        raise ValueError("Similarity scoring requires an embedding runner")

    # Pre-compute all similarities in one batch for efficiency
    flat_refs = _flatten_items(items)
    all_similarities = embedder.similarities(text=text, references=flat_refs)

    # Create an iterator to consume pre-computed similarities
    sim_iter = iter(all_similarities)

    def score_single(ref: str) -> tuple[float, str]:
        return next(sim_iter), ""

    return score_with_bundling(items, score_single, params.label_prefix, log_fn)


def _flatten_items(items: list[str | list[str]]) -> list[str]:
    """Flatten a list of items (strings or lists of strings) into a flat list."""
    flat: list[str] = []
    for item in items:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat
