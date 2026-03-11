"""Similarity scoring method.

This module implements embedding-based similarity scoring of trajectories
using a sentence embedding model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from src.common.callback_types import LogFn
from src.inference import ModelRunner
from src.inference.embedding_runner import EmbeddingRunner

from ..scoring_method_registry import ScoringMethodParams, register_method

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class SimilarityParams(ScoringMethodParams):
    """Parameters for embedding similarity scoring."""

    embedding_model: str = "all-MiniLM-L6-v2"

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

    # Flatten references for embedding computation
    flat_refs: list[str] = []
    for item in items:
        if isinstance(item, list):
            flat_refs.extend(item)
        else:
            flat_refs.append(item)

    # Get all similarities at once (more efficient)
    all_similarities = embedder.similarities(text=text, references=flat_refs)

    # Build output list and log with bundled structure
    scores: list[float] = []
    sim_idx = 0
    label_prefix = params.label_prefix

    for struct_idx, item in enumerate(items):
        if isinstance(item, list):
            if log_fn:
                log_fn(
                    f"[{label_prefix}{struct_idx + 1}] Bundled ({len(item)} items)"
                )

            for ref in item:
                score = all_similarities[sim_idx]
                scores.append(score)
                sim_idx += 1

                if log_fn:
                    ref_preview = ref[:40] + "..." if len(ref) > 40 else ref
                    log_fn(f"     • {ref_preview} -> {score:.3f}")
        else:
            score = all_similarities[sim_idx]
            scores.append(score)
            sim_idx += 1

            if log_fn:
                ref_preview = item[:40] + "..." if len(item) > 40 else item
                log_fn(f"[{label_prefix}{struct_idx + 1}] {ref_preview} -> {score:.3f}")

    # Similarity doesn't have raw responses
    return scores, [""] * len(scores)
