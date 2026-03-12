"""Count occurrences scoring method.

This module implements a simple word occurrence frequency scoring method.
It counts how many times specified words/phrases appear in the text
and returns the ratio: (# words found) / (# total words in text).

This is a lightweight scoring method that requires no LLM or embeddings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

from src.common.callback_types import LogFn
from src.inference import ModelRunner
from src.inference.embedding_runner import EmbeddingRunner

from ..scoring_method_registry import ScoringMethodParams, register_method, score_with_bundling

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class CountOccurrencesParams(ScoringMethodParams):
    """Parameters for count occurrences scoring."""

    case_sensitive: bool = False

    # Registry metadata
    name: ClassVar[str] = "count-occurrences"
    config_key: ClassVar[str] = "count_occurrences"
    label_prefix: ClassVar[str] = "o"
    requires_runner: ClassVar[bool] = False
    requires_embedder: ClassVar[bool] = False

    _cli_args: ClassVar[dict[str, str]] = {
        "case_sensitive": "--case-sensitive",
    }


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════


def count_words(text: str) -> int:
    """Count total words in text."""
    words = [w for w in re.split(r"\s+", text) if w]
    return len(words)


def count_occurrences(text: str, pattern: str, case_sensitive: bool = False) -> int:
    """Count occurrences of a word/phrase in text."""
    flags = 0 if case_sensitive else re.IGNORECASE
    if " " in pattern:
        regex_pattern = re.escape(pattern)
    else:
        regex_pattern = r"\b" + re.escape(pattern) + r"\b"
    matches = re.findall(regex_pattern, text, flags)
    return len(matches)


# ══════════════════════════════════════════════════════════════════════════════
# REGISTERED SCORING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════


@register_method(CountOccurrencesParams)
def score_count_occurrences(
    text: str,
    items: list[str | list[str]],
    params: CountOccurrencesParams,
    runner: ModelRunner | None = None,
    embedder: EmbeddingRunner | None = None,
    log_fn: LogFn | None = None,
) -> tuple[list[float], list[str]]:
    """Score text by counting word/phrase occurrences.

    For each target, computes: (# occurrences) / (# total words)

    Args:
        text: Text to analyze
        items: Target words/phrases from config
        params: Method parameters
        runner: Not used
        embedder: Not used
        log_fn: Optional logging callback

    Returns:
        Tuple of (scores, raw_responses)
    """
    total_words = count_words(text)

    def score_single(target: str) -> tuple[float, str]:
        n_found = count_occurrences(text, target, params.case_sensitive)
        score = n_found / total_words if total_words > 0 else 0.0
        return score, ""

    return score_with_bundling(items, score_single, params.label_prefix, log_fn)
