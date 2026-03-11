"""Categorical scoring method.

This module implements categorical (yes/no) scoring of trajectories
using a language model as a judge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

from src.common.callback_types import LogFn
from src.inference import ModelRunner
from src.inference.embedding_runner import EmbeddingRunner

from ..scoring_method_registry import (
    ScoringMethodParams,
    register_method,
    score_with_bundling,
)

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class CategoricalParams(ScoringMethodParams):
    """Parameters for categorical (binary) judgment scoring."""

    max_tokens: int = 10

    # Registry metadata
    name: ClassVar[str] = "categorical"
    config_key: ClassVar[str] = "categorical_judgements"
    label_prefix: ClassVar[str] = "c"
    requires_runner: ClassVar[bool] = True
    requires_embedder: ClassVar[bool] = False

    _cli_args: ClassVar[dict[str, str]] = {
        "max_tokens": "--max-tokens",
    }


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT AND PARSING
# ══════════════════════════════════════════════════════════════════════════════


def build_categorical_prompt(text: str, question: str) -> str:
    """Build prompt for categorical judgment."""
    return f"""Read the following text and answer the question with 0 (no) or 1 (yes).

TEXT:
{text}

QUESTION: {question}

Answer with just 0 or 1:"""


def parse_categorical_response(response: str) -> int | None:
    """Parse a 0 or 1 judgment from model response."""
    text = response
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = text.strip()

    if text in ("0", "1"):
        return int(text)

    match = re.search(r"(?:answer|response|judgment|result)[:\s]*([01])", text, re.I)
    if match:
        return int(match.group(1))

    if re.search(r"\byes\b", text, re.I):
        return 1
    if re.search(r"\bno\b", text, re.I):
        return 0

    match = re.search(r"([01])\s*$", text)
    if match:
        return int(match.group(1))

    match = re.search(r"\b([01])\b", text)
    if match:
        return int(match.group(1))

    return None


# ══════════════════════════════════════════════════════════════════════════════
# REGISTERED SCORING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════


@register_method(CategoricalParams)
def score_categorical(
    text: str,
    items: list[str | list[str]],
    params: CategoricalParams,
    runner: ModelRunner | None = None,
    embedder: EmbeddingRunner | None = None,
    log_fn: LogFn | None = None,
) -> tuple[list[int | None], list[str]]:
    """Score text on categorical judgments.

    Args:
        text: Text to judge
        items: Questions from config (method's config_key data)
        params: Method parameters
        runner: Model runner for inference
        embedder: Not used
        log_fn: Optional logging callback

    Returns:
        Tuple of (scores, raw_responses)
    """
    if runner is None:
        raise ValueError("Categorical scoring requires a model runner")

    def score_single(question: str) -> tuple[int | None, str]:
        prompt = build_categorical_prompt(text, question)
        response = runner.generate(
            prompt=prompt,
            max_new_tokens=params.max_tokens,
            temperature=0.0,
            prefilling=runner.skip_thinking_prefix,
        )
        score = parse_categorical_response(response)
        return score, response

    return score_with_bundling(items, score_single, params.label_prefix, log_fn)
