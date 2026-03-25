"""Graded scoring method.

This module implements graded (0-1 scale) scoring of trajectories
using a language model as a judge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import ClassVar

from src.common.callback_types import LogFn
from src.common.default_config import JUDGE_MAX_TOKENS
from src.inference import ModelRunner
from src.inference.embedding_runner import EmbeddingRunner

from ..scoring_method_registry import (
    ScoringMethodParams,
    register_method,
    score_with_bundling,
)
from .llm_response_parsing import strip_thinking_content
from .logging.scoring_logging_utils import log_parse_failure

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class GradedParams(ScoringMethodParams):
    """Parameters for graded (0-1 scale) judgment scoring."""

    max_tokens: int = field(default_factory=lambda: JUDGE_MAX_TOKENS)

    # Registry metadata
    name: ClassVar[str] = "graded"
    config_key: ClassVar[str] = "graded_judgements"
    label_prefix: ClassVar[str] = "g"
    requires_runner: ClassVar[bool] = True
    requires_embedder: ClassVar[bool] = False

    _cli_args: ClassVar[dict[str, str]] = {
        "max_tokens": "--max-tokens",
    }


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT AND PARSING
# ══════════════════════════════════════════════════════════════════════════════


def build_graded_prompt(text: str, question: str) -> str:
    """Build prompt for graded judgment (0-1 scale)."""
    return f"""Read the following text and answer the question with a score between 0.0 and 1.0.
0.0 means completely no/false, 1.0 means completely yes/true, values in between indicate partial agreement.

TEXT:
{text}

QUESTION: {question}

IMPORTANT: Respond with ONLY a decimal number between 0.0 and 1.0. No explanation, no words, just the number.
Example valid responses: 0.0, 0.25, 0.5, 0.75, 1.0"""


# Prefill to force numeric response
GRADED_PREFILL = "0."


def parse_graded_response(response: str, prefill: str = "") -> float | None:
    """Parse a 0-1 graded judgment from model response.

    Args:
        response: Model response text
        prefill: Prefill text that was used (will be prepended to response)

    Returns:
        Float score between 0.0 and 1.0, or None if parsing failed
    """
    # Combine prefill with response for full text
    full_text = prefill + response if prefill else response
    text = strip_thinking_content(full_text)

    # Try to find a decimal number between 0 and 1
    # Match: 0, 1, 0.X, 1.0, .X patterns
    match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?|\.\d+)\b", text)
    if match:
        try:
            value = float(match.group(1))
            if 0.0 <= value <= 1.0:
                return value
        except ValueError:
            pass

    # Try plain 0 or 1
    text_stripped = text.strip()
    if text_stripped in ("0", "1"):
        return float(text_stripped)

    # Try to extract any number and clamp to [0, 1]
    any_number = re.search(r"(\d+\.?\d*)", text)
    if any_number:
        try:
            value = float(any_number.group(1))
            if value <= 1.0:
                return max(0.0, value)
        except ValueError:
            pass

    return None


# ══════════════════════════════════════════════════════════════════════════════
# REGISTERED SCORING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════


@register_method(GradedParams)
def score_graded(
    text: str,
    items: list[str | list[str]],
    params: GradedParams,
    runner: ModelRunner | None = None,
    embedder: EmbeddingRunner | None = None,
    log_fn: LogFn | None = None,
) -> tuple[list[float | None], list[str]]:
    """Score text on graded judgments.

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
        raise ValueError("Graded scoring requires a model runner")

    def score_single(question: str) -> tuple[float | None, str]:
        prompt = build_graded_prompt(text, question)
        # Use prefill to encourage numeric response (if model supports it)
        prefill = runner.skip_thinking_prefix + GRADED_PREFILL
        response = runner.generate(
            prompt=prompt,
            max_new_tokens=params.max_tokens,
            temperature=0.0,
            prefilling=prefill,
        )
        # runner.generate() now returns complete response (prefill + continuation)
        # or just the full response if prefill wasn't supported
        score = parse_graded_response(response)
        if score is None and log_fn:
            log_parse_failure("GRADED", question, response, log_fn)
        return score, response

    return score_with_bundling(items, score_single, params.label_prefix, log_fn)
