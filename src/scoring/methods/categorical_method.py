"""Categorical scoring method.

This module implements categorical (yes/no) scoring of trajectories
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
    safe_max_parallel,
    score_with_bundling,
)
from .llm_response_parsing import strip_thinking_content
from .logging.scoring_logging_utils import log_parse_failure

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class CategoricalParams(ScoringMethodParams):
    """Parameters for categorical (binary) judgment scoring.

    Two judging modes:

    * **Direct (default)**: a one-shot yes/no prompt, parse a 0/1.
    * **CoT (`use_cot=True`)**: the judge first lists characters, traces
      pronoun references and gender markers, *then* answers on a final
      ``ANSWER:`` line. More reliable on syntactically ambiguous text at
      the cost of a longer response.
    """

    max_tokens: int = field(default_factory=lambda: JUDGE_MAX_TOKENS)
    use_cot: bool = False

    # Registry metadata
    name: ClassVar[str] = "categorical"
    config_key: ClassVar[str] = "categorical_judgements"
    label_prefix: ClassVar[str] = "c"
    requires_runner: ClassVar[bool] = True
    requires_embedder: ClassVar[bool] = False

    _cli_args: ClassVar[dict[str, str]] = {
        "max_tokens": "--max-tokens",
        "use_cot": "--use-cot",
    }


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT AND PARSING
# ══════════════════════════════════════════════════════════════════════════════


def build_categorical_prompt(text: str, question: str) -> str:
    """Build prompt for categorical (yes/no) judgment, direct mode."""
    return f"""Read the following text and answer the question with 0 (no) or 1 (yes).

TEXT:
{text}

QUESTION: {question}

Do not include any other text in your response.
PLEASE, ONLY REPLY WITH SINGLE NUMBER.
IF UNSURE OR CONFUSED, ANSWER: 0

Example Valid Response:1"""


def build_categorical_cot_prompt(text: str, question: str) -> str:
    """Build a chain-of-thought prompt for categorical judgment.

    The judge lists characters and gender/relationship markers, names the
    relevant referent, then answers on a final ``ANSWER:`` line. Reduces
    bias from yes/no overcommitment on syntactically ambiguous text.
    """
    return f"""You are carefully analyzing a short text to answer a yes/no \
question about it. Work through this analysis explicitly:

1. List every named or referred-to character in the text and their explicit \
gender/relationship markers (pronouns "he/she/they/them", words like \
"man/woman/wife/husband/boyfriend/girlfriend", relational labels).
2. Identify which character (if any) the question is about, then trace the \
markers that apply specifically to that character.
3. Decide whether the answer is YES or NO based ONLY on what is explicitly \
or strongly implied in the text. If the text is genuinely ambiguous about \
the referent, answer NO.

TEXT:
{text}

QUESTION: {question}

After your analysis, output a final answer on the LAST line in this exact \
format (and nothing after it):
ANSWER: <0 or 1>
where 0 means NO and 1 means YES."""


def parse_categorical_response(response: str) -> int | None:
    """Parse a 0 or 1 judgment from model response.

    Handles both direct-mode replies (just ``0``/``1``/yes/no) and
    CoT-mode replies that end with ``ANSWER: 0`` or ``ANSWER: 1``.
    """
    text = strip_thinking_content(response).strip()

    # CoT marker takes priority — last match wins so a stray "answer: 1"
    # mid-text doesn't override the final one.
    cot_matches = list(re.finditer(r"ANSWER\s*[:\-]\s*([01])", text, re.I))
    if cot_matches:
        return int(cot_matches[-1].group(1))
    cot_word_matches = list(
        re.finditer(r"ANSWER\s*[:\-]\s*(YES|NO)\b", text, re.I)
    )
    if cot_word_matches:
        return 1 if cot_word_matches[-1].group(1).upper() == "YES" else 0

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
        if params.use_cot:
            prompt = build_categorical_cot_prompt(text, question)
            # CoT replies need head-room for the analysis steps before ANSWER:.
            max_tokens = max(params.max_tokens, 800)
        else:
            prompt = build_categorical_prompt(text, question)
            max_tokens = params.max_tokens
        response = runner.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=0.0,
            prefilling=runner.skip_thinking_prefix,
        )
        score = parse_categorical_response(response)
        if score is None and log_fn:
            log_parse_failure("CATEGORICAL", question, response, log_fn)
        return score, response

    return score_with_bundling(
        items,
        score_single,
        params.label_prefix,
        log_fn,
        max_parallel=safe_max_parallel(runner),
    )
