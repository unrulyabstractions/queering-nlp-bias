"""Shared base code for LLM providers: constants, dataclasses, retry logic, logging."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from anthropic import RateLimitError as AnthropicRateLimitError
from openai import RateLimitError as OpenAIRateLimitError

from webapp.common.normativity_types import Scoring
from webapp.common.text_formatting_utils import truncate_for_log


# ════════════════════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════════════════════

# Retry settings for rate limits
MAX_RETRIES = 5
BASE_RETRY_DELAY = 15  # seconds

# Rate limit exception types by provider
RATE_LIMIT_EXCEPTIONS: dict[str, type] = {
    "anthropic": AnthropicRateLimitError,
    "openai": OpenAIRateLimitError,
}

# Judge text truncation (prevents overly long inputs)
MAX_JUDGE_TEXT_LENGTH = 1500

# Judge response tokens
JUDGE_MAX_TOKENS = 100

# Logging truncation widths
LOG_TRUNCATE_DEFAULT = 80
LOG_TRUNCATE_PROMPT = 100
LOG_TRUNCATE_PREFILL = 60
LOG_TRUNCATE_RESPONSE = 30


# ════════════════════════════════════════════════════════════════════════════════
# Result Dataclasses
# ════════════════════════════════════════════════════════════════════════════════


@dataclass
class GenerationResult:
    """Result from generating text, optionally with logprobs."""

    text: str
    logprob: float | None = None


@dataclass
class JudgeResult:
    """Result from judging a text against a question."""

    score: Scoring
    raw_response: str
    logprob: Scoring | None = None


# ════════════════════════════════════════════════════════════════════════════════
# Retry Logic
# ════════════════════════════════════════════════════════════════════════════════


async def retry_on_rate_limit(
    provider: str,
    api_call: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute an API call with exponential backoff retry on rate limits.

    Args:
        provider: The provider name ('anthropic' or 'openai')
        api_call: The synchronous API function to call
        *args, **kwargs: Arguments to pass to the API call

    Returns:
        The API response

    Raises:
        The original rate limit error if all retries are exhausted
    """
    rate_limit_error = RATE_LIMIT_EXCEPTIONS.get(provider)

    for attempt in range(MAX_RETRIES + 1):
        try:
            return await asyncio.to_thread(api_call, *args, **kwargs)
        except Exception as e:
            if rate_limit_error and isinstance(e, rate_limit_error):
                if attempt < MAX_RETRIES:
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    print(f"⚠️ {provider} RATE_LIMIT: {str(e)[:100]}")
                    print(f"⚠️ Retrying in {delay}s (attempt {attempt + 1}/{MAX_RETRIES})")
                    await asyncio.sleep(delay)
                else:
                    print(f"❌ {provider} RATE_LIMIT_EXCEEDED after {MAX_RETRIES} retries")
                    raise
            else:
                raise


# ════════════════════════════════════════════════════════════════════════════════
# Logging Helpers
# ════════════════════════════════════════════════════════════════════════════════


def profile(label: str, start_time: float) -> None:
    """Print profiling info."""
    elapsed = time.time() - start_time
    print(f"⏱️  [{label}] {elapsed:.3f}s")


def log_generation_call(provider: str, model: str, prompt: str, prefill: str) -> None:
    """Log a generation API call."""
    print("\n" + "-" * 60)
    print("LLM GENERATION CALL")
    print("-" * 60)
    print(f"  Provider: {provider}")
    print(f"  Model: {model}")
    print(f"  Prompt: {truncate_for_log(prompt, LOG_TRUNCATE_PROMPT)}")
    if prefill:
        print(f"  Prefill: {truncate_for_log(prefill, LOG_TRUNCATE_PREFILL)}")


def log_generation_result(result: str, logprob: float | None = None) -> None:
    """Log a generation result with FULL details."""
    print("=" * 60)
    print("  ██ GENERATION RESULT ██")
    print("=" * 60)
    print(f"  Length: {len(result)} chars, {len(result.split())} words")
    if logprob is not None:
        print(f"  Sum logprob: {logprob:.4f}")
        print(f"  Perplexity: {2 ** (-logprob / max(1, len(result.split()))):.2f}")
    print()
    print("  ┌" + "─" * 56 + "┐")
    print("  │ FULL GENERATED TEXT:".ljust(58) + "│")
    print("  ├" + "─" * 56 + "┤")
    for line in result.split("\n"):
        while len(line) > 54:
            print(f"  │ {line[:54]} │")
            line = line[54:]
        print(f"  │ {line.ljust(54)} │")
    print("  └" + "─" * 56 + "┘")
    print("=" * 60)


def log_judge_call(
    provider: str,
    model: str,
    text: str,
    question: str,
    formatted_prompt: str,
    call_type: str = "JUDGE",
) -> None:
    """Log a judge API call."""
    print("\n" + "-" * 60)
    print(f"LLM {call_type} CALL")
    print("-" * 60)
    print(f"  Provider: {provider}")
    print(f"  Model: {model}")
    print(f"  Question: {truncate_for_log(question, 80)}")
    print(f"  Text: {truncate_for_log(text, 100)}")
    print("  Formatted prompt sent to API:")
    for line in formatted_prompt.split("\n"):
        print(f"    | {line}")


def log_judge_result(
    score: Scoring, raw_response: str, logprob: Scoring | None = None
) -> None:
    """Log a judge result with FULL details."""
    print("  ┌─────────────────────────────────────┐")
    print("  │ JUDGE RESULT                        │")
    print("  ├─────────────────────────────────────┤")
    print(f"  │ Raw response: '{raw_response[:LOG_TRUNCATE_RESPONSE]}'".ljust(40) + "│")
    print(f"  │ ★ SCORE: {score:.4f}".ljust(40) + "│")
    if logprob is not None:
        print(f"  │ Logprob: {logprob:.4f}".ljust(40) + "│")
        print(f"  │ Confidence: {100 * (2 ** logprob):.1f}%".ljust(40) + "│")
    print("  └─────────────────────────────────────┘")
