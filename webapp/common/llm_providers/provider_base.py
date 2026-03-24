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


JUDGE_MAX_TOKENS = 32
DEFAULT_MAX_TOKENS = 300

MODEL_MAX_TOKENS: dict[str, int] = {
    # OpenAI models
    "gpt-4o-mini": 16384,
    "gpt-4o": 16384,
    "gpt-4-turbo": 4096,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 4096,
    # Anthropic Claude 4 models
    "claude-sonnet-4-20250514": 8192,
    "claude-haiku-4-20250514": 8192,
    "claude-opus-4-20250514": 8192,
    # Anthropic Claude 3.5 models
    "claude-3-5-sonnet-20241022": 8192,
    "claude-3-5-haiku-20241022": 8192,
    # Anthropic Claude 3 models
    "claude-3-opus-20240229": 4096,
    "claude-3-sonnet-20240229": 4096,
    "claude-3-haiku-20240307": 4096,
    # HuggingFace Qwen models (instruct)
    "Qwen/Qwen3-0.6B": 32768,
    "Qwen/Qwen3-1.7B": 32768,
    "Qwen/Qwen3-4B": 32768,
    "Qwen/Qwen3-8B": 32768,
    "Qwen/Qwen3-14B": 32768,
    "Qwen/Qwen3-32B": 32768,
}


def get_max_tokens_for_model(model: str, requested: int | None) -> int:
    """Get max tokens for generation, using model default if requested is None or 0."""
    if requested is not None and requested > 0:
        return requested
    return MODEL_MAX_TOKENS.get(model, DEFAULT_MAX_TOKENS)

_MAX_RETRIES = 5
_MAX_JUDGE_TEXT_LENGTH = 1500
_BASE_RETRY_DELAY = 15
_LOG_TRUNCATE = 80

_RATE_LIMIT_EXCEPTIONS: dict[str, type] = {
    "anthropic": AnthropicRateLimitError,
    "openai": OpenAIRateLimitError,
}


@dataclass
class GenerationResult:
    text: str
    logprob: float | None = None


@dataclass
class JudgeResult:
    score: Scoring
    raw_response: str
    logprob: Scoring | None = None


async def retry_on_rate_limit(
    provider: str, api_call: Any, *args: Any, **kwargs: Any
) -> Any:
    """Execute API call with exponential backoff on rate limits."""
    rate_limit_error = _RATE_LIMIT_EXCEPTIONS.get(provider)

    for attempt in range(_MAX_RETRIES + 1):
        try:
            return await asyncio.to_thread(api_call, *args, **kwargs)
        except Exception as e:
            if rate_limit_error and isinstance(e, rate_limit_error):
                if attempt < _MAX_RETRIES:
                    delay = _BASE_RETRY_DELAY * (2 ** attempt)
                    print(f"⚠️ {provider} RATE_LIMIT: {str(e)[:100]}")
                    print(f"⚠️ Retrying in {delay}s (attempt {attempt + 1}/{_MAX_RETRIES})")
                    await asyncio.sleep(delay)
                else:
                    print(f"❌ {provider} RATE_LIMIT_EXCEEDED after {_MAX_RETRIES} retries")
                    raise
            else:
                raise


def profile(label: str, start_time: float) -> None:
    print(f"⏱️  [{label}] {time.time() - start_time:.3f}s")


def format_judge_prompt(judge_prompt: str, text: str, question: str) -> str:
    """Format judge prompt with text truncation."""
    return judge_prompt.format(text=text[:_MAX_JUDGE_TEXT_LENGTH], question=question)


def log_generation_call(provider: str, model: str, prompt: str, prefill: str) -> None:
    pf = f" prefill={truncate_for_log(prefill, _LOG_TRUNCATE)}" if prefill else ""
    print(f"▶ GEN [{provider}/{model}] {truncate_for_log(prompt, _LOG_TRUNCATE)}{pf}")


def log_generation_result(result: str, logprob: float | None = None) -> None:
    lp = f" logprob={logprob:.2f}" if logprob else ""
    print(f"◀ GEN result: {len(result)} chars, {len(result.split())} words{lp}")


def log_judge_call(provider: str, model: str, question: str) -> None:
    print(f"▶ JUDGE [{provider}/{model}] q={truncate_for_log(question, _LOG_TRUNCATE)}")


def log_judge_result(score: Scoring, raw_response: str, logprob: Scoring | None = None) -> None:
    lp = f" logprob={logprob:.2f}" if logprob is not None else ""
    score_str = f"{score:.4f}" if score is not None else "ERROR"
    print(f"◀ JUDGE score={score_str} raw='{raw_response[:30]}'{lp}")
