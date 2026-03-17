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


# Constants
MAX_RETRIES = 5
BASE_RETRY_DELAY = 15
MAX_JUDGE_TEXT_LENGTH = 1500
JUDGE_MAX_TOKENS = 100
LOG_TRUNCATE_PROMPT = 100
LOG_TRUNCATE_PREFILL = 60

RATE_LIMIT_EXCEPTIONS: dict[str, type] = {
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


def profile(label: str, start_time: float) -> None:
    print(f"⏱️  [{label}] {time.time() - start_time:.3f}s")


def log_generation_call(provider: str, model: str, prompt: str, prefill: str) -> None:
    pf = f" prefill={truncate_for_log(prefill, LOG_TRUNCATE_PREFILL)}" if prefill else ""
    print(f"▶ GEN [{provider}/{model}] {truncate_for_log(prompt, LOG_TRUNCATE_PROMPT)}{pf}")


def log_generation_result(result: str, logprob: float | None = None) -> None:
    lp = f" logprob={logprob:.2f}" if logprob else ""
    print(f"◀ GEN result: {len(result)} chars, {len(result.split())} words{lp}")


def log_judge_call(
    provider: str, model: str, text: str, question: str, formatted_prompt: str
) -> None:
    print(f"▶ JUDGE [{provider}/{model}] q={truncate_for_log(question, 60)}")


def log_judge_result(score: Scoring, raw_response: str, logprob: Scoring | None = None) -> None:
    lp = f" logprob={logprob:.2f}" if logprob else ""
    print(f"◀ JUDGE score={score:.4f} raw='{raw_response[:30]}'{lp}")
