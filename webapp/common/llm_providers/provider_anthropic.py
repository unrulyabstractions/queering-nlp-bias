"""Anthropic provider: generation and judging with Claude models."""

from __future__ import annotations

from typing import Any

import anthropic

from webapp.common.normativity_types import parse_judge_score

from .provider_base import (
    GenerationResult,
    JudgeResult,
    JUDGE_MAX_TOKENS,
    MAX_JUDGE_TEXT_LENGTH,
    log_generation_call,
    log_generation_result,
    log_judge_call,
    log_judge_result,
    retry_on_rate_limit,
)


def get_anthropic_client(api_key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key)


async def generate_anthropic(
    client: Any, model: str, prompt: str, prefill: str = "",
    max_tokens: int = 300, temperature: float = 1.0,
) -> GenerationResult:
    """Generate with Anthropic. Uses prefill as assistant message primer."""
    log_generation_call("anthropic", model, prompt, prefill)

    messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
    if prefill and (stripped := prefill.rstrip()):
        messages.append({"role": "assistant", "content": stripped})

    response = await retry_on_rate_limit(
        "anthropic", client.messages.create,
        model=model, max_tokens=max_tokens, messages=messages, temperature=temperature,
    )

    continuation = response.content[0].text if response.content else ""
    result = prefill + continuation
    log_generation_result(result, None)
    return GenerationResult(text=result, logprob=None)


async def judge_anthropic(
    client: Any, model: str, text: str, question: str,
    judge_prompt: str, temperature: float = 0.0,
) -> JudgeResult:
    formatted = judge_prompt.format(text=text[:MAX_JUDGE_TEXT_LENGTH], question=question)
    log_judge_call("anthropic", model, text, question, formatted)

    response = await retry_on_rate_limit(
        "anthropic", client.messages.create,
        model=model, max_tokens=JUDGE_MAX_TOKENS,
        messages=[{"role": "user", "content": formatted}], temperature=temperature,
    )

    answer = response.content[0].text if response.content else ""
    score = parse_judge_score(answer)
    log_judge_result(score, answer.strip())
    return JudgeResult(score=score, raw_response=answer.strip(), logprob=None)
