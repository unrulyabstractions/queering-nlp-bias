"""Anthropic provider: generation and judging with Claude models."""

from __future__ import annotations

import time

import anthropic

from webapp.common.normativity_types import parse_judge_score

from .provider_base import (
    JUDGE_MAX_TOKENS,
    GenerationResult,
    JudgeResult,
    format_judge_prompt,
    get_max_tokens_for_model,
    log_generation_call,
    log_generation_result,
    log_judge_call,
    log_judge_result,
    profile,
    retry_on_rate_limit,
)


def get_anthropic_client(api_key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key)


async def generate_anthropic(
    client: anthropic.Anthropic,
    model: str,
    prompt: str,
    prefill: str = "",
    max_tokens: int | None = 300,
    temperature: float = 1.0,
) -> GenerationResult:
    """Generate with Anthropic. Uses prefill as assistant message primer.

    Args:
        client: Anthropic client
        model: Model name
        prompt: User prompt
        prefill: Text to prefill the response with
        max_tokens: Max tokens to generate (None = use model default)
        temperature: Sampling temperature
    """
    log_generation_call("anthropic", model, prompt, prefill)

    messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
    if prefill and (stripped_prefill := prefill.rstrip()):
        messages.append({"role": "assistant", "content": stripped_prefill})

    # Resolve max_tokens using model defaults
    effective_max_tokens = get_max_tokens_for_model(model, max_tokens)

    api_start = time.time()
    response = await retry_on_rate_limit(
        "anthropic",
        client.messages.create,
        model=model,
        max_tokens=effective_max_tokens,
        messages=messages,
        temperature=temperature,
    )
    profile("Anthropic gen", api_start)

    continuation = response.content[0].text if response.content else ""
    log_generation_result(continuation, None)
    return GenerationResult(text=continuation, logprob=None)


async def judge_anthropic(
    client: anthropic.Anthropic,
    model: str,
    text: str,
    question: str,
    judge_prompt: str,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> JudgeResult:
    formatted = format_judge_prompt(judge_prompt, text, question)
    log_judge_call("anthropic", model, question)

    tokens = max_tokens if max_tokens else JUDGE_MAX_TOKENS
    response = await retry_on_rate_limit(
        "anthropic",
        client.messages.create,
        model=model,
        max_tokens=tokens,
        messages=[{"role": "user", "content": formatted}],
        temperature=temperature,
    )

    answer = response.content[0].text if response.content else ""
    score = parse_judge_score(answer)
    log_judge_result(
        score, answer.strip(), None
    )  # Anthropic API doesn't provide logprobs
    return JudgeResult(score=score, raw_response=answer.strip(), logprob=None)
