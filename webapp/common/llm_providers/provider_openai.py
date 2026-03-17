"""OpenAI provider: generation and judging with GPT models."""

from __future__ import annotations

import time

import openai
from openai.types.chat.chat_completion import Choice

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

OPENAI_PREFILL_INSTRUCTION = (
    "Continue from the following text exactly as written, without repeating it. "
    "Your response must seamlessly continue from this starting point:\n\n{prefill}"
)


def get_openai_client(api_key: str) -> openai.OpenAI:
    return openai.OpenAI(api_key=api_key)


def _extract_logprob(choice: Choice) -> float | None:
    if choice.logprobs and choice.logprobs.content:
        return sum(t.logprob for t in choice.logprobs.content)
    return None


async def generate_openai(
    client: openai.OpenAI,
    model: str,
    prompt: str,
    prefill: str = "",
    max_tokens: int | None = 300,
    temperature: float = 1.0,
) -> GenerationResult:
    """Generate with OpenAI. Simulates prefill via instruction.

    Args:
        client: OpenAI client
        model: Model name
        prompt: User prompt
        prefill: Text to prefill the response with
        max_tokens: Max tokens to generate (None = use model default)
        temperature: Sampling temperature
    """
    log_generation_call("openai", model, prompt, prefill)

    full_prompt = prompt
    if prefill:
        full_prompt = (
            f"{prompt}\n\n{OPENAI_PREFILL_INSTRUCTION.format(prefill=prefill)}"
        )

    # Resolve max_tokens using model defaults
    effective_max_tokens = get_max_tokens_for_model(model, max_tokens)

    api_start = time.time()
    response = await retry_on_rate_limit(
        "openai",
        client.chat.completions.create,
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        max_tokens=effective_max_tokens,
        temperature=temperature,
        logprobs=True,
    )
    profile("OpenAI gen", api_start)

    content = response.choices[0].message.content or ""
    logprob = _extract_logprob(response.choices[0])

    # Strip prefill if model repeated it (common with instruction-based prefill)
    if prefill and content.startswith(prefill):
        content = content[len(prefill):]

    log_generation_result(content, logprob)
    return GenerationResult(text=content, logprob=logprob)


async def judge_openai(
    client: openai.OpenAI,
    model: str,
    text: str,
    question: str,
    judge_prompt: str,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> JudgeResult:
    formatted = format_judge_prompt(judge_prompt, text, question)
    log_judge_call("openai", model, question)

    tokens = max_tokens if max_tokens else JUDGE_MAX_TOKENS
    response = await retry_on_rate_limit(
        "openai",
        client.chat.completions.create,
        model=model,
        messages=[{"role": "user", "content": formatted}],
        max_tokens=tokens,
        temperature=temperature,
        logprobs=True,
    )

    answer = response.choices[0].message.content or ""
    logprob = _extract_logprob(response.choices[0])
    score = parse_judge_score(answer)
    log_judge_result(score, answer.strip(), logprob)
    return JudgeResult(score=score, raw_response=answer.strip(), logprob=logprob)
