"""OpenAI provider: generation and judging with GPT models."""

from __future__ import annotations

import time
from typing import Any

import openai

from webapp.common.normativity_types import parse_judge_score

from .provider_base import (
    GenerationResult,
    JudgeResult,
    JUDGE_MAX_TOKENS,
    format_judge_prompt,
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


def _extract_logprob(choice: Any) -> float | None:
    if choice.logprobs and choice.logprobs.content:
        return sum(t.logprob for t in choice.logprobs.content)
    return None


async def generate_openai(
    client: Any, model: str, prompt: str, prefill: str = "",
    max_tokens: int = 300, temperature: float = 1.0,
) -> GenerationResult:
    """Generate with OpenAI. Simulates prefill via instruction."""
    log_generation_call("openai", model, prompt, prefill)

    full_prompt = prompt
    if prefill:
        full_prompt = f"{prompt}\n\n{OPENAI_PREFILL_INSTRUCTION.format(prefill=prefill)}"

    api_start = time.time()
    response = await retry_on_rate_limit(
        "openai", client.chat.completions.create,
        model=model, messages=[{"role": "user", "content": full_prompt}],
        max_tokens=max_tokens, temperature=temperature, logprobs=True,
    )
    profile("OpenAI gen", api_start)

    content = response.choices[0].message.content or ""
    logprob = _extract_logprob(response.choices[0])
    result = prefill + content

    log_generation_result(result, logprob)
    return GenerationResult(text=result, logprob=logprob)


async def judge_openai(
    client: Any, model: str, text: str, question: str,
    judge_prompt: str, temperature: float = 0.0,
) -> JudgeResult:
    formatted = format_judge_prompt(judge_prompt, text, question)
    log_judge_call("openai", model, question)

    response = await retry_on_rate_limit(
        "openai", client.chat.completions.create,
        model=model, messages=[{"role": "user", "content": formatted}],
        max_tokens=JUDGE_MAX_TOKENS, temperature=temperature, logprobs=True,
    )

    answer = response.choices[0].message.content or ""
    logprob = _extract_logprob(response.choices[0])
    score = parse_judge_score(answer)
    log_judge_result(score, answer.strip(), logprob)
    return JudgeResult(score=score, raw_response=answer.strip(), logprob=logprob)
