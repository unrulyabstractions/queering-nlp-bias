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
    MAX_JUDGE_TEXT_LENGTH,
    log_generation_call,
    log_generation_result,
    profile,
    retry_on_rate_limit,
)


# OpenAI doesn't support true prefill, so we use an instruction instead
OPENAI_PREFILL_INSTRUCTION = (
    "Continue from the following text exactly as written, without repeating it. "
    "Your response must seamlessly continue from this starting point:\n\n{prefill}"
)


def get_openai_client(api_key: str) -> openai.OpenAI:
    """Create OpenAI client."""
    return openai.OpenAI(api_key=api_key)


async def generate_openai(
    client: Any,
    model: str,
    prompt: str,
    prefill: str = "",
    max_tokens: int = 300,
    temperature: float = 1.0,
) -> GenerationResult:
    """Generate text continuation using OpenAI API with logprobs."""
    log_generation_call("openai", model, prompt, prefill)

    full_prompt = prompt
    if prefill:
        instruction = OPENAI_PREFILL_INSTRUCTION.format(prefill=prefill)
        full_prompt = f"{prompt}\n\n{instruction}"

    print(f"█ OpenAI gen: model={model} prompt_len={len(full_prompt)}")

    api_start = time.time()
    response = await retry_on_rate_limit(
        "openai",
        client.chat.completions.create,
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=True,
    )
    profile("OpenAI gen API", api_start)

    content = response.choices[0].message.content or ""
    result_text = prefill + content

    # Extract logprobs
    logprob = None
    choice = response.choices[0]
    tokens_count = 0
    if choice.logprobs and choice.logprobs.content:
        logprob = sum(t.logprob for t in choice.logprobs.content)
        tokens_count = len(choice.logprobs.content)

    usage = response.usage
    lp_str = f"{logprob:.4f}" if logprob is not None else "N/A"
    usage_str = str(usage.total_tokens) if usage else "N/A"
    print(f"▓ Response: {tokens_count} tokens, logprob={lp_str}, usage={usage_str}")

    log_generation_result(result_text, logprob)
    return GenerationResult(text=result_text, logprob=logprob)


async def judge_openai(
    client: Any,
    model: str,
    text: str,
    question: str,
    judge_prompt: str,
    temperature: float = 0.0,
) -> JudgeResult:
    """Score text using OpenAI API. Includes logprob extraction."""
    formatted_prompt = judge_prompt.format(text=text[:MAX_JUDGE_TEXT_LENGTH], question=question)

    response = await retry_on_rate_limit(
        "openai",
        client.chat.completions.create,
        model=model,
        messages=[{"role": "user", "content": formatted_prompt}],
        max_tokens=JUDGE_MAX_TOKENS,
        temperature=temperature,
        logprobs=True,
        top_logprobs=5,
    )

    answer = response.choices[0].message.content or ""
    logprob = None
    choice = response.choices[0]
    if choice.logprobs and choice.logprobs.content:
        logprob = sum(t.logprob for t in choice.logprobs.content)

    score = parse_judge_score(answer)
    return JudgeResult(score=score, raw_response=answer.strip(), logprob=logprob)
