"""LLM provider clients - unified interface for Anthropic, OpenAI, and HuggingFace."""

from __future__ import annotations

import asyncio
from typing import Any

from .llm_providers import (
    GenerationResult,
    JudgeResult,
    JUDGE_MAX_TOKENS,
    MAX_JUDGE_TEXT_LENGTH,
    MAX_RETRIES,
    SKIP_THINKING_PREFIX,
    generate_anthropic,
    generate_huggingface,
    generate_openai,
    get_anthropic_client,
    get_huggingface_model,
    get_openai_client,
    is_base_model,
    judge_anthropic,
    judge_huggingface,
    judge_openai,
)


__all__ = [
    # Types
    "GenerationResult",
    "JudgeResult",
    # Constants
    "JUDGE_MAX_TOKENS",
    "MAX_JUDGE_TEXT_LENGTH",
    "MAX_RETRIES",
    "SKIP_THINKING_PREFIX",
    # Factory
    "get_client",
    "get_huggingface_model",
    "is_base_model",
    # Unified interfaces
    "generate_from_llm",
    "llm_judge",
    "judge_all_questions",
]


def get_client(provider: str, api_key: str) -> Any:
    """Create client for the specified provider."""
    if provider == "openai":
        return get_openai_client(api_key)
    if provider == "huggingface":
        return None  # Local models don't need a client
    return get_anthropic_client(api_key)


async def generate_from_llm(
    client: Any,
    provider: str,
    model: str,
    prompt: str,
    prefill: str = "",
    max_tokens: int = 300,
    temperature: float = 1.0,
) -> GenerationResult:
    """Generate text continuation. Routes to provider-specific implementation."""
    if provider == "openai":
        return await generate_openai(client, model, prompt, prefill, max_tokens, temperature)
    if provider == "huggingface":
        return await generate_huggingface(model, prompt, prefill, max_tokens, temperature)
    return await generate_anthropic(client, model, prompt, prefill, max_tokens, temperature)


async def llm_judge(
    client: Any,
    provider: str,
    model: str,
    text: str,
    question: str,
    judge_prompt: str,
    temperature: float = 0.0,
) -> JudgeResult:
    """Score text against question. Routes to provider-specific implementation."""
    if provider == "openai":
        return await judge_openai(client, model, text, question, judge_prompt, temperature)
    if provider == "huggingface":
        return await judge_huggingface(model, text, question, judge_prompt, temperature)
    return await judge_anthropic(client, model, text, question, judge_prompt, temperature)


async def judge_all_questions(
    client: Any,
    provider: str,
    model: str,
    text: str,
    questions: list[str],
    judge_prompt: str,
) -> list[JudgeResult]:
    """Judge text against all questions sequentially."""
    results = []
    for i, q in enumerate(questions):
        result = await llm_judge(client, provider, model, text, q, judge_prompt)
        results.append(result)
        if i < len(questions) - 1:
            await asyncio.sleep(0.1)
    return results
