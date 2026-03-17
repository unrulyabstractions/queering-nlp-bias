"""LLM provider clients - unified interface for Anthropic, OpenAI, and HuggingFace."""

from __future__ import annotations

import asyncio

import anthropic
import openai

from webapp.app_settings import is_huggingface_provider

from .llm_providers import (
    GenerationResult,
    JudgeResult,
    generate_anthropic,
    generate_huggingface,
    generate_openai,
    get_anthropic_client,
    get_openai_client,
    judge_anthropic,
    judge_huggingface,
    judge_openai,
)

# Type alias: Anthropic client | OpenAI client | None (for local HuggingFace)
LLMClient = anthropic.Anthropic | openai.OpenAI | None


def get_client(provider: str, api_key: str) -> LLMClient:
    """Create client for the specified provider."""
    if provider == "openai":
        return get_openai_client(api_key)
    if is_huggingface_provider(provider):
        return None
    return get_anthropic_client(api_key)


async def generate_from_llm(
    client: LLMClient,
    provider: str,
    model: str,
    prompt: str,
    prefill: str = "",
    max_tokens: int | None = 300,
    temperature: float = 1.0,
) -> GenerationResult:
    """Generate text continuation. Routes to provider-specific implementation.

    Args:
        client: LLM client (Anthropic, OpenAI, or None for HuggingFace)
        provider: Provider name (huggingface_base, huggingface_instruct, huggingface_reasoning for HF)
        model: Model name
        prompt: User prompt
        prefill: Text to prefill the response with
        max_tokens: Max tokens to generate (None = use model default)
        temperature: Sampling temperature
    """
    if provider == "openai":
        return await generate_openai(
            client, model, prompt, prefill, max_tokens, temperature
        )
    if is_huggingface_provider(provider):
        return await generate_huggingface(
            provider, model, prompt, prefill, max_tokens, temperature
        )
    return await generate_anthropic(
        client, model, prompt, prefill, max_tokens, temperature
    )


async def llm_judge(
    client: LLMClient,
    provider: str,
    model: str,
    text: str,
    question: str,
    judge_prompt: str,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> JudgeResult:
    """Score text against question. Routes to provider-specific implementation.

    Args:
        client: LLM client (Anthropic, OpenAI, or None for HuggingFace)
        provider: Provider name (huggingface_base, huggingface_instruct, huggingface_reasoning for HF)
        model: Model name
        text: Text to judge
        question: Question to answer about the text
        judge_prompt: Template for the judge prompt
        temperature: Sampling temperature
        max_tokens: Max tokens for response (None = use default)
    """
    if provider == "openai":
        return await judge_openai(
            client, model, text, question, judge_prompt, temperature, max_tokens
        )
    if is_huggingface_provider(provider):
        return await judge_huggingface(
            provider, model, text, question, judge_prompt, temperature, max_tokens
        )
    return await judge_anthropic(
        client, model, text, question, judge_prompt, temperature, max_tokens
    )


async def judge_all_questions(
    client: LLMClient,
    provider: str,
    model: str,
    text: str,
    questions: list[str],
    judge_prompt: str,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> list[JudgeResult]:
    """Judge text against all questions in parallel."""
    tasks = [
        llm_judge(
            client, provider, model, text, q, judge_prompt, temperature, max_tokens
        )
        for q in questions
    ]
    return await asyncio.gather(*tasks)


async def multi_judge(
    client: LLMClient,
    provider: str,
    models: list[str],
    text: str,
    question: str,
    judge_prompt: str,
    temperature: float = 0.0,
) -> JudgeResult:
    """Score text against question using multiple models, averaging scores.

    Args:
        client: LLM client
        provider: Provider name
        models: List of model names to use as judges
        text: Text to judge
        question: Question to answer about the text
        judge_prompt: Template for the judge prompt
        temperature: Sampling temperature

    Returns a JudgeResult with averaged score and combined raw responses.
    """
    if not models:
        return JudgeResult(score=None, raw_response="No models specified", logprob=None)

    if len(models) == 1:
        return await llm_judge(
            client, provider, models[0], text, question, judge_prompt, temperature
        )

    # Run all judges concurrently
    tasks = [
        llm_judge(client, provider, model, text, question, judge_prompt, temperature)
        for model in models
    ]
    results = await asyncio.gather(*tasks)

    # Average the scores (skip None/error scores)
    valid_scores = [r.score for r in results if r.score is not None]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

    # Combine raw responses for transparency
    combined_response = " | ".join(
        f"[{m}:{f'{r.score:.2f}' if r.score is not None else 'ERR'}]"
        for m, r in zip(models, results)
    )

    # Average logprobs if available
    logprobs = [r.logprob for r in results if r.logprob is not None]
    avg_logprob = sum(logprobs) / len(logprobs) if logprobs else None

    return JudgeResult(
        score=avg_score, raw_response=combined_response, logprob=avg_logprob
    )


async def multi_judge_all_questions(
    client: LLMClient,
    provider: str,
    models: list[str],
    text: str,
    questions: list[str],
    judge_prompt: str,
    temperature: float = 0.0,
) -> list[JudgeResult]:
    """Judge text against all questions using multiple models (scores averaged)."""
    results = []
    for i, q in enumerate(questions):
        if i > 0:
            await asyncio.sleep(0.1)
        results.append(
            await multi_judge(
                client, provider, models, text, q, judge_prompt, temperature
            )
        )
    return results


# Import JudgeModelSpec for type hints (avoid circular import by importing here)
from webapp.common.algorithm_config import JudgeModelSpec


async def multi_provider_judge(
    api_keys: dict[str, str],
    model_specs: list[JudgeModelSpec],
    text: str,
    question: str,
    judge_prompt: str,
    temperature: float = 0.0,
) -> JudgeResult:
    """Score text using multiple models from different providers, averaging scores.

    Args:
        api_keys: Dict mapping provider name to API key
        model_specs: List of JudgeModelSpec (provider, model pairs)
        text: Text to judge
        question: Question to answer about the text
        judge_prompt: Template for the judge prompt
        temperature: Sampling temperature

    Returns a JudgeResult with averaged score and combined raw responses.
    """
    if not model_specs:
        return JudgeResult(score=None, raw_response="No models specified", logprob=None)

    if len(model_specs) == 1:
        spec = model_specs[0]
        client = get_client(spec.provider, api_keys.get(spec.provider, ""))
        return await llm_judge(
            client, spec.provider, spec.model, text, question, judge_prompt, temperature
        )

    # Create clients for each unique provider
    clients: dict[str, LLMClient] = {}
    for spec in model_specs:
        if spec.provider not in clients:
            clients[spec.provider] = get_client(
                spec.provider, api_keys.get(spec.provider, "")
            )

    # Run all judges concurrently
    async def judge_with_spec(
        spec: JudgeModelSpec,
    ) -> tuple[JudgeModelSpec, JudgeResult]:
        client = clients[spec.provider]
        result = await llm_judge(
            client, spec.provider, spec.model, text, question, judge_prompt, temperature
        )
        return spec, result

    tasks = [judge_with_spec(spec) for spec in model_specs]
    results = await asyncio.gather(*tasks)

    # Average the scores (skip None/error scores)
    valid_scores = [r.score for _, r in results if r.score is not None]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

    # Combine raw responses for transparency (include provider in label)
    combined_response = " | ".join(
        f"[{spec.provider}/{spec.model}:{f'{r.score:.2f}' if r.score is not None else 'ERR'}]"
        for spec, r in results
    )

    # Average logprobs if available
    logprobs = [r.logprob for _, r in results if r.logprob is not None]
    avg_logprob = sum(logprobs) / len(logprobs) if logprobs else None

    return JudgeResult(
        score=avg_score, raw_response=combined_response, logprob=avg_logprob
    )


async def multi_provider_judge_all_questions(
    api_keys: dict[str, str],
    model_specs: list[JudgeModelSpec],
    text: str,
    questions: list[str],
    judge_prompt: str,
    temperature: float = 0.0,
) -> list[JudgeResult]:
    """Judge text against all questions using multiple models from different providers."""
    results = []
    for i, q in enumerate(questions):
        if i > 0:
            await asyncio.sleep(0.1)
        results.append(
            await multi_provider_judge(
                api_keys, model_specs, text, q, judge_prompt, temperature
            )
        )
    return results
