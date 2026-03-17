"""LLM provider clients for Anthropic, OpenAI, and HuggingFace."""

from __future__ import annotations

from .provider_anthropic import generate_anthropic, get_anthropic_client, judge_anthropic
from .provider_base import (
    GenerationResult,
    JUDGE_MAX_TOKENS,
    JudgeResult,
    MAX_JUDGE_TEXT_LENGTH,
    MAX_RETRIES,
)
from .provider_huggingface import (
    SKIP_THINKING_PREFIX,
    generate_huggingface,
    get_huggingface_model,
    is_base_model,
    judge_huggingface,
)
from .provider_openai import generate_openai, get_openai_client, judge_openai

__all__ = [
    # Base
    "GenerationResult",
    "JudgeResult",
    "MAX_JUDGE_TEXT_LENGTH",
    "JUDGE_MAX_TOKENS",
    "MAX_RETRIES",
    # Anthropic
    "get_anthropic_client",
    "generate_anthropic",
    "judge_anthropic",
    # OpenAI
    "get_openai_client",
    "generate_openai",
    "judge_openai",
    # HuggingFace
    "get_huggingface_model",
    "is_base_model",
    "generate_huggingface",
    "judge_huggingface",
    "SKIP_THINKING_PREFIX",
]
