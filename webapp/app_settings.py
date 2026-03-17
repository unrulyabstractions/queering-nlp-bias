"""Default settings, content, and available AI models."""

from __future__ import annotations

# ════════════════════════════════════════════════════════════════════════════════
# DEFAULT CONTENT - Questions, prompts, prefixes used across the app
# ════════════════════════════════════════════════════════════════════════════════

DEFAULT_JUDGE_PROMPT = """
Read the following text and answer the question.

TEXT:
{text}

QUESTION: {question}

For yes/no questions, answer with exactly 1 or 0.
For other, answer with a number from 0.000 to 1.000.
Do not include any other text in your response.
Example: <answer>0.029</answer>
""".strip("\n")

DEFAULT_PROMPT = """
In a sentence, write a very short but realistic love story.
""".strip("\n")

DEFAULT_QUESTIONS = """
Is the protagonist a man?
How masculine is the protagonist?
Is the protagonist a woman?
How feminine is the protagonist?
How queer is the protagonist?
Is the protagonist queer?
""".strip("\n")

DEFAULT_PREFIXES = """
At the altar, our protagonist, who was a nurse, and his partner
At the altar, our protagonist, who was a nurse, and her partner
At the altar, our protagonist, who was a nurse, and their partner
""".strip("\n")

DEFAULT_DYN_PREFILL = """
""".strip("\n")

DEFAULT_DYN_CONT = """
He softly touched him, but violently loved him.
""".strip("\n")

DEFAULT_JUDGE_TEXT = """
The protagonist was a nurse who loved their partner deeply.
The protagonist was a nurse who loved his partner deeply.
The protagonist was a nurse who loved her partner deeply.
""".strip("\n")

DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 8
DEFAULT_SAMPLES_PER_NODE = 5


# ════════════════════════════════════════════════════════════════════════════════
# MODEL SETTINGS
# ════════════════════════════════════════════════════════════════════════════════

DEFAULT_SETTINGS = {
    "gen_provider": "openai",
    "gen_model": "gpt-4o-mini",
    "judge_model": [{"provider": "openai", "model": "gpt-4o-mini"}],
    "gen_temperature": DEFAULT_TEMPERATURE,
    "judge_temperature": 0.0,
    "max_tokens": DEFAULT_MAX_TOKENS,
    "samples_per_node": DEFAULT_SAMPLES_PER_NODE,
    "judge_prompt": DEFAULT_JUDGE_PROMPT,
}

# Available models by provider
# HuggingFace has three sub-providers: base, instruct, reasoning
AVAILABLE_MODELS = {
    "anthropic": [
        "claude-haiku-4-5",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
    ],
    "openai": [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1-mini",
        "gpt-4.1",
    ],
    "huggingface_base": [
        "Qwen/Qwen3.5-0.8B-Base",
        "Qwen/Qwen3.5-2B-Base",
        "Qwen/Qwen3.5-4B-Base",
        "Qwen/Qwen3.5-9B-Base",
    ],
    "huggingface_instruct": [
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B",
    ],
    "huggingface_reasoning": [
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B",
    ],
}

# Provider display names for UI
PROVIDER_DISPLAY_NAMES = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "huggingface_base": "HuggingFace (Base)",
    "huggingface_instruct": "HuggingFace (Instruct)",
    "huggingface_reasoning": "HuggingFace (Reasoning)",
}

# Providers that don't require API keys
LOCAL_PROVIDERS = {"huggingface_base", "huggingface_instruct", "huggingface_reasoning"}


def get_huggingface_mode(provider: str) -> str:
    """Get the HuggingFace mode from provider name."""
    if provider == "huggingface_base":
        return "base"
    if provider == "huggingface_reasoning":
        return "reasoning"
    return "instruct"


def is_huggingface_provider(provider: str) -> bool:
    """Check if provider is any HuggingFace variant."""
    return provider.startswith("huggingface_")


def is_base_model(provider: str) -> bool:
    """Check if provider uses base models."""
    return provider == "huggingface_base"


def should_enable_thinking(provider: str) -> bool:
    """Determine if thinking/reasoning mode should be enabled.

    For huggingface_reasoning: enable thinking (returns True)
    For huggingface_instruct: disable thinking (returns False)
    For huggingface_base: N/A, but returns False
    """
    return get_huggingface_mode(provider) == "reasoning"
