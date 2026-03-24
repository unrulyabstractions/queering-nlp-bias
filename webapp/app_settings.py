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
PLEASE, ONLY REPLY WITH SINGLE NUMBER.
IF UNSURE OR CONFUSED, ANSWER: 0.001

Example: <answer>0.029</answer>
""".strip("\n")

DEFAULT_PROMPT = """
In a sentence, write a very short but realistic love story. Give names to the characters.
""".strip("\n")

DEFAULT_QUESTIONS = """
How masculine is the protagonist?
How feminine is the protagonist?
Is the protagonist explicitly heterosexual?
Is the protagonist explicitly gay?
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
        # Aliases (resolve to full model IDs)
        "claude",  # → claude-sonnet-4-20250514
        "sonnet",  # → claude-sonnet-4-20250514
        "haiku",  # → claude-haiku-4-20250514
        "opus",  # → claude-opus-4-20250514
        # Claude 4 models
        "claude-sonnet-4-20250514",
        "claude-haiku-4-20250514",
        "claude-opus-4-20250514",
        # Claude 3.5 models
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        # Claude 3 models
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
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


# Claude model aliases → full model IDs
CLAUDE_MODEL_ALIASES: dict[str, str] = {
    # Provider-only defaults to sonnet
    "claude": "claude-sonnet-4-20250514",
    "anthropic": "claude-sonnet-4-20250514",
    # Model family shortcuts
    "sonnet": "claude-sonnet-4-20250514",
    "haiku": "claude-haiku-4-20250514",
    "opus": "claude-opus-4-20250514",
}


def resolve_claude_model(model: str) -> str:
    """Resolve Claude model alias to full model ID.

    Handles formats:
        - "claude" or "anthropic" → claude-sonnet-4-20250514
        - "sonnet", "haiku", "opus" → latest version of that model
        - "anthropic/sonnet", "claude/haiku" → latest version of that model
        - "anthropic/claude-3-5-sonnet-20241022" → claude-3-5-sonnet-20241022
        - Full model ID passed through unchanged

    Args:
        model: Model name or alias

    Returns:
        Full model ID for Anthropic API
    """
    model = model.strip()

    # Handle provider/model format: "anthropic/sonnet" or "claude/haiku"
    if "/" in model:
        prefix, suffix = model.split("/", 1)
        if prefix.lower() in ("anthropic", "claude"):
            # Check if suffix is an alias
            if suffix.lower() in CLAUDE_MODEL_ALIASES:
                return CLAUDE_MODEL_ALIASES[suffix.lower()]
            # Otherwise return the suffix as-is (full model ID)
            return suffix

    # Direct alias lookup
    return CLAUDE_MODEL_ALIASES.get(model.lower(), model)
