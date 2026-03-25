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
        "claude",  # → claude-sonnet-4-6 (latest)
        "sonnet",  # → claude-sonnet-4-6
        "haiku",  # → claude-haiku-4-5
        "opus",  # → claude-opus-4-6
        # Claude 4.6 models (latest)
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        # Claude 4.5 models
        "claude-haiku-4-5",
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-5",
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-5",
        "claude-opus-4-5-20251101",
        # Claude 4.1 models
        "claude-opus-4-1",
        "claude-opus-4-1-20250805",
        # Claude 4.0 models
        "claude-sonnet-4-0",
        "claude-sonnet-4-20250514",
        "claude-opus-4-0",
        "claude-opus-4-20250514",
        # Claude 3.5 models
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        # Claude 3 models (deprecated)
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
# Latest models default to their API aliases (no date suffix needed)
CLAUDE_MODEL_ALIASES: dict[str, str] = {
    # Default aliases (latest recommended models)
    "claude": "claude-sonnet-4-6",
    "anthropic": "claude-sonnet-4-6",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5",
    "opus": "claude-opus-4-6",
    # Latest generation (4.6 / 4.5)
    "opus-4.6": "claude-opus-4-6",
    "opus-4-6": "claude-opus-4-6",
    "sonnet-4.6": "claude-sonnet-4-6",
    "sonnet-4-6": "claude-sonnet-4-6",
    "haiku-4.5": "claude-haiku-4-5",
    "haiku-4-5": "claude-haiku-4-5",
    # Previous generation (4.5 for sonnet/opus, 4.1 for opus)
    "opus-4.5": "claude-opus-4-5",
    "opus-4-5": "claude-opus-4-5",
    "sonnet-4.5": "claude-sonnet-4-5",
    "sonnet-4-5": "claude-sonnet-4-5",
    "opus-4.1": "claude-opus-4-1",
    "opus-4-1": "claude-opus-4-1",
    # Claude 4.0 generation
    "opus-4.0": "claude-opus-4-0",
    "opus-4-0": "claude-opus-4-0",
    "sonnet-4.0": "claude-sonnet-4-0",
    "sonnet-4-0": "claude-sonnet-4-0",
    "opus-4": "claude-opus-4-0",
    "sonnet-4": "claude-sonnet-4-0",
    "haiku-4": "claude-haiku-4-5",  # No haiku 4.0, point to 4.5
    # Claude 3.5 generation
    "sonnet-3.5": "claude-3-5-sonnet-20241022",
    "sonnet-3-5": "claude-3-5-sonnet-20241022",
    "haiku-3.5": "claude-3-5-haiku-20241022",
    "haiku-3-5": "claude-3-5-haiku-20241022",
    # Claude 3 generation
    "opus-3": "claude-3-opus-20240229",
    "sonnet-3": "claude-3-sonnet-20240229",
    "haiku-3": "claude-3-haiku-20240307",
}


def resolve_claude_model(model: str) -> str:
    """Resolve Claude model alias to full model ID.

    Handles formats:
        - "claude" or "anthropic" → claude-sonnet-4-6 (latest)
        - "sonnet", "haiku", "opus" → latest version of that model
        - "opus-4.6", "opus-4-6" → claude-opus-4-6
        - "anthropic/sonnet-4.6", "claude/haiku" → resolved model
        - "claude-opus-4-6" → passed through (already valid)
        - Full model IDs with dates → passed through unchanged

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
            # Recursively resolve the suffix
            return resolve_claude_model(suffix)

    # Normalize: lowercase and replace dots with dashes for lookup
    normalized = model.lower().replace(".", "-")

    # Direct alias lookup (try both original and normalized)
    if model.lower() in CLAUDE_MODEL_ALIASES:
        return CLAUDE_MODEL_ALIASES[model.lower()]
    if normalized in CLAUDE_MODEL_ALIASES:
        return CLAUDE_MODEL_ALIASES[normalized]

    # Handle "claude-X" format by stripping "claude-" prefix and re-resolving
    if normalized.startswith("claude-"):
        suffix = normalized[7:]  # Remove "claude-"
        if suffix in CLAUDE_MODEL_ALIASES:
            return CLAUDE_MODEL_ALIASES[suffix]

    # Return as-is (assume it's already a valid model ID)
    return model
