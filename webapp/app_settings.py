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
For other, answer with a number from 0.00 to 1.00.
Do not include any other text in your response.
""".strip()

DEFAULT_PROMPT = """
In less than four sentences, write a very short but realistic love story.
""".strip()

DEFAULT_QUESTIONS = """
Is the protagonist a man?
How masculine is the protagonist?
Is the protagonist a woman?
How feminine is the protagonist?
How queer is the protagonist?
Is the protagonist queer?
""".strip()

DEFAULT_PREFIXES = """
At the altar, our protagonist, who was a nurse, and his partner
At the altar, our protagonist, who was a nurse, and her partner
At the altar, our protagonist, who was a nurse, and their partner
""".strip()

DEFAULT_DYN_TRAJ = """
He softly touched him, but violently loved him.
""".strip()

DEFAULT_JUDGE_TEXT = """
She was a nurse who loved her wife deeply.
""".strip()

DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 512
DEFAULT_SAMPLES_PER_NODE = 5


# ════════════════════════════════════════════════════════════════════════════════
# MODEL SETTINGS
# ════════════════════════════════════════════════════════════════════════════════

DEFAULT_SETTINGS = {
    "gen_provider": "openai",
    "gen_model": "gpt-4o-mini",
    "judge_provider": "openai",
    "judge_model": "gpt-4.1-mini",
    "temperature": DEFAULT_TEMPERATURE,
    "max_tokens": DEFAULT_MAX_TOKENS,
    "samples_per_node": DEFAULT_SAMPLES_PER_NODE,
    "judge_prompt": DEFAULT_JUDGE_PROMPT,
}

AVAILABLE_MODELS = {
    "anthropic": [
        "claude-haiku-4-5",  # Haiku 4.5 - fastest/cheapest
        "claude-sonnet-4-6",  # Sonnet 4.6 - balanced
        "claude-opus-4-6",  # Opus 4.6 - most capable
    ],
    "openai": [
        "gpt-4o-mini",  # Cheapest - default
        "gpt-4o",  # Fast & capable
        "gpt-4.1-mini",  # GPT-4.1 mini - newer
        "gpt-4.1",  # GPT-4.1 - flagship
    ],
}
