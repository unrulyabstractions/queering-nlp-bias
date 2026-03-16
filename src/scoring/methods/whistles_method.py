"""Dog whistle detection scoring method.

References:
-----------
Mendelsohn, J., Le Bras, R., Choi, Y., & Sap, M. (2023).
"From Dogwhistles to Bullhorns: Unveiling Coded Rhetoric with Language Models"
ACL 2023. https://aclanthology.org/2023.acl-long.845

GitHub: https://github.com/juliamendelsohn/dogwhistles
Demo: https://dogwhistles.allen.ai

Method:
-------
A dogwhistle is an expression that conveys one meaning to a broad audience
and a second (covert) meaning to a narrow in-group.

Type I: Signals speaker's group identity without changing literal proposition
  - Example: "slay" signals queer community membership

Type II: Alters the implied proposition for in-group
  - Example: "confirmed bachelor" literally means unmarried, covertly means gay

Scoring process:
1. Find glossary term matches in the input text
2. Use LLM to estimate P(coded | term, context) for each match via few-shot prompting
3. Aggregate probabilities via noisy-OR (any coded) or max (strongest signal)

Formula (noisy-OR):
  s(x) = 1 - prod(1 - p_j) for all matches j
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from src.common.callback_types import LogFn
from src.common.default_config import JUDGE_MAX_TOKENS
from src.common.file_io import load_json
from src.inference import ModelRunner
from src.inference.embedding_runner import EmbeddingRunner

from ..scoring_method_registry import ScoringMethodParams, register_method
from .llm_response_parsing import strip_thinking_content
from .logging.scoring_logging_utils import log_parse_failure


# ══════════════════════════════════════════════════════════════════════════════
# GLOSSARY ENTRY TYPE
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class GlossaryEntry:
    """A single dogwhistle glossary entry."""

    surface_form: str  # The word/phrase as it appears in text
    covert_meaning: str  # What the in-group understands
    whistle_type: str = "II"  # "I" (identity signal) or "II" (altered meaning)
    persona: str = ""  # The targeted/signaled group
    example_coded: str = ""  # Example where term is used with covert meaning
    example_literal: str = ""  # Example where term is used literally

    @classmethod
    def from_dict(cls, d: dict) -> GlossaryEntry:
        """Create from dict, handling various field names."""
        return cls(
            surface_form=d.get("surface_form", d.get("term", "")),
            covert_meaning=d.get("covert_meaning", ""),
            whistle_type=d.get("type", d.get("whistle_type", "II")),
            persona=d.get("persona", ""),
            example_coded=d.get("example_coded", d.get("example_context", "")),
            example_literal=d.get("example_literal", ""),
        )


# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class WhistlesParams(ScoringMethodParams):
    """Parameters for dog whistle detection scoring.

    Requires a glossary JSON file with entries containing:
    - surface_form: The word/phrase to detect
    - covert_meaning: The hidden meaning understood by in-group
    - type: "I" (identity signal) or "II" (altered proposition)
    - example_coded: Example with covert usage (for few-shot prompting)
    - example_literal: Example with literal usage (for few-shot prompting)
    """

    glossary_path: str = ""  # Path to glossary JSON file (required)
    aggregation: str = "noisy_or"  # "noisy_or" or "max"
    max_tokens: int = field(default_factory=lambda: JUDGE_MAX_TOKENS)

    # Registry metadata
    name: ClassVar[str] = "whistles"
    config_key: ClassVar[str] = "whistles"
    label_prefix: ClassVar[str] = "w"
    requires_runner: ClassVar[bool] = True
    requires_embedder: ClassVar[bool] = False

    _cli_args: ClassVar[dict[str, str]] = {
        "glossary_path": "--glossary-path",
        "aggregation": "--aggregation",
        "max_tokens": "--max-tokens",
    }


# ══════════════════════════════════════════════════════════════════════════════
# GLOSSARY LOADING AND MATCHING
# ══════════════════════════════════════════════════════════════════════════════


def load_glossary(path: str | Path) -> list[GlossaryEntry]:
    """Load glossary from JSON file."""
    data = load_json(Path(path))
    if not isinstance(data, list):
        raise ValueError(f"Glossary must be a JSON array, got {type(data)}")
    return [GlossaryEntry.from_dict(entry) for entry in data]


def find_glossary_matches(text: str, glossary: list[GlossaryEntry]) -> list[GlossaryEntry]:
    """Find all glossary entries whose surface form appears in text."""
    text_lower = text.lower()
    matches = []
    for entry in glossary:
        # Match as whole word/phrase using word boundaries
        pattern = r"\b" + re.escape(entry.surface_form.lower()) + r"\b"
        if re.search(pattern, text_lower):
            matches.append(entry)
    return matches


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDING AND RESPONSE PARSING
# ══════════════════════════════════════════════════════════════════════════════


def build_whistle_detection_prompt(
    text: str,
    entry: GlossaryEntry,
    demo_entries: list[GlossaryEntry],
) -> str:
    """Build few-shot prompt for whistle detection.

    Args:
        text: The text being analyzed
        entry: The glossary entry to check
        demo_entries: Other entries to use as few-shot examples
    """
    prompt = (
        "A dogwhistle is a word or phrase that conveys a hidden meaning to a specific "
        "in-group while appearing innocent to others.\n\n"
    )

    # Add few-shot examples from other entries
    for demo in demo_entries[:2]:
        if demo.example_coded:
            prompt += (
                f"Example (coded use):\n"
                f'Text: "{demo.example_coded}"\n'
                f'Term: "{demo.surface_form}"\n'
                f'Covert meaning: "{demo.covert_meaning}"\n'
                f"Confidence this is coded: 0.85\n\n"
            )
        if demo.example_literal:
            prompt += (
                f"Example (literal use):\n"
                f'Text: "{demo.example_literal}"\n'
                f'Term: "{demo.surface_form}"\n'
                f'Covert meaning: "{demo.covert_meaning}"\n'
                f"Confidence this is coded: 0.10\n\n"
            )

    # The actual query
    prompt += (
        f"Now analyze:\n"
        f'Text: "{text}"\n'
        f'Term: "{entry.surface_form}"\n'
        f'Covert meaning: "{entry.covert_meaning}"\n\n'
        f"What is the probability (0.0 to 1.0) that this term is being used with its "
        f"covert meaning in this context?\n"
        f"Reply with ONLY a single decimal number between 0.0 and 1.0."
    )

    return prompt


def parse_probability_response(response: str) -> float | None:
    """Parse a probability (0.0-1.0) from model response."""
    text = strip_thinking_content(response)

    # Try to find a decimal number in [0, 1]
    match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?|\.\d+)\b", text)
    if match:
        try:
            value = float(match.group(1))
            if 0.0 <= value <= 1.0:
                return value
        except ValueError:
            pass

    # Handle bare 0 or 1
    if text.strip() in ("0", "1"):
        return float(text.strip())

    return None


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════


def aggregate_noisy_or(probabilities: list[float]) -> float:
    """Aggregate using noisy-OR: P(at least one coded).

    P = 1 - prod(1 - p_i)

    Good for "is there ANY queer coding present?"
    """
    if not probabilities:
        return 0.0
    product = 1.0
    for p in probabilities:
        product *= 1.0 - p
    return 1.0 - product


def aggregate_max(probabilities: list[float]) -> float:
    """Aggregate using max: strongest individual signal.

    Good for "what's the strongest single signal?"
    """
    if not probabilities:
        return 0.0
    return max(probabilities)


# ══════════════════════════════════════════════════════════════════════════════
# REGISTERED SCORING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════


@register_method(WhistlesParams)
def score_whistles(
    text: str,
    items: list[str | list[str]],
    params: WhistlesParams,
    runner: ModelRunner | None = None,
    embedder: EmbeddingRunner | None = None,
    log_fn: LogFn | None = None,
) -> tuple[list[float], list[str]]:
    """Score text for dog whistle usage.

    Process:
    1. Load glossary from params.glossary_path
    2. Find all glossary term matches in text
    3. For each match, prompt LLM to estimate P(coded | term, context)
    4. Aggregate probabilities using noisy-OR or max

    Args:
        text: Text to analyze for dog whistles
        items: Not used (method uses glossary_path instead)
        params: Method parameters including glossary_path
        runner: Model runner for LLM inference
        embedder: Not used
        log_fn: Optional logging callback

    Returns:
        Tuple of ([aggregate_score], [raw_response_summary])
    """
    if runner is None:
        raise ValueError("Whistles scoring requires a model runner")

    if not params.glossary_path:
        raise ValueError("Whistles scoring requires glossary_path parameter")

    # Load glossary
    glossary_path = Path(params.glossary_path)
    if not glossary_path.exists():
        raise FileNotFoundError(f"Glossary file not found: {glossary_path}")

    glossary = load_glossary(glossary_path)
    if log_fn:
        log_fn(f"Loaded glossary with {len(glossary)} entries")

    # Find matches
    matches = find_glossary_matches(text, glossary)
    if log_fn:
        log_fn(f"Found {len(matches)} term matches in text")

    if not matches:
        # No matches = no coding detected
        return [0.0], ["no_matches"]

    # Score each match
    match_results: list[dict[str, Any]] = []
    probabilities: list[float] = []

    for match in matches:
        # Use other entries as few-shot demos
        demos = [e for e in glossary if e.surface_form != match.surface_form]

        prompt = build_whistle_detection_prompt(text, match, demos)
        response = runner.generate(
            prompt=prompt,
            max_new_tokens=params.max_tokens,
            temperature=0.0,
            prefilling=runner.skip_thinking_prefix,
        )

        p_coded = parse_probability_response(response)
        if p_coded is None:
            if log_fn:
                log_parse_failure("WHISTLES", match.surface_form, response, log_fn)
            p_coded = 0.5  # Default to uncertain

        probabilities.append(p_coded)
        match_results.append(
            {
                "term": match.surface_form,
                "covert_meaning": match.covert_meaning,
                "p_coded": p_coded,
            }
        )

        if log_fn:
            log_fn(f"  {match.surface_form}: p={p_coded:.3f} -> {match.covert_meaning}")

    # Aggregate
    if params.aggregation == "max":
        final_score = aggregate_max(probabilities)
    else:
        final_score = aggregate_noisy_or(probabilities)

    if log_fn:
        log_fn(f"Aggregate ({params.aggregation}): {final_score:.3f}")

    # Build raw response summary
    raw_summary = "; ".join(
        f"{r['term']}={r['p_coded']:.2f}" for r in match_results
    )

    return [final_score], [raw_summary]
