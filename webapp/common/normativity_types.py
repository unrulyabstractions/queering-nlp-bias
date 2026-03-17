"""Core types and math for normativity analysis.

All data structures and calculations in one place. No UI dependencies.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# ════════════════════════════════════════════════════════════════════════════════
# Type Aliases
# ════════════════════════════════════════════════════════════════════════════════

Scoring = float | None  # Single judge score for one question (None = parse error)
Structure = Scoring  # Alias for use in judge evaluation context
System = list[Structure]  # Scores for all questions in one trajectory sample

# ════════════════════════════════════════════════════════════════════════════════
# Vector Math
# ════════════════════════════════════════════════════════════════════════════════


def compute_l2_norm(scores: System) -> Scoring:
    """L2 norm (magnitude) of score vector."""
    if not scores:
        return 0.0
    return math.sqrt(sum(s * s for s in scores))


def compute_l2_distance(a: System, b: System) -> Scoring:
    """L2 distance between two score vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    return compute_l2_norm([x - y for x, y in zip(a, b)])


def compute_core_diversity(system: System) -> Scoring:
    """Compute effective number of structures represented (exp of entropy).

    Takes a System (list of scores), normalizes to probability distribution,
    and returns exp(Shannon entropy) = Hill number D_1.

    The result is in units of "effective number of structures" - ranges from
    1 (all weight on one structure) to len(system) (uniform distribution).
    """
    if not system or len(system) == 0:
        return 1.0

    # Handle edge case: all zeros
    total = sum(abs(s) for s in system)
    if total < 1e-10:
        return float(len(system))  # Uniform when no signal

    # Normalize to probability distribution (use absolute values)
    probs = [abs(s) / total for s in system]

    # Compute Shannon entropy: H = -sum(p * log(p))
    entropy = 0.0
    for p in probs:
        if p > 1e-10:  # Avoid log(0)
            entropy -= p * math.log(p)

    # Return exp(entropy) = effective number of structures
    return math.exp(entropy)


def compute_deviation(scores: System, reference: System) -> System:
    """Element-wise deviation: scores - reference."""
    if not scores:
        return []
    if not reference or len(reference) != len(scores):
        return [0.0] * len(scores)
    return [s - r for s, r in zip(scores, reference)]


# ════════════════════════════════════════════════════════════════════════════════
# Aggregation
# ════════════════════════════════════════════════════════════════════════════════


def compute_mean(values: list[Scoring]) -> Scoring:
    """Arithmetic mean."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def compute_system_means(samples: list[System]) -> System:
    """Compute mean per dimension across sample vectors."""
    if not samples:
        return []
    n_dims = len(samples[0])
    return [compute_mean([s[i] for s in samples if len(s) > i]) for i in range(n_dims)]


def compute_system_stds(samples: list[System]) -> System:
    """Compute standard deviation per dimension across sample vectors."""
    if not samples or len(samples) < 2:
        return [0.0] * len(samples[0]) if samples else []
    n_dims = len(samples[0])
    means = compute_system_means(samples)
    stds = []
    for i in range(n_dims):
        vals = [s[i] for s in samples if len(s) > i]
        if len(vals) < 2:
            stds.append(0.0)
        else:
            variance = sum((v - means[i]) ** 2 for v in vals) / len(vals)
            stds.append(math.sqrt(variance))
    return stds


# ════════════════════════════════════════════════════════════════════════════════
# Score Parsing
# ════════════════════════════════════════════════════════════════════════════════


def parse_judge_score(answer: str) -> Scoring:
    """Parse judge response to 0-1 score.

    Robust parsing that handles reasoning traces by prioritizing:
    1. Strip <think>...</think> blocks (reasoning model output)
    2. Extract <answer>...</answer> tags if present
    3. Extract content after "ANSWER:" if present
    4. Check if last word/token is a clear answer (number, YES/NO)
    5. Check last line for clear answer
    6. Conservative fallback - only accept standalone answers, not numbers in reasoning

    Handles:
        - YES/TRUE → 1.0, NO/FALSE → 0.0
        - Floats in 0-1 range used directly (0.0, 0.5, 0.75, 1.0)
        - Integers 0-1 used directly
        - Numbers 2-10 scaled to 0-1 (e.g., 7 → 0.7)
        - Returns None if unparseable (ERROR state)
    """
    text = answer.strip()
    if not text:
        print("❌ JUDGE PARSE ERROR: Empty response")
        return None

    # Strip <think>...</think> blocks from reasoning models (e.g., Qwen)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not text:
        print("❌ JUDGE PARSE ERROR: Only thinking block, no answer")
        return None

    # FIRST: Check if entire response is just a number (simplest case)
    # Handles: "0.5", ".5", "1", "0.75", etc.
    clean_text = text.strip()
    if re.match(r"^(\d+\.?\d*|\.\d+)$", clean_text):
        value = float(clean_text)
        if 0.0 <= value <= 1.0:
            return value
        if 1.0 < value <= 10.0:
            return value / 10.0
        return max(0.0, min(1.0, value))

    # Check for <answer>...</answer> tags - most reliable
    answer_tag_match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if answer_tag_match:
        text = answer_tag_match.group(1).strip()
        if not text:
            print("❌ JUDGE PARSE ERROR: Empty <answer> tag")
            return None
        # With explicit answer tags, parse the content directly
        return _parse_answer_content(text)

    # Check for "ANSWER:" prefix - also reliable
    answer_prefix_match = re.search(r"ANSWER:\s*(.+)", text, flags=re.IGNORECASE)
    if answer_prefix_match:
        text = answer_prefix_match.group(1).strip()
        if not text:
            print("❌ JUDGE PARSE ERROR: Empty ANSWER: content")
            return None
        # With explicit ANSWER: marker, parse the content directly
        return _parse_answer_content(text)

    # Get lines and check for incomplete response (ends mid-sentence)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        print("❌ JUDGE PARSE ERROR: No content found")
        return None

    last_line = lines[-1]

    # Check if last word is a number first (most common case: model just outputs "0.7")
    last_word = last_line.split()[-1].strip(".,!?:;\"'") if last_line.split() else ""
    if re.match(r"^(\d+\.?\d*|\.\d+)$", last_word):
        value = float(last_word)
        if 0.0 <= value <= 1.0:
            return value
        if 1.0 < value <= 10.0:
            return value / 10.0
        return max(0.0, min(1.0, value))

    # Check if response looks incomplete (ends without punctuation or answer)
    if not last_line.rstrip().endswith((".", "!", "?", "0", "1", "YES", "NO", "yes", "no", "Yes", "No")):
        # Response might be cut off - be very conservative
        last_word_upper = last_word.upper()
        if last_word_upper in ("YES", "TRUE"):
            return 1.0
        if last_word_upper in ("NO", "FALSE"):
            return 0.0
        print(f"❌ JUDGE PARSE ERROR: Incomplete response, no clear answer: ...{last_line[-50:]}")
        return None

    # Check if last line is a standalone answer (very short, just the answer)
    last_line_clean = last_line.strip(".,!?:;\"' ").upper()
    if last_line_clean in ("YES", "TRUE"):
        return 1.0
    if last_line_clean in ("NO", "FALSE"):
        return 0.0

    # Check last word for YES/NO (number already checked above)
    last_word_upper = last_word.upper()
    if last_word_upper in ("YES", "TRUE"):
        return 1.0
    if last_word_upper in ("NO", "FALSE"):
        return 0.0

    # Check if last line ends with a standalone number (after punctuation like "Answer: 0.7.")
    match = re.search(r"[:\s](\d+\.?\d*|\.\d+)\s*[.!?]?\s*$", last_line)
    if match:
        value = float(match.group(1))
        if 0.0 <= value <= 1.0:
            return value
        if 1.0 < value <= 10.0:
            return value / 10.0
        return max(0.0, min(1.0, value))

    # Do NOT fall back to searching for numbers in full text - too error prone
    # Numbers in reasoning (like "1 or 0", "between 0 and 1") should not be extracted
    print(f"❌ JUDGE PARSE ERROR: No clear answer found in: ...{text[-100:]}")
    return None


def _parse_answer_content(text: str) -> Scoring:
    """Parse answer content when we have explicit answer markers."""
    text = text.strip().upper()

    # YES/NO/TRUE/FALSE
    if text in ("YES", "TRUE"):
        return 1.0
    if text in ("NO", "FALSE"):
        return 0.0

    # Try to extract number (handles both "0.5" and ".5")
    match = re.search(r"(\d+\.?\d*|\.\d+)", text)
    if match:
        value = float(match.group(1))
        if 0.0 <= value <= 1.0:
            return value
        if 1.0 < value <= 10.0:
            return value / 10.0
        return max(0.0, min(1.0, value))

    print(f"❌ JUDGE PARSE ERROR: Cannot parse answer content: {text}")
    return None


# ════════════════════════════════════════════════════════════════════════════════
# Measurement Positions
# ════════════════════════════════════════════════════════════════════════════════


def get_word_positions(text: str) -> list[int]:
    """Character positions at end of each word (for measuring at word boundaries).

    Uses regex to find actual word boundaries, handling punctuation correctly.
    """
    if not text:
        return []
    # Find all word matches and return the end position of each
    import re
    return [m.end() for m in re.finditer(r'\S+', text)]


# ════════════════════════════════════════════════════════════════════════════════
# Data Types
# ════════════════════════════════════════════════════════════════════════════════


@dataclass
class GenerationNode:
    """A node in the prefix tree where we sample trajectories."""

    node_id: int
    name: str
    prefix: str
    label: str
    parent: int | None  # parent node_id, None for root
    depth: int


@dataclass
class NormativityEstimate:
    """Accumulated normativity samples for a sampling point."""

    node_id: int
    samples: list[System] = field(default_factory=list)
    trajectories: list[str] = field(default_factory=list)  # Generated texts
    logprobs: list[float] = field(default_factory=list)  # Generation logprobs (OpenAI)

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def core(self) -> System:
        """Mean score vector across all samples."""
        return compute_system_means(self.samples)

    @property
    def orient_std(self) -> System:
        """Standard deviation per dimension of sample orientations (deviations from core)."""
        if not self.samples:
            return []
        core = self.core
        orientations = [compute_deviation(sample, core) for sample in self.samples]
        return compute_system_stds(orientations)

    @property
    def mean_logprob(self) -> float | None:
        """Average logprob across all generations (None if no logprobs)."""
        valid = [lp for lp in self.logprobs if lp is not None]
        return sum(valid) / len(valid) if valid else None

    def get_orientation_for(self, reference_core: System) -> System:
        """Compute orientation relative to a reference point."""
        return compute_deviation(self.core, reference_core)
