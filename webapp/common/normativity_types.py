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

Scoring = float  # Single judge score for one question
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
    1. Final line/word of response
    2. Standalone YES/NO/TRUE/FALSE (word boundaries)
    3. Numbers anywhere in text

    Handles:
        - YES/TRUE → 1.0, NO/FALSE → 0.0
        - Floats in 0-1 range used directly (0.0, 0.5, 0.75, 1.0)
        - Integers 0-1 used directly
        - Numbers 2-10 scaled to 0-1 (e.g., 7 → 0.7)
        - Defaults to 0.5 if unparseable
    """
    text = answer.strip()
    if not text:
        return 0.5

    # First, check the last line (most likely to be the final answer)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    last_line = lines[-1].upper() if lines else text.upper()

    # Check last line for standalone YES/NO (word boundaries)
    if re.search(r"\bYES\b", last_line) or re.search(r"\bTRUE\b", last_line):
        return 1.0
    if re.search(r"\bNO\b", last_line) or re.search(r"\bFALSE\b", last_line):
        return 0.0

    # Check last line for a number
    match = re.search(r"\b(\d+\.?\d*)\b", last_line)
    if match:
        value = float(match.group(1))
        if 0.0 <= value <= 1.0:
            return value
        if 1.0 < value <= 10.0:
            return value / 10.0
        return max(0.0, min(1.0, value))

    # Fall back to checking full text with word boundaries
    full_upper = text.upper()
    if re.search(r"\bYES\b", full_upper) or re.search(r"\bTRUE\b", full_upper):
        return 1.0
    if re.search(r"\bNO\b", full_upper) or re.search(r"\bFALSE\b", full_upper):
        return 0.0

    # Try to extract any number from full text
    match = re.search(r"-?\d+\.?\d*", text)
    if match:
        value = float(match.group())
        if 0.0 <= value <= 1.0:
            return value
        if 1.0 < value <= 10.0:
            return value / 10.0
        return max(0.0, min(1.0, value))

    return 0.5


# ════════════════════════════════════════════════════════════════════════════════
# Measurement Positions
# ════════════════════════════════════════════════════════════════════════════════


def get_word_positions(text: str) -> list[int]:
    """Character positions at end of each word (for measuring at word boundaries)."""
    if not text:
        return []
    positions = []
    for i, char in enumerate(text):
        if char == " " and i > 0:
            positions.append(i)
    if text and len(text) not in positions:
        positions.append(len(text))
    return positions


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
