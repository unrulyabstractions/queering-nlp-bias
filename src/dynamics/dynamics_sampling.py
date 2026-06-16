"""Estimate the system default ⟨Λ_n⟩(x_p) by sampling continuations from a prefix.

This is what makes dynamics paper-correct: the system default is the *expected*
attunement over continuations of the prefix (Eq. 7). We Monte-Carlo it by sampling
completions from the model, scoring each, and averaging.
"""

from __future__ import annotations

from src.common.default_config import STRING_SELECTION
from src.common.math.entropy_diversity.structure_aware import generalized_system_core
from src.common.text import strip_thinking_blocks
from src.inference import ModelRunner
from src.scoring.scorer import Scorer

System = list[float]


def _selected(text: str) -> str:
    """Apply the same string selection used during scoring (default: strip thinking)."""
    if STRING_SELECTION == "NonThinkingContinuation":
        return strip_thinking_blocks(text)
    return text


def estimate_system_default(
    runner: ModelRunner,
    scorer: Scorer,
    prompt: str,
    prefill: str,
    prefix_text: str,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
) -> System:
    """Sample continuations from the prefix and average their attunements → ⟨Λ_n⟩(x_p).

    Sampling at temperature 1.0 makes the uniform mean an unbiased Monte-Carlo
    estimate of E[Λ_n(y) | y continues x_p] (paper Eq. 7). Each continuation is
    scored on its full prefix+completion text, exactly like a trajectory, with the
    same string selection used elsewhere in the pipeline.
    """
    attunements: list[System] = []
    for _ in range(max(1, n_samples)):
        traj = runner.generate_trajectory_from_prompt(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            prefilling=prefill + prefix_text,
        )
        scores = scorer.score(_selected(prefix_text + traj.generated_text))
        if scores:
            attunements.append(scores)
    if not attunements:
        return []
    weights = [1.0 / len(attunements)] * len(attunements)
    return list(generalized_system_core(attunements, weights))
