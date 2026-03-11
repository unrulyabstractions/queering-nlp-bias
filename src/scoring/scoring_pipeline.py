"""Core scoring pipeline logic.

This module contains the complete scoring pipeline that dynamically
discovers and runs all active scoring methods without hardcoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from src.common.callback_types import LogFn
from src.common.text import strip_eos_tokens
from src.inference import ModelRunner
from src.inference.embedding_runner import EmbeddingRunner

from .scoring_config import ScoringConfig, StringSelection
from .scoring_method_registry import get_method, get_params_class, iter_methods
from .scoring_data import TrajectoryData
from .scoring_output import ScoringOutput, ScoringResult


# ══════════════════════════════════════════════════════════════════════════════
# TEXT PROCESSING
# ══════════════════════════════════════════════════════════════════════════════


def get_text_for_scoring(
    traj: TrajectoryData,
    config: ScoringConfig,
    eos_token: str | None = None,
) -> str:
    """Get the text to score based on string_selection config."""
    selection = config.string_selection

    if selection == StringSelection.WholeTrajectory:
        text = traj.full_text
    elif (
        selection == StringSelection.WholeContinuation
        or selection == StringSelection.AfterTrunk
    ):
        text = traj.response
    elif selection == StringSelection.AfterBranch:
        text = traj.response_after_branch
    else:
        text = traj.response

    markers = [eos_token] if eos_token else None
    return strip_eos_tokens(text, markers)


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE TRAJECTORY SCORING
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class TrajectoryScores:
    """All scores for a single trajectory - generic storage."""

    # Method name -> (scores, raw_responses)
    method_scores: dict[str, tuple[list[Any], list[str]]] = field(default_factory=dict)

    def get_scores(self, method_name: str) -> list[Any]:
        """Get scores for a method."""
        if method_name in self.method_scores:
            return self.method_scores[method_name][0]
        return []

    def get_raw_responses(self, method_name: str) -> list[str]:
        """Get raw responses for a method."""
        if method_name in self.method_scores:
            return self.method_scores[method_name][1]
        return []


def score_trajectory(
    runner: ModelRunner | None,
    embedder: EmbeddingRunner | None,
    config: ScoringConfig,
    traj: TrajectoryData,
    log_fn: LogFn | None = None,
    eos_token: str | None = None,
) -> TrajectoryScores:
    """Score a single trajectory with all active methods.

    Dynamically discovers and runs all methods that have data configured.
    """
    text = get_text_for_scoring(traj, config, eos_token)
    result = TrajectoryScores()

    # Iterate over all methods that have data
    for method_name in config.get_active_methods():
        items = config.get_method_items(method_name)
        if not items:
            continue

        params_class = get_params_class(method_name)
        params = config.get_scoring_params(method_name)
        score_fn = get_method(method_name)

        # Check resource requirements
        if params_class.requires_runner and runner is None:
            raise ValueError(f"{method_name} requires a model runner but none provided")
        if params_class.requires_embedder and embedder is None:
            raise ValueError(f"{method_name} requires an embedding runner but none provided")

        # Run the scoring function
        scores, raw_responses = score_fn(text, items, params, runner, embedder, log_fn)
        result.method_scores[method_name] = (scores, raw_responses)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ScoringPipelineResult:
    """Result of running the full scoring pipeline."""

    results: list[ScoringResult]
    output: ScoringOutput


# Callback type for progress reporting: (current_idx, total, trajectory)
ScoringProgressFn = Callable[[int, int, TrajectoryData], None]


def run_scoring_pipeline(
    config: ScoringConfig,
    trajectories: list[TrajectoryData],
    branches: list[str],
    arm_texts: dict[str, str],
    generation_file: str = "",
    scoring_file: str = "",
    prefix_logprobs: dict[str, Any] | None = None,
    progress_fn: ScoringProgressFn | None = None,
    log_fn: LogFn | None = None,
    eos_token: str | None = None,
) -> ScoringPipelineResult:
    """Run the complete scoring pipeline.

    Dynamically loads models based on which methods are active.
    """
    # Load models based on what's needed
    runner: ModelRunner | None = None
    embedder: EmbeddingRunner | None = None

    if config.needs_runner():
        if not config.model:
            raise ValueError("No model specified for LLM-based scoring methods")
        runner = ModelRunner(config.model)

    if config.needs_embedder():
        embedder = EmbeddingRunner(config.embedding_model)

    # Score all trajectories
    results: list[ScoringResult] = []
    for i, traj in enumerate(trajectories):
        if progress_fn:
            progress_fn(i, len(trajectories), traj)

        scores = score_trajectory(runner, embedder, config, traj, log_fn=log_fn, eos_token=eos_token)

        results.append(
            ScoringResult.from_trajectory(traj, scores)
        )

    # Create output
    output = ScoringOutput.create(
        generation_file=generation_file,
        scoring_file=scoring_file,
        scoring_config=config,
        results=results,
        branches=branches,
        arm_texts=arm_texts,
        prefix_logprobs=prefix_logprobs,
    )

    return ScoringPipelineResult(results=results, output=output)
