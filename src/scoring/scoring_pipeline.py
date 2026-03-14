"""Scoring pipeline - orchestrates scoring many trajectories.

For scoring individual text, use Scorer directly:
    scorer = Scorer(config)
    scores = scorer.score(text)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.common.callback_types import LogFn
from src.common.device_utils import clear_gpu_memory
from src.common.profiler import profile

from .scorer import Scorer
from .scoring_config import ScoringConfig
from .scoring_data import TrajectoryData
from .scoring_output import ScoringOutput, ScoringResult

# Clear memory after every trajectory to prevent MLX cache accumulation
# Each trajectory has ~10 LLM calls, cache grows ~2GB per batch
MEMORY_CLEAR_INTERVAL = 1


@dataclass
class ScoringPipelineResult:
    """Result of running the full scoring pipeline."""

    results: list[ScoringResult]
    output: ScoringOutput


ScoringProgressFn = Callable[[int, int, TrajectoryData], None]


@profile
def run_scoring_pipeline(
    config: ScoringConfig,
    trajectories: list[TrajectoryData],
    arm_names: list[str],
    arm_texts: dict[str, str],
    generation_file: str = "",
    scoring_file: str = "",
    progress_fn: ScoringProgressFn | None = None,
    log_fn: LogFn | None = None,
    eos_token: str | None = None,
) -> ScoringPipelineResult:
    """Run scoring pipeline on trajectories."""
    scorer = Scorer(config)

    results: list[ScoringResult] = []
    for i, traj in enumerate(trajectories):
        if progress_fn:
            progress_fn(i, len(trajectories), traj)

        method_scores = scorer.score_trajectory(traj, log_fn, eos_token)
        results.append(ScoringResult.from_method_scores(traj, method_scores))

        # Periodic memory cleanup to prevent accumulation
        if (i + 1) % MEMORY_CLEAR_INTERVAL == 0:
            clear_gpu_memory()

    output = ScoringOutput.create(
        generation_file=generation_file,
        scoring_file=scoring_file,
        scoring_config=config,
        results=results,
        arm_names=arm_names,
        arm_texts=arm_texts,
    )

    # Cleanup scorer models to free memory
    scorer.cleanup()

    return ScoringPipelineResult(results=results, output=output)
