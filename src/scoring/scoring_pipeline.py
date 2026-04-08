"""Scoring pipeline - orchestrates scoring many trajectories.

For scoring individual text, use Scorer directly:
    scorer = Scorer(config)
    scores = scorer.score(text)
"""

from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
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

# Number of parallel workers for API-based scoring
# Higher values = faster but more API rate limiting risk
# 4 workers is safe for most API rate limits
API_PARALLEL_WORKERS = 4

# Checkpoint settings for crash recovery
CHECKPOINT_INTERVAL = 10  # Save every N trajectories
CHECKPOINT_DIR = Path("/tmp/scoring_checkpoints")


@dataclass
class ScoringPipelineResult:
    """Result of running the full scoring pipeline."""

    results: list[ScoringResult]
    output: ScoringOutput


ScoringProgressFn = Callable[[int, int, TrajectoryData], None]


def _is_api_backend(config: ScoringConfig) -> bool:
    """Check if the scoring config uses an API backend (Anthropic/OpenAI)."""
    model = config.model.lower() if config.model else ""
    return any(x in model for x in ["anthropic", "claude", "openai", "gpt-"])


def _get_checkpoint_path(generation_file: str, scoring_file: str) -> Path:
    """Get unique checkpoint path based on input files."""
    key = f"{generation_file}:{scoring_file}"
    hash_id = hashlib.md5(key.encode()).hexdigest()[:12]
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / f"checkpoint_{hash_id}.json"


def _save_checkpoint(
    checkpoint_path: Path,
    results: list[ScoringResult],
    start_idx: int,
) -> None:
    """Save scoring checkpoint for crash recovery."""
    data = {
        "start_idx": start_idx,
        "results": [r.to_dict() for r in results],
    }
    # Write atomically via temp file
    tmp_path = checkpoint_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f)
    tmp_path.rename(checkpoint_path)


def _load_checkpoint(
    checkpoint_path: Path,
) -> tuple[int, list[ScoringResult]] | None:
    """Load checkpoint if it exists. Returns (start_idx, results) or None."""
    if not checkpoint_path.exists():
        return None
    try:
        with open(checkpoint_path) as f:
            data = json.load(f)
        results = [ScoringResult.from_dict(r) for r in data["results"]]
        start_idx = data["start_idx"]
        print(f"  [Checkpoint] Resuming from trajectory {start_idx} ({len(results)} cached)")
        return start_idx, results
    except Exception as e:
        print(f"  [Checkpoint] Failed to load: {e}, starting fresh")
        return None


def _clear_checkpoint(checkpoint_path: Path) -> None:
    """Remove checkpoint after successful completion."""
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("  [Checkpoint] Cleared")


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
    """Run scoring pipeline on trajectories.

    Supports checkpoint-based crash recovery. Checkpoints are saved every
    CHECKPOINT_INTERVAL trajectories to /tmp/scoring_checkpoints/.

    Uses parallel scoring for API backends (Anthropic/OpenAI) to speed up
    processing. Local backends run sequentially to avoid GPU contention.
    """
    scorer = Scorer(config)
    # Parallel scoring disabled due to API rate limit issues
    # TODO: Re-enable with proper rate limiting
    # use_parallel = _is_api_backend(config)
    use_parallel = False

    # Get checkpoint path and check for existing checkpoint
    checkpoint_path = _get_checkpoint_path(generation_file, scoring_file)
    checkpoint = _load_checkpoint(checkpoint_path)

    if use_parallel:
        results = _run_parallel_scoring(
            scorer, trajectories, progress_fn, log_fn, eos_token
        )
    else:
        results = _run_sequential_scoring(
            scorer, trajectories, progress_fn, log_fn, eos_token,
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
        )

    output = ScoringOutput.create(
        generation_file=generation_file,
        scoring_file=scoring_file,
        scoring_config=config,
        results=results,
        arm_names=arm_names,
        arm_texts=arm_texts,
    )

    # Clear checkpoint after successful completion
    _clear_checkpoint(checkpoint_path)

    # Cleanup scorer models to free memory
    scorer.cleanup()

    return ScoringPipelineResult(results=results, output=output)


def _run_sequential_scoring(
    scorer: Scorer,
    trajectories: list[TrajectoryData],
    progress_fn: ScoringProgressFn | None,
    log_fn: LogFn | None,
    eos_token: str | None,
    checkpoint_path: Path | None = None,
    checkpoint: tuple[int, list[ScoringResult]] | None = None,
) -> list[ScoringResult]:
    """Run scoring sequentially with checkpoint support.

    Args:
        scorer: Scorer instance
        trajectories: Trajectories to score
        progress_fn: Progress callback
        log_fn: Logging callback
        eos_token: EOS token for text extraction
        checkpoint_path: Path to save checkpoints (None = no checkpointing)
        checkpoint: Existing checkpoint (start_idx, results) or None
    """
    # Resume from checkpoint if available
    if checkpoint:
        start_idx, results = checkpoint
    else:
        start_idx, results = 0, []

    n_total = len(trajectories)

    for i in range(start_idx, n_total):
        traj = trajectories[i]

        if progress_fn:
            progress_fn(i, n_total, traj)

        method_scores = scorer.score_trajectory(traj, log_fn, eos_token)
        results.append(ScoringResult.from_method_scores(traj, method_scores))

        # Periodic memory cleanup to prevent accumulation
        if (i + 1) % MEMORY_CLEAR_INTERVAL == 0:
            clear_gpu_memory()

        # Save checkpoint periodically
        if checkpoint_path and (i + 1) % CHECKPOINT_INTERVAL == 0:
            _save_checkpoint(checkpoint_path, results, i + 1)

    return results


def _run_parallel_scoring(
    scorer: Scorer,
    trajectories: list[TrajectoryData],
    progress_fn: ScoringProgressFn | None,
    log_fn: LogFn | None,
    eos_token: str | None,
) -> list[ScoringResult]:
    """Run scoring in parallel (for API backends).

    Uses ThreadPoolExecutor since API calls are I/O bound.
    Results are collected in original order.
    """
    n_total = len(trajectories)
    results: list[ScoringResult | None] = [None] * n_total
    completed = 0

    def score_one(idx: int, traj: TrajectoryData) -> tuple[int, ScoringResult]:
        method_scores = scorer.score_trajectory(traj, None, eos_token)
        return idx, ScoringResult.from_method_scores(traj, method_scores)

    with ThreadPoolExecutor(max_workers=API_PARALLEL_WORKERS) as executor:
        futures = {
            executor.submit(score_one, i, traj): i
            for i, traj in enumerate(trajectories)
        }

        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
            completed += 1

            # Progress callback (not thread-safe for detailed logging)
            if progress_fn:
                progress_fn(completed - 1, n_total, trajectories[idx])

            # Log result if log_fn provided
            if log_fn:
                _log_result(log_fn, completed, n_total, trajectories[idx], result)

    return results  # type: ignore


def _log_result(
    log_fn: LogFn,
    completed: int,
    total: int,
    traj: TrajectoryData,
    result: ScoringResult,
) -> None:
    """Log scoring result for a trajectory."""
    log_fn(f"\nTrajectory {completed}/{total} (arm: {traj.arm})\n")
    text_preview = traj.text[:100].replace("\n", " ") + "..."
    log_fn(f'  Selected: "{text_preview}"\n')
    for label, score in result.structure_scores.items():
        if score is not None:
            log_fn(f"    [{label}] -> {score:.3f}\n")
        else:
            log_fn(f"    [{label}] -> ?\n")
    sys.stdout.flush()
