#!/usr/bin/env python3
"""Score trajectories with categorical judgments and similarity scoring.

Usage:
    python scripts/score_trajectories.py trials/scoring/<scoring>.json out/gen_<gen>.json

Outputs:
    out/score_<gen>_<scoring>.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import (
    GenerationOutputData,
    JudgmentOutput,
    JudgmentResult,
    ScoringConfig,
    TrajectoryData,
)
from schemas.script_utils import log_step, oneline

from src.common.log import log, log_section
from src.common.viz_utils import preview
from src.inference import ModelRunner
from src.inference.embedding_runner import EmbeddingRunner

# ══════════════════════════════════════════════════════════════════════════════
# Core Algorithm
# ══════════════════════════════════════════════════════════════════════════════


def judge_single_question(
    runner: ModelRunner,
    config: ScoringConfig,
    text: str,
    question: str,
) -> tuple[int | None, str]:
    """Judge a single question for a trajectory. Returns (score, raw_response)."""
    prompt = config.build_judgment_prompt(text, question)
    response = runner.generate(
        prompt=prompt,
        max_new_tokens=config.max_tokens,
        temperature=0.0,  # Always greedy
        prefilling=runner.skip_thinking_prefix,
    )
    score = config.parse_judgment(response)
    return score, response


def score_trajectory_categorical(
    runner: ModelRunner,
    config: ScoringConfig,
    traj: TrajectoryData,
) -> tuple[list[int | None], list[str]]:
    """Score a trajectory on all categorical judgments.

    Returns:
        Tuple of (scores, raw_judgments)
    """
    scores = []
    raw_judgments = []
    full_text = traj.full_text

    for i, question in enumerate(config.categorical_judgements):
        score, response = judge_single_question(runner, config, full_text, question)
        scores.append(score)
        raw_judgments.append(response)

        score_str = str(score) if score is not None else "?"
        log(f"    {preview(question, 40)} -> {score_str}")

    return scores, raw_judgments


def score_trajectory_similarity(
    embedder: EmbeddingRunner,
    config: ScoringConfig,
    traj: TrajectoryData,
) -> list[float]:
    """Score a trajectory on all similarity references.

    Returns:
        List of similarity scores (0-1).
    """
    similarities = embedder.similarities(
        text=traj.full_text,
        references=config.similarity_scoring,
    )

    for i, ref in enumerate(config.similarity_scoring):
        log(f"    {preview(ref, 40)} -> {similarities[i]:.3f}")

    return similarities


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Steps
# ══════════════════════════════════════════════════════════════════════════════


def step_load_models(
    config: ScoringConfig,
) -> tuple[ModelRunner | None, EmbeddingRunner | None]:
    """Load scoring models as needed."""
    log_step(1, "Load models")

    runner = None
    embedder = None

    if config.categorical_judgements:
        if not config.model:
            raise ValueError("No model specified for categorical judgments")
        log(f"  Judge model: {config.model}")
        runner = ModelRunner(config.model)

    if config.similarity_scoring:
        log(f"  Embedding model: {config.embedding_model}")
        embedder = EmbeddingRunner(config.embedding_model)

    return runner, embedder


def step_score_trajectories(
    runner: ModelRunner | None,
    embedder: EmbeddingRunner | None,
    config: ScoringConfig,
    trajectories: list[TrajectoryData],
) -> list[JudgmentResult]:
    """Score all trajectories with configured scoring methods."""
    log_step(2, "Score trajectories", f"{len(trajectories)} trajectories")

    results = []
    for i, traj in enumerate(trajectories):
        log_section(f"Trajectory {i + 1}/{len(trajectories)} (branch: {traj.branch})")

        # Print prompt and response separately
        log(f'  Prompt: "{preview(oneline(traj.prompt), 60)}"', gap=1)
        log(f'  Response: "{preview(oneline(traj.response), 60)}"')

        # Categorical judgments
        scores: list[int | None] = []
        raw_judgments: list[str] = []
        if config.categorical_judgements and runner:
            log("  Categorical:")
            scores, raw_judgments = score_trajectory_categorical(runner, config, traj)

        # Similarity scoring
        similarity_scores: list[float] = []
        if config.similarity_scoring and embedder:
            log("  Similarity:")
            similarity_scores = score_trajectory_similarity(embedder, config, traj)

        results.append(
            JudgmentResult.from_trajectory(
                traj, scores, raw_judgments, similarity_scores
            )
        )

    return results


def step_save_output(
    results: list[JudgmentResult],
    config: ScoringConfig,
    scoring_path: Path,
    gen_path: Path,
    branches: list[str],
    groups: dict[str, str],
    prefix_logprobs: dict[str, Any] | None = None,
) -> Path:
    """Save judgment results to output file."""
    log_step(3, "Save output")

    output = JudgmentOutput.create(
        generation_file=str(gen_path),
        scoring_file=str(scoring_path),
        scoring_config=config,
        results=results,
        branches=branches,
        groups=groups,
        prefix_logprobs=prefix_logprobs,
    )

    out_path = JudgmentOutput.compute_output_path(gen_path, scoring_path)
    output.save(out_path)

    log(f"  Saved judgments to {out_path}")
    output.summarize()

    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════


def score_trajectories(
    config: ScoringConfig,
    scoring_path: Path,
    gen_data: GenerationOutputData,
    gen_path: Path,
) -> None:
    """Run scoring pipeline.

    Pipeline:
        1. Load models (judge and/or embedding)
        2. Score all trajectories
        3. Save output
    """
    log_section("Scoring Pipeline")
    log(f"  Scoring config: {scoring_path}")
    log(f"  Generation output: {gen_path}")
    log(f"  Trajectories: {len(gen_data.trajectories)}")
    if config.categorical_judgements:
        log(f"  Categorical judgments: {len(config.categorical_judgements)}")
    if config.similarity_scoring:
        log(f"  Similarity references: {len(config.similarity_scoring)}")

    runner, embedder = step_load_models(config)
    results = step_score_trajectories(runner, embedder, config, gen_data.trajectories)
    step_save_output(
        results, config, scoring_path, gen_path, gen_data.branches, gen_data.groups, gen_data.prefix_logprobs
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Parse arguments and run scoring pipeline."""
    parser = argparse.ArgumentParser(
        description="Score trajectories with scoring config"
    )
    parser.add_argument("scoring_config", help="Path to scoring config JSON")
    parser.add_argument("generation_output", help="Path to generation output JSON")
    args = parser.parse_args()

    scoring_path = Path(args.scoring_config)
    gen_path = Path(args.generation_output)
    config = ScoringConfig.load(scoring_path)
    gen_data = GenerationOutputData.load(gen_path)

    score_trajectories(
        config=config,
        scoring_path=scoring_path,
        gen_data=gen_data,
        gen_path=gen_path,
    )


if __name__ == "__main__":
    main()
