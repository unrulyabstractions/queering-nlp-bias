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


EOS_MARKERS = ["<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>"]


def _strip_eos_tokens(text: str) -> str:
    """Remove EOS tokens from end of text."""
    result = text.rstrip()
    for marker in EOS_MARKERS:
        if result.endswith(marker):
            result = result[:-len(marker)].rstrip()
    return result


def get_text_for_scoring(traj: TrajectoryData, config: ScoringConfig) -> str:
    """Get the text to score based on string_selection config."""
    from schemas.scoring import StringSelection

    selection = config.string_selection
    if selection == StringSelection.WholeTrajectory:
        text = traj.full_text
    elif selection == StringSelection.WholeContinuation:
        text = traj.response
    elif selection == StringSelection.AfterTrunk:
        # For AfterTrunk, we want the continuation after the trunk
        # response already contains continuation_text which is after trunk
        text = traj.response
    elif selection == StringSelection.AfterBranch:
        # For AfterBranch, we want text after the branch point
        # Use response_after_branch which has the branch token stripped
        text = traj.response_after_branch
    else:
        text = traj.response  # Default to continuation

    return _strip_eos_tokens(text)


def score_trajectory_categorical(
    runner: ModelRunner,
    config: ScoringConfig,
    traj: TrajectoryData,
) -> tuple[list[int | None], list[str]]:
    """Score a trajectory on all categorical judgments.

    Handles both single questions and bundled questions (lists).
    For bundled questions, each question is scored individually,
    and the bundling is handled during estimation (averaged into one structure).

    Returns:
        Tuple of (scores, raw_judgments) - flat lists of all individual scores
    """
    scores = []
    raw_judgments = []
    text_to_score = get_text_for_scoring(traj, config)

    for struct_idx, item in enumerate(config.categorical_judgements):
        if isinstance(item, list):
            # Bundled questions - score each individually
            log(f"    [c{struct_idx+1}] Bundled ({len(item)} questions):", gap=1 if struct_idx > 0 else 0)
            for q_idx, question in enumerate(item):
                score, response = judge_single_question(runner, config, text_to_score, question)
                scores.append(score)
                raw_judgments.append(response)
                score_str = str(score) if score is not None else "?"
                log(f"         • {question} -> {score_str}")
        else:
            # Single question
            score, response = judge_single_question(runner, config, text_to_score, item)
            scores.append(score)
            raw_judgments.append(response)
            score_str = str(score) if score is not None else "?"
            log(f"    [c{struct_idx+1}] {item} -> {score_str}", gap=1 if struct_idx > 0 else 0)

    return scores, raw_judgments


def judge_single_graded_question(
    runner: ModelRunner,
    config: ScoringConfig,
    text: str,
    question: str,
) -> tuple[float | None, str]:
    """Judge a single graded question for a trajectory. Returns (score, raw_response)."""
    prompt = config.build_graded_prompt(text, question)
    response = runner.generate(
        prompt=prompt,
        max_new_tokens=config.max_tokens,
        temperature=0.0,  # Always greedy
        prefilling=runner.skip_thinking_prefix,
    )
    score = config.parse_graded_judgment(response)
    return score, response


def score_trajectory_graded(
    runner: ModelRunner,
    config: ScoringConfig,
    traj: TrajectoryData,
) -> tuple[list[float | None], list[str]]:
    """Score a trajectory on all graded judgments (0-1 scale).

    Handles both single questions and bundled questions (lists).
    For bundled questions, each question is scored individually,
    and the bundling is handled during estimation (averaged into one structure).

    Returns:
        Tuple of (scores, raw_judgments) - flat lists of all individual scores
    """
    scores = []
    raw_judgments = []
    text_to_score = get_text_for_scoring(traj, config)

    for struct_idx, item in enumerate(config.graded_judgements):
        if isinstance(item, list):
            # Bundled questions - score each individually
            log(f"    [g{struct_idx+1}] Bundled ({len(item)} questions):", gap=1 if struct_idx > 0 else 0)
            for q_idx, question in enumerate(item):
                score, response = judge_single_graded_question(runner, config, text_to_score, question)
                scores.append(score)
                raw_judgments.append(response)
                score_str = f"{score:.2f}" if score is not None else "?"
                log(f"         • {question} -> {score_str}")
        else:
            # Single question
            score, response = judge_single_graded_question(runner, config, text_to_score, item)
            scores.append(score)
            raw_judgments.append(response)
            score_str = f"{score:.2f}" if score is not None else "?"
            log(f"    [g{struct_idx+1}] {item} -> {score_str}", gap=1 if struct_idx > 0 else 0)

    return scores, raw_judgments


def score_trajectory_similarity(
    embedder: EmbeddingRunner,
    config: ScoringConfig,
    traj: TrajectoryData,
) -> list[float]:
    """Score a trajectory on all similarity references.

    Handles both single references and bundled references (lists).
    For bundled references, each reference is scored individually,
    and the bundling is handled during estimation (averaged into one structure).

    Returns:
        Flat list of similarity scores (0-1) for all individual references.
    """
    text_to_score = get_text_for_scoring(traj, config)

    # Flatten references for embedding computation
    flat_refs = []
    for item in config.similarity_scoring:
        if isinstance(item, list):
            flat_refs.extend(item)
        else:
            flat_refs.append(item)

    # Get all similarities at once
    all_similarities = embedder.similarities(text=text_to_score, references=flat_refs)

    # Log with bundled structure
    scores = []
    sim_idx = 0
    for struct_idx, item in enumerate(config.similarity_scoring):
        if isinstance(item, list):
            log(f"    [s{struct_idx+1}] Bundled ({len(item)} references):", gap=1 if struct_idx > 0 else 0)
            for ref in item:
                score = all_similarities[sim_idx]
                scores.append(score)
                log(f"         • {preview(ref, 40)} -> {score:.3f}")
                sim_idx += 1
        else:
            score = all_similarities[sim_idx]
            scores.append(score)
            log(f"    [s{struct_idx+1}] {preview(item, 40)} -> {score:.3f}", gap=1 if struct_idx > 0 else 0)
            sim_idx += 1

    return scores


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
        branch_display = "trunk" if traj.branch_idx == 0 else f"branch_{traj.branch_idx}"
        log_section(f"Trajectory {i + 1}/{len(trajectories)} (branch: {branch_display})")

        # Print response (what's being scored with WholeContinuation)
        log(f'  Response: "{preview(oneline(traj.response), 120)}"', gap=1)

        # Categorical judgments
        scores: list[int | None] = []
        raw_judgments: list[str] = []
        if config.categorical_judgements and runner:
            log("  Categorical:", gap=1)
            scores, raw_judgments = score_trajectory_categorical(runner, config, traj)

        # Graded judgments
        graded_scores: list[float | None] = []
        graded_raw_judgments: list[str] = []
        if config.graded_judgements and runner:
            log("  Graded:", gap=1)
            graded_scores, graded_raw_judgments = score_trajectory_graded(runner, config, traj)

        # Similarity scoring
        similarity_scores: list[float] = []
        if config.similarity_scoring and embedder:
            log("  Similarity:", gap=1)
            similarity_scores = score_trajectory_similarity(embedder, config, traj)

        results.append(
            JudgmentResult.from_trajectory(
                traj, scores, raw_judgments, similarity_scores,
                graded_scores, graded_raw_judgments
            )
        )

    return results


def step_save_output(
    results: list[JudgmentResult],
    config: ScoringConfig,
    scoring_path: Path,
    gen_path: Path,
    branches: list[str],
    arm_texts: dict[str, str],
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
        arm_texts=arm_texts,
        prefix_logprobs=prefix_logprobs,
    )

    out_path = JudgmentOutput.compute_output_path(gen_path, scoring_path)
    output.save(out_path)
    log(f"  Saved judgments to {out_path}")

    # Save human-readable summary
    summary_path = JudgmentOutput.compute_summary_path(gen_path, scoring_path)
    output.save_summary(summary_path)
    log(f"  Saved summary to {summary_path}")

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
        log(f"  Categorical judgments ({len(config.categorical_judgements)}):")
        for i, item in enumerate(config.categorical_judgements):
            if isinstance(item, list):
                log(f"    [c{i+1}] BUNDLED ({len(item)} questions):")
                for q in item:
                    log(f"      • {q}")
            else:
                log(f"    [c{i+1}] {item}")
    if config.graded_judgements:
        log(f"  Graded judgments ({len(config.graded_judgements)}):")
        for i, item in enumerate(config.graded_judgements):
            if isinstance(item, list):
                log(f"    [g{i+1}] BUNDLED ({len(item)} questions):")
                for q in item:
                    log(f"      • {q}")
            else:
                log(f"    [g{i+1}] {item}")
    if config.similarity_scoring:
        log(f"  Similarity references ({len(config.similarity_scoring)}):")
        for i, ref in enumerate(config.similarity_scoring):
            log(f"    [s{i+1}] {ref}")
    log(f"  String selection: {config.string_selection.value}")

    runner, embedder = step_load_models(config)
    results = step_score_trajectories(runner, embedder, config, gen_data.trajectories)
    step_save_output(
        results, config, scoring_path, gen_path, gen_data.branches, gen_data.arm_texts, gen_data.prefix_logprobs
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
