#!/usr/bin/env python3
"""Generate trajectories using forking paths algorithm.

Probes local branches around a greedy path by exploring one-step
deviations at positions where alternative tokens have high probability
and the model shows sufficient uncertainty (entropy).

Usage:
    python scripts/generate_by_forking_paths.py trials/generation/<config>.json
    python scripts/generate_by_forking_paths.py trials/generation/<config>.json \
        --max-alternates-per-position 5 \
        --min-prob-for-alternate 0.05 \
        --min-entropy-to-fork 1.0 \
        --samples-per-fork 2

Algorithm:
    1. Generate greedy path (temperature=0)
    2. Single forward pass to get top-K candidates + entropy at each position
    3. For positions with entropy >= min_entropy:
       - For each alternate token with prob >= min_prob:
         - Sample N continuations from that fork point
    4. Collect greedy + all forked trajectories
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


from schemas import BranchGenerationResult, ForkingParams, GenerationConfig
from schemas.script_utils import (
    ArgSpec,
    build_and_save_tree,
    create_forking_tree_paths,
    format_horizontal_tree,
    load_model,
    log_branch_header,
    log_step,
    log_wrapped,
    parse_generation_args,
)

from src.common.log import log, log_section
from src.common.viz_utils import (
    compute_percentiles,
    compute_stats,
    format_histogram_vertical,
    format_sequence_plot,
    preview,
    print_lines,
)
from src.inference import ModelRunner
from src.inference.generated_trajectory import GeneratedTrajectory

# ══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class TopKCandidate:
    """A candidate token at a position."""

    token_id: int
    prob: float
    logprob: float


@dataclass
class PositionAnalysis:
    """Analysis of a single position in the greedy path."""

    position: int
    entropy: float
    greedy_token_id: int
    candidates: list[TopKCandidate]


@dataclass
class ForkPoint:
    """A position where we fork from the greedy path."""

    position: int
    entropy: float
    greedy_token_id: int
    alternate: TopKCandidate
    continuations: list[GeneratedTrajectory]


@dataclass
class QualifyingFork:
    """A position/candidate pair that qualifies for forking."""

    analysis: PositionAnalysis
    candidate: TopKCandidate


@dataclass
class PositionAnalysisResult:
    """Result of analyzing all positions in the greedy path."""

    analyses: list[PositionAnalysis]
    qualifying_forks: list[QualifyingFork]


@dataclass
class ForkExpansionResult:
    """Result of expanding fork points into trajectories."""

    trajectories: list[GeneratedTrajectory]
    fork_points: list[ForkPoint]


# ══════════════════════════════════════════════════════════════════════════════
# Core Algorithm
# ══════════════════════════════════════════════════════════════════════════════


TOP_K_DISPLAY = 5  # Always fetch at least this many for display


def analyze_all_positions(
    runner: ModelRunner,
    token_ids: list[int],
    prompt_len: int,
    max_alternates: int,
) -> list[PositionAnalysis]:
    """Single forward pass to get top-K candidates and entropy at each position.

    Returns analysis for positions [prompt_len, prompt_len+1, ...].
    """
    # Fetch enough candidates for both forking and display
    num_candidates = max(max_alternates, TOP_K_DISPLAY)

    with torch.inference_mode():
        input_ids = torch.tensor([token_ids], device=runner.device)
        logits = runner._backend.forward(input_ids)  # [1, seq_len, vocab]

    analyses: list[PositionAnalysis] = []
    generated_ids = token_ids[prompt_len:]

    # logits[i] predicts token at position i+1
    for t, greedy_token in enumerate(generated_ids):
        logit_idx = prompt_len - 1 + t
        probs = torch.softmax(logits[0, logit_idx, :], dim=-1)

        log_probs = torch.log_softmax(logits[0, logit_idx, :], dim=-1)
        entropy = -torch.sum(probs * log_probs).item()

        # Get top-K candidates
        topk_probs, topk_ids = torch.topk(probs, num_candidates)
        candidates = [
            TopKCandidate(
                token_id=topk_ids[i].item(),
                prob=topk_probs[i].item(),
                logprob=torch.log(topk_probs[i]).item(),
            )
            for i in range(num_candidates)
        ]

        analyses.append(
            PositionAnalysis(
                position=t,
                entropy=entropy,
                greedy_token_id=greedy_token,
                candidates=candidates,
            )
        )

    return analyses


def find_qualifying_forks(
    analyses: list[PositionAnalysis],
    max_alternates: int,
    min_prob: float,
    min_entropy: float,
) -> list[QualifyingFork]:
    """Find all (position, candidate) pairs that qualify for forking."""
    qualifying: list[QualifyingFork] = []

    for analysis in analyses:
        if analysis.entropy < min_entropy:
            continue

        # Only consider top max_alternates candidates for forking
        for candidate in analysis.candidates[:max_alternates]:
            if candidate.token_id == analysis.greedy_token_id:
                continue
            if candidate.prob < min_prob:
                continue

            qualifying.append(QualifyingFork(analysis=analysis, candidate=candidate))

    return qualifying


# ══════════════════════════════════════════════════════════════════════════════
# Logging Helpers
# ══════════════════════════════════════════════════════════════════════════════


def log_top_k_tokens(
    runner: ModelRunner,
    candidates: list[TopKCandidate],
    greedy_token_id: int,
    fork_token_id: int | None = None,
    k: int = 5,
) -> None:
    """Log the top-k tokens with probabilities and markers."""
    log(f"    Top {k} tokens:")
    for rank, cand in enumerate(candidates[:k]):
        token_str = runner.decode_ids([cand.token_id]).strip()
        markers = []
        if cand.token_id == greedy_token_id:
            markers.append("greedy")
        if fork_token_id and cand.token_id == fork_token_id:
            markers.append("fork")
        marker_str = f" ←{','.join(markers)}" if markers else ""
        log(f'      {rank + 1}. "{token_str}" p={cand.prob:.4f}{marker_str}')


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Steps
# ══════════════════════════════════════════════════════════════════════════════


def step_generate_greedy_path(
    runner: ModelRunner,
    prompt_ids: list[int],
    formatted_prompt: str,
    max_new_tokens: int,
) -> GeneratedTrajectory:
    """Step 1: Generate greedy path."""
    log_step(1, "Generate greedy path")

    greedy_traj = runner.generate_trajectory(
        token_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    prompt_len = len(prompt_ids)
    generated_len = len(greedy_traj.token_ids) - prompt_len

    text = runner.decode_ids(greedy_traj.token_ids)
    continuation = text[len(formatted_prompt) :]

    log(f"  Generated {generated_len} tokens:")
    log_wrapped(continuation, gap=1)

    return greedy_traj


def step_analyze_positions(
    runner: ModelRunner,
    greedy_traj: GeneratedTrajectory,
    prompt_len: int,
    formatted_prompt: str,
    max_alternates: int,
    min_prob: float,
    min_entropy: float,
) -> PositionAnalysisResult:
    """Step 2: Analyze positions and find qualifying forks."""
    log_step(2, "Analyze positions", f"top-{max_alternates} candidates + entropy")

    # Show prompt vs response separation
    full_text = runner.decode_ids(greedy_traj.token_ids)
    response_text = full_text[len(formatted_prompt) :]

    log(f"  Prompt ({prompt_len} tokens):")
    log_wrapped(preview(formatted_prompt, 200), indent="    ")
    log(f"  Response ({len(greedy_traj.token_ids) - prompt_len} tokens):", gap=1)
    log_wrapped(preview(response_text, 200), indent="    ")

    analyses = analyze_all_positions(
        runner, greedy_traj.token_ids, prompt_len, max_alternates
    )
    entropies = [a.entropy for a in analyses]

    # Store entropies in trajectory for JSON persistence
    greedy_traj.entropies = entropies

    if entropies:
        _log_entropy_stats(entropies, gap=1)
        _log_entropy_visualizations(entropies)

    # Find qualifying forks
    qualifying = find_qualifying_forks(analyses, max_alternates, min_prob, min_entropy)
    _log_filtering_results(analyses, qualifying, min_prob, min_entropy)

    return PositionAnalysisResult(analyses=analyses, qualifying_forks=qualifying)


def _log_entropy_stats(entropies: list[float], gap: int = 0) -> None:
    """Log entropy statistics for response tokens."""
    stats = compute_stats(entropies)
    percentiles = compute_percentiles(entropies, [25, 50, 75])

    log(f"  Entropy statistics ({len(entropies)} response positions):", gap=gap)
    log(f"    min={stats['min']:.2f}  max={stats['max']:.2f}")
    log(f"    mean={stats['mean']:.2f}  std={stats['std']:.2f}")
    log(
        f"    p25={percentiles[25]:.2f}  p50={percentiles[50]:.2f}  p75={percentiles[75]:.2f}"
    )


def _log_entropy_visualizations(entropies: list[float]) -> None:
    """Log entropy histogram and sequence plot."""
    log("  Entropy distribution (response tokens):", gap=1)
    print_lines(format_histogram_vertical(entropies, num_bins=8), log)

    log("  Entropy over response positions:", gap=1)
    print_lines(
        format_sequence_plot(entropies, max_width=70, height=8, label="H"),
        log,
    )


def _log_filtering_results(
    analyses: list[PositionAnalysis],
    qualifying: list[QualifyingFork],
    min_prob: float,
    min_entropy: float,
) -> None:
    """Log fork filtering results."""
    high_entropy_count = sum(1 for a in analyses if a.entropy >= min_entropy)
    log("  Filtering:", gap=1)
    log(f"    Positions with H >= {min_entropy}: {high_entropy_count}/{len(analyses)}")
    log(f"    Alternates with p >= {min_prob}: {len(qualifying)}")


def step_expand_forks(
    runner: ModelRunner,
    analysis_result: PositionAnalysisResult,
    prompt_ids: list[int],
    formatted_prompt: str,
    max_new_tokens: int,
    temperature: float,
    samples_per_fork: int,
    greedy_traj: GeneratedTrajectory,
) -> ForkExpansionResult:
    """Step 3: Expand qualifying fork points."""
    qualifying = analysis_result.qualifying_forks
    log_step(
        3, "Expand fork points", f"{len(qualifying)} forks × {samples_per_fork} samples"
    )

    if not qualifying:
        log("  No qualifying fork points.")
        return ForkExpansionResult(
            trajectories=[greedy_traj.sanitize()],
            fork_points=[],
        )

    all_trajs: list[GeneratedTrajectory] = [greedy_traj.sanitize()]
    fork_points: list[ForkPoint] = []

    for fork_idx, qf in enumerate(qualifying):
        fork_point = _expand_single_fork(
            runner=runner,
            analyses=analysis_result.analyses,
            analysis=qf.analysis,
            candidate=qf.candidate,
            fork_idx=fork_idx,
            total_forks=len(qualifying),
            prompt_ids=prompt_ids,
            formatted_prompt=formatted_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            samples_per_fork=samples_per_fork,
        )

        if fork_point:
            fork_points.append(fork_point)
            all_trajs.extend(fork_point.continuations)

    # Show final tree structure
    if fork_points:
        greedy_text = runner.decode_ids(greedy_traj.token_ids)
        greedy_continuation = greedy_text[len(formatted_prompt) :]

        # Build fork data for tree viz
        prompt_len = len(prompt_ids)
        fork_data = []
        for fp in fork_points:
            continuations = [
                (c.token_ids, runner.decode_ids(c.token_ids)[len(formatted_prompt) :])
                for c in fp.continuations
            ]
            fork_data.append((fp.position, continuations))

        tree_paths = create_forking_tree_paths(
            greedy_traj.token_ids, greedy_continuation, fork_data
        )

        log("\n  Tree:")
        for line in format_horizontal_tree(tree_paths, prompt_len, max_new_tokens):
            log(f"  {line}")

    return ForkExpansionResult(trajectories=all_trajs, fork_points=fork_points)


def _expand_single_fork(
    runner: ModelRunner,
    analyses: list[PositionAnalysis],
    analysis: PositionAnalysis,
    candidate: TopKCandidate,
    fork_idx: int,
    total_forks: int,
    prompt_ids: list[int],
    formatted_prompt: str,
    max_new_tokens: int,
    temperature: float,
    samples_per_fork: int,
) -> ForkPoint | None:
    """Expand a single fork point and return the ForkPoint."""
    # Build prefix for this fork
    greedy_prefix = [a.greedy_token_id for a in analyses[: analysis.position]]
    prefix = list(prompt_ids) + greedy_prefix + [candidate.token_id]
    remaining = max_new_tokens - analysis.position - 1

    if remaining <= 0:
        return None

    # Log fork header
    greedy_token = runner.decode_ids([analysis.greedy_token_id]).strip()
    fork_token = runner.decode_ids([candidate.token_id]).strip()

    log(
        f"  Fork {fork_idx + 1}/{total_forks} @ pos {analysis.position} (H={analysis.entropy:.2f}):",
        gap=1,
    )
    log(f'    "{greedy_token}" → "{fork_token}" (p={candidate.prob:.3f})')

    # Show top tokens
    log_top_k_tokens(
        runner, analysis.candidates, analysis.greedy_token_id, candidate.token_id
    )

    continuations: list[GeneratedTrajectory] = []
    for sample_idx in range(samples_per_fork):
        traj = runner.generate_trajectory(prefix, remaining, temperature)
        traj.sanitize()
        continuations.append(traj)

        text = runner.decode_ids(traj.token_ids)
        cont = text[len(formatted_prompt) :]
        log(f'    Sample {sample_idx + 1}: "{preview(cont, 50)}"')

    return ForkPoint(
        position=analysis.position,
        entropy=analysis.entropy,
        greedy_token_id=analysis.greedy_token_id,
        alternate=candidate,
        continuations=continuations,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════


def generate_forking_paths_for_branch(
    runner: ModelRunner,
    formatted_prompt: str,
    max_new_tokens: int,
    temperature: float,
    max_alternates: int,
    min_prob: float,
    min_entropy: float,
    samples_per_fork: int,
) -> ForkExpansionResult:
    """Generate forking paths for a single branch.

    Pipeline:
        1. Generate greedy path
        2. Analyze positions (entropy + top-k candidates)
        3. Expand qualifying fork points
    """
    prompt_ids = runner.encode_ids(formatted_prompt, add_special_tokens=True)
    prompt_len = len(prompt_ids)

    # Step 1: Generate greedy path
    greedy_traj = step_generate_greedy_path(
        runner, prompt_ids, formatted_prompt, max_new_tokens
    )

    # Step 2: Analyze positions and find qualifying forks
    analysis_result = step_analyze_positions(
        runner,
        greedy_traj,
        prompt_len,
        formatted_prompt,
        max_alternates,
        min_prob,
        min_entropy,
    )

    # Step 3: Expand fork points
    return step_expand_forks(
        runner,
        analysis_result,
        prompt_ids,
        formatted_prompt,
        max_new_tokens,
        temperature,
        samples_per_fork,
        greedy_traj,
    )


def generate_for_all_branches(
    runner: ModelRunner,
    config: GenerationConfig,
    params: ForkingParams,
) -> BranchGenerationResult:
    """Generate forking paths for all branches."""
    branches = config.get_branches(runner.skip_thinking_prefix)
    prompt_length = config.compute_prompt_length(runner)
    trunk_length = config.compute_trunk_length(runner)

    all_trajectories: list[GeneratedTrajectory] = []
    all_group_indices: list[int] = []

    for branch in branches:
        formatted_prompt = runner.apply_chat_template(config.prompt) + branch.prefill
        log_branch_header(branch.name, formatted_prompt)

        result = generate_forking_paths_for_branch(
            runner,
            formatted_prompt,
            config.max_new_tokens,
            config.temperature,
            params.max_alternates,
            params.min_prob,
            params.min_entropy,
            params.samples_per_fork,
        )

        log(
            f"\n  Summary: {len(result.trajectories)} trajectories from {len(result.fork_points)} fork points"
        )

        all_trajectories.extend(result.trajectories)
        all_group_indices.extend(branch.group_idx for _ in result.trajectories)

    return BranchGenerationResult(
        trajectories=all_trajectories,
        group_indices=all_group_indices,
        trunk_length=trunk_length,
        prompt_length=prompt_length,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def generate_by_forking_paths(
    config: GenerationConfig,
    config_path: Path,
    params: ForkingParams,
) -> None:
    """Run forking paths generation."""
    runner = load_model(config)

    log_section("Forking Paths Algorithm")
    params.print()

    result = generate_for_all_branches(runner, config, params)

    build_and_save_tree(
        result=result,
        config=config,
        config_path=config_path,
        runner=runner,
        method="forking",
    )


def main() -> None:
    parsed = parse_generation_args(
        description="Generate trajectories using forking paths algorithm",
        examples=[
            "config.json",
            "config.json --max-alternates-per-position 5 --min-prob-for-alternate 0.1",
            "config.json --min-entropy-to-fork 1.5 --samples-per-fork 3",
        ],
        extra_args=[
            ArgSpec(
                "max-alternates-per-position",
                int,
                "K",
                "Max alternate tokens per position",
            ),
            ArgSpec(
                "min-prob-for-alternate",
                float,
                "P",
                "Minimum probability for alternate token",
            ),
            ArgSpec(
                "min-entropy-to-fork", float, "H", "Minimum entropy to consider forking"
            ),
            ArgSpec("samples-per-fork", int, "N", "Continuations per fork point"),
        ],
    )

    generate_by_forking_paths(
        config=parsed.config,
        config_path=parsed.config_path,
        params=parsed.config.forking_params,
    )


if __name__ == "__main__":
    main()
