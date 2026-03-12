"""Forking paths generation method.

This module implements trajectory generation using the forking paths
algorithm: identifying high-entropy positions and exploring alternative
tokens at those positions.

Algorithm:
    For each arm:
        1. Generate greedy path (temperature=0)
        2. Analyze all positions (entropy + top-K candidates)
        3. Find qualifying forks (high entropy, high probability alternates)
        4. Expand each fork point with sampled continuations
"""

from __future__ import annotations

import torch

from src.common.callback_types import LogFn
from src.inference import ModelRunner
from src.inference.generated_trajectory import GeneratedTrajectory

from ..generation_config import GenerationConfig
from ..generation_method_registry import register_method
from src.common.experiment_types import ArmGenerationResult

from .generation_method_utils import compute_arm_token_lengths
from .forking_paths_params import ForkingParams
from .forking_paths_types import (
    ForkPoint,
    PositionAnalysis,
    QualifyingFork,
    TopKCandidate,
)
from .logging.forking_paths_logging import (
    log_arm_tree,
    log_fork_expansion,
    log_greedy_path,
    log_position_analyses,
)


# ══════════════════════════════════════════════════════════════════════════════
# CORE ALGORITHMS
# ══════════════════════════════════════════════════════════════════════════════


TOP_K_DISPLAY = 5  # Default number of candidates to fetch


def analyze_all_positions(
    runner: ModelRunner,
    token_ids: list[int],
    prompt_len: int,
    max_alternates: int,
) -> list[PositionAnalysis]:
    """Single forward pass to get top-K candidates and entropy at each position.

    Args:
        runner: Model runner for inference
        token_ids: Full token sequence (prompt + generation)
        prompt_len: Length of prompt portion
        max_alternates: Maximum number of alternate tokens to consider

    Returns:
        Analysis for positions [prompt_len, prompt_len+1, ...] in the generated part
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
    """Find all (position, candidate) pairs that qualify for forking.

    Args:
        analyses: Position analyses from analyze_all_positions
        max_alternates: Maximum number of alternates per position
        min_prob: Minimum probability for alternate token
        min_entropy: Minimum entropy to consider forking

    Returns:
        List of qualifying forks meeting entropy and probability thresholds
    """
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


def generate_greedy_path(
    runner: ModelRunner,
    prompt_ids: list[int],
    max_new_tokens: int,
) -> GeneratedTrajectory:
    """Generate a greedy (temperature=0) trajectory.

    Args:
        runner: Model runner for generation
        prompt_ids: Tokenized prompt
        max_new_tokens: Maximum tokens to generate

    Returns:
        Greedy trajectory
    """
    return runner.generate_trajectory(
        token_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )


def expand_fork_point(
    runner: ModelRunner,
    analyses: list[PositionAnalysis],
    analysis: PositionAnalysis,
    candidate: TopKCandidate,
    prompt_ids: list[int],
    formatted_prompt: str,
    arm_prefill: str,
    max_new_tokens: int,
    temperature: float,
    samples_per_fork: int,
) -> ForkPoint | None:
    """Expand a single fork point by sampling continuations.

    Args:
        runner: Model runner for generation
        analyses: All position analyses (to build prefix)
        analysis: Analysis of the fork position
        candidate: Alternate token to fork to
        prompt_ids: Tokenized prompt
        max_new_tokens: Maximum tokens from prompt start
        temperature: Sampling temperature
        samples_per_fork: Number of continuations to sample

    Returns:
        ForkPoint with continuations, or None if no tokens remaining
    """
    # Build prefix for this fork: prompt + greedy tokens up to position + alternate
    greedy_prefix = [a.greedy_token_id for a in analyses[: analysis.position]]
    prefix = list(prompt_ids) + greedy_prefix + [candidate.token_id]
    remaining = max_new_tokens - analysis.position - 1

    if remaining <= 0:
        return None

    continuations: list[GeneratedTrajectory] = []
    for _ in range(samples_per_fork):
        traj = runner.generate_trajectory(prefix, remaining, temperature)
        traj.sanitize()

        # Set text fields (pipe, not parse)
        text = runner.decode_ids(traj.token_ids)
        traj.prefill_text = arm_prefill
        traj.generated_text = text[len(formatted_prompt):]

        # Free heavy data (full_logits) immediately to reduce peak memory
        traj.pop_heavy()
        continuations.append(traj)

    return ForkPoint(
        position=analysis.position,
        entropy=analysis.entropy,
        greedy_token_id=analysis.greedy_token_id,
        alternate=candidate,
        continuations=continuations,
    )


@register_method(ForkingParams)
def generate_forking(
    runner: ModelRunner,
    config: GenerationConfig,
    params: ForkingParams,
    log_fn: LogFn | None = None,
) -> ArmGenerationResult:
    """Generate trajectories using forking paths algorithm.

    Args:
        runner: Model runner for generation
        config: Generation config with prompt, arms, and parameters
        params: Forking-specific parameters
        log_fn: Optional logging callback

    Returns:
        ArmGenerationResult containing all trajectories and metadata
    """
    arms = config.get_arms(runner.skip_thinking_prefix)
    arm_token_lengths = compute_arm_token_lengths(runner, config, arms)

    base_formatted_prompt = runner.apply_chat_template(config.prompt)

    all_trajectories: list[GeneratedTrajectory] = []
    all_arm_indices: list[int] = []

    for arm_idx, arm in enumerate(arms):
        formatted_prompt = base_formatted_prompt + arm.prefill
        prompt_ids = runner.encode_ids(formatted_prompt, add_special_tokens=True)
        prompt_len = len(prompt_ids)

        # Use arm name directly
        arm_name = arm.name

        # Step 1: Generate greedy path
        greedy_traj = generate_greedy_path(runner, prompt_ids, config.max_new_tokens)

        # Set text fields (pipe, not parse)
        greedy_text = runner.decode_ids(greedy_traj.token_ids)
        greedy_traj.prefill_text = arm.prefill
        greedy_traj.generated_text = greedy_text[len(formatted_prompt):]

        # Free heavy data (full_logits) immediately to reduce peak memory
        greedy_traj.pop_heavy()

        if log_fn:
            log_greedy_path(greedy_traj, runner, arm_name, prompt_len)

        # Step 2: Analyze positions
        analyses = analyze_all_positions(
            runner, greedy_traj.token_ids, prompt_len, params.max_alternates
        )
        greedy_traj.entropies = [a.entropy for a in analyses]

        # Step 3: Find qualifying forks
        qualifying = find_qualifying_forks(
            analyses, params.max_alternates, params.min_prob, params.min_entropy
        )

        if log_fn:
            log_position_analyses(
                analyses,
                qualifying,
                runner,
                params,
                greedy_traj=greedy_traj,
                prompt_len=prompt_len,
            )

        # Step 4: Expand fork points
        arm_trajectories: list[GeneratedTrajectory] = [greedy_traj.sanitize()]
        expanded_fork_points: list[ForkPoint] = []

        for qf in qualifying:
            fork_point = expand_fork_point(
                runner=runner,
                analyses=analyses,
                analysis=qf.analysis,
                candidate=qf.candidate,
                prompt_ids=prompt_ids,
                formatted_prompt=formatted_prompt,
                arm_prefill=arm.prefill,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                samples_per_fork=params.samples_per_fork,
            )
            if fork_point:
                arm_trajectories.extend(fork_point.continuations)
                expanded_fork_points.append(fork_point)

        if log_fn:
            log_fork_expansion(expanded_fork_points, analyses, runner, prompt_len)
            log_arm_tree(
                arm_name,
                greedy_traj,
                expanded_fork_points,
                runner,
                prompt_len,
                config.max_new_tokens,
            )

        all_trajectories.extend(arm_trajectories)
        all_arm_indices.extend(arm_idx for _ in arm_trajectories)

    return ArmGenerationResult(
        trajectories=all_trajectories,
        arm_indices=all_arm_indices,
        arm_token_lengths=arm_token_lengths,
        arms=arms,
    )
