"""Entropy-seeking generation method.

This module implements trajectory generation by seeking high-entropy
positions and expanding the tree at those points.

Algorithm:
    For each arm:
        1. Initialize tree with N sampled trajectories
        2. Compute entropy at all positions via forward pass
        3. For K rounds:
           - Find (path, position) with highest unused entropy
           - Sample N new continuations from that fork point
           - Compute entropy for new trajectories
        4. Return all trajectories
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar

import torch

from src.common.callback_types import LogFn
from src.common.default_config import (
    ENTROPY_NUM_EXPANSION_ROUNDS,
    ENTROPY_SAMPLES_PER_EXPANSION,
)
from src.common.logging import log, log_step
from src.inference import ModelRunner
from src.inference.generated_trajectory import GeneratedTrajectory

from ..generation_config import GenerationConfig
from ..generation_method_registry import GenerationMethodParams, register_method
from src.common.experiment_types import ArmGenerationResult

from .generation_method_utils import compute_arm_token_lengths
from .entropy_seeking_types import ExpansionPoint, TreePath
from .logging.entropy_seeking_logging import (
    log_arm_header_entropy,
    log_expansion_round,
    log_expansion_summary,
    log_initialize_tree,
)


@dataclass
class EntropySeekingParams(GenerationMethodParams):
    """Parameters for entropy-seeking generation."""

    samples_per_expansion: int = field(
        default_factory=lambda: ENTROPY_SAMPLES_PER_EXPANSION
    )
    num_expansion_rounds: int = field(
        default_factory=lambda: ENTROPY_NUM_EXPANSION_ROUNDS
    )

    name: ClassVar[str] = "seeking-entropy"

    _cli_args: ClassVar[dict[str, str]] = {
        "samples_per_expansion": "--samples-per-expansion",
        "num_expansion_rounds": "--num-expansion-rounds",
    }


# ══════════════════════════════════════════════════════════════════════════════
# CORE ALGORITHMS
# ══════════════════════════════════════════════════════════════════════════════


def compute_entropies(
    runner: ModelRunner,
    token_ids: list[int],
    prompt_len: int,
) -> list[float]:
    """Compute next-token entropy at each generated position in one forward pass.

    Args:
        runner: Model runner for inference
        token_ids: Full token sequence (prompt + generation)
        prompt_len: Length of prompt portion

    Returns:
        List of entropy values for each generated position
    """
    ctx = (
        torch.inference_mode()
        if runner._backend.supports_inference_mode
        else torch.no_grad()
    )

    with ctx:
        input_ids = torch.tensor([token_ids], device=runner.device)
        logits = runner._backend.forward(input_ids)

    entropies = []
    for pos in range(prompt_len - 1, len(token_ids) - 1):
        probs = torch.softmax(logits[0, pos, :], dim=-1)
        log_probs = torch.log_softmax(logits[0, pos, :], dim=-1)
        entropy = -torch.sum(probs * log_probs).item()
        entropies.append(entropy)

    return entropies


def find_best_expansion_point(
    tree_paths: list[TreePath],
    prompt_len: int,
) -> ExpansionPoint:
    """Find the (path, position) with highest entropy across all paths.

    Args:
        tree_paths: All paths in the tree
        prompt_len: Length of prompt portion

    Returns:
        ExpansionPoint with best path, position, and entropy
    """
    best_path = None
    best_pos = None
    best_entropy = -math.inf

    for path in tree_paths:
        result = path.best_unused_position(prompt_len)
        if result.position is not None and result.entropy > best_entropy:
            best_entropy = result.entropy
            best_path = path
            best_pos = result.position

    return ExpansionPoint(path=best_path, position=best_pos, entropy=best_entropy)


def initialize_tree(
    runner: ModelRunner,
    prompt_ids: list[int],
    formatted_prompt: str,
    arm_prefill: str,
    max_new_tokens: int,
    temperature: float,
    samples_per_expansion: int,
    log_fn: LogFn | None = None,
) -> tuple[list[TreePath], int]:
    """Initialize tree with sampled trajectories and their entropies.

    Args:
        runner: Model runner for generation
        prompt_ids: Tokenized prompt
        formatted_prompt: Formatted prompt string
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        samples_per_expansion: Number of initial samples
        log_fn: Optional logging callback

    Returns:
        Tuple of (list of TreePaths, next_path_id)
    """
    prompt_len = len(prompt_ids)
    tree_paths = []

    for i in range(samples_per_expansion):
        traj = runner.generate_trajectory(prompt_ids, max_new_tokens, temperature)
        entropies = compute_entropies(runner, traj.token_ids, prompt_len)

        text = runner.decode_ids(traj.token_ids)
        continuation = text[len(formatted_prompt):]

        # Set text fields (pipe, not parse)
        traj.prefill_text = arm_prefill
        traj.generated_text = continuation

        path = TreePath(
            trajectory=traj.sanitize(),
            path_id=i,
            entropies=entropies,
            continuation=continuation,
            parent_id=None,
            branch_pos=None,
        )
        # Free heavy data (full_logits) immediately to reduce peak memory
        traj.pop_heavy()
        tree_paths.append(path)

    if log_fn:
        log_initialize_tree(tree_paths, runner, samples_per_expansion, max_new_tokens)

    return tree_paths, samples_per_expansion


def expand_tree(
    runner: ModelRunner,
    tree_paths: list[TreePath],
    next_path_id: int,
    prompt_ids: list[int],
    formatted_prompt: str,
    arm_prefill: str,
    max_new_tokens: int,
    temperature: float,
    samples_per_expansion: int,
    num_expansion_rounds: int,
    log_fn: LogFn | None = None,
) -> list[TreePath]:
    """Iteratively expand tree at highest-entropy positions.

    Args:
        runner: Model runner for generation
        tree_paths: Existing paths in tree
        next_path_id: Next available path ID
        prompt_ids: Tokenized prompt
        formatted_prompt: Formatted prompt string
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        samples_per_expansion: Number of samples per expansion
        num_expansion_rounds: Number of expansion rounds
        log_fn: Optional logging callback

    Returns:
        Updated list of tree paths including new expansions
    """
    prompt_len = len(prompt_ids)

    if log_fn:
        log_step(2, f"Expand tree ({num_expansion_rounds} rounds)")
        log("  Each round: find highest-entropy position, branch with new samples")

    for round_num in range(1, num_expansion_rounds + 1):
        # Find the single highest-entropy position across all paths
        expansion = find_best_expansion_point(tree_paths, prompt_len)

        if expansion.path is None or expansion.position is None:
            break

        expansion.path.mark_used(expansion.position)

        # Get token text at branch point
        token_id = expansion.path.token_ids[expansion.position]
        token_text = runner.decode_ids([token_id])

        # Sample new continuations from this branch point
        split_prefix = expansion.path.prefix(expansion.position)
        remaining = max_new_tokens - (expansion.position - prompt_len)

        if remaining <= 0:
            continue

        new_paths = []
        for _ in range(samples_per_expansion):
            traj = runner.generate_trajectory(split_prefix, remaining, temperature)
            entropies = compute_entropies(runner, traj.token_ids, prompt_len)

            text = runner.decode_ids(traj.token_ids)
            continuation = text[len(formatted_prompt):]

            # Set text fields (pipe, not parse)
            traj.prefill_text = arm_prefill
            traj.generated_text = continuation

            path = TreePath(
                trajectory=traj.sanitize(),
                path_id=next_path_id,
                entropies=entropies,
                continuation=continuation,
                parent_id=expansion.path.path_id,
                branch_pos=expansion.position - prompt_len,
            )
            # Free heavy data (full_logits) immediately to reduce peak memory
            traj.pop_heavy()
            tree_paths.append(path)
            new_paths.append(path)
            next_path_id += 1

        if log_fn:
            log_expansion_round(
                round_num=round_num,
                total_rounds=num_expansion_rounds,
                source_path=expansion.path,
                position=expansion.position,
                entropy=expansion.entropy,
                token=token_text,
                new_paths=new_paths,
                all_paths=tree_paths,
                runner=runner,
                prompt_len=prompt_len,
                max_tokens=max_new_tokens,
            )

    return tree_paths


def generate_entropy_seeking_for_arm(
    runner: ModelRunner,
    formatted_prompt: str,
    arm_prefill: str,
    max_new_tokens: int,
    temperature: float,
    samples_per_expansion: int,
    num_expansion_rounds: int,
    log_fn: LogFn | None = None,
) -> list[GeneratedTrajectory]:
    """Generate trajectories by seeking entropy for a single arm.

    Args:
        runner: Model runner for generation
        formatted_prompt: Formatted prompt string
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        samples_per_expansion: Number of samples per expansion
        num_expansion_rounds: Number of expansion rounds
        log_fn: Optional logging callback

    Returns:
        List of generated trajectories
    """
    prompt_ids = runner.encode_ids(formatted_prompt, add_special_tokens=True)

    tree_paths, next_path_id = initialize_tree(
        runner,
        prompt_ids,
        formatted_prompt,
        arm_prefill,
        max_new_tokens,
        temperature,
        samples_per_expansion,
        log_fn,
    )

    tree_paths = expand_tree(
        runner,
        tree_paths,
        next_path_id,
        prompt_ids,
        formatted_prompt,
        arm_prefill,
        max_new_tokens,
        temperature,
        samples_per_expansion,
        num_expansion_rounds,
        log_fn,
    )

    if log_fn:
        log_expansion_summary(tree_paths, samples_per_expansion, num_expansion_rounds)

    return [path.trajectory for path in tree_paths]


@register_method(EntropySeekingParams)
def generate_entropy_seeking(
    runner: ModelRunner,
    config: GenerationConfig,
    params: EntropySeekingParams,
    log_fn: LogFn | None = None,
) -> ArmGenerationResult:
    """Generate trajectories using entropy-seeking algorithm.

    Args:
        runner: Model runner for generation
        config: Generation config with prompt, arms, and parameters
        params: Entropy-seeking-specific parameters
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
        arm_name = arm.name

        if log_fn:
            log_arm_header_entropy(
                arm_name, config.trunk + (arm.prefill if arm_idx > 0 else "")
            )

        trajs = generate_entropy_seeking_for_arm(
            runner,
            formatted_prompt,
            arm.prefill,
            config.max_new_tokens,
            config.temperature,
            params.samples_per_expansion,
            params.num_expansion_rounds,
            log_fn,
        )

        all_trajectories.extend(trajs)
        all_arm_indices.extend(arm_idx for _ in trajs)

    return ArmGenerationResult(
        trajectories=all_trajectories,
        arm_indices=all_arm_indices,
        arm_token_lengths=arm_token_lengths,
        arms=arms,
    )
