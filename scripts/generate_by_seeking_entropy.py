#!/usr/bin/env python3
"""Generate trajectories by seeking high-entropy positions.

Iteratively expands a tree at positions where the model is most uncertain
(highest next-token entropy).

Usage:
    python scripts/generate_by_seeking_entropy.py trials/generation/<config>.json
    python scripts/generate_by_seeking_entropy.py trials/generation/<config>.json \
        --samples-per-expansion 3 \
        --num-expansion-rounds 4

Algorithm:
    1. Initialize tree with N sampled trajectories
    2. Compute entropy at all positions via single forward pass per trajectory
    3. For K rounds:
       - Find (path, position) with highest entropy among unused positions
       - Sample N new continuations from that fork point
       - Compute entropy for new trajectories
    4. Return all trajectories
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import BranchGenerationResult, EntropySeekingParams, GenerationConfig
from schemas.script_utils import (
    ArgSpec,
    build_and_save_tree,
    format_horizontal_tree,
    format_tree_simple,
    load_model,
    log_branch_header,
    log_prompt_header,
    log_step,
    oneline,
    parse_generation_args,
)

from src.common.log import log, log_section
from src.common.viz_utils import preview
from src.inference import ModelRunner
from src.inference.generated_trajectory import GeneratedTrajectory

# ══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class BestPosition:
    """Result of finding the best unused position in a path."""

    position: int | None
    entropy: float


@dataclass
class ExpansionPoint:
    """The best expansion point across all paths."""

    path: TreePath | None
    position: int | None
    entropy: float


@dataclass
class TreePath:
    """A path in the entropy-seeking tree with precomputed entropies."""

    trajectory: GeneratedTrajectory
    path_id: int
    entropies: list[float]
    continuation: str = ""  # For display
    parent_id: int | None = None  # Which path we branched from
    branch_pos: int | None = None  # Position where we branched
    used_positions: set[int] = field(default_factory=set)

    @property
    def token_ids(self) -> list[int]:
        return self.trajectory.token_ids

    @property
    def max_entropy(self) -> float:
        return max(self.entropies) if self.entropies else 0.0

    def prefix(self, position: int) -> list[int]:
        """Get tokens up to and including position."""
        return self.token_ids[: position + 1]

    def mark_used(self, pos: int) -> None:
        """Mark a position as used for splitting."""
        self.used_positions.add(pos)

    def best_unused_position(self, prompt_len: int) -> BestPosition:
        """Find highest-entropy unused position."""
        best_pos = None
        best_entropy = -math.inf

        for i, entropy in enumerate(self.entropies):
            pos = prompt_len + i
            if pos not in self.used_positions and entropy > best_entropy:
                best_entropy = entropy
                best_pos = pos

        return BestPosition(position=best_pos, entropy=best_entropy)


# ══════════════════════════════════════════════════════════════════════════════
# Core Algorithm
# ══════════════════════════════════════════════════════════════════════════════


def compute_entropies(
    runner: ModelRunner,
    token_ids: list[int],
    prompt_len: int,
) -> list[float]:
    """Compute next-token entropy at each generated position in one forward pass."""
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
    """Find the (path, position) with highest entropy across all paths."""
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


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Steps
# ══════════════════════════════════════════════════════════════════════════════


def step_initialize(
    runner: ModelRunner,
    prompt_ids: list[int],
    formatted_prompt: str,
    max_new_tokens: int,
    temperature: float,
    samples_per_expansion: int,
) -> tuple[list[TreePath], int]:
    """Step 1: Sample initial trajectories and compute their entropies."""
    log_step(1, "Initialize tree", f"{samples_per_expansion} random samples")
    log("  Sampling trajectories from prompt, computing entropy at each token...")

    prompt_len = len(prompt_ids)
    tree_paths = []

    for i in range(samples_per_expansion):
        traj = runner.generate_trajectory(prompt_ids, max_new_tokens, temperature)
        entropies = compute_entropies(runner, traj.token_ids, prompt_len)

        text = runner.decode_ids(traj.token_ids)
        continuation = text[len(formatted_prompt):]

        path = TreePath(
            trajectory=traj.sanitize(),
            path_id=i,
            entropies=entropies,
            continuation=continuation,
            parent_id=None,
            branch_pos=None,
        )
        tree_paths.append(path)

    # Show initial tree
    log("\n  Tree:")
    for line in format_horizontal_tree(tree_paths, prompt_len, max_new_tokens):
        log(f"  {line}")

    # Show path details
    log("\n  Paths:")
    for line in format_tree_simple(tree_paths):
        log(f"    {line}")

    return tree_paths, samples_per_expansion


def step_expand(
    runner: ModelRunner,
    tree_paths: list[TreePath],
    next_path_id: int,
    prompt_ids: list[int],
    formatted_prompt: str,
    max_new_tokens: int,
    temperature: float,
    samples_per_expansion: int,
    num_expansion_rounds: int,
) -> list[TreePath]:
    """Step 2: Iteratively expand at highest-entropy positions."""
    log_step(2, "Expand tree", f"{num_expansion_rounds} rounds")
    log("  Each round: find highest-entropy position, branch with new samples")

    prompt_len = len(prompt_ids)

    for round_num in range(1, num_expansion_rounds + 1):
        # Find the single highest-entropy position across all paths
        expansion = find_best_expansion_point(tree_paths, prompt_len)

        if expansion.path is None or expansion.position is None:
            log(f"\n  Round {round_num}/{num_expansion_rounds}: no unexplored positions")
            break

        rel_pos = expansion.position - prompt_len
        split_token = runner.decode_ids(
            [expansion.path.token_ids[expansion.position]]
        ).strip()

        log(f"\n  Round {round_num}/{num_expansion_rounds}: branch from [{expansion.path.path_id}] @ token {rel_pos} \"{split_token}\" (H={expansion.entropy:.2f})")

        expansion.path.mark_used(expansion.position)

        # Sample new continuations from this branch point
        split_prefix = expansion.path.prefix(expansion.position)
        remaining = max_new_tokens - (expansion.position - prompt_len)

        if remaining <= 0:
            log("    Skipping: no tokens remaining")
            continue

        new_paths = []
        for _ in range(samples_per_expansion):
            traj = runner.generate_trajectory(split_prefix, remaining, temperature)
            entropies = compute_entropies(runner, traj.token_ids, prompt_len)

            text = runner.decode_ids(traj.token_ids)
            continuation = text[len(formatted_prompt):]

            path = TreePath(
                trajectory=traj.sanitize(),
                path_id=next_path_id,
                entropies=entropies,
                continuation=continuation,
                parent_id=expansion.path.path_id,
                branch_pos=rel_pos,
            )
            tree_paths.append(path)
            new_paths.append(path)
            next_path_id += 1

        # Show new paths
        log("  New paths:")
        for line in format_tree_simple(new_paths):
            log(f"    {line}")

        # Show updated tree
        log("\n  Tree:")
        for line in format_horizontal_tree(tree_paths, prompt_len, max_new_tokens):
            log(f"  {line}")

    return tree_paths


def generate_entropy_seeking_for_branch(
    runner: ModelRunner,
    formatted_prompt: str,
    max_new_tokens: int,
    temperature: float,
    samples_per_expansion: int,
    num_expansion_rounds: int,
) -> list[GeneratedTrajectory]:
    """Generate trajectories by seeking entropy for a single branch."""
    prompt_ids = runner.encode_ids(formatted_prompt, add_special_tokens=True)

    tree_paths, next_path_id = step_initialize(
        runner,
        prompt_ids,
        formatted_prompt,
        max_new_tokens,
        temperature,
        samples_per_expansion,
    )

    tree_paths = step_expand(
        runner,
        tree_paths,
        next_path_id,
        prompt_ids,
        formatted_prompt,
        max_new_tokens,
        temperature,
        samples_per_expansion,
        num_expansion_rounds,
    )

    return [path.trajectory for path in tree_paths]


def generate_for_all_branches(
    runner: ModelRunner,
    config: GenerationConfig,
    params: EntropySeekingParams,
) -> BranchGenerationResult:
    """Generate trajectories for all branches."""
    branches = config.get_branches(runner.skip_thinking_prefix)
    prompt_length = config.compute_prompt_length(runner)
    trunk_length = config.compute_trunk_length(runner)

    # Show prompt structure once at the start
    log_prompt_header(config.prompt, config.trunk, config.branches)

    all_trajectories: list[GeneratedTrajectory] = []
    all_group_indices: list[int] = []

    for branch in branches:
        formatted_prompt = runner.apply_chat_template(config.prompt) + branch.prefill
        # Determine the branch-specific continuation
        if branch.name == "trunk":
            branch_continuation = config.trunk
        else:
            branch_continuation = config.trunk + branch.name
        log_branch_header(branch.name, branch_continuation)

        trajs = generate_entropy_seeking_for_branch(
            runner,
            formatted_prompt,
            config.max_new_tokens,
            config.temperature,
            params.samples_per_expansion,
            params.num_expansion_rounds,
        )

        total_initial = params.samples_per_expansion
        total_expanded = len(trajs) - total_initial
        log(f"\n  Total: {len(trajs)} paths ({total_initial} initial + {total_expanded} from expansion)")

        all_trajectories.extend(trajs)
        all_group_indices.extend(branch.group_idx for _ in trajs)

    return BranchGenerationResult(
        trajectories=all_trajectories,
        group_indices=all_group_indices,
        trunk_length=trunk_length,
        prompt_length=prompt_length,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════


def generate_by_seeking_entropy(
    config: GenerationConfig,
    config_path: Path,
    params: EntropySeekingParams,
) -> None:
    """Run entropy-seeking generation."""
    runner = load_model(config)

    log_section("Entropy-Seeking Algorithm")
    params.print()

    result = generate_for_all_branches(runner, config, params)

    build_and_save_tree(
        result=result,
        config=config,
        config_path=config_path,
        runner=runner,
        method="entropy",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parsed = parse_generation_args(
        description="Generate trajectories by seeking high-entropy positions",
        examples=[
            "config.json",
            "config.json --samples-per-expansion 3 --num-expansion-rounds 5",
        ],
        extra_args=[
            ArgSpec("samples-per-expansion", int, "N", "Trajectories per expansion"),
            ArgSpec("num-expansion-rounds", int, "K", "Number of expansion rounds"),
        ],
    )

    generate_by_seeking_entropy(
        config=parsed.config,
        config_path=parsed.config_path,
        params=parsed.config.entropy_params,
    )


if __name__ == "__main__":
    main()
