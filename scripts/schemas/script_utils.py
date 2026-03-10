"""Shared utilities for pipeline scripts.

This module provides common functions used across generation, scoring,
estimation, and experiment scripts to reduce code duplication.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from src.common.analysis import analyze_token_tree
from src.common.device_utils import clear_gpu_memory
from src.common.log import log, log_params, log_section
from src.common.seed import set_seed
from src.common.token_tree import TokenTree
from src.common.viz_utils import preview
from src.inference import ModelRunner

from .generation import (
    BranchGenerationResult,
    GenerationConfig,
    GenerationOutput,
    OutputPaths,
)

# Re-export logging utilities from log_utils for backward compatibility
from .log_utils import (
    HEADER_WIDTH,
    STAGE_GAP,
    fmt_core,
    fmt_prob,
    log_banner,
    log_box,
    log_divider,
    log_header,
    log_major,
    log_section_title,
    log_stage,
    log_step,
    log_sub_banner,
    log_table_header,
    log_wrapped,
    oneline,
)


def log_output_paths(paths: OutputPaths, gap: int = 0) -> None:
    """Log output file paths."""
    log("Output files:", gap=gap)
    log(f"  -> {paths.generation}")
    log(f"  -> {paths.judgment}")
    log(f"  -> {paths.estimation}")


def log_experiment_start(
    title: str,
    paths: OutputPaths,
    **params: Any,
) -> None:
    """Log experiment header with params and output paths."""
    log("═" * HEADER_WIDTH)
    log(title)
    log("═" * HEADER_WIDTH)
    if params:
        log_params(**params)
    log_output_paths(paths, gap=1)
    log("═" * HEADER_WIDTH)


# ══════════════════════════════════════════════════════════════════════════════
# Argument Parsing
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ArgSpec:
    """Specification for an extra command-line argument."""

    name: str
    type: type
    metavar: str
    help: str


@dataclass
class ParsedArgs:
    """Result of parsing generation script arguments."""

    config: GenerationConfig
    config_path: Path


def parse_generation_args(
    description: str,
    examples: list[str],
    extra_args: list[ArgSpec] | None = None,
) -> ParsedArgs:
    """Parse command-line arguments for generation scripts.

    Args:
        description: Script description
        examples: List of example usage lines
        extra_args: Additional arguments specific to this script

    Returns:
        ParsedArgs with config (CLI overrides already applied) and config_path
    """
    epilog = "Examples:\n" + "\n".join(f"  %(prog)s {ex}" for ex in examples)

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    parser.add_argument("config", help="Path to generation config JSON")

    if extra_args:
        for arg in extra_args:
            parser.add_argument(
                f"--{arg.name}",
                type=arg.type,
                metavar=arg.metavar,
                help=arg.help,
            )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    config = GenerationConfig.load(config_path)

    # Apply CLI overrides to config
    if extra_args:
        overrides = {
            arg.name.replace("-", "_"): getattr(args, arg.name.replace("-", "_"))
            for arg in extra_args
        }
        config.apply_cli_overrides(overrides)

    set_seed(config.seed)

    return ParsedArgs(config=config, config_path=config_path)


# ══════════════════════════════════════════════════════════════════════════════
# Model Loading
# ══════════════════════════════════════════════════════════════════════════════


def load_model(config: GenerationConfig) -> ModelRunner:
    """Load and validate the model from config."""
    if not config.model:
        raise ValueError("No model specified in generation config")

    log(f"Loading model: {config.model}")
    runner = ModelRunner(config.model)

    # Get model type from runner (it detects chat models)
    model_type = "CHAT/INSTRUCT" if runner._is_chat_model else "BASE"

    box_content = f"MODEL TYPE: {model_type}"
    log(f"\n  ╔{'═' * (len(box_content) + 4)}╗")
    log(f"  ║  {box_content}  ║")
    log(f"  ╚{'═' * (len(box_content) + 4)}╝\n")
    return runner


def log_prompt_header(prompt: str, trunk: str, branches: list[str]) -> None:
    """Log the prompt and structure at the start of generation.

    Args:
        prompt: The user prompt
        trunk: The trunk/shared prefix text
        branches: List of branch continuation texts (e.g., [" boy", " cat"])
    """
    log_section("Generation Setup")
    log("  Prompt:")
    for line in prompt.split("\n"):
        log(f"    {line}")
    log(f'\n  Trunk (shared prefix): "{trunk}"')
    if branches:
        log(f"  Branches ({len(branches)}): {branches}")
        log(f"  Total groups: {len(branches) + 1} (trunk + {len(branches)} branches)")


def log_branch_header(branch_name: str, continuation: str) -> None:
    """Log section header for a branch.

    Args:
        branch_name: Name of the branch ("trunk" or branch name)
        continuation: The branch-specific continuation text
    """
    if branch_name == "trunk":
        label = "Trunk"
        log_section(label)
        log(f'  Continuation: "{continuation}"')
    else:
        label = f"Branch: {branch_name}"
        log_section(label)
        log(f'  Continuation: "{continuation}"')


# ══════════════════════════════════════════════════════════════════════════════
# Tree Building
# ══════════════════════════════════════════════════════════════════════════════


def _log_trajectories(result: BranchGenerationResult, runner: ModelRunner) -> None:
    """Log trajectory texts and conditional probabilities."""
    prompt_len = result.prompt_length
    trunk_len = result.trunk_length

    # Table 1: Trajectory continuations (not full text including prompt)
    log(f"  Trajectories ({len(result.trajectories)} total):")
    log(f"  {'#':>3}  {'branch':<10} continuation")
    log("  " + "─" * 70)

    for i, traj in enumerate(result.trajectories):
        group_idx = result.group_indices[i]
        display = "trunk" if group_idx == 0 else f"branch_{group_idx}"
        # Show only continuation (tokens after trunk), not full text
        continuation_ids = traj.token_ids[trunk_len:]
        continuation_text = runner.decode_ids(continuation_ids)
        # Show more text before cutting off (80 chars instead of 50)
        log(f"  {i:>3}  {display:<10} {preview(continuation_text, 80)}")
    log("")

    # Table 2: Conditional probabilities
    log(
        f"  Conditional probabilities (prompt_len={prompt_len}, trunk_len={trunk_len}):"
    )
    log(
        f"  {'#':>3}  {'branch':<10} {'p(t|prompt)':>11}  "
        f"{'p(t|trunk)':>11}  {'p(t|branch)':>11}  {'Finished?':>9}"
    )
    log("  " + "─" * 67)

    # Get EOS token from runner
    eos_token = runner.eos_token
    eos_token_id = runner.eos_token_id

    for i, traj in enumerate(result.trajectories):
        group_idx = result.group_indices[i]
        p_prompt = traj.get_conditional_prob(prompt_len, traj.length) or 0.0

        if group_idx == 0:
            p_trunk = traj.get_conditional_prob(trunk_len, traj.length) or 0.0
            p_branch = p_trunk
        else:
            p_trunk = traj.get_conditional_prob(trunk_len - 1, traj.length) or 0.0
            p_branch = traj.get_conditional_prob(trunk_len, traj.length) or 0.0

        # Check if trajectory has EOS token (by token ID or text)
        continuation_ids = traj.token_ids[trunk_len:]
        is_finished = (eos_token_id is not None and eos_token_id in continuation_ids)
        if not is_finished and eos_token:
            continuation_text = runner.decode_ids(continuation_ids)
            is_finished = eos_token in continuation_text
        finished_str = "YES" if is_finished else "NO"

        display = "trunk" if group_idx == 0 else f"branch_{group_idx}"
        log(
            f"  {i:>3}  {display:<10} {fmt_prob(p_prompt, 11)}  "
            f"{fmt_prob(p_trunk, 11)}  {fmt_prob(p_branch, 11)}  {finished_str:>9}"
        )
    log("")


def build_and_save_tree(
    result: BranchGenerationResult,
    config: GenerationConfig,
    config_path: Path,
    runner: ModelRunner,
    method: str,
) -> Path:
    """Build token tree from generation result and save to output file.

    Args:
        result: Generation result with trajectories and groups
        config: Generation configuration
        config_path: Path to config file (for output naming)
        runner: Model runner (for text decoding)
        method: Method name for output metadata and filename (e.g., "sampling")

    Returns:
        Path to saved output file
    """
    log_section("Building Tree")
    _log_trajectories(result, runner)

    tree = TokenTree.from_trajectories(
        trajs=result.trajectories,
        groups_per_traj=[(idx,) for idx in result.group_indices],
        fork_arms=[(arm.left, arm.right) for arm in config.fork_arms],
        trunk=list(range(result.trunk_length)),
        prompt_length=result.prompt_length,
    )
    tree.decode_texts(runner)

    analyze_token_tree(tree)

    tree.pop_heavy()
    clear_gpu_memory()

    output = GenerationOutput.from_tree(config, config.model, tree, method=method, eos_token=runner.eos_token)
    out_path = GenerationOutput.compute_output_path(config_path, method=method)
    output.save(out_path)

    log(f"Saved {len(result.trajectories)} trajectories to {out_path}", gap=1)

    # Save human-readable summary
    summary_path = GenerationOutput.compute_summary_path(config_path, method=method)
    output.save_summary(summary_path)
    log(f"Saved summary to {summary_path}")

    output.summarize()

    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# Tree Visualization
# ══════════════════════════════════════════════════════════════════════════════


class TreePathLike(Protocol):
    """Protocol for objects that can be visualized as tree paths."""

    path_id: int
    parent_id: int | None
    branch_pos: int | None
    continuation: str

    @property
    def token_ids(self) -> list[int]: ...


def format_horizontal_tree(
    tree_paths: list[TreePathLike],
    prompt_len: int,
    max_new_tokens: int,
    width: int = 50,
) -> list[str]:
    """Format tree as horizontal timeline showing token positions."""
    if not tree_paths:
        return []

    scale = width / max(max_new_tokens, 1)

    def pos_to_col(rel_token_pos: int) -> int:
        return int(rel_token_pos * scale)

    children: dict[int | None, list[TreePathLike]] = {}
    for path in tree_paths:
        parent = path.parent_id
        if parent not in children:
            children[parent] = []
        children[parent].append(path)

    lines: list[str] = []

    prefix = "    "
    ruler = prefix
    step = max(5, max_new_tokens // 6)
    for i in range(0, max_new_tokens + 1, step):
        col = pos_to_col(i)
        label = str(i)
        ruler = ruler.ljust(len(prefix) + col) + label
    lines.append(ruler)

    def get_path_length(path: TreePathLike) -> int:
        return len(path.token_ids) - prompt_len

    def render_path(
        path: TreePathLike,
        row_prefix: str,
        is_last_sibling: bool,
    ) -> None:
        start = path.branch_pos if path.branch_pos is not None else 0
        end = get_path_length(path)
        start_col = pos_to_col(start)
        end_col = pos_to_col(end)

        total_width = len(prefix) + width + 15
        line = list(row_prefix.ljust(total_width))

        line_start = len(prefix) + start_col
        line_end = len(prefix) + end_col

        connector = "└" if is_last_sibling else "├"
        if line_start < len(line):
            line[line_start] = connector
        for i in range(line_start + 1, min(line_end, len(line))):
            line[i] = "─"
        if line_end < len(line):
            line[line_end] = "●"

        label = f" [{path.path_id}]"
        for i, c in enumerate(label):
            if line_end + 1 + i < len(line):
                line[line_end + 1 + i] = c

        lines.append("".join(line).rstrip())

        path_children = children.get(path.path_id, [])
        for i, child in enumerate(path_children):
            child_is_last = i == len(path_children) - 1

            total_width = len(prefix) + width + 15
            new_prefix = list(row_prefix.ljust(total_width))

            if not is_last_sibling:
                vert_col = len(prefix) + start_col
                if vert_col < len(new_prefix):
                    new_prefix[vert_col] = "│"

            branch_col = len(prefix) + pos_to_col(child.branch_pos or 0)
            if branch_col < len(new_prefix):
                new_prefix[branch_col] = "│"

            render_path(child, "".join(new_prefix), child_is_last)

    root_paths = children.get(None, [])
    for i, root in enumerate(root_paths):
        is_last = i == len(root_paths) - 1
        render_path(root, "", is_last)

    return lines


def format_tree_simple(
    tree_paths: list[TreePathLike],
    text_width: int = 40,
) -> list[str]:
    """Format tree as simple list with path details."""
    if not tree_paths:
        return []

    lines = []
    for path in tree_paths:
        text_preview = preview(oneline(path.continuation), text_width)
        if path.parent_id is None:
            lines.append(f'[{path.path_id}] "{text_preview}"')
        else:
            lines.append(
                f'[{path.path_id}] <- [{path.parent_id}]@{path.branch_pos}: '
                f'"{text_preview}"'
            )

    return lines


@dataclass
class SimplePath:
    """Simple implementation of TreePathLike for visualization."""

    path_id: int
    parent_id: int | None
    branch_pos: int | None
    continuation: str
    _token_ids: list[int]

    @property
    def token_ids(self) -> list[int]:
        return self._token_ids


def create_forking_tree_paths(
    greedy_traj_ids: list[int],
    greedy_continuation: str,
    fork_points: list[tuple[int, list[tuple[list[int], str]]]],
) -> list[SimplePath]:
    """Create tree paths from forking paths result."""
    paths = [
        SimplePath(
            path_id=0,
            parent_id=None,
            branch_pos=None,
            continuation=greedy_continuation,
            _token_ids=greedy_traj_ids,
        )
    ]

    path_id = 1
    for position, continuations in fork_points:
        for traj_ids, cont_text in continuations:
            paths.append(
                SimplePath(
                    path_id=path_id,
                    parent_id=0,
                    branch_pos=position,
                    continuation=cont_text,
                    _token_ids=traj_ids,
                )
            )
            path_id += 1

    return paths
