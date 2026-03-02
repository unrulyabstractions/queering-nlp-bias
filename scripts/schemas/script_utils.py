"""Shared utilities for generation scripts.

This module provides common functions used across all generate_by_*.py scripts
to reduce code duplication.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.common.analysis import analyze_token_tree
from src.common.device_utils import clear_gpu_memory
from src.common.log import log, log_section


def _fmt_prob(p: float, width: int = 10) -> str:
    """Format probability, using scientific notation for very small values."""
    if p < 0.0001:
        return f"{p:>{width}.1e}"
    return f"{p:>{width}.4f}"
from src.common.seed import set_seed
from src.common.token_tree import TokenTree
from src.common.viz_utils import preview
from src.inference import ModelRunner

from .generation import (
    BranchGenerationResult,
    GenerationConfig,
    GenerationOutput,
)

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


def log_step(step_num: int, title: str, detail: str = "") -> None:
    """Log a step header with consistent formatting."""
    header = f"  Step {step_num}: {title}"
    if detail:
        header += f" ({detail})"
    log(f"\n{header}")
    log("  " + "─" * 50)


def log_wrapped(text: str, indent: str = "  ", width: int = 78, gap: int = 0) -> None:
    """Log text with word wrapping."""
    words = text.split()
    line = indent
    first = True
    for word in words:
        if len(line) + len(word) + 1 > width:
            log(line, gap=gap if first else 0)
            first = False
            line = indent + word
        else:
            line = line + " " + word if line != indent else indent + word
    if line.strip():
        log(line, gap=gap if first else 0)


def load_model(config: GenerationConfig) -> ModelRunner:
    """Load and validate the model from config."""
    if not config.model:
        raise ValueError("No model specified in generation config")

    log(f"Loading model: {config.model}")
    return ModelRunner(config.model)


def log_branch_header(branch_name: str, formatted_prompt: str) -> None:
    """Log section header and prompt for a branch.

    Args:
        branch_name: Name of the branch ("trunk" or branch name)
        formatted_prompt: The full formatted prompt to display
    """
    label = "Trunk" if branch_name == "trunk" else f"Branch: {branch_name}"
    log_section(label)
    log(f'  Prompt: "{preview(formatted_prompt, 70)}"')


def _log_trajectories(result: BranchGenerationResult, runner: ModelRunner) -> None:
    """Log trajectory texts and conditional probabilities.

    Shows two tables:
    1. Trajectory index, branch, and decoded text
    2. Conditional probabilities p(t|prompt), p(t|trunk), p(t|branch)
    """
    prompt_len = result.prompt_length
    trunk_len = result.trunk_length

    # Table 1: Trajectory texts
    log(f"  Trajectories ({len(result.trajectories)} total):")
    log(f"  {'[#]':<4} {'branch':<10} text")
    log("  " + "─" * 60)

    for i, traj in enumerate(result.trajectories):
        group_idx = result.group_indices[i]
        display = "trunk" if group_idx == 0 else f"branch_{group_idx}"
        text = runner.decode_ids(traj.token_ids)
        log(f"  [{i}] {display:<10} {preview(text, 50)}")
    log("")

    # Table 2: Conditional probabilities
    log(f"  Conditional probabilities (prompt_len={prompt_len}, trunk_len={trunk_len}):")
    log(f"  {'[#]':<4} {'branch':<10} {'p(t|prompt)':>10}  {'p(t|trunk)':>10}  {'p(t|branch)':>10}")
    log("  " + "─" * 50)

    for i, traj in enumerate(result.trajectories):
        group_idx = result.group_indices[i]
        p_prompt = traj.get_conditional_prob(prompt_len, traj.length) or 0.0

        if group_idx == 0:
            p_trunk = traj.get_conditional_prob(trunk_len, traj.length) or 0.0
            p_branch = p_trunk
        else:
            # Branch token at trunk_len-1 due to BPE merge
            p_trunk = traj.get_conditional_prob(trunk_len - 1, traj.length) or 0.0
            p_branch = traj.get_conditional_prob(trunk_len, traj.length) or 0.0

        display = "trunk" if group_idx == 0 else f"branch_{group_idx}"
        log(f"  [{i}] {display:<10} {_fmt_prob(p_prompt)}  {_fmt_prob(p_trunk)}  {_fmt_prob(p_branch)}")
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

    output = GenerationOutput.from_tree(config, config.model, tree, method=method)
    out_path = GenerationOutput.compute_output_path(config_path, method=method)
    output.save(out_path)

    log(f"Saved {len(result.trajectories)} trajectories to {out_path}", gap=1)

    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# Tree Visualization
# ══════════════════════════════════════════════════════════════════════════════


class TreePathLike(Protocol):
    """Protocol for objects that can be visualized as tree paths."""

    path_id: int
    parent_id: int | None
    branch_pos: int | None  # Relative token position where branched
    continuation: str  # Text for display

    @property
    def token_ids(self) -> list[int]: ...


def oneline(text: str) -> str:
    """Collapse whitespace to single spaces for display."""
    return re.sub(r"\s+", " ", text).strip()


def format_horizontal_tree(
    tree_paths: list[TreePathLike],
    prompt_len: int,
    max_new_tokens: int,
    width: int = 50,
) -> list[str]:
    """Format tree as horizontal timeline showing token positions.

    Example output:
        tokens: 0    5    10   15   20   25   30
        ├────────────────────────────────────● [0]
        │         ├──────────────● [2]
        │         └────────────────● [3]
        └──────────────────┬─────────────────● [1]
                           └────● [4]
    """
    if not tree_paths:
        return []

    # Scale factor: chars per token (relative to generation start)
    scale = width / max(max_new_tokens, 1)

    def pos_to_col(rel_token_pos: int) -> int:
        """Convert relative token position to column."""
        return int(rel_token_pos * scale)

    # Build tree structure
    children: dict[int | None, list[TreePathLike]] = {}
    for path in tree_paths:
        parent = path.parent_id
        if parent not in children:
            children[parent] = []
        children[parent].append(path)

    lines: list[str] = []

    # Token position ruler (relative to generation start)
    prefix = "    "
    ruler = prefix
    step = max(5, max_new_tokens // 6)
    for i in range(0, max_new_tokens + 1, step):
        col = pos_to_col(i)
        label = str(i)
        ruler = ruler.ljust(len(prefix) + col) + label
    lines.append(ruler)

    def get_path_length(path: TreePathLike) -> int:
        """Get the number of generated tokens (excluding prompt)."""
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

        # Build the line
        total_width = len(prefix) + width + 15
        line = list(row_prefix.ljust(total_width))

        # Draw horizontal line from start to end
        line_start = len(prefix) + start_col
        line_end = len(prefix) + end_col

        connector = "└" if is_last_sibling else "├"
        if line_start < len(line):
            line[line_start] = connector
        for i in range(line_start + 1, min(line_end, len(line))):
            line[i] = "─"
        if line_end < len(line):
            line[line_end] = "●"

        # Add path label
        label = f" [{path.path_id}]"
        for i, c in enumerate(label):
            if line_end + 1 + i < len(line):
                line[line_end + 1 + i] = c

        lines.append("".join(line).rstrip())

        # Render children
        path_children = children.get(path.path_id, [])
        for i, child in enumerate(path_children):
            child_is_last = i == len(path_children) - 1

            # Build prefix for children
            total_width = len(prefix) + width + 15
            new_prefix = list(row_prefix.ljust(total_width))

            # Vertical line from parent's start if parent continues
            if not is_last_sibling:
                vert_col = len(prefix) + start_col
                if vert_col < len(new_prefix):
                    new_prefix[vert_col] = "│"

            # Vertical line at child's branch point
            branch_col = len(prefix) + pos_to_col(child.branch_pos or 0)
            if branch_col < len(new_prefix):
                new_prefix[branch_col] = "│"

            render_path(child, "".join(new_prefix), child_is_last)

    # Render all root paths
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
                f'[{path.path_id}] <- [{path.parent_id}]@{path.branch_pos}: "{text_preview}"'
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
    fork_points: list[
        tuple[int, list[tuple[list[int], str]]]
    ],  # [(position, [(token_ids, continuation), ...])]
) -> list[SimplePath]:
    """Create tree paths from forking paths result.

    Args:
        greedy_traj_ids: Token IDs of the greedy trajectory
        greedy_continuation: Text continuation of greedy path
        fork_points: List of (position, continuations) where each continuation is (token_ids, text)

    Returns:
        List of SimplePath objects for tree visualization
    """
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
                    parent_id=0,  # All forks come from greedy path
                    branch_pos=position,
                    continuation=cont_text,
                    _token_ids=traj_ids,
                )
            )
            path_id += 1

    return paths
