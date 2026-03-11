"""Logging utilities for entropy-seeking generation method.

This module provides detailed logging for the entropy-seeking algorithm,
including tree initialization, expansion rounds, and path visualization.
"""

from __future__ import annotations

from src.common.log import log
from src.common.log_utils import log_step
from src.inference import ModelRunner

from ..entropy_seeking_types import TreePath


def log_tree_visualization(
    tree_paths: list[TreePath],
    max_tokens: int = 128,
) -> None:
    """Log ASCII tree visualization of paths.

    Args:
        tree_paths: All paths in the tree
        max_tokens: Maximum tokens for scale
    """
    log("\n  Tree:")
    # Scale header
    scale_marks = list(range(0, max_tokens + 1, 21))
    scale_line = "      " + "".join(f"{m:<8}" for m in scale_marks)
    log(scale_line)

    # Build tree structure based on parent relationships
    for path in tree_paths:
        # Determine indentation based on branch position
        if path.parent_id is None:
            # Root path
            prefix = "      "
            connector = "\u251c" if path.path_id < len(tree_paths) - 1 else "\u2514"
            length = len(path.trajectory.token_ids) - (path.branch_pos or 0)
            bar = "\u2500" * min(4, length // 10)
            log(f"{prefix}{connector}{bar}\u25cf [{path.path_id}]")
        else:
            # Child path - show branching
            indent = "      " + " " * (path.branch_pos or 0)
            is_last = path.path_id == tree_paths[-1].path_id
            connector = "\u2514" if is_last else "\u251c"
            bar = "\u2500" * 2
            log(f"{indent}{connector}{bar}\u25cf [{path.path_id}]")


def log_paths(
    tree_paths: list[TreePath],
    runner: "ModelRunner",
    max_display: int = 5,
) -> None:
    """Log path texts.

    Args:
        tree_paths: Paths to display
        runner: Model runner for decoding
        max_display: Maximum number to show
    """
    log("\n  Paths:")
    for path in tree_paths[:max_display]:
        cont = path.continuation[:60] + "..." if len(path.continuation) > 60 else path.continuation
        log(f'    [{path.path_id}] "{cont}"')
    if len(tree_paths) > max_display:
        log(f"    ... and {len(tree_paths) - max_display} more")


def log_initialize_tree(
    tree_paths: list[TreePath],
    runner: "ModelRunner",
    samples: int,
    max_tokens: int,
) -> None:
    """Log tree initialization step.

    Args:
        tree_paths: Initial paths
        runner: Model runner for decoding
        samples: Number of samples
        max_tokens: Max tokens for tree visualization
    """
    log_step(1, f"Initialize tree ({samples} random samples)")
    log("  Sampling trajectories from prompt, computing entropy at each token...")

    log_tree_visualization(tree_paths, max_tokens)
    log_paths(tree_paths, runner)


def log_expansion_round(
    round_num: int,
    total_rounds: int,
    source_path: "TreePath",
    position: int,
    entropy: float,
    token: str,
    new_paths: list["TreePath"],
    all_paths: list["TreePath"],
    runner: "ModelRunner",
    prompt_len: int,
    max_tokens: int,
) -> None:
    """Log a single expansion round.

    Args:
        round_num: Current round number
        total_rounds: Total number of rounds
        source_path: Path we're branching from
        position: Token position we're branching at
        entropy: Entropy at that position
        token: Token text at the branch point
        new_paths: Newly created paths
        all_paths: All paths after expansion
        runner: Model runner for decoding
        prompt_len: Prompt length
        max_tokens: Max tokens for tree visualization
    """
    relative_pos = position - prompt_len
    log(f"\n  Round {round_num}/{total_rounds}: branch from [{source_path.path_id}] @ token {relative_pos} \"{token}\" (H={entropy:.2f})")

    if new_paths:
        log("  New paths:")
        for new_path in new_paths:
            cont = new_path.continuation[:50] + "..." if len(new_path.continuation) > 50 else new_path.continuation
            log(f'    [{new_path.path_id}] <- [{source_path.path_id}]@{relative_pos}: "{cont}"')

    log_tree_visualization(all_paths, max_tokens)


def log_expansion_summary(
    tree_paths: list[TreePath],
    initial_count: int,
    rounds: int,
) -> None:
    """Log summary after all expansion rounds.

    Args:
        tree_paths: All paths after expansion
        initial_count: Number of initial samples
        rounds: Number of expansion rounds
    """
    expansion_count = len(tree_paths) - initial_count
    log(f"\n  Total: {len(tree_paths)} paths ({initial_count} initial + {expansion_count} from expansion)")


def log_arm_header(arm_name: str, continuation: str) -> None:
    """Log arm header for entropy seeking.

    Args:
        arm_name: Name of the arm (trunk, branch_1, etc.)
        continuation: The continuation prefix for this arm
    """
    display_name = arm_name.replace("_", " ").title() if "_" in arm_name else arm_name.capitalize()
    if arm_name == "trunk":
        display_name = "Trunk"
    else:
        display_name = f"Branch: {arm_name}"

    log(f"\n{display_name}")
    log(f'  Continuation: "{continuation}"')
