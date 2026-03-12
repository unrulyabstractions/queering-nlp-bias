"""Helper functions for generation output formatting and statistics.

These functions support GenerationOutput's summary and analysis methods.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

from src.common.default_config import MAX_NEW_TOKENS
from src.common.logging import log, log_banner, log_sub_banner
from src.common.viz_utils import escape_newlines
from src.estimation.arm_types import get_arm_name_from_index

# Type alias for output functions (log or file writer)
OutputFn = Callable[[str], None]


def get_eos_markers(eos_token: str | None) -> list[str]:
    """Get EOS markers for finished trajectory detection."""
    return (
        [eos_token]
        if eos_token
        else ["<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>"]
    )


def group_trajectories_by_branch(
    trajs: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    """Group trajectories by branch index."""
    by_branch: dict[int, list[dict[str, Any]]] = {}
    for traj in trajs:
        arm_index = traj.get("arm_index", [0])
        if isinstance(arm_index, list):
            branch_idx = arm_index[0] if arm_index else 0
        else:
            branch_idx = arm_index
        by_branch.setdefault(branch_idx, []).append(traj)
    return by_branch


def get_continuation_text(traj: dict[str, Any]) -> str:
    """Get continuation text from trajectory dict.

    Computes from prefill_text + generated_text if continuation_text is not stored.
    """
    stored = traj.get("continuation_text")
    if stored:
        return stored
    prefill = traj.get("prefill_text") or ""
    generated = traj.get("generated_text") or ""
    return prefill + generated


def count_finished(trajs: list[dict[str, Any]], eos_markers: list[str]) -> int:
    """Count trajectories that contain an EOS marker."""
    return sum(
        1
        for t in trajs
        if any(eos in get_continuation_text(t) for eos in eos_markers)
    )


def format_branch_stats(
    trajs: list[dict[str, Any]],
    eos_markers: list[str],
    display_name: str,
) -> tuple[str, float]:
    """Format branch statistics (count and finished percentage).

    Returns:
        Tuple of (header_string, finished_percentage)
    """
    finished = count_finished(trajs, eos_markers)
    pct = (finished / len(trajs) * 100) if trajs else 0
    return f"{display_name} ({len(trajs)} trajectories, {pct:.0f}% finished)", pct


def compute_branch_probability_mass(
    trajs: list[dict[str, Any]],
    trunk_len: int,
) -> float:
    """Compute sum of conditional probabilities for trajectories in a branch."""
    total_cond_prob = 0.0
    for traj in trajs:
        logprobs = traj.get("logprobs", [])
        if len(logprobs) > trunk_len:
            cont_logprobs = logprobs[trunk_len:]
            cont_logp = sum(lp for lp in cont_logprobs if lp is not None)
            if cont_logp > -700:
                total_cond_prob += math.exp(cont_logp)
    return total_cond_prob


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY WRITING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════


def write_settings(
    out: OutputFn,
    model: str,
    method: str,
    generated_at: str,
    num_trajectories: int,
    prefix: str = "  ",
) -> None:
    """Write settings section."""
    out(f"{prefix}Model:       {model}")
    out(f"{prefix}Method:      {method}")
    out(f"{prefix}Generated:   {generated_at}")
    out(f"{prefix}Trajectories:{num_trajectories}")


def write_config(
    out: OutputFn,
    config: dict[str, Any],
    prefix: str = "  ",
    truncate: bool = True,
) -> None:
    """Write config section."""
    prompt = config.get("prompt", "")
    trunk = config.get("trunk", "")
    branches = config.get("branches", [])
    real_branches = [b for b in branches if b != "trunk"]

    if truncate and len(prompt) > 70:
        out(f"{prefix}Prompt: {prompt[:70]}...")
    else:
        out(f"{prefix}Prompt: {prompt}")

    if trunk:
        out(f'{prefix}Trunk:  "{trunk}"')
    if real_branches:
        out(
            f"{prefix}Branches: {', '.join(f'{chr(34)}{b}{chr(34)}' for b in real_branches)}"
        )
    out(f"{prefix}Temperature: {config.get('temperature', 1.0)}")
    out(f"{prefix}Max tokens:  {config.get('max_new_tokens', MAX_NEW_TOKENS)}")


def write_trajectories_by_branch(
    out: OutputFn,
    tree: dict[str, Any],
    config: dict[str, Any],
    eos_token: str | None,
    prefix: str = "  ",
    max_trajs: int = 3,
    max_text_len: int = 60,
) -> None:
    """Write trajectories grouped by arm (root, trunk, branches, twigs)."""
    if not tree:
        return

    trajs = tree.get("trajs", [])
    eos_markers = get_eos_markers(eos_token)
    by_branch = group_trajectories_by_branch(trajs)

    # Get arm names from config if available, otherwise use indices
    arms = config.get("arms", [])
    if arms:
        # Use arm names from config (includes root, trunk, branches, twigs)
        arm_indices = sorted(by_branch.keys())
        for arm_idx in arm_indices:
            trajs_in_branch = by_branch.get(arm_idx, [])
            # Get arm name from config arms list
            if arm_idx < len(arms):
                display_name = arms[arm_idx].get("name", get_arm_name_from_index(arm_idx))
            else:
                display_name = get_arm_name_from_index(arm_idx)
            header, _ = format_branch_stats(trajs_in_branch, eos_markers, display_name)

            out(f"\n{prefix}{header}:")
            for i, traj in enumerate(trajs_in_branch[:max_trajs]):
                text = escape_newlines(get_continuation_text(traj)[:max_text_len])
                out(f"{prefix}  [{traj.get('idx', i)}] {text}")
    else:
        # Fallback: iterate over all arm indices found in data
        for arm_idx in sorted(by_branch.keys()):
            trajs_in_branch = by_branch[arm_idx]
            display_name = get_arm_name_from_index(arm_idx)
            header, _ = format_branch_stats(trajs_in_branch, eos_markers, display_name)

            out(f"\n{prefix}{header}:")
            for i, traj in enumerate(trajs_in_branch[:max_trajs]):
                text = escape_newlines(get_continuation_text(traj)[:max_text_len])
                out(f"{prefix}  [{traj.get('idx', i)}] {text}")

        if len(trajs_in_branch) > max_trajs:
            out(f"{prefix}  ... and {len(trajs_in_branch) - max_trajs} more")


def write_probability_mass(
    out: OutputFn,
    tree: dict[str, Any],
    config: dict[str, Any],
    prefix: str = "  ",
) -> None:
    """Write probability mass captured per arm."""
    if not tree:
        return

    trajs = tree.get("trajs", [])
    trunk_len = tree.get("trunk_length", 0)
    by_branch = group_trajectories_by_branch(trajs)
    arms = config.get("arms", [])

    out(
        f"\n{prefix}{'Arm':<12} {'N':>4}  {'Sum(p|arm)':>14}  {'Coverage':>10}"
    )
    out(f"{prefix}" + "─" * 46)

    for arm_idx in sorted(by_branch.keys()):
        trajs_in_branch = by_branch[arm_idx]
        # Get arm name from config arms list if available
        if arms and arm_idx < len(arms):
            display_name = arms[arm_idx].get("name", get_arm_name_from_index(arm_idx))
        else:
            display_name = get_arm_name_from_index(arm_idx)

        total_cond_prob = compute_branch_probability_mass(
            trajs_in_branch, trunk_len
        )
        coverage_pct = total_cond_prob * 100

        prob_str = f"{total_cond_prob:.2e}" if total_cond_prob > 0 else "0"
        coverage_str = (
            f"{coverage_pct:.1f}%" if coverage_pct < 100 else f"{coverage_pct:.0f}%"
        )

        out(
            f"{prefix}{display_name:<12} {len(trajs_in_branch):>4}  "
            f"{prob_str:>14}  {coverage_str:>10}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════


def save_generation_summary(
    path: str | Path,
    model: str,
    method: str,
    generated_at: str,
    num_trajectories: int,
    config: dict[str, Any],
    tree: dict[str, Any] | None,
    eos_token: str | None,
) -> Path:
    """Save human-readable summary to text file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    def add_line(text: str = "") -> None:
        lines.append(text)

    # Header
    add_line("=" * 76)
    add_line("  GENERATION SUMMARY")
    add_line("=" * 76)
    add_line()

    # Settings
    write_settings(add_line, model, method, generated_at, num_trajectories)
    add_line()

    # Config
    add_line("-" * 76)
    add_line("  CONFIG")
    add_line("-" * 76)
    write_config(add_line, config, truncate=True)
    add_line()

    # Trajectories
    if tree:
        add_line("-" * 76)
        add_line("  TRAJECTORIES")
        add_line("-" * 76)
        write_trajectories_by_branch(
            add_line, tree, config, eos_token, max_trajs=999999, max_text_len=999999
        )

    add_line()
    add_line("=" * 76)

    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def print_generation_summary(
    model: str,
    method: str,
    generated_at: str,
    num_trajectories: int,
    config: dict[str, Any],
    tree: dict[str, Any] | None,
    eos_token: str | None,
) -> None:
    """Print a clean summary of generation results."""
    log_banner("GENERATION SUMMARY")

    # Settings
    log("\nSettings:")
    write_settings(log, model, method, generated_at, num_trajectories)

    # Config - show full prompt
    prompt = config.get("prompt", "")
    trunk = config.get("trunk", "")
    branches = config.get("branches", [])
    real_branches = [b for b in branches if b != "trunk"]

    log("\n  Prompt:")
    for line in prompt.split("\n"):
        log(f"    {line}")
    if trunk:
        log(f'\n  Trunk: "{trunk}"')
    if real_branches:
        log(f"\n  Branches ({len(real_branches)}):")
        for i, branch in enumerate(real_branches):
            log(f'    [{i + 1}] "{branch}"')

    # Generation params
    temp = config.get("temperature", 1.0)
    max_tokens = config.get("max_new_tokens", MAX_NEW_TOKENS)
    log(f"  Temperature: {temp}")
    log(f"  Max tokens: {max_tokens}")

    # Trajectories by branch
    log_sub_banner("CONTINUATIONS BY BRANCH")
    write_trajectories_by_branch(log, tree, config, eos_token, max_trajs=5, max_text_len=80)

    # Probability mass
    if tree:
        log_sub_banner("PROBABILITY MASS CAPTURED PER BRANCH")
        write_probability_mass(log, tree, config)

    # Final stats
    log_sub_banner(f"Total trajectories: {num_trajectories}")
