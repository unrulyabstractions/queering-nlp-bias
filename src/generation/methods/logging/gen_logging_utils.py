"""Logging utilities for generation methods.

This module provides logging functions for trajectory generation output.
"""

from __future__ import annotations

from src.common.callback_types import LogFn
from src.common.logging import fmt_prob, log, log_section
from src.common.viz_utils import escape_newlines, preview
from src.estimation.arm_types import ArmKind, classify_arm, get_parent_branch
from src.common.experiment_types import ArmGenerationResult, GenerationArm
from src.inference import ModelRunner


def log_arm_header(arm: GenerationArm, log_fn: LogFn) -> None:
    """Log the header for an arm with its name and prefill text."""
    log_fn(f"\n{arm.name.replace('_', ' ').title()}")
    if arm.prefill:
        log_fn(f'  Prefill: "{escape_newlines(arm.prefill)}"')
    else:
        log_fn("  Prefill: N/A")


def _get_arm_display_name(arm_idx: int, arm_names: list[str] | None) -> str:
    """Get display name for an arm by index."""
    if arm_names and arm_idx < len(arm_names):
        return arm_names[arm_idx]
    # Fallback for old format without arm_names
    return f"arm_{arm_idx}"


def _get_parent_branch_index(twig_name: str, arm_names: list[str]) -> int | None:
    """Get the arm index of the parent branch for a twig.

    Args:
        twig_name: Name like "twig_b2_1" (twig 1 of branch 2)
        arm_names: List of all arm names

    Returns:
        Index of parent branch in arm_names, or None if not found
    """
    if not twig_name.startswith("twig_"):
        return None
    parent_branch_name = get_parent_branch(twig_name)
    if parent_branch_name is None:
        return None
    try:
        return arm_names.index(parent_branch_name)
    except ValueError:
        return None


def log_tree_trajectories(result: ArmGenerationResult, runner: ModelRunner) -> None:
    """Log trajectory texts and conditional probabilities.

    Args:
        result: Generation result with trajectories
        runner: Model runner for text decoding
    """
    prompt_len = result.prompt_length
    trunk_len = result.trunk_length
    arm_names = result.arm_names or []
    arm_token_lengths = result.arm_token_lengths or []

    log_section("Building Tree")

    # Table 1: Trajectory with prefill and generated columns
    log(f"  Trajectories ({len(result.trajectories)} total):")
    log(f"  {'#':>3}  {'arm':<12}  {'prefill':<47}  generated")
    log("  " + "─" * 112)

    for i, traj in enumerate(result.trajectories):
        arm_idx = result.arm_indices[i]
        arm_name = _get_arm_display_name(arm_idx, arm_names)

        # Use stored fields directly - no recomputation
        prefill = traj.prefill_text if traj.prefill_text else "N/A"
        generated = traj.generated_text or ""

        # Format for display using shared utilities
        prefill_display = preview(prefill, 45)
        generated_display = preview(generated, 55)

        log(f"  {i:>3}  {arm_name:<12}  {prefill_display:<47}  {generated_display}")
    log("")

    # Check what arm types we have
    has_branch = any(classify_arm(n) == ArmKind.BRANCH for n in arm_names)
    has_twig = any(classify_arm(n) == ArmKind.TWIG for n in arm_names)

    # Table 2: Conditional probabilities with proper columns
    log(
        f"  Conditional probabilities (prompt_len={prompt_len}, trunk_len={trunk_len}):"
    )

    # Build header based on what arm types exist
    header_parts = [f"{'#':>3}", f"{'arm':<12}"]
    header_parts.append(f"{'p(t|root)':>11}")
    header_parts.append(f"{'p(t|trunk)':>11}")
    if has_branch:
        header_parts.append(f"{'p(t|branch)':>11}")
    if has_twig:
        header_parts.append(f"{'p(t|twig)':>11}")
    header_parts.append(f"{'EOS?':>5}")
    log("  " + "  ".join(header_parts))
    log("  " + "─" * (35 + 13 * (2 + int(has_branch) + int(has_twig))))

    eos_token = runner.eos_token

    for i, traj in enumerate(result.trajectories):
        arm_idx = result.arm_indices[i]
        arm_name = _get_arm_display_name(arm_idx, arm_names)
        kind = classify_arm(arm_name)

        # Compute conditional probabilities from each reference point
        # p(t|root) = p(trajectory | prompt only)
        p_root = traj.get_conditional_prob(prompt_len, traj.length) or 0.0

        # p(t|trunk) = p(trajectory | prompt + trunk)
        if kind == ArmKind.ROOT:
            p_trunk = None  # N/A for root
        else:
            p_trunk = traj.get_conditional_prob(trunk_len, traj.length) or 0.0

        # p(t|branch) = p(trajectory | prompt + trunk + branch)
        p_branch = None
        if kind == ArmKind.BRANCH:
            # For branch, use its own token length
            branch_len = arm_token_lengths[arm_idx] if arm_idx < len(arm_token_lengths) else trunk_len
            p_branch = traj.get_conditional_prob(branch_len, traj.length) or 0.0
        elif kind == ArmKind.TWIG:
            # For twig, use parent branch's token length
            parent_idx = _get_parent_branch_index(arm_name, arm_names)
            if parent_idx is not None and parent_idx < len(arm_token_lengths):
                branch_len = arm_token_lengths[parent_idx]
                p_branch = traj.get_conditional_prob(branch_len, traj.length) or 0.0

        # p(t|twig) = p(trajectory | prompt + trunk + branch + twig)
        p_twig = None
        if kind == ArmKind.TWIG:
            twig_len = arm_token_lengths[arm_idx] if arm_idx < len(arm_token_lengths) else trunk_len
            p_twig = traj.get_conditional_prob(twig_len, traj.length) or 0.0

        # Check if trajectory has EOS token using stored generated_text
        is_finished = eos_token is not None and eos_token in (traj.generated_text or "")
        finished_str = "Y" if is_finished else "N"

        # Format row
        row_parts = [f"{i:>3}", f"{arm_name:<12}"]
        row_parts.append(fmt_prob(p_root, 11))
        row_parts.append(fmt_prob(p_trunk, 11) if p_trunk is not None else f"{'N/A':>11}")
        if has_branch:
            row_parts.append(fmt_prob(p_branch, 11) if p_branch is not None else f"{'N/A':>11}")
        if has_twig:
            row_parts.append(fmt_prob(p_twig, 11) if p_twig is not None else f"{'N/A':>11}")
        row_parts.append(f"{finished_str:>5}")

        log("  " + "  ".join(row_parts))
    log("")
