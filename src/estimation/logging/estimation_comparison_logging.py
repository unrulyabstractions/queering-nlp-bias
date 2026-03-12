"""Display utilities for estimation comparison.

This module provides functions for displaying and comparing estimation
results across different generation methods.
"""

from __future__ import annotations

import json
from typing import Any

from src.common.logging import (
    fmt_core,
    log,
    log_divider,
    log_major,
    log_section_title,
    log_table_header,
)
from src.common.experiment_types import OutputPaths

from src.scoring.scoring_method_registry import iter_methods as iter_scoring_methods

from ..estimation_experiment_types import EstimationArmResult, EstimationResult
from ..weighting_method_registry import get_method_description
from ..weighting_method_registry import iter_methods as iter_weighting_methods


# ══════════════════════════════════════════════════════════════════════════════
# SETUP SUMMARY
# ══════════════════════════════════════════════════════════════════════════════


def log_setup_summary(paths: OutputPaths) -> None:
    """Log a concise summary of the experiment setup and results."""
    # Load generation config
    with open(paths.generation) as f:
        gen_data = json.load(f)
    config = gen_data.get("config", {})

    # Load scoring config from judgment file
    with open(paths.judgment) as f:
        judge_data = json.load(f)

    # Load estimation results
    with open(paths.estimation) as f:
        est_data = json.load(f)

    log_section_title("SETUP")
    log_divider(width=76)

    # Prompt (full, separated)
    prompt = config.get("prompt", "")
    log("  Prompt:")
    log(f"    {prompt}")
    log("")

    # Arms section - show all arm types
    trunk = config.get("trunk", "")
    branches = config.get("branches", [])
    twig_variations = config.get("twig_variations", [])

    log("  ARMS:")
    log("")
    log("    ROOT:       prompt only (no trunk)")
    log("")
    if trunk:
        log(f'    TRUNK:      "{trunk}"')
        log("")

    # Branches and their twigs
    if branches and branches != ["trunk"]:
        real_branches = [b for b in branches if b != "trunk"]
        for i, b in enumerate(real_branches, start=1):
            log(f'    BRANCH_{i}:  TRUNK + "{b}"')
            # Show twig variations for this branch
            for t, twig in enumerate(twig_variations, start=1):
                log(f'        TWIG_{t}_B{i}:  BRANCH_{i} + "{twig}"')
            log("")

    log("")

    # Models
    gen_model = config.get("model", "")
    judge_model = judge_data.get("judge_model", "")
    embed_model = judge_data.get("embedding_model", "")

    log("  Models:")
    if gen_model:
        log(f"    gen:   {gen_model}")
    if judge_model:
        log(f"    judge: {judge_model}")
    if embed_model:
        log(f"    embed: {embed_model}")

    log("")

    # Structures - use scoring registry to discover methods
    log("  Structures:")
    scoring_data = judge_data.get("scoring_data", {})

    def truncate(s: str, max_len: int = 55) -> str:
        return s[: max_len - 3] + "..." if len(s) > max_len else s

    # Build structure labels by iterating through registered methods
    labels: list[str] = []
    for method_name, params_class, _ in iter_scoring_methods():
        config_key = params_class.config_key
        items = scoring_data.get(config_key, [])
        if not items:
            continue

        label_prefix = params_class.label_prefix
        for i, item in enumerate(items):
            label = f"{label_prefix}{i + 1}"
            labels.append(label)
            if isinstance(item, list):
                log(f"    {label}: [bundled: {len(item)}]")
                for sub in item:
                    log(f"        * {truncate(sub)}")
            else:
                log(f"    {label}: {truncate(item)}")

    log("")

    # Results display
    _log_results_summary(est_data, labels)


def _log_results_summary(est_data: dict[str, Any], labels: list[str]) -> None:
    """Log the results summary from estimation data."""
    from src.estimation.arm_types import ArmKind, classify_arm

    col_w = 7

    def fmt_header(lbls: list[str]) -> str:
        """Format header row with centered labels."""
        return "".join(f"{lbl:^{col_w}}" for lbl in lbls)

    def fmt_values(values: list[float]) -> str:
        """Format values row with centered numbers."""
        result = []
        for v in values:
            if abs(v) < 0.005:
                v = 0.0
            result.append(f"{v:^{col_w}.2f}")
        return "".join(result)

    def fmt_scalar(v: float) -> str:
        if abs(v) < 0.00005:
            v = 0.0
        return f"{v:>7.4f}"

    arms = est_data.get("arms", [])
    if not arms:
        return

    # Check if we have root arm
    has_root = any(arm.get("name") == "root" for arm in arms)

    log_section_title("RESULTS")
    log_divider(width=76)

    # Legend
    log("  core   = normativity characterization (weighted avg score per structure)")
    log("  E[∂|X] = avg divergence from normativity (lower = tighter clustering)")
    log("")

    header_indent = "                "
    table_width = len(header_indent) + col_w * len(labels)

    # Iterate over all registered weighting methods
    for method_name, _, _ in iter_weighting_methods():
        desc = get_method_description(method_name)
        # Check if any arm has this method
        has_method = any(
            arm.get("estimates", {}).get(method_name, {}).get("core")
            for arm in arms
        )
        if not has_method:
            continue

        log(f"  [{desc}]")
        log("")

        # TABLE 1: Cores
        log(f"  Cores:")
        log(f"    {'Arm':<14}  {'N':>4}  {fmt_header(labels)}")
        log(f"    {'─' * (20 + col_w * len(labels))}")

        for arm in arms:
            name = arm.get("name", "")
            trajectories = arm.get("trajectories", [])
            n_traj = len(trajectories)
            est = arm.get("estimates", {}).get(method_name, {})
            core = est.get("core", [])
            log(f"    {name:<14}  {n_traj:>4}  {fmt_values(core)}")

        log("")

        # TABLE 2: Deviances
        log(f"  Deviances:")

        # Check what arm types we have
        has_branch = any(classify_arm(arm.get("name", "")) == ArmKind.BRANCH for arm in arms)
        has_twig = any(classify_arm(arm.get("name", "")) == ArmKind.TWIG for arm in arms)

        # Build header based on what arms we have
        dev_cols = ["E[∂|root]"] if has_root else []
        dev_cols.append("E[∂|trunk]")
        if has_branch:
            dev_cols.append("E[∂|branch]")
        if has_twig:
            dev_cols.append("E[∂|twig]")
        dev_header = "  ".join(f"{c:>11}" for c in dev_cols)
        log(f"    {'Arm':<14}  {dev_header}")
        log(f"    {'─' * (16 + 13 * len(dev_cols))}")

        for arm in arms:
            name = arm.get("name", "")
            kind = classify_arm(name)
            est = arm.get("estimates", {}).get(method_name, {})
            dev_self = est.get("deviance_avg", 0.0)
            dev_root = est.get("deviance_avg_root", 0.0)
            dev_trunk = est.get("deviance_avg_trunk", 0.0)

            # Build row based on arm type
            row_parts = []

            # E[∂|root] column
            if has_root:
                if kind == ArmKind.ROOT:
                    row_parts.append(f"{fmt_scalar(dev_self):>11}")
                else:
                    row_parts.append(f"{fmt_scalar(dev_root):>11}")

            # E[∂|trunk] column
            if kind == ArmKind.ROOT:
                row_parts.append(f"{'—':>11}")
            elif kind == ArmKind.TRUNK:
                row_parts.append(f"{fmt_scalar(dev_self):>11}")
            else:
                row_parts.append(f"{fmt_scalar(dev_trunk):>11}")

            # E[∂|branch] column
            if has_branch:
                if kind == ArmKind.BRANCH:
                    row_parts.append(f"{fmt_scalar(dev_self):>11}")
                elif kind == ArmKind.TWIG:
                    # For twigs, show deviance from parent branch
                    dev_parent = est.get("deviance_avg_parent", dev_trunk)
                    row_parts.append(f"{fmt_scalar(dev_parent):>11}")
                else:
                    row_parts.append(f"{'—':>11}")

            # E[∂|twig] column
            if has_twig:
                if kind == ArmKind.TWIG:
                    row_parts.append(f"{fmt_scalar(dev_self):>11}")
                else:
                    row_parts.append(f"{'—':>11}")

            row = "  ".join(row_parts)
            log(f"    {name:<14}  {row}")

        log("")


# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON DISPLAY
# ══════════════════════════════════════════════════════════════════════════════


def log_arm_table(
    results: list[EstimationResult],
    arm_names: list[str],
    method_name: str,
    core_label: str,
) -> None:
    """Log a comparison table for all arms using a specific weighting method."""
    for arm_name in arm_names:
        display_name = arm_name.upper()
        log_section_title(display_name)
        log_table_header(
            [
                ("Method", 18, "<"),
                ("N", 4, ">"),
                ("E[∂]", 8, ">"),
                (core_label, 40, "<"),
            ],
            divider_width=76,
        )

        for r in results:
            arm = next((a for a in r.arms if a.name == arm_name), None)
            if arm:
                core = arm.get_core(method_name)
                deviance = arm.get_deviance_avg(method_name)
                core_str = fmt_core(core)
                log(
                    f"  {r.method:<18}  {arm.n_trajectories:>4}  "
                    f"{deviance:>8.4f}  {core_str}"
                )


def display_comparison(results: list[EstimationResult], gap: int = 2) -> None:
    """Display side-by-side comparison of estimation results."""
    log_major("COMPARISON: Core Estimation by Generation Method", gap=gap)

    # Print setup summary first
    if results:
        log_setup_summary(results[0].paths)

    # Collect all unique arm names
    all_arm_names: list[str] = []
    for r in results:
        for a in r.arms:
            if a.name not in all_arm_names:
                all_arm_names.append(a.name)

    # Iterate over all registered weighting methods
    for method_name, _, _ in iter_weighting_methods():
        desc = get_method_description(method_name)

        # Check if any arm has this method
        has_method = any(
            a.estimates.get(method_name, {}).get("core")
            for r in results
            for a in r.arms
        )
        if not has_method:
            continue

        log_section_title(desc.upper())
        log_divider(width=76)

        log_arm_table(
            results,
            all_arm_names,
            method_name,
            f"Core ({desc})",
        )

    log("")
