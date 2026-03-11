"""Display utilities for estimation comparison.

This module provides functions for displaying and comparing estimation
results across different generation methods.
"""

from __future__ import annotations

import json
from typing import Any

from src.common.log import log
from src.common.log_utils import (
    fmt_core,
    log_divider,
    log_major,
    log_section_title,
    log_table_header,
)
from src.generation import OutputPaths

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

    # Arms section
    trunk = config.get("trunk", "")
    branches = config.get("branches", [])

    log("  ARMS:")
    if trunk:
        log(f'    TRUNK:    "{trunk}"')

    # Branches - show each with TRUNK + branch
    if branches and branches != ["trunk"]:
        real_branches = [b for b in branches if b != "trunk"]
        for i, b in enumerate(real_branches):
            log(f'    BRANCH_{i + 1}: TRUNK + "{b}"')

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
    col_w = 7

    def fmt_header(lbls: list[str]) -> str:
        """Format header row with centered labels."""
        return "".join(f"{lbl:^{col_w}}" for lbl in lbls)

    def fmt_values(values: list[float]) -> str:
        """Format values row with centered numbers."""
        result = []
        for v in values:
            # Avoid -0.00 display
            if abs(v) < 0.005:
                v = 0.0
            result.append(f"{v:^{col_w}.2f}")
        return "".join(result)

    arms = est_data.get("arms", [])

    if not arms:
        return

    log_section_title("RESULTS")
    log_divider(width=76)

    # Legend for expectations
    log("  Legend:")
    log("    core         = expected compliance per structure (normative center)")
    log("    E[θ|trunk]   = how this arm's core differs from trunk (direction)")
    log("    ||E[θ]||     = distance between cores: ||core - trunk_core||")
    log("    E[∂|branch]  = avg spread around this arm's core")
    log("    E[∂|trunk]   = avg spread around trunk's core")
    log("    E[Δ∂]        = E[∂|branch] - E[∂|trunk] (deviance difference)")
    log("")

    header_indent = "                    "
    table_width = len(header_indent) + col_w * len(labels)

    def log_arm_data(
        display_name: str,
        n_traj: int,
        core: list[float],
        dev_branch: float,
        dev_trunk: float,
        dev_delta: float,
        orient_avg: list[float],
        orient_norm: float,
    ) -> None:
        """Helper to print one arm's data."""

        def fmt_scalar(v: float) -> str:
            if abs(v) < 0.00005:
                v = 0.0
            return f"{v:>7.4f}"

        log(f"  {display_name} ({n_traj} trajectories)")
        log(f"      core         = {fmt_values(core)}")
        if orient_avg:
            log(f"      E[θ|trunk]   = {fmt_values(orient_avg)}")
        log(f"      ||E[θ]||     = {fmt_scalar(orient_norm)}")
        log(f"      E[∂|branch]  = {fmt_scalar(dev_branch)}")
        log(f"      E[∂|trunk]   = {fmt_scalar(dev_trunk)}")
        log(f"      E[Δ∂]        = {fmt_scalar(dev_delta)}")
        log("")

    def get_display_name(arm: dict) -> str:
        """Helper to get display name for an arm."""
        name = arm.get("name", "")
        if name == "trunk":
            return "TRUNK"
        elif name == "all_arms":
            return "ALL_ARMS"
        elif name.startswith("branch_"):
            return name.upper().replace("_", "_")
        else:
            idx = arm.get("arm_index", 0)
            if idx > 0:
                return f"BRANCH_{idx}"
            return name.upper()

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
        log(f"{header_indent}{fmt_header(labels)}")
        log(f"    {'─' * (table_width - 4)}")

        for arm in arms:
            display_name = get_display_name(arm)
            trajectories = arm.get("trajectories", [])
            n_traj = len(trajectories)

            est = arm.get("estimates", {}).get(method_name, {})
            log_arm_data(
                display_name,
                n_traj,
                est.get("core", []),
                est.get("deviance_avg", 0.0),
                est.get("deviance_avg_trunk", 0.0),
                est.get("deviance_delta", 0.0),
                est.get("orientation_avg", []),
                est.get("orientation_norm", 0.0),
            )


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
    for idx, arm_name in enumerate(arm_names):
        display_name = "TRUNK" if idx == 0 else f"BRANCH_{idx}"
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
