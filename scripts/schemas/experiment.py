"""Experiment result dataclasses and display utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from .generation import OutputPaths
from .script_utils import (
    fmt_core,
    log_divider,
    log_section_title,
    log_table_header,
)
from src.common.log import log


# ══════════════════════════════════════════════════════════════════════════════
# Types
# ══════════════════════════════════════════════════════════════════════════════

GenerationMethod = Literal["simple-sampling", "forking-paths", "seeking-entropy"]


# ══════════════════════════════════════════════════════════════════════════════
# Result Dataclasses
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ArmResult:
    """Statistics for a single arm (trunk or branch_N)."""

    name: str
    core: list[float]
    core_inv_ppl: list[float]
    # E[∂|B] - deviance relative to this arm's core
    deviance_avg: float
    deviance_avg_inv_ppl: float
    # E[∂|T] - deviance relative to trunk core
    deviance_avg_trunk: float
    deviance_avg_trunk_inv_ppl: float
    # E[Δ∂] = E[∂|B - ∂|T] - expected per-trajectory deviance difference
    deviance_delta: float
    deviance_delta_inv_ppl: float
    # E[θ|T] - orientation relative to trunk core
    orientation_avg: list[float]
    orientation_avg_inv_ppl: list[float]
    # ‖E[θ|T]‖ - distance between cores
    orientation_norm: float
    orientation_norm_inv_ppl: float
    n_trajectories: int

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int = 0) -> "ArmResult":
        """Create from estimation output dict."""
        trajectories = data.get("trajectories", [])
        n_traj = len(trajectories)

        return cls(
            name=data.get("name", f"arm_{index}"),
            core=data["core"],
            core_inv_ppl=data.get("core_inv_ppl", []),
            deviance_avg=data["deviance_avg"],
            deviance_avg_inv_ppl=data.get("deviance_avg_inv_ppl", 0.0),
            deviance_avg_trunk=data.get("deviance_avg_trunk", 0.0),
            deviance_avg_trunk_inv_ppl=data.get("deviance_avg_trunk_inv_ppl", 0.0),
            deviance_delta=data.get("deviance_delta", 0.0),
            deviance_delta_inv_ppl=data.get("deviance_delta_inv_ppl", 0.0),
            orientation_avg=data.get("orientation_avg", []),
            orientation_avg_inv_ppl=data.get("orientation_avg_inv_ppl", []),
            orientation_norm=data.get("orientation_norm", 0.0),
            orientation_norm_inv_ppl=data.get("orientation_norm_inv_ppl", 0.0),
            n_trajectories=n_traj,
        )


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""

    method: GenerationMethod
    paths: OutputPaths
    arms: list[ArmResult] = field(default_factory=list)

    @property
    def n_trajectories(self) -> int:
        """Total trajectories (trunk count includes all)."""
        return self.arms[0].n_trajectories if self.arms else 0

    @property
    def trunk(self) -> ArmResult:
        """Get trunk arm (index 0)."""
        return self.arms[0]

    @classmethod
    def from_estimation_file(
        cls,
        method: GenerationMethod,
        paths: OutputPaths,
    ) -> ExperimentResult:
        """Load result from estimation output file."""
        with open(paths.estimation) as f:
            est_data = json.load(f)

        arms = [ArmResult.from_dict(a, i) for i, a in enumerate(est_data["arms"])]
        return cls(method=method, paths=paths, arms=arms)


# ══════════════════════════════════════════════════════════════════════════════
# Display Utilities
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

    # Prompt (truncated)
    prompt = config.get("prompt", "")
    if len(prompt) > 70:
        prompt = prompt[:67] + "..."
    log(f"  Prompt:   {prompt}")

    # Trunk
    trunk = config.get("trunk", "")
    if trunk:
        if len(trunk) > 60:
            trunk = trunk[:57] + "..."
        log(f"  Trunk:    \"{trunk}\"")

    # Branches
    branches = config.get("branches", [])
    if branches and branches != ["trunk"]:
        branch_strs = [f"\"{b}\"" for b in branches if b != "trunk"]
        if branch_strs:
            log(f"  Branches: {', '.join(branch_strs)}")

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

    # Structures
    log("  Structures:")
    cat_judgements = judge_data.get("categorical_judgements", [])
    graded_judgements = judge_data.get("graded_judgements", [])
    similarity_scoring = judge_data.get("similarity_scoring", [])

    def truncate(s: str, max_len: int = 55) -> str:
        return s[:max_len-3] + "..." if len(s) > max_len else s

    cat_idx = 1
    for item in cat_judgements:
        if isinstance(item, list):
            log(f"    c{cat_idx}: [bundled: {len(item)}]")
            for q in item:
                log(f"        • {truncate(q)}")
        else:
            log(f"    c{cat_idx}: {truncate(item)}")
        cat_idx += 1

    graded_idx = 1
    for item in graded_judgements:
        if isinstance(item, list):
            log(f"    g{graded_idx}: [bundled: {len(item)}]")
            for q in item:
                log(f"        • {truncate(q)}")
        else:
            log(f"    g{graded_idx}: {truncate(item)}")
        graded_idx += 1

    sim_idx = 1
    for item in similarity_scoring:
        if isinstance(item, list):
            log(f"    s{sim_idx}: sim({', '.join(item)})")
        else:
            log(f"    s{sim_idx}: sim(\"{item}\")")
        sim_idx += 1

    log("")

    # Build structure labels
    n_cat = len(cat_judgements)
    n_graded = len(graded_judgements)
    n_sim = len(similarity_scoring)
    labels = (
        [f"c{i+1}" for i in range(n_cat)]
        + [f"g{i+1}" for i in range(n_graded)]
        + [f"s{i+1}" for i in range(n_sim)]
    )

    # Column width for values
    col_w = 7

    def fmt_header(lbls: list[str]) -> str:
        """Format header row with centered labels."""
        return "".join(f"{lbl:^{col_w}}" for lbl in lbls)

    def fmt_values(values: list[float]) -> str:
        """Format values row with centered numbers."""
        return "".join(f"{v:^{col_w}.2f}" for v in values)

    # Results summary - read pre-computed values from estimation output
    arms = est_data.get("arms", [])

    if arms:
        log_section_title("RESULTS")
        log_divider(width=76)

        # Legend for expectations
        log("  Legend:")
        log("    E[∂]          = E[∂|branch] = avg spread around this arm's core")
        log("    core          = expected compliance per structure (the normative center)")
        log("    E[θ|trunk]    = how this arm's core differs from trunk (direction)")
        log("    ‖E[θ|trunk]‖  = distance between this arm's core and trunk core (scalar)")
        log("    E[∂|trunk]    = avg spread around trunk's core")
        log("    E[Δ∂]         = E[∂|branch] - E[∂|trunk] = deviance difference")
        log("")

        # Print header row once
        header_indent = "                    "
        log(f"{header_indent}{fmt_header(labels)}")
        table_width = len(header_indent) + col_w * len(labels)
        log(f"    {'─' * (table_width - 4)}")

        for idx, arm in enumerate(arms):
            display_name = "TRUNK" if idx == 0 else f"BRANCH_{idx}"
            trajectories = arm.get("trajectories", [])
            n_traj = len(trajectories)

            # Prob-weighted (pre-computed in estimation)
            core = arm.get("core", [])
            dev_branch = arm.get("deviance_avg", 0.0)
            dev_trunk = arm.get("deviance_avg_trunk", 0.0)
            dev_delta = arm.get("deviance_delta", 0.0)
            orient_avg = arm.get("orientation_avg", [])
            orient_norm = arm.get("orientation_norm", 0.0)

            # Inv-ppl weighted (pre-computed in estimation)
            core_inv = arm.get("core_inv_ppl", [])
            dev_branch_inv = arm.get("deviance_avg_inv_ppl", 0.0)
            dev_trunk_inv = arm.get("deviance_avg_trunk_inv_ppl", 0.0)
            dev_delta_inv = arm.get("deviance_delta_inv_ppl", 0.0)
            orient_avg_inv = arm.get("orientation_avg_inv_ppl", [])
            orient_norm_inv = arm.get("orientation_norm_inv_ppl", 0.0)

            log(f"  {display_name} ({n_traj} trajectories)")
            log(f"    [prob-weighted]")
            log(f"      E[∂]          = {dev_branch:.4f}")
            log(f"      core            {fmt_values(core)}")
            if orient_avg:
                log(f"      E[θ|trunk]      {fmt_values(orient_avg)}")
            log(f"      ‖E[θ|trunk]‖  = {orient_norm:.4f}")
            log(f"      E[∂|trunk]    = {dev_trunk:.4f}")
            log(f"      E[∂|branch]   = {dev_branch:.4f}")
            log(f"      E[Δ∂]         = {dev_delta:.4f}")
            if core_inv:
                log(f"    [inv-ppl-weighted]")
                log(f"      E[∂]          = {dev_branch_inv:.4f}")
                log(f"      core            {fmt_values(core_inv)}")
                if orient_avg_inv:
                    log(f"      E[θ|trunk]      {fmt_values(orient_avg_inv)}")
                log(f"      ‖E[θ|trunk]‖  = {orient_norm_inv:.4f}")
                log(f"      E[∂|trunk]    = {dev_trunk_inv:.4f}")
                log(f"      E[∂|branch]   = {dev_branch_inv:.4f}")
                log(f"      E[Δ∂]         = {dev_delta_inv:.4f}")
            log("")


def log_arm_table(
    results: list[ExperimentResult],
    arm_names: list[str],
    core_label: str,
    get_core: Callable[[ArmResult], list[float]],
    get_deviance: Callable[[ArmResult], float],
) -> None:
    """Log a comparison table for all arms."""
    for idx, arm_name in enumerate(arm_names):
        # Use generic labels: TRUNK for index 0, BRANCH_N for others
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
                core_str = fmt_core(get_core(arm))
                log(
                    f"  {r.method:<18}  {arm.n_trajectories:>4}  "
                    f"{get_deviance(arm):>8.4f}  {core_str}"
                )


def display_comparison(results: list[ExperimentResult], gap: int = 2) -> None:
    """Display side-by-side comparison of estimation results."""
    from .script_utils import log_major, STAGE_GAP

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

    # Probability-weighted comparison
    log_section_title("PROB-WEIGHTED")
    log_divider(width=76)

    log_arm_table(
        results,
        all_arm_names,
        "Core (prob-weighted)",
        lambda a: a.core,
        lambda a: a.deviance_avg,
    )

    # Inv-perplexity weighted comparison
    has_inv_ppl = any(a.core_inv_ppl for r in results for a in r.arms)
    if has_inv_ppl:
        log_section_title("INV-PERPLEXITY WEIGHTED")
        log_divider(width=76)

        log_arm_table(
            results,
            all_arm_names,
            "Core (inv-ppl-weighted)",
            lambda a: a.core_inv_ppl,
            lambda a: a.deviance_avg_inv_ppl,
        )

    log("")
