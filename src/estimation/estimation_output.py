"""Output classes for estimation results.

This module defines the main output data structures for estimation results,
including both machine-readable and human-readable summary formats.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.common.base_schema import BaseSchema
from src.common.log import log
from src.common.log_utils import log_banner, log_divider

from .estimation_auxiliary_types import (
    ArmSummary,
    ContinuationsByArm,
    EstimationSummary,
    TrajectoryGrouping,
)
from .estimation_scoring_data import ScoringItem
from .estimation_scoring_result import ArmScoring, StructureInfo
from .estimation_structure import ArmEstimate
from .logging.estimation_display_utils import (
    log_arm_cores,
    log_compliance_rates,
    log_structures,
)
from .weighting_method_registry import get_method_description, iter_methods


@dataclass
class EstimationSummaryOutput(BaseSchema):
    """Human-readable summary output saved separately."""

    generation_file: str
    scoring_file: str
    judgment_file: str
    judge_model: str
    embedding_model: str
    estimated_at: str
    structures: list[dict[str, Any]]  # [{label, description, is_bundled, questions}]
    arm_scoring: list[
        dict[str, Any]
    ]  # [{branch, rates: {label: rate}, question_rates}]
    branch_cores: list[
        dict[str, Any]
    ]  # [{branch, prob_weighted: {label: val}, inv_ppl: {label: val}}]
    continuations_by_arm: dict[str, list[dict[str, Any]]]  # {branch: [{idx, text}]}


@dataclass
class EstimationOutput(BaseSchema):
    """Output from normativity estimation."""

    summary: EstimationSummary
    scoring_data: dict[str, list[ScoringItem]]  # Generic storage for all methods
    arms: list[ArmEstimate]  # trunk + branches
    judgment_file: str
    estimated_at: str

    # Additional metadata for summary
    generation_file: str = ""
    scoring_file: str = ""
    judge_model: str = ""
    embedding_model: str = ""
    structure_info: list[StructureInfo] = field(default_factory=list)
    arm_scoring: list[ArmScoring] = field(default_factory=list)
    continuations_by_arm: ContinuationsByArm = field(default_factory=ContinuationsByArm)

    @classmethod
    def create(
        cls,
        judgment_file: str,
        scoring_data: dict[str, list[ScoringItem]],
        arms: list[ArmEstimate],
        texts: dict[int, str],
        generation_file: str = "",
        scoring_file: str = "",
        judge_model: str = "",
        embedding_model: str = "",
        structure_info: list[StructureInfo] | None = None,
        arm_scoring: list[ArmScoring] | None = None,
        continuations_by_arm: ContinuationsByArm | None = None,
    ) -> EstimationOutput:
        """Create estimation output with auto-generated summary."""
        # Build trajectory -> arms mapping
        traj_to_arms: dict[int, list[int]] = {}
        for arm in arms:
            for traj in arm.trajectories:
                traj_to_arms.setdefault(traj.traj_idx, []).append(arm.arm_idx)

        summary = EstimationSummary(
            trajectories=[
                TrajectoryGrouping(
                    traj_idx=idx, arm_idxs=aids, continuation_text=texts.get(idx, "")
                )
                for idx, aids in sorted(traj_to_arms.items())
            ],
            arms=[ArmSummary(a.arm_idx, a.name, len(a.trajectories)) for a in arms],
        )

        return cls(
            summary=summary,
            scoring_data=scoring_data,
            arms=arms,
            judgment_file=judgment_file,
            estimated_at=datetime.now().isoformat(),
            generation_file=generation_file,
            scoring_file=scoring_file,
            judge_model=judge_model,
            embedding_model=embedding_model,
            structure_info=structure_info or [],
            arm_scoring=arm_scoring or [],
            continuations_by_arm=continuations_by_arm or ContinuationsByArm(),
        )

    def save(self, path: str | Path) -> Path:
        """Save output to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @staticmethod
    def compute_output_path(judgment_path: Path) -> Path:
        """Compute the output path for estimation results."""
        name = judgment_path.stem.replace("score_", "")
        return Path("out") / f"est_{name}.json"

    @staticmethod
    def compute_summary_path(judgment_path: Path) -> Path:
        """Compute the output path for summary results."""
        name = judgment_path.stem.replace("score_", "")
        return Path("out") / f"summary_est_{name}.txt"

    def get_structure_labels(self) -> list[str]:
        """Get structure labels from structure_info."""
        return [s.label for s in self.structure_info]

    def save_summary(self, path: str | Path) -> Path:
        """Save human-readable summary to text file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        labels = self.get_structure_labels()
        lines = []

        # Header
        lines.append("=" * 76)
        lines.append("  ESTIMATION SUMMARY")
        lines.append("=" * 76)
        lines.append("")

        # Metadata
        lines.append(f"  Generated: {self.estimated_at}")
        lines.append(f"  Judge:     {self.judge_model}")
        lines.append(f"  Embed:     {self.embedding_model}")
        lines.append("")

        # Structures
        lines.append("-" * 76)
        lines.append("  STRUCTURES")
        lines.append("-" * 76)
        for s in self.structure_info:
            if s.is_bundled:
                lines.append(f"  {s.label}: [BUNDLED]")
                for q in s.questions:
                    lines.append(f"      • {q}")
            else:
                lines.append(f"  {s.label}: {s.description}")
        lines.append("")

        # Results table
        col_w = 8
        header = "  " + "".join(f"{l:^{col_w}}" for l in labels)

        lines.append("-" * 76)
        lines.append("  RESULTS")
        lines.append("-" * 76)

        # Iterate over all registered weighting methods
        for method_name, _, _ in iter_methods():
            desc = get_method_description(method_name)
            lines.append(f"  [{desc}]")
            lines.append(f"  {'Arm':<14} {'N':>4}  {'E[∂]':>7}  {header.strip()}")
            lines.append("  " + "-" * 70)

            for arm in self.arms:
                est = arm.estimates.get(method_name)
                if est and est.core:
                    core_str = "".join(f"{c:^{col_w}.3f}" for c in est.core)
                    lines.append(
                        f"  {arm.name:<14} {len(arm.trajectories):>4}  {est.deviance_avg:>7.4f}  {core_str}"
                    )
            lines.append("")

        lines.append("=" * 76)

        with open(path, "w") as f:
            f.write("\n".join(lines))
        return path

    def summarize(self, show_variants: bool = True) -> None:
        """Print summary statistics."""
        labels = self.get_structure_labels()

        # Show structure legend
        log_banner("STRUCTURES")
        log_structures(self.structure_info)

        # Show per-branch rates
        log_banner(
            "COMPLIANCE RATES BY BRANCH (% yes for categorical, avg for similarity)"
        )
        log_compliance_rates(self.arm_scoring, labels)

        # Show cores with labels
        log_banner("CORES BY ARM")
        log_arm_cores(self.arms, labels, show_variants)

        # Trajectory counts by branch
        if self.continuations_by_arm:
            log_banner("TRAJECTORY COUNTS BY BRANCH")
            for branch, items in self.continuations_by_arm.items():
                log(f"  {branch}: {len(items)} trajectories")

        log_banner("")
