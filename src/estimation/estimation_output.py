"""Official output format for estimation results.

EstimationOutput is the canonical, versioned output format for estimation.
All fields are organized into clear sections for machine and human consumption.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.common.base_schema import BaseSchema

from .estimation_scoring_result import ArmScoring, StructureInfo
from .estimation_structure import ArmEstimate

# ══════════════════════════════════════════════════════════════════════════════
# METADATA
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class EstimationMetadata(BaseSchema):
    """Metadata about the estimation run."""

    version: str  # Output format version
    estimated_at: str  # ISO timestamp

    # Source files
    generation_file: str
    scoring_file: str
    judgment_file: str

    # Models used
    judge_model: str
    embedding_model: str


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT CLASS
# ══════════════════════════════════════════════════════════════════════════════


OUTPUT_VERSION = "2.0"


@dataclass
class EstimationOutput(BaseSchema):
    """Official output format for estimation results.

    Sections:
        metadata: Run metadata (version, timestamp, source files, models)
        structures: Structure definitions (labels, descriptions, bundling)
        arms: Per-arm estimates (cores, deviance, orientation)
        arm_scoring: Per-arm compliance rates (aggregate scores)

    Output path: out/<method>/<gen_name>/<scoring_name>/estimation.json
    """

    # === METADATA ===
    metadata: EstimationMetadata

    # === STRUCTURE DEFINITIONS ===
    structures: list[StructureInfo]

    # === CORE RESULTS ===
    arms: list[ArmEstimate]

    # === COMPLIANCE RATES ===
    arm_scoring: list[ArmScoring] = field(default_factory=list)

    # ──────────────────────────────────────────────────────────────────────────
    # Factory
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        *,
        judgment_file: str,
        generation_file: str,
        scoring_file: str,
        judge_model: str,
        embedding_model: str,
        structures: list[StructureInfo],
        arms: list[ArmEstimate],
        arm_scoring: list[ArmScoring],
    ) -> EstimationOutput:
        """Create estimation output with metadata auto-populated."""
        metadata = EstimationMetadata(
            version=OUTPUT_VERSION,
            estimated_at=datetime.now().isoformat(),
            generation_file=generation_file,
            scoring_file=scoring_file,
            judgment_file=judgment_file,
            judge_model=judge_model,
            embedding_model=embedding_model,
        )

        return cls(
            metadata=metadata,
            structures=structures,
            arms=arms,
            arm_scoring=arm_scoring,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Accessors
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def structure_labels(self) -> list[str]:
        """Get structure labels in order."""
        return [s.label for s in self.structures]

    @property
    def arm_names(self) -> list[str]:
        """Get arm names in order."""
        return [a.name for a in self.arms]

    def get_arm(self, name: str) -> ArmEstimate | None:
        """Get arm by name."""
        for arm in self.arms:
            if arm.name == name:
                return arm
        return None

    def get_structure(self, label: str) -> StructureInfo | None:
        """Get structure by label."""
        for s in self.structures:
            if s.label == label:
                return s
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        """Save to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> EstimationOutput:
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    # ──────────────────────────────────────────────────────────────────────────
    # Path Computation
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_output_path(judgment_path: Path) -> Path:
        """Compute output path from judgment path.

        Pattern: out/<method>/<gen_name>/<scoring_name>/estimation.json
        """
        return judgment_path.parent / "estimation.json"

    @staticmethod
    def compute_summary_path(judgment_path: Path) -> Path:
        """Compute summary text file path.

        Pattern: out/<method>/<gen_name>/<scoring_name>/summary_estimation.txt
        """
        return judgment_path.parent / "summary_estimation.txt"

    # ──────────────────────────────────────────────────────────────────────────
    # Summary (convenience methods delegating to standalone functions)
    # ──────────────────────────────────────────────────────────────────────────

    def save_summary(self, path: str | Path) -> Path:
        """Save human-readable summary to text file."""
        return save_estimation_summary(self, path)

    def summarize(self) -> None:
        """Print summary to console."""
        print_estimation_summary(self)


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY GENERATION (separated from output class)
# ══════════════════════════════════════════════════════════════════════════════


def save_estimation_summary(output: EstimationOutput, path: str | Path) -> Path:
    """Save human-readable summary to text file."""
    from .weighting_method_registry import get_method_description, iter_methods

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    labels = output.structure_labels
    col_w = 8
    lines = []

    # Header
    lines.append("=" * 76)
    lines.append("  ESTIMATION SUMMARY")
    lines.append("=" * 76)
    lines.append("")

    # Metadata
    lines.append(f"  Version:   {output.metadata.version}")
    lines.append(f"  Generated: {output.metadata.estimated_at}")
    lines.append(f"  Judge:     {output.metadata.judge_model}")
    lines.append(f"  Embed:     {output.metadata.embedding_model}")
    lines.append("")

    # Structures
    lines.append("-" * 76)
    lines.append("  STRUCTURES")
    lines.append("-" * 76)
    for s in output.structures:
        if s.is_bundled:
            lines.append(f"  {s.label}: [BUNDLED]")
            for q in s.questions:
                lines.append(f"      - {q}")
        else:
            lines.append(f"  {s.label}: {s.description}")
    lines.append("")

    # Results by weighting method
    lines.append("-" * 76)
    lines.append("  CORES BY WEIGHTING METHOD")
    lines.append("-" * 76)

    for method_name, _, _ in iter_methods():
        desc = get_method_description(method_name)
        lines.append(f"  [{desc}]")

        header = "".join(f"{label:^{col_w}}" for label in labels)
        lines.append(f"  {'Arm':<14} {'N':>4}  {'E[d]':>7}  {header}")
        lines.append("  " + "-" * 70)

        for arm in output.arms:
            est = arm.estimates.get(method_name)
            if est and est.core:
                core_str = "".join(f"{c:^{col_w}.3f}" for c in est.core)
                lines.append(
                    f"  {arm.name:<14} {len(arm.trajectories):>4}  "
                    f"{est.deviance_avg:>7.4f}  {core_str}"
                )
        lines.append("")

    lines.append("=" * 76)

    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def print_estimation_summary(output: EstimationOutput) -> None:
    """Print summary to console."""
    from src.common.logging import log_banner

    from .logging.estimation_display_utils import (
        log_arm_cores,
        log_compliance_rates,
        log_structures,
    )

    labels = output.structure_labels

    log_banner("STRUCTURES")
    log_structures(output.structures)

    log_banner("SCORES BY ARM")
    log_compliance_rates(output.arm_scoring, labels)

    log_banner("CORES BY ARM")
    log_arm_cores(output.arms, labels, show_variants=True)

    log_banner("")
