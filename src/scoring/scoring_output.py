"""Official output format for scoring results.

ScoringOutput is the canonical, versioned output format for trajectory scoring.
All fields are organized into clear sections for machine and human consumption.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.common.base_schema import BaseSchema
from src.common.output_paths import scoring_output_path, scoring_summary_path
from src.common.result_grouping import group_results_by_arm

from .scoring_data import TrajectoryData
from .scoring_method_registry import iter_methods


# ══════════════════════════════════════════════════════════════════════════════
# METADATA
# ══════════════════════════════════════════════════════════════════════════════


OUTPUT_VERSION = "2.0"


@dataclass
class ScoringMetadata(BaseSchema):
    """Metadata about the scoring run."""

    version: str  # Output format version
    scored_at: str  # ISO timestamp

    # Source files
    generation_file: str
    scoring_file: str

    # Models used
    judge_model: str
    embedding_model: str

    # Counts
    num_trajectories: int


# ══════════════════════════════════════════════════════════════════════════════
# RESULT TYPES
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ScoringResult(BaseSchema):
    """Result of scoring a single trajectory."""

    traj_idx: int
    arm: str
    arm_idx: int
    text: str
    n_generated_tokens: int
    conditional_logprobs: dict[str, float]

    # Scores by method: method_name -> list of scores (one per structure)
    method_scores: dict[str, list[Any]] = field(default_factory=dict)

    # Raw responses by method: method_name -> list of raw LLM responses
    method_raw: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_method_scores(
        cls,
        traj: TrajectoryData,
        scores: dict[str, tuple[list[Any], list[str]]],
    ) -> ScoringResult:
        """Create from TrajectoryData and method scores dict."""
        method_scores = {}
        method_raw = {}

        for method_name, (sc, raw) in scores.items():
            method_scores[method_name] = sc
            method_raw[method_name] = raw

        return cls(
            traj_idx=traj.idx,
            arm=traj.arm_name,
            arm_idx=traj.arm_idx,
            text=traj.generated_text,
            n_generated_tokens=traj.n_generated_tokens,
            conditional_logprobs=traj.conditional_logprobs,
            method_scores=method_scores,
            method_raw=method_raw,
        )


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT CLASS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ScoringOutput(BaseSchema):
    """Official output format for scoring results.

    Sections:
        metadata: Run metadata (version, timestamp, source files, models)
        scoring_items: Structure definitions by method
        arms: Arm names and prefill texts
        results: Per-trajectory scoring results

    Output path: out/<method>/<gen_name>/<scoring_name>/scoring.json
    """

    # === METADATA ===
    metadata: ScoringMetadata

    # === STRUCTURE DEFINITIONS ===
    # Config key -> list of items (strings or bundles)
    scoring_items: dict[str, list[str | list[str]]]

    # === ARM INFO ===
    arm_names: list[str]
    arm_texts: dict[str, str]  # arm_name -> prefill text

    # === RESULTS ===
    results: list[dict[str, Any]]  # ScoringResult.to_dict() for each trajectory

    # ──────────────────────────────────────────────────────────────────────────
    # Factory
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        *,
        generation_file: str,
        scoring_file: str,
        scoring_config: Any,  # ScoringConfig
        results: list[ScoringResult],
        arm_names: list[str],
        arm_texts: dict[str, str],
    ) -> ScoringOutput:
        """Create scoring output with metadata auto-populated."""
        metadata = ScoringMetadata(
            version=OUTPUT_VERSION,
            scored_at=datetime.now().isoformat(),
            generation_file=generation_file,
            scoring_file=scoring_file,
            judge_model=scoring_config.model,
            embedding_model=scoring_config.embedding_model,
            num_trajectories=len(results),
        )

        return cls(
            metadata=metadata,
            scoring_items=scoring_config.scoring_data,
            arm_names=arm_names,
            arm_texts=arm_texts,
            results=[r.to_dict() for r in results],
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, path: str | Path, config_path: str | Path | None = None) -> Path:
        """Save to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

        # Copy original config if provided
        if config_path:
            import shutil
            cfg_dest = path.parent / "scoring_cfg.json"
            shutil.copy(config_path, cfg_dest)

        return path

    @classmethod
    def load(cls, path: str | Path) -> ScoringOutput:
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    # ──────────────────────────────────────────────────────────────────────────
    # Path Computation (delegates to centralized output_paths module)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_output_path(gen_path: str | Path, scoring_path: str | Path) -> Path:
        """Compute output path from generation and scoring paths."""
        return scoring_output_path(Path(gen_path), Path(scoring_path))

    @staticmethod
    def compute_summary_path(gen_path: str | Path, scoring_path: str | Path) -> Path:
        """Compute summary text file path."""
        return scoring_summary_path(Path(gen_path), Path(scoring_path))

    # ──────────────────────────────────────────────────────────────────────────
    # Summary (convenience methods delegating to standalone functions)
    # ──────────────────────────────────────────────────────────────────────────

    def save_summary(self, path: str | Path) -> Path:
        """Save human-readable summary to text file."""
        return save_scoring_summary(self, path)

    def summarize(self) -> None:
        """Print summary to console."""
        print_scoring_summary(self)

    # ──────────────────────────────────────────────────────────────────────────
    # Analysis Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def group_by_arm(self) -> dict[str, list[dict]]:
        """Group results by arm name."""
        return group_results_by_arm(self.results)

    def get_active_methods(self) -> list[tuple[str, str, str]]:
        """Get active methods: (method_name, config_key, label_prefix)."""
        active = []
        for method_name, params_class, _ in iter_methods():
            config_key = params_class.config_key
            if self.scoring_items.get(config_key):
                active.append((method_name, config_key, params_class.label_prefix))
        return active

    def get_structure_labels(self) -> list[str]:
        """Get structure labels for all active methods."""
        labels = []
        for _, config_key, label_prefix in self.get_active_methods():
            items = self.scoring_items.get(config_key, [])
            for i in range(len(items)):
                labels.append(f"{label_prefix}{i + 1}")
        return labels


# ══════════════════════════════════════════════════════════════════════════════
# SCORE EXTRACTION UTILITIES
# ══════════════════════════════════════════════════════════════════════════════


def get_score_at_index(
    result: dict[str, Any],
    method_name: str,
    struct_idx: int,
) -> Any | None:
    """Get score at a structure index from a result dict."""
    method_scores = result.get("method_scores", {})
    scores = method_scores.get(method_name, [])
    if struct_idx < len(scores):
        return scores[struct_idx]
    return None


def get_bundle_score_at_index(
    result: dict[str, Any],
    method_name: str,
    struct_idx: int,
    question_idx: int,
) -> float | None:
    """Get a specific question's score from a bundled structure."""
    score = get_score_at_index(result, method_name, struct_idx)
    if isinstance(score, list) and question_idx < len(score):
        return score[question_idx]
    return None


def collect_scores(
    results: list[dict[str, Any]],
    method_name: str,
    struct_idx: int,
) -> list[Any]:
    """Collect scores at a structure index across multiple results."""
    values = []
    for r in results:
        score = get_score_at_index(r, method_name, struct_idx)
        if score is not None:
            values.append(score)
    return values


def collect_single_scores(
    results: list[dict[str, Any]],
    method_name: str,
    struct_idx: int,
) -> list[float]:
    """Collect single (non-bundle) scores, converting to float."""
    values = []
    for r in results:
        score = get_score_at_index(r, method_name, struct_idx)
        if score is not None and not isinstance(score, list):
            values.append(float(score))
    return values


def collect_bundle_scores(
    results: list[dict[str, Any]],
    method_name: str,
    struct_idx: int,
    question_idx: int,
) -> list[float]:
    """Collect scores for a specific question within a bundle."""
    values = []
    for r in results:
        score = get_bundle_score_at_index(r, method_name, struct_idx, question_idx)
        if score is not None:
            values.append(float(score))
    return values


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY GENERATION
# ══════════════════════════════════════════════════════════════════════════════


def _compute_arm_stats(
    output: ScoringOutput,
    arm_results: list[dict],
) -> list[tuple[float, float, float, float]]:
    """Compute (mean, std, min, max) for each structure given arm results."""
    import math

    stats = []

    for method_name, config_key, _ in output.get_active_methods():
        items = output.scoring_items.get(config_key, [])

        for struct_idx, item in enumerate(items):
            if isinstance(item, list):
                all_values = []
                for q_idx in range(len(item)):
                    all_values.extend(
                        collect_bundle_scores(arm_results, method_name, struct_idx, q_idx)
                    )
                values = all_values
            else:
                values = collect_single_scores(arm_results, method_name, struct_idx)

            if values:
                mean = sum(values) / len(values)
                var = sum((v - mean) ** 2 for v in values) / len(values)
                std = math.sqrt(var)
                stats.append((mean, std, min(values), max(values)))
            else:
                stats.append((0.0, 0.0, 0.0, 0.0))

    return stats


def _truncate(s: str, n: int) -> str:
    """Truncate string with ellipsis if needed."""
    return s[:n-3] + "..." if len(s) > n else s


def save_scoring_summary(output: ScoringOutput, path: str | Path) -> Path:
    """Save human-readable summary to text file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    by_arm = output.group_by_arm()
    W = 80

    lines = []
    lines.append("=" * W)
    lines.append("SCORING SUMMARY")
    lines.append("=" * W)
    lines.append("")

    # Metadata
    lines.append(f"Version:      {output.metadata.version}")
    lines.append(f"Judge:        {output.metadata.judge_model}")
    lines.append(f"Embedder:     {output.metadata.embedding_model or '(none)'}")
    lines.append(f"Scored:       {output.metadata.scored_at[:19]}")
    lines.append(f"Trajectories: {output.metadata.num_trajectories}")
    lines.append(f"Arms:         {len(output.arm_names)}")
    lines.append("")

    # Structure definitions
    lines.append("-" * W)
    lines.append("STRUCTURES")
    lines.append("-" * W)

    for method_name, config_key, label_prefix in output.get_active_methods():
        items = output.scoring_items.get(config_key, [])
        lines.append(f"\n[{method_name}]")
        for i, item in enumerate(items):
            label = f"{label_prefix}{i+1}"
            if isinstance(item, list):
                lines.append(f"  {label}: (bundle of {len(item)})")
                for q in item:
                    lines.append(f"       - {_truncate(q, 60)}")
            else:
                lines.append(f"  {label}: {_truncate(item, 65)}")

    lines.append("")

    # Arms and prefills
    lines.append("-" * W)
    lines.append("ARMS")
    lines.append("-" * W)
    for arm_name in output.arm_names:
        prefill = output.arm_texts.get(arm_name, "").replace("\n", "\\n")
        n = len(by_arm.get(arm_name, []))
        lines.append(f"  {arm_name:<14} (n={n:>2})  \"{_truncate(prefill, 45)}\"")
    lines.append("")

    # Per-arm rates table with stats
    lines.append("-" * W)
    lines.append("RATES BY ARM (mean ± std)")
    lines.append("-" * W)

    labels = output.get_structure_labels()
    col_w = 12
    header = "".join(f"{label:^{col_w}}" for label in labels)
    lines.append(f"{'Arm':<14}  {header}")
    lines.append("-" * W)

    for arm_name in output.arm_names:
        arm_results = by_arm.get(arm_name, [])
        if not arm_results:
            continue
        stats = _compute_arm_stats(output, arm_results)
        parts = []
        for mean, std, _, _ in stats:
            if std < 0.01:
                parts.append(f"{mean:^{col_w}.3f}")
            else:
                parts.append(f"{mean:.2f}±{std:.2f}".center(col_w))
        lines.append(f"{arm_name:<14}  {''.join(parts)}")

    lines.append("")
    lines.append("=" * W)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def print_scoring_summary(output: ScoringOutput) -> None:
    """Print summary to console."""
    from src.common.logging import log, log_banner, log_sub_banner
    from src.common.viz_utils import preview

    log_banner("SCORING SUMMARY")
    log("\nSettings:")
    log(f"  Version: {output.metadata.version}")
    log(f"  Judge model: {output.metadata.judge_model}")
    log(f"  Generation file: {output.metadata.generation_file}")
    log(f"  Trajectories scored: {output.metadata.num_trajectories}")

    by_arm = output.group_by_arm()

    # Summarize each active method
    for method_name, config_key, label_prefix in output.get_active_methods():
        items = output.scoring_items.get(config_key, [])
        if not items:
            continue

        log_sub_banner(f"{method_name.upper()} SCORES")

        for struct_idx, item in enumerate(items):
            if isinstance(item, list):
                # Bundled item
                log(f"\n  [{label_prefix}{struct_idx + 1}] Bundled ({len(item)} items):")
                group_values = []
                for sub_idx, sub_item in enumerate(item):
                    sub_values = collect_bundle_scores(
                        output.results, method_name, struct_idx, sub_idx
                    )
                    if sub_values:
                        avg = sum(sub_values) / len(sub_values)
                        group_values.append(avg)
                        log(f"      * {preview(sub_item, 40)}: avg={avg:.4f}")
                if group_values:
                    group_avg = sum(group_values) / len(group_values)
                    log(f"      -> group avg={group_avg:.4f}")
            else:
                # Single item
                valid = collect_single_scores(output.results, method_name, struct_idx)
                if valid:
                    avg = sum(valid) / len(valid)
                    log(f"\n  [{label_prefix}{struct_idx + 1}] {preview(item, 50)}")
                    log(f"      -> avg={avg:.4f}")

    log_banner("")
