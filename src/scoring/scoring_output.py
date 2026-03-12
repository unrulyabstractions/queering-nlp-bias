"""Output types for trajectory scoring.

This module defines ScoringResult and ScoringOutput with generic storage
that works with any registered scoring method without modification.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.common.base_schema import BaseSchema
from src.common.logging import log, log_banner, log_sub_banner
from src.common.viz_utils import preview

from .scoring_data import TrajectoryData
from .scoring_method_registry import iter_methods


@dataclass
class ScoringResult(BaseSchema):
    """Result of scoring a single trajectory - generic storage."""

    trajectory_idx: int
    branch: str
    branch_idx: int
    text: str
    conditional_logprobs: dict[str, float]
    n_continuation_tokens: int

    # Generic storage: method_name -> list of scores
    method_scores: dict[str, list[Any]] = field(default_factory=dict)

    # Generic storage: method_name -> list of raw responses
    method_raw: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_trajectory(
        cls,
        traj: TrajectoryData,
        scores: Any,  # TrajectoryScores from pipeline
    ) -> ScoringResult:
        """Create a ScoringResult from a TrajectoryData and scores."""
        method_scores = {}
        method_raw = {}

        for method_name, (sc, raw) in scores.method_scores.items():
            method_scores[method_name] = sc
            method_raw[method_name] = raw

        return cls(
            trajectory_idx=traj.idx,
            branch=traj.arm_name,
            branch_idx=traj.arm_idx,
            text=traj.generated_text,
            conditional_logprobs=traj.conditional_logprobs,
            n_continuation_tokens=traj.n_continuation_tokens,
            method_scores=method_scores,
            method_raw=method_raw,
        )


@dataclass
class ScoringOutput(BaseSchema):
    """Output from trajectory scoring - generic storage."""

    generation_file: str
    scoring_file: str
    judge_model: str
    embedding_model: str = ""

    # Generic storage: config_key -> list of items
    scoring_data: dict[str, list[str | list[str]]] = field(default_factory=dict)

    arm_names: list[str] = field(default_factory=list)
    arm_texts: dict[str, str] = field(default_factory=dict)
    scored_at: str = ""
    num_results: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        generation_file: str,
        scoring_file: str,
        scoring_config: Any,  # ScoringConfig
        results: list[ScoringResult],
        arm_names: list[str],
        arm_texts: dict[str, str],
    ) -> ScoringOutput:
        """Create scoring output from results."""
        return cls(
            generation_file=generation_file,
            scoring_file=scoring_file,
            judge_model=scoring_config.model,
            embedding_model=scoring_config.embedding_model,
            scoring_data=scoring_config.scoring_data,
            arm_names=arm_names,
            arm_texts=arm_texts,
            scored_at=datetime.now().isoformat(),
            num_results=len(results),
            results=[r.to_dict() for r in results],
        )

    def save(self, path: str | Path, config_path: str | Path | None = None) -> Path:
        """Save output to JSON file.

        Args:
            path: Output path for scoring.json
            config_path: Original config file to copy as scoring_cfg.json
        """
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

    @staticmethod
    def compute_output_path(gen_path: str | Path, scoring_path: str | Path) -> Path:
        """Compute the output path for judgment results.

        Output structure: out/<method>/<gen_name>/<scoring_name>/scoring.json
        Method and gen_name extracted from gen_path.
        """
        gen_path = Path(gen_path)
        scoring_path = Path(scoring_path)
        method = gen_path.parent.parent.name  # out/<method>/<gen_name>/generation.json
        gen_name = gen_path.parent.name
        scoring_name = scoring_path.stem
        return Path("out") / method / gen_name / scoring_name / "scoring.json"

    @staticmethod
    def compute_summary_path(gen_path: str | Path, scoring_path: str | Path) -> Path:
        """Compute the output path for scoring summary.

        Output structure: out/<method>/<gen_name>/<scoring_name>/score_summary.txt
        """
        gen_path = Path(gen_path)
        scoring_path = Path(scoring_path)
        method = gen_path.parent.parent.name
        gen_name = gen_path.parent.name
        scoring_name = scoring_path.stem
        return Path("out") / method / gen_name / scoring_name / "score_summary.txt"

    def _get_active_methods(self) -> list[tuple[str, str, str]]:
        """Get active methods with their metadata.

        Returns:
            List of (method_name, config_key, label_prefix) tuples
        """
        active = []
        for method_name, params_class, _ in iter_methods():
            config_key = params_class.config_key
            if config_key in self.scoring_data and self.scoring_data[config_key]:
                active.append((method_name, config_key, params_class.label_prefix))
        return active

    def _get_structure_labels(self) -> list[str]:
        """Get structure labels for all active methods."""
        labels = []
        for method_name, config_key, label_prefix in self._get_active_methods():
            items = self.scoring_data.get(config_key, [])
            for i in range(len(items)):
                labels.append(f"{label_prefix}{i + 1}")
        return labels

    def _group_by_branch(self) -> dict[str, list[dict]]:
        """Group results by branch name."""
        by_branch: dict[str, list[dict]] = {}
        for r in self.results:
            branch = r.get("branch", "trunk")
            by_branch.setdefault(branch, []).append(r)
        return by_branch

    def _collect_scores_at_index(
        self,
        results: list[dict],
        method_name: str,
        flat_idx: int,
    ) -> list[Any]:
        """Collect valid scores at a specific flat index from results."""
        values = []
        for r in results:
            method_scores = r.get("method_scores", {})
            scores = method_scores.get(method_name, [])
            if flat_idx < len(scores) and scores[flat_idx] is not None:
                values.append(scores[flat_idx])
        return values

    def _compute_branch_rates(
        self, branch_results: list[dict], labels: list[str]
    ) -> list[float]:
        """Compute rates for each structure label given branch results."""
        rates = []

        for method_name, config_key, _ in self._get_active_methods():
            items = self.scoring_data.get(config_key, [])
            flat_idx = 0

            for item in items:
                if isinstance(item, list):
                    # Bundled - average all sub-items
                    all_values = []
                    for sub_idx in range(len(item)):
                        values = self._collect_scores_at_index(
                            branch_results, method_name, flat_idx + sub_idx
                        )
                        all_values.extend(values)
                    rates.append(sum(all_values) / len(all_values) if all_values else 0.0)
                    flat_idx += len(item)
                else:
                    values = self._collect_scores_at_index(
                        branch_results, method_name, flat_idx
                    )
                    rates.append(sum(values) / len(values) if values else 0.0)
                    flat_idx += 1

        return rates

    def save_summary(self, path: str | Path) -> Path:
        """Save human-readable summary to text file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        labels = self._get_structure_labels()

        lines = []
        lines.append("=" * 76)
        lines.append("  SCORING SUMMARY")
        lines.append("=" * 76)
        lines.append("")
        lines.append(f"  Judge:       {self.judge_model}")
        lines.append(f"  Embed:       {self.embedding_model}")
        lines.append(f"  Scored:      {self.scored_at}")
        lines.append(f"  Trajectories:{self.num_results}")
        lines.append("")

        by_branch = self._group_by_branch()

        # Per-branch rates table
        lines.append("-" * 76)
        lines.append("  RATES BY BRANCH")
        lines.append("-" * 76)
        col_w = 8
        header = "".join(f"{l:^{col_w}}" for l in labels)
        lines.append(f"  {'Branch':<14} {'N':>4}  {header}")
        lines.append("  " + "-" * 70)

        for branch_name in self.arm_names:
            branch_results = by_branch.get(branch_name, [])
            if not branch_results:
                continue
            rates = self._compute_branch_rates(branch_results, labels)
            rate_str = "".join(f"{r:^{col_w}.3f}" for r in rates)
            lines.append(f"  {branch_name:<14} {len(branch_results):>4}  {rate_str}")

        lines.append("")
        lines.append("=" * 76)

        with open(path, "w") as f:
            f.write("\n".join(lines))
        return path

    def summarize(self) -> None:
        """Print clean summary statistics for all scoring methods."""
        log_banner("SCORING SUMMARY")
        log("\nSettings:")
        log(f"  Judge model: {self.judge_model}")
        log(f"  Generation file: {self.generation_file}")
        log(f"  Trajectories scored: {self.num_results}")

        by_branch = self._group_by_branch()

        # Summarize each active method
        for method_name, config_key, label_prefix in self._get_active_methods():
            items = self.scoring_data.get(config_key, [])
            if not items:
                continue

            log_sub_banner(f"{method_name.upper()} SCORES")
            self._log_method_scores(method_name, config_key, label_prefix, items, by_branch)

        log_banner("")

    def _log_method_scores(
        self,
        method_name: str,
        config_key: str,
        label_prefix: str,
        items: list[str | list[str]],
        by_branch: dict[str, list[dict]],
    ) -> None:
        """Log scores for a single method."""
        flat_idx = 0
        for struct_idx, item in enumerate(items):
            if isinstance(item, list):
                self._log_bundled_item_scores(
                    method_name, label_prefix, struct_idx, item, flat_idx
                )
                flat_idx += len(item)
            else:
                self._log_single_item_scores(
                    method_name, label_prefix, struct_idx, item, flat_idx
                )
                flat_idx += 1

    def _log_bundled_item_scores(
        self,
        method_name: str,
        label_prefix: str,
        struct_idx: int,
        sub_items: list[str],
        flat_idx: int,
    ) -> None:
        """Log scores for a bundled item (list of sub-items)."""
        log(f"\n  [{label_prefix}{struct_idx + 1}] Bundled ({len(sub_items)} items):")
        group_values = []

        for sub_idx, sub_item in enumerate(sub_items):
            values = self._collect_scores_at_index(
                self.results, method_name, flat_idx + sub_idx
            )
            if values:
                avg = sum(values) / len(values)
                group_values.append(avg)
                log(f"      * {preview(sub_item, 40)}: avg={avg:.4f}")

        if group_values:
            group_avg = sum(group_values) / len(group_values)
            log(f"      -> group avg={group_avg:.4f}")

    def _log_single_item_scores(
        self,
        method_name: str,
        label_prefix: str,
        struct_idx: int,
        item: str,
        flat_idx: int,
    ) -> None:
        """Log scores for a single item."""
        values = self._collect_scores_at_index(self.results, method_name, flat_idx)
        if values:
            avg = sum(values) / len(values)
            log(f"\n  [{label_prefix}{struct_idx + 1}] {preview(item, 50)}")
            log(f"      -> avg={avg:.4f}")
