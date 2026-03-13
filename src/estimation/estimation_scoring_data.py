"""Scoring data loading and processing for estimation.

This module handles loading scoring output files and computing
structure scores and arm-level scoring from scoring results.

Works with the generic scoring format:
- scoring_data: dict mapping config_key to items list
- results[].method_scores: dict mapping method_name to scores list
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.common.base_schema import BaseSchema
from src.common.result_grouping import group_results_by_arm
from src.scoring.scoring_method_registry import iter_methods
from src.scoring.scoring_output import (
    collect_bundle_scores,
    collect_single_scores,
    get_score_at_index,
)

from .estimation_auxiliary_types import ContinuationsByArm
from .estimation_scoring_result import (
    ArmScoring,
    BundledScoreResult,
    ScoreComputation,
    StructureInfo,
    StructureScoresResult,
)
from .estimation_structure import TrajectoryScoringData

# Type alias for scoring items (single question or bundled questions)
ScoringItem = str | list[str]


@dataclass
class ScoringMethodConfig(BaseSchema):
    """Configuration for a single scoring method."""

    method_name: str
    items: list[ScoringItem]
    label_prefix: str  # "c", "g", "s", "o", etc.
    config_key: str  # Key in scoring_data dict
    struct_offset: int = (
        0  # Offset in structure list (cumulative items from prior methods)
    )

    def get_flat_index(self, struct_idx_within_method: int, question_idx: int) -> int:
        """Get the flat index into the scores array for a structure/question."""
        flat_idx = 0
        for i, item in enumerate(self.items):
            if i == struct_idx_within_method:
                return flat_idx + question_idx
            flat_idx += len(item) if isinstance(item, list) else 1
        return flat_idx


@dataclass
class ScoringMetadata(BaseSchema):
    """Metadata from scoring process."""

    generation_file: str = ""
    scoring_file: str = ""
    judge_model: str = ""
    embedding_model: str = ""
    prefix_logprobs: dict[str, float] = field(default_factory=dict)


@dataclass
class ScoringData(BaseSchema):
    """Loaded scoring data for estimation."""

    # Generic storage: config_key -> list of items
    scoring_data: dict[str, list[ScoringItem]] = field(default_factory=dict)

    # Trajectory results
    results: list[dict[str, Any]] = field(default_factory=list)
    arm_names: list[str] = field(default_factory=list)
    arm_texts: dict[str, str] = field(default_factory=dict)

    # Metadata
    metadata: ScoringMetadata = field(default_factory=ScoringMetadata)

    # Convenience accessors
    @property
    def generation_file(self) -> str:
        return self.metadata.generation_file

    @property
    def scoring_file(self) -> str:
        return self.metadata.scoring_file

    @property
    def judge_model(self) -> str:
        return self.metadata.judge_model

    @property
    def embedding_model(self) -> str:
        return self.metadata.embedding_model

    @property
    def prefix_logprobs(self) -> dict[str, float]:
        return self.metadata.prefix_logprobs

    # Cached structure info
    _structure_info: list[StructureInfo] | None = field(default=None, repr=False)

    # Cached method configs
    _method_configs: list[ScoringMethodConfig] | None = field(default=None, repr=False)

    @classmethod
    def load(cls, path: str | Path) -> ScoringData:
        """Load scoring output from JSON file (v2.0 format)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Scoring output not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        meta = data["metadata"]
        metadata = ScoringMetadata(
            generation_file=meta["generation_file"],
            scoring_file=meta["scoring_file"],
            judge_model=meta["judge_model"],
            embedding_model=meta.get("embedding_model", ""),
            prefix_logprobs=data.get("prefix_logprobs", {}),
        )

        instance = cls(
            scoring_data=data["scoring_items"],
            results=data["results"],
            arm_names=data["arm_names"],
            arm_texts=data.get("arm_texts", {}),
            metadata=metadata,
        )
        instance.validate()
        return instance

    def validate(self) -> None:
        """Validate that the loaded data is usable for estimation."""
        if not self.results:
            raise ValueError("No results found in scoring file")
        # Check if at least one method has items
        has_any_method = any(items for items in self.scoring_data.values() if items)
        if not has_any_method:
            raise ValueError("No scoring methods found in scoring file")

    def get_scoring_methods(self) -> list[ScoringMethodConfig]:
        """Get all configured scoring methods as a unified list with computed offsets.

        Uses the scoring registry to get method metadata (label_prefix, etc.).
        """
        if self._method_configs is not None:
            return self._method_configs

        methods = []
        offset = 0

        # Iterate through registered methods and check if they have data
        for method_name, params_class, _ in iter_methods():
            config_key = params_class.config_key
            items = self.scoring_data.get(config_key, [])

            if not items:
                continue

            # Build config from registry metadata
            methods.append(
                ScoringMethodConfig(
                    method_name=method_name,
                    items=items,
                    label_prefix=params_class.label_prefix,
                    config_key=config_key,
                    struct_offset=offset,
                )
            )
            offset += len(items)

        self._method_configs = methods
        return methods

    def get_structure_info(self) -> list[StructureInfo]:
        """Get information about all structures."""
        if self._structure_info is not None:
            return self._structure_info

        info = []
        idx = 0

        # Use unified iteration over all scoring methods
        for method in self.get_scoring_methods():
            for struct_idx_in_method, item in enumerate(method.items):
                label = f"{method.label_prefix}{struct_idx_in_method + 1}"
                info.append(
                    self._build_structure_info(
                        idx=idx,
                        label=label,
                        item=item,
                        method_name=method.method_name,
                        struct_idx_in_method=struct_idx_in_method,
                    )
                )
                idx += 1

        self._structure_info = info
        return info

    def _build_structure_info(
        self,
        idx: int,
        label: str,
        item: ScoringItem,
        method_name: str,
        struct_idx_in_method: int,
    ) -> StructureInfo:
        """Build StructureInfo for a single item (bundled or not)."""
        if isinstance(item, list):
            return StructureInfo(
                idx=idx,
                label=label,
                description=" + ".join(
                    q[:30] + "..." if len(q) > 30 else q for q in item
                ),
                is_bundled=True,
                question_count=len(item),
                questions=item,
                method_name=method_name,
                struct_idx_in_method=struct_idx_in_method,
            )
        else:
            return StructureInfo(
                idx=idx,
                label=label,
                description=item,
                is_bundled=False,
                question_count=1,
                questions=[item],
                method_name=method_name,
                struct_idx_in_method=struct_idx_in_method,
            )

    def get_text(self, traj_idx: int) -> str:
        """Get text for a trajectory by index."""
        for r in self.results:
            if r.get("traj_idx") == traj_idx:
                return r.get("text", "")
        return ""

    def get_texts(self) -> dict[int, str]:
        """Get all trajectory texts as {traj_idx: text}."""
        return {r["traj_idx"]: r["text"] for r in self.results}

    def get_structure_scores(self, result: dict) -> list[float]:
        """Convert scores to structure scores Lambda_n(x) with grouped averaging.

        For grouped structures, the individual question scores are averaged
        to produce a single structure score for the structure.
        None values are treated as 0.0.
        """
        structure_scores = []

        # Iterate through all scoring methods using unified config
        for method in self.get_scoring_methods():
            for struct_idx, item in enumerate(method.items):
                score = get_score_at_index(result, method.method_name, struct_idx)
                structure_scores.append(self._compute_structure_compliance(score, item))

        return structure_scores

    def _compute_structure_compliance(
        self,
        score: float | list[float] | None,
        item: ScoringItem,
    ) -> float:
        """Compute compliance for a single or grouped structure.

        Args:
            score: Single score or list of scores (for bundles)
            item: The scoring item (string or list of strings)

        Returns:
            Single compliance value (averaged for bundles)
        """
        if isinstance(item, list):
            # Bundled structure: score should be a list, average the values
            if not isinstance(score, list):
                return 0.0
            values = [float(s) if s is not None else 0.0 for s in score]
            return sum(values) / len(values) if values else 0.0
        else:
            # Single structure
            if score is None:
                return 0.0
            return float(score) if not isinstance(score, list) else 0.0

    def compute_arm_scoring(self) -> list[ArmScoring]:
        """Compute per-structure compliance scores for each arm."""
        structures = self.get_structure_info()
        arm_name_list = self.arm_names if self.arm_names else ["trunk"]
        results_by_arm = self._group_results_by_arm()

        scoring = []
        for arm_idx, arm_name in enumerate(arm_name_list):
            results = results_by_arm.get(arm_name, [])
            if not results:
                continue

            scores_result = self._compute_structure_scores(structures, results)

            scoring.append(
                ArmScoring.from_scores_result(
                    arm=arm_name,
                    arm_idx=arm_idx,
                    trajectory_count=len(results),
                    scores=scores_result,
                )
            )

        return scoring

    def _group_results_by_arm(self) -> dict[str, list[dict]]:
        """Group results by arm name."""
        return group_results_by_arm(self.results)

    def _compute_structure_scores(
        self,
        structures: list[StructureInfo],
        results: list[dict],
    ) -> StructureScoresResult:
        """Compute scores for all structures given a set of results."""
        simple_scoring: dict[str, float] = {}
        bundled_scoring: dict[str, BundledScoreResult] = {}

        for struct in structures:
            score = self._compute_scores_for_structure(
                struct=struct,
                results=results,
            )

            if struct.is_bundled:
                bundled_scoring[struct.label] = BundledScoreResult(
                    aggregate=score.aggregate, items=score.item_scores
                )
            else:
                simple_scoring[struct.label] = score.aggregate

        return StructureScoresResult(
            simple_scoring=simple_scoring, bundled_scoring=bundled_scoring
        )

    def _compute_scores_for_structure(
        self,
        struct: StructureInfo,
        results: list[dict],
    ) -> ScoreComputation:
        """Compute scores for a single structure (bundled or not).

        None values are treated as 0.0 (no compliance).
        """
        method_name = struct.method_name
        struct_idx = struct.struct_idx_in_method

        if struct.is_bundled:
            # Bundled: scores[struct_idx] is a list of scores for each question
            all_values: list[float] = []
            item_scores: dict[str, float] = {}

            for q_idx, question in enumerate(struct.questions):
                values = collect_bundle_scores(results, method_name, struct_idx, q_idx)
                if values:
                    avg = sum(values) / len(values)
                    item_scores[question[:50]] = avg
                    all_values.extend(values)

            aggregate = sum(all_values) / len(all_values) if all_values else 0.0
            return ScoreComputation(aggregate=aggregate, item_scores=item_scores)
        else:
            # Single: scores[struct_idx] is a single value
            values = collect_single_scores(results, method_name, struct_idx)
            aggregate = sum(values) / len(values) if values else 0.0
            return ScoreComputation(aggregate=aggregate)

    def group_by_arm(self) -> dict[str, list[TrajectoryScoringData]]:
        """Group results by arm (trunk or branch), returning TrajectoryScoringData objects."""
        grouped: dict[str, list[TrajectoryScoringData]] = {}
        for result in self.results:
            arm = result.get("arm", "trunk")
            idx = result["traj_idx"]
            compliance = self.get_structure_scores(result)
            conditional_logprobs = result.get("conditional_logprobs", {})
            text = result.get("text", "")

            if arm not in grouped:
                grouped[arm] = []
            grouped[arm].append(
                TrajectoryScoringData(
                    traj_idx=idx,
                    arm=arm,
                    structure_scores=compliance,
                    conditional_logprobs=conditional_logprobs,
                    n_generated_tokens=result.get("n_generated_tokens", 0),
                    text=text,
                )
            )

        return grouped

    def get_all_trajectories(self) -> list[TrajectoryScoringData]:
        """Get all trajectories as typed TrajectoryScoringData objects."""
        trajectories: list[TrajectoryScoringData] = []
        for result in self.results:
            arm = result.get("arm", "trunk")
            idx = result["traj_idx"]
            compliance = self.get_structure_scores(result)
            conditional_logprobs = result.get("conditional_logprobs", {})
            text = result.get("text", "")

            trajectories.append(
                TrajectoryScoringData(
                    traj_idx=idx,
                    arm=arm,
                    structure_scores=compliance,
                    conditional_logprobs=conditional_logprobs,
                    n_generated_tokens=result.get("n_generated_tokens", 0),
                    text=text,
                )
            )

        return trajectories

    def get_continuations_by_arm(self) -> ContinuationsByArm:
        """Get continuations organized by arm."""
        result = ContinuationsByArm()
        for r in self.results:
            arm = r.get("arm", "trunk")
            idx = r["traj_idx"]
            text = r.get("text", "")
            result.add(arm, idx, text)
        return result
