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
from src.scoring.scoring_method_registry import iter_methods

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
class ScoringResultRecord(BaseSchema):
    """A single trajectory's scoring results - generic storage."""

    trajectory_idx: int
    branch: str
    text: str = ""
    n_continuation_tokens: int = 0
    conditional_logprobs: dict[str, float] = field(default_factory=dict)

    # Generic storage: method_name -> list of scores
    method_scores: dict[str, list[Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScoringResultRecord:
        """Create from a result dict."""
        return cls(
            trajectory_idx=data.get("trajectory_idx", 0),
            branch=data.get("branch", "trunk"),
            text=data.get("text", ""),
            n_continuation_tokens=data.get("n_continuation_tokens", 0),
            conditional_logprobs=data.get("conditional_logprobs", {}),
            method_scores=data.get("method_scores", {}),
        )


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
    branches: list[str] = field(default_factory=list)
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
        """Load scoring output from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Scoring output not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        metadata = ScoringMetadata(
            generation_file=data.get("generation_file", ""),
            scoring_file=data.get("scoring_file", ""),
            judge_model=data.get("judge_model", ""),
            embedding_model=data.get("embedding_model", ""),
            prefix_logprobs=data.get("prefix_logprobs", {}),
        )

        # Read scoring_data dict (generic format)
        scoring_data = data.get("scoring_data", {})

        instance = cls(
            scoring_data=scoring_data,
            results=data.get("results", []),
            branches=data.get("branches", []),
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
            if r.get("trajectory_idx") == traj_idx:
                return r.get("text", "")
        return ""

    def get_texts(self) -> dict[int, str]:
        """Get all trajectory texts as {traj_idx: text}."""
        return {r["trajectory_idx"]: r["text"] for r in self.results}

    def get_structure_scores(self, result: dict) -> list[float]:
        """Convert scores to structure scores Lambda_n(x) with grouped averaging.

        For grouped structures, the individual question scores are averaged
        to produce a single structure score for the structure.
        None values are treated as 0.0.
        """
        structure_scores = []
        method_scores_dict = result.get("method_scores", {})

        # Iterate through all scoring methods using unified config
        for method in self.get_scoring_methods():
            scores_data = method_scores_dict.get(method.method_name, [])
            score_idx = 0

            for item in method.items:
                structure_scores.append(
                    self._compute_grouped_compliance(
                        scores_data,
                        item,
                        score_idx,
                    )
                )
                score_idx += len(item) if isinstance(item, list) else 1

        return structure_scores

    def _compute_grouped_compliance(
        self,
        scores: list,
        item: ScoringItem,
        start_idx: int,
    ) -> float:
        """Compute compliance for a single or grouped structure.

        None values are treated as 0.0 (no compliance).
        """
        if isinstance(item, list):
            # Grouped structure: average the questions
            group_scores = []
            for i in range(len(item)):
                idx = start_idx + i
                if idx < len(scores):
                    s = scores[idx]
                    group_scores.append(float(s) if s is not None else 0.0)
                else:
                    group_scores.append(0.0)
            return sum(group_scores) / len(group_scores) if group_scores else 0.0
        else:
            # Single structure
            if start_idx < len(scores):
                s = scores[start_idx]
                return float(s) if s is not None else 0.0
            return 0.0

    def compute_arm_scoring(self) -> list[ArmScoring]:
        """Compute per-structure compliance scores for each arm."""
        structures = self.get_structure_info()
        arm_names = self.branches if self.branches else ["trunk"]
        results_by_arm = self._group_results_by_arm()

        scoring = []
        for arm_idx, arm_name in enumerate(arm_names):
            results = results_by_arm.get(arm_name, [])
            if not results:
                continue

            scores_result = self._compute_structure_scores(structures, results)

            scoring.append(
                ArmScoring.from_scores_result(
                    branch=arm_name,
                    branch_idx=arm_idx,
                    trajectory_count=len(results),
                    scores=scores_result,
                )
            )

        return scoring

    def _group_results_by_arm(self) -> dict[str, list[dict]]:
        """Group results by arm name."""
        by_arm: dict[str, list[dict]] = {}
        for result in self.results:
            arm = result.get("branch", "trunk")
            by_arm.setdefault(arm, []).append(result)
        return by_arm

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

        # Get flat index calculator for this method's items
        method_config = next(
            (m for m in self.get_scoring_methods() if m.method_name == method_name),
            None,
        )
        if method_config is None:
            return ScoreComputation(aggregate=0.0)

        items = method_config.items

        if struct.is_bundled:
            all_values: list[float] = []
            item_scores: dict[str, float] = {}

            for q_idx, question in enumerate(struct.questions):
                flat_idx = self._get_flat_index_for_items(items, struct_idx, q_idx)
                values = self._extract_values(results, method_name, flat_idx)
                if values:
                    avg = sum(values) / len(values)
                    item_scores[question[:50]] = avg
                    all_values.extend(values)

            aggregate = sum(all_values) / len(all_values) if all_values else 0.0
            return ScoreComputation(aggregate=aggregate, item_scores=item_scores)
        else:
            flat_idx = self._get_flat_index_for_items(items, struct_idx, 0)
            values = self._extract_values(results, method_name, flat_idx)
            aggregate = sum(values) / len(values) if values else 0.0
            return ScoreComputation(aggregate=aggregate)

    def _extract_values(
        self,
        results: list[dict],
        method_name: str,
        flat_idx: int,
    ) -> list[float]:
        """Extract values from results at a given index for a method.

        None values are converted to 0.0 (no compliance).
        """
        values = []
        for r in results:
            method_scores = r.get("method_scores", {})
            scores = method_scores.get(method_name, [])
            if flat_idx < len(scores):
                val = scores[flat_idx]
                # Convert None to 0.0 - scoring must always be 0-1
                values.append(float(val) if val is not None else 0.0)
        return values

    def _get_flat_index_for_items(
        self, items: list[ScoringItem], struct_idx: int, question_idx: int
    ) -> int:
        """Get the flat index into a scores array for a structure/question.

        This is a unified helper that works for any scoring method's items.
        """
        flat_idx = 0
        for i, item in enumerate(items):
            if i == struct_idx:
                return flat_idx + question_idx
            if isinstance(item, list):
                flat_idx += len(item)
            else:
                flat_idx += 1
        return flat_idx

    def group_by_arm(self) -> dict[str, list[TrajectoryScoringData]]:
        """Group results by arm (trunk or branch), returning TrajectoryScoringData objects."""
        grouped: dict[str, list[TrajectoryScoringData]] = {}
        for result in self.results:
            branch = result.get("branch", "trunk")
            idx = result["trajectory_idx"]
            compliance = self.get_structure_scores(result)
            conditional_logprobs = result.get("conditional_logprobs", {})

            if branch not in grouped:
                grouped[branch] = []
            grouped[branch].append(
                TrajectoryScoringData(
                    traj_idx=idx,
                    branch=branch,
                    structure_scores=compliance,
                    conditional_logprobs=conditional_logprobs,
                    n_continuation_tokens=result.get("n_continuation_tokens", 0),
                )
            )

        return grouped

    def get_continuations_by_arm(self) -> ContinuationsByArm:
        """Get continuations organized by arm."""
        result = ContinuationsByArm()
        for r in self.results:
            arm = r.get("branch", "trunk")
            idx = r["trajectory_idx"]
            text = r.get("text", "")
            result.add(arm, idx, text)
        return result
