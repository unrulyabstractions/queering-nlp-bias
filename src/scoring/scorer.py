"""Scorer - unified interface for scoring text.

Mirrors ModelRunner pattern: one object, simple methods, handles complexity internally.

Usage:
    scorer = Scorer(config)  # or Scorer.load("path/to/config.json")

    # Score text -> flat structure scores
    scores = scorer.score(text)

    # Score text -> full method results
    results = scorer.score_detailed(text)

    # Score trajectory
    traj_scores = scorer.score_trajectory(traj)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.callback_types import LogFn
from src.inference import ModelRunner
from src.inference.embedding_runner import EmbeddingRunner

from .scoring_config import ScoringConfig, StringSelection
from .scoring_data import TrajectoryData
from .scoring_method_registry import get_method, get_params_class


def get_text_for_scoring(
    traj: TrajectoryData, config: ScoringConfig, eos_token: str | None = None
) -> str:
    """Extract text from trajectory based on config string_selection."""
    from src.common.text import strip_eos_tokens

    text = {
        StringSelection.WholeContinuation: traj.continuation_text,
        StringSelection.NonThinkingContinuation: traj.continuation_text_no_thinking,
        StringSelection.AfterTrunk: traj.text_after_trunk,
        StringSelection.AfterBranch: traj.text_after_branch,
        StringSelection.AfterTwig: traj.text_after_twig,
    }.get(config.string_selection, traj.continuation_text_no_thinking)
    return strip_eos_tokens(text, [eos_token] if eos_token else None)


class Scorer:
    """Unified scoring interface. Load once, score many."""

    def __init__(self, config: ScoringConfig):
        """Create scorer from config. Loads models as needed."""
        self.config = config
        self._runner: ModelRunner | None = None
        self._embedder: EmbeddingRunner | None = None

        if config.needs_runner() and config.model:
            self._runner = ModelRunner(config.model)
        if config.needs_embedder():
            self._embedder = EmbeddingRunner(config.embedding_model)

    @classmethod
    def load(cls, path: str | Path) -> Scorer:
        """Load scorer from config file."""
        return cls(ScoringConfig.load(path))

    def score(self, text: str, log_fn: LogFn | None = None) -> list[float]:
        """Score text, return flat structure scores (bundles averaged)."""
        results = self.score_detailed(text, log_fn)
        return self._flatten(results)

    def score_detailed(
        self, text: str, log_fn: LogFn | None = None
    ) -> dict[str, tuple[list[Any], list[str]]]:
        """Score text, return full method results: {method -> (scores, raw_responses)}."""
        results: dict[str, tuple[list[Any], list[str]]] = {}

        for method_name in self.config.get_active_methods():
            items = self.config.get_method_items(method_name)
            if not items:
                continue

            params_class = get_params_class(method_name)
            if params_class.requires_runner and self._runner is None:
                raise ValueError(f"{method_name} requires a model runner")
            if params_class.requires_embedder and self._embedder is None:
                raise ValueError(f"{method_name} requires an embedding runner")

            params = self.config.get_scoring_params(method_name)
            scores, raw = get_method(method_name)(
                text, items, params, self._runner, self._embedder, log_fn
            )
            results[method_name] = (scores, raw)

        return results

    def score_trajectory(
        self,
        traj: TrajectoryData,
        log_fn: LogFn | None = None,
        eos_token: str | None = None,
    ) -> dict[str, tuple[list[Any], list[str]]]:
        """Score trajectory, return full method results."""
        text = self._get_traj_text(traj, eos_token)
        return self.score_detailed(text, log_fn)

    def _flatten(self, results: dict[str, tuple[list[Any], list[str]]]) -> list[float]:
        """Flatten method results to one float per structure."""
        scores: list[float] = []
        for method_name in self.config.get_active_methods():
            items = self.config.get_method_items(method_name)
            if not items or method_name not in results:
                continue
            method_scores = results[method_name][0]
            for i, item in enumerate(items):
                if i >= len(method_scores):
                    scores.append(0.0)
                elif isinstance(item, list):
                    sub = method_scores[i]
                    scores.append(sum(sub) / len(sub) if sub else 0.0)
                else:
                    scores.append(float(method_scores[i] or 0.0))
        return scores

    def _get_traj_text(self, traj: TrajectoryData, eos_token: str | None) -> str:
        """Extract text from trajectory based on config."""
        return get_text_for_scoring(traj, self.config, eos_token)

    @property
    def num_structures(self) -> int:
        """Number of structures being scored."""
        return self.config.num_structures()

    @property
    def structure_labels(self) -> list[str]:
        """Labels for each structure (c1, c2, g1, ...)."""
        return self.config.get_structure_labels()

    def cleanup(self) -> None:
        """Release model memory and clear GPU caches.

        Call this when done with the scorer to free GPU/MPS memory.
        """
        if self._runner is not None:
            self._runner.cleanup()
            self._runner = None

        if self._embedder is not None:
            self._embedder.cleanup()
            self._embedder = None
