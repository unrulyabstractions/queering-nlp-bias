"""Greedy output schema.

`greedy.json` sits alongside `generation.json` and stores one greedy-decoded
trajectory per arm (temperature=0 from that arm's prefill). When the file
is augmented with `structure_scores` (typically by the scoring step), it
serves as the authoritative source for the `greedy` core estimator —
otherwise estimation falls back to walking the empirical prefix tree of
the sampled trajectories.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.common.base_schema import BaseSchema


@dataclass
class GreedyArmEntry(BaseSchema):
    """One arm's greedy-decoded continuation."""

    name: str
    text: str = ""
    n_generated_tokens: int = 0
    token_ids: list[int] = field(default_factory=list)
    prefill_length: int = 0
    structure_scores: list[float] = field(default_factory=list)


@dataclass
class GreedyOutput(BaseSchema):
    """Per-arm greedy paths produced at temperature=0."""

    version: str = "1.0"
    metadata: dict[str, Any] = field(default_factory=dict)
    arms: list[GreedyArmEntry] = field(default_factory=list)

    @staticmethod
    def compute_path(generation_file: str | Path) -> Path:
        """Conventional location: alongside `generation.json`."""
        return Path(generation_file).parent / "greedy.json"

    @classmethod
    def load(cls, path: str | Path) -> GreedyOutput:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return p

    def get_arm(self, name: str) -> GreedyArmEntry | None:
        for a in self.arms:
            if a.name == name:
                return a
        return None

    @classmethod
    def create(
        cls,
        *,
        model: str,
        config_path: str | Path,
        arms: list[GreedyArmEntry],
    ) -> GreedyOutput:
        return cls(
            metadata={
                "model": model,
                "config_path": str(config_path),
                "generated_at": datetime.utcnow().isoformat() + "Z",
            },
            arms=arms,
        )
