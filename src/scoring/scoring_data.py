"""Data types for trajectory scoring."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.common.experiment_types import GenerationArm
from src.common.token_tree import TokenTree
from src.common.token_trajectory import TokenTrajectory


@dataclass
class TrajectoryData:
    """Wrapper around TokenTrajectory with scoring-specific fields.

    All text selections are precomputed during loading - no parsing at scoring time.
    """

    traj: TokenTrajectory
    arm_name: str
    arm_idx: int
    n_continuation_tokens: int
    conditional_logprobs: dict[str, float] = field(default_factory=dict)

    # Precomputed text selections (piped from arm structure, not parsed)
    text_after_trunk: str = ""
    text_after_branch: str = ""
    text_after_twig: str = ""

    @property
    def idx(self) -> int:
        return self.traj.traj_idx or 0

    @property
    def prefill_text(self) -> str:
        return self.traj.prefill_text or ""

    @property
    def generated_text(self) -> str:
        return self.traj.generated_text or ""

    @property
    def continuation_text(self) -> str:
        return self.traj.continuation_text or ""

    @property
    def continuation_text_no_thinking(self) -> str:
        return self.traj.continuation_text_no_thinking or ""

    @property
    def logprobs(self) -> list[float]:
        return self.traj.logprobs




@dataclass
class GenerationOutputData:
    """Loaded generation output with trajectory data."""

    tree: TokenTree | None
    trajectories: list[TrajectoryData]
    config: dict[str, Any]
    arms: list[GenerationArm]
    eos_token: str | None = None

    @property
    def arm_names(self) -> list[str]:
        return [arm.name for arm in self.arms]

    @property
    def arm_prefills(self) -> list[str]:
        return [arm.prefill for arm in self.arms]

    @property
    def arm_texts(self) -> dict[str, str]:
        """Map arm names to prefill texts."""
        return {arm.name: arm.prefill for arm in self.arms}

    @classmethod
    def load(cls, path: str | Path) -> GenerationOutputData:
        """Load generation output from JSON file."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        config = data.get("config", {})
        tree = TokenTree.from_dict(data["tree"]) if data.get("tree") else None

        # Load arms from structured data
        arms_data = config.get("arms", [])
        arms = [GenerationArm.from_dict(a) for a in arms_data]

        # Find trunk index from arms list
        trunk_idx = next((i for i, arm in enumerate(arms) if arm.name == "trunk"), 1)

        trajectories = []
        if tree:
            for i, traj in enumerate(tree.trajs):
                traj.traj_idx = i
                arm_idx = traj.arm_index[0] if traj.arm_index else 0
                if arm_idx >= len(arms):
                    continue
                arm = arms[arm_idx]

                # Get token length from arm data
                token_length = arms_data[arm_idx].get("token_length", 0) if arm_idx < len(arms_data) else 0
                n_cont = len(traj.token_ids) - token_length

                # Compute conditional logprobs using token lengths from arms
                cond_logprobs = {}
                for idx, other_arm in enumerate(arms):
                    other_length = arms_data[idx].get("token_length", 0) if idx < len(arms_data) else 0
                    cond_logprobs[other_arm.name] = sum(traj.logprobs[other_length:])

                # Text selections use precomputed arm_text_lengths via text_after_arm()
                text_after_trunk = traj.text_after_arm(trunk_idx)
                if arm.parent_idx is not None:
                    text_after_branch = traj.text_after_arm(arm.parent_idx)
                else:
                    text_after_branch = traj.generated_text or ""
                text_after_twig = traj.generated_text or ""

                trajectories.append(
                    TrajectoryData(
                        traj=traj,
                        arm_name=arm.name,
                        arm_idx=arm_idx,
                        n_continuation_tokens=n_cont,
                        conditional_logprobs=cond_logprobs,
                        text_after_trunk=text_after_trunk,
                        text_after_branch=text_after_branch,
                        text_after_twig=text_after_twig,
                    )
                )

        return cls(
            tree=tree,
            trajectories=trajectories,
            config=config,
            arms=arms,
            eos_token=data.get("eos_token"),
        )
