"""Output dataclass for trajectory generation.

This module defines the GenerationOutput class which holds the results
of trajectory generation including the token tree structure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.common.base_schema import BaseSchema
from src.common.token_tree import TokenTree

from .generation_config import GenerationConfig
from .generation_helpers import (
    print_generation_summary,
    save_generation_summary,
)

# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT CLASS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class GenerationOutput(BaseSchema):
    """Output from trajectory generation, including tree structure."""

    config: dict[str, Any]  # GenerationConfig.to_dict()
    model: str
    method: str  # Generation method: simple-sampling, forking-paths, seeking-entropy
    generated_at: str
    num_trajectories: int
    tree: dict[str, Any] | None = None  # TokenTree.to_dict() output
    eos_token: str | None = None  # EOS token from model (for finished detection)

    @classmethod
    def from_tree(
        cls,
        config: GenerationConfig,
        model: str,
        tree: TokenTree,
        arms: list,  # list[GenerationArm] - piped from generation
        method: str = "simple-sampling",
        eos_token: str | None = None,
        arm_token_lengths: list[int] | None = None,
    ) -> GenerationOutput:
        """Create output from a TokenTree.

        Args:
            config: Generation configuration
            model: Model name used
            tree: TokenTree with trajectories
            arms: Arm objects from generation (with correct prefills)
            method: Generation method name
            eos_token: EOS token from the model (for finished detection)
            arm_token_lengths: Token lengths for each arm (from generation)
        """
        # Pop heavy data before serializing (full_logits tensors)
        tree.pop_heavy()

        # Build config dict with arm info stored as structured data
        config_dict = config.to_dict()
        config_dict["arms"] = [arm.to_dict() for arm in arms]

        # Compute arm lengths (token and text)
        arm_text_lengths = [len(arm.prefill) for arm in arms]
        if arm_token_lengths:
            for i, length in enumerate(arm_token_lengths):
                if i < len(config_dict["arms"]):
                    config_dict["arms"][i]["token_length"] = length
        for i, length in enumerate(arm_text_lengths):
            if i < len(config_dict["arms"]):
                config_dict["arms"][i]["text_length"] = length

        # Set arm lengths on each trajectory for text_after_arm() slicing
        for traj in tree.trajs:
            traj.arm_token_lengths = arm_token_lengths
            traj.arm_text_lengths = arm_text_lengths

        return cls(
            config=config_dict,
            model=model,
            method=method,
            generated_at=datetime.now().isoformat(),
            num_trajectories=len(tree.trajs),
            tree=tree.to_dict(max_list_length=10000),
            eos_token=eos_token,
        )

    @staticmethod
    def compute_output_path(config_path: Path, method: str = "sampling") -> Path:
        """Compute the output path for generation results.

        Output structure: out/<method>/<gen_name>/generation.json
        """
        return Path("out") / method / config_path.stem / "generation.json"

    @staticmethod
    def compute_summary_path(config_path: Path, method: str = "sampling") -> Path:
        """Compute the output path for generation summary.

        Output structure: out/<method>/<gen_name>/gen_summary.txt
        """
        return Path("out") / method / config_path.stem / "gen_summary.txt"

    def save(self, path: str | Path, config_path: str | Path | None = None) -> Path:
        """Save output to JSON file.

        Args:
            path: Output path for generation.json
            config_path: Original config file to copy as generation_cfg.json
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

        # Copy original config if provided
        if config_path:
            import shutil
            cfg_dest = path.parent / "generation_cfg.json"
            shutil.copy(config_path, cfg_dest)

        return path

    def save_summary(self, path: str | Path) -> Path:
        """Save human-readable summary to text file."""
        return save_generation_summary(
            path,
            self.model,
            self.method,
            self.generated_at,
            self.num_trajectories,
            self.config,
            self.tree,
            self.eos_token,
        )

    def summarize(self) -> None:
        """Print a clean summary of generation results."""
        print_generation_summary(
            self.model,
            self.method,
            self.generated_at,
            self.num_trajectories,
            self.config,
            self.tree,
            self.eos_token,
        )
