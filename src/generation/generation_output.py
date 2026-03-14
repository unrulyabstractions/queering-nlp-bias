"""Official output format for generation results.

GenerationOutput is the canonical, versioned output format for trajectory generation.
All fields are organized into clear sections for machine and human consumption.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.common.base_schema import BaseSchema
from src.common.output_paths import generation_output_path, generation_summary_path
from src.common.token_tree import TokenTree

from .generation_config import GenerationConfig

# ══════════════════════════════════════════════════════════════════════════════
# METADATA
# ══════════════════════════════════════════════════════════════════════════════


OUTPUT_VERSION = "2.0"


@dataclass
class GenerationMetadata(BaseSchema):
    """Metadata about the generation run."""

    version: str  # Output format version
    generated_at: str  # ISO timestamp
    model: str  # Model name
    method: str  # Generation method: simple-sampling, forking-paths, etc.
    num_trajectories: int  # Number of trajectories generated
    eos_token: str | None  # EOS token from model (for finished detection)


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT CLASS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class GenerationOutput(BaseSchema):
    """Official output format for generation results.

    Sections:
        metadata: Run metadata (version, timestamp, model, method)
        config: Generation configuration (prompt, arms, params)
        tree: Token tree with trajectories and branching structure

    Output path: out/<method>/<gen_name>/generation.json
    """

    # === METADATA ===
    metadata: GenerationMetadata

    # === CONFIGURATION ===
    config: dict[str, Any]  # GenerationConfig.to_dict() with arms

    # === TREE DATA ===
    tree: dict[str, Any] | None = None  # TokenTree.to_dict()

    # ──────────────────────────────────────────────────────────────────────────
    # Factory
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_tree(
        cls,
        *,
        config: GenerationConfig,
        model: str,
        tree: TokenTree,
        arms: list,  # list[GenerationArm]
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
            eos_token: EOS token from the model
            arm_token_lengths: Token lengths for each arm
        """
        # Pop heavy data before serializing (full_logits tensors)
        tree.pop_heavy()

        # Build config dict with arm info
        config_dict = config.to_dict()
        config_dict["arms"] = [arm.to_dict() for arm in arms]

        # Compute arm lengths
        arm_text_lengths = [len(arm.prefill) for arm in arms]
        if arm_token_lengths:
            for i, length in enumerate(arm_token_lengths):
                if i < len(config_dict["arms"]):
                    config_dict["arms"][i]["token_length"] = length
        for i, length in enumerate(arm_text_lengths):
            if i < len(config_dict["arms"]):
                config_dict["arms"][i]["text_length"] = length

        # Set arm lengths on each trajectory
        for traj in tree.trajs:
            traj.arm_token_lengths = arm_token_lengths
            traj.arm_text_lengths = arm_text_lengths

        # Create metadata
        metadata = GenerationMetadata(
            version=OUTPUT_VERSION,
            generated_at=datetime.now().isoformat(),
            model=model,
            method=method,
            num_trajectories=len(tree.trajs),
            eos_token=eos_token,
        )

        return cls(
            metadata=metadata,
            config=config_dict,
            tree=tree.to_dict(),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, path: str | Path, config_path: str | Path | None = None) -> Path:
        """Save to JSON file."""
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

    @classmethod
    def load(cls, path: str | Path) -> GenerationOutput:
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    # ──────────────────────────────────────────────────────────────────────────
    # Path Computation (delegates to centralized output_paths module)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_output_path(
        config_path: Path,
        method: str = "sampling",
        base_dir: str | Path = "out",
        include_method: bool = False,
    ) -> Path:
        """Compute output path from config path."""
        return generation_output_path(
            config_path,
            base_dir=base_dir,
            method=method if include_method else None,
        )

    @staticmethod
    def compute_summary_path(
        config_path: Path,
        method: str = "sampling",
        base_dir: str | Path = "out",
        include_method: bool = False,
    ) -> Path:
        """Compute summary text file path."""
        return generation_summary_path(
            config_path,
            base_dir=base_dir,
            method=method if include_method else None,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Summary (convenience methods delegating to standalone functions)
    # ──────────────────────────────────────────────────────────────────────────

    def save_summary(self, path: str | Path) -> Path:
        """Save human-readable summary to text file."""
        from .generation_helpers import save_generation_summary

        return save_generation_summary(
            path,
            self.metadata.model,
            self.metadata.method,
            self.metadata.generated_at,
            self.metadata.num_trajectories,
            self.config,
            self.tree,
            self.metadata.eos_token,
        )

    def summarize(self) -> None:
        """Print summary to console."""
        from .generation_helpers import print_generation_summary

        print_generation_summary(
            self.metadata.model,
            self.metadata.method,
            self.metadata.generated_at,
            self.metadata.num_trajectories,
            self.config,
            self.tree,
            self.metadata.eos_token,
        )
