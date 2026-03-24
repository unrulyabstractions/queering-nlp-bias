"""Configuration for trajectory generation.

This module defines the GenerationConfig class which holds all settings
for generating trajectories.

Method parameters are stored as MethodParamsOverride objects, but accessed via
get_params() which returns properly typed params objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.common.base_schema import BaseSchema
from src.common.default_config import MAX_NEW_TOKENS, TEMPERATURE
from src.common.experiment_types import GenerationArm
from src.common.method_params_override import MethodParamsOverride

from .generation_method_registry import GenerationMethodParams, get_default_params


@dataclass
class GenerationConfig(BaseSchema):
    """Configuration for trajectory generation.

    Method-specific parameters are stored in `method_params` dict (keyed by
    method name). Use `get_params(method_name)` to access typed params objects.

    Example JSON:
        {
            "model": "Qwen/Qwen3-0.6B",
            "prompt": "Write a story...",
            "method_params": {
                "simple-sampling": {"overrides": {"samples_per_arm": 20}},
                "forking-paths": {"overrides": {"max_alternates": 10}}
            }
        }

    Example usage:
        config = GenerationConfig.load("config.json")
        params = config.get_params("simple-sampling")  # Returns SamplingParams
    """

    prompt: str
    model: str = ""
    trunk: str = ""
    branches: list[str] = field(default_factory=list)
    twig_variations: list[str] = field(default_factory=list)

    # General generation params
    temperature: float = field(default_factory=lambda: TEMPERATURE)
    max_new_tokens: int = field(default_factory=lambda: MAX_NEW_TOKENS)
    seed: int | None = None

    # Method-specific parameter overrides - keyed by method name
    # Access via get_params() for typed params objects
    method_params: dict[str, MethodParamsOverride] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> GenerationConfig:
        """Load config from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        return cls.from_json(path)

    def get_arms(self, skip_prefix: str = "") -> list[GenerationArm]:
        """Get arm configurations for generation.

        Returns list of arms. Position in list IS the arm index.
        parent_idx points to parent's position for AfterBranch.

        Args:
            skip_prefix: Prefix to prepend (e.g., reasoning skip tokens)

        Returns:
            List of GenerationArm objects.
        """
        result: list[GenerationArm] = []

        # Root (idx 0) - no parent
        result.append(GenerationArm(name="root", prefill=skip_prefix))

        # Trunk (idx 1) - parent is root
        result.append(
            GenerationArm(name="trunk", prefill=skip_prefix + self.trunk, parent_idx=0)
        )

        # Branches and twigs
        trunk_idx = 1
        for branch_num, branch in enumerate(self.branches, start=1):
            branch_prefill = skip_prefix + self.trunk + branch
            branch_idx = len(result)

            # Branches have trunk as parent
            result.append(
                GenerationArm(
                    name=f"branch_{branch_num}",
                    prefill=branch_prefill,
                    parent_idx=trunk_idx,
                )
            )

            for twig_num, twig in enumerate(self.twig_variations, start=1):
                result.append(
                    GenerationArm(
                        name=f"twig_b{branch_num}_{twig_num}",
                        prefill=branch_prefill + twig,
                        parent_idx=branch_idx,
                    )
                )

        return result

    def get_params(self, method: str) -> GenerationMethodParams:
        """Get typed params object for a generation method.

        Looks up the method in the registry, creates a default params instance,
        then applies any overrides from method_params.

        Args:
            method: Method name (e.g., "simple-sampling", "forking-paths")

        Returns:
            Properly typed params instance with any config overrides applied
        """
        # Get default params for this method
        params = get_default_params(method)

        # Apply any overrides from method_params
        if method in self.method_params:
            self.method_params[method].apply_to(params)

        return params
