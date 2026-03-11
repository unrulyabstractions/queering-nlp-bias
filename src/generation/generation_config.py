"""Configuration for trajectory generation.

This module defines the GenerationConfig class which holds all settings
for generating trajectories.

Method parameters are stored as MethodParamsOverride objects, but accessed via
get_params() which returns properly typed params objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.common.base_schema import BaseSchema

from .generation_method_registry import GenerationMethodParams, get_default_params
from .generation_types import GenerationArm


@dataclass
class MethodParamsOverride(BaseSchema):
    """Override values for a generation method's parameters.

    Wraps a flat dict of parameter overrides. Used to avoid nested dicts
    in GenerationConfig while maintaining JSON compatibility.
    """

    overrides: dict[str, Any] = field(default_factory=dict)

    def apply_to(self, params: GenerationMethodParams) -> None:
        """Apply these overrides to a params instance."""
        for key, value in self.overrides.items():
            if hasattr(params, key):
                setattr(params, key, value)


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

    # General generation params
    temperature: float = 1.0
    max_new_tokens: int = 128
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

        Args:
            skip_prefix: Prefix to prepend (e.g., reasoning skip tokens)

        Returns:
            List of GenerationArm objects: trunk first, then each branch
        """
        # Always include trunk as first arm
        result = [GenerationArm(prefill=skip_prefix + self.trunk, name="trunk", arm_index=0)]

        # Add explicit branches if defined
        if self.branches:
            result.extend(
                GenerationArm(
                    prefill=skip_prefix + self.trunk + branch,
                    name=f"branch_{i + 1}",
                    arm_index=i + 1,
                )
                for i, branch in enumerate(self.branches)
            )

        return result

    @property
    def fork_arms(self) -> list[tuple[int, int]]:
        """Get all pairwise fork arms between branches as (left, right) tuples."""
        if len(self.branches) < 2:
            return []
        return [(i, j) for i in range(len(self.branches)) for j in range(i + 1, len(self.branches))]

    def compute_prompt_length(self, runner) -> int:
        """Compute the shared prefix length between prompt-only and prompt+trunk.

        Due to BPE tokenization, adding trunk text may change how the prompt is
        tokenized. This finds the last position where both tokenizations agree,
        which is where trunk-specific logprobs start.
        """
        skip_prefix = runner.skip_thinking_prefix
        prompt_only = runner.apply_chat_template(self.prompt) + skip_prefix
        prompt_trunk = prompt_only + self.trunk

        tokens_prompt = runner.encode_ids(prompt_only, add_special_tokens=True)
        tokens_trunk = runner.encode_ids(prompt_trunk, add_special_tokens=True)

        # Find the divergence point
        shared_length = 0
        for i, (t1, t2) in enumerate(zip(tokens_prompt, tokens_trunk)):
            if t1 != t2:
                break
            shared_length = i + 1

        return shared_length

    def compute_trunk_length(self, runner) -> int:
        """Compute the length of the trunk (prompt + trunk, no branch) in tokens."""
        skip_prefix = runner.skip_thinking_prefix
        formatted = runner.apply_chat_template(self.prompt) + skip_prefix + self.trunk
        return len(runner.encode_ids(formatted, add_special_tokens=True))

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
