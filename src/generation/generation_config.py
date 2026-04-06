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

    **Traditional mode** (prompt/trunk/branches):
        {
            "model": "Qwen/Qwen3-0.6B",
            "prompt": "Write a story...",
            "trunk": "The protagonist was a ",
            "branches": ["nurse,", "mechanic,"],
            "method_params": {"simple-sampling": {"overrides": {"samples_per_arm": 20}}}
        }

    **Template mode** (prompt_template/template_words):
        {
            "model": "openai/gpt-4o-mini",
            "prompt_template": "Generate a persona of a {word}.",
            "template_words": ["man", "woman", "gay man"],
            "method_params": {"simple-sampling": {"overrides": {"samples_per_arm": 20}}}
        }
        Each word is substituted into the template to produce one arm labeled
        with that word. prompt/trunk/branches/twig_variations must be absent.

    Example usage:
        config = GenerationConfig.load("config.json")
        params = config.get_params("simple-sampling")  # Returns SamplingParams
    """

    prompt: str = ""
    model: str = ""
    trunk: str = ""
    branches: list[str] = field(default_factory=list)
    twig_variations: list[str] = field(default_factory=list)

    # Template mode fields — mutually exclusive with prompt/trunk/branches/twig_variations
    prompt_template: str = ""
    template_words: list[str] = field(default_factory=list)

    # General generation params
    temperature: float = field(default_factory=lambda: TEMPERATURE)
    max_new_tokens: int = field(default_factory=lambda: MAX_NEW_TOKENS)
    seed: int | None = None

    # Method-specific parameter overrides - keyed by method name
    # Access via get_params() for typed params objects
    method_params: dict[str, MethodParamsOverride] = field(default_factory=dict)

    def __post_init__(self) -> None:
        has_template = bool(self.prompt_template)
        has_words = bool(self.template_words)

        if has_template != has_words:
            raise ValueError(
                "prompt_template and template_words must both be specified or both omitted"
            )

        if has_template and has_words:
            if self.prompt or self.trunk or self.branches or self.twig_variations:
                raise ValueError(
                    "When using prompt_template/template_words, prompt/trunk/"
                    "branches/twig_variations must not be set"
                )
        else:
            if not self.prompt:
                raise ValueError(
                    "prompt is required when not using prompt_template/template_words"
                )

    @property
    def is_template_mode(self) -> bool:
        """True when arms are produced by filling template_words into prompt_template."""
        return bool(self.prompt_template and self.template_words)

    @classmethod
    def load(cls, path: str | Path) -> GenerationConfig:
        """Load config from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        return cls.from_json(path)

    def get_arms(self, skip_prefix: str = "") -> list[GenerationArm]:
        """Get arm configurations for generation.

        In template mode: one arm per template_word, each with its own filled
        prompt and no prefill text (only skip_prefix if provided).

        In traditional mode: root → trunk → branches → twigs, exactly as
        before.

        Args:
            skip_prefix: Prefix to prepend (e.g., reasoning skip tokens)

        Returns:
            List of GenerationArm objects.
        """
        if self.is_template_mode:
            return [
                GenerationArm(
                    name=word,
                    prefill=skip_prefix,
                    prompt=self.prompt_template.format(word=word),
                    parent_idx=None,
                )
                for word in self.template_words
            ]

        result: list[GenerationArm] = []

        # Root (idx 0) - no parent
        result.append(GenerationArm(name="root", prefill=skip_prefix))

        # Trunk (idx 1) - only added when trunk text is non-empty; otherwise root
        # and trunk would have identical prefills, making trunk redundant.
        if self.trunk:
            result.append(
                GenerationArm(
                    name="trunk", prefill=skip_prefix + self.trunk, parent_idx=0
                )
            )

        # Branches and twigs — parent is trunk if present, otherwise root
        trunk_idx = next(
            (i for i, arm in enumerate(result) if arm.name == "trunk"), 0
        )
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
