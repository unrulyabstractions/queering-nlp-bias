"""Configuration for trajectory scoring/judgment.

This module defines ScoringConfig - a generic config that works with
any registered scoring method without modification.

Method data is stored in a generic dict. Each method reads its own
config_key from the dict automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from src.common.base_schema import BaseSchema
from src.common.default_config import (
    EMBEDDING_MODEL,
    JUDGE_MAX_TOKENS,
    STRING_SELECTION,
)
from src.common.method_params_override import MethodParamsOverride

from .scoring_method_registry import (
    ScoringMethodParams,
    get_default_params,
    get_params_class,
    iter_methods,
)


class StringSelection(Enum):
    """Determines which portion of the trajectory text to use for scoring.

    Options:
        WholeTrajectory: Full text including prompt and response
        WholeContinuation: Just the generated response/continuation (default)
        AfterTrunk: Text after the trunk tokens (continuation minus trunk)
        AfterBranch: Text after the branch point (continuation minus trunk and branch)
        AfterTwig: Text after the twig point (continuation minus trunk, branch, and twig)
    """

    WholeTrajectory = "WholeTrajectory"
    WholeContinuation = "WholeContinuation"
    NonThinkingContinuation = "NonThinkingContinuation"
    AfterTrunk = "AfterTrunk"
    AfterBranch = "AfterBranch"
    AfterTwig = "AfterTwig"


@dataclass
class ScoringConfig(BaseSchema):
    """Configuration for trajectory scoring/judgment.

    This is a GENERIC config - new methods don't require changes here.

    Method data is stored in `scoring_data` dict, keyed by each method's
    config_key. For example:
    - categorical method reads from scoring_data["categorical_judgements"]
    - similarity method reads from scoring_data["similarity_scoring"]

    Each method's items can be:
    - Individual items (strings): Each becomes its own structure
    - Bundled items (list of strings): All items in the bundle are
      averaged together to form a single structure value
    """

    # Judge model for LLM-based methods
    model: str = ""

    # Embedding model for similarity-based methods
    embedding_model: str = field(default_factory=lambda: EMBEDDING_MODEL)

    # Text selection for scoring
    string_selection: StringSelection = field(
        default_factory=lambda: StringSelection(STRING_SELECTION)
    )

    # Generic max_tokens for LLM methods
    max_tokens: int = field(default_factory=lambda: JUDGE_MAX_TOKENS)

    # Method-specific parameter overrides
    method_params: dict[str, MethodParamsOverride] = field(default_factory=dict)

    # Generic storage for all method data
    # Keys are method config_keys (e.g., "categorical_judgements", "similarity_scoring")
    # Values are lists of items (str or list[str] for bundles)
    scoring_data: dict[str, list[str | list[str]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Convert any extra fields to scoring_data."""
        # This allows JSON to use flat keys like "categorical_judgements"
        # instead of nested "scoring_data.categorical_judgements"
        pass

    @classmethod
    def load(cls, path: str | Path) -> ScoringConfig:
        """Load config from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Scoring config not found: {path}")
        config = cls.from_json(path)
        config.validate()
        return config

    @classmethod
    def from_dict(cls, data: dict) -> ScoringConfig:
        """Create config from dict, moving method data to scoring_data."""
        # Extract known fields
        model = data.get("model", "")
        embedding_model = data.get("embedding_model", EMBEDDING_MODEL)
        string_selection = data.get("string_selection", STRING_SELECTION)
        if isinstance(string_selection, str):
            string_selection = StringSelection(string_selection)
        max_tokens = data.get("max_tokens", JUDGE_MAX_TOKENS)
        method_params_raw = data.get("method_params", {})

        # Convert method_params
        method_params = {}
        for key, value in method_params_raw.items():
            if isinstance(value, dict):
                method_params[key] = MethodParamsOverride.from_dict(value)
            else:
                method_params[key] = value

        # Collect method data from known config_keys
        scoring_data: dict[str, list[str | list[str]]] = {}
        for method_name, params_class, _ in iter_methods():
            config_key = params_class.config_key
            if config_key in data:
                scoring_data[config_key] = data[config_key]

        # Also check for explicit scoring_data dict
        if "scoring_data" in data:
            scoring_data.update(data["scoring_data"])

        return cls(
            model=model,
            embedding_model=embedding_model,
            string_selection=string_selection,
            max_tokens=max_tokens,
            method_params=method_params,
            scoring_data=scoring_data,
        )

    def validate(self) -> None:
        """Validate that required fields are present."""
        # Check if any method has data
        has_any_data = False
        needs_runner = False

        for method_name, params_class, _ in iter_methods():
            items = self.get_method_items(method_name)
            if items:
                has_any_data = True
                if params_class.requires_runner:
                    needs_runner = True

        if not has_any_data:
            raise ValueError("No scoring methods have data configured")

        if needs_runner and not self.model:
            raise ValueError("No model specified for LLM-based scoring methods")

    def get_method_items(self, method_name: str) -> list[str | list[str]]:
        """Get the items for a method from scoring_data.

        Args:
            method_name: The method name (e.g., "categorical")

        Returns:
            List of items (empty if method has no data)
        """
        params_class = get_params_class(method_name)
        config_key = params_class.config_key
        return self.scoring_data.get(config_key, [])

    def has_method_data(self, method_name: str) -> bool:
        """Check if a method has any data configured."""
        return bool(self.get_method_items(method_name))

    def get_scoring_params(self, method: str) -> ScoringMethodParams:
        """Get fully resolved params for a scoring method."""
        params = get_default_params(method)
        if method in self.method_params:
            params = self.method_params[method].apply_to(params)
        return params

    def get_active_methods(self) -> list[str]:
        """Get list of methods that have data configured."""
        active = []
        for method_name, params_class, _ in iter_methods():
            if self.has_method_data(method_name):
                active.append(method_name)
        return active

    def get_structure_labels(self) -> list[str]:
        """Get labels for each structure across all active methods."""
        labels = []
        for method_name in self.get_active_methods():
            params_class = get_params_class(method_name)
            prefix = params_class.label_prefix
            items = self.get_method_items(method_name)
            for i in range(len(items)):
                labels.append(f"{prefix}{i + 1}")
        return labels

    def get_structure_descriptions(self) -> list[str]:
        """Get human-readable descriptions for each structure."""
        descriptions = []
        for method_name in self.get_active_methods():
            items = self.get_method_items(method_name)
            for item in items:
                if isinstance(item, list):
                    descriptions.append(" + ".join(item))
                else:
                    descriptions.append(item)
        return descriptions

    def num_structures(self) -> int:
        """Return number of structures (bundled items count as one)."""
        total = 0
        for method_name in self.get_active_methods():
            total += len(self.get_method_items(method_name))
        return total

    def needs_runner(self) -> bool:
        """Check if any active method requires a model runner."""
        for method_name in self.get_active_methods():
            params_class = get_params_class(method_name)
            if params_class.requires_runner:
                return True
        return False

    def needs_embedder(self) -> bool:
        """Check if any active method requires an embedding runner."""
        for method_name in self.get_active_methods():
            params_class = get_params_class(method_name)
            if params_class.requires_embedder:
                return True
        return False
