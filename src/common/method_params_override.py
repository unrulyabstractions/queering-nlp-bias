"""Shared parameter override mechanism for config systems.

Provides a generic way to apply config-time customizations to default
method parameters in both generation and scoring pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.common.base_schema import BaseSchema


@dataclass
class MethodParamsOverride(BaseSchema):
    """Override values for method parameters.

    Wraps a flat dict of parameter overrides. Used to apply config-time
    customizations to default method parameters.
    """

    overrides: dict[str, Any] = field(default_factory=dict)

    def apply_to(self, params: Any) -> Any:
        """Apply these overrides to a params instance.

        Args:
            params: Any dataclass-style params object with settable attributes

        Returns:
            The modified params instance (for chaining)
        """
        for key, value in self.overrides.items():
            if hasattr(params, key):
                setattr(params, key, value)
        return params
