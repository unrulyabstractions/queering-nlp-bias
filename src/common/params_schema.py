"""Base schema for parameter classes with CLI-style printing.

This module provides the root base class for all parameter schemas.
Each pipeline (generation, scoring, estimation) has its own params
base class that extends this one.

Hierarchy:
    ParamsSchema (base)
    ├── GenerationParams
    │   ├── SamplingParams
    │   ├── ForkingParams
    │   └── EntropySeekingParams
    ├── ScoringParams
    │   └── ... (scoring method params)
    └── EstimationParams
        └── ... (estimation method params)
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, fields
from typing import ClassVar

from src.common.base_schema import BaseSchema
from src.common.callback_types import LogFn
from src.common.log import log as default_log


@dataclass
class ParamsSchema(BaseSchema, ABC):
    """Base class for all parameter schemas with CLI-style printing.

    Subclasses should define a _cli_args class variable mapping
    field names to CLI argument names for display purposes.

    Example:
        @dataclass
        class MyParams(ParamsSchema):
            count: int
            threshold: float

            _cli_args: ClassVar[dict[str, str]] = {
                "count": "--count",
                "threshold": "--threshold",
            }
    """

    # Subclasses define: field_name -> "--cli-arg-name"
    _cli_args: ClassVar[dict[str, str]] = {}

    def print(self, log_fn: LogFn | None = None) -> None:
        """Print parameters as CLI arguments.

        Args:
            log_fn: Optional logging function. Uses src.common.log if None.
        """
        if log_fn is None:
            log_fn = default_log

        log_fn("  Parameters:")
        for field_name, cli_arg in self._cli_args.items():
            value = getattr(self, field_name, None)
            if value is not None:
                log_fn(f"    {cli_arg} {value}")

    def get_params_dict(self) -> dict[str, any]:
        """Return parameters as a dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
