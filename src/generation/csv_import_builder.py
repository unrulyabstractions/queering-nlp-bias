"""Build GenerationOutput from pre-generated CSV text.

Converts a flat CSV dataset of LLM generations into the generation.json format
expected by the scoring, estimation, and visualization pipeline stages.
No model is run — token arrays are empty and logprobs are zero throughout.

The resulting output is compatible with all text-based scoring methods.
Logprob-based weighting (prob, inv-ppl) degrades to uniform weighting since
all conditional_logprobs are 0.0, which is equivalent to uniform via logsumexp.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from src.common.base_schema import BaseSchema
from src.common.experiment_types import GenerationArm
from src.common.token_trajectory import TokenTrajectory
from src.common.token_tree import TokenTree

from .generation_output import OUTPUT_VERSION, GenerationMetadata, GenerationOutput


# ══════════════════════════════════════════════════════════════════════════════
# INPUT SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class CsvRow(BaseSchema):
    """A single row from the input CSV."""

    text: str
    label: str


@dataclass
class CsvImportConfig(BaseSchema):
    """Configuration recorded in the generation metadata for a CSV import."""

    model: str = "external"
    prompt: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# BUILDERS
# ══════════════════════════════════════════════════════════════════════════════


def build_generation_output_from_csv(
    rows: list[CsvRow],
    config: CsvImportConfig,
) -> GenerationOutput:
    """Convert CSV rows into a GenerationOutput ready for the scoring pipeline.

    Args:
        rows: List of (text, label) pairs from the CSV.
        config: Metadata to record (model name, prompt text).

    Returns:
        GenerationOutput with empty token arrays. All arms are flat (no trunk).
    """
    arms = _build_arms(rows)
    arm_index = {arm.name: i for i, arm in enumerate(arms)}
    trajs = _build_trajectories(rows, arm_index)
    tree = TokenTree.from_trajectories(
        trajs,
        groups_per_traj=[[arm_index[row.label]] for row in rows],
    )
    config_dict = _build_config_dict(arms, config)
    metadata = GenerationMetadata(
        version=OUTPUT_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        model=config.model,
        method="csv-import",
        num_trajectories=len(rows),
        eos_token=None,
    )
    return GenerationOutput(metadata=metadata, config=config_dict, tree=tree.to_dict())


def _build_arms(rows: list[CsvRow]) -> list[GenerationArm]:
    """Derive one flat arm per unique label, preserving first-seen order."""
    seen: dict[str, GenerationArm] = {}
    for row in rows:
        if row.label not in seen:
            seen[row.label] = GenerationArm(
                name=row.label,
                prefill="",
                parent_idx=None,
            )
    return list(seen.values())


def _build_trajectories(
    rows: list[CsvRow],
    arm_index: dict[str, int],
) -> list[TokenTrajectory]:
    """Create one TokenTrajectory per row with empty token arrays."""
    trajs = []
    for i, row in enumerate(rows):
        traj = TokenTrajectory(
            token_ids=[],
            logprobs=[],
            logits=[],
            prefill_text="",
            generated_text=row.text,
            arm_token_lengths=None,
            arm_text_lengths=None,
            traj_idx=i,
            arm_idx=(arm_index[row.label],),
        )
        trajs.append(traj)
    return trajs


def _build_config_dict(
    arms: list[GenerationArm],
    config: CsvImportConfig,
) -> dict:
    """Build the config dict that GenerationOutputData.load() expects."""
    return {
        "model": config.model,
        "prompt": config.prompt,
        "trunk": "",
        "branches": [],
        "arms": [arm.to_dict() for arm in arms],
    }
