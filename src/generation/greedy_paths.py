"""Generate one greedy (temperature=0) trajectory per arm.

Used both by the production generation script (writes greedy.json after
the main pipeline) and by the ad-hoc backfill scripts.
"""

from __future__ import annotations

from pathlib import Path

from src.common.callback_types import LogFn
from src.inference import ModelRunner

from .generation_config import GenerationConfig
from .greedy_output import GreedyArmEntry, GreedyOutput


def produce_greedy_paths(
    runner: ModelRunner,
    config: GenerationConfig,
    *,
    config_path: str | Path = "",
    log_fn: LogFn | None = None,
) -> GreedyOutput:
    """Generate one greedy trajectory per arm and return a `GreedyOutput`.

    Each arm uses its own prefill; temperature is fixed at 0. The
    returned object has no `structure_scores` populated — that is the
    scoring step's responsibility.
    """
    arms = config.get_arms(runner.skip_thinking_prefix)
    entries: list[GreedyArmEntry] = []

    for arm in arms:
        if log_fn:
            log_fn(f"  greedy-decoding arm: {arm.name}")

        traj = runner.generate_trajectory_from_prompt(
            prompt=config.prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=0.0,
            prefilling=arm.prefill,
        )

        text = traj.generated_text or ""
        token_ids = list(traj.token_ids or [])
        prefill_length = traj.prefill_length or 0
        n_generated_tokens = max(0, len(token_ids) - prefill_length)

        entries.append(
            GreedyArmEntry(
                name=arm.name,
                text=text,
                n_generated_tokens=n_generated_tokens,
                token_ids=token_ids,
                prefill_length=prefill_length,
            )
        )

    return GreedyOutput.create(
        model=config.model,
        config_path=config_path,
        arms=entries,
    )
