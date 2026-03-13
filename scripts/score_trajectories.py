"""Score trajectories with registered scoring methods.

Usage:
    python scripts/score_trajectories.py trials/scoring/<scoring>.json out/<method>/gen_<gen>.json

Outputs:
    out/<method>/score_<gen>_<scoring>.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.logging import log, log_items, log_kv, log_section, log_step
from src.scoring import (
    GenerationOutputData,
    ScoringOutput,
    ScoringConfig,
    TrajectoryData,
    run_scoring_pipeline,
    get_params_class,
)
from src.scoring.methods.logging.scoring_logging_utils import log_trajectory_header
from src.scoring.scorer import get_text_for_scoring


def score_trajectories(
    config: ScoringConfig,
    scoring_path: Path,
    gen_data: GenerationOutputData,
    gen_path: Path,
) -> None:
    """Run scoring pipeline with logging."""
    log_section("Scoring Pipeline")
    log_kv("Scoring config", str(scoring_path))
    log_kv("Generation output", str(gen_path))
    log_kv("Trajectories", str(len(gen_data.trajectories)))

    # Log all active methods dynamically
    for method_name in config.get_active_methods():
        items = config.get_method_items(method_name)
        params_class = get_params_class(method_name)
        log_items(
            f"{method_name} ({len(items)}):",
            items,
            prefix=params_class.label_prefix,
        )

    log_kv("String selection", config.string_selection.value)

    # Progress callback for logging
    def on_progress(current: int, total: int, traj: TrajectoryData) -> None:
        selected_text = get_text_for_scoring(traj, config)
        log_trajectory_header(traj, current, total, log_section, selected_text)

    # Indented log function for scoring details
    def indented_log(msg: str) -> None:
        log(f"    {msg}")

    log_step(1, "Load models")
    if config.model:
        log(f"  Judge model: {config.model}")
    if config.embedding_model:
        log(f"  Embedding model: {config.embedding_model}")
    log("")
    log_step(2, "Score trajectories", f"{len(gen_data.trajectories)} trajectories")

    # Run the pipeline
    pipeline_result = run_scoring_pipeline(
        config=config,
        trajectories=gen_data.trajectories,
        arm_names=gen_data.arm_names,
        arm_texts=gen_data.arm_texts,
        generation_file=str(gen_path),
        scoring_file=str(scoring_path),
        progress_fn=on_progress,
        log_fn=indented_log,
    )

    # Save outputs (and copy original config)
    log_step(3, "Save output")
    out_path = ScoringOutput.compute_output_path(gen_path, scoring_path)
    pipeline_result.output.save(out_path, config_path=scoring_path)
    log(f"  Saved judgments to {out_path}")

    summary_path = ScoringOutput.compute_summary_path(gen_path, scoring_path)
    pipeline_result.output.save_summary(summary_path)
    log(f"  Saved summary to {summary_path}")

    pipeline_result.output.summarize()


def main() -> None:
    """Parse arguments and run scoring pipeline."""
    parser = argparse.ArgumentParser(
        description="Score trajectories with scoring config"
    )
    parser.add_argument("scoring_config", help="Path to scoring config JSON")
    parser.add_argument("generation_output", help="Path to generation output JSON")
    args = parser.parse_args()

    scoring_path = Path(args.scoring_config)
    gen_path = Path(args.generation_output)
    config = ScoringConfig.load(scoring_path)
    gen_data = GenerationOutputData.load(gen_path)

    score_trajectories(
        config=config,
        scoring_path=scoring_path,
        gen_data=gen_data,
        gen_path=gen_path,
    )


if __name__ == "__main__":
    main()
