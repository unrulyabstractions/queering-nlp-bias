"""Score trajectories with registered scoring methods.

Usage:
    python scripts/score_trajectories.py trials/scoring/<scoring>.json out/<method>/gen_<gen>.json
    python scripts/score_trajectories.py trials/scoring/<scoring>.json out/<method>/gen_<gen>.json --arm trunk
    python scripts/score_trajectories.py trials/scoring/<scoring>.json out/<method>/gen_<gen>.json --arm trunk --skip-greedy

Outputs:
    out/<method>/<scoring>/scoring.json                    (full run)
    out/<method>/<scoring>/_arm_<arm>/scoring.json         (--arm <arm>)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.logging import log, log_items, log_kv, log_section, log_step
from src.generation.greedy_output import GreedyOutput
from src.scoring import (
    GenerationOutputData,
    ScoringOutput,
    ScoringConfig,
    TrajectoryData,
    run_scoring_pipeline,
    get_params_class,
)
from src.scoring.methods.logging.scoring_logging_utils import log_trajectory_header
from src.scoring.scorer import Scorer, get_text_for_scoring


def score_trajectories(
    config: ScoringConfig,
    scoring_path: Path,
    gen_data: GenerationOutputData,
    gen_path: Path,
    *,
    arm: str | None = None,
    skip_greedy: bool = False,
) -> None:
    """Run scoring pipeline with logging.

    Args:
        config: Scoring config.
        scoring_path: Path to scoring config (used for output naming).
        gen_data: Loaded generation output (full set of trajectories).
        gen_path: Path to generation.json.
        arm: If given, only score trajectories whose `arm_name` matches this
            value, and write to a per-arm sidecar at
            `<scoring_dir>/_arm_<arm>/scoring.json`.
        skip_greedy: If True, do not score greedy.json after the run.
            Used by per-arm parallel orchestration to defer greedy scoring
            to a single post-merge pass.
    """
    if arm is not None:
        gen_data = _filter_to_arm(gen_data, arm)

    log_section("Scoring Pipeline")
    log_kv("Scoring config", str(scoring_path))
    log_kv("Generation output", str(gen_path))
    if arm is not None:
        log_kv("Arm filter", arm)
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

    # Save outputs (and copy original config). Per-arm runs go to a
    # `_arm_<arm>/` sidecar so multiple arm jobs can run in parallel
    # without clobbering each other.
    log_step(3, "Save output")
    out_path = ScoringOutput.compute_output_path(gen_path, scoring_path)
    if arm is not None:
        out_path = out_path.parent / f"_arm_{arm}" / "scoring.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline_result.output.save(out_path, config_path=scoring_path)
    log(f"  Saved judgments to {out_path}")

    summary_path = out_path.parent / "summary_scoring.txt"
    pipeline_result.output.save_summary(summary_path)
    log(f"  Saved summary to {summary_path}")

    # Score greedy.json once at the end (skipped on per-arm runs; the
    # parallel orchestrator handles greedy in a single post-merge pass).
    if not skip_greedy:
        greedy_path = GreedyOutput.compute_path(gen_path)
        if greedy_path.exists():
            log_step(4, "Score greedy paths", str(greedy_path))
            greedy_output = GreedyOutput.load(greedy_path)
            scorer = Scorer(config)
            for entry in greedy_output.arms:
                entry.structure_scores = scorer.score(entry.text)
                log(
                    f"  {entry.name}: scores={[round(s, 3) for s in entry.structure_scores]}"
                )
            sidecar = out_path.parent / "greedy.json"
            greedy_output.save(sidecar)
            log(f"  Wrote {sidecar}")

    pipeline_result.output.summarize()


def _filter_to_arm(gen_data: GenerationOutputData, arm: str) -> GenerationOutputData:
    """Return a shallow copy of `gen_data` containing only the named arm's trajs."""
    keep = [t for t in gen_data.trajectories if t.arm_name == arm]
    if not keep:
        raise ValueError(
            f"--arm {arm}: no trajectories found in generation output. "
            f"Available arms: {sorted({t.arm_name for t in gen_data.trajectories})}"
        )
    return GenerationOutputData(
        tree=gen_data.tree,
        trajectories=keep,
        config=gen_data.config,
        arms=gen_data.arms,
        eos_token=gen_data.eos_token,
    )


def main() -> None:
    """Parse arguments and run scoring pipeline."""
    parser = argparse.ArgumentParser(
        description="Score trajectories with scoring config"
    )
    parser.add_argument("scoring_config", help="Path to scoring config JSON")
    parser.add_argument("generation_output", help="Path to generation output JSON")
    parser.add_argument(
        "--arm",
        default=None,
        help="Score only this arm's trajectories (writes to _arm_<arm>/ sidecar).",
    )
    parser.add_argument(
        "--skip-greedy",
        action="store_true",
        help="Skip the greedy.json scoring pass (used by parallel orchestration).",
    )
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
        arm=args.arm,
        skip_greedy=args.skip_greedy,
    )


if __name__ == "__main__":
    main()
