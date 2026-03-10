#!/usr/bin/env python3
"""Estimate normativity from scoring results.

Usage:
    python scripts/estimate_normativity.py out/score_<name>.json

Outputs:
    out/est_<name>.json

Computes structure-aware diversity metrics:
- Core: Expected system compliance (average scores)
- Orientation: Deviation from core per trajectory
- Deviance: Scalar non-normativity (orientation magnitude)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import EstimationOutput, ArmEstimate, JudgmentData, TrajectoryCompliance
from schemas.log_utils import fmt_prob, log_banner, log_divider, log_step

from src.common.log import log
from src.common.viz_utils import preview

# ══════════════════════════════════════════════════════════════════════════════
# Core Algorithm
# ══════════════════════════════════════════════════════════════════════════════


def estimate_arm(
    arm_idx: int,
    name: str,
    trajectories: list[TrajectoryCompliance],
    reference_core: list[float] | None = None,
    reference_core_inv_ppl: list[float] | None = None,
) -> ArmEstimate:
    """Estimate normativity for a single arm (trunk or branch).

    Args:
        arm_idx: Index of this arm (0=trunk, 1+=branches)
        name: Name of this arm
        trajectories: Trajectories with compliance scores
        reference_core: Optional core for computing E[θ] (e.g., trunk core)
        reference_core_inv_ppl: Optional inv-ppl core for E[θ]
    """
    return ArmEstimate.from_trajectories(
        arm_idx, name, trajectories, reference_core, reference_core_inv_ppl
    )


def compute_normalized_probs(
    log_probs: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    """Compute normalized probabilities from log probabilities.

    Args:
        log_probs: List of (traj_idx, log_probability)

    Returns:
        List of (traj_idx, normalized_prob) sorted by probability descending.
    """
    if not log_probs:
        return []

    max_lp = max(lp for _, lp in log_probs)

    # Compute relative probabilities
    probs = [(idx, math.exp(lp - max_lp)) for idx, lp in log_probs]
    total = sum(p for _, p in probs)

    if total <= 0:
        return [(idx, 1.0 / len(probs)) for idx, _ in probs]

    # Normalize and sort by probability descending
    normalized = [(idx, p / total) for idx, p in probs]
    return sorted(normalized, key=lambda x: -x[1])


def get_arm_log_probs(
    trajs: list[TrajectoryCompliance],
    arm_name: str,
) -> list[tuple[int, float]]:
    """Get log probabilities for trajectories, conditioned on a specific arm.

    Args:
        trajs: Trajectories in this arm
        arm_name: Name of the conditioning arm ("trunk" or branch name)

    Returns:
        List of (traj_idx, log_probability) for trajectories in this arm
    """
    result = []
    for t in trajs:
        lp = t.conditional_logprobs.get(arm_name, 0.0)
        # Skip trajectories not in this arm (logprob = 0.0 marker)
        if arm_name != "trunk" and lp == 0.0 and t.branch != arm_name:
            continue
        result.append((t.traj_idx, lp))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Steps
# ══════════════════════════════════════════════════════════════════════════════


def step_show_trajectories(data: JudgmentData) -> None:
    """Step 1: Show all trajectories with their scores."""
    log_step(1, "Trajectories", f"{len(data.results)} total")

    # Get structure info for proper labeling
    structures = data.get_structure_info()

    # Show legend if we have any scoring - full questions, not truncated
    if structures:
        log("    Scoring structures:")
        for s in structures:
            bundled_marker = " [BUNDLED]" if s.is_bundled else ""
            if s.is_bundled:
                log(
                    f"      {s.label}: BUNDLED ({len(s.questions)} questions){bundled_marker}"
                )
                for q in s.questions:
                    log(f"          • {q}")
            else:
                log(f"      {s.label}: {s.description}{bundled_marker}")
        log("")

    # Build header with structure labels
    labels = [s.label for s in structures]
    header = f"    {'#':>3}  {'arm':<10} " + "  ".join(f"{l:>5}" for l in labels)
    log(header)
    log_divider(18 + 7 * len(labels), indent="    ")

    for i, r in enumerate(data.results):
        idx = r["trajectory_idx"]
        branch = r.get("branch", "trunk")
        branch_idx = r.get("branch_idx", 0)
        display_name = "trunk" if branch_idx == 0 else f"branch_{branch_idx}"

        # Get compliance values (handles grouping)
        compliance = data.get_compliance(r)

        # Format each compliance value
        score_parts = []
        for c in compliance:
            score_parts.append(f"{c:>5.2f}")
        scores_str = "  ".join(score_parts)

        # Extract just the continuation after the prefix
        text = data.get_text(idx)
        prefix = data.arm_texts.get(branch, "")
        continuation = _extract_continuation(text, prefix)

        # Add spacing before each entry (except first)
        log(f"    {idx:>3}  {display_name:<10} {scores_str}", gap=1 if i > 0 else 0)
        log(f'         "{preview(continuation, 65)}"')


def step_estimate_arms(
    data: JudgmentData,
    by_branch: dict[str, list[TrajectoryCompliance]],
) -> list[ArmEstimate]:
    """Step 2: Estimate normativity for all arms with mass breakdown."""
    # Trunk = all trajectories pooled
    all_trajs = [t for trajs in by_branch.values() for t in trajs]
    # Use config order from branches list
    branch_names = data.branches if data.branches else ["trunk"]

    # Build display name mapping: trunk stays trunk, others become branch_N
    display_names = {"trunk": "trunk"}
    branch_idx = 1
    for name in branch_names:
        if name != "trunk":
            display_names[name] = f"branch_{branch_idx}"
            branch_idx += 1

    log_step(2, "Arm Statistics", f"{len(branch_names)} arms")

    # Show arm definitions - only the part after the prompt (strip chat template)
    if data.arm_texts:
        log("    Conditioning text per arm (after prompt):")
        for name in branch_names:
            text = data.arm_texts.get(name, "")
            display = display_names.get(name, name)
            # Extract just the meaningful part (after chat template markers)
            clean_text = _extract_branch_prefix(text)
            log(f'      {display}: "{clean_text}"')
        log("")

    # Show prefix logprobs if available
    if data.prefix_logprobs:
        log("    Prefix conditional logprobs:")
        trunk_lp = data.prefix_logprobs.get("trunk_given_prompt", 0.0)
        trunk_p = math.exp(trunk_lp) if trunk_lp > -700 else 0.0
        log(f"      p(trunk|prompt): {trunk_lp:.2f} (p={fmt_prob(trunk_p)})")
        branch_lps = data.prefix_logprobs.get("branch_given_trunk", {})
        # Iterate by branch_idx (1, 2, 3, ...)
        for branch_idx in range(1, len(branch_names)):
            # Keys may be int or str (from JSON)
            lp = branch_lps.get(branch_idx) or branch_lps.get(str(branch_idx))
            if lp is not None:
                prob = math.exp(lp)
                log(
                    f"      p(branch_{branch_idx}|trunk): {lp:.2f} (p={fmt_prob(prob)})"
                )
        log("")

    arms = []
    trunk_core: list[float] | None = None
    trunk_core_inv_ppl: list[float] | None = None

    for idx, name in enumerate(branch_names):
        is_trunk = name == "trunk"
        trajs = all_trajs if is_trunk else by_branch[name]

        # Get log probs conditioned on this arm
        log_probs = get_arm_log_probs(trajs, name)
        traj_probs = compute_normalized_probs(log_probs)

        # Build arm header
        display = display_names.get(name, name)
        header = f"<{idx}> {display} ({len(trajs)} trajectories)"
        if not is_trunk and trajs:
            # p0 = exp(logp_trunk - logp_branch) for the first trajectory
            t = trajs[0]
            lp_trunk = t.conditional_logprobs.get("trunk", 0.0)
            lp_branch = t.conditional_logprobs.get(name, 0.0)
            if lp_trunk != 0.0 and lp_branch != 0.0:
                p0 = math.exp(lp_trunk - lp_branch)
                header += f"  p₀={p0:.1%}"

        log(f"    {header}")
        log(
            f"    {'#':>3}  {'logp':>6}  {'p':>10}  {'p_norm':>7}  {'ppl':>5}  {'inv_ppl_n':>9}"
        )
        log_divider(55, indent="    ")

        # Build lookup for trajectory data
        traj_lookup = {t.traj_idx: t for t in trajs}
        log_prob_dict = dict(log_probs)

        # First pass: compute inverse perplexities for normalization
        inv_ppls = {}
        for traj_idx, _ in traj_probs:
            logp = log_prob_dict.get(traj_idx, 0.0)
            traj = traj_lookup.get(traj_idx)
            n_tokens = traj.n_continuation_tokens if traj else 0
            if n_tokens > 0 and logp > -700:
                avg_logp = logp / n_tokens
                inv_ppls[traj_idx] = math.exp(avg_logp)  # 1/ppl = exp(avg_logp)
            else:
                inv_ppls[traj_idx] = 0.0

        # Normalize inverse perplexities
        total_inv_ppl = sum(inv_ppls.values())
        norm_inv_ppls = {
            k: v / total_inv_ppl if total_inv_ppl > 0 else 0
            for k, v in inv_ppls.items()
        }

        # Show mass breakdown with perplexity
        for traj_idx, norm_p in traj_probs:
            logp = log_prob_dict.get(traj_idx, 0.0)
            p = math.exp(logp) if logp > -700 else 0.0

            traj = traj_lookup.get(traj_idx)
            n_tokens = traj.n_continuation_tokens if traj else 0
            if n_tokens > 0 and logp > -700:
                avg_logp = logp / n_tokens
                ppl = math.exp(-avg_logp)
                ppl_str = f"{ppl:>5.1f}"
                inv_ppl_norm = norm_inv_ppls.get(traj_idx, 0.0)
                inv_ppl_str = f"{inv_ppl_norm:>6.1%}"
            else:
                ppl_str = "    -"
                inv_ppl_str = "     -"

            log(
                f"    {traj_idx:>3}  {logp:>6.0f}  {fmt_prob(p)}  {norm_p:>7.1%}  {ppl_str}  {inv_ppl_str:>9}"
            )
        log("")

        # For trunk, E[θ] is relative to its own core (= 0 by definition)
        # For branches, E[θ] is relative to trunk core (shows deviation from "normal")
        if is_trunk:
            estimate = estimate_arm(idx, name, trajs)
            # Save trunk core for computing branch E[θ]
            trunk_core = estimate.core
            trunk_core_inv_ppl = estimate.core_inv_ppl
        else:
            estimate = estimate_arm(idx, name, trajs, trunk_core, trunk_core_inv_ppl)
        arms.append(estimate)

    return arms


def step_save_output(output: EstimationOutput, scores_path: Path) -> Path:
    """Step 3: Save estimation output."""
    out_path = EstimationOutput.compute_output_path(scores_path)
    log_step(3, "Save output", str(out_path))

    output.save(out_path)
    log(f"    Saved to {out_path}")

    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════


def _extract_continuation(text: str, prefix: str) -> str:
    """Extract continuation from full text by removing prefix."""
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def _extract_branch_prefix(full_prefix: str) -> str:
    """Extract just the generation prefix from full chat template text.

    Removes chat template markers like <|im_start|>user, <|im_end|>, <think>, etc.
    Returns just the meaningful text portion.
    """
    # Find the actual content after all chat template markers
    # Look for the last </think> and newlines
    if "</think>" in full_prefix:
        # Get everything after </think> and strip
        after_think = full_prefix.split("</think>")[-1].strip()
        return after_think
    return full_prefix


def step_show_continuations(data: JudgmentData) -> None:
    """Step 0: Show all continuations organized by branch."""
    log_step(0, "Continuations by Branch")

    continuations = data.get_continuations_by_branch()
    branch_names = data.branches if data.branches else ["trunk"]

    # Build branch index mapping for display
    branch_display = {"trunk": "trunk"}
    for i, name in enumerate(branch_names):
        if name != "trunk":
            branch_display[name] = f"branch_{i}"

    # First, show the structure
    if data.arm_texts:
        trunk_prefix = _extract_branch_prefix(data.arm_texts.get("trunk", ""))
        log(f'\n    Trunk prefix: "{trunk_prefix}"')
        for i, branch in enumerate(branch_names):
            if branch != "trunk":
                branch_prefix = _extract_branch_prefix(data.arm_texts.get(branch, ""))
                # Show what's added for this branch
                added = (
                    branch_prefix[len(trunk_prefix) :]
                    if branch_prefix.startswith(trunk_prefix)
                    else branch
                )
                log(f'    + "{added}" -> branch_{i}')

    for i, branch in enumerate(branch_names):
        items = continuations.get(branch, [])
        if not items:
            continue

        display_name = branch_display.get(branch, f"branch_{i}")
        prefix = data.arm_texts.get(branch, "")
        clean_prefix = _extract_branch_prefix(prefix)

        log(f"\n    {display_name} ({len(items)} trajectories)")
        log(f'    Full continuation starts with: "{clean_prefix}..."')
        log_divider(70, indent="    ")

        for idx, text in items:
            # Extract just the continuation after the full prefix
            continuation = _extract_continuation(text, prefix)
            # Show first 100 chars of continuation
            first_line = continuation[:100].replace("\n", " ")
            if len(continuation) > 100:
                first_line += "..."
            log(f"      [{idx}] {first_line}")


def step_save_summary(output: EstimationOutput, scores_path: Path) -> Path:
    """Step 4: Save summary output."""
    summary_path = EstimationOutput.compute_summary_path(scores_path)
    log_step(4, "Save summary", str(summary_path))

    output.save_summary(summary_path)
    log(f"    Saved to {summary_path}")

    return summary_path


def estimate_normativity(data: JudgmentData, scores_path: Path) -> None:
    """Run normativity estimation pipeline.

    Pipeline:
        0. Show continuations by branch
        1. Show trajectories with scores
        2. Estimate groups with mass breakdown
        3. Save output
        4. Save summary
        5. Print summary
    """
    log_banner("NORMATIVITY ESTIMATION")
    log(f"\n  Input: {scores_path}")
    log(f"  Generation file: {data.generation_file}")
    if data.judge_model:
        log(f"  Judge model: {data.judge_model}")
    if data.embedding_model:
        log(f"  Embedding model: {data.embedding_model}")

    # Step 0: Show continuations by branch
    step_show_continuations(data)

    # Step 1: Show all trajectories
    step_show_trajectories(data)

    # Step 2: Estimate arms
    by_branch = data.group_by_branch()
    arms = step_estimate_arms(data, by_branch)

    # Compute additional summary data
    structure_info = data.get_structure_info()
    branch_rates = data.compute_branch_rates()
    continuations_by_branch = data.get_continuations_by_branch()

    # Build output
    output = EstimationOutput.create(
        judgment_file=str(scores_path),
        categorical_judgements=data.categorical_judgements,
        similarity_scoring=data.similarity_scoring,
        arms=arms,
        texts=data.get_texts(),
        generation_file=data.generation_file,
        scoring_file=data.scoring_file,
        judge_model=data.judge_model,
        embedding_model=data.embedding_model,
        structure_info=structure_info,
        branch_rates=branch_rates,
        continuations_by_branch=continuations_by_branch,
    )

    # Step 3: Save output
    step_save_output(output, scores_path)

    # Step 4: Save summary
    step_save_summary(output, scores_path)

    # Summary
    output.summarize()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate normativity from scores")
    parser.add_argument("scores", help="Path to scoring output JSON")
    args = parser.parse_args()

    scores_path = Path(args.scores)
    data = JudgmentData.load(scores_path)

    estimate_normativity(data=data, scores_path=scores_path)


if __name__ == "__main__":
    main()
