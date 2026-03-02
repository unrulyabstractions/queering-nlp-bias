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

from schemas import EstimationOutput, GroupEstimate, JudgmentData, TrajectoryCompliance
from schemas.script_utils import log_step

from src.common.log import log, log_section
from src.common.viz_utils import preview


# ══════════════════════════════════════════════════════════════════════════════
# Formatting Utilities
# ══════════════════════════════════════════════════════════════════════════════


def _fmt_prob(p: float, width: int = 8) -> str:
    """Format probability, using scientific notation for very small values."""
    if p < 0.0001:
        return f"{p:>{width}.1e}"
    return f"{p:>{width}.4f}"


# ══════════════════════════════════════════════════════════════════════════════
# Core Algorithm
# ══════════════════════════════════════════════════════════════════════════════


def estimate_group(
    group_idx: int,
    name: str,
    trajectories: list[TrajectoryCompliance],
) -> GroupEstimate:
    """Estimate normativity for a single group."""
    return GroupEstimate.from_trajectories(group_idx, name, trajectories)


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


def get_group_log_probs(
    trajs: list[TrajectoryCompliance],
    group_name: str,
) -> list[tuple[int, float]]:
    """Get log probabilities for trajectories, conditioned on a specific group.

    Args:
        trajs: Trajectories in this group
        group_name: Name of the conditioning group ("trunk" or branch name)

    Returns:
        List of (traj_idx, log_probability) for trajectories in this group
    """
    result = []
    for t in trajs:
        lp = t.conditional_logprobs.get(group_name, 0.0)
        # Skip trajectories not in this group (logprob = 0.0 marker)
        if group_name != "trunk" and lp == 0.0 and t.branch != group_name:
            continue
        result.append((t.traj_idx, lp))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Steps
# ══════════════════════════════════════════════════════════════════════════════


def step_show_trajectories(data: JudgmentData) -> None:
    """Step 1: Show all trajectories with their scores."""
    log_step(1, "Trajectories", f"{len(data.results)} total")

    # Build scoring structure legend
    cat_labels = []
    sim_labels = []

    for i, q in enumerate(data.categorical_judgements):
        label = f"c{i+1}"
        cat_labels.append(label)

    for i, ref in enumerate(data.similarity_scoring):
        label = f"s{i+1}"
        sim_labels.append(label)

    # Show legend if we have any scoring
    if cat_labels or sim_labels:
        log("    Scoring structures:")
        for i, q in enumerate(data.categorical_judgements):
            log(f"      c{i+1}: \"{preview(q, 50)}\"")
        for i, ref in enumerate(data.similarity_scoring):
            log(f"      s{i+1}: \"{preview(ref, 50)}\"")
        log("")

    # Build header
    all_labels = cat_labels + sim_labels
    header = f"    {'[#]':<4} {'group':<10} " + "  ".join(f"{l:>5}" for l in all_labels)
    log(header)
    log("    " + "─" * (16 + 7 * len(all_labels)))

    for r in data.results:
        idx = r["trajectory_idx"]
        branch_idx = r.get("branch_idx", 0)
        display_name = "trunk" if branch_idx == 0 else f"branch_{branch_idx}"
        scores = r.get("scores", [])
        sims = r.get("similarity_scores", [])

        # Format each score individually
        score_parts = []
        for s in scores:
            score_parts.append(f"{s:>5}" if s is not None else "    ?")
        for s in sims:
            score_parts.append(f"{s:>5.2f}")
        scores_str = "  ".join(score_parts)

        text = data.get_text(idx)
        log(f"    [{idx}] {display_name:<10} {scores_str}")
        log(f"        {preview(text, 70)}")


def step_estimate_groups(
    data: JudgmentData,
    grouped: dict[str, list[TrajectoryCompliance]],
) -> list[GroupEstimate]:
    """Step 2: Estimate normativity for all groups with mass breakdown."""
    # Trunk = all trajectories pooled
    all_trajs = [t for trajs in grouped.values() for t in trajs]
    # Use config order from branches list
    branch_names = data.branches if data.branches else ["trunk"]

    # Build display name mapping: trunk stays trunk, others become branch_N
    display_names = {"trunk": "trunk"}
    branch_idx = 1
    for name in branch_names:
        if name != "trunk":
            display_names[name] = f"branch_{branch_idx}"
            branch_idx += 1

    log_step(2, "Group Statistics", f"{len(branch_names)} groups")

    # Show group definitions
    if data.groups:
        log("    Conditioning text per group:")
        for name in branch_names:
            text = data.groups.get(name, "")
            display = display_names.get(name, name)
            log(f"      {display}: \"{preview(text, 55)}\"")
        log("")

    # Show prefix logprobs if available
    if data.prefix_logprobs:
        log("    Prefix conditional logprobs:")
        trunk_lp = data.prefix_logprobs.get("trunk_given_prompt", 0.0)
        trunk_p = math.exp(trunk_lp) if trunk_lp > -700 else 0.0
        log(f"      p(trunk|prompt): {trunk_lp:.2f} (p={_fmt_prob(trunk_p)})")
        branch_lps = data.prefix_logprobs.get("branch_given_trunk", {})
        # Iterate by branch_idx (1, 2, 3, ...)
        for branch_idx in range(1, len(branch_names)):
            # Keys may be int or str (from JSON)
            lp = branch_lps.get(branch_idx) or branch_lps.get(str(branch_idx))
            if lp is not None:
                prob = math.exp(lp)
                log(f"      p(branch_{branch_idx}|trunk): {lp:.2f} (p={_fmt_prob(prob)})")
        log("")

    groups = []
    for idx, name in enumerate(branch_names):
        is_trunk = name == "trunk"
        trajs = all_trajs if is_trunk else grouped[name]

        # Get log probs conditioned on this group
        log_probs = get_group_log_probs(trajs, name)
        traj_probs = compute_normalized_probs(log_probs)

        # Build group header
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
        log(f"    [#]   logp      p        p_norm   ppl   inv_ppl_norm")
        log(f"    " + "─" * 55)

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
        norm_inv_ppls = {k: v / total_inv_ppl if total_inv_ppl > 0 else 0 for k, v in inv_ppls.items()}

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

            log(f"    [{traj_idx}]  {logp:>6.0f}  {_fmt_prob(p)}  {norm_p:>6.1%}  {ppl_str}  {inv_ppl_str}")
        log("")

        estimate = estimate_group(idx, name, trajs)
        groups.append(estimate)

    return groups


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


def estimate_normativity(data: JudgmentData, scores_path: Path) -> None:
    """Run normativity estimation pipeline.

    Pipeline:
        1. Show trajectories with scores
        2. Estimate groups with mass breakdown
        3. Save output
    """
    log_section("Normativity Estimation")
    log(f"  Input: {scores_path}")

    # Step 1: Show all trajectories
    step_show_trajectories(data)

    # Step 2: Estimate groups
    grouped = data.group_by_branch()
    groups = step_estimate_groups(data, grouped)

    # Build output
    output = EstimationOutput.create(
        judgment_file=str(scores_path),
        categorical_judgements=data.categorical_judgements,
        similarity_scoring=data.similarity_scoring,
        groups=groups,
        texts=data.get_texts(),
    )

    # Step 3: Save output
    step_save_output(output, scores_path)

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
