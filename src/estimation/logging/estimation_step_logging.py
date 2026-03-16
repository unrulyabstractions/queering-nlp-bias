"""Logging utilities for estimation pipeline.

This module provides detailed logging functions for the estimation pipeline,
including trajectory tables, arm statistics, and continuation displays.
"""

from __future__ import annotations

import math

from src.common.logging import fmt_prob, log, log_divider, log_step
from src.common.viz_utils import escape_newlines, preview
from src.estimation.arm_types import get_parent_branch

from ..estimation_scoring_data import ScoringData
from ..estimation_structure import TrajectoryScoringData


def log_continuations_by_branch(data: ScoringData) -> None:
    """Step 0: Show all continuations organized by arm."""
    log_step(0, "Continuations by Arm")

    continuations = data.get_continuations_by_arm()
    arm_names = data.arm_names if data.arm_names else ["trunk"]

    # First, show the structure
    if data.arm_texts:
        trunk_prefix = escape_newlines(data.arm_texts.get("trunk", ""))
        log(f'\n    Trunk prefix: "{trunk_prefix}"')
        for arm_name in arm_names:
            if arm_name not in ("root", "trunk"):
                arm_prefix = escape_newlines(data.arm_texts.get(arm_name, ""))
                # Show what's added for this arm
                added = (
                    arm_prefix[len(trunk_prefix) :]
                    if arm_prefix.startswith(trunk_prefix)
                    else arm_name
                )
                log(f'    + "{added}" -> {arm_name}')

    for arm_name in arm_names:
        items = continuations.get(arm_name)
        if not items:
            continue

        prefix = data.arm_texts.get(arm_name, "")
        clean_prefix = escape_newlines(prefix)

        log(f"\n    {arm_name} ({len(items)} trajectories)")
        log(f'    Prefill: "{clean_prefix}"')
        log_divider(70, indent_str="    ")

        for cont in items:
            # Text is already just the generated portion
            first_line = cont.text[:100].replace("\n", " ")
            if len(cont.text) > 100:
                first_line += "..."
            log(f"      [{cont.traj_idx}] {first_line}")


def log_trajectories_with_scores(data: ScoringData) -> None:
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
    log_divider(18 + 7 * len(labels), indent_str="    ")

    for i, r in enumerate(data.results):
        idx = r["traj_idx"]
        arm = r.get("arm", "trunk")
        display_name = arm  # Use actual arm name

        # Get compliance values (handles grouping)
        compliance = data.get_structure_scores(r)

        # Format each compliance value
        score_parts = []
        for c in compliance:
            score_parts.append(f"{c:>5.2f}")
        scores_str = "  ".join(score_parts)

        # Text is already just the generated_text (not prefill + generated)
        text = data.get_text(idx)

        # Add spacing before each entry (except first)
        log(f"    {idx:>3}  {display_name:<10} {scores_str}", gap=1 if i > 0 else 0)
        log(f'         "{preview(text, 65)}"')


def compute_normalized_probs(
    log_probs: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    """Compute normalized probabilities from log probabilities."""
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
    trajs: list[TrajectoryScoringData],
    arm_name: str,
) -> list[tuple[int, float]]:
    """Get log probabilities for trajectories, conditioned on a specific arm."""
    result = []
    for t in trajs:
        lp = t.conditional_logprobs.get(arm_name, 0.0)
        # Skip trajectories not in this arm (logprob = 0.0 marker)
        if arm_name != "trunk" and lp == 0.0 and t.arm != arm_name:
            continue
        result.append((t.traj_idx, lp))
    return result


def log_arm_statistics(
    data: ScoringData,
    by_branch: dict[str, list[TrajectoryScoringData]],
) -> None:
    """Step 2: Log arm statistics with mass breakdown."""
    arm_names = data.arm_names if data.arm_names else ["trunk"]

    log_step(2, "Arm Statistics", f"{len(arm_names)} arms")

    # Show arm definitions
    if data.arm_texts:
        log("    Conditioning text per arm (after prompt):")
        max_name_len = max(len(name) for name in arm_names)
        for name in arm_names:
            text = data.arm_texts.get(name, "")
            clean_text = escape_newlines(text)
            log(f'      {name:<{max_name_len}}: "{clean_text}"')
        log("")

    # Show prefix logprobs if available
    if data.prefix_logprobs:
        log("    Conditional logprobs:")

        # Build all labels first to find max width
        cond_labels: list[tuple[str, float]] = []
        trunk_lp = data.prefix_logprobs.get("trunk_given_prompt", 0.0)
        cond_labels.append(("p(trunk|prompt)", trunk_lp))

        branch_lps = data.prefix_logprobs.get("branch_given_trunk", {})
        for arm_name in arm_names:
            if arm_name.startswith("branch_"):
                lp = branch_lps.get(arm_name)
                if lp is not None:
                    cond_labels.append((f"p({arm_name}|trunk)", lp))

        twig_lps = data.prefix_logprobs.get("twig_given_branch", {})
        for arm_name in arm_names:
            if "twig_" in arm_name:
                lp = twig_lps.get(arm_name)
                if lp is not None:
                    parent = get_parent_branch(arm_name) or "?"
                    cond_labels.append((f"p({arm_name}|{parent})", lp))

        # Find max label width and print aligned
        max_cond_len = max(len(label) for label, _ in cond_labels) if cond_labels else 0
        for label, lp in cond_labels:
            prob = math.exp(lp) if lp > -700 else 0.0
            log(f"      {label:<{max_cond_len}}: {lp:>8.2f}  (p={fmt_prob(prob):>12})")
        log("")

    for idx, name in enumerate(arm_names):
        is_baseline = name in ("root", "trunk")
        trajs = by_branch.get(name, [])

        # Get log probs conditioned on this arm
        log_probs = get_arm_log_probs(trajs, name)
        traj_probs = compute_normalized_probs(log_probs)

        # Build arm header
        header = f"<{idx}> {name} ({len(trajs)} trajectories)"
        if not is_baseline and trajs:
            t = trajs[0]
            lp_trunk = t.conditional_logprobs.get("trunk", 0.0)
            lp_branch = t.conditional_logprobs.get(name, 0.0)
            if lp_trunk != 0.0 and lp_branch != 0.0:
                p0 = math.exp(lp_trunk - lp_branch)
                # Use scientific notation for very small probabilities (< 0.1%)
                if p0 < 0.001:
                    header += f"  p₀={p0:.2e}"
                else:
                    header += f"  p₀={p0:.1%}"

        log(f"    {header}")
        log(
            f"    {'#':>3}  {'logp':>6}  {'p':>10}  {'p_norm':>7}  {'ppl':>5}  {'inv_ppl_n':>9}"
        )
        log_divider(55, indent_str="    ")

        # Build lookup for trajectory data
        traj_lookup = {t.traj_idx: t for t in trajs}
        log_prob_dict = dict(log_probs)

        # First pass: compute inverse perplexities for normalization
        inv_ppls = {}
        for traj_idx, _ in traj_probs:
            logp = log_prob_dict.get(traj_idx, 0.0)
            traj = traj_lookup.get(traj_idx)
            n_tokens = traj.n_generated_tokens if traj else 0
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
            n_tokens = traj.n_generated_tokens if traj else 0
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
