"""Logging utilities for forking paths generation method.

This module provides detailed logging for the forking paths algorithm,
including entropy analysis, position histograms, and fork point visualization.
"""

from __future__ import annotations

import math

from src.common.log import log
from src.common.log_utils import log_divider, log_step
from src.inference import ModelRunner
from src.inference.generated_trajectory import GeneratedTrajectory

from ..forking_paths_params import ForkingParams
from ..forking_paths_types import ForkPoint, PositionAnalysis, QualifyingFork


def log_greedy_path(
    greedy_traj: GeneratedTrajectory,
    runner: ModelRunner,
    arm_name: str,
    prompt_len: int,
) -> None:
    """Log the greedy path for an arm.

    Args:
        greedy_traj: The greedy trajectory
        runner: Model runner for decoding
        arm_name: Name of the arm (trunk, branch_1, etc.)
        prompt_len: Length of prompt in tokens
    """
    log_step(1, "Generate greedy path")
    greedy_text = runner.decode_ids(greedy_traj.token_ids[prompt_len:])
    log(f"    Arm: {arm_name}")
    log(
        f'    Greedy: "{greedy_text[:80]}..."'
        if len(greedy_text) > 80
        else f'    Greedy: "{greedy_text}"'
    )
    log(f"    Tokens: {greedy_traj.length - prompt_len}")


def log_position_analyses(
    analyses: list[PositionAnalysis],
    qualifying: list[QualifyingFork],
    runner: ModelRunner,
    params: ForkingParams,
    greedy_traj: "GeneratedTrajectory | None" = None,
    prompt_len: int = 0,
) -> None:
    """Log entropy analysis at each position.

    Args:
        analyses: Position analyses from greedy path
        qualifying: Qualifying fork points
        runner: Model runner for decoding
        params: Forking parameters
        greedy_traj: The greedy trajectory (for prompt/response display)
        prompt_len: Length of prompt in tokens
    """
    log_step(2, "Analyze positions (top-5 candidates + entropy)")

    # Show prompt and response tokens if available
    if greedy_traj is not None and prompt_len > 0:
        log(f"  Prompt ({prompt_len} tokens):")
        prompt_text = runner.decode_ids(greedy_traj.token_ids[:prompt_len])
        # Show first 80 chars and last 40 chars
        if len(prompt_text) > 120:
            log(f"    {prompt_text[:80]}")
            log(f"    ...{prompt_text[-40:]}")
        else:
            log(f"    {prompt_text}")

        response_len = greedy_traj.length - prompt_len
        log(f"\n  Response ({response_len} tokens):")
        response_text = runner.decode_ids(greedy_traj.token_ids[prompt_len:])
        if len(response_text) > 80:
            log(f"    {response_text[:80]}...")
        else:
            log(f"    {response_text}")
        log("")

    # Compute entropy statistics
    entropies = [a.entropy for a in analyses]
    if not entropies:
        log("    No positions to analyze")
        return

    min_e = min(entropies)
    max_e = max(entropies)
    avg_e = sum(entropies) / len(entropies)

    # Compute std deviation
    variance = sum((e - avg_e) ** 2 for e in entropies) / len(entropies)
    std_e = math.sqrt(variance)

    # Compute percentiles
    sorted_e = sorted(entropies)
    n = len(sorted_e)
    p25 = sorted_e[int(n * 0.25)] if n >= 4 else min_e
    p50 = sorted_e[int(n * 0.50)] if n >= 2 else avg_e
    p75 = sorted_e[int(n * 0.75)] if n >= 4 else max_e

    high_entropy_count = sum(1 for e in entropies if e >= params.min_entropy)

    log(f"  Entropy statistics ({len(analyses)} response positions):")
    log(f"    min={min_e:.2f}  max={max_e:.2f}")
    log(f"    mean={avg_e:.2f}  std={std_e:.2f}")
    log(f"    p25={p25:.2f}  p50={p50:.2f}  p75={p75:.2f}")

    # Show entropy histogram
    log("")
    log("  Entropy distribution (response tokens):")
    log("  Range        Count  Distribution")
    log_divider(50, indent="  ")
    _log_entropy_histogram(entropies, params.min_entropy)

    # Show ASCII viz of entropy across positions
    _log_entropy_ascii(analyses, params.min_entropy)

    # Filtering section
    log("")
    log("  Filtering:")
    log(f"    Positions with H >= {params.min_entropy}: {high_entropy_count}/{len(analyses)}")
    alternates_count = len(qualifying)
    log(f"    Alternates with p >= {params.min_prob}: {alternates_count}")

    # Show qualifying forks
    log("")
    log(f"  Qualifying forks ({len(qualifying)} total):")
    log_divider(60, indent="  ")

    if not qualifying:
        log("    None found")
    else:
        log(
            f"   {'pos':>4}  {'entropy':>7}  {'greedy':>12}  {'alt':>12}  {'p(alt)':>8}"
        )
        log_divider(55, indent="   ")

        for qf in qualifying[:10]:  # Show first 10
            greedy_tok = runner.decode_ids([qf.analysis.greedy_token_id])
            alt_tok = runner.decode_ids([qf.candidate.token_id])
            log(
                f"   {qf.analysis.position:>4}  "
                f"{qf.analysis.entropy:>7.2f}  "
                f'"{greedy_tok[:10]:>10}"  '
                f'"{alt_tok[:10]:>10}"  '
                f"{qf.candidate.prob:>8.3f}"
            )
        if len(qualifying) > 10:
            log(f"   ... and {len(qualifying) - 10} more")


def _log_entropy_histogram(entropies: list[float], threshold: float) -> None:
    """Log histogram of entropy values with dynamic bins."""
    if not entropies:
        return

    min_e = min(entropies)
    max_e = max(entropies)
    n_bins = 8

    # Dynamic bin edges based on data range
    bin_width = (max_e - min_e) / n_bins if max_e > min_e else 1.0
    bins = [min_e + i * bin_width for i in range(n_bins + 1)]

    # Count values in each bin
    counts = [0] * n_bins
    for e in entropies:
        for i in range(n_bins):
            if bins[i] <= e < bins[i + 1] or (i == n_bins - 1 and e == bins[i + 1]):
                counts[i] += 1
                break

    max_count = max(counts) if counts else 1
    bar_width = 30

    for i in range(n_bins):
        bar_len = int(counts[i] / max_count * bar_width) if max_count > 0 else 0
        bar = "\u2588" * bar_len  # Full block character
        log(f"   {bins[i]:>5.2f}-{bins[i + 1]:>5.2f}  {counts[i]:<3}   {bar}")


def _log_entropy_ascii(analyses: list[PositionAnalysis], threshold: float) -> None:
    """Log ASCII visualization of entropy across positions using unicode blocks."""
    if not analyses:
        return

    entropies = [a.entropy for a in analyses]
    min_e = min(entropies)
    max_e = max(entropies)
    avg_e = sum(entropies) / len(entropies)

    if max_e == 0:
        max_e = 1

    # Unicode block characters for different heights (8 levels)
    # From lowest to highest: space, lower blocks, full block
    blocks = [
        " ",
        "\u2581",
        "\u2582",
        "\u2583",
        "\u2584",
        "\u2585",
        "\u2586",
        "\u2587",
        "\u2588",
    ]

    # Limit to first 60 positions
    analyses = analyses[:60]
    n = len(analyses)

    # Find max and min positions
    max_pos = entropies.index(max_e)
    min_pos = entropies.index(min_e)

    # Show 8 height levels
    height = 8

    log("\n  Entropy over response positions:")
    for row in range(height - 1, -1, -1):
        level = min_e + (row + 0.5) / height * (max_e - min_e)

        # Build label
        if row == height - 1:
            label = f"  {max_e:.2f} \u2502"
            suffix = f" \u2190 max @{max_pos}"
        elif row == 0:
            label = f"  {min_e:.2f} \u2502"
            suffix = f" \u2190 min @{min_pos}"
        elif abs(level - threshold) < (max_e - min_e) / height:
            label = f"  {threshold:.2f} \u2502"
            suffix = ""
        elif abs(level - avg_e) < (max_e - min_e) / height:
            label = "       \u2502"
            suffix = f" \u2190 \u03bc={avg_e:.2f}"
        else:
            label = "       \u2502"
            suffix = ""

        # Build bar characters
        chars = []
        for a in analyses:
            # Normalize to 0-8 range
            normalized = (a.entropy - min_e) / (max_e - min_e) if max_e > min_e else 0.5
            bar_height = normalized * height

            if bar_height > row + 1:
                chars.append("\u2588")  # Full block
            elif bar_height > row:
                # Partial block based on fraction
                frac = bar_height - row
                block_idx = min(int(frac * 8), 8)
                chars.append(blocks[block_idx])
            else:
                chars.append(" ")

        log(label + "".join(chars) + suffix)

    # Bottom border and position markers
    log("       \u2514" + "\u2534" * n)
    log("        " + "".join(str(i % 10) for i in range(n)))
    log(f"          n={n}")


def log_fork_expansion(
    fork_points: list[ForkPoint],
    analyses: list[PositionAnalysis],
    runner: ModelRunner,
    prompt_len: int,
) -> None:
    """Log the expansion of fork points into trajectories.

    Args:
        fork_points: Expanded fork points with continuations
        analyses: Position analyses (to get top tokens)
        runner: Model runner for decoding
        prompt_len: Length of prompt in tokens
    """
    log_step(3, "Expand fork points", f"{len(fork_points)} points")

    if not fork_points:
        log("    No fork points expanded")
        return

    for i, fp in enumerate(fork_points[:5]):  # Show first 5
        greedy_tok = runner.decode_ids([fp.greedy_token_id])
        alt_tok = runner.decode_ids([fp.alternate.token_id])

        log(
            f"\n    Fork {i + 1}/{len(fork_points)} @ pos {fp.position} (H={fp.entropy:.2f}):"
        )
        log(f'      "{greedy_tok}" \u2192 "{alt_tok}" (p={fp.alternate.prob:.3f})')

        # Show top 5 tokens at this position
        analysis = next((a for a in analyses if a.position == fp.position), None)
        if analysis and analysis.candidates:
            log("      Top 5 tokens:")
            for j, cand in enumerate(analysis.candidates[:5]):
                tok = runner.decode_ids([cand.token_id])
                markers = []
                if cand.token_id == fp.greedy_token_id:
                    markers.append("\u2190greedy")
                if cand.token_id == fp.alternate.token_id:
                    markers.append("\u2190fork")
                marker_str = " " + " ".join(markers) if markers else ""
                log(f'        {j + 1}. "{tok}" p={cand.prob:.4f}{marker_str}')

        # Show samples
        for j, traj in enumerate(fp.continuations):
            cont_text = runner.decode_ids(traj.token_ids[prompt_len:])
            if len(cont_text) > 60:
                cont_text = cont_text[:60] + "..."
            log(f'      Sample {j + 1}: "{cont_text}"')

    if len(fork_points) > 5:
        log(f"\n    ... and {len(fork_points) - 5} more fork points")


def log_arm_tree(
    arm_name: str,
    greedy_traj: GeneratedTrajectory,
    fork_points: list[ForkPoint],
    runner: ModelRunner,
    prompt_len: int,
    max_tokens: int = 128,
) -> None:
    """Log a tree visualization for the arm.

    Args:
        arm_name: Name of the arm
        greedy_traj: The greedy trajectory
        fork_points: All fork points with continuations
        runner: Model runner for decoding
        prompt_len: Length of prompt
        max_tokens: Max tokens for scale
    """
    log("")
    log("  Tree:")

    # Build scale header
    scale_marks = list(range(0, max_tokens + 1, 21))
    scale_line = "      " + "".join(f"{m:<8}" for m in scale_marks)
    log(scale_line)

    # Count total trajectories
    total_trajs = 1 + sum(len(fp.continuations) for fp in fork_points)
    traj_idx = 0

    # Show greedy as root
    greedy_len = greedy_traj.length - prompt_len
    log(f"      \u2514\u2500\u25cf [{traj_idx}]")
    traj_idx += 1

    # Show fork points and their continuations
    for fp_idx, fp in enumerate(fork_points):
        is_last_fork = fp_idx == len(fork_points) - 1

        for cont_idx, cont in enumerate(fp.continuations):
            is_last_cont = cont_idx == len(fp.continuations) - 1 and is_last_fork

            if is_last_cont:
                prefix = "       \u2514\u2500\u25cf"
            else:
                prefix = "       \u251c\u2500\u25cf"

            log(f"{prefix} [{traj_idx}]")
            traj_idx += 1

    # Summary line
    log("")
    log(f"  Summary: {total_trajs} trajectories from {len(fork_points)} fork points")
