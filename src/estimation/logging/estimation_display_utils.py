"""Display utilities for estimation output formatting.

Provides formatting functions for core variants, structures, compliance rates,
and arm cores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.common.logging import log, log_divider

from ..estimation_core_types import NUM_DISPLAYED_VARIANTS, CoreVariant
from ..weighting_method_registry import get_method_description, iter_methods

if TYPE_CHECKING:
    from ..estimation_scoring_result import ArmScoring, StructureInfo
    from ..estimation_structure import ArmEstimate


def format_qr_parameter(x: float) -> str:
    """Format q/r parameter, using symbols for infinities."""
    if x == float("inf"):
        return "inf"
    if x == float("-inf"):
        return "-inf"
    if x == 0.0:
        return "0"
    return f"{x:.1f}"


def log_core_variants_table(
    label: str, variants: list[CoreVariant], struct_labels: list[str]
) -> None:
    """Log a table of core variants with structure labels (full cores, no truncation)."""
    log(f"\n    Core variants ({label}):")
    header_labels = "  ".join(f"{l:>6}" for l in struct_labels)
    log(f"    {'name':<14}  {'q':>4}  {'r':>4}    {header_labels}  {'E[∂]':>8}")
    log_divider(32 + 8 * len(struct_labels) + 10, indent_str="    ")
    for v in variants[:NUM_DISPLAYED_VARIANTS]:
        q_str = format_qr_parameter(v.q)
        r_str = format_qr_parameter(v.r)
        core_parts = "  ".join(f"{c:>6.3f}" for c in v.core)  # Full core, no truncation
        log(
            f"    {v.name:<14}  {q_str:>4}  {r_str:>4}    {core_parts}  {v.deviance_avg:>8.4f}"
        )


def log_structures(structure_info: list[StructureInfo]) -> None:
    """Log structure legend."""
    for s in structure_info:
        grouped_marker = " [BUNDLED]" if s.is_bundled else ""
        if s.is_bundled and s.questions:
            log(
                f"  {s.label}: GROUPED ({len(s.questions)} questions){grouped_marker}"
            )
            for q in s.questions:
                log(f"      * {q}")
        else:
            log(f"  {s.label}: {s.description}{grouped_marker}")


def log_compliance_rates(
    arm_scoring: list[ArmScoring], labels: list[str]
) -> None:
    """Log compliance rates by branch."""
    if not arm_scoring:
        log("")
        return

    # Main table header
    header = f"  {'Branch':<12} {'N':>4}  " + "  ".join(f"{l:>6}" for l in labels)
    log(header)
    log_divider(18 + 8 * len(labels))

    for br in arm_scoring:
        rates_str = "  ".join(
            f"{br.structure_rates.get(l, 0.0) * 100:>5.1f}%" for l in labels
        )
        display_name = "trunk" if br.branch_idx == 0 else f"branch_{br.branch_idx}"
        log(f"  {display_name:<12} {br.trajectory_count:>4}  {rates_str}")

    # Show breakdowns for grouped structures
    for label in labels:
        has_breakdown = any(label in br.question_rates for br in arm_scoring)
        if not has_breakdown:
            continue

        log(f"\n  {label} breakdown:")
        questions = []
        for br in arm_scoring:
            q_rates = br.question_rates.get(label, {})
            if q_rates:
                questions = list(q_rates.keys())
                break

        if questions:

            def get_branch_header(br: ArmScoring) -> str:
                return "trunk" if br.branch_idx == 0 else f"br_{br.branch_idx}"

            branch_headers = "  ".join(
                f"{get_branch_header(br):>8}" for br in arm_scoring
            )
            log(f"    {'Question':<50}  {branch_headers}")
            log_divider(52 + 10 * len(arm_scoring), indent_str="    ")

            for q in questions:
                rates_row = "  ".join(
                    f"{br.question_rates.get(label, {}).get(q, 0.0) * 100:>7.1f}%"
                    for br in arm_scoring
                )
                q_display = q[:48] + ".." if len(q) > 50 else q
                log(f"    {q_display:<50}  {rates_row}")
    log("")


def log_arm_cores(
    arms: list[ArmEstimate], labels: list[str], show_variants: bool = True
) -> None:
    """Log core values by arm, iterating over all weighting methods."""
    for arm in arms:
        display_name = "trunk" if arm.arm_idx == 0 else f"branch_{arm.arm_idx}"
        log(
            f"\n  [{arm.arm_idx}] {display_name} ({len(arm.trajectories)} trajectories)"
        )

        # Check if any estimates exist
        if not arm.estimates:
            continue

        header_labels = "  ".join(f"{l:>6}" for l in labels)
        log(
            f"\n    {'weighting':<25}  {header_labels}  {'E[∂]':>8}  {'Var[∂]':>10}"
        )
        log_divider(31 + 8 * len(labels) + 20, indent_str="    ")

        # Iterate over registered weighting methods
        for method_name, _, _ in iter_methods():
            est = arm.estimates.get(method_name)
            if est and est.core:
                desc = get_method_description(method_name)
                core_vals = "  ".join(
                    f"{est.core[i]:>6.3f}" for i in range(len(est.core))
                )
                log(
                    f"    {desc:<25}  {core_vals}  {est.deviance_avg:>8.4f}  {est.deviance_var:>10.6f}"
                )

        # Show core variants for each method
        if show_variants:
            for method_name, _, _ in iter_methods():
                est = arm.estimates.get(method_name)
                if est and est.core_variants:
                    desc = get_method_description(method_name)
                    log_core_variants_table(desc, est.core_variants, labels)
