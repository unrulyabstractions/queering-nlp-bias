"""Display utilities for estimation output formatting.

Provides formatting functions for core variants, structures, compliance rates,
and arm cores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.common.logging import log, log_divider

from ..arm_types import get_display_name, get_short_display_name
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
    log(f"    {'name':<14}  {'q':>5}  {'r':>5}    {header_labels}  {'E[∂]':>8}")
    log_divider(34 + 8 * len(struct_labels) + 10, indent_str="    ")
    for v in variants[:NUM_DISPLAYED_VARIANTS]:
        q_str = format_qr_parameter(v.q)
        r_str = format_qr_parameter(v.r)
        core_parts = "  ".join(f"{c:>6.3f}" for c in v.core)  # Full core, no truncation
        log(
            f"    {v.name:<14}  {q_str:>5}  {r_str:>5}    {core_parts}  {v.deviance_avg:>8.4f}"
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
        # Get rate from simple_scoring or bundled_scoring aggregate
        def get_rate(label: str) -> float:
            if label in br.simple_scoring:
                return br.simple_scoring[label]
            if label in br.bundled_scoring:
                return br.bundled_scoring[label].aggregate
            raise KeyError(f"Label '{label}' not found in arm '{br.branch}' scoring")

        rates_str = "  ".join(f"{get_rate(l) * 100:>5.1f}%" for l in labels)
        # Use arm name directly if available, otherwise construct from branch_idx
        display_name = getattr(br, "name", None)
        if display_name is None:
            display_name = "trunk" if br.branch_idx == 0 else f"branch_{br.branch_idx}"
        log(f"  {display_name:<12} {br.trajectory_count:>4}  {rates_str}")

    # Helper to get branch header name
    def get_branch_header(br: ArmScoring) -> str:
        name = getattr(br, "name", None)
        if name is None:
            name = "trunk" if br.branch_idx == 0 else f"branch_{br.branch_idx}"
        return get_short_display_name(name)

    # Show categorical (non-bundled) questions breakdown
    categorical_labels = [l for l in labels if any(l in br.simple_scoring for br in arm_scoring)]
    if categorical_labels:
        log("\n  Categorical questions breakdown:")
        branch_headers = "  ".join(
            f"{get_branch_header(br):>8}" for br in arm_scoring
        )
        log(f"    {'Question':<50}  {branch_headers}")
        log_divider(52 + 10 * len(arm_scoring), indent_str="    ")

        # Get full question text from structure_info if available
        for label in categorical_labels:
            # Find the question text for this label
            question_text = label  # Default to label
            for br in arm_scoring:
                if hasattr(br, "structure_info"):
                    for s in br.structure_info:
                        if s.label == label and s.description:
                            question_text = s.description
                            break

            rates_row = "  ".join(
                f"{br.simple_scoring.get(label, 0.0) * 100:>7.1f}%"
                for br in arm_scoring
            )
            q_display = question_text[:48] + ".." if len(question_text) > 50 else question_text
            log(f"    {q_display:<50}  {rates_row}")

    # Show breakdowns for bundled structures
    for label in labels:
        has_breakdown = any(label in br.bundled_scoring for br in arm_scoring)
        if not has_breakdown:
            continue

        log(f"\n  {label} breakdown:")
        questions = []
        for br in arm_scoring:
            if label in br.bundled_scoring:
                questions = list(br.bundled_scoring[label].items.keys())
                break

        if questions:
            branch_headers = "  ".join(
                f"{get_branch_header(br):>8}" for br in arm_scoring
            )
            log(f"    {'Question':<50}  {branch_headers}")
            log_divider(52 + 10 * len(arm_scoring), indent_str="    ")

            for q in questions:
                def get_item_rate(br: ArmScoring, lbl: str, question: str) -> float:
                    if lbl not in br.bundled_scoring:
                        raise KeyError(f"Label '{lbl}' not in bundled_scoring for '{br.branch}'")
                    items = br.bundled_scoring[lbl].items
                    if question not in items:
                        raise KeyError(f"Question '{question}' not in items for '{lbl}'")
                    return items[question]

                rates_row = "  ".join(
                    f"{get_item_rate(br, label, q) * 100:>7.1f}%"
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
        # Use arm.name directly (root, trunk, branch_N)
        display_name = get_display_name(arm.name)
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
