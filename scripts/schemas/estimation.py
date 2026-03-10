"""Schemas for normativity estimation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.common.base_schema import BaseSchema
from src.common.log import log

from .log_utils import log_banner, log_divider
from src.common.math.entropy_diversity.structure_aware import (
    deviance,
    deviance_variance,
    expected_deviance,
    expected_orientation,
    generalized_system_core,
    orientation,
)
from src.common.viz_utils import preview

# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════


def _format_qr(x: float) -> str:
    """Format q/r parameter, using symbols for infinities."""
    if x == float("inf"):
        return "∞"
    if x == float("-inf"):
        return "-∞"
    if x == 0.0:
        return "0"
    return f"{x:.1f}"


def _format_core(core: list[float]) -> str:
    """Format core vector for display (full, no truncation)."""
    if not core:
        return "[]"
    items = ", ".join(f"{c:.3f}" for c in core)
    return f"[{items}]"


# ══════════════════════════════════════════════════════════════════════════════
# GENERALIZED CORE PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# Named (q, r) parameterizations for generalized cores
# Reference: https://www.unrulyabstractions.com/pdfs/diversity.pdf
#
# r controls which trajectories get attention:
#   r=1: actual distribution, r=0: uniform, r=∞: mode, r=-∞: anti-mode (rarest)
# q controls how compliance values are aggregated:
#   q=1: arithmetic, q=0: geometric, q=-1: harmonic, q=∞: max, q=-∞: min

# Number of core variants to display (5 paper cases + 4 user-requested combos)
NUM_DISPLAYED_VARIANTS = 9

NAMED_CORES: list[tuple[str, float, float, str]] = [
    # First NUM_DISPLAYED_VARIANTS are shown: paper's 5 + user-requested (q,r) combinations
    ("standard", 1.0, 1.0, "⟨α⟩ standard expected compliance"),
    ("uniform", 1.0, 0.0, "uniform avg over support"),
    ("mode", 1.0, float("inf"), "compliance of mode"),
    ("max", float("inf"), 1.0, "max compliance in support"),
    ("mode_min", float("-inf"), float("inf"), "min compliance among modes"),
    ("confident", 1.0, 2.0, "confident core (q=1, r=2)"),
    ("rms", 2.0, 1.0, "root-mean-square (q=2, r=1)"),
    ("rms_conf", 2.0, 2.0, "RMS confident (q=2, r=2)"),
    ("top_heavy", 1.0, 100.0, "heavily mode-biased (q=1, r=100)"),
    # Additional cases - varying r (which trajectories)
    ("antimode", 1.0, float("-inf"), "compliance of rarest (anti-mode)"),
    ("inverse", 1.0, -1.0, "inverse probability weighting"),
    # Additional cases - varying q (how to aggregate)
    ("geometric", 0.0, 1.0, "geometric mean (sensitive to exclusion)"),
    ("harmonic", -1.0, 1.0, "harmonic mean (penalizes low compliance)"),
    # Combinations for contrasting dominant vs. rare
    ("rare_max", float("inf"), float("-inf"), "max compliance among rarest"),
    ("actual_min", float("-inf"), 1.0, "min compliance under actual dist"),
    ("rare_min", float("-inf"), float("-inf"), "min compliance among rarest"),
    ("rare_geometric", 0.0, float("-inf"), "geometric mean in long tail"),
]

# ══════════════════════════════════════════════════════════════════════════════
# DATA TRANSFER SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class TrajectoryCompliance(BaseSchema):
    """A trajectory with its compliance scores for estimation."""

    traj_idx: int
    branch: str
    compliances: list[float]
    conditional_logprobs: dict[str, float]  # Log prob conditioned on each group
    n_continuation_tokens: int = 0  # Number of tokens in continuation


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class TrajectorySummary(BaseSchema):
    """Summary mapping trajectory to its group(s) and text."""

    traj_idx: int
    arm_idxs: list[int]  # Trajectory can belong to multiple arms (trunk + its branch)
    continuation_text: str


@dataclass
class ArmSummary(BaseSchema):
    """Summary of an arm (trunk or branch_N) for estimation."""

    arm_idx: int
    name: str
    trajectory_count: int


@dataclass
class EstimationSummary(BaseSchema):
    """Summary of trajectories and arms for easy lookup."""

    trajectories: list[TrajectorySummary]
    arms: list[ArmSummary]


# ══════════════════════════════════════════════════════════════════════════════
# ESTIMATION SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class CoreVariant(BaseSchema):
    """A generalized core with specific (q, r) parameterization."""

    name: str  # e.g., "standard", "antimode"
    q: float  # power mean order
    r: float  # escort order
    description: str  # human-readable description
    core: list[float]  # computed core values ⟨Λ_n⟩_{q,r}
    deviance_avg: float  # E[∂_n] relative to this core
    deviance_var: float  # Var[∂_n] relative to this core


@dataclass
class TrajectoryEstimate(BaseSchema):
    """Estimation results for a single trajectory."""

    traj_idx: int
    orientation: list[float]  # θ_n(x) = Λ_n(x) - ⟨Λ_n⟩
    deviance: float  # ∂_n(x) = ||θ_n(x)||


@dataclass
class ArmEstimate(BaseSchema):
    """Estimation results for an arm (trunk or branch_N).

    An "arm" is a conditioning point in the generation tree - either
    the trunk (all trajectories) or a specific branch prefix.

    Two weighting schemes are computed:
    1. Probability-weighted: uses p(traj) as weights
       - The (q, r) generalization controls escort order and power mean order
    2. Inv-perplexity-weighted: uses 1/ppl = exp(logp/n_tokens) as weights
       - Beyond the (q, r) generalization: different base measure entirely
       - Weights by model confidence per token, not raw probability
    """

    arm_idx: int
    name: str
    # Primary cores (q=1, r=1 with different weighting schemes)
    core: list[float]  # ⟨Λ_n⟩ probability-weighted
    core_inv_ppl: list[float]  # ⟨Λ_n⟩ inv-perplexity-weighted
    trajectories: list[TrajectoryEstimate]
    # Deviance stats relative to THIS arm's core: E[∂|branch]
    deviance_avg: float  # E[∂|B] prob-weighted
    deviance_var: float  # Var[∂|B] prob-weighted
    deviance_avg_inv_ppl: float  # E[∂|B] inv-ppl-weighted
    deviance_var_inv_ppl: float  # Var[∂|B] inv-ppl-weighted
    # Deviance stats relative to TRUNK core: E[∂|trunk]
    deviance_avg_trunk: float = 0.0  # E[∂|T] prob-weighted
    deviance_avg_trunk_inv_ppl: float = 0.0  # E[∂|T] inv-ppl-weighted
    # Expected deviance delta: E[Δ∂] = E[∂|branch - ∂|trunk]
    deviance_delta: float = 0.0  # E[Δ∂] prob-weighted
    deviance_delta_inv_ppl: float = 0.0  # E[Δ∂] inv-ppl-weighted
    # Expected orientation relative to TRUNK core: E[θ|trunk]
    orientation_avg: list[float] = field(default_factory=list)  # E[θ|T] prob-weighted
    orientation_avg_inv_ppl: list[float] = field(default_factory=list)  # E[θ|T] inv-ppl-weighted
    # Distance between this arm's core and trunk core: ‖E[θ|trunk]‖
    orientation_norm: float = 0.0  # ‖E[θ|T]‖ prob-weighted
    orientation_norm_inv_ppl: float = 0.0  # ‖E[θ|T]‖ inv-ppl-weighted
    # All (q,r) variants
    core_variants: list[CoreVariant] = field(default_factory=list)
    core_variants_inv_ppl: list[CoreVariant] = field(default_factory=list)

    @staticmethod
    def _normalize_logprobs(log_probs: list[float]) -> list[float]:
        """Convert log probabilities to normalized probabilities."""
        if not log_probs:
            return []
        max_lp = max(log_probs)
        probs = [math.exp(lp - max_lp) for lp in log_probs]
        total = sum(probs)
        if total > 0:
            return [p / total for p in probs]
        return [1.0 / len(probs)] * len(probs)

    @staticmethod
    def _compute_inv_perplexity_weights(
        log_probs: list[float], n_tokens: list[int]
    ) -> list[float]:
        """Compute normalized inverse perplexity weights.

        inv_ppl = exp(logprob / n_tokens) = 1/perplexity
        Normalized so they sum to 1.
        """
        inv_ppls = []
        for lp, n in zip(log_probs, n_tokens):
            if n > 0 and lp > -700:
                inv_ppls.append(math.exp(lp / n))
            else:
                inv_ppls.append(0.0)
        total = sum(inv_ppls)
        if total > 0:
            return [p / total for p in inv_ppls]
        return [1.0 / len(inv_ppls)] * len(inv_ppls)

    @staticmethod
    def _compute_core_variants(
        compliances: list[list[float]], probs: list[float]
    ) -> list[CoreVariant]:
        """Compute all named core variants with their deviances."""
        variants = []
        for name, q, r, desc in NAMED_CORES:
            try:
                core = generalized_system_core(compliances, probs, q=q, r=r)
                dev_avg = expected_deviance(compliances, core, weights=probs, norm="l2")
                dev_var = deviance_variance(compliances, core, weights=probs, norm="l2")
                variants.append(
                    CoreVariant(
                        name=name,
                        q=q,
                        r=r,
                        description=desc,
                        core=core,
                        deviance_avg=dev_avg,
                        deviance_var=dev_var,
                    )
                )
            except (ValueError, ZeroDivisionError, OverflowError):
                # Some (q, r) combinations may fail numerically
                pass
        return variants

    @classmethod
    def from_trajectories(
        cls,
        arm_idx: int,
        name: str,
        trajectories: list[TrajectoryCompliance],
        reference_core: list[float] | None = None,
        reference_core_inv_ppl: list[float] | None = None,
    ) -> "ArmEstimate":
        """Create arm estimate from trajectory compliances.

        Uses probability-weighted calculations:
        - Core: generalized_system_core with trajectory probabilities
        - Deviance stats: expected_deviance/variance with probability weights

        Log probabilities are converted to normalized probabilities to avoid
        underflow issues with long sequences.

        Computes multiple core variants:
        - Standard probability-weighted cores with various (q, r) settings
        - Inverse-perplexity weighted cores (weighting by model confidence)

        Args:
            arm_idx: Index of this arm (0=trunk, 1+=branches)
            name: Name of this arm (e.g., "trunk", "branch_1")
            trajectories: Trajectories with their compliance scores and log probabilities
            reference_core: Optional core to compute E[θ] against (default: this arm's core)
            reference_core_inv_ppl: Optional inv-ppl core for E[θ] (default: this arm's core)
        """
        n_trajs = len(trajectories)

        # Handle empty arm: return neutral estimate
        if n_trajs == 0:
            return cls(
                arm_idx=arm_idx,
                name=name,
                core=[],
                core_inv_ppl=[],
                trajectories=[],
                deviance_avg=0.0,
                deviance_var=0.0,
                deviance_avg_inv_ppl=0.0,
                deviance_var_inv_ppl=0.0,
                orientation_avg=[],
                orientation_avg_inv_ppl=[],
                core_variants=[],
                core_variants_inv_ppl=[],
            )

        # Extract compliances and log probabilities (conditioned on this arm)
        compliances = [t.compliances for t in trajectories]
        log_probs = [t.conditional_logprobs.get(name, 0.0) for t in trajectories]
        n_tokens = [t.n_continuation_tokens for t in trajectories]
        n_structures = len(compliances[0])

        # Convert log probabilities to normalized probabilities
        probs = cls._normalize_logprobs(log_probs)

        # Compute inverse perplexity weights
        inv_ppl_weights = cls._compute_inv_perplexity_weights(log_probs, n_tokens)

        # Validate consistent dimensions
        for i, c in enumerate(compliances[1:], start=1):
            if len(c) != n_structures:
                raise ValueError(
                    f"Compliance {i} has {len(c)} dimensions, expected {n_structures}"
                )

        # Calculate probability-weighted core ⟨Λ_n⟩
        # q=1, r=1 gives standard expected compliance weighted by probability
        core = generalized_system_core(compliances, probs, q=1.0, r=1.0)

        # Calculate orientation and deviance for each trajectory
        traj_estimates = [
            TrajectoryEstimate(
                traj_idx=t.traj_idx,
                orientation=list(orientation(t.compliances, core)),
                deviance=float(deviance(t.compliances, core, norm="l2")),
            )
            for t in trajectories
        ]

        # Calculate probability-weighted aggregate deviance statistics
        dev_avg = expected_deviance(compliances, core, weights=probs, norm="l2")
        dev_var = deviance_variance(compliances, core, weights=probs, norm="l2")

        # Calculate inv-perplexity-weighted core (q=1, r=1 but different base measure)
        core_inv_ppl = generalized_system_core(
            compliances, inv_ppl_weights, q=1.0, r=1.0
        )
        dev_avg_inv_ppl = expected_deviance(
            compliances, core_inv_ppl, weights=inv_ppl_weights, norm="l2"
        )
        dev_var_inv_ppl = deviance_variance(
            compliances, core_inv_ppl, weights=inv_ppl_weights, norm="l2"
        )

        # Calculate metrics relative to reference (trunk) core
        # If reference_core is provided, compute E[θ|T] and E[∂|T]
        # Otherwise, these equal the branch-relative values
        ref_core = reference_core if reference_core is not None else core
        ref_core_inv = reference_core_inv_ppl if reference_core_inv_ppl is not None else core_inv_ppl

        # E[θ|T] - orientation relative to trunk
        orient_avg = expected_orientation(compliances, ref_core, weights=probs)
        orient_avg_inv_ppl = expected_orientation(compliances, ref_core_inv, weights=inv_ppl_weights)

        # ‖E[θ|T]‖ - L2 norm of orientation (distance between cores)
        orient_norm = math.sqrt(sum(x * x for x in orient_avg)) if orient_avg else 0.0
        orient_norm_inv_ppl = math.sqrt(sum(x * x for x in orient_avg_inv_ppl)) if orient_avg_inv_ppl else 0.0

        # E[∂|T] - deviance relative to trunk
        dev_avg_trunk = expected_deviance(compliances, ref_core, weights=probs, norm="l2")
        dev_avg_trunk_inv_ppl = expected_deviance(compliances, ref_core_inv, weights=inv_ppl_weights, norm="l2")

        # E[Δ∂] = E[∂|branch] - E[∂|trunk] (by linearity of expectation)
        dev_delta = dev_avg - dev_avg_trunk
        dev_delta_inv_ppl = dev_avg_inv_ppl - dev_avg_trunk_inv_ppl

        # Compute all named core variants with probability weights
        core_variants = cls._compute_core_variants(compliances, probs)

        # Compute all named core variants with inverse perplexity weights
        core_variants_inv_ppl = cls._compute_core_variants(compliances, inv_ppl_weights)

        return cls(
            arm_idx=arm_idx,
            name=name,
            core=core,
            core_inv_ppl=core_inv_ppl,
            trajectories=traj_estimates,
            deviance_avg=dev_avg,
            deviance_var=dev_var,
            deviance_avg_inv_ppl=dev_avg_inv_ppl,
            deviance_var_inv_ppl=dev_var_inv_ppl,
            deviance_avg_trunk=dev_avg_trunk,
            deviance_avg_trunk_inv_ppl=dev_avg_trunk_inv_ppl,
            deviance_delta=dev_delta,
            deviance_delta_inv_ppl=dev_delta_inv_ppl,
            orientation_avg=orient_avg,
            orientation_avg_inv_ppl=orient_avg_inv_ppl,
            orientation_norm=orient_norm,
            orientation_norm_inv_ppl=orient_norm_inv_ppl,
            core_variants=core_variants,
            core_variants_inv_ppl=core_variants_inv_ppl,
        )


@dataclass
class EstimationSummaryOutput(BaseSchema):
    """Human-readable summary output saved separately."""

    # Metadata
    generation_file: str
    scoring_file: str
    judgment_file: str
    judge_model: str
    embedding_model: str
    estimated_at: str

    # Structure definitions with labels
    structures: list[dict[str, Any]]  # [{label, description, is_bundled, questions}]

    # Per-branch rates
    branch_rates: list[dict[str, Any]]  # [{branch, rates: {label: rate}, question_rates}]

    # Per-branch cores (labeled)
    branch_cores: list[dict[str, Any]]  # [{branch, prob_weighted: {label: val}, inv_ppl: {label: val}}]

    # Continuations by branch
    continuations_by_branch: dict[str, list[dict[str, Any]]]  # {branch: [{idx, text}]}


@dataclass
class EstimationOutput(BaseSchema):
    """Output from normativity estimation."""

    summary: EstimationSummary
    categorical_judgements: list[Any]  # str | list[str]
    similarity_scoring: list[Any]  # str | list[str]
    arms: list[ArmEstimate]  # trunk + branches
    judgment_file: str
    estimated_at: str

    # Additional metadata for summary
    generation_file: str = ""
    scoring_file: str = ""
    judge_model: str = ""
    embedding_model: str = ""
    structure_info: list[dict[str, Any]] = field(default_factory=list)
    branch_rates: list[dict[str, Any]] = field(default_factory=list)
    continuations_by_branch: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        judgment_file: str,
        categorical_judgements: list[Any],
        similarity_scoring: list[Any],  # str | list[str]
        arms: list[ArmEstimate],
        texts: dict[int, str],
        generation_file: str = "",
        scoring_file: str = "",
        judge_model: str = "",
        embedding_model: str = "",
        structure_info: list[StructureInfo] | None = None,
        branch_rates: list[BranchRates] | None = None,
        continuations_by_branch: dict[str, list[tuple[int, str]]] | None = None,
    ) -> EstimationOutput:
        """Create estimation output with auto-generated summary."""
        # Build trajectory -> arms mapping
        traj_to_arms: dict[int, list[int]] = {}
        for arm in arms:
            for traj in arm.trajectories:
                traj_to_arms.setdefault(traj.traj_idx, []).append(arm.arm_idx)

        summary = EstimationSummary(
            trajectories=[
                TrajectorySummary(
                    traj_idx=idx, arm_idxs=aids, continuation_text=texts.get(idx, "")
                )
                for idx, aids in sorted(traj_to_arms.items())
            ],
            arms=[
                ArmSummary(a.arm_idx, a.name, len(a.trajectories)) for a in arms
            ],
        )

        # Convert structure_info to dicts
        struct_dicts = []
        if structure_info:
            for s in structure_info:
                struct_dicts.append({
                    "label": s.label,
                    "description": s.description,
                    "is_bundled": s.is_bundled,
                    "question_count": s.question_count,
                    "questions": s.questions,
                })

        # Convert branch_rates to dicts
        rate_dicts = []
        if branch_rates:
            for br in branch_rates:
                rate_dicts.append({
                    "branch": br.branch,
                    "branch_idx": br.branch_idx,
                    "trajectory_count": br.trajectory_count,
                    "structure_rates": br.structure_rates,
                    "question_rates": br.question_rates,
                })

        # Convert continuations to dicts
        cont_dicts: dict[str, list[dict[str, Any]]] = {}
        if continuations_by_branch:
            for branch, items in continuations_by_branch.items():
                cont_dicts[branch] = [{"idx": idx, "text": text} for idx, text in items]

        return cls(
            summary=summary,
            categorical_judgements=categorical_judgements,
            similarity_scoring=similarity_scoring,
            arms=arms,
            judgment_file=judgment_file,
            estimated_at=datetime.now().isoformat(),
            generation_file=generation_file,
            scoring_file=scoring_file,
            judge_model=judge_model,
            embedding_model=embedding_model,
            structure_info=struct_dicts,
            branch_rates=rate_dicts,
            continuations_by_branch=cont_dicts,
        )

    def save(self, path: str | Path) -> Path:
        """Save output to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                self.to_dict(), f, indent=2
            )  # No sort_keys to preserve field order
        return path

    def save_summary(self, path: str | Path) -> Path:
        """Save human-readable summary to text file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        labels = self.get_structure_labels()
        lines = []

        # Header
        lines.append("=" * 76)
        lines.append("  ESTIMATION SUMMARY")
        lines.append("=" * 76)
        lines.append("")

        # Metadata
        lines.append(f"  Generated: {self.estimated_at}")
        lines.append(f"  Judge:     {self.judge_model}")
        lines.append(f"  Embed:     {self.embedding_model}")
        lines.append("")

        # Structures
        lines.append("-" * 76)
        lines.append("  STRUCTURES")
        lines.append("-" * 76)
        for s in self.structure_info:
            if s.get("is_bundled"):
                lines.append(f"  {s['label']}: [BUNDLED]")
                for q in s.get("questions", []):
                    lines.append(f"      • {q}")
            else:
                lines.append(f"  {s['label']}: {s['description']}")
        lines.append("")

        # Results table
        col_w = 8
        header = "  " + "".join(f"{l:^{col_w}}" for l in labels)

        lines.append("-" * 76)
        lines.append("  RESULTS")
        lines.append("-" * 76)
        lines.append(f"  {'Arm':<14} {'N':>4}  {'E[∂]':>7}  {header.strip()}")
        lines.append("  " + "-" * 70)

        for arm in self.arms:
            core_str = "".join(f"{c:^{col_w}.3f}" for c in arm.core)
            lines.append(f"  {arm.name:<14} {len(arm.trajectories):>4}  {arm.deviance_avg:>7.4f}  {core_str}")
        lines.append("")

        # Inv-PPL weighted results
        has_inv_ppl = any(a.core_inv_ppl for a in self.arms)
        if has_inv_ppl:
            lines.append("  [inv-ppl weighted]")
            lines.append(f"  {'Arm':<14} {'N':>4}  {'E[∂]':>7}  {header.strip()}")
            lines.append("  " + "-" * 70)
            for arm in self.arms:
                if arm.core_inv_ppl:
                    core_str = "".join(f"{c:^{col_w}.3f}" for c in arm.core_inv_ppl)
                    lines.append(f"  {arm.name:<14} {len(arm.trajectories):>4}  {arm.deviance_avg_inv_ppl:>7.4f}  {core_str}")
            lines.append("")

        lines.append("=" * 76)

        with open(path, "w") as f:
            f.write("\n".join(lines))
        return path

    @staticmethod
    def compute_output_path(judgment_path: Path) -> Path:
        """Compute the output path for estimation results."""
        name = judgment_path.stem.replace("score_", "")
        return Path("out") / f"est_{name}.json"

    @staticmethod
    def compute_summary_path(judgment_path: Path) -> Path:
        """Compute the output path for summary results."""
        name = judgment_path.stem.replace("score_", "")
        return Path("out") / f"summary_est_{name}.txt"

    def get_structure_labels(self) -> list[str]:
        """Get structure labels from structure_info or generate defaults."""
        if self.structure_info:
            return [s["label"] for s in self.structure_info]
        # Generate default labels
        n_cat = len(self.categorical_judgements)
        n_sim = len(self.similarity_scoring)
        return [f"c{i+1}" for i in range(n_cat)] + [f"s{i+1}" for i in range(n_sim)]

    def summarize(self, show_variants: bool = True) -> None:
        """Print summary statistics.

        Args:
            show_variants: If True, show all core variants. If False, only standard core.
        """
        labels = self.get_structure_labels()

        # Show structure legend - full questions, not truncated
        log_banner("STRUCTURES")
        if self.structure_info:
            for s in self.structure_info:
                grouped_marker = " [BUNDLED]" if s.get("is_bundled") else ""
                if s.get("is_bundled") and s.get("questions"):
                    log(f"  {s['label']}: GROUPED ({len(s['questions'])} questions){grouped_marker}")
                    for q in s["questions"]:
                        log(f"      • {q}")
                else:
                    log(f"  {s['label']}: {s['description']}{grouped_marker}")
        else:
            for i, q in enumerate(self.categorical_judgements):
                if isinstance(q, list):
                    log(f"  c{i+1}: [BUNDLED] {len(q)} questions")
                    for qq in q:
                        log(f"      • {preview(qq, 50)}")
                else:
                    log(f"  c{i+1}: {preview(q, 55)}")
            for i, ref in enumerate(self.similarity_scoring):
                log(f"  s{i+1}: {preview(ref, 55)}")

        # Show per-branch rates
        log_banner("COMPLIANCE RATES BY BRANCH (% yes for categorical, avg for similarity)")
        if self.branch_rates:
            # Main table header
            header = f"  {'Branch':<12} {'N':>4}  " + "  ".join(f"{l:>6}" for l in labels)
            log(header)
            log_divider(18 + 8 * len(labels))

            for br in self.branch_rates:
                rates_str = "  ".join(
                    f"{br['structure_rates'].get(l, 0.0)*100:>5.1f}%" for l in labels
                )
                # Use branch_N naming instead of raw branch name
                branch_idx = br.get('branch_idx', 0)
                display_name = "trunk" if branch_idx == 0 else f"branch_{branch_idx}"
                log(f"  {display_name:<12} {br['trajectory_count']:>4}  {rates_str}")

            # Show breakdowns for grouped structures as separate tables
            for label, _ in [(l, None) for l in labels]:
                # Check if any branch has breakdown for this label
                has_breakdown = any(label in br.get("question_rates", {}) for br in self.branch_rates)
                if not has_breakdown:
                    continue

                log(f"\n  {label} breakdown:")
                # Get questions from first branch that has this breakdown
                questions = []
                for br in self.branch_rates:
                    q_rates = br.get("question_rates", {}).get(label, {})
                    if q_rates:
                        questions = list(q_rates.keys())
                        break

                if questions:
                    # Table header with branch names
                    def get_branch_header(br):
                        idx = br.get('branch_idx', 0)
                        return 'trunk' if idx == 0 else f'br_{idx}'
                    branch_headers = "  ".join(
                        f"{get_branch_header(br):>8}"
                        for br in self.branch_rates
                    )
                    log(f"    {'Question':<50}  {branch_headers}")
                    log_divider(52 + 10 * len(self.branch_rates), indent="    ")

                    for q in questions:
                        rates_row = "  ".join(
                            f"{br.get('question_rates', {}).get(label, {}).get(q, 0.0)*100:>7.1f}%"
                            for br in self.branch_rates
                        )
                        q_display = q[:48] + ".." if len(q) > 50 else q
                        log(f"    {q_display:<50}  {rates_row}")
        log("")

        # Show cores with labels
        log_banner("CORES BY ARM")
        for arm in self.arms:
            display_name = "trunk" if arm.arm_idx == 0 else f"branch_{arm.arm_idx}"
            log(f"\n  [{arm.arm_idx}] {display_name} ({len(arm.trajectories)} trajectories)")

            # Build labeled core display as table
            if arm.core or arm.core_inv_ppl:
                header_labels = "  ".join(f"{l:>6}" for l in labels)
                log(f"\n    {'weighting':<20}  {header_labels}  {'E[∂]':>8}  {'Var[∂]':>10}")
                log_divider(26 + 8 * len(labels) + 20, indent="    ")

                if arm.core:
                    core_vals = "  ".join(f"{arm.core[i]:>6.3f}" for i in range(len(arm.core)))
                    log(f"    {'prob-weighted':<20}  {core_vals}  {arm.deviance_avg:>8.4f}  {arm.deviance_var:>10.6f}")

                if arm.core_inv_ppl:
                    core_vals = "  ".join(f"{arm.core_inv_ppl[i]:>6.3f}" for i in range(len(arm.core_inv_ppl)))
                    log(f"    {'inv-ppl-weighted':<20}  {core_vals}  {arm.deviance_avg_inv_ppl:>8.4f}  {arm.deviance_var_inv_ppl:>10.6f}")

            if show_variants and arm.core_variants:
                _log_core_variants_labeled("probability-weighted", arm.core_variants, labels)

            if show_variants and arm.core_variants_inv_ppl:
                _log_core_variants_labeled("inv-perplexity-weighted", arm.core_variants_inv_ppl, labels)

        # Note: Detailed continuations are shown in Step 0 of the pipeline
        # Just show a summary here
        if self.continuations_by_branch:
            log_banner("TRAJECTORY COUNTS BY BRANCH")
            for branch, items in self.continuations_by_branch.items():
                log(f"  {branch}: {len(items)} trajectories")

        log_banner("")


def _log_core_variants_labeled(label: str, variants: list[CoreVariant], struct_labels: list[str]) -> None:
    """Log a table of core variants with structure labels (full cores, no truncation)."""
    log(f"\n    Core variants ({label}):")
    header_labels = "  ".join(f"{l:>6}" for l in struct_labels)
    log(f"    {'name':<14}  {'q':>4}  {'r':>4}    {header_labels}  {'E[∂]':>8}")
    log_divider(32 + 8 * len(struct_labels) + 10, indent="    ")
    for v in variants[:NUM_DISPLAYED_VARIANTS]:
        q_str = _format_qr(v.q)
        r_str = _format_qr(v.r)
        core_parts = "  ".join(f"{c:>6.3f}" for c in v.core)  # Full core, no truncation
        log(f"    {v.name:<14}  {q_str:>4}  {r_str:>4}    {core_parts}  {v.deviance_avg:>8.4f}")


def _log_core_variants(label: str, variants: list[CoreVariant]) -> None:
    """Log a table of core variants (full cores, no truncation)."""
    log(f"\n    Core variants ({label}):")
    log(f"    {'name':<14}  {'q':>4}  {'r':>4}    core")
    log_divider(70, indent="    ")
    for v in variants[:NUM_DISPLAYED_VARIANTS]:
        q_str = _format_qr(v.q)
        r_str = _format_qr(v.r)
        core_str = _format_core(v.core)  # Full core, no truncation
        log(f"    {v.name:<14}  {q_str:>4}  {r_str:>4}    {core_str}  E[∂]={v.deviance_avg:.4f}")


@dataclass
class StructureInfo(BaseSchema):
    """Information about a scoring structure."""

    idx: int  # Structure index (0-based)
    label: str  # Short label like "c1", "c2", "s1"
    description: str  # Full question or reference text
    is_bundled: bool  # Whether this is a bundled structure (multiple questions)
    question_count: int  # Number of questions (1 for single, N for bundled)
    questions: list[str]  # Individual questions (for bundled structures)


@dataclass
class BranchRates(BaseSchema):
    """Per-structure compliance rates for a branch."""

    branch: str
    branch_idx: int
    trajectory_count: int
    # Per-structure rates (% of 1s for categorical, avg for similarity)
    structure_rates: dict[str, float]  # label -> rate
    # For bundled structures, also track individual question rates
    question_rates: dict[str, dict[str, float]]  # label -> {question -> rate}


@dataclass
class JudgmentData(BaseSchema):
    """Loaded judgment data for estimation."""

    categorical_judgements: list[Any] = field(default_factory=list)  # str | list[str]
    graded_judgements: list[Any] = field(default_factory=list)  # str | list[str]
    similarity_scoring: list[Any] = field(default_factory=list)  # str | list[str]
    results: list[dict[str, Any]] = field(default_factory=list)
    branches: list[str] = field(default_factory=list)  # Branch names in config order
    arm_texts: dict[str, str] = field(default_factory=dict)  # arm_name -> conditioning text
    generation_file: str = ""
    scoring_file: str = ""
    judge_model: str = ""
    embedding_model: str = ""
    prefix_logprobs: dict[str, Any] = field(default_factory=dict)

    # Cached structure info
    _structure_info: list[StructureInfo] | None = field(default=None, repr=False)

    @classmethod
    def load(cls, path: str | Path) -> JudgmentData:
        """Load judgment output from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Judgment output not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        instance = cls(
            categorical_judgements=data.get("categorical_judgements", []),
            graded_judgements=data.get("graded_judgements", []),
            similarity_scoring=data.get("similarity_scoring", []),
            results=data.get("results", []),
            branches=data.get("branches", []),
            arm_texts=data.get("arm_texts", {}),
            generation_file=data.get("generation_file", ""),
            scoring_file=data.get("scoring_file", ""),
            judge_model=data.get("judge_model", ""),
            embedding_model=data.get("embedding_model", ""),
            prefix_logprobs=data.get("prefix_logprobs", {}),
        )
        instance.validate()
        return instance

    def validate(self) -> None:
        """Validate that the loaded data is usable for estimation."""
        if not self.results:
            raise ValueError("No results found in judgment file")
        if not self.categorical_judgements and not self.similarity_scoring:
            raise ValueError("No scoring methods found in judgment file")

    def get_structure_info(self) -> list[StructureInfo]:
        """Get information about all structures."""
        if self._structure_info is not None:
            return self._structure_info

        info = []
        idx = 0

        # Categorical structures
        for i, item in enumerate(self.categorical_judgements):
            if isinstance(item, list):
                info.append(StructureInfo(
                    idx=idx,
                    label=f"c{i+1}",
                    description=" + ".join(q[:30] + "..." if len(q) > 30 else q for q in item),
                    is_bundled=True,
                    question_count=len(item),
                    questions=item,
                ))
            else:
                info.append(StructureInfo(
                    idx=idx,
                    label=f"c{i+1}",
                    description=item,
                    is_bundled=False,
                    question_count=1,
                    questions=[item],
                ))
            idx += 1

        # Graded structures
        for i, item in enumerate(self.graded_judgements):
            if isinstance(item, list):
                info.append(StructureInfo(
                    idx=idx,
                    label=f"g{i+1}",
                    description=" + ".join(q[:30] + "..." if len(q) > 30 else q for q in item),
                    is_bundled=True,
                    question_count=len(item),
                    questions=item,
                ))
            else:
                info.append(StructureInfo(
                    idx=idx,
                    label=f"g{i+1}",
                    description=item,
                    is_bundled=False,
                    question_count=1,
                    questions=[item],
                ))
            idx += 1

        # Similarity structures
        for i, item in enumerate(self.similarity_scoring):
            if isinstance(item, list):
                info.append(StructureInfo(
                    idx=idx,
                    label=f"s{i+1}",
                    description=" + ".join(r[:30] + "..." if len(r) > 30 else r for r in item),
                    is_bundled=True,
                    question_count=len(item),
                    questions=item,
                ))
            else:
                info.append(StructureInfo(
                    idx=idx,
                    label=f"s{i+1}",
                    description=item,
                    is_bundled=False,
                    question_count=1,
                    questions=[item],
                ))
            idx += 1

        self._structure_info = info
        return info

    def get_text(self, traj_idx: int) -> str:
        """Get text for a trajectory by index."""
        for r in self.results:
            if r.get("trajectory_idx") == traj_idx:
                return r.get("text", "")
        return ""

    def get_texts(self) -> dict[int, str]:
        """Get all trajectory texts as {traj_idx: text}."""
        return {r["trajectory_idx"]: r["text"] for r in self.results}

    def get_response(self, traj_idx: int) -> str:
        """Get just the response/continuation text for a trajectory."""
        for r in self.results:
            if r.get("trajectory_idx") == traj_idx:
                # Response is stored separately or derived from text
                # For now, return full text - will be improved when we have response stored
                return r.get("text", "")
        return ""

    def get_raw_scores(self, result: dict) -> list[int | None]:
        """Get raw categorical scores (before grouping)."""
        return result.get("scores", [])

    def get_compliance(self, result: dict) -> list[float]:
        """Convert scores to compliance Λ_n(x) with grouped averaging.

        For grouped structures, the individual question scores are averaged
        to produce a single compliance value for the structure.
        None values are treated as 0.5.
        """
        raw_scores = result.get("scores", [])
        graded_scores = result.get("graded_scores", [])
        similarities = result.get("similarity_scores", [])

        # Process categorical judgements with grouping
        compliance = []
        score_idx = 0

        for item in self.categorical_judgements:
            if isinstance(item, list):
                # Grouped structure: average the questions
                group_scores = []
                for _ in item:
                    if score_idx < len(raw_scores):
                        s = raw_scores[score_idx]
                        group_scores.append(float(s) if s is not None else 0.5)
                    else:
                        group_scores.append(0.5)
                    score_idx += 1
                # Average for the group
                compliance.append(sum(group_scores) / len(group_scores) if group_scores else 0.5)
            else:
                # Single structure
                if score_idx < len(raw_scores):
                    s = raw_scores[score_idx]
                    compliance.append(float(s) if s is not None else 0.5)
                else:
                    compliance.append(0.5)
                score_idx += 1

        # Process graded judgements with grouping
        graded_idx = 0
        for item in self.graded_judgements:
            if isinstance(item, list):
                # Grouped structure: average the questions
                group_scores = []
                for _ in item:
                    if graded_idx < len(graded_scores):
                        s = graded_scores[graded_idx]
                        group_scores.append(float(s) if s is not None else 0.5)
                    else:
                        group_scores.append(0.5)
                    graded_idx += 1
                # Average for the group
                compliance.append(sum(group_scores) / len(group_scores) if group_scores else 0.5)
            else:
                # Single structure
                if graded_idx < len(graded_scores):
                    s = graded_scores[graded_idx]
                    compliance.append(float(s) if s is not None else 0.5)
                else:
                    compliance.append(0.5)
                graded_idx += 1

        # Process similarity scores with grouping
        sim_idx = 0
        for item in self.similarity_scoring:
            if isinstance(item, list):
                # Grouped structure: average the references
                group_scores = []
                for _ in item:
                    if sim_idx < len(similarities):
                        group_scores.append(float(similarities[sim_idx]))
                    else:
                        group_scores.append(0.5)
                    sim_idx += 1
                compliance.append(sum(group_scores) / len(group_scores) if group_scores else 0.5)
            else:
                # Single reference
                if sim_idx < len(similarities):
                    compliance.append(float(similarities[sim_idx]))
                else:
                    compliance.append(0.5)
                sim_idx += 1

        return compliance

    def compute_branch_rates(self) -> list[BranchRates]:
        """Compute per-structure compliance rates for each branch."""
        structures = self.get_structure_info()
        branch_names = self.branches if self.branches else ["trunk"]

        # Group results by branch
        by_branch: dict[str, list[dict]] = {}
        for result in self.results:
            branch = result.get("branch", "trunk")
            by_branch.setdefault(branch, []).append(result)

        rates = []
        for branch_idx, branch in enumerate(branch_names):
            results = by_branch.get(branch, [])
            if not results:
                continue

            structure_rates = {}
            question_rates = {}

            # Compute rates for each structure
            for struct in structures:
                if struct.label.startswith("c"):
                    # Categorical: compute % of 1s
                    if struct.is_bundled:
                        # Track both aggregate and individual rates
                        group_values = []
                        q_rates = {}
                        for q_idx, question in enumerate(struct.questions):
                            q_values = []
                            for r in results:
                                raw_scores = r.get("scores", [])
                                # Find the score index for this question
                                flat_idx = self._get_flat_score_index(struct.idx, q_idx)
                                if flat_idx < len(raw_scores) and raw_scores[flat_idx] is not None:
                                    q_values.append(raw_scores[flat_idx])
                            if q_values:
                                q_rate = sum(q_values) / len(q_values)
                                q_rates[question[:50]] = q_rate
                                group_values.extend(q_values)
                        # Aggregate rate for grouped structure = avg % yes across all questions
                        structure_rates[struct.label] = sum(group_values) / len(group_values) if group_values else 0.0
                        question_rates[struct.label] = q_rates
                    else:
                        # Single question
                        values = []
                        flat_idx = self._get_flat_score_index(struct.idx, 0)
                        for r in results:
                            raw_scores = r.get("scores", [])
                            if flat_idx < len(raw_scores) and raw_scores[flat_idx] is not None:
                                values.append(raw_scores[flat_idx])
                        structure_rates[struct.label] = sum(values) / len(values) if values else 0.0
                elif struct.label.startswith("g"):
                    # Graded: compute average score (0-1 scale)
                    # struct.idx is the index within graded_judgements
                    graded_struct_idx = struct.idx - len(self.categorical_judgements)
                    if struct.is_bundled:
                        # Track both aggregate and individual rates
                        group_values = []
                        q_rates = {}
                        for q_idx, question in enumerate(struct.questions):
                            q_values = []
                            for r in results:
                                graded_scores = r.get("graded_scores", [])
                                flat_idx = self._get_flat_graded_index(graded_struct_idx, q_idx)
                                if flat_idx < len(graded_scores) and graded_scores[flat_idx] is not None:
                                    q_values.append(graded_scores[flat_idx])
                            if q_values:
                                q_rate = sum(q_values) / len(q_values)
                                q_rates[question[:50]] = q_rate
                                group_values.extend(q_values)
                        structure_rates[struct.label] = sum(group_values) / len(group_values) if group_values else 0.0
                        question_rates[struct.label] = q_rates
                    else:
                        # Single question
                        values = []
                        flat_idx = self._get_flat_graded_index(graded_struct_idx, 0)
                        for r in results:
                            graded_scores = r.get("graded_scores", [])
                            if flat_idx < len(graded_scores) and graded_scores[flat_idx] is not None:
                                values.append(graded_scores[flat_idx])
                        structure_rates[struct.label] = sum(values) / len(values) if values else 0.0
                else:
                    # Similarity: compute average
                    sim_struct_idx = struct.idx - len(self.categorical_judgements) - len(self.graded_judgements)
                    if struct.is_bundled:
                        # Track both aggregate and individual rates
                        group_values = []
                        q_rates = {}
                        for ref_idx, ref in enumerate(struct.questions):
                            ref_values = []
                            for r in results:
                                sims = r.get("similarity_scores", [])
                                flat_idx = self._get_flat_similarity_index(sim_struct_idx, ref_idx)
                                if flat_idx < len(sims):
                                    ref_values.append(sims[flat_idx])
                            if ref_values:
                                ref_rate = sum(ref_values) / len(ref_values)
                                q_rates[ref[:50]] = ref_rate
                                group_values.extend(ref_values)
                        structure_rates[struct.label] = sum(group_values) / len(group_values) if group_values else 0.0
                        question_rates[struct.label] = q_rates
                    else:
                        # Single reference
                        values = []
                        flat_idx = self._get_flat_similarity_index(sim_struct_idx, 0)
                        for r in results:
                            sims = r.get("similarity_scores", [])
                            if flat_idx < len(sims):
                                values.append(sims[flat_idx])
                        structure_rates[struct.label] = sum(values) / len(values) if values else 0.0

            rates.append(BranchRates(
                branch=branch,
                branch_idx=branch_idx,
                trajectory_count=len(results),
                structure_rates=structure_rates,
                question_rates=question_rates,
            ))

        return rates

    def _get_flat_score_index(self, struct_idx: int, question_idx: int) -> int:
        """Get the flat index into raw_scores for a structure/question."""
        flat_idx = 0
        for i, item in enumerate(self.categorical_judgements):
            if i == struct_idx:
                return flat_idx + question_idx
            if isinstance(item, list):
                flat_idx += len(item)
            else:
                flat_idx += 1
        return flat_idx

    def _get_flat_graded_index(self, struct_idx: int, question_idx: int) -> int:
        """Get the flat index into graded_scores for a structure/question."""
        flat_idx = 0
        for i, item in enumerate(self.graded_judgements):
            if i == struct_idx:
                return flat_idx + question_idx
            if isinstance(item, list):
                flat_idx += len(item)
            else:
                flat_idx += 1
        return flat_idx

    def _get_flat_similarity_index(self, struct_idx: int, ref_idx: int) -> int:
        """Get the flat index into similarity_scores for a structure/reference."""
        flat_idx = 0
        for i, item in enumerate(self.similarity_scoring):
            if i == struct_idx:
                return flat_idx + ref_idx
            if isinstance(item, list):
                flat_idx += len(item)
            else:
                flat_idx += 1
        return flat_idx

    def group_by_branch(self) -> dict[str, list[TrajectoryCompliance]]:
        """Group results by branch, returning TrajectoryCompliance objects."""
        grouped: dict[str, list[TrajectoryCompliance]] = {}
        for result in self.results:
            branch = result.get("branch", "trunk")
            idx = result["trajectory_idx"]
            compliance = self.get_compliance(result)
            conditional_logprobs = result.get("conditional_logprobs", {})

            if branch not in grouped:
                grouped[branch] = []
            grouped[branch].append(
                TrajectoryCompliance(
                    traj_idx=idx,
                    branch=branch,
                    compliances=compliance,
                    conditional_logprobs=conditional_logprobs,
                    n_continuation_tokens=result.get("n_continuation_tokens", 0),
                )
            )

        return grouped

    def get_continuations_by_branch(self) -> dict[str, list[tuple[int, str]]]:
        """Get continuations organized by branch.

        Returns:
            Dict mapping branch name to list of (traj_idx, continuation_text) tuples.
        """
        by_branch: dict[str, list[tuple[int, str]]] = {}
        for result in self.results:
            branch = result.get("branch", "trunk")
            idx = result["trajectory_idx"]
            text = result.get("text", "")
            by_branch.setdefault(branch, []).append((idx, text))
        return by_branch
