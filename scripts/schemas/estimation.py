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
from src.common.math.entropy_diversity.structure_aware import (
    deviance,
    deviance_variance,
    expected_deviance,
    generalized_system_core,
    orientation,
)

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


def _format_core(core: list[float], max_items: int = 3) -> str:
    """Format core vector for display."""
    if not core:
        return "[]"
    items = ", ".join(f"{c:.2f}" for c in core[:max_items])
    if len(core) > max_items:
        return f"[{items}, ...]"
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

NAMED_CORES: list[tuple[str, float, float, str]] = [
    # Paper's five special cases
    ("standard", 1.0, 1.0, "⟨α⟩ standard expected compliance"),
    ("uniform", 1.0, 0.0, "uniform avg over support"),
    ("mode", 1.0, float("inf"), "compliance of mode"),
    ("max", float("inf"), 1.0, "max compliance in support"),
    ("mode_min", float("-inf"), float("inf"), "min compliance among modes"),
    # Additional interesting cases - varying r (which trajectories)
    ("antimode", 1.0, float("-inf"), "compliance of rarest (anti-mode)"),
    ("confident", 1.0, 2.0, "confident core (squared prob weighting)"),
    ("inverse", 1.0, -1.0, "inverse probability weighting"),
    # Additional interesting cases - varying q (how to aggregate)
    ("geometric", 0.0, 1.0, "geometric mean (sensitive to exclusion)"),
    ("harmonic", -1.0, 1.0, "harmonic mean (penalizes low compliance)"),
    ("rms", 2.0, 1.0, "root-mean-square"),
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
    group_idxs: list[int]  # Trajectory can belong to multiple groups
    continuation_text: str


@dataclass
class GroupSummary(BaseSchema):
    """Summary of a group definition."""

    group_idx: int
    name: str
    trajectory_count: int


@dataclass
class EstimationSummary(BaseSchema):
    """Summary of trajectories and groups for easy lookup."""

    trajectories: list[TrajectorySummary]
    groups: list[GroupSummary]


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
class GroupEstimate(BaseSchema):
    """Estimation results for a group (trunk or branch).

    Two weighting schemes are computed:
    1. Probability-weighted: uses p(traj) as weights
       - The (q, r) generalization controls escort order and power mean order
    2. Inv-perplexity-weighted: uses 1/ppl = exp(logp/n_tokens) as weights
       - Beyond the (q, r) generalization: different base measure entirely
       - Weights by model confidence per token, not raw probability
    """

    group_idx: int
    name: str
    # Primary cores (q=1, r=1 with different weighting schemes)
    core: list[float]  # ⟨Λ_n⟩ probability-weighted
    core_inv_ppl: list[float]  # ⟨Λ_n⟩ inv-perplexity-weighted
    trajectories: list[TrajectoryEstimate]
    # Deviance stats for probability-weighted core
    deviance_avg: float  # E[∂_n]
    deviance_var: float  # Var[∂_n]
    # Deviance stats for inv-ppl-weighted core
    deviance_avg_inv_ppl: float  # E[∂_n] with inv-ppl weights
    deviance_var_inv_ppl: float  # Var[∂_n] with inv-ppl weights
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
        group_idx: int,
        name: str,
        trajectories: list[TrajectoryCompliance],
    ) -> GroupEstimate:
        """Create group estimate from trajectory compliances.

        Uses probability-weighted calculations:
        - Core: generalized_system_core with trajectory probabilities
        - Deviance stats: expected_deviance/variance with probability weights

        Log probabilities are converted to normalized probabilities to avoid
        underflow issues with long sequences.

        Computes multiple core variants:
        - Standard probability-weighted cores with various (q, r) settings
        - Inverse-perplexity weighted cores (weighting by model confidence)

        Args:
            group_idx: Index of this group
            name: Name of this group (e.g., "trunk", "branch_a")
            trajectories: Trajectories with their compliance scores and log probabilities
        """
        n_trajs = len(trajectories)

        # Handle empty group: return neutral estimate
        if n_trajs == 0:
            return cls(
                group_idx=group_idx,
                name=name,
                core=[],
                core_inv_ppl=[],
                trajectories=[],
                deviance_avg=0.0,
                deviance_var=0.0,
                deviance_avg_inv_ppl=0.0,
                deviance_var_inv_ppl=0.0,
                core_variants=[],
                core_variants_inv_ppl=[],
            )

        # Extract compliances and log probabilities (conditioned on this group)
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

        # Compute all named core variants with probability weights
        core_variants = cls._compute_core_variants(compliances, probs)

        # Compute all named core variants with inverse perplexity weights
        core_variants_inv_ppl = cls._compute_core_variants(compliances, inv_ppl_weights)

        return cls(
            group_idx=group_idx,
            name=name,
            core=core,
            core_inv_ppl=core_inv_ppl,
            trajectories=traj_estimates,
            deviance_avg=dev_avg,
            deviance_var=dev_var,
            deviance_avg_inv_ppl=dev_avg_inv_ppl,
            deviance_var_inv_ppl=dev_var_inv_ppl,
            core_variants=core_variants,
            core_variants_inv_ppl=core_variants_inv_ppl,
        )


@dataclass
class EstimationOutput(BaseSchema):
    """Output from normativity estimation."""

    summary: EstimationSummary
    categorical_judgements: list[str]
    similarity_scoring: list[str]
    groups: list[GroupEstimate]
    judgment_file: str
    estimated_at: str

    @classmethod
    def create(
        cls,
        judgment_file: str,
        categorical_judgements: list[str],
        similarity_scoring: list[str],
        groups: list[GroupEstimate],
        texts: dict[int, str],
    ) -> EstimationOutput:
        """Create estimation output with auto-generated summary."""
        # Build trajectory -> groups mapping
        traj_to_groups: dict[int, list[int]] = {}
        for group in groups:
            for traj in group.trajectories:
                traj_to_groups.setdefault(traj.traj_idx, []).append(group.group_idx)

        summary = EstimationSummary(
            trajectories=[
                TrajectorySummary(
                    traj_idx=idx, group_idxs=gids, continuation_text=texts.get(idx, "")
                )
                for idx, gids in sorted(traj_to_groups.items())
            ],
            groups=[
                GroupSummary(g.group_idx, g.name, len(g.trajectories)) for g in groups
            ],
        )

        return cls(
            summary=summary,
            categorical_judgements=categorical_judgements,
            similarity_scoring=similarity_scoring,
            groups=groups,
            judgment_file=judgment_file,
            estimated_at=datetime.now().isoformat(),
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

    @staticmethod
    def compute_output_path(judgment_path: Path) -> Path:
        """Compute the output path for estimation results."""
        name = judgment_path.stem.replace("score_", "")
        return Path("out") / f"est_{name}.json"

    def summarize(self, show_variants: bool = True) -> None:
        """Print summary statistics.

        Args:
            show_variants: If True, show all core variants. If False, only standard core.
        """
        log(f"\nEstimation summary ({len(self.groups)} groups):")
        for group in self.groups:
            log(f"\n  [{group.group_idx}] {group.name} ({len(group.trajectories)} trajectories):")
            log("    ┌─ Probability-weighted:")
            log(f"    │  core:     [{', '.join(f'{c:.3f}' for c in group.core)}]")
            log(f"    │  deviance: avg={group.deviance_avg:.4f}, var={group.deviance_var:.6f}")
            log("    └─ Inv-perplexity-weighted (beyond q,r - weights by 1/ppl):")
            core_inv = group.core_inv_ppl if group.core_inv_ppl else []
            log(f"       core:     [{', '.join(f'{c:.3f}' for c in core_inv)}]")
            log(f"       deviance: avg={group.deviance_avg_inv_ppl:.4f}, var={group.deviance_var_inv_ppl:.6f}")

            if show_variants and group.core_variants:
                _log_core_variants("probability-weighted", group.core_variants)

            if show_variants and group.core_variants_inv_ppl:
                _log_core_variants("inv-perplexity-weighted", group.core_variants_inv_ppl)


def _log_core_variants(label: str, variants: list[CoreVariant]) -> None:
    """Log a table of core variants."""
    log(f"\n    Core variants ({label}):")
    log(f"    {'name':<14}  {'q':>4}  {'r':>4}    {'core':^24}  {'E[∂]':>8}")
    log("    " + "─" * 62)
    for v in variants:
        q_str = _format_qr(v.q)
        r_str = _format_qr(v.r)
        core_str = _format_core(v.core)
        log(f"    {v.name:<14}  {q_str:>4}  {r_str:>4}    {core_str:^24}  {v.deviance_avg:>8.4f}")


@dataclass
class JudgmentData(BaseSchema):
    """Loaded judgment data for estimation."""

    categorical_judgements: list[str] = field(default_factory=list)
    similarity_scoring: list[str] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    branches: list[str] = field(default_factory=list)  # Branch names in config order
    groups: dict[str, str] = field(default_factory=dict)  # group_name -> text
    generation_file: str = ""
    prefix_logprobs: dict[str, Any] = field(default_factory=dict)

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
            similarity_scoring=data.get("similarity_scoring", []),
            results=data.get("results", []),
            branches=data.get("branches", []),
            groups=data.get("groups", {}),
            generation_file=data.get("generation_file", ""),
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

    def get_text(self, traj_idx: int) -> str:
        """Get text for a trajectory by index."""
        for r in self.results:
            if r.get("trajectory_idx") == traj_idx:
                return r.get("text", "")
        return ""

    def get_texts(self) -> dict[int, str]:
        """Get all trajectory texts as {traj_idx: text}."""
        return {r["trajectory_idx"]: r["text"] for r in self.results}

    def get_compliance(self, result: dict) -> list[float]:
        """Convert scores to compliance Λ_n(x) (None -> 0.5)."""
        scores = result.get("scores", [])
        similarities = result.get("similarity_scores", [])
        # Combine categorical scores and similarity scores
        cat = [float(s) if s is not None else 0.5 for s in scores]
        sim = [float(s) for s in similarities]
        return cat + sim

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
