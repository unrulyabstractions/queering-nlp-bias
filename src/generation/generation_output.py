"""Output dataclass for trajectory generation.

This module defines the GenerationOutput class which holds the results
of trajectory generation including the token tree structure.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from src.common.base_schema import BaseSchema
from src.common.logging import log, log_banner, log_sub_banner
from src.common.token_tree import TokenTree

from .generation_config import GenerationConfig

# ══════════════════════════════════════════════════════════════════════════════
# HELPER TYPES
# ══════════════════════════════════════════════════════════════════════════════

# Type alias for output functions (log or file writer)
OutputFn = Callable[[str], None]


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def _get_eos_markers(eos_token: str | None) -> list[str]:
    """Get EOS markers for finished trajectory detection."""
    return (
        [eos_token]
        if eos_token
        else ["<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>"]
    )


def _group_trajectories_by_branch(
    trajs: list[dict],
) -> dict[int, list[dict]]:
    """Group trajectories by branch index."""
    by_branch: dict[int, list[dict]] = {}
    for traj in trajs:
        arm_index = traj.get("arm_index", [0])
        if isinstance(arm_index, list):
            branch_idx = arm_index[0] if arm_index else 0
        else:
            branch_idx = arm_index
        by_branch.setdefault(branch_idx, []).append(traj)
    return by_branch


def _count_finished(trajs: list[dict], eos_markers: list[str]) -> int:
    """Count trajectories that contain an EOS marker."""
    return sum(
        1
        for t in trajs
        if any(eos in t.get("continuation_text", "") for eos in eos_markers)
    )


def _format_branch_stats(
    trajs: list[dict],
    eos_markers: list[str],
    display_name: str,
) -> tuple[str, float]:
    """Format branch statistics (count and finished percentage).

    Returns:
        Tuple of (header_string, finished_percentage)
    """
    finished = _count_finished(trajs, eos_markers)
    pct = (finished / len(trajs) * 100) if trajs else 0
    return f"{display_name} ({len(trajs)} trajectories, {pct:.0f}% finished)", pct


def _compute_branch_probability_mass(
    trajs: list[dict],
    trunk_len: int,
) -> float:
    """Compute sum of conditional probabilities for trajectories in a branch."""
    total_cond_prob = 0.0
    for traj in trajs:
        logprobs = traj.get("logprobs", [])
        if len(logprobs) > trunk_len:
            cont_logprobs = logprobs[trunk_len:]
            cont_logp = sum(lp for lp in cont_logprobs if lp is not None)
            if cont_logp > -700:
                total_cond_prob += math.exp(cont_logp)
    return total_cond_prob


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT CLASS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class GenerationOutput(BaseSchema):
    """Output from trajectory generation, including tree structure."""

    config: dict[str, Any]  # GenerationConfig.to_dict()
    model: str
    method: str  # Generation method: simple-sampling, forking-paths, seeking-entropy
    generated_at: str
    num_trajectories: int
    tree: dict[str, Any] | None = None  # TokenTree.to_dict() output
    eos_token: str | None = None  # EOS token from model (for finished detection)

    @classmethod
    def from_tree(
        cls,
        config: GenerationConfig,
        model: str,
        tree: TokenTree,
        method: str = "simple-sampling",
        eos_token: str | None = None,
    ) -> GenerationOutput:
        """Create output from a TokenTree.

        Args:
            config: Generation configuration
            model: Model name used
            tree: TokenTree with trajectories
            method: Generation method name
            eos_token: EOS token from the model (for finished detection)
        """
        # Pop heavy data before serializing (full_logits tensors)
        tree.pop_heavy()

        # Build config dict with trunk included in branches list
        config_dict = config.to_dict()
        config_dict["branches"] = ["trunk"] + config_dict.get("branches", [])

        return cls(
            config=config_dict,
            model=model,
            method=method,
            generated_at=datetime.now().isoformat(),
            num_trajectories=len(tree.trajs),
            tree=tree.to_dict(max_list_length=10000),
            eos_token=eos_token,
        )

    @staticmethod
    def compute_output_path(config_path: Path, method: str = "sampling") -> Path:
        """Compute the output path for generation results."""
        return Path("out") / f"gen_{method}_{config_path.stem}.json"

    @staticmethod
    def compute_summary_path(config_path: Path, method: str = "sampling") -> Path:
        """Compute the output path for generation summary."""
        return Path("out") / f"summary_gen_{method}_{config_path.stem}.txt"

    def save(self, path: str | Path) -> Path:
        """Save output to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
        return path

    # ──────────────────────────────────────────────────────────────────────────
    # Summary Methods
    # ──────────────────────────────────────────────────────────────────────────

    def _write_settings(self, out: OutputFn, prefix: str = "  ") -> None:
        """Write settings section."""
        out(f"{prefix}Model:       {self.model}")
        out(f"{prefix}Method:      {self.method}")
        out(f"{prefix}Generated:   {self.generated_at}")
        out(f"{prefix}Trajectories:{self.num_trajectories}")

    def _write_config(
        self, out: OutputFn, prefix: str = "  ", truncate: bool = True
    ) -> None:
        """Write config section."""
        prompt = self.config.get("prompt", "")
        trunk = self.config.get("trunk", "")
        branches = self.config.get("branches", [])
        real_branches = [b for b in branches if b != "trunk"]

        if truncate and len(prompt) > 70:
            out(f"{prefix}Prompt: {prompt[:70]}...")
        else:
            out(f"{prefix}Prompt: {prompt}")

        if trunk:
            out(f'{prefix}Trunk:  "{trunk}"')
        if real_branches:
            out(
                f"{prefix}Branches: {', '.join(f'{chr(34)}{b}{chr(34)}' for b in real_branches)}"
            )
        out(f"{prefix}Temperature: {self.config.get('temperature', 1.0)}")
        out(f"{prefix}Max tokens:  {self.config.get('max_new_tokens', 128)}")

    def _write_trajectories_by_branch(
        self,
        out: OutputFn,
        prefix: str = "  ",
        max_trajs: int = 3,
        max_text_len: int = 60,
    ) -> None:
        """Write trajectories grouped by branch."""
        if not self.tree:
            return

        trajs = self.tree.get("trajs", [])
        branches = self.config.get("branches", [])
        eos_markers = _get_eos_markers(self.eos_token)
        by_branch = _group_trajectories_by_branch(trajs)

        for branch_idx in range(len(branches)):
            trajs_in_branch = by_branch.get(branch_idx, [])
            display_name = "trunk" if branch_idx == 0 else f"branch_{branch_idx}"
            header, _ = _format_branch_stats(trajs_in_branch, eos_markers, display_name)

            out(f"\n{prefix}{header}:")
            for i, traj in enumerate(trajs_in_branch[:max_trajs]):
                text = traj.get("continuation_text", "")[:max_text_len]
                out(f"{prefix}  [{traj.get('idx', i)}] {text}")

            if len(trajs_in_branch) > max_trajs:
                out(f"{prefix}  ... and {len(trajs_in_branch) - max_trajs} more")

    def _write_probability_mass(self, out: OutputFn, prefix: str = "  ") -> None:
        """Write probability mass captured per branch."""
        if not self.tree:
            return

        trajs = self.tree.get("trajs", [])
        branches = self.config.get("branches", [])
        trunk_len = self.tree.get("trunk_length", 0)
        by_branch = _group_trajectories_by_branch(trajs)

        out(
            f"\n{prefix}{'Branch':<12} {'N':>4}  {'Sum(p|branch)':>14}  {'Coverage':>10}"
        )
        out(f"{prefix}" + "─" * 46)

        for branch_idx in range(len(branches)):
            trajs_in_branch = by_branch.get(branch_idx, [])
            display_name = "trunk" if branch_idx == 0 else f"branch_{branch_idx}"

            total_cond_prob = _compute_branch_probability_mass(
                trajs_in_branch, trunk_len
            )
            coverage_pct = total_cond_prob * 100

            prob_str = f"{total_cond_prob:.2e}" if total_cond_prob > 0 else "0"
            coverage_str = (
                f"{coverage_pct:.1f}%" if coverage_pct < 100 else f"{coverage_pct:.0f}%"
            )

            out(
                f"{prefix}{display_name:<12} {len(trajs_in_branch):>4}  "
                f"{prob_str:>14}  {coverage_str:>10}"
            )

    def save_summary(self, path: str | Path) -> Path:
        """Save human-readable summary to text file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []

        def add_line(text: str = "") -> None:
            lines.append(text)

        # Header
        add_line("=" * 76)
        add_line("  GENERATION SUMMARY")
        add_line("=" * 76)
        add_line()

        # Settings
        self._write_settings(add_line)
        add_line()

        # Config
        add_line("-" * 76)
        add_line("  CONFIG")
        add_line("-" * 76)
        self._write_config(add_line, truncate=True)
        add_line()

        # Trajectories
        if self.tree:
            add_line("-" * 76)
            add_line("  TRAJECTORIES")
            add_line("-" * 76)
            self._write_trajectories_by_branch(add_line, max_trajs=999999, max_text_len=999999)

        add_line()
        add_line("=" * 76)

        with open(path, "w") as f:
            f.write("\n".join(lines))
        return path

    def summarize(self) -> None:
        """Print a clean summary of generation results."""
        log_banner("GENERATION SUMMARY")

        # Settings
        log("\nSettings:")
        self._write_settings(log)

        # Config - show full prompt
        prompt = self.config.get("prompt", "")
        trunk = self.config.get("trunk", "")
        branches = self.config.get("branches", [])
        real_branches = [b for b in branches if b != "trunk"]

        log("\n  Prompt:")
        for line in prompt.split("\n"):
            log(f"    {line}")
        if trunk:
            log(f'\n  Trunk (shared prefix): "{trunk}"')
        if real_branches:
            log(f"\n  Branches ({len(real_branches)}):")
            for i, branch in enumerate(real_branches):
                log(f'    [{i + 1}] "{branch}"')

        # Generation params
        temp = self.config.get("temperature", 1.0)
        max_tokens = self.config.get("max_new_tokens", 128)
        log(f"  Temperature: {temp}")
        log(f"  Max tokens: {max_tokens}")

        # Trajectories by branch
        log_sub_banner("CONTINUATIONS BY BRANCH")
        self._write_trajectories_by_branch(log, max_trajs=5, max_text_len=80)

        # Probability mass
        if self.tree:
            log_sub_banner("PROBABILITY MASS CAPTURED PER BRANCH")
            self._write_probability_mass(log)

        # Final stats
        log_sub_banner(f"Total trajectories: {self.num_trajectories}")
