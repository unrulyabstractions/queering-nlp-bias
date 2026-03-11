"""Data types for trajectory scoring.

This module defines data structures for loaded generation output
and trajectory data extracted for scoring.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.text import arm_display_name
from src.common.token_tree import TokenTree


@dataclass
class TrajectoryData:
    """Data extracted from a trajectory for judgment."""

    trajectory_idx: int
    branch: str  # Display name (trunk, branch_1, etc.) - NOT raw branch text
    branch_idx: int  # Index of branch in config order (0=trunk, 1=branch_1, etc.)
    prompt: str  # The prompt/trunk text (includes chat template)
    response: str  # Continuation after trunk (includes branch token for branch trajs)
    response_after_branch: str  # Continuation after branch token (branch stripped)
    conditional_logprobs: dict[str, float]  # Log prob conditioned on each arm
    n_continuation_tokens: int = 0  # Number of tokens in continuation

    @property
    def full_text(self) -> str:
        """Full text (prompt + response) for judgment."""
        return self.prompt + self.response


@dataclass
class ArmDefinitions:
    """Defines the conditioning text for each arm (trunk or branch_N)."""

    texts: dict[str, str]  # arm_name -> text at that conditioning level
    token_lengths: dict[str, int]  # arm_name -> token length

    @classmethod
    def from_tree(cls, tree: TokenTree, branches: list[str]) -> ArmDefinitions:
        """Build arm definitions from a token tree."""
        trunk_text = tree.trunk_text or ""
        trunk_length = tree.trunk_length or 0

        texts = {"trunk": trunk_text}
        token_lengths = {"trunk": trunk_length}

        # For each branch, we need to find the text including the branch token
        # The branch token is at position trunk_length
        for branch in branches:
            # Find a trajectory in this branch to get the branch token
            for traj in tree.trajs:
                if traj.arm_index and len(branches) > traj.arm_index[0]:
                    if branches[traj.arm_index[0]] == branch:
                        # Get text up to and including the branch token
                        branch_text = trunk_text + (traj.continuation_text or "")[:50]
                        # Just use trunk + first part of continuation as approximation
                        # The exact text would need decoding the branch token
                        texts[branch] = f"{trunk_text}..."  # Placeholder
                        token_lengths[branch] = trunk_length + 1
                        break

        return cls(texts=texts, token_lengths=token_lengths)


@dataclass
class GenerationOutputData:
    """Loaded generation output with extracted trajectory data."""

    tree: TokenTree | None
    trajectories: list[TrajectoryData]
    config: dict[str, Any]
    branches: list[str]  # Branch names in config order
    arm_texts: dict[str, str]  # arm_name -> conditioning text
    prefix_logprobs: dict[str, Any] | None = None  # Conditional logprobs for prefixes
    eos_token: str | None = None  # EOS token from the model that generated trajectories

    @classmethod
    def load(cls, path: str | Path) -> GenerationOutputData:
        """Load generation output from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Generation output not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        config = data.get("config", {})
        branches = config.get("branches", [])

        # Try to load tree if present
        tree = None
        if data.get("tree"):
            tree = TokenTree.from_dict(data["tree"])

        # Build arm conditioning texts
        arm_texts: dict[str, str] = {}
        trajectories: list[TrajectoryData] = []

        # Extract prefix logprobs from trajectories
        prefix_logprobs: dict[str, Any] = {
            "trunk_given_prompt": 0.0,
            "branch_given_trunk": {},
        }

        if tree:
            trunk_text = tree.trunk_text or ""
            trunk_length = tree.trunk_length or 0
            prompt_length = tree.prompt_length or 0

            # Build arm conditioning texts using DISPLAY names as keys:
            # - "trunk": just trunk_text
            # - "branch_N": trunk_text + raw_branch_text
            for raw_idx, raw_branch in enumerate(branches):
                display_name = arm_display_name(raw_idx)
                if raw_branch == "trunk":
                    arm_texts[display_name] = trunk_text
                else:
                    arm_texts[display_name] = trunk_text + raw_branch

            # Extract trajectories with conditional logprobs
            for i, traj in enumerate(tree.trajs):
                continuation_text = traj.continuation_text or ""

                # Get branch index from arm_index
                if traj.arm_index and len(branches) > traj.arm_index[0]:
                    branch_idx = traj.arm_index[0]
                    raw_branch = branches[branch_idx]
                else:
                    branch_idx = 0
                    raw_branch = "trunk"

                # Use display name for output
                display_name = arm_display_name(branch_idx)

                # response = continuation after trunk (includes branch token for branch trajs)
                response = continuation_text

                # response_after_branch = continuation after branch token (branch stripped)
                # Use raw_branch for stripping since it's the actual text
                if branch_idx > 0 and continuation_text.startswith(raw_branch):
                    response_after_branch = continuation_text[len(raw_branch) :]
                else:
                    response_after_branch = continuation_text

                # Extract prefix logprobs (once per branch)
                if (
                    trunk_length > prompt_length
                    and prefix_logprobs["trunk_given_prompt"] == 0.0
                ):
                    # p(trunk | prompt) - sum logprobs for trunk tokens only (not prompt)
                    prefix_logprobs["trunk_given_prompt"] = sum(
                        traj.logprobs[prompt_length:trunk_length]
                    )

                if (
                    branch_idx > 0
                    and branch_idx not in prefix_logprobs["branch_given_trunk"]
                ):
                    # p(branch | prompt + trunk) - logprob of branch token at trunk_length
                    if trunk_length < len(traj.logprobs):
                        prefix_logprobs["branch_given_trunk"][branch_idx] = (
                            traj.logprobs[trunk_length]
                        )

                # Compute conditional log probabilities using display names as keys
                conditional_logprobs: dict[str, float] = {}

                # For branch trajectories, BPE may merge trunk+branch differently.
                # The continuation starts at position trunk_length for trunk trajs,
                # but for branch trajs it also starts at trunk_length because BPE
                # absorbed the trunk space into the branch token.

                # p(continuation | trunk) - sum from trunk_length onwards
                if branch_idx == 0:
                    # Trunk trajectory: continuation starts at trunk_length
                    conditional_logprobs["trunk"] = sum(traj.logprobs[trunk_length:])
                else:
                    # Branch trajectory: branch token is at trunk_length-1 due to BPE merge
                    branch_token_pos = trunk_length - 1
                    conditional_logprobs["trunk"] = sum(
                        traj.logprobs[branch_token_pos:]
                    )

                # p(continuation | trunk + branch) for each non-trunk branch
                for b_idx, b_raw in enumerate(branches):
                    if b_raw == "trunk":
                        continue  # Already handled above
                    b_display = arm_display_name(b_idx)
                    if b_idx == branch_idx:
                        # This trajectory is in this branch
                        conditional_logprobs[b_display] = sum(
                            traj.logprobs[trunk_length:]
                        )
                    else:
                        # Not in this branch - use 0.0 as marker
                        conditional_logprobs[b_display] = 0.0

                # Continuation tokens = total - trunk
                n_continuation = len(traj.token_ids) - trunk_length

                trajectories.append(
                    TrajectoryData(
                        trajectory_idx=i,
                        branch=display_name,  # Use display name, not raw text
                        branch_idx=branch_idx,
                        prompt=arm_texts.get(display_name, trunk_text),
                        response=response,
                        response_after_branch=response_after_branch,
                        conditional_logprobs=conditional_logprobs,
                        n_continuation_tokens=n_continuation,
                    )
                )

            # Convert branches to display names for output
            branches = [arm_display_name(i) for i in range(len(branches))]

        result = cls(
            tree=tree,
            trajectories=trajectories,
            config=config,
            branches=branches,
            arm_texts=arm_texts,
            prefix_logprobs=prefix_logprobs
            if prefix_logprobs["branch_given_trunk"]
            else None,
            eos_token=data.get("eos_token"),
        )
        result.validate()
        return result

    def validate(self) -> None:
        """Validate that the loaded data is usable for judgment."""
        if not self.trajectories:
            raise ValueError("No trajectories found in generation output")
