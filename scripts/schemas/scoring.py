"""Schemas for trajectory judgment/scoring."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.common.base_schema import BaseSchema
from src.common.log import log
from src.common.token_tree import TokenTree

from . import default_config as defaults


@dataclass
class ScoringConfig(BaseSchema):
    """Configuration for trajectory scoring/judgment."""

    model: str
    categorical_judgements: list[str] = field(default_factory=list)
    similarity_scoring: list[str] = field(default_factory=list)

    # Judgment generation parameters
    max_tokens: int = defaults.JUDGE_MAX_TOKENS

    # Embedding parameters
    embedding_model: str = defaults.EMBEDDING_MODEL

    @classmethod
    def load(cls, path: str | Path) -> ScoringConfig:
        """Load config from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Scoring config not found: {path}")
        config = cls.from_json(path)
        config.validate()
        return config

    def validate(self) -> None:
        """Validate that required fields are present."""
        if not self.categorical_judgements and not self.similarity_scoring:
            raise ValueError(
                "No scoring methods specified: need categorical_judgements or similarity_scoring"
            )
        if self.categorical_judgements and not self.model:
            raise ValueError("No judge model specified for categorical judgements")

    def build_judgment_prompt(self, text: str, question: str) -> str:
        """Build prompt for categorical judgment."""
        return f"""Read the following text and answer the question with 0 (no) or 1 (yes).

TEXT:
{text}

QUESTION: {question}

Answer with just 0 or 1:"""

    @staticmethod
    def parse_judgment(response: str) -> int | None:
        """Parse a 0 or 1 judgment from model response."""
        # Remove thinking tags if present
        text = response
        if "</think>" in text:
            text = text.split("</think>")[-1]
        text = text.strip()

        # Check for just "0" or "1"
        if text in ("0", "1"):
            return int(text)

        # Check for patterns like "Answer: 0" or "Answer: 1"
        match = re.search(
            r"(?:answer|response|judgment|result)[:\s]*([01])", text, re.I
        )
        if match:
            return int(match.group(1))

        # Check for "yes" -> 1, "no" -> 0
        if re.search(r"\byes\b", text, re.I):
            return 1
        if re.search(r"\bno\b", text, re.I):
            return 0

        # Look for standalone 0 or 1 at end
        match = re.search(r"([01])\s*$", text)
        if match:
            return int(match.group(1))

        # Look for any 0 or 1
        match = re.search(r"\b([01])\b", text)
        if match:
            return int(match.group(1))

        return None


@dataclass
class TrajectoryData:
    """Data extracted from a trajectory for judgment."""

    trajectory_idx: int
    branch: str
    branch_idx: int  # Index of branch in config order (0=trunk, 1=branch_1, etc.)
    prompt: str  # The prompt/trunk text
    response: str  # The generated continuation
    conditional_logprobs: dict[str, float]  # Log prob conditioned on each group
    n_continuation_tokens: int = 0  # Number of tokens in continuation

    @property
    def full_text(self) -> str:
        """Full text (prompt + response) for judgment."""
        return self.prompt + self.response


@dataclass
class GroupDefinitions:
    """Defines the text for each conditioning group."""

    texts: dict[str, str]  # group_name -> text at that conditioning level
    token_lengths: dict[str, int]  # group_name -> token length

    @classmethod
    def from_tree(cls, tree: TokenTree, branches: list[str]) -> GroupDefinitions:
        """Build group definitions from a token tree."""
        trunk_text = tree.trunk_text or ""
        trunk_length = tree.trunk_length or 0

        texts = {"trunk": trunk_text}
        token_lengths = {"trunk": trunk_length}

        # For each branch, we need to find the text including the branch token
        # The branch token is at position trunk_length
        for branch in branches:
            # Find a trajectory in this branch to get the branch token
            for traj in tree.trajs:
                if traj.group_idx and len(branches) > traj.group_idx[0]:
                    if branches[traj.group_idx[0]] == branch:
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
    groups: dict[str, str]  # group_name -> conditioning text
    prefix_logprobs: dict[str, Any] | None = None  # Conditional logprobs for prefixes

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

        # Build group definitions
        groups: dict[str, str] = {}
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

            # Build group conditioning texts:
            # - "trunk": just trunk_text
            # - other branches: trunk_text + branch_name
            for branch in branches:
                if branch == "trunk":
                    groups[branch] = trunk_text
                else:
                    groups[branch] = trunk_text + branch

            # Extract trajectories with conditional logprobs
            for i, traj in enumerate(tree.trajs):
                response = traj.continuation_text or ""

                # Get branch name and index from group_idx
                if traj.group_idx and len(branches) > traj.group_idx[0]:
                    branch_idx = traj.group_idx[0]
                    branch = branches[branch_idx]
                else:
                    branch_idx = 0
                    branch = "trunk"

                # Extract prefix logprobs (once per branch)
                if trunk_length > prompt_length and prefix_logprobs["trunk_given_prompt"] == 0.0:
                    # p(trunk | prompt) - sum logprobs for trunk tokens only (not prompt)
                    prefix_logprobs["trunk_given_prompt"] = sum(traj.logprobs[prompt_length:trunk_length])

                if branch_idx > 0 and branch_idx not in prefix_logprobs["branch_given_trunk"]:
                    # p(branch | prompt + trunk) - logprob of branch token at trunk_length
                    if trunk_length < len(traj.logprobs):
                        prefix_logprobs["branch_given_trunk"][branch_idx] = traj.logprobs[trunk_length]

                # Compute conditional log probabilities for continuation
                conditional_logprobs: dict[str, float] = {}

                # For branch trajectories, BPE may merge trunk+branch differently.
                # Find actual position where continuation starts by comparing trajectory
                # length against expected prefix length.
                #
                # For trunk trajs: prefix = prompt + trunk = trunk_length tokens
                # For branch trajs: prefix = prompt + trunk + branch, but BPE may merge,
                #   so actual length may be less than trunk_length + len(branch_tokens)
                #
                # The continuation starts at position trunk_length for trunk trajs,
                # but for branch trajs it also starts at trunk_length because BPE
                # absorbed the trunk space into the branch token.

                # p(continuation | trunk) - sum from trunk_length onwards
                # For branch trajs, this excludes the branch token logprob (at trunk_length-1)
                # To include it, we need to start from trunk_length - 1 for branch trajs
                if branch_idx == 0:
                    # Trunk trajectory: continuation starts at trunk_length
                    conditional_logprobs["trunk"] = sum(traj.logprobs[trunk_length:])
                else:
                    # Branch trajectory: branch token is at trunk_length-1 due to BPE merge
                    # Include it in the trunk conditional
                    branch_token_pos = trunk_length - 1
                    conditional_logprobs["trunk"] = sum(traj.logprobs[branch_token_pos:])

                # p(continuation | trunk + branch) for each non-trunk branch
                for b in branches:
                    if b == "trunk":
                        continue  # Already handled above
                    if b == branch:
                        # This trajectory is in this branch
                        # Continuation starts at trunk_length (after branch token)
                        conditional_logprobs[b] = sum(traj.logprobs[trunk_length:])
                    else:
                        # Not in this branch - use 0.0 as marker
                        conditional_logprobs[b] = 0.0

                # Continuation tokens = total - trunk
                n_continuation = len(traj.token_ids) - trunk_length

                trajectories.append(
                    TrajectoryData(
                        trajectory_idx=i,
                        branch=branch,
                        branch_idx=branch_idx,
                        prompt=groups.get(branch, trunk_text),
                        response=response,
                        conditional_logprobs=conditional_logprobs,
                        n_continuation_tokens=n_continuation,
                    )
                )

        result = cls(
            tree=tree,
            trajectories=trajectories,
            config=config,
            branches=branches,
            groups=groups,
            prefix_logprobs=prefix_logprobs if prefix_logprobs["branch_given_trunk"] else None,
        )
        result.validate()
        return result

    def validate(self) -> None:
        """Validate that the loaded data is usable for judgment."""
        if not self.trajectories:
            raise ValueError("No trajectories found in generation output")


@dataclass
class JudgmentResult(BaseSchema):
    """Result of scoring a single trajectory."""

    trajectory_idx: int
    branch: str
    branch_idx: int  # Index of branch in config order (0=trunk, 1=branch_1, etc.)
    text: str  # Full text (prompt + response) that was scored
    conditional_logprobs: dict[str, float]  # Log prob conditioned on each group
    n_continuation_tokens: int  # Number of tokens in continuation
    scores: list[int | None]  # Categorical judgment scores (0/1)
    raw_judgments: list[str]  # Raw LLM responses for categorical judgments
    similarity_scores: list[float] = field(
        default_factory=list
    )  # Similarity scores (0-1)

    @classmethod
    def from_trajectory(
        cls,
        traj: TrajectoryData,
        scores: list[int | None],
        raw_judgments: list[str],
        similarity_scores: list[float] | None = None,
    ) -> JudgmentResult:
        """Create a JudgmentResult from a TrajectoryData and scores."""
        return cls(
            trajectory_idx=traj.trajectory_idx,
            branch=traj.branch,
            branch_idx=traj.branch_idx,
            text=traj.full_text,
            conditional_logprobs=traj.conditional_logprobs,
            n_continuation_tokens=traj.n_continuation_tokens,
            scores=scores,
            raw_judgments=raw_judgments,
            similarity_scores=similarity_scores or [],
        )


@dataclass
class JudgmentOutput(BaseSchema):
    """Output from trajectory scoring."""

    generation_file: str
    scoring_file: str
    judge_model: str
    categorical_judgements: list[str]
    similarity_scoring: list[str]
    embedding_model: str
    branches: list[str]  # Branch names in config order
    groups: dict[str, str]  # group_name -> conditioning text
    scored_at: str
    num_results: int
    results: list[dict[str, Any]]  # List of JudgmentResult.to_dict()
    prefix_logprobs: dict[str, Any] | None = None  # Conditional logprobs for prefixes

    @classmethod
    def create(
        cls,
        generation_file: str,
        scoring_file: str,
        scoring_config: ScoringConfig,
        results: list[JudgmentResult],
        branches: list[str],
        groups: dict[str, str],
        prefix_logprobs: dict[str, Any] | None = None,
    ) -> JudgmentOutput:
        """Create scoring output from results."""
        return cls(
            generation_file=generation_file,
            scoring_file=scoring_file,
            judge_model=scoring_config.model,
            categorical_judgements=scoring_config.categorical_judgements,
            similarity_scoring=scoring_config.similarity_scoring,
            embedding_model=scoring_config.embedding_model,
            branches=branches,
            groups=groups,
            scored_at=datetime.now().isoformat(),
            num_results=len(results),
            results=[r.to_dict() for r in results],
            prefix_logprobs=prefix_logprobs,
        )

    def save(self, path: str | Path) -> Path:
        """Save output to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
        return path

    @staticmethod
    def compute_output_path(gen_path: str | Path, scoring_path: str | Path) -> Path:
        """Compute the output path for judgment results."""
        gen_path = Path(gen_path)
        scoring_path = Path(scoring_path)
        out_dir = Path("out")
        gen_name = gen_path.stem.replace("gen_", "")
        scoring_name = scoring_path.stem
        return out_dir / f"score_{gen_name}_{scoring_name}.json"

    def summarize(self) -> None:
        """Print summary statistics for all scoring methods."""
        log("\nSummary:")

        # Categorical judgments
        if self.categorical_judgements:
            log("  Categorical judgments:")
            for i, question in enumerate(self.categorical_judgements):
                values = [
                    r.get("scores", [])[i] if i < len(r.get("scores", [])) else None
                    for r in self.results
                ]
                valid = [v for v in values if v is not None]
                if valid:
                    avg = sum(valid) / len(valid)
                    log(
                        f"    {question[:50]}: {avg:.2%} ({len(valid)}/{len(values)} valid)"
                    )

        # Similarity scores
        if self.similarity_scoring:
            log("  Similarity scores:")
            for i, ref in enumerate(self.similarity_scoring):
                values = [
                    r.get("similarity_scores", [])[i]
                    if i < len(r.get("similarity_scores", []))
                    else None
                    for r in self.results
                ]
                valid = [v for v in values if v is not None]
                if valid:
                    avg = sum(valid) / len(valid)
                    log(f"    {ref[:50]}: {avg:.3f} ({len(valid)}/{len(values)} valid)")
