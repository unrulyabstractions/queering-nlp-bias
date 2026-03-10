"""Schemas for trajectory judgment/scoring."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from src.common.base_schema import BaseSchema
from src.common.log import log
from src.common.token_tree import TokenTree

from . import default_config as defaults
from .log_utils import log_banner, log_sub_banner


class StringSelection(Enum):
    """Determines which portion of the trajectory text to use for scoring.

    Options:
        WholeTrajectory: Full text including prompt and response
        WholeContinuation: Just the generated response/continuation (default)
        AfterTrunk: Text after the trunk tokens (continuation minus trunk)
        AfterBranch: Text after the branch point (continuation minus trunk and branch)
    """

    WholeTrajectory = "WholeTrajectory"
    WholeContinuation = "WholeContinuation"
    AfterTrunk = "AfterTrunk"
    AfterBranch = "AfterBranch"


# Type alias for categorical judgements: can be a string or a list of strings (bundled structure)
CategoricalJudgement = str | list[str]


@dataclass
class ScoringConfig(BaseSchema):
    """Configuration for trajectory scoring/judgment.

    categorical_judgements can contain:
    - Individual questions (strings): Each becomes its own structure
    - Bundled questions (list of strings): All questions in the bundle are
      averaged together to form a single structure value in the core

    graded_judgements work the same way but ask for a score between 0 and 1
    instead of a binary 0/1 answer.
    """

    model: str
    categorical_judgements: list[CategoricalJudgement] = field(default_factory=list)
    graded_judgements: list[CategoricalJudgement] = field(default_factory=list)
    similarity_scoring: list[str | list[str]] = field(default_factory=list)

    # Text selection for scoring
    string_selection: StringSelection = StringSelection(defaults.STRING_SELECTION)

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
        if not self.categorical_judgements and not self.graded_judgements and not self.similarity_scoring:
            raise ValueError(
                "No scoring methods specified: need categorical_judgements, graded_judgements, or similarity_scoring"
            )
        if (self.categorical_judgements or self.graded_judgements) and not self.model:
            raise ValueError("No judge model specified for judgements")

    def iter_all_questions(self) -> list[tuple[int, int | None, str]]:
        """Iterate over all individual questions with structure and sub-indices.

        Returns:
            List of (structure_idx, sub_idx, question) tuples.
            sub_idx is None for single questions, or 0,1,2... for bundled questions.
        """
        result = []
        for struct_idx, item in enumerate(self.categorical_judgements):
            if isinstance(item, list):
                for sub_idx, question in enumerate(item):
                    result.append((struct_idx, sub_idx, question))
            else:
                result.append((struct_idx, None, item))
        return result

    def get_structure_labels(self) -> list[str]:
        """Get labels for each structure (c for categorical, g for graded, s for similarity).

        For bundled questions, returns a single label for the bundle.
        """
        labels = []
        for i in range(len(self.categorical_judgements)):
            labels.append(f"c{i+1}")
        for i in range(len(self.graded_judgements)):
            labels.append(f"g{i+1}")
        for i in range(len(self.similarity_scoring)):
            labels.append(f"s{i+1}")
        return labels

    def get_structure_descriptions(self) -> list[str]:
        """Get human-readable descriptions for each structure.

        For bundled questions, joins them with ' + '.
        """
        descriptions = []
        for item in self.categorical_judgements:
            if isinstance(item, list):
                descriptions.append(" + ".join(item))
            else:
                descriptions.append(item)
        for item in self.graded_judgements:
            if isinstance(item, list):
                descriptions.append(" + ".join(item))
            else:
                descriptions.append(item)
        for item in self.similarity_scoring:
            if isinstance(item, list):
                descriptions.append(" + ".join(item))
            else:
                descriptions.append(item)
        return descriptions

    def num_structures(self) -> int:
        """Return number of structures (bundled questions count as one)."""
        return len(self.categorical_judgements) + len(self.graded_judgements) + len(self.similarity_scoring)

    def is_bundled(self, struct_idx: int) -> bool:
        """Check if a categorical structure is a bundled question set."""
        if struct_idx >= len(self.categorical_judgements):
            return False
        return isinstance(self.categorical_judgements[struct_idx], list)

    def get_bundled_questions(self, struct_idx: int) -> list[str]:
        """Get all questions in a structure (single or bundled)."""
        if struct_idx >= len(self.categorical_judgements):
            return []
        item = self.categorical_judgements[struct_idx]
        if isinstance(item, list):
            return item
        return [item]

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

    def build_graded_prompt(self, text: str, question: str) -> str:
        """Build prompt for graded judgment (0-1 scale)."""
        return f"""Read the following text and answer the question with a score between 0.0 and 1.0.
0.0 means completely no/false, 1.0 means completely yes/true, values in between indicate partial agreement.

TEXT:
{text}

QUESTION: {question}

Answer with just a number between 0.0 and 1.0:"""

    @staticmethod
    def parse_graded_judgment(response: str) -> float | None:
        """Parse a 0-1 graded judgment from model response."""
        # Remove thinking tags if present
        text = response
        if "</think>" in text:
            text = text.split("</think>")[-1]
        text = text.strip()

        # Look for decimal numbers between 0 and 1
        match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?|\.\d+)\b", text)
        if match:
            try:
                value = float(match.group(1))
                if 0.0 <= value <= 1.0:
                    return value
            except ValueError:
                pass

        # Check for just "0" or "1"
        if text in ("0", "1"):
            return float(text)

        return None


def branch_display_name(branch_idx: int) -> str:
    """Convert branch index to display name.

    0 -> "trunk"
    1 -> "branch_1"
    2 -> "branch_2"
    etc.
    """
    return "trunk" if branch_idx == 0 else f"branch_{branch_idx}"


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
    def from_tree(cls, tree: TokenTree, branches: list[str]) -> "ArmDefinitions":
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
    arm_texts: dict[str, str]  # arm_name -> conditioning text
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
                display_name = branch_display_name(raw_idx)
                if raw_branch == "trunk":
                    arm_texts[display_name] = trunk_text
                else:
                    arm_texts[display_name] = trunk_text + raw_branch

            # Extract trajectories with conditional logprobs
            for i, traj in enumerate(tree.trajs):
                continuation_text = traj.continuation_text or ""

                # Get branch index from group_idx
                if traj.group_idx and len(branches) > traj.group_idx[0]:
                    branch_idx = traj.group_idx[0]
                    raw_branch = branches[branch_idx]
                else:
                    branch_idx = 0
                    raw_branch = "trunk"

                # Use display name for output
                display_name = branch_display_name(branch_idx)

                # response = continuation after trunk (includes branch token for branch trajs)
                response = continuation_text

                # response_after_branch = continuation after branch token (branch stripped)
                # Use raw_branch for stripping since it's the actual text
                if branch_idx > 0 and continuation_text.startswith(raw_branch):
                    response_after_branch = continuation_text[len(raw_branch):]
                else:
                    response_after_branch = continuation_text

                # Extract prefix logprobs (once per branch)
                if trunk_length > prompt_length and prefix_logprobs["trunk_given_prompt"] == 0.0:
                    # p(trunk | prompt) - sum logprobs for trunk tokens only (not prompt)
                    prefix_logprobs["trunk_given_prompt"] = sum(traj.logprobs[prompt_length:trunk_length])

                if branch_idx > 0 and branch_idx not in prefix_logprobs["branch_given_trunk"]:
                    # p(branch | prompt + trunk) - logprob of branch token at trunk_length
                    if trunk_length < len(traj.logprobs):
                        prefix_logprobs["branch_given_trunk"][branch_idx] = traj.logprobs[trunk_length]

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
                    conditional_logprobs["trunk"] = sum(traj.logprobs[branch_token_pos:])

                # p(continuation | trunk + branch) for each non-trunk branch
                for b_idx, b_raw in enumerate(branches):
                    if b_raw == "trunk":
                        continue  # Already handled above
                    b_display = branch_display_name(b_idx)
                    if b_idx == branch_idx:
                        # This trajectory is in this branch
                        conditional_logprobs[b_display] = sum(traj.logprobs[trunk_length:])
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
            branches = [branch_display_name(i) for i in range(len(branches))]

        result = cls(
            tree=tree,
            trajectories=trajectories,
            config=config,
            branches=branches,
            arm_texts=arm_texts,
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
    conditional_logprobs: dict[str, float]  # Log prob conditioned on each arm
    n_continuation_tokens: int  # Number of tokens in continuation
    scores: list[int | None]  # Categorical judgment scores (0/1)
    raw_judgments: list[str]  # Raw LLM responses for categorical judgments
    similarity_scores: list[float] = field(
        default_factory=list
    )  # Similarity scores (0-1)
    graded_scores: list[float | None] = field(
        default_factory=list
    )  # Graded judgment scores (0-1 continuous)
    graded_raw_judgments: list[str] = field(
        default_factory=list
    )  # Raw LLM responses for graded judgments

    @classmethod
    def from_trajectory(
        cls,
        traj: TrajectoryData,
        scores: list[int | None],
        raw_judgments: list[str],
        similarity_scores: list[float] | None = None,
        graded_scores: list[float | None] | None = None,
        graded_raw_judgments: list[str] | None = None,
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
            graded_scores=graded_scores or [],
            graded_raw_judgments=graded_raw_judgments or [],
        )


@dataclass
class JudgmentOutput(BaseSchema):
    """Output from trajectory scoring."""

    generation_file: str
    scoring_file: str
    judge_model: str
    categorical_judgements: list[CategoricalJudgement]  # str | list[str]
    graded_judgements: list[CategoricalJudgement] = field(default_factory=list)  # str | list[str]
    similarity_scoring: list[str | list[str]] = field(default_factory=list)
    embedding_model: str = ""
    branches: list[str] = field(default_factory=list)  # Branch names in config order
    arm_texts: dict[str, str] = field(default_factory=dict)  # arm_name -> conditioning text
    scored_at: str = ""
    num_results: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)  # List of JudgmentResult.to_dict()
    prefix_logprobs: dict[str, Any] | None = None  # Conditional logprobs for prefixes

    @classmethod
    def create(
        cls,
        generation_file: str,
        scoring_file: str,
        scoring_config: ScoringConfig,
        results: list[JudgmentResult],
        branches: list[str],
        arm_texts: dict[str, str],
        prefix_logprobs: dict[str, Any] | None = None,
    ) -> JudgmentOutput:
        """Create scoring output from results."""
        return cls(
            generation_file=generation_file,
            scoring_file=scoring_file,
            judge_model=scoring_config.model,
            categorical_judgements=scoring_config.categorical_judgements,
            graded_judgements=scoring_config.graded_judgements,
            similarity_scoring=scoring_config.similarity_scoring,
            embedding_model=scoring_config.embedding_model,
            branches=branches,
            arm_texts=arm_texts,
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

    def save_summary(self, path: str | Path) -> Path:
        """Save human-readable summary to text file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        lines.append("=" * 76)
        lines.append("  SCORING SUMMARY")
        lines.append("=" * 76)
        lines.append("")

        # Settings
        lines.append(f"  Judge:       {self.judge_model}")
        lines.append(f"  Embed:       {self.embedding_model}")
        lines.append(f"  Scored:      {self.scored_at}")
        lines.append(f"  Trajectories:{self.num_results}")
        lines.append("")

        # Group results by branch
        by_branch: dict[str, list[dict]] = {}
        for r in self.results:
            branch = r.get("branch", "trunk")
            by_branch.setdefault(branch, []).append(r)

        # Build structure labels
        labels = []
        for i, item in enumerate(self.categorical_judgements):
            labels.append(f"c{i+1}")
        for i, item in enumerate(self.graded_judgements):
            labels.append(f"g{i+1}")
        for i, item in enumerate(self.similarity_scoring):
            labels.append(f"s{i+1}")

        # Per-branch rates table
        lines.append("-" * 76)
        lines.append("  RATES BY BRANCH")
        lines.append("-" * 76)
        col_w = 8
        header = "".join(f"{l:^{col_w}}" for l in labels)
        lines.append(f"  {'Branch':<14} {'N':>4}  {header}")
        lines.append("  " + "-" * 70)

        for branch_name in self.branches:
            branch_results = by_branch.get(branch_name, [])
            if not branch_results:
                continue
            rates = []
            # Categorical rates
            for i, item in enumerate(self.categorical_judgements):
                flat_idx = self._get_flat_score_index(i, 0)
                if isinstance(item, list):
                    values = []
                    for q_idx in range(len(item)):
                        idx = self._get_flat_score_index(i, q_idx)
                        for r in branch_results:
                            scores = r.get("scores", [])
                            if idx < len(scores) and scores[idx] is not None:
                                values.append(scores[idx])
                    rates.append(sum(values) / len(values) if values else 0.0)
                else:
                    values = [r.get("scores", [])[flat_idx] for r in branch_results
                              if flat_idx < len(r.get("scores", [])) and r.get("scores", [])[flat_idx] is not None]
                    rates.append(sum(values) / len(values) if values else 0.0)
            # Graded rates
            for i, item in enumerate(self.graded_judgements):
                flat_idx = self._get_flat_graded_index(i, 0)
                if isinstance(item, list):
                    values = []
                    for q_idx in range(len(item)):
                        idx = self._get_flat_graded_index(i, q_idx)
                        for r in branch_results:
                            scores = r.get("graded_scores", [])
                            if idx < len(scores) and scores[idx] is not None:
                                values.append(scores[idx])
                    rates.append(sum(values) / len(values) if values else 0.0)
                else:
                    values = [r.get("graded_scores", [])[flat_idx] for r in branch_results
                              if flat_idx < len(r.get("graded_scores", [])) and r.get("graded_scores", [])[flat_idx] is not None]
                    rates.append(sum(values) / len(values) if values else 0.0)
            # Similarity rates
            for i, item in enumerate(self.similarity_scoring):
                values = [r.get("similarity_scores", [])[i] for r in branch_results
                          if i < len(r.get("similarity_scores", []))]
                rates.append(sum(values) / len(values) if values else 0.0)

            rate_str = "".join(f"{r:^{col_w}.3f}" for r in rates)
            lines.append(f"  {branch_name:<14} {len(branch_results):>4}  {rate_str}")

        lines.append("")
        lines.append("=" * 76)

        with open(path, "w") as f:
            f.write("\n".join(lines))
        return path

    @staticmethod
    def compute_summary_path(gen_path: str | Path, scoring_path: str | Path) -> Path:
        """Compute the output path for scoring summary."""
        gen_path = Path(gen_path)
        scoring_path = Path(scoring_path)
        gen_name = gen_path.stem.replace("gen_", "")
        scoring_name = scoring_path.stem
        return Path("out") / f"summary_score_{gen_name}_{scoring_name}.txt"

    @staticmethod
    def compute_output_path(gen_path: str | Path, scoring_path: str | Path) -> Path:
        """Compute the output path for judgment results."""
        gen_path = Path(gen_path)
        scoring_path = Path(scoring_path)
        out_dir = Path("out")
        gen_name = gen_path.stem.replace("gen_", "")
        scoring_name = scoring_path.stem
        return out_dir / f"score_{gen_name}_{scoring_name}.json"

    def _get_flat_score_index(self, struct_idx: int, question_idx: int = 0) -> int:
        """Get the flat index into raw scores for a categorical structure/question."""
        flat_idx = 0
        for i, item in enumerate(self.categorical_judgements):
            if i == struct_idx:
                return flat_idx + question_idx
            if isinstance(item, list):
                flat_idx += len(item)
            else:
                flat_idx += 1
        return flat_idx

    def _get_flat_graded_index(self, struct_idx: int, question_idx: int = 0) -> int:
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

    def summarize(self) -> None:
        """Print clean summary statistics for all scoring methods."""
        from src.common.viz_utils import preview

        log_banner("SCORING SUMMARY")

        # Settings
        log("\nSettings:")
        log(f"  Judge model: {self.judge_model}")
        log(f"  Generation file: {self.generation_file}")
        log(f"  Trajectories scored: {self.num_results}")

        # Categorical judgments - show per-arm breakdown
        if self.categorical_judgements:
            log_sub_banner("CATEGORICAL JUDGMENTS (% answering YES) - BY ARM")

            # Group results by branch
            by_branch: dict[str, list[dict]] = {}
            for r in self.results:
                branch = r.get("branch", "trunk")
                by_branch.setdefault(branch, []).append(r)

            for branch_name in self.branches:
                branch_results = by_branch.get(branch_name, [])
                if not branch_results:
                    continue

                # Branch names are already display names (trunk, branch_1, etc.)
                log(f"\n  === {branch_name.upper()} ({len(branch_results)} trajectories) ===")

                for struct_idx, item in enumerate(self.categorical_judgements):
                    if isinstance(item, list):
                        # Bundled structure
                        log(f"\n    [c{struct_idx+1}] BUNDLED ({len(item)} questions):")
                        all_values = []
                        for q_idx, question in enumerate(item):
                            flat_idx = self._get_flat_score_index(struct_idx, q_idx)
                            values = [
                                r.get("scores", [])[flat_idx]
                                if flat_idx < len(r.get("scores", []))
                                else None
                                for r in branch_results
                            ]
                            valid = [v for v in values if v is not None]
                            all_values.extend(valid)
                            if valid:
                                avg = sum(valid) / len(valid)
                                log(f"        • {preview(question, 40)}: {avg*100:5.1f}%")
                        # Show aggregate
                        if all_values:
                            agg = sum(all_values) / len(all_values)
                            log(f"        → AGGREGATE: {agg*100:5.1f}%")
                    else:
                        # Single question
                        flat_idx = self._get_flat_score_index(struct_idx, 0)
                        values = [
                            r.get("scores", [])[flat_idx]
                            if flat_idx < len(r.get("scores", []))
                            else None
                            for r in branch_results
                        ]
                        valid = [v for v in values if v is not None]
                        if valid:
                            avg = sum(valid) / len(valid)
                            log(f"\n    [c{struct_idx+1}] {preview(item, 45)}: {avg*100:5.1f}%")

        # Graded judgments - show per-arm breakdown
        if self.graded_judgements:
            log_sub_banner("GRADED JUDGMENTS (avg score 0-1) - BY ARM")

            # Group results by branch
            by_branch: dict[str, list[dict]] = {}
            for r in self.results:
                branch = r.get("branch", "trunk")
                by_branch.setdefault(branch, []).append(r)

            for branch_name in self.branches:
                branch_results = by_branch.get(branch_name, [])
                if not branch_results:
                    continue

                # Branch names are already display names (trunk, branch_1, etc.)
                log(f"\n  === {branch_name.upper()} ({len(branch_results)} trajectories) ===")

                for struct_idx, item in enumerate(self.graded_judgements):
                    if isinstance(item, list):
                        # Bundled structure
                        log(f"\n    [g{struct_idx+1}] BUNDLED ({len(item)} questions):")
                        all_values = []
                        for q_idx, question in enumerate(item):
                            flat_idx = self._get_flat_graded_index(struct_idx, q_idx)
                            values = [
                                r.get("graded_scores", [])[flat_idx]
                                if flat_idx < len(r.get("graded_scores", []))
                                else None
                                for r in branch_results
                            ]
                            valid = [v for v in values if v is not None]
                            all_values.extend(valid)
                            if valid:
                                avg = sum(valid) / len(valid)
                                log(f"        • {preview(question, 40)}: {avg:.3f}")
                        # Show aggregate
                        if all_values:
                            agg = sum(all_values) / len(all_values)
                            log(f"        → AGGREGATE: {agg:.3f}")
                    else:
                        # Single question
                        flat_idx = self._get_flat_graded_index(struct_idx, 0)
                        values = [
                            r.get("graded_scores", [])[flat_idx]
                            if flat_idx < len(r.get("graded_scores", []))
                            else None
                            for r in branch_results
                        ]
                        valid = [v for v in values if v is not None]
                        if valid:
                            avg = sum(valid) / len(valid)
                            log(f"\n    [g{struct_idx+1}] {preview(item, 45)}: {avg:.3f}")

        # Similarity scores
        if self.similarity_scoring:
            log_sub_banner("SIMILARITY SCORES (0-1 scale)")
            flat_idx = 0
            for struct_idx, item in enumerate(self.similarity_scoring):
                if isinstance(item, list):
                    # Bundled structure
                    log(f"\n  [s{struct_idx+1}] Bundled ({len(item)} references):")
                    group_values = []
                    for ref in item:
                        values = [
                            r.get("similarity_scores", [])[flat_idx]
                            if flat_idx < len(r.get("similarity_scores", []))
                            else None
                            for r in self.results
                        ]
                        valid = [v for v in values if v is not None]
                        if valid:
                            avg = sum(valid) / len(valid)
                            group_values.append(avg)
                            log(f"      • {preview(ref, 40)}: avg={avg:.3f}")
                        flat_idx += 1
                    if group_values:
                        group_avg = sum(group_values) / len(group_values)
                        log(f"      → group avg={group_avg:.3f}")
                else:
                    # Single reference
                    values = [
                        r.get("similarity_scores", [])[flat_idx]
                        if flat_idx < len(r.get("similarity_scores", []))
                        else None
                        for r in self.results
                    ]
                    valid = [v for v in values if v is not None]
                    if valid:
                        avg = sum(valid) / len(valid)
                        log(f"\n  [s{struct_idx+1}] {preview(item, 50)}")
                        log(f"      → avg={avg:.3f} ({len(valid)}/{len(values)} valid)")
                    flat_idx += 1

        log_banner("")
