"""Schemas for trajectory generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from src.common.base_schema import BaseSchema
from src.common.log import log
from src.common.token_tree import TokenTree

from . import default_config as defaults

if TYPE_CHECKING:
    from src.inference.generated_trajectory import GeneratedTrajectory


# ══════════════════════════════════════════════════════════════════════════════
# Parameter Dataclasses
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ParamsSchema(BaseSchema):
    """Base class for parameter schemas with CLI-style printing."""

    # Subclasses define: field_name -> "--cli-arg-name"
    _cli_args: ClassVar[dict[str, str]] = {}

    def print(self) -> None:
        """Print parameters as CLI arguments."""
        log("  Parameters:")
        for field_name, cli_arg in self._cli_args.items():
            value = getattr(self, field_name)
            if value is not None:
                log(f"    {cli_arg} {value}")


@dataclass
class ForkingParams(ParamsSchema):
    """Parameters for forking paths generation."""

    max_alternates: int
    min_prob: float
    min_entropy: float
    samples_per_fork: int

    _cli_args: ClassVar[dict[str, str]] = {
        "max_alternates": "--max-alternates-per-position",
        "min_prob": "--min-prob-for-alternate",
        "min_entropy": "--min-entropy-to-fork",
        "samples_per_fork": "--samples-per-fork",
    }


@dataclass
class EntropySeekingParams(ParamsSchema):
    """Parameters for entropy-seeking generation."""

    samples_per_expansion: int
    num_expansion_rounds: int

    _cli_args: ClassVar[dict[str, str]] = {
        "samples_per_expansion": "--samples-per-expansion",
        "num_expansion_rounds": "--num-expansion-rounds",
    }


@dataclass
class SamplingParams(ParamsSchema):
    """Parameters for simple sampling generation."""

    samples_per_branch: int

    _cli_args: ClassVar[dict[str, str]] = {
        "samples_per_branch": "--samples-per-branch",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Result Dataclasses
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class BranchGenerationResult:
    """Result from generating trajectories across all branches."""

    trajectories: list[GeneratedTrajectory]
    group_indices: list[int]  # group_idx for each trajectory
    trunk_length: int
    prompt_length: int  # Length of just the prompt (no trunk) in tokens


@dataclass
class ForkArm:
    """A pair of branch indices for comparison."""

    left: int
    right: int


@dataclass
class OutputPaths:
    """Computed output paths for the full experiment pipeline."""

    generation: Path
    judgment: Path
    estimation: Path


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Branch:
    """A branch configuration for trajectory generation."""

    prefill: str  # Full prefill text (skip_prefix + trunk + branch)
    name: str  # Name of this branch
    group_idx: int  # Group index for this branch


@dataclass
class GenerationConfig(BaseSchema):
    """Configuration for trajectory generation."""

    prompt: str
    model: str = ""
    trunk: str = ""
    branches: list[str] = field(default_factory=list)

    # General generation params
    temperature: float = defaults.TEMPERATURE
    max_new_tokens: int = defaults.MAX_NEW_TOKENS
    seed: int | None = None

    # Simple sampling parameters
    sampling_samples_per_branch: int = defaults.SAMPLING_SAMPLES_PER_BRANCH

    # Forking paths parameters
    forking_max_alternates: int = defaults.FORKING_MAX_ALTERNATES
    forking_min_prob: float = defaults.FORKING_MIN_PROB
    forking_min_entropy: float = defaults.FORKING_MIN_ENTROPY
    forking_samples_per_fork: int = defaults.FORKING_SAMPLES_PER_FORK

    # Entropy seeking parameters
    entropy_samples_per_expansion: int = defaults.ENTROPY_SAMPLES_PER_EXPANSION
    entropy_num_expansion_rounds: int = defaults.ENTROPY_NUM_EXPANSION_ROUNDS

    @classmethod
    def load(cls, path: str | Path) -> GenerationConfig:
        """Load config from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        return cls.from_json(path)

    def get_branches(self, skip_prefix: str = "") -> list[Branch]:
        """Get branch configurations for generation.

        Args:
            skip_prefix: Prefix to prepend (e.g., reasoning skip tokens)

        Returns:
            List of Branch objects: trunk first, then each branch
        """
        # Always include trunk as first branch
        result = [Branch(prefill=skip_prefix + self.trunk, name="trunk", group_idx=0)]

        # Add explicit branches if defined
        if self.branches:
            result.extend(
                Branch(
                    prefill=skip_prefix + self.trunk + branch,
                    name=branch,
                    group_idx=i + 1,  # +1 because trunk is 0
                )
                for i, branch in enumerate(self.branches)
            )

        return result

    @property
    def fork_arms(self) -> list[ForkArm]:
        """Get all pairwise fork arms between branches."""
        if len(self.branches) < 2:
            return []
        arms = []
        for i in range(len(self.branches)):
            for j in range(i + 1, len(self.branches)):
                arms.append(ForkArm(left=i, right=j))
        return arms

    def compute_prompt_length(self, runner) -> int:
        """Compute the shared prefix length between prompt-only and prompt+trunk.

        Due to BPE tokenization, adding trunk text may change how the prompt is
        tokenized. This finds the last position where both tokenizations agree,
        which is where trunk-specific logprobs start.
        """
        skip_prefix = runner.skip_thinking_prefix
        prompt_only = runner.apply_chat_template(self.prompt) + skip_prefix
        prompt_trunk = prompt_only + self.trunk

        tokens_prompt = runner.encode_ids(prompt_only, add_special_tokens=True)
        tokens_trunk = runner.encode_ids(prompt_trunk, add_special_tokens=True)

        # Find the divergence point
        shared_length = 0
        for i, (t1, t2) in enumerate(zip(tokens_prompt, tokens_trunk)):
            if t1 != t2:
                break
            shared_length = i + 1

        return shared_length

    def compute_trunk_length(self, runner) -> int:
        """Compute the length of the trunk (prompt + trunk, no branch) in tokens."""
        skip_prefix = runner.skip_thinking_prefix
        formatted = runner.apply_chat_template(self.prompt) + skip_prefix + self.trunk
        return len(runner.encode_ids(formatted, add_special_tokens=True))

    def apply_cli_overrides(self, overrides: dict[str, Any]) -> None:
        """Apply CLI argument overrides to config fields.

        Maps CLI arg names (with underscores) to config field names.
        """
        mapping = {
            # Sampling
            "samples_per_branch": "sampling_samples_per_branch",
            # Forking
            "max_alternates_per_position": "forking_max_alternates",
            "min_prob_for_alternate": "forking_min_prob",
            "min_entropy_to_fork": "forking_min_entropy",
            "samples_per_fork": "forking_samples_per_fork",
            # Entropy seeking
            "samples_per_expansion": "entropy_samples_per_expansion",
            "num_expansion_rounds": "entropy_num_expansion_rounds",
        }
        for cli_name, value in overrides.items():
            if value is not None and cli_name in mapping:
                setattr(self, mapping[cli_name], value)

    @property
    def sampling_params(self) -> SamplingParams:
        """Get simple sampling parameters from config."""
        return SamplingParams(samples_per_branch=self.sampling_samples_per_branch)

    @property
    def forking_params(self) -> ForkingParams:
        """Get forking paths parameters from config."""
        return ForkingParams(
            max_alternates=self.forking_max_alternates,
            min_prob=self.forking_min_prob,
            min_entropy=self.forking_min_entropy,
            samples_per_fork=self.forking_samples_per_fork,
        )

    @property
    def entropy_params(self) -> EntropySeekingParams:
        """Get entropy seeking parameters from config."""
        return EntropySeekingParams(
            samples_per_expansion=self.entropy_samples_per_expansion,
            num_expansion_rounds=self.entropy_num_expansion_rounds,
        )



@dataclass
class GenerationOutput(BaseSchema):
    """Output from trajectory generation, including tree structure."""

    config: dict[str, Any]  # GenerationConfig.to_dict()
    model: str
    method: str  # Generation method: simple-sampling, forking-paths, seeking-entropy
    generated_at: str
    num_trajectories: int
    tree: dict[str, Any] | None = None  # TokenTree.to_dict() output

    @classmethod
    def from_tree(
        cls,
        config: GenerationConfig,
        model: str,
        tree: TokenTree,
        method: str = "simple-sampling",
    ) -> GenerationOutput:
        """Create output from a TokenTree.

        Args:
            config: Generation configuration
            model: Model name used
            tree: TokenTree with trajectories
            method: Generation method name
        """
        # Pop heavy data before serializing (full_logits tensors)
        tree.pop_heavy()

        # Build config dict with trunk included in branches list
        # This ensures group_idx 0 = "trunk", 1 = first branch, etc.
        config_dict = config.to_dict()
        config_dict["branches"] = ["trunk"] + config_dict.get("branches", [])

        return cls(
            config=config_dict,
            model=model,
            method=method,
            generated_at=datetime.now().isoformat(),
            num_trajectories=len(tree.trajs),
            tree=tree.to_dict(max_list_length=10000),
        )

    @staticmethod
    def compute_output_path(config_path: Path, method: str = "sampling") -> Path:
        """Compute the output path for generation results.

        Args:
            config_path: Path to the generation config file
            method: Generation method keyword (sampling, forking, entropy)

        Returns:
            Path like out/gen_<method>_<config_stem>.json
        """
        return Path("out") / f"gen_{method}_{config_path.stem}.json"

    def save(self, path: str | Path) -> Path:
        """Save output to JSON file."""
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
        return path
