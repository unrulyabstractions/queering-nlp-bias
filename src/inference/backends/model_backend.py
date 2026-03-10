"""Abstract base class for model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from typing import Any

import torch


class ModelBackend(Enum):
    """Available model backends."""

    MLX = "mlx"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Backend(ABC):
    """Abstract base class for model backends.

    All backends must implement these methods to provide a consistent interface
    for model inference.
    """

    supports_inference_mode: bool = (
        True  # Override to False if backend conflicts with inference_mode
    )

    def __init__(self, runner: Any):
        """Initialize backend with a reference to the ModelRunner.

        Args:
            runner: ModelRunner instance that owns this backend
        """
        self.runner = runner

    @abstractmethod
    def get_tokenizer(self):
        """Get the tokenizer for this backend."""
        ...

    @abstractmethod
    def get_n_layers(self) -> int:
        """Get the number of layers in the model."""
        ...

    @abstractmethod
    def get_d_model(self) -> int:
        """Get the hidden dimension of the model."""
        ...

    @abstractmethod
    def encode(
        self, text: str, add_special_tokens: bool = True, prepend_bos: bool = False
    ) -> torch.Tensor:
        """Encode text into token IDs tensor."""
        ...

    @abstractmethod
    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to text."""
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        past_kv_cache: Any = None,
    ) -> str:
        """Generate text from a prompt."""
        ...

    @abstractmethod
    def get_next_token_probs(
        self, prompt: str, target_tokens: Sequence[str], past_kv_cache: Any = None
    ) -> dict[str, float]:
        """Get next token probabilities for target tokens."""
        ...

    @abstractmethod
    def get_next_token_probs_by_id(
        self, prompt: str, token_ids: Sequence[int], past_kv_cache: Any = None
    ) -> dict[int, float]:
        """Get next token probabilities by token ID."""
        ...

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run forward pass and return logits.

        Args:
            input_ids: Token IDs tensor of shape [batch, seq_len]

        Returns:
            Logits tensor of shape [batch, seq_len, vocab_size]
        """
        ...

    @abstractmethod
    def generate_trajectory(
        self,
        token_ids: list[int],
        max_new_tokens: int,
        temperature: float,
    ) -> tuple[list[int], list[float]]:
        """Generate trajectory with KV caching.

        Args:
            token_ids: Input token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy)

        Returns:
            Tuple of (all_token_ids, logprobs) where logprobs[i] is the
            log probability of token_ids[i] given the previous tokens.
            Prompt tokens have logprob=0.0.
        """
        ...
