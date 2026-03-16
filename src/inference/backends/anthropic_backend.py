"""Anthropic backend implementation using the Anthropic API.

Note: Anthropic API does not provide logprobs, so trajectory generation
returns 0.0 logprobs for all tokens. This backend is suitable for:
- Text generation
- Categorical judgments (yes/no scoring)

But NOT suitable for:
- Probability-weighted cores (all weights will be equal)
- Perplexity-based metrics
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import torch
from anthropic import Anthropic

from .api_tokenizer import APITokenizer
from .model_backend import Backend


class AnthropicBackend(Backend):
    """Backend using Anthropic API for inference.

    Note: Anthropic API does NOT provide logprobs. All logprob values
    returned by this backend are 0.0. This means:
    - generate_trajectory returns uniform logprobs
    - Probability-based weighting will be uniform
    - Use with caution for normativity estimation

    For categorical judgments (yes/no scoring), this backend works fine.
    """

    supports_inference_mode: bool = False  # Not applicable for API calls

    # Default model (claude-sonnet-4-20250514 is recommended for most uses)
    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(self, runner: Any, model: str | None = None):
        """Initialize Anthropic backend.

        Args:
            runner: ModelRunner instance
            model: Anthropic model name (default: claude-sonnet-4-20250514)
        """
        super().__init__(runner)
        self._model = model or self.DEFAULT_MODEL
        # Use cl100k_base as approximation for Claude tokenization
        self._tokenizer = APITokenizer(encoding_name="cl100k_base")
        self._client = None

    def _get_client(self):
        """Lazy-load Anthropic client."""
        if self._client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable not set. "
                    "Set it with: export ANTHROPIC_API_KEY=your-key"
                )
            self._client = Anthropic(api_key=api_key)
        return self._client

    def get_tokenizer(self):
        return self._tokenizer

    def get_n_layers(self) -> int:
        # Unknown for closed models
        return 0

    def get_d_model(self) -> int:
        # Unknown for closed models
        return 0

    def encode(
        self, text: str, add_special_tokens: bool = True, prepend_bos: bool = False
    ) -> torch.Tensor:
        tokens = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return torch.tensor([tokens])

    def decode(self, token_ids: torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if (
            isinstance(token_ids, list)
            and len(token_ids) > 0
            and isinstance(token_ids[0], list)
        ):
            token_ids = token_ids[0]
        return self._tokenizer.decode(token_ids, skip_special_tokens=False)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        past_kv_cache: Any = None,
    ) -> str:
        client = self._get_client()

        # Anthropic requires temperature > 0, use 0.01 for near-greedy
        temp = temperature if temperature > 0 else 0.0

        response = client.messages.create(
            model=self._model,
            max_tokens=max_new_tokens,
            temperature=temp,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract text from response
        if response.content and len(response.content) > 0:
            return response.content[0].text
        return ""

    def get_next_token_probs(
        self, prompt: str, target_tokens: Sequence[str], past_kv_cache: Any = None
    ) -> dict[str, float]:
        """Get next token probabilities for target tokens.

        Note: Anthropic API does NOT support logprobs.
        This returns uniform probabilities as a fallback.
        """
        # Cannot get real probabilities from Anthropic API
        # Return uniform distribution as fallback
        n = len(target_tokens)
        uniform_prob = 1.0 / n if n > 0 else 0.0
        return {token: uniform_prob for token in target_tokens}

    def get_next_token_probs_by_id(
        self, prompt: str, token_ids: Sequence[int], past_kv_cache: Any = None
    ) -> dict[int, float]:
        """Get next token probabilities by token ID.

        Note: Anthropic API does NOT support logprobs.
        This returns uniform probabilities as a fallback.
        """
        n = len(token_ids)
        uniform_prob = 1.0 / n if n > 0 else 0.0
        return {tid: uniform_prob for tid in token_ids}

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass not supported for API-based backend."""
        raise NotImplementedError(
            "Anthropic backend does not support direct forward passes. "
            "Use generate() or generate_trajectory() instead."
        )

    def generate_trajectory(
        self,
        token_ids: list[int],
        max_new_tokens: int,
        temperature: float,
    ) -> tuple[list[int], list[float]]:
        """Generate trajectory WITHOUT logprobs (Anthropic doesn't support them).

        Args:
            token_ids: Input token IDs (will be decoded to text)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy)

        Returns:
            Tuple of (all_token_ids, logprobs) where logprobs are all 0.0

        Warning:
            All logprobs will be 0.0 since Anthropic API doesn't provide them.
            This means probability-weighted metrics will be uniform.
        """
        # Decode input tokens to text
        prompt = self._tokenizer.decode(token_ids)

        # Generate text
        generated_text = self.generate(prompt, max_new_tokens, temperature)

        # Tokenize the generated text
        generated_ids = self._tokenizer.encode(generated_text)

        # Build full token list
        all_token_ids = list(token_ids) + generated_ids

        # All logprobs are 0.0 (Anthropic doesn't provide them)
        all_logprobs = [0.0] * len(all_token_ids)

        return all_token_ids, all_logprobs

    def generate_trajectory_from_prompt(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        prefilling: str = "",
    ) -> tuple[list[int], list[float], str, str]:
        """Generate trajectory with proper prefill handling for Anthropic.

        Uses assistant message to implement true prefill - the API returns
        only the continuation after the prefill.

        Args:
            prompt: User prompt text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            prefilling: Text to prefill the assistant response with

        Returns:
            Tuple of (all_token_ids, logprobs, prefill_text, generated_text)
            where logprobs are all 0.0 (Anthropic doesn't provide them).
        """
        client = self._get_client()

        # Anthropic requires temperature > 0, use small value for near-greedy
        temp = temperature if temperature > 0 else 0.0

        # Build messages with prefill as assistant message
        messages = [{"role": "user", "content": prompt}]
        if prefilling:
            messages.append({"role": "assistant", "content": prefilling})

        response = client.messages.create(
            model=self._model,
            max_tokens=max_new_tokens,
            temperature=temp,
            messages=messages,
        )

        # Extract continuation (text after prefill)
        continuation = ""
        if response.content and len(response.content) > 0:
            continuation = response.content[0].text

        # Full response = prefill + continuation
        full_response = prefilling + continuation

        # Tokenize prompt and full response
        prompt_ids = self._tokenizer.encode(prompt)
        response_ids = self._tokenizer.encode(full_response)

        # Build full token sequence
        all_token_ids = prompt_ids + response_ids

        # All logprobs are 0.0 (Anthropic doesn't provide them)
        all_logprobs = [0.0] * len(all_token_ids)

        return all_token_ids, all_logprobs, prefilling, continuation
