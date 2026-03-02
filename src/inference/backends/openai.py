"""OpenAI backend implementation using the OpenAI API."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .model_backend import Backend


@dataclass
class OpenAITokenizer:
    """Minimal tokenizer interface wrapping tiktoken for OpenAI models."""

    encoding_name: str = "o200k_base"  # GPT-4o encoding

    def __post_init__(self):
        import tiktoken

        self._encoding = tiktoken.get_encoding(self.encoding_name)

    @property
    def vocab_size(self) -> int:
        return self._encoding.n_vocab

    @property
    def bos_token_id(self) -> int | None:
        return None  # OpenAI models don't expose BOS

    @property
    def eos_token_id(self) -> int | None:
        return self._encoding.eot_token

    @property
    def pad_token_id(self) -> int | None:
        return None

    @property
    def bos_token(self) -> str | None:
        return None

    @property
    def eos_token(self) -> str | None:
        return "<|endoftext|>"

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self._encoding.encode(text)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._encoding.decode(token_ids)


class OpenAIBackend(Backend):
    """Backend using OpenAI API for inference."""

    supports_inference_mode: bool = False  # Not applicable for API calls

    def __init__(self, runner: Any, model: str = "gpt-4o"):
        """Initialize OpenAI backend.

        Args:
            runner: ModelRunner instance
            model: OpenAI model name (default: gpt-4o)
        """
        super().__init__(runner)
        self._model = model
        self._tokenizer = OpenAITokenizer()
        self._client = None

    def _get_client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            from openai import OpenAI

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable not set. "
                    "Set it with: export OPENAI_API_KEY=your-key"
                )
            self._client = OpenAI(api_key=api_key)
        return self._client

    def get_tokenizer(self):
        return self._tokenizer

    def get_n_layers(self) -> int:
        # Unknown for closed models - return placeholder
        return 0

    def get_d_model(self) -> int:
        # Unknown for closed models - return placeholder
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

        # Use temperature=0 for greedy, otherwise provided value
        temp = temperature if temperature > 0 else 0

        response = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temp,
        )

        return response.choices[0].message.content or ""

    def get_next_token_probs(
        self, prompt: str, target_tokens: Sequence[str], past_kv_cache: Any = None
    ) -> dict[str, float]:
        """Get next token probabilities for target tokens.

        Note: OpenAI API has limited logprobs support. This uses the
        logprobs parameter to get top-k token probabilities.
        """
        import math

        client = self._get_client()

        response = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,  # Max allowed
        )

        result = {token: 0.0 for token in target_tokens}

        choice = response.choices[0]
        if choice.logprobs and choice.logprobs.content:
            top_logprobs = choice.logprobs.content[0].top_logprobs
            logprob_dict = {lp.token: lp.logprob for lp in top_logprobs}

            for token in target_tokens:
                if token in logprob_dict:
                    result[token] = math.exp(logprob_dict[token])

        return result

    def get_next_token_probs_by_id(
        self, prompt: str, token_ids: Sequence[int], past_kv_cache: Any = None
    ) -> dict[int, float]:
        """Get next token probabilities by token ID.

        Note: OpenAI API doesn't directly support token ID queries.
        This decodes the IDs and uses string-based lookup.
        """
        # Convert IDs to strings
        token_strs = [self._tokenizer.decode([tid]) for tid in token_ids]
        str_probs = self.get_next_token_probs(prompt, token_strs, past_kv_cache)

        # Map back to IDs
        result = {}
        for tid, tstr in zip(token_ids, token_strs):
            result[tid] = str_probs.get(tstr, 0.0)

        return result

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass not supported for API-based backend."""
        raise NotImplementedError(
            "OpenAI backend does not support direct forward passes. "
            "Use generate() or generate_trajectory() instead."
        )

    def generate_trajectory(
        self,
        token_ids: list[int],
        max_new_tokens: int,
        temperature: float,
    ) -> tuple[list[int], list[float]]:
        """Generate trajectory with logprobs using OpenAI API.

        Args:
            token_ids: Input token IDs (will be decoded to text)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy)

        Returns:
            Tuple of (all_token_ids, logprobs)
        """
        client = self._get_client()

        # Decode input tokens to text
        prompt = self._tokenizer.decode(token_ids)

        # Use temperature=0 for greedy
        temp = temperature if temperature > 0 else 0

        response = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temp,
            logprobs=True,
        )

        # Process response
        choice = response.choices[0]

        # Build token IDs and logprobs from API response
        # Input tokens have logprob=0.0 (not available from API)
        all_token_ids = list(token_ids)
        all_logprobs = [0.0] * len(token_ids)

        # Extract tokens and logprobs from API response (aligned to each other)
        if choice.logprobs and choice.logprobs.content:
            for token_info in choice.logprobs.content:
                # Get the token bytes and encode to get token ID
                token_bytes = token_info.bytes
                if token_bytes:
                    # Decode bytes to string, then encode to get token ID
                    try:
                        token_str = bytes(token_bytes).decode("utf-8")
                        token_id = self._tokenizer.encode(token_str)
                        if token_id:
                            all_token_ids.append(token_id[0])
                            all_logprobs.append(token_info.logprob)
                    except (UnicodeDecodeError, IndexError):
                        # Skip problematic tokens
                        pass
        else:
            # Fallback: tokenize the text and use 0.0 logprobs
            generated_text = choice.message.content or ""
            generated_ids = self._tokenizer.encode(generated_text)
            all_token_ids.extend(generated_ids)
            all_logprobs.extend([0.0] * len(generated_ids))

        return all_token_ids, all_logprobs
