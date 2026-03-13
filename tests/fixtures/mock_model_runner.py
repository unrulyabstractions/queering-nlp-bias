"""Mock ModelRunner for testing without actual model inference.

Provides a configurable mock that simulates ModelRunner behavior for unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from src.inference.generated_trajectory import GeneratedTrajectory


@dataclass
class MockGenerationResponse:
    """Configurable response for mock generation."""

    token_ids: list[int]
    logprobs: list[float]
    generated_text: str = "Mock generated text"


@dataclass
class MockModelRunner:
    """Mock implementation of ModelRunner for testing.

    Attributes:
        vocab_size: Size of vocabulary for logits
        bos_token_id: Beginning of sequence token ID
        eos_token_id: End of sequence token ID
        pad_token_id: Padding token ID
        responses: Queue of pre-configured responses for generation
        encode_map: Custom token ID mappings for encode_text
        decode_map: Custom text mappings for decode_ids
    """

    vocab_size: int = 32000
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    responses: list[MockGenerationResponse] = field(default_factory=list)
    encode_map: dict[str, list[int]] = field(default_factory=dict)
    decode_map: dict[tuple[int, ...], str] = field(default_factory=dict)
    _response_idx: int = field(default=0, repr=False)

    def encode_text(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Returns custom mapping if available, otherwise generates deterministic IDs.
        """
        if text in self.encode_map:
            return self.encode_map[text]

        # Generate deterministic token IDs based on text hash
        if not text:
            return []

        # Simple deterministic encoding: each char maps to an ID
        return [ord(c) % self.vocab_size for c in text[:100]]

    def encode_ids(self, text: str) -> list[int]:
        """Alias for encode_text."""
        return self.encode_text(text)

    def decode_ids(self, token_ids: list[int]) -> str:
        """Decode token IDs to text.

        Returns custom mapping if available, otherwise generates placeholder text.
        """
        key = tuple(token_ids)
        if key in self.decode_map:
            return self.decode_map[key]

        # Generate deterministic text
        return f"[decoded:{len(token_ids)}tokens]"

    def generate_trajectory_from_prompt(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        prefilling: str | None = None,
    ) -> GeneratedTrajectory:
        """Generate a trajectory from a prompt.

        Uses pre-configured responses if available, otherwise generates a mock response.
        """
        # Use pre-configured response if available
        if self._response_idx < len(self.responses):
            response = self.responses[self._response_idx]
            self._response_idx += 1

            return GeneratedTrajectory(
                token_ids=response.token_ids,
                logprobs=response.logprobs,
                logits=response.logprobs,  # Use logprobs as scalar logit approximation
                full_logits=None,
                prefill_text=prefilling or "",
                generated_text=response.generated_text,
            )

        # Generate a default mock response
        n_tokens = min(max_new_tokens, 20)
        prompt_ids = self.encode_text(prompt)
        prefill_ids = self.encode_text(prefilling or "")

        # Generate "new" tokens
        generated_ids = [100 + i for i in range(n_tokens)]
        all_ids = prompt_ids + prefill_ids + generated_ids

        # Generate logprobs: 0 for prompt tokens, random-ish for generated
        logprobs = [0.0] * len(prompt_ids + prefill_ids)
        for i in range(n_tokens):
            # Deterministic "random" logprobs
            logprobs.append(-1.0 - (i * 0.1))

        return GeneratedTrajectory(
            token_ids=all_ids,
            logprobs=logprobs,
            logits=logprobs,
            full_logits=None,
            prefill_text=prefilling or "",
            generated_text=f"[mock generated {n_tokens} tokens]",
            prefill_length=len(prompt_ids + prefill_ids),
        )

    def add_response(
        self,
        token_ids: list[int],
        logprobs: list[float],
        generated_text: str = "Mock text",
    ) -> None:
        """Add a pre-configured response to the queue."""
        self.responses.append(
            MockGenerationResponse(
                token_ids=token_ids,
                logprobs=logprobs,
                generated_text=generated_text,
            )
        )

    def reset_responses(self) -> None:
        """Reset the response queue."""
        self._response_idx = 0


def create_mock_runner(
    vocab_size: int = 32000,
    responses: list[MockGenerationResponse] | None = None,
) -> MockModelRunner:
    """Factory function to create a MockModelRunner.

    Args:
        vocab_size: Vocabulary size
        responses: Optional list of pre-configured responses

    Returns:
        Configured MockModelRunner instance
    """
    return MockModelRunner(
        vocab_size=vocab_size,
        responses=responses or [],
    )
