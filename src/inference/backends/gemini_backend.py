"""Gemini backend implementation using the google-genai SDK.

Mirrors the surface of :class:`OpenAIBackend` so the rest of the codebase
treats Gemini like any other API model. The Gemini API does not expose
per-token logprobs, so trajectory logprobs are 0.0 (same caveat as
Anthropic).
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Any

import torch

from .api_tokenizer import APITokenizer
from .model_backend import Backend


GEMINI_DEFAULT_MODEL = "gemini-2.5-pro"

# Retry configuration mirrors OpenAI/Anthropic backends.
MAX_RETRIES = 8
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 120.0
BACKOFF_MULTIPLIER = 2.0
RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}


class GeminiBackend(Backend):
    """Backend using Google Gemini API via the official google-genai SDK."""

    supports_inference_mode: bool = False

    def __init__(self, runner: Any, model: str | None = None):
        """Initialize Gemini backend.

        Args:
            runner: ModelRunner instance.
            model: Gemini model id (e.g. ``"gemini-2.5-pro"``). Defaults to
                :data:`GEMINI_DEFAULT_MODEL`.
        """
        super().__init__(runner)
        self._model = model or GEMINI_DEFAULT_MODEL
        # Tokenizer is only used for rough byte-level counts; Gemini doesn't
        # accept token ids over the wire.
        self._tokenizer = APITokenizer(encoding_name="cl100k_base")
        self._client = None

    def _get_client(self):
        """Lazy-load the Gemini client."""
        if self._client is None:
            from google import genai

            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
                "GOOGLE_API_KEY"
            )
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable not set. "
                    "Set it with: export GEMINI_API_KEY=your-key"
                )
            self._client = genai.Client(api_key=api_key)
        return self._client

    # ── tokenizer / shape ────────────────────────────────────────────────

    def get_tokenizer(self):
        return self._tokenizer

    def get_n_layers(self) -> int:
        return 0

    def get_d_model(self) -> int:
        return 0

    def encode(
        self, text: str, add_special_tokens: bool = True, prepend_bos: bool = False
    ) -> torch.Tensor:
        return torch.tensor(
            [self._tokenizer.encode(text, add_special_tokens=add_special_tokens)]
        )

    def decode(self, token_ids: torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return self._tokenizer.decode(token_ids, skip_special_tokens=False)

    # ── generation ───────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        past_kv_cache: Any = None,
    ) -> str:
        """Generate text via Gemini, with bounded retry on transient errors.

        Gemini-2.5 models default to internal "thinking" tokens that share
        the output budget. We disable that here so the configured
        `max_new_tokens` is fully available to the visible response —
        callers that want chain-of-thought should put it in the prompt.
        """
        from google.genai import errors as genai_errors

        client = self._get_client()
        # Note: gemini-2.5-pro requires `thinking_budget > 0`. We let the
        # model decide its own thinking budget but cap the *visible*
        # response within max_output_tokens (callers should pass a
        # generous value when the visible reply needs room — e.g. CoT).
        config = {
            "max_output_tokens": max_new_tokens,
            "temperature": float(temperature) if temperature > 0 else 0.0,
        }
        backoff = INITIAL_BACKOFF
        last_err: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=config,
                )
                return response.text or ""
            except genai_errors.APIError as e:
                code = getattr(e, "code", None)
                if code not in RETRYABLE_CODES:
                    raise
                last_err = e
                wait = min(backoff, MAX_BACKOFF)
                print(
                    f"  [Retry {attempt + 1}/{MAX_RETRIES}] Gemini {self._model} "
                    f"got {code}; waiting {wait:.0f}s..."
                )
                time.sleep(wait)
                backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF)

        raise RuntimeError(
            f"Gemini call failed after {MAX_RETRIES} retries. Last error: {last_err}"
        ) from last_err

    # ── unsupported probability / forward APIs ───────────────────────────
    # Gemini does not expose token-level logprobs or activations; we return
    # uniform fallbacks to satisfy the abstract Backend interface.

    def get_next_token_probs(
        self,
        prompt: str,
        target_tokens: Sequence[str],
        past_kv_cache: Any = None,
    ) -> dict[str, float]:
        n = len(target_tokens)
        return {t: (1.0 / n if n else 0.0) for t in target_tokens}

    def get_next_token_probs_by_id(
        self,
        prompt: str,
        token_ids: Sequence[int],
        past_kv_cache: Any = None,
    ) -> dict[int, float]:
        n = len(token_ids)
        return {tid: (1.0 / n if n else 0.0) for tid in token_ids}

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Gemini backend has no direct forward pass.")

    def generate_trajectory(
        self,
        token_ids: list[int],
        max_new_tokens: int,
        temperature: float,
    ) -> tuple[list[int], list[float]]:
        prompt = self._tokenizer.decode(token_ids)
        text = self.generate(prompt, max_new_tokens, temperature)
        gen_ids = self._tokenizer.encode(text)
        all_ids = list(token_ids) + gen_ids
        # No per-token logprobs available from Gemini.
        return all_ids, [0.0] * len(all_ids)
