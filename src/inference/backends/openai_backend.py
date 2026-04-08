"""OpenAI backend implementation using the OpenAI API."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Sequence
from typing import Any

import torch
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from .api_tokenizer import APITokenizer
from .model_backend import Backend

# Retry configuration
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0  # seconds
MAX_BACKOFF = 60.0  # seconds
BACKOFF_MULTIPLIER = 2.0


def _retry_api_call(func, *args, **kwargs):
    """Execute API call with exponential backoff retry.

    Handles transient errors like empty responses, connection errors,
    rate limits, and server errors (5xx).
    """
    last_exception = None
    backoff = INITIAL_BACKOFF

    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except RateLimitError as e:
            # Rate limit - use longer backoff
            last_exception = e
            wait_time = min(backoff * 2, MAX_BACKOFF)
            print(f"  [Retry {attempt + 1}/{MAX_RETRIES}] Rate limited, waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
            backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF)
        except APIConnectionError as e:
            # Connection error - retry with backoff
            last_exception = e
            print(f"  [Retry {attempt + 1}/{MAX_RETRIES}] Connection error, waiting {backoff:.1f}s...")
            time.sleep(backoff)
            backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF)
        except APIStatusError as e:
            # Server errors (5xx) - retry; client errors (4xx) - don't retry,
            # EXCEPT for 400 "could not parse JSON body" which is a transient
            # network corruption error, not a real client error.
            is_json_parse_error = (
                e.status_code == 400
                and "could not parse the json body" in str(e).lower()
            )
            if e.status_code >= 500 or is_json_parse_error:
                last_exception = e
                print(f"  [Retry {attempt + 1}/{MAX_RETRIES}] Server error {e.status_code}, waiting {backoff:.1f}s...")
                time.sleep(backoff)
                backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF)
            else:
                # Client error (4xx) - don't retry
                raise
        except Exception as e:
            # Catch JSON decode errors and other transient issues
            error_str = str(e).lower()
            if "json" in error_str or "expecting value" in error_str or "empty" in error_str:
                last_exception = e
                print(f"  [Retry {attempt + 1}/{MAX_RETRIES}] Empty/invalid response, waiting {backoff:.1f}s...")
                time.sleep(backoff)
                backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF)
            else:
                # Unknown error - re-raise
                raise

    # All retries exhausted
    raise RuntimeError(
        f"API call failed after {MAX_RETRIES} retries. Last error: {last_exception}"
    ) from last_exception


# Instruction to simulate prefill behavior since OpenAI doesn't support true assistant prefill.
# This is appended to the user message when a prefill is requested.
# Research indicates that explicit, direct instructions work best for continuation.
OPENAI_PREFILL_INSTRUCTION = (
    "Continue from the following text exactly as written, without repeating it. "
    "Your response must seamlessly continue from this starting point:\n\n{prefill}"
)


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
        # GPT-4o uses o200k_base encoding
        self._tokenizer = APITokenizer(encoding_name="o200k_base")
        self._client = None

    def _get_client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
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

        # Use retry wrapper for robustness
        response = _retry_api_call(
            client.chat.completions.create,
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
        client = self._get_client()

        # Use retry wrapper for robustness
        response = _retry_api_call(
            client.chat.completions.create,
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

        # Use retry wrapper for robustness
        response = _retry_api_call(
            client.chat.completions.create,
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

    def generate_trajectory_from_prompt(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        prefilling: str = "",
    ) -> tuple[list[int], list[float], str, str]:
        """Generate trajectory with prefill handling for OpenAI.

        OpenAI doesn't support true prefill like Anthropic, so we include
        the prefill instruction in the user message and prepend it to the result.

        Args:
            prompt: User prompt text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            prefilling: Text to prefill the assistant response with

        Returns:
            Tuple of (all_token_ids, logprobs, prefill_text, generated_text)
            Note: logprobs for prefill tokens are 0.0 (not from model).
        """
        client = self._get_client()

        # Use temperature=0 for greedy
        temp = temperature if temperature > 0 else 0

        # OpenAI doesn't support true prefill, so include instruction in the prompt
        full_prompt = prompt
        if prefilling:
            instruction = OPENAI_PREFILL_INSTRUCTION.format(prefill=prefilling)
            full_prompt = f"{prompt}\n\n{instruction}"

        # Use retry wrapper for robustness
        response = _retry_api_call(
            client.chat.completions.create,
            model=self._model,
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=max_new_tokens,
            temperature=temp,
            logprobs=True,
        )

        choice = response.choices[0]
        raw_response = choice.message.content or ""

        # The model should have started with the prefill, but we ensure it
        # by prepending if needed (and avoiding duplication)
        if prefilling and raw_response.startswith(prefilling):
            continuation = raw_response[len(prefilling) :]
        else:
            continuation = raw_response

        full_response = prefilling + continuation

        # Tokenize prompt and full response
        prompt_ids = self._tokenizer.encode(prompt)
        prefill_ids = self._tokenizer.encode(prefilling) if prefilling else []

        # Build token IDs: prompt + prefill (with 0.0 logprobs) + generated (with real logprobs if available)
        all_token_ids = prompt_ids + prefill_ids
        all_logprobs = [0.0] * len(all_token_ids)

        # Extract generated tokens and logprobs from API response
        if choice.logprobs and choice.logprobs.content:
            for token_info in choice.logprobs.content:
                token_bytes = token_info.bytes
                if token_bytes:
                    try:
                        token_str = bytes(token_bytes).decode("utf-8")
                        token_id = self._tokenizer.encode(token_str)
                        if token_id:
                            all_token_ids.append(token_id[0])
                            all_logprobs.append(token_info.logprob)
                    except (UnicodeDecodeError, IndexError):
                        pass
        else:
            # Fallback: tokenize continuation and use 0.0 logprobs
            continuation_ids = self._tokenizer.encode(continuation)
            all_token_ids.extend(continuation_ids)
            all_logprobs.extend([0.0] * len(continuation_ids))

        return all_token_ids, all_logprobs, prefilling, continuation
