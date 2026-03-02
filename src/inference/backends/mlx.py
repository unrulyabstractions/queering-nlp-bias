"""MLX backend implementation for Apple Silicon."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .model_backend import Backend


def _get_mx():
    """Lazy import of mlx.core."""
    import mlx.core as mx

    return mx


def _get_generate():
    """Lazy import of mlx_lm generate."""
    from mlx_lm import generate

    return generate


def _get_stream_generate():
    """Lazy import of mlx_lm stream_generate."""
    from mlx_lm import stream_generate

    return stream_generate


class MLXBackend(Backend):
    """Backend using MLX for Apple Silicon inference."""

    def __init__(self, runner: Any, tokenizer: Any):
        """Initialize MLX backend.

        Args:
            runner: ModelRunner instance
            tokenizer: Tokenizer loaded from mlx_lm
        """
        super().__init__(runner)
        # MLX wraps HuggingFace tokenizer; store underlying one for full API compatibility
        if hasattr(tokenizer, "_tokenizer"):
            self._tokenizer = tokenizer._tokenizer
        else:
            self._tokenizer = tokenizer

    def get_tokenizer(self):
        return self._tokenizer

    def get_n_layers(self) -> int:
        return len(self.runner._model.layers)

    def get_d_model(self) -> int:
        args = self.runner._model.args
        # Try common attribute names for hidden dimension
        if hasattr(args, "hidden_size"):
            return args.hidden_size
        if hasattr(args, "dim"):
            return args.dim
        if hasattr(args, "d_model"):
            return args.d_model
        # Some models (e.g., Gemma 3n) nest config in text_config
        if hasattr(args, "text_config"):
            text_cfg = args.text_config
            if isinstance(text_cfg, dict):
                return text_cfg.get("hidden_size") or text_cfg.get("dim")
            elif hasattr(text_cfg, "hidden_size"):
                return text_cfg.hidden_size
        raise AttributeError(f"Cannot find hidden size in model args: {args}")

    def encode(
        self, text: str, add_special_tokens: bool = True, prepend_bos: bool = False
    ) -> torch.Tensor:
        tokens = self.runner._tokenizer.encode(
            text, add_special_tokens=add_special_tokens
        )
        if prepend_bos and self.runner._tokenizer.bos_token_id is not None:
            tokens = [self.runner._tokenizer.bos_token_id] + tokens
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
        return self.runner._tokenizer.decode(token_ids, skip_special_tokens=False)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        past_kv_cache: Any = None,
    ) -> str:
        generate = _get_generate()

        if temperature > 0:
            from mlx_lm.sample_utils import make_sampler

            sampler = make_sampler(temp=temperature)
        else:
            sampler = None

        return generate(
            self.runner._model,
            self.runner._tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens,
            sampler=sampler,
        )

    def get_next_token_probs(
        self, prompt: str, target_tokens: Sequence[str], past_kv_cache: Any = None
    ) -> dict[str, float]:
        mx = _get_mx()

        input_ids = self.runner._tokenizer.encode(prompt)
        input_mx = mx.array([input_ids])

        logits = self.runner._model(input_mx)
        last_logits = logits[0, -1, :]
        probs = mx.softmax(last_logits)

        result = {}
        for token_str in target_tokens:
            ids = self.runner._tokenizer.encode(token_str)
            if ids:
                result[token_str] = probs[ids[0]].item()
            else:
                result[token_str] = 0.0
        return result

    def get_next_token_probs_by_id(
        self, prompt: str, token_ids: Sequence[int], past_kv_cache: Any = None
    ) -> dict[int, float]:
        mx = _get_mx()

        input_ids = self.runner._tokenizer.encode(prompt)
        input_mx = mx.array([input_ids])

        logits = self.runner._model(input_mx)
        last_logits = logits[0, -1, :]
        probs = mx.softmax(last_logits)

        result = {}
        for tok_id in token_ids:
            if tok_id is not None:
                result[tok_id] = probs[tok_id].item()
        return result

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run forward pass and return logits as PyTorch tensor."""
        import numpy as np

        mx = _get_mx()

        if isinstance(input_ids, torch.Tensor):
            input_mx = mx.array(input_ids.cpu().numpy().astype("int32"))
        else:
            input_mx = mx.array(input_ids)

        logits_mx = self.runner._model(input_mx)
        # Convert to float32 for numpy compatibility (mlx defaults to bfloat16)
        logits_mx = logits_mx.astype(mx.float32)
        mx.eval(logits_mx)
        # Convert to torch and move to runner's device
        return torch.from_numpy(np.array(logits_mx)).to(self.runner.device)

    def generate_trajectory(
        self,
        token_ids: list[int],
        max_new_tokens: int,
        temperature: float,
    ) -> tuple[list[int], list[float]]:
        """Generate trajectory using stream_generate with KV caching."""
        mx = _get_mx()
        stream_generate = _get_stream_generate()

        # Compute logprobs for prefilled tokens via forward pass
        input_mx = mx.array([token_ids])
        logits = self.runner._model(input_mx)
        log_probs = mx.softmax(logits, axis=-1)
        log_probs = mx.log(log_probs + 1e-12)  # Add epsilon to avoid log(0)

        # For each position i, get logprob of token at position i+1
        all_logprobs: list[float] = [0.0]  # First token has no prior context
        for i in range(len(token_ids) - 1):
            next_token = token_ids[i + 1]
            lp = float(log_probs[0, i, next_token].item())
            all_logprobs.append(lp)

        # Build kwargs for stream_generate
        kwargs = {}
        if temperature > 0:
            from mlx_lm.sample_utils import make_sampler
            kwargs["sampler"] = make_sampler(temp=temperature)

        all_token_ids = list(token_ids)

        for response in stream_generate(
            self.runner._model,
            self.runner._tokenizer,
            prompt=token_ids,
            max_tokens=max_new_tokens,
            **kwargs,
        ):
            all_token_ids.append(response.token)
            # logprobs is a vector; get the logprob for the selected token
            token_logprob = float(response.logprobs[response.token].item())
            all_logprobs.append(token_logprob)

            if response.finish_reason == "stop":
                break

        return all_token_ids, all_logprobs
