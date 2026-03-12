"""HuggingFace Transformers backend implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .model_backend import Backend


class HuggingFaceBackend(Backend):
    """Backend using HuggingFace Transformers for model inference."""

    def __init__(self, runner: Any, tokenizer: Any):
        super().__init__(runner)
        self._tokenizer = tokenizer

    def get_tokenizer(self):
        return self._tokenizer

    def get_n_layers(self) -> int:
        return self.runner._model.config.num_hidden_layers

    def get_d_model(self) -> int:
        return self.runner._model.config.hidden_size

    def encode(
        self, text: str, add_special_tokens: bool = True, prepend_bos: bool = False
    ) -> torch.Tensor:
        tokens = self.runner._tokenizer(
            text, return_tensors="pt", add_special_tokens=add_special_tokens
        )
        input_ids = tokens["input_ids"].to(self.runner.device)
        if prepend_bos and self.runner._tokenizer.bos_token_id is not None:
            bos = torch.tensor(
                [[self.runner._tokenizer.bos_token_id]], device=self.runner.device
            )
            input_ids = torch.cat([bos, input_ids], dim=1)
        return input_ids

    def decode(self, token_ids: torch.Tensor) -> str:
        return self.runner._tokenizer.decode(token_ids, skip_special_tokens=False)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        past_kv_cache: Any = None,
    ) -> str:
        input_ids = self.encode(prompt)
        prompt_len = input_ids.shape[1]

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.runner._tokenizer.eos_token_id,
            # Override model's default generation config to ensure greedy decoding
            "repetition_penalty": 1.0,
            "num_beams": 1,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = self.runner._model.generate(input_ids, **gen_kwargs)

        return self.decode(output_ids[0, prompt_len:])

    def get_next_token_probs(
        self, prompt: str, target_tokens: Sequence[str], past_kv_cache: Any = None
    ) -> dict[str, float]:
        input_ids = self.encode(prompt)
        with torch.no_grad():
            outputs = self.runner._model(input_ids)
            logits = outputs.logits
        probs = torch.softmax(logits[0, -1, :], dim=-1)

        result = {}
        for token_str in target_tokens:
            ids = self.runner._tokenizer.encode(token_str, add_special_tokens=False)
            result[token_str] = probs[ids[0]].item() if ids else 0.0
        return result

    def get_next_token_probs_by_id(
        self, prompt: str, token_ids: Sequence[int], past_kv_cache: Any = None
    ) -> dict[int, float]:
        input_ids = self.encode(prompt)
        with torch.no_grad():
            outputs = self.runner._model(input_ids)
            logits = outputs.logits
        probs = torch.softmax(logits[0, -1, :], dim=-1)

        result = {}
        for tok_id in token_ids:
            if tok_id is not None:
                result[tok_id] = probs[tok_id].item()
        return result

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.runner._model(input_ids)
            return outputs.logits

    def generate_trajectory(
        self,
        token_ids: list[int],
        max_new_tokens: int,
        temperature: float,
    ) -> tuple[list[int], list[float]]:
        """Generate trajectory using HF generate() with KV caching."""
        input_ids = torch.tensor([token_ids], device=self.runner.device)
        prompt_len = len(token_ids)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
            "use_cache": True,  # Enable KV caching
            "repetition_penalty": 1.0,
            "num_beams": 1,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = self.runner._model.generate(input_ids, **gen_kwargs)

        # outputs.sequences: [1, prompt_len + generated_len]
        # outputs.scores: tuple of (generated_len) tensors, each [1, vocab_size]
        all_token_ids = outputs.sequences[0].tolist()
        generated_ids = all_token_ids[prompt_len:]

        # Compute logprobs for prefilled tokens via forward pass
        with torch.no_grad():
            prefix_outputs = self.runner._model(input_ids)
            prefix_logits = prefix_outputs.logits[0]  # [prompt_len, vocab_size]
            prefix_log_probs = torch.log_softmax(prefix_logits, dim=-1)

        # For position i, get logprob of token[i+1] given context up to i
        all_logprobs = [0.0]  # First token has no prior context
        for i in range(prompt_len - 1):
            next_token = token_ids[i + 1]
            all_logprobs.append(prefix_log_probs[i, next_token].item())

        # Add logprobs for generated tokens
        for i, (score, token_id) in enumerate(zip(outputs.scores, generated_ids)):
            log_probs = torch.log_softmax(score[0], dim=-1)
            all_logprobs.append(log_probs[token_id].item())

        return all_token_ids, all_logprobs
