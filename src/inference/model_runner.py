"""Model runner for inference."""

from __future__ import annotations

from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.default_config import MAX_NEW_TOKENS
from src.common.device_utils import clear_gpu_memory, get_device
from src.common.logging import log
from src.common.profiler import profile, track_memory

from .backends import (
    AnthropicBackend,
    HuggingFaceBackend,
    MLXBackend,
    ModelBackend,
    OpenAIBackend,
    get_recommended_backend_inference,
)
from .generated_trajectory import (
    GeneratedTrajectory,
)


class ModelRunner:
    """Model runner for inference."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        backend: ModelBackend | None = None,
    ):
        self.model_name = model_name

        # Auto-detect OpenAI models
        if backend is None:
            backend = self._detect_backend(model_name)

        if device is None:
            device = get_device()
        self.device = device
        if dtype is None:
            dtype = torch.float16 if device in ["mps", "cuda"] else torch.float32
        self.dtype = dtype

        self._model = None
        self._backend_type = backend

        if backend == ModelBackend.OPENAI:
            self._init_openai()
        elif backend == ModelBackend.ANTHROPIC:
            self._init_anthropic()
        elif backend == ModelBackend.HUGGINGFACE:
            self._init_huggingface()
        elif backend == ModelBackend.MLX:
            self._init_mlx()
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self._is_chat_model = self._detect_chat_model(model_name)

        log(f"Model loaded: {backend} {model_name} (chat={self._is_chat_model})")
        if backend not in (ModelBackend.OPENAI, ModelBackend.ANTHROPIC):
            log(f"  n_layers={self.n_layers}, d_model={self.d_model}\n")
        else:
            log("")

    ############################
    #            API           #
    ############################

    @property
    def _tokenizer(self):
        return self._backend.get_tokenizer()

    @property
    def bos_token_id(self) -> int | None:
        return self._tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int | None:
        return self._tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int | None:
        return self._tokenizer.pad_token_id

    @property
    def bos_token(self) -> str | None:
        return self._tokenizer.bos_token

    @property
    def eos_token(self) -> str | None:
        return self._tokenizer.eos_token

    @property
    def n_layers(self) -> int:
        return self._backend.get_n_layers()

    @property
    def d_model(self) -> int:
        return self._backend.get_d_model()

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self._tokenizer.vocab_size

    @property
    def is_reasoning_model(self) -> bool:
        """Whether this model supports thinking/reasoning mode."""
        if not hasattr(self, "_is_reasoning_model"):
            self._is_reasoning_model = self._detect_reasoning_model()
        return self._is_reasoning_model

    @property
    def skip_thinking_prefix(self) -> str:
        """Prefix to skip thinking mode for reasoning models.

        Returns empty string for non-reasoning models.
        """
        if self.is_reasoning_model:
            return "<think>\n</think>\n\n"
        return ""

    def encode(
        self, text: str, add_special_tokens: bool = True, prepend_bos: bool = False
    ) -> torch.Tensor:
        """Encode text into tensor of token IDs.

        Args:
            text: Input text to encode
            add_special_tokens: Whether to add special tokens (default True)
            prepend_bos: Whether to prepend BOS token (default False)

        Returns:
            Token IDs tensor of shape [1, seq_len]
        """
        return self._backend.encode(
            text, add_special_tokens=add_special_tokens, prepend_bos=prepend_bos
        )

    def encode_ids(
        self, text: str, add_special_tokens: bool = True, prepend_bos: bool = False
    ) -> list[int]:
        """Encode text into list of token IDs."""
        tensor = self.encode(
            text, add_special_tokens=add_special_tokens, prepend_bos=prepend_bos
        )
        # Use squeeze(0) to only remove batch dim, keeping sequence dim
        # This ensures tolist() always returns a list, not a scalar
        result = tensor.squeeze(0).tolist()
        # Handle edge case where result is still a scalar (single token)
        return result if isinstance(result, list) else [result]

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode tensor of token IDs to string."""
        return self._backend.decode(token_ids)

    def decode_ids(self, token_ids: list[int]) -> str:
        """Decode list of token IDs to string."""
        return self._backend.decode(torch.tensor(token_ids))

    @profile
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = 0.0,
        past_kv_cache: Any = None,
        prefilling: str = "",
    ) -> str:
        """Generate text from prompt, preserving special tokens like EOS."""
        formatted = self.apply_chat_template(prompt) + prefilling

        # For API-based backends, use the backend's generate directly
        if self._backend_type in (ModelBackend.OPENAI, ModelBackend.ANTHROPIC):
            return self._backend.generate(formatted, max_new_tokens, temperature)

        # For local backends, generate token by token to preserve EOS
        input_ids = self.encode_ids(formatted, add_special_tokens=True)
        all_token_ids = list(input_ids)
        ctx = (
            torch.inference_mode()
            if self._backend.supports_inference_mode
            else torch.no_grad()
        )

        with ctx:
            for _ in range(max_new_tokens):
                input_tensor = torch.tensor([all_token_ids], device=self.device)
                logits_batch = self._backend.forward(input_tensor)
                next_logits = logits_batch[0, -1, :]

                if temperature == 0.0:
                    next_token = next_logits.argmax().item()
                else:
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()

                all_token_ids.append(next_token)

                if next_token == self.eos_token_id:
                    break

        # Decode only the generated portion (after input), with special tokens
        generated_ids = all_token_ids[len(input_ids) :]
        return self.decode_ids(generated_ids)

    @profile
    def generate_trajectory(
        self,
        token_ids: list[int],
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = 0.0,
    ) -> GeneratedTrajectory:
        """Generate text autoregressively and return trajectory with logprobs.

        Args:
            token_ids: Initial token IDs to start generation from
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = greedy)

        Returns:
            GeneratedTrajectory containing all tokens (input + generated) with logprobs
        """
        all_token_ids, all_logprobs = self._backend.generate_trajectory(
            token_ids, max_new_tokens, temperature
        )
        return GeneratedTrajectory.from_logprobs(all_token_ids, all_logprobs)

    @profile
    def generate_trajectory_from_prompt(
        self,
        prompt: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = 0.0,
        prefilling: str = "",
    ) -> GeneratedTrajectory:
        """Generate tokens from prompt and return trajectory with logprobs.

        Applies chat template, generates tokens, and computes logprobs for
        the full sequence. Also sets continuation_text on the trajectory.

        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0 for greedy)
            prefilling: Optional text to prepend to the generation

        Returns:
            GeneratedTrajectory with full sequence (prompt + generated) and logprobs
        """
        # For API backends, delegate to backend's prefill-aware implementation
        if self._backend_type in (ModelBackend.OPENAI, ModelBackend.ANTHROPIC):
            all_token_ids, all_logprobs, prefill_text, generated_text = (
                self._backend.generate_trajectory_from_prompt(
                    prompt, max_new_tokens, temperature, prefilling
                )
            )
            traj = GeneratedTrajectory.from_logprobs(all_token_ids, all_logprobs)
            traj.prefill_text = prefill_text
            traj.generated_text = generated_text
            traj.prefill_length = len(self.encode_ids(prompt)) + len(
                self.encode_ids(prefilling)
            )
            return traj

        # For local backends, use token-based generation
        formatted = self.apply_chat_template(prompt) + prefilling
        token_ids = self.encode_ids(formatted, add_special_tokens=True)
        prefill_length = len(token_ids)  # Where generated content starts
        traj = self.generate_trajectory(token_ids, max_new_tokens, temperature)

        # Set text fields
        full_text = self.decode_ids(traj.token_ids)
        traj.prefill_text = prefilling  # Trunk/branch/twig text
        traj.generated_text = full_text[len(formatted) :]  # Model-generated text
        traj.prefill_length = prefill_length

        return traj

    def apply_chat_template(self, prompt: str) -> str:
        """Apply chat template if model is a chat model."""
        if not self._is_chat_model:
            return prompt
        tokenizer = self._tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    ##################
    #### Internal ####
    ##################

    def _detect_backend(self, model_name: str) -> ModelBackend:
        """Detect the appropriate backend based on model name."""
        name = model_name.lower()

        # OpenAI models
        openai_prefixes = ["openai", "gpt-4", "gpt-3", "o1", "o3"]
        if any(name.startswith(prefix) or name == prefix for prefix in openai_prefixes):
            return ModelBackend.OPENAI

        # Anthropic models
        anthropic_prefixes = ["anthropic", "claude"]
        if any(
            name.startswith(prefix) or name == prefix for prefix in anthropic_prefixes
        ):
            return ModelBackend.ANTHROPIC

        # Default to recommended backend
        return get_recommended_backend_inference()

    def _init_openai(self) -> None:
        """Initialize OpenAI backend."""
        # Extract model name (e.g., "openai/gpt-4o" -> "gpt-4o")
        model = self.model_name
        if "/" in model:
            model = model.split("/", 1)[1]
        elif model.lower() == "openai":
            model = "gpt-4o"  # Default to gpt-4o

        log(f"Using OpenAI API with model: {model}")
        self._backend = OpenAIBackend(self, model=model)

    def _init_anthropic(self) -> None:
        """Initialize Anthropic backend.

        Note: Anthropic API does not provide logprobs, so all trajectory
        logprobs will be 0.0. This backend is suitable for text generation
        and categorical judgments, but not for probability-weighted metrics.
        """
        # Extract model name (e.g., "anthropic/claude-sonnet-4-20250514" -> "claude-sonnet-4-20250514")
        model = self.model_name
        if "/" in model:
            model = model.split("/", 1)[1]
        elif model.lower() == "anthropic":
            model = None  # Use default

        log(f"Using Anthropic API with model: {model or 'claude-sonnet-4-20250514'}")
        log(
            "  Note: Anthropic API does not provide logprobs; trajectory logprobs will be 0.0"
        )
        self._backend = AnthropicBackend(self, model=model)

    def _init_huggingface(self) -> None:
        log(f"Loading {self.model_name} on {self.device} (HuggingFace)...")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=self.dtype
        ).to(self.device)
        self._model.eval()

        # Compile model for faster inference
        if self.device == "cuda":
            try:
                self._model = torch.compile(self._model)
                log("  torch.compile enabled")
            except Exception as e:
                log(f"  torch.compile failed: {e}")

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._backend = HuggingFaceBackend(self, tokenizer)

    def _init_mlx(self) -> None:
        from mlx_lm import load as mlx_load

        log(f"Loading {self.model_name} (MLX)...")
        self._model, tokenizer = mlx_load(self.model_name)
        self._backend = MLXBackend(self, tokenizer)

    def _detect_chat_model(self, model_name: str) -> bool:
        """Detect if model is a chat/instruct model based on name."""
        if not model_name:
            model_name = self.model_name
        name = model_name.lower()

        # Explicit base model indicators
        if any(x in name for x in ["-base", "_base"]):
            return False

        # API-based models are always chat models
        if any(x in name for x in ["claude", "anthropic", "gpt-4", "gpt-3", "openai"]):
            return True

        # Qwen3/Qwen3.5 models are instruct/reasoning by default (no base variant)
        if any(x in name for x in ["qwen3", "qwen-3", "qwen_3"]):
            return True

        # Explicit chat/instruct indicators
        return any(x in name for x in ["instruct", "chat", "-it", "rlhf"])

    def _detect_reasoning_model(self) -> bool:
        """Detect if model supports thinking/reasoning mode.

        Detection strategy:
        1. Check if chat_template contains thinking-related tokens (most reliable)
        2. Fall back to name heuristics, excluding known non-reasoning variants
        """
        name = self.model_name.lower()

        # Explicit non-reasoning model indicators
        non_reasoning_indicators = ["-2507", "_2507", "-base", "_base"]
        if any(ind in name for ind in non_reasoning_indicators):
            return False

        # Primary method: check chat_template for thinking tokens
        tokenizer = self._tokenizer
        if tokenizer is not None:
            chat_template = getattr(tokenizer, "chat_template", None)
            if chat_template:
                template_str = (
                    chat_template
                    if isinstance(chat_template, str)
                    else str(chat_template)
                )
                thinking_indicators = [
                    "<think>",
                    "</think>",
                    "enable_thinking",
                    "<|thinking|>",
                    "<reasoning>",
                ]
                if any(indicator in template_str for indicator in thinking_indicators):
                    return True

        # Name-based heuristics for known reasoning models
        reasoning_models = ["qwen3", "qwen-3", "qwen_3", "deepseek-r1", "o1", "o3"]
        return any(model in name for model in reasoning_models)

    @track_memory
    def cleanup(self) -> None:
        """Release model memory and clear GPU caches."""
        if self._model is not None:
            del self._model
            self._model = None
        if hasattr(self, "_backend"):
            del self._backend
        clear_gpu_memory()
