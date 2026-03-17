"""HuggingFace provider: local generation and judging with transformers models."""

from __future__ import annotations

import asyncio
import gc
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from webapp.common.normativity_types import parse_judge_score

from .provider_base import (
    JUDGE_MAX_TOKENS,
    GenerationResult,
    JudgeResult,
    format_judge_prompt,
    get_max_tokens_for_model,
    log_generation_call,
    log_generation_result,
    log_judge_call,
    log_judge_result,
    profile,
)

# Type aliases for HuggingFace
HFModel = PreTrainedModel
HFTokenizer = PreTrainedTokenizerBase

SKIP_THINKING_PREFIX = "<think>\n</think>\n\n"
_model_cache: dict[str, tuple[HFModel, HFTokenizer]] = {}

# Serial execution lock for local models (prevents memory explosion)
_hf_lock = asyncio.Lock()


def _is_qwen35_model(model_name: str) -> bool:
    """Check if model is Qwen 3.5 (uses enable_thinking param instead of prefix hack)."""
    return "Qwen3.5" in model_name or "Qwen/Qwen3.5" in model_name


def clear_gpu_memory() -> None:
    """Clear GPU memory caches for CUDA, MPS, and MLX."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    # Clear MLX memory if available
    try:
        import mlx.core as mx

        mx.clear_cache()
    except (ImportError, AttributeError):
        pass


def _get_device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def get_huggingface_model(model_name: str) -> tuple[HFModel, HFTokenizer]:
    """Load or retrieve cached model and tokenizer."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    print(f"▓ Loading HuggingFace model: {model_name}...")
    device, dtype = _get_device_and_dtype()

    tokenizer: HFTokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    model: HFModel = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    _model_cache[model_name] = (model, tokenizer)
    return model, tokenizer


def _apply_chat_template(
    tokenizer: HFTokenizer,
    model_name: str,
    prompt: str,
    prefill: str = "",
    enable_thinking: bool = False,
) -> str:
    """Apply chat template with thinking mode control.

    For Qwen 3.5 models: uses enable_thinking parameter in apply_chat_template.
    For older models: uses <think></think> prefix hack to skip reasoning.

    Args:
        tokenizer: HuggingFace tokenizer
        model_name: Model name (to detect Qwen 3.5)
        prompt: User prompt
        prefill: Text to prefill the response with
        enable_thinking: If True, enable thinking/reasoning mode
    """
    is_qwen35 = _is_qwen35_model(model_name)

    if hasattr(tokenizer, "apply_chat_template"):
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        # Qwen 3.5 uses enable_thinking parameter
        if is_qwen35:
            template_kwargs["enable_thinking"] = enable_thinking
            print(f"🧠 Qwen 3.5 detected: enable_thinking={enable_thinking}")

        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            **template_kwargs,
        )
    else:
        formatted = prompt

    # For non-Qwen 3.5 models, use prefix hack to skip reasoning when not enabled
    if not is_qwen35 and not enable_thinking:
        formatted = formatted + SKIP_THINKING_PREFIX

    return formatted + prefill


def _compute_logprob(outputs: object, generated_ids: torch.Tensor) -> float | None:
    if not outputs.scores:
        return None
    logprob = 0.0
    for i, score in enumerate(outputs.scores):
        if i >= len(generated_ids):
            break
        log_softmax = torch.nn.functional.log_softmax(score[0], dim=-1)
        logprob += log_softmax[generated_ids[i].item()].item()
    return logprob


def _generate_sync(
    model: HFModel,
    tokenizer: HFTokenizer,
    input_text: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, float | None]:
    device = next(model.parameters()).device
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else None,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    gen_ids = outputs.sequences[0, input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True), _compute_logprob(
        outputs, gen_ids
    )


async def generate_huggingface(
    provider: str,
    model_name: str,
    prompt: str,
    prefill: str = "",
    max_tokens: int | None = 300,
    temperature: float = 1.0,
) -> GenerationResult:
    """Generate text with HuggingFace model.

    Runs in serial (one at a time) to prevent GPU memory explosion.

    Args:
        provider: Provider name (huggingface_base, huggingface_instruct, huggingface_reasoning)
        model_name: HuggingFace model name
        prompt: User prompt
        prefill: Text to prefill the response with
        max_tokens: Max tokens to generate (None = use model default)
        temperature: Sampling temperature
    """
    from webapp.app_settings import is_base_model, should_enable_thinking

    is_base = is_base_model(provider)
    enable_thinking = should_enable_thinking(provider)

    async with _hf_lock:
        log_generation_call(provider, model_name, prompt, prefill)
        model, tokenizer = get_huggingface_model(model_name)

        effective_max_tokens = get_max_tokens_for_model(model_name, max_tokens)

        if is_base:
            input_text = prompt + prefill
        else:
            input_text = _apply_chat_template(
                tokenizer, model_name, prompt, prefill, enable_thinking
            )

        api_start = time.time()
        generated_text, logprob = await asyncio.to_thread(
            _generate_sync,
            model,
            tokenizer,
            input_text,
            effective_max_tokens,
            temperature,
        )
        profile("HuggingFace gen", api_start)

        clear_gpu_memory()

        log_generation_result(generated_text, logprob)
        return GenerationResult(text=generated_text, logprob=logprob)


async def judge_huggingface(
    provider: str,
    model_name: str,
    text: str,
    question: str,
    judge_prompt: str,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> JudgeResult:
    """Judge text with HuggingFace model.

    Runs in serial (one at a time) to prevent GPU memory explosion.

    Args:
        provider: Provider name (huggingface_base, huggingface_instruct, huggingface_reasoning)
        model_name: HuggingFace model name
        text: Text to judge
        question: Question to answer about the text
        judge_prompt: Template for the judge prompt
        temperature: Sampling temperature
        max_tokens: Max tokens for response (None = use default)
    """
    from webapp.app_settings import is_base_model, should_enable_thinking

    is_base = is_base_model(provider)
    enable_thinking = should_enable_thinking(provider)
    tokens = max_tokens if max_tokens else JUDGE_MAX_TOKENS
    print(
        f"🔍 JUDGE HF: provider={provider}, enable_thinking={enable_thinking}, max_tokens={tokens}"
    )

    async with _hf_lock:
        formatted = format_judge_prompt(judge_prompt, text, question)
        log_judge_call(provider, model_name, question)

        model, tokenizer = get_huggingface_model(model_name)
        if is_base:
            input_text = formatted
        else:
            input_text = _apply_chat_template(
                tokenizer, model_name, formatted, "", enable_thinking
            )

        answer, logprob = await asyncio.to_thread(
            _generate_sync, model, tokenizer, input_text, tokens, temperature
        )

        # Debug: show raw answer
        print(f"🔍 JUDGE HF RAW ANSWER ({len(answer)} chars): {answer[:200]}...")

        # Clear GPU memory after judging
        clear_gpu_memory()

        score = parse_judge_score(answer)
        log_judge_result(score, answer.strip(), logprob)
        return JudgeResult(score=score, raw_response=answer.strip(), logprob=logprob)
