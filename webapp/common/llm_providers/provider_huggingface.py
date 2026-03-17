"""HuggingFace provider: local generation and judging with transformers models."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from webapp.common.normativity_types import parse_judge_score

from .provider_base import (
    GenerationResult,
    JudgeResult,
    JUDGE_MAX_TOKENS,
    MAX_JUDGE_TEXT_LENGTH,
    log_generation_call,
    log_generation_result,
    log_judge_call,
    log_judge_result,
    profile,
)


# Skip thinking prefix for Qwen3.5 instruct models
SKIP_THINKING_PREFIX = "<think>\n</think>\n\n"

# Model cache to avoid reloading
_model_cache: dict[str, tuple[Any, Any]] = {}


def get_huggingface_model(model_name: str) -> tuple[Any, Any]:
    """Load or retrieve cached HuggingFace model and tokenizer."""
    if model_name in _model_cache:
        print(f"▓ Using cached HuggingFace model: {model_name}")
        return _model_cache[model_name]

    print(f"▓ Loading HuggingFace model: {model_name}...")

    # Determine device and dtype
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"▓ Using device: {device}, dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    _model_cache[model_name] = (model, tokenizer)
    print(f"▓ Model loaded successfully: {model_name}")

    return model, tokenizer


def is_base_model(model_name: str) -> bool:
    """Check if model is a base model (not instruct/chat)."""
    name_lower = model_name.lower()
    return "-base" in name_lower or "_base" in name_lower


def _apply_chat_template(tokenizer: Any, prompt: str, prefill: str = "") -> str:
    """Apply chat template for instruct models with skip_thinking prefix."""
    if hasattr(tokenizer, "apply_chat_template"):
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        formatted = prompt
    return formatted + SKIP_THINKING_PREFIX + prefill


def _compute_logprob(outputs: Any, generated_ids: Any) -> float | None:
    """Compute sum of logprobs from HuggingFace generation output scores."""
    if not outputs.scores:
        return None
    logprob = 0.0
    for i, score in enumerate(outputs.scores):
        if i >= len(generated_ids):
            break
        token_id = generated_ids[i].item()
        log_softmax = torch.nn.functional.log_softmax(score[0], dim=-1)
        logprob += log_softmax[token_id].item()
    return logprob


def _generate_sync(
    model: Any,
    tokenizer: Any,
    input_text: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, float | None]:
    """Synchronous HuggingFace generation with logprobs."""
    device = next(model.parameters()).device
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]

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

    generated_ids = outputs.sequences[0, input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    logprob = _compute_logprob(outputs, generated_ids)

    return generated_text, logprob


async def generate_huggingface(
    model_name: str,
    prompt: str,
    prefill: str = "",
    max_tokens: int = 300,
    temperature: float = 1.0,
) -> GenerationResult:
    """Generate text continuation using HuggingFace transformers."""
    log_generation_call("huggingface", model_name, prompt, prefill)

    model, tokenizer = get_huggingface_model(model_name)

    if is_base_model(model_name):
        input_text = prompt + prefill
        print(f"█ HuggingFace base: model={model_name} input_len={len(input_text)}")
    else:
        input_text = _apply_chat_template(tokenizer, prompt, prefill)
        print(f"█ HuggingFace instruct: model={model_name} input_len={len(input_text)}")

    api_start = time.time()
    generated_text, logprob = await asyncio.to_thread(
        _generate_sync, model, tokenizer, input_text, max_tokens, temperature
    )
    profile("HuggingFace gen", api_start)

    result_text = prefill + generated_text
    lp_str = f"{logprob:.4f}" if logprob is not None else "N/A"
    print(f"▓ Response: {len(generated_text.split())} words, logprob={lp_str}")

    log_generation_result(result_text, logprob)
    return GenerationResult(text=result_text, logprob=logprob)


def _judge_sync(model: Any, tokenizer: Any, input_text: str) -> tuple[str, float | None]:
    """Synchronous HuggingFace judge with logprobs."""
    device = next(model.parameters()).device
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=JUDGE_MAX_TOKENS,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated_ids = outputs.sequences[0, input_length:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    logprob = _compute_logprob(outputs, generated_ids)

    return answer, logprob


async def judge_huggingface(
    model_name: str,
    text: str,
    question: str,
    judge_prompt: str,
    temperature: float = 0.0,
) -> JudgeResult:
    """Score text using HuggingFace model."""
    formatted_prompt = judge_prompt.format(text=text[:MAX_JUDGE_TEXT_LENGTH], question=question)
    log_judge_call("huggingface", model_name, text, question, formatted_prompt)

    model, tokenizer = get_huggingface_model(model_name)

    if is_base_model(model_name):
        input_text = formatted_prompt
    else:
        input_text = _apply_chat_template(tokenizer, formatted_prompt, "")

    answer, logprob = await asyncio.to_thread(_judge_sync, model, tokenizer, input_text)

    score = parse_judge_score(answer)
    log_judge_result(score, answer.strip(), logprob)
    return JudgeResult(score=score, raw_response=answer.strip(), logprob=logprob)
