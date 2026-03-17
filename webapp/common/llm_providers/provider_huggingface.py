"""HuggingFace provider: local generation and judging with transformers models."""

from __future__ import annotations

import asyncio
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from webapp.common.normativity_types import parse_judge_score

from .provider_base import (
    GenerationResult,
    JudgeResult,
    JUDGE_MAX_TOKENS,
    format_judge_prompt,
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

    tokenizer: HFTokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model: HFModel = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device, trust_remote_code=True,
    )
    model.eval()

    _model_cache[model_name] = (model, tokenizer)
    return model, tokenizer


def is_base_model(model_name: str) -> bool:
    name = model_name.lower()
    return "-base" in name or "_base" in name


def _apply_chat_template(tokenizer: HFTokenizer, prompt: str, prefill: str = "") -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
        )
    else:
        formatted = prompt
    return formatted + SKIP_THINKING_PREFIX + prefill


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
    model: HFModel, tokenizer: HFTokenizer, input_text: str, max_tokens: int, temperature: float,
) -> tuple[str, float | None]:
    device = next(model.parameters()).device
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else None,
            output_scores=True, return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    gen_ids = outputs.sequences[0, input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True), _compute_logprob(outputs, gen_ids)


async def generate_huggingface(
    model_name: str, prompt: str, prefill: str = "",
    max_tokens: int = 300, temperature: float = 1.0,
) -> GenerationResult:
    log_generation_call("huggingface", model_name, prompt, prefill)
    model, tokenizer = get_huggingface_model(model_name)
    input_text = prompt + prefill if is_base_model(model_name) else _apply_chat_template(tokenizer, prompt, prefill)

    api_start = time.time()
    generated_text, logprob = await asyncio.to_thread(
        _generate_sync, model, tokenizer, input_text, max_tokens, temperature
    )
    profile("HuggingFace gen", api_start)

    result = prefill + generated_text
    log_generation_result(result, logprob)
    return GenerationResult(text=result, logprob=logprob)


async def judge_huggingface(
    model_name: str, text: str, question: str,
    judge_prompt: str, temperature: float = 0.0,
) -> JudgeResult:
    formatted = format_judge_prompt(judge_prompt, text, question)
    log_judge_call("huggingface", model_name, question)

    model, tokenizer = get_huggingface_model(model_name)
    input_text = formatted if is_base_model(model_name) else _apply_chat_template(tokenizer, formatted, "")

    answer, logprob = await asyncio.to_thread(
        _generate_sync, model, tokenizer, input_text, JUDGE_MAX_TOKENS, 0.0
    )

    score = parse_judge_score(answer)
    log_judge_result(score, answer.strip(), logprob)
    return JudgeResult(score=score, raw_response=answer.strip(), logprob=logprob)
