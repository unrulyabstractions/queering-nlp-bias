"""LLM provider clients for Anthropic, OpenAI, and HuggingFace with parallel processing support."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import anthropic
from anthropic import RateLimitError as AnthropicRateLimitError
import openai
from openai import RateLimitError as OpenAIRateLimitError
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from webapp.common.normativity_types import Scoring, parse_judge_score
from webapp.common.text_formatting_utils import format_logprob, truncate_for_log


# ════════════════════════════════════════════════════════════════════════════════
# HuggingFace Model Cache - Singleton pattern to avoid reloading models
# ════════════════════════════════════════════════════════════════════════════════

_huggingface_model_cache: dict[str, tuple[Any, Any]] = {}

# Skip thinking prefix for Qwen3.5 instruct models
SKIP_THINKING_PREFIX = "<think>\n</think>\n\n"

# Retry settings for rate limits
MAX_RETRIES = 5
BASE_RETRY_DELAY = 15  # seconds

# Rate limit exception types by provider
RATE_LIMIT_EXCEPTIONS = {
    "anthropic": AnthropicRateLimitError,
    "openai": OpenAIRateLimitError,
}


async def _retry_on_rate_limit(
    provider: str,
    api_call: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute an API call with exponential backoff retry on rate limits.

    Args:
        provider: The provider name ('anthropic' or 'openai')
        api_call: The synchronous API function to call
        *args, **kwargs: Arguments to pass to the API call

    Returns:
        The API response

    Raises:
        The original rate limit error if all retries are exhausted
    """
    rate_limit_error = RATE_LIMIT_EXCEPTIONS.get(provider)

    for attempt in range(MAX_RETRIES + 1):
        try:
            return await asyncio.to_thread(api_call, *args, **kwargs)
        except Exception as e:
            # Check if it's a rate limit error for this provider
            if rate_limit_error and isinstance(e, rate_limit_error):
                if attempt < MAX_RETRIES:
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    print(f"⚠️ {provider} RATE_LIMIT: {str(e)[:100]}")
                    print(f"⚠️ Retrying in {delay}s (attempt {attempt + 1}/{MAX_RETRIES})")
                    await asyncio.sleep(delay)
                else:
                    print(f"❌ {provider} RATE_LIMIT_EXCEEDED after {MAX_RETRIES} retries")
                    raise
            else:
                # Non-rate-limit error, don't retry
                raise


def _profile(label: str, start_time: float) -> None:
    """Print profiling info."""
    elapsed = time.time() - start_time
    print(f"⏱️  [{label}] {elapsed:.3f}s")


# ════════════════════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════════════════════

# Judge text truncation (prevents overly long inputs)
MAX_JUDGE_TEXT_LENGTH = 1500

# Judge response tokens
JUDGE_MAX_TOKENS = 100

# Logging truncation widths
LOG_TRUNCATE_DEFAULT = 80
LOG_TRUNCATE_PROMPT = 100
LOG_TRUNCATE_PREFILL = 60
LOG_TRUNCATE_RESPONSE = 30


def _log_generation_call(provider: str, model: str, prompt: str, prefill: str) -> None:
    """Log a generation API call."""
    print("\n" + "-" * 60)
    print("LLM GENERATION CALL")
    print("-" * 60)
    print(f"  Provider: {provider}")
    print(f"  Model: {model}")
    print(f"  Prompt: {truncate_for_log(prompt, LOG_TRUNCATE_PROMPT)}")
    if prefill:
        print(f"  Prefill: {truncate_for_log(prefill, LOG_TRUNCATE_PREFILL)}")


def _log_generation_result(result: str, logprob: float | None = None) -> None:
    """Log a generation result with FULL details."""
    print("=" * 60)
    print("  ██ GENERATION RESULT ██")
    print("=" * 60)
    print(f"  Length: {len(result)} chars, {len(result.split())} words")
    if logprob is not None:
        print(f"  Sum logprob: {logprob:.4f}")
        print(f"  Perplexity: {2 ** (-logprob / max(1, len(result.split()))):.2f}")
    print()
    print("  ┌" + "─" * 56 + "┐")
    print("  │ FULL GENERATED TEXT:".ljust(58) + "│")
    print("  ├" + "─" * 56 + "┤")
    for line in result.split("\n"):
        # Wrap long lines
        while len(line) > 54:
            print(f"  │ {line[:54]} │")
            line = line[54:]
        print(f"  │ {line.ljust(54)} │")
    print("  └" + "─" * 56 + "┘")
    print("=" * 60)


def _log_judge_call(
    provider: str,
    model: str,
    text: str,
    question: str,
    formatted_prompt: str,
    call_type: str = "JUDGE",
) -> None:
    """Log a judge API call."""
    print("\n" + "-" * 60)
    print(f"LLM {call_type} CALL")
    print("-" * 60)
    print(f"  Provider: {provider}")
    print(f"  Model: {model}")
    print(f"  Question: {truncate_for_log(question, 80)}")
    print(f"  Text: {truncate_for_log(text, 100)}")
    print("  Formatted prompt sent to API:")
    for line in formatted_prompt.split("\n"):
        print(f"    | {line}")


def _log_judge_result(
    score: Scoring, raw_response: str, logprob: Scoring | None = None
) -> None:
    """Log a judge result with FULL details."""
    print("  ┌─────────────────────────────────────┐")
    print("  │ JUDGE RESULT                        │")
    print("  ├─────────────────────────────────────┤")
    print(f"  │ Raw response: '{raw_response[:LOG_TRUNCATE_RESPONSE]}'".ljust(40) + "│")
    print(f"  │ ★ SCORE: {score:.4f}".ljust(40) + "│")
    if logprob is not None:
        print(f"  │ Logprob: {logprob:.4f}".ljust(40) + "│")
        print(f"  │ Confidence: {100 * (2 ** logprob):.1f}%".ljust(40) + "│")
    print("  └─────────────────────────────────────┘")


@dataclass
class GenerationResult:
    """Result from generating text, optionally with logprobs."""

    text: str
    logprob: float | None = None  # Sum of token logprobs (OpenAI only)


@dataclass
class JudgeResult:
    """Result from judging a text against a question."""

    score: Scoring
    raw_response: str
    logprob: Scoring | None = None


# ════════════════════════════════════════════════════════════════════════════════
# Client Factory
# ════════════════════════════════════════════════════════════════════════════════


def get_huggingface_model(model_name: str) -> tuple[Any, Any]:
    """Load or retrieve cached HuggingFace model and tokenizer.

    Returns:
        Tuple of (model, tokenizer)
    """
    if model_name in _huggingface_model_cache:
        print(f"▓ Using cached HuggingFace model: {model_name}")
        return _huggingface_model_cache[model_name]

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

    _huggingface_model_cache[model_name] = (model, tokenizer)
    print(f"▓ Model loaded successfully: {model_name}")

    return model, tokenizer


def is_base_model(model_name: str) -> bool:
    """Check if model is a base model (not instruct/chat)."""
    name_lower = model_name.lower()
    return "-base" in name_lower or "_base" in name_lower


def get_client(provider: str, api_key: str) -> Any:
    """Create client for the specified provider.

    For HuggingFace, api_key is ignored (uses local models).
    """
    if provider == "openai":
        return openai.OpenAI(api_key=api_key)
    if provider == "huggingface":
        # HuggingFace doesn't need a client object, return None
        # The model is loaded separately via get_huggingface_model
        return None
    return anthropic.Anthropic(api_key=api_key)


# ════════════════════════════════════════════════════════════════════════════════
# Anthropic Completion
# ════════════════════════════════════════════════════════════════════════════════


async def generate_from_llm_anthropic(
    client: Any,
    model: str,
    prompt: str,
    prefill: str = "",
    max_tokens: int = 300,
    temperature: float = 1.0,
) -> GenerationResult:
    """Generate text continuation using Anthropic API.

    Prefill: Add assistant message to prime the response.
    The API returns only the continuation, so we prepend the prefill.
    Note: Anthropic doesn't provide logprobs in the same way as OpenAI.
    """
    _log_generation_call("anthropic", model, prompt, prefill)

    messages = [{"role": "user", "content": prompt}]
    if prefill:
        # Anthropic API rejects assistant messages ending with whitespace
        prefill_stripped = prefill.rstrip()
        if prefill_stripped:
            messages.append({"role": "assistant", "content": prefill_stripped})

    response = await _retry_on_rate_limit(
        "anthropic",
        client.messages.create,
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        temperature=temperature,
    )

    continuation = response.content[0].text if response.content else ""
    result = prefill + continuation
    _log_generation_result(result, None)
    return GenerationResult(text=result, logprob=None)


# ════════════════════════════════════════════════════════════════════════════════
# OpenAI Completion
# ════════════════════════════════════════════════════════════════════════════════

OPENAI_PREFILL_INSTRUCTION = (
    "Continue from the following text exactly as written, without repeating it. "
    "Your response must seamlessly continue from this starting point:\n\n{prefill}"
)


async def generate_from_llm_openai(
    client: Any,
    model: str,
    prompt: str,
    prefill: str = "",
    max_tokens: int = 300,
    temperature: float = 1.0,
) -> GenerationResult:
    """Generate text continuation using OpenAI API with logprobs.

    For prefill, we include it in the user message as context
    since OpenAI doesn't support true prefill like Anthropic.
    Returns GenerationResult with text and sum of token logprobs.
    """
    _log_generation_call("openai", model, prompt, prefill)

    full_prompt = prompt
    if prefill:
        instruction = OPENAI_PREFILL_INSTRUCTION.format(prefill=prefill)
        full_prompt = f"{prompt}\n\n{instruction}"

    print(f"█ OpenAI gen: model={model} prompt_len={len(full_prompt)}")

    api_start = time.time()
    response = await _retry_on_rate_limit(
        "openai",
        client.chat.completions.create,
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=True,
    )
    _profile("OpenAI gen API", api_start)

    content = response.choices[0].message.content or ""
    result_text = prefill + content

    # Extract logprobs (minimal logging)
    logprob = None
    choice = response.choices[0]
    tokens_count = 0
    if choice.logprobs and choice.logprobs.content:
        logprob = sum(t.logprob for t in choice.logprobs.content)
        tokens_count = len(choice.logprobs.content)

    usage = response.usage
    lp_str = f"{logprob:.4f}" if logprob is not None else "N/A"
    usage_str = str(usage.total_tokens) if usage else "N/A"
    print(f"▓ Response: {tokens_count} tokens, logprob={lp_str}, usage={usage_str}")

    _log_generation_result(result_text, logprob)
    return GenerationResult(text=result_text, logprob=logprob)


# ════════════════════════════════════════════════════════════════════════════════
# HuggingFace Completion
# ════════════════════════════════════════════════════════════════════════════════


def _apply_chat_template_for_generation(
    tokenizer: Any, prompt: str, prefill: str = ""
) -> str:
    """Apply chat template for instruct models, adding skip_thinking prefix.

    For Qwen3.5 instruct models, we need to:
    1. Apply the chat template
    2. Prefill with <think>\n</think>\n\n to skip thinking mode
    """
    if hasattr(tokenizer, "apply_chat_template"):
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        formatted = prompt

    # Add skip_thinking prefix for reasoning models, then any user prefill
    return formatted + SKIP_THINKING_PREFIX + prefill


def _compute_huggingface_logprob(
    outputs: Any, generated_ids: Any
) -> float | None:
    """Compute sum of logprobs from HuggingFace generation output scores.

    Args:
        outputs: The generation output with .scores attribute
        generated_ids: Tensor of generated token IDs

    Returns:
        Sum of logprobs, or None if scores are not available
    """
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


def _generate_huggingface_sync(
    model: Any,
    tokenizer: Any,
    input_text: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, float | None]:
    """Synchronous HuggingFace generation with logprobs.

    Returns tuple of (generated_text, sum_logprob).
    """
    device = next(model.parameters()).device

    # Encode input
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]

    # Generate with sampling
    with torch.inference_mode():
        if temperature == 0.0:
            # Greedy decoding
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        else:
            # Temperature sampling
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

    # Decode only generated tokens
    generated_ids = outputs.sequences[0, input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    logprob = _compute_huggingface_logprob(outputs, generated_ids)
    return generated_text, logprob


async def generate_from_llm_huggingface(
    model_name: str,
    prompt: str,
    prefill: str = "",
    max_tokens: int = 300,
    temperature: float = 1.0,
) -> GenerationResult:
    """Generate text continuation using HuggingFace transformers.

    For instruct models (non-base), applies chat template and skip_thinking prefix.
    For base models, does direct text completion.
    """
    _log_generation_call("huggingface", model_name, prompt, prefill)

    model, tokenizer = get_huggingface_model(model_name)

    # Prepare input text based on model type
    if is_base_model(model_name):
        # Base model: direct text continuation
        input_text = prompt + prefill
        print(f"█ HuggingFace base: model={model_name} input_len={len(input_text)}")
    else:
        # Instruct model: apply chat template + skip_thinking + prefill
        input_text = _apply_chat_template_for_generation(tokenizer, prompt, prefill)
        print(f"█ HuggingFace instruct: model={model_name} input_len={len(input_text)}")

    api_start = time.time()

    # Run generation in thread to avoid blocking
    generated_text, logprob = await asyncio.to_thread(
        _generate_huggingface_sync,
        model,
        tokenizer,
        input_text,
        max_tokens,
        temperature,
    )

    _profile("HuggingFace gen", api_start)

    # Result includes prefill + generated for consistency with other providers
    result_text = prefill + generated_text

    lp_str = f"{logprob:.4f}" if logprob is not None else "N/A"
    print(f"▓ Response: {len(generated_text.split())} words, logprob={lp_str}")

    _log_generation_result(result_text, logprob)
    return GenerationResult(text=result_text, logprob=logprob)


# ════════════════════════════════════════════════════════════════════════════════
# Unified Completion Interface
# ════════════════════════════════════════════════════════════════════════════════


async def generate_from_llm(
    client: Any,
    provider: str,
    model: str,
    prompt: str,
    prefill: str = "",
    max_tokens: int = 300,
    temperature: float = 1.0,
) -> GenerationResult:
    """Generate text continuation. Routes to provider-specific implementation.

    Returns GenerationResult with text and optional logprobs (OpenAI and HuggingFace).
    """
    if provider == "openai":
        return await generate_from_llm_openai(
            client, model, prompt, prefill, max_tokens, temperature
        )
    if provider == "huggingface":
        return await generate_from_llm_huggingface(
            model, prompt, prefill, max_tokens, temperature
        )
    return await generate_from_llm_anthropic(
        client, model, prompt, prefill, max_tokens, temperature
    )


# ════════════════════════════════════════════════════════════════════════════════
# Anthropic Judge
# ════════════════════════════════════════════════════════════════════════════════


async def llm_judge_anthropic(
    client: Any,
    model: str,
    text: str,
    question: str,
    judge_prompt: str,
    temperature: float = 0.0,
) -> JudgeResult:
    """Score text using Anthropic API."""
    formatted_prompt = judge_prompt.format(text=text[:MAX_JUDGE_TEXT_LENGTH], question=question)
    _log_judge_call("anthropic", model, text, question, formatted_prompt)

    response = await _retry_on_rate_limit(
        "anthropic",
        client.messages.create,
        model=model,
        max_tokens=JUDGE_MAX_TOKENS,
        messages=[{"role": "user", "content": formatted_prompt}],
        temperature=temperature,
    )

    answer = response.content[0].text if response.content else ""
    score = parse_judge_score(answer)

    _log_judge_result(score, answer.strip())
    return JudgeResult(score=score, raw_response=answer.strip(), logprob=None)


# ════════════════════════════════════════════════════════════════════════════════
# OpenAI Judge
# ════════════════════════════════════════════════════════════════════════════════


async def llm_judge_openai(
    client: Any,
    model: str,
    text: str,
    question: str,
    judge_prompt: str,
    temperature: float = 0.0,
) -> JudgeResult:
    """Score text using OpenAI API. Includes logprob extraction."""
    formatted_prompt = judge_prompt.format(text=text[:MAX_JUDGE_TEXT_LENGTH], question=question)

    response = await _retry_on_rate_limit(
        "openai",
        client.chat.completions.create,
        model=model,
        messages=[{"role": "user", "content": formatted_prompt}],
        max_tokens=JUDGE_MAX_TOKENS,
        temperature=temperature,
        logprobs=True,
        top_logprobs=5,
    )

    answer = response.choices[0].message.content or ""
    logprob = None
    choice = response.choices[0]
    if choice.logprobs and choice.logprobs.content:
        logprob = sum(t.logprob for t in choice.logprobs.content)

    score = parse_judge_score(answer)
    return JudgeResult(score=score, raw_response=answer.strip(), logprob=logprob)


# ════════════════════════════════════════════════════════════════════════════════
# HuggingFace Judge
# ════════════════════════════════════════════════════════════════════════════════


def _judge_huggingface_sync(
    model: Any,
    tokenizer: Any,
    input_text: str,
    is_base: bool,
) -> tuple[str, float | None]:
    """Synchronous HuggingFace judge with logprobs.

    Returns tuple of (answer_text, sum_logprob).
    """
    device = next(model.parameters()).device

    # Encode input
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]

    # Generate with greedy decoding (temperature=0)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=JUDGE_MAX_TOKENS,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Decode only generated tokens
    generated_ids = outputs.sequences[0, input_length:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

    logprob = _compute_huggingface_logprob(outputs, generated_ids)
    return answer, logprob


async def llm_judge_huggingface(
    model_name: str,
    text: str,
    question: str,
    judge_prompt: str,
    temperature: float = 0.0,
) -> JudgeResult:
    """Score text using HuggingFace model. Includes logprob extraction."""
    formatted_prompt = judge_prompt.format(text=text[:MAX_JUDGE_TEXT_LENGTH], question=question)
    _log_judge_call("huggingface", model_name, text, question, formatted_prompt)

    model, tokenizer = get_huggingface_model(model_name)
    is_base = is_base_model(model_name)

    # Prepare input text based on model type
    if is_base:
        # Base model: direct text completion
        input_text = formatted_prompt
    else:
        # Instruct model: apply chat template + skip_thinking
        input_text = _apply_chat_template_for_generation(tokenizer, formatted_prompt, "")

    # Run generation in thread to avoid blocking
    answer, logprob = await asyncio.to_thread(
        _judge_huggingface_sync,
        model,
        tokenizer,
        input_text,
        is_base,
    )

    score = parse_judge_score(answer)
    _log_judge_result(score, answer.strip(), logprob)
    return JudgeResult(score=score, raw_response=answer.strip(), logprob=logprob)


# ════════════════════════════════════════════════════════════════════════════════
# Unified Judge Interface
# ════════════════════════════════════════════════════════════════════════════════


async def llm_judge(
    client: Any,
    provider: str,
    model: str,
    text: str,
    question: str,
    judge_prompt: str,
    temperature: float = 0.0,
) -> JudgeResult:
    """Score text against question. Routes to provider-specific implementation."""
    if provider == "openai":
        return await llm_judge_openai(
            client, model, text, question, judge_prompt, temperature
        )
    if provider == "huggingface":
        return await llm_judge_huggingface(
            model, text, question, judge_prompt, temperature
        )
    return await llm_judge_anthropic(
        client, model, text, question, judge_prompt, temperature
    )


async def judge_all_questions(
    client: Any,
    provider: str,
    model: str,
    text: str,
    questions: list[str],
    judge_prompt: str,
) -> list[JudgeResult]:
    """Judge text against all questions sequentially to avoid rate limits."""
    results = []
    for i, q in enumerate(questions):
        result = await llm_judge(client, provider, model, text, q, judge_prompt)
        results.append(result)
        # Minimal delay between judge calls (retry logic handles rate limits)
        if i < len(questions) - 1:
            await asyncio.sleep(0.1)
    return results
