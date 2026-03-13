# Inference Module: Technical Specification

Detailed documentation for model inference, backend routing, and trajectory generation.

## Table of Contents

1. [ModelRunner Architecture](#modelrunner-architecture)
2. [Backend Routing](#backend-routing)
3. [Model Loading](#model-loading)
4. [GeneratedTrajectory](#generatedtrajectory)
5. [EmbeddingRunner](#embeddingrunner)

---

## ModelRunner Architecture

`ModelRunner` (`model_runner.py`) is the unified inference interface supporting multiple backends.

### Initialization

```python
ModelRunner(
    model_name: str,                  # Model identifier
    device: str | None = None,        # "cuda", "mps", "cpu" (auto-detected)
    dtype: torch.dtype | None = None, # float16 on GPU/MPS, float32 on CPU
    backend: ModelBackend | None = None,  # Auto-detected if None
)
```

### Model Properties

| Property | Type | Source |
|----------|------|--------|
| `device` | `str` | User-provided or auto-detected |
| `dtype` | `torch.dtype` | Depends on device (float16 for GPU/MPS, float32 for CPU) |
| `vocab_size` | `int` | From tokenizer |
| `n_layers` | `int` | From backend (0 for API) |
| `d_model` | `int` | From backend (0 for API) |
| `bos_token_id`, `eos_token_id` | `int \| None` | From tokenizer |
| `is_reasoning_model` | `bool` | Auto-detected from chat template or name |
| `skip_thinking_prefix` | `str` | `"<think>\n</think>\n\n"` for reasoning models, else `""` |

### Auto-Detection

**Chat Model Detection**:
- Excludes base models (`-base`, `_base`)
- Includes API models (`claude`, `gpt-4`, etc.)
- Includes Qwen3 models (reasoning by default)
- Includes instruct patterns (`instruct`, `chat`, `-it`, `rlhf`)

**Reasoning Model Detection**:
1. Check tokenizer's `chat_template` for thinking tokens (`<think>`, `</think>`, etc.)
2. Fall back to name heuristics (`qwen3`, `deepseek-r1`, `o1`, `o3`)
3. Exclude known non-reasoning variants (`-2507`, `-base`)

---

## Backend Routing

Backend selection in `_detect_backend()`:

1. **OpenAI**: Model name contains `openai/`, `gpt-4`, `gpt-3`, `o1`, `o3`
   - Calls `_init_openai()`, creates `OpenAIBackend`

2. **Anthropic**: Model name contains `anthropic/`, `claude`
   - Calls `_init_anthropic()`, creates `AnthropicBackend`

3. **Fallback**: Calls `get_recommended_backend_inference()`
   - If Apple Silicon + MLX available → `_init_mlx()`, creates `MLXBackend`
   - Otherwise → `_init_huggingface()`, creates `HuggingFaceBackend`

### Backend Enum

```python
class ModelBackend(Enum):
    MLX = "mlx"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
```

---

## Model Loading

Each backend initializes differently:

### HuggingFace (`_init_huggingface`)

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=dtype
).to(device)
model.eval()

# Attempt torch.compile on CUDA
if device == "cuda":
    model = torch.compile(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)
self._backend = HuggingFaceBackend(self, tokenizer)
```

### MLX (`_init_mlx`)

```python
from mlx_lm import load as mlx_load

model, tokenizer = mlx_load(model_name)
self._backend = MLXBackend(self, tokenizer)
```

### OpenAI (`_init_openai`)

```python
# Extract model name (e.g., "openai/gpt-4o" → "gpt-4o")
# Defaults to "gpt-4o" if model_name is just "openai"
self._backend = OpenAIBackend(self, model=model)
```

### Anthropic (`_init_anthropic`)

```python
# Extract model name from "anthropic/..." format
# Defaults to "claude-sonnet-4-20250514" if None
self._backend = AnthropicBackend(self, model=model)
# WARNING: Anthropic API does not provide logprobs
```

---

## Backend Interface

All backends inherit from `Backend` (abstract class in `model_backend.py`).

### Core Methods

| Method | Purpose |
|--------|---------|
| `get_tokenizer()` | Return tokenizer object |
| `get_n_layers()` | Number of transformer layers (0 for API) |
| `get_d_model()` | Hidden dimension (0 for API) |
| `encode(text, add_special_tokens, prepend_bos)` | Return token tensor |
| `decode(token_ids)` | Return decoded string |
| `forward(input_ids)` | Return logits tensor |
| `generate(prompt, max_new_tokens, temperature)` | Return generated string |
| `generate_trajectory(token_ids, max_new_tokens, temperature)` | Return (token_ids, logprobs) |

### Backend-Specific Notes

**HuggingFaceBackend**:
- Full forward pass access; supports KV caching
- Uses `model.generate()` with `output_scores=True` for logprobs

**MLXBackend**:
- Streams generation via `mlx_lm.stream_generate()`
- Extracts logprobs from `response.logprobs`

**OpenAIBackend**:
- Uses `tiktoken` with `o200k_base` encoding
- `forward()` raises `NotImplementedError`
- Limited logprobs (top-20 tokens only)

**AnthropicBackend**:
- Uses `tiktoken` with `cl100k_base` encoding (approximation)
- **Critical**: No logprobs provided; all logprob values are 0.0
- Suitable only for text generation and categorical judgments
- Uses `generate()` directly (no forward passes)

---

## GeneratedTrajectory

`GeneratedTrajectory` extends `TokenTrajectory` with model internals support.

```python
@dataclass
class GeneratedTrajectory(TokenTrajectory):
    internals: dict = field(default_factory=dict)
```

### Fields (inherited from TokenTrajectory)

- `token_ids: list[int]` - Full sequence
- `logprobs: list[float]` - Log probability per token (first token is 0.0)
- `logits: list[float]` - Scalar logit per token
- `full_logits: torch.Tensor | None` - Full vocab logits `[seq_len, vocab_size]`

### Factory Methods

**`from_inference(token_ids, logits, device, internals)`**:
- Builds from forward pass outputs
- Computes logprobs via `log_softmax` on logits
- First token: logprob=0.0, logit=0.0

**`from_logprobs(token_ids, logprobs)`**:
- Builds from logprobs only (no full logits)
- Sets `full_logits=None`

**`from_token_trajectory(trajectory, internals)`**:
- Upgrades `TokenTrajectory` to `GeneratedTrajectory`

### Methods

- `can_have_internals()` → True
- `has_internals()` → bool
- `has_internals_for(names_filter)` → bool (check specific keys)
- `load_internals_from_disk(path)` → Load from file
- `pop_heavy()` → Clear internals for memory

---

## ModelRunner Generation Methods

**`generate(prompt, max_new_tokens, temperature, prefilling)`**:
- Returns generated string only

**`generate_trajectory(token_ids, max_new_tokens, temperature)`**:
- Takes token IDs, returns `GeneratedTrajectory`
- Delegates to backend's `generate_trajectory()`

**`generate_trajectory_from_prompt(prompt, max_new_tokens, temperature, prefilling)`**:
- Takes text prompt, applies chat template, generates trajectory
- Sets trajectory fields: `prefill_text`, `generated_text`, `prefill_length`

### Token Operations

**Encoding**:
```python
encode(text, add_special_tokens=True, prepend_bos=False) → torch.Tensor [1, seq_len]
encode_ids(text, add_special_tokens=True, prepend_bos=False) → list[int]
```

**Decoding**:
```python
decode(token_ids: torch.Tensor) → str
decode_ids(token_ids: list[int]) → str
```

**Chat Template**:
```python
apply_chat_template(prompt: str) → str
# For chat models, applies tokenizer's template; otherwise returns prompt unchanged
```

### Batch Processing

```python
calculate_trajectories_for_batch(
    token_ids_batch: list[list[int]],
    logits_batch: torch.Tensor,  # [batch, max_seq_len, vocab_size]
    device: str = "cpu"
) → list[GeneratedTrajectory]
```

Handles variable-length sequences by trimming padding.

---

## EmbeddingRunner

`EmbeddingRunner` provides text embeddings using sentence-transformers.

### Initialization

```python
EmbeddingRunner(model_name: str = EMBEDDING_MODEL)
```

Suppresses stdout/stderr during loading to avoid "LOAD REPORT" noise.

### Methods

| Method | Returns | Purpose |
|--------|---------|---------|
| `embed(texts: list[str])` | NDArray[np.float32] shape `(len(texts), embedding_dim)` | Batch embedding |
| `embed_single(text: str)` | NDArray[np.float32] shape `(embedding_dim,)` | Single embedding |
| `similarity(text, reference)` | float in [0, 1] | Cosine similarity (clamped) |
| `similarities(text, references)` | list[float] | One-to-many similarities |

Cosine similarity is computed as `(dot / norm + 1) / 2` to map [-1, 1] → [0, 1].
