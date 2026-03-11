# Inference Module: In-Depth Specification

> **Note**: This documentation was AI-generated and may contain errors. If something seems off, check the code or open an issue.


This document provides detailed technical documentation for the inference module.

## Table of Contents

1. [ModelRunner Architecture](#modelrunner-architecture)
2. [Backend Abstraction](#backend-abstraction)
3. [Trajectory Generation API](#trajectory-generation-api)
4. [Token-Level Operations](#token-level-operations)
5. [Embedding Runner](#embedding-runner)

---

## ModelRunner Architecture

`ModelRunner` (`model_runner.py`) is the primary interface for model inference. It provides a unified API regardless of the underlying backend.

### Initialization

```python
ModelRunner(
    model_name: str,           # Model identifier (HF repo, API model name)
    device: str | None,        # "cuda", "mps", "cpu" (auto-detected if None)
    dtype: torch.dtype | None, # float16 on GPU/MPS, float32 on CPU
    backend: ModelBackend | None,  # Auto-detected if None
)
```

Backend detection priority:
1. If `backend` is explicitly provided, use it
2. If model name starts with `openai/`, `gpt-4`, `gpt-3`, `o1`, `o3` -> OpenAI
3. If model name starts with `anthropic/`, `claude` -> Anthropic
4. Otherwise: call `get_recommended_backend_inference()`
   - Apple Silicon + MLX available -> MLX
   - Else -> HuggingFace

### Model Properties

| Property | Type | Description |
|----------|------|-------------|
| `device` | `str` | Device the model runs on |
| `dtype` | `torch.dtype` | Model precision |
| `vocab_size` | `int` | Vocabulary size |
| `n_layers` | `int` | Number of transformer layers (0 for API backends) |
| `d_model` | `int` | Hidden dimension (0 for API backends) |
| `bos_token_id` | `int \| None` | Beginning-of-sequence token ID |
| `eos_token_id` | `int \| None` | End-of-sequence token ID |
| `is_reasoning_model` | `bool` | Whether model uses thinking tokens |
| `skip_thinking_prefix` | `str` | `"<think>\n</think>\n\n"` or `""` |

### Chat Model Detection

`ModelRunner` automatically detects chat/instruct models based on name patterns:
- Explicit base indicators (`-base`, `_base`) -> base model
- API models (`claude`, `gpt-4`, etc.) -> chat model
- Qwen3/Qwen3.5 models -> chat model (reasoning by default)
- Instruct indicators (`instruct`, `chat`, `-it`, `rlhf`) -> chat model

### Reasoning Model Detection

Detection strategy:
1. Check tokenizer's `chat_template` for thinking-related tokens (`<think>`, `</think>`, `enable_thinking`, etc.)
2. Fall back to name heuristics (`qwen3`, `deepseek-r1`, `o1`, `o3`)
3. Exclude known non-reasoning variants (`-2507`, `-base`)

---

## Backend Abstraction

All backends implement the `Backend` abstract base class (`backends/model_backend.py`).

### Backend Enum

```python
class ModelBackend(Enum):
    MLX = "mlx"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
```

### Abstract Interface

```python
class Backend(ABC):
    supports_inference_mode: bool = True  # False for API backends

    def __init__(self, runner: ModelRunner): ...

    # Model info
    def get_tokenizer(self) -> Any: ...
    def get_n_layers(self) -> int: ...
    def get_d_model(self) -> int: ...

    # Tokenization
    def encode(self, text: str, add_special_tokens: bool = True,
               prepend_bos: bool = False) -> torch.Tensor: ...
    def decode(self, token_ids: torch.Tensor) -> str: ...

    # Generation
    def generate(self, prompt: str, max_new_tokens: int,
                 temperature: float, past_kv_cache: Any = None) -> str: ...
    def generate_trajectory(self, token_ids: list[int], max_new_tokens: int,
                           temperature: float) -> tuple[list[int], list[float]]: ...

    # Forward pass
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor: ...

    # Token probabilities
    def get_next_token_probs(self, prompt: str, target_tokens: Sequence[str],
                             past_kv_cache: Any = None) -> dict[str, float]: ...
    def get_next_token_probs_by_id(self, prompt: str, token_ids: Sequence[int],
                                   past_kv_cache: Any = None) -> dict[int, float]: ...
```

### Backend Implementations

#### HuggingFaceBackend

Uses `transformers.AutoModelForCausalLM` and `AutoTokenizer`.

Key features:
- Full forward pass access via `model(input_ids)`
- KV caching via `model.generate(use_cache=True)`
- Score output for logprob extraction: `model.generate(output_scores=True)`
- `torch.compile` on CUDA for faster inference

Trajectory generation:
1. Run `model.generate()` with `return_dict_in_generate=True, output_scores=True`
2. Compute prefix logprobs via separate forward pass
3. Extract generated token logprobs from `outputs.scores`

#### MLXBackend

Uses `mlx_lm` for Apple Silicon optimization.

Key features:
- Metal acceleration via MLX framework
- Streaming generation via `mlx_lm.stream_generate`
- Lazy-loaded to avoid import errors on non-Apple platforms

Trajectory generation:
1. Compute prefix logprobs via forward pass
2. Stream generate with `stream_generate()`, extracting per-token logprobs from `response.logprobs`

Tokenizer handling:
- MLX wraps HuggingFace tokenizers; the backend extracts `tokenizer._tokenizer` for full API compatibility

#### OpenAIBackend

Uses OpenAI API via the `openai` Python client.

Key features:
- Uses `tiktoken` with `o200k_base` encoding for tokenization
- Requests logprobs via `logprobs=True, top_logprobs=20`
- Greedy decoding when `temperature=0`

Limitations:
- `forward()` raises `NotImplementedError` (no direct model access)
- `n_layers` and `d_model` return 0 (closed model)
- Logprobs only for top-20 tokens in `get_next_token_probs`

Environment: Requires `OPENAI_API_KEY` environment variable.

#### AnthropicBackend

Uses Anthropic API via the `anthropic` Python client.

Key features:
- Uses `tiktoken` with `cl100k_base` encoding (approximation)
- Default model: `claude-sonnet-4-20250514`

Critical limitation:
- **Anthropic API does NOT provide logprobs**
- All logprob values are 0.0
- Suitable for text generation and categorical judgments
- NOT suitable for probability-weighted metrics or perplexity

Environment: Requires `ANTHROPIC_API_KEY` environment variable.

### Backend Selection Logic

`get_recommended_backend_inference()` in `backend_selection.py`:

```python
def get_recommended_backend_inference() -> ModelBackend:
    if _is_apple_silicon() and _mlx_available():
        return ModelBackend.MLX
    return ModelBackend.HUGGINGFACE
```

---

## Trajectory Generation API

### GeneratedTrajectory

`GeneratedTrajectory` (`generated_trajectory.py`) extends `TokenTrajectory` with model internals support.

```python
@dataclass
class GeneratedTrajectory(TokenTrajectory):
    internals: dict = field(default_factory=dict)  # Captured activations
```

Key fields (inherited from `TokenTrajectory`):
- `token_ids: list[int]` - Full sequence of token IDs
- `logprobs: list[float]` - Log probability of each token given prior context
- `logits: list[float]` - Scalar logit for each token
- `full_logits: torch.Tensor | None` - Full vocab logits `[seq_len, vocab_size]`

The first token always has `logprob=0.0` (it's given, not predicted).

### Factory Methods

#### `from_inference()`

Build from forward pass outputs:

```python
GeneratedTrajectory.from_inference(
    token_ids: list[int],     # Full sequence [n_sequence]
    logits: torch.Tensor,     # Full logits [n_sequence, vocab_size]
    device: str = "cpu",
    internals: dict | None = None,
) -> GeneratedTrajectory
```

Computation:
1. First token: `logprob=0.0`, `logit=0.0`
2. For positions 1..n: compute `log_softmax(logits[i-1])` and gather probability for `token_ids[i]`

#### `from_logprobs()`

Build from logprobs only (no full logits):

```python
GeneratedTrajectory.from_logprobs(
    token_ids: list[int],
    logprobs: list[float],
) -> GeneratedTrajectory
```

Sets `full_logits=None` and uses logprobs as scalar logit approximation.

#### `from_token_trajectory()`

Upgrade a `TokenTrajectory` to `GeneratedTrajectory`:

```python
GeneratedTrajectory.from_token_trajectory(
    trajectory: TokenTrajectory,
    internals: dict | None = None,
) -> GeneratedTrajectory
```

### ModelRunner Generation Methods

#### `generate_trajectory()`

Generate from token IDs:

```python
runner.generate_trajectory(
    token_ids: list[int],
    max_new_tokens: int = 256,
    temperature: float = 0.0,  # 0.0 = greedy
) -> GeneratedTrajectory
```

#### `generate_trajectory_from_prompt()`

Generate from text prompt:

```python
runner.generate_trajectory_from_prompt(
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    prefilling: str = "",  # Text prepended to generation
) -> GeneratedTrajectory
```

Applies chat template if model is a chat model, then tokenizes and generates.

### Batch Processing

```python
calculate_trajectories_for_batch(
    token_ids_batch: list[list[int]],
    logits_batch: torch.Tensor,  # [batch, max_seq_len, vocab_size]
    device: str = "cpu",
) -> list[GeneratedTrajectory]
```

Handles variable-length sequences by trimming padding per sequence.

---

## Token-Level Operations

### Encoding

```python
# Returns torch.Tensor [1, seq_len]
runner.encode(text: str, add_special_tokens: bool = True,
              prepend_bos: bool = False) -> torch.Tensor

# Returns list[int]
runner.encode_ids(text: str, add_special_tokens: bool = True,
                  prepend_bos: bool = False) -> list[int]
```

### Decoding

```python
# From tensor
runner.decode(token_ids: torch.Tensor) -> str

# From list
runner.decode_ids(token_ids: list[int]) -> str
```

Both methods preserve special tokens (use `skip_special_tokens=False` internally).

### Chat Template

```python
runner.apply_chat_template(prompt: str) -> str
```

For chat models, applies the tokenizer's chat template to format the prompt as a conversation. Returns unchanged prompt for base models.

### Next Token Probabilities

Via backends:

```python
# By token string
backend.get_next_token_probs(prompt: str, target_tokens: Sequence[str]) -> dict[str, float]

# By token ID
backend.get_next_token_probs_by_id(prompt: str, token_ids: Sequence[int]) -> dict[int, float]
```

---

## Embedding Runner

`EmbeddingRunner` (`embedding_runner.py`) provides text embeddings via sentence-transformers.

### Initialization

```python
EmbeddingRunner(model_name: str = "all-MiniLM-L6-v2")
```

Suppresses stdout/stderr during model loading to avoid "LOAD REPORT" noise.

### Methods

#### `embed()`

Batch embedding:

```python
runner.embed(texts: list[str]) -> NDArray[np.float32]
# Returns: shape (len(texts), embedding_dim)
```

#### `embed_single()`

Single text embedding:

```python
runner.embed_single(text: str) -> NDArray[np.float32]
# Returns: shape (embedding_dim,)
```

#### `similarity()`

Cosine similarity between two texts:

```python
runner.similarity(text: str, reference: str) -> float
# Returns: [0, 1] (clamped from [-1, 1] via (sim + 1) / 2)
```

#### `similarities()`

One-to-many similarity:

```python
runner.similarities(text: str, references: list[str]) -> list[float]
```

Efficiently computes embeddings for `[text] + references` in one call.

---

## Memory and Performance

### KV Caching

All local backends use KV caching for efficient autoregressive generation:
- HuggingFace: `use_cache=True` in `model.generate()`
- MLX: Built into `stream_generate()`

### Inference Mode

`ModelRunner.generate()` uses the appropriate context:
- `torch.inference_mode()` for backends where `supports_inference_mode=True`
- `torch.no_grad()` for API backends

### Torch Compile

HuggingFace backend attempts `torch.compile()` on CUDA devices for faster inference:

```python
if self.device == "cuda":
    try:
        self._model = torch.compile(self._model)
    except Exception:
        pass
```
