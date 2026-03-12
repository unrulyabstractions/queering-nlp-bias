# src/inference/

Model inference with multi-backend support (HuggingFace, MLX, OpenAI, Anthropic).

## Quick Start

```python
from src.inference import ModelRunner

runner = ModelRunner("Qwen/Qwen3-0.6B")
traj = runner.generate_trajectory_from_prompt("Write a story", max_new_tokens=100)
```

## ModelRunner

`ModelRunner` is the unified inference interface. It automatically detects and routes to the appropriate backend based on model name and hardware.

### Backend Selection

Priority order:
1. **OpenAI**: `openai/...`, `gpt-4`, `gpt-3`, `o1`, `o3` → OpenAI API
2. **Anthropic**: `anthropic/...`, `claude` → Anthropic API
3. **MLX**: Apple Silicon + MLX available → MLX (optimized)
4. **HuggingFace**: Default fallback

### Model Loading

Models are loaded in `__init__` based on detected backend:

- **HuggingFace**: `AutoModelForCausalLM.from_pretrained()` with optional `torch.compile()` on CUDA
- **MLX**: `mlx_lm.load()` for Apple Silicon
- **OpenAI/Anthropic**: API clients initialized (no local model)

### Key Features

- **Auto chat model detection**: Detects instruct models by name patterns
- **Reasoning model detection**: Checks tokenizer's chat template for thinking tokens
- **Encoding/decoding**: Unified tokenizer access regardless of backend
- **Trajectory generation**: Returns `GeneratedTrajectory` with logprobs

## GeneratedTrajectory

Extends `TokenTrajectory` with:
- `internals`: dict of captured activations from forward pass
- Methods: `from_inference()`, `from_logprobs()`, `from_token_trajectory()`

## EmbeddingRunner

Uses sentence-transformers for text embeddings and similarity scoring.

```python
from src.inference import EmbeddingRunner

runner = EmbeddingRunner()
sim = runner.similarity("hello", "hi")
sims = runner.similarities("hello", ["hi", "bye"])
```

## Backends Directory

- `model_backend.py`: Base `Backend` abstract class
- `huggingface_backend.py`: HuggingFace + transformers
- `mlx_backend.py`: MLX for Apple Silicon
- `openai_backend.py`: OpenAI API
- `anthropic_backend.py`: Anthropic API (no logprobs)
- `backend_selection.py`: Hardware detection logic

See [EXPLANATION.md](./EXPLANATION.md) for detailed architecture and API specifications.
