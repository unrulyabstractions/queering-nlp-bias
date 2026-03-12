# Model Backends

Unified interface for model inference across different platforms and APIs.

## Available Backends

| Backend | Type | Use Case |
|---------|------|----------|
| **MLX** | Local | Apple Silicon (fastest) |
| **HuggingFace** | Local | CPU/GPU (most compatible) |
| **OpenAI** | API | GPT-4o, remote inference with logprobs |
| **Anthropic** | API | Claude models, remote inference (no logprobs) |

## Backend Selection

Use `get_recommended_backend_inference()` to auto-select the best backend:

- **Apple Silicon + MLX available** → MLX
- **Otherwise** → HuggingFace

```python
from src.inference.backends import get_recommended_backend_inference, ModelBackend

# Auto-select best available backend
backend = get_recommended_backend_inference()

# Explicit selection
backend = ModelBackend.MLX           # Apple Silicon only
backend = ModelBackend.HUGGINGFACE   # Universal
backend = ModelBackend.OPENAI        # Requires OPENAI_API_KEY
backend = ModelBackend.ANTHROPIC     # Requires ANTHROPIC_API_KEY
```

## Capabilities

All backends implement:
- **encode/decode** - Text ↔ token ID conversion
- **generate** - Generate text from prompt
- **get_next_token_probs** - Token probability distributions
- **generate_trajectory** - Generate with token logprobs

**Note on logprobs:**
- Local backends (MLX, HuggingFace) provide accurate logprobs
- OpenAI API provides logprobs (via API)
- Anthropic API returns uniform logprobs (API limitation)

## Files

- `model_backend.py` - `ModelBackend` enum and `Backend` abstract class
- `backend_selection.py` - Auto-selection logic based on hardware
- `huggingface_backend.py` - HuggingFace Transformers backend
- `mlx_backend.py` - MLX backend for Apple Silicon
- `openai_backend.py` - OpenAI API backend
- `anthropic_backend.py` - Anthropic API backend
- `api_tokenizer.py` - Shared tokenizer for API-based backends
