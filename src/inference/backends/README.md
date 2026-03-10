# Model Backends

Unified interface for different model inference backends.

## Supported Backends

| Backend | Module | Use Case |
|---------|--------|----------|
| MLX | `mlx.py` | Apple Silicon (fastest on M-series) |
| HuggingFace | `huggingface.py` | General GPU/CPU inference |
| OpenAI | `openai.py` | OpenAI API models |
| Anthropic | `anthropic.py` | Claude API models |

## Contents

- `model_backend.py` - `ModelBackend` enum and base interfaces
- `backend_selection.py` - Automatic backend selection logic
- `mlx.py` - MLX backend implementation
- `huggingface.py` - HuggingFace/Transformers backend
- `openai.py` - OpenAI API backend
- `anthropic.py` - Anthropic API backend

## Usage

```python
from src.inference.backends import ModelBackend, get_recommended_backend_inference

# Auto-select best backend
backend = get_recommended_backend_inference()

# Or specify explicitly
backend = ModelBackend.MLX
```
