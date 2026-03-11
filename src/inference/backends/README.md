# Model Backends

> **Note**: This documentation was AI-generated and may contain errors. If something seems off, check the code or open an issue.


Unified interface for model inference across different platforms and APIs.

## Contents

| File | Purpose |
|------|---------|
| `model_backend.py` | `ModelBackend` enum and `Backend` ABC |
| `backend_selection.py` | Auto-select backend based on hardware |
| `huggingface.py` | HuggingFace Transformers backend |
| `mlx.py` | MLX backend (Apple Silicon) |
| `openai.py` | OpenAI API backend |
| `anthropic.py` | Anthropic API backend |

## Usage

```python
from src.inference.backends import ModelBackend, get_recommended_backend_inference

backend = get_recommended_backend_inference()  # Auto-select
backend = ModelBackend.MLX                      # Explicit
```

See [../EXPLANATION.md](../EXPLANATION.md) for detailed backend architecture.
