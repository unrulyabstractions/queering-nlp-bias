# src/inference/

> **Note**: This documentation was AI-generated and may contain errors. If something seems off, check the code or open an issue.


Model inference with multi-backend support.

## Quick Start

```python
from src.inference import ModelRunner

runner = ModelRunner("Qwen/Qwen3-0.6B")
traj = runner.generate_trajectory_from_prompt("Write a story", max_new_tokens=100)
```

## Contents

| File | Purpose |
|------|---------|
| `model_runner.py` | Unified model interface |
| `generated_trajectory.py` | Trajectory with logprobs and internals |
| `embedding_runner.py` | Text embedding via sentence-transformers |
| `backends/` | Backend implementations (MLX, HuggingFace, OpenAI, Anthropic) |

## Key Classes

- **ModelRunner**: Load models, encode/decode, generate trajectories
- **GeneratedTrajectory**: Token sequence with logprobs (extends TokenTrajectory)
- **EmbeddingRunner**: Compute text embeddings and cosine similarities

## Backend Selection

Automatic based on model name and hardware:
- `openai/...` or `gpt-4...` -> OpenAI API
- `anthropic/...` or `claude...` -> Anthropic API
- Apple Silicon + MLX available -> MLX
- Otherwise -> HuggingFace

See [EXPLANATION.md](./EXPLANATION.md) for detailed architecture documentation.
