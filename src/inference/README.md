# src/inference/

Model inference and trajectory generation.

See [GENERATION.md](../../GENERATION.md) for conceptual background on generation methods.

## Directory Structure

```
inference/
├── backends/              # Model backend implementations
│   ├── model_backend.py   # Abstract backend interface
│   ├── huggingface.py     # HuggingFace Transformers backend
│   ├── mlx.py             # MLX backend (Apple Silicon)
│   └── backend_selection.py
├── model_runner.py        # Unified model interface
└── generated_trajectory.py # Trajectory with captured internals
```

## ModelRunner

Primary interface for model inference. Automatically selects the appropriate backend.

```python
from src.inference import ModelRunner

# Load model
runner = ModelRunner("Qwen/Qwen3-0.6B")

# Properties
runner.device          # "cuda", "mps", or "cpu"
runner.vocab_size      # 151936
runner.n_layers        # 28
runner.d_model         # 1024

# Tokenization
ids = runner.encode_ids("Hello world")
text = runner.decode_ids(ids)

# Chat template
formatted = runner.apply_chat_template("Write a story")

# Trajectory generation (uses KV caching for efficiency)
traj = runner.generate_trajectory(
    token_ids=prompt_ids,
    max_new_tokens=100,
    temperature=1.0,
)

# From prompt directly
traj = runner.generate_trajectory_from_prompt(
    prompt="Write a story",
    max_new_tokens=100,
    temperature=1.0,
    prefilling="Once upon a time",
)
```

### Reasoning Models

For models with thinking tokens (e.g., Qwen3):

```python
runner = ModelRunner("Qwen/Qwen3-0.6B")

# Check if reasoning model
if runner.skip_thinking_prefix:
    # Model uses <think>...</think> tokens
    pass

# Thinking prefix is automatically handled in generation
```

## Generated Trajectory

Extended trajectory with captured model internals.

```python
from src.inference.generated_trajectory import GeneratedTrajectory

traj = GeneratedTrajectory(
    token_ids=[1, 2, 3, 4],
    logprobs=[0.0, -0.5, -1.2, -0.3],
    logits=[-0.5, -1.2, -0.3],
    full_logits=tensor,  # Optional: full vocab logits
)

# Factory from logprobs only (no full logits)
traj = GeneratedTrajectory.from_logprobs(token_ids, logprobs)
```

## Backends

### Abstract Interface

All backends implement the `Backend` abstract class:

```python
class Backend:
    # Tokenization
    def encode(text, add_special_tokens=True) -> torch.Tensor
    def decode(token_ids) -> str

    # Generation
    def generate(prompt, max_new_tokens, temperature) -> str
    def generate_trajectory(token_ids, max_new_tokens, temperature) -> tuple

    # Forward pass
    def forward(input_ids) -> torch.Tensor  # Returns logits

    # Token probabilities
    def get_next_token_probs(prompt, target_tokens) -> dict[str, float]
    def get_next_token_probs_by_id(prompt, token_ids) -> dict[int, float]
```

### HuggingFace Backend

Uses the `transformers` library. Supports most open-source models.

```python
from src.inference.backends.huggingface import HuggingFaceBackend

# Automatically selected for HuggingFace model IDs
runner = ModelRunner("meta-llama/Llama-2-7b")
```

Features:
- KV caching via `model.generate(use_cache=True)`
- Score output for logprob extraction
- Batch operations

### MLX Backend

Optimized for Apple Silicon using the MLX framework.

```python
from src.inference.backends.mlx import MLXBackend

# Automatically selected on Apple Silicon when mlx is available
runner = ModelRunner("mlx-community/Qwen2.5-0.5B-4bit")
```

Features:
- Metal acceleration on Mac
- Streaming generation with `mlx_lm.stream_generate`
- Efficient memory usage

### Backend Selection

Backend is automatically selected based on:
1. Model ID format (mlx-community → MLX)
2. Hardware availability (MPS → MLX, CUDA → HuggingFace)
3. Library availability

```python
from src.inference.backends.backend_selection import select_backend

backend = select_backend(runner, model_name)
```

## Performance Notes

### KV Caching

All backends use KV caching for efficient generation:

```python
# Efficient: O(n) for n tokens
traj = runner.generate_trajectory(prompt_ids, max_new_tokens=100)

# Each new token only processes 1 position, reusing cached K/V
```

### Memory Management

```python
from src.common.device_utils import clear_gpu_memory

# Clear GPU memory after generation
clear_gpu_memory()
```

## Example: Full Generation Pipeline

```python
from src.inference import ModelRunner
from src.common.token_tree import TokenTree

runner = ModelRunner("Qwen/Qwen3-0.6B")

# Generate trajectories
trajectories = []
for i in range(10):
    traj = runner.generate_trajectory_from_prompt(
        prompt="Write a story about a",
        prefilling="boy",
        max_new_tokens=50,
        temperature=1.2,
    )
    trajectories.append(traj)

# Build tree
tree = TokenTree.from_trajectories(
    trajs=trajectories,
    groups_per_traj=[(0,)] * 10,
    fork_arms=["boy"],
    trunk=[],
)
tree.decode_texts(runner)
```
