# Text Utilities

Text processing utilities for trajectory analysis and language model output.

## Contents

- `text_display.py` - Display name formatting (arm names, structure labels)
- `eos_handling.py` - End-of-sequence token handling
- `thinking_filter.py` - Thinking block filtering

## API

### Display names (`text_display.py`)

```python
from src.common.text import arm_display_name, structure_label

# Arm display names (0=trunk, 1+=branches)
arm_display_name(0)   # "trunk"
arm_display_name(1)   # "branch_1"

# Structure labels (1-indexed)
structure_label(0, "c")  # "c1" (categorical)
structure_label(2, "g")  # "g3" (graded)
```

### EOS token handling (`eos_handling.py`)

```python
from src.common.text import strip_eos_tokens

# Strip end-of-sequence markers
strip_eos_tokens("Hello world<|im_end|>", ["<|im_end|>"])  # "Hello world"
```

### Thinking block filtering (`thinking_filter.py`)

```python
from src.common.text import strip_thinking_blocks

# Remove <think>...</think> blocks
strip_thinking_blocks("text <think>reasoning</think> more text")  # "text  more text"
```
