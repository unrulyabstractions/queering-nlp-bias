# Visualization Utilities

> **Note**: This documentation was AI-generated and may contain errors. If something seems off, check the code or open an issue.


Tree and text visualization utilities.

## Contents

- `tree_display.py` - ASCII tree visualization for trajectory trees

## Usage

```python
from src.common.viz import format_horizontal_tree, format_tree_simple

# Horizontal timeline view
lines = format_horizontal_tree(tree_paths, prompt_len, max_tokens)
# Output:
#   0    10   20   30   40   50
#   └──────────────────────● [0]
#   │      └─────────────● [1]

# Simple list view
lines = format_tree_simple(tree_paths)
# Output:
#   [0] "The quick brown fox..."
#   [1] <- [0]@15: "jumps over..."
```

See also `src/common/viz_utils.py` for text preview utilities.
