# Visualization Utilities

ASCII tree visualization utilities for displaying trajectory trees with branching structures.

## Module: `tree_display.py`

### TreePathLike Protocol

Objects implementing this protocol can be visualized using the tree formatting functions:
- `path_id: int` - Unique identifier for the path
- `parent_id: int | None` - ID of parent path (None for root)
- `branch_pos: int | None` - Token position where branch occurs
- `continuation: str` - Text content of the path
- `token_ids: list[int]` - Token IDs for the path

### SimplePath

A simple dataclass implementing `TreePathLike` for minimal tree visualization use cases.

### Formatting Functions

**`format_horizontal_tree(tree_paths, prompt_len, max_new_tokens, width=50)`**

Renders tree as a horizontal timeline showing token positions. Creates ASCII art with branches at their relative token positions on a ruler.

Example:
```
    0    10   20   30   40   50
    └──────────────────────● [0]
    │      └─────────────● [1]
    │      └───────● [2]
```

**`format_tree_simple(tree_paths, text_width=40)`**

Renders tree as a simple list with parent relationships and text previews.

Example:
```
[0] "The quick brown fox..."
[1] <- [0]@15: "jumps over the lazy dog..."
[2] <- [0]@15: "runs through the forest..."
```

### Utility Functions

**`create_forking_tree_paths(greedy_traj_ids, greedy_continuation, fork_points)`**

Constructs tree paths from forking paths generation results. Takes a greedy trajectory and list of fork points, returns a list of `SimplePath` objects ready for visualization.
