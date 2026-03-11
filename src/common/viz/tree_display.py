"""Tree visualization utilities for displaying trajectory trees.

This module provides utilities for visualizing token trajectory trees,
including horizontal timeline views and simple list formats.

Moved from scripts/schemas/script_utils.py to make these utilities
available throughout the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.common.log_utils import oneline
from src.common.viz_utils import preview


# ══════════════════════════════════════════════════════════════════════════════
# PROTOCOLS
# ══════════════════════════════════════════════════════════════════════════════


class TreePathLike(Protocol):
    """Protocol for objects that can be visualized as tree paths.

    Any object with these attributes can be rendered using the tree
    visualization functions in this module.
    """

    path_id: int
    parent_id: int | None
    branch_pos: int | None
    continuation: str

    @property
    def token_ids(self) -> list[int]: ...


# ══════════════════════════════════════════════════════════════════════════════
# SIMPLE PATH IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class SimplePath:
    """Simple implementation of TreePathLike for visualization.

    Use this when you need a minimal implementation of the TreePathLike
    protocol for creating tree visualizations.
    """

    path_id: int
    parent_id: int | None
    branch_pos: int | None
    continuation: str
    _token_ids: list[int]

    @property
    def token_ids(self) -> list[int]:
        return self._token_ids


# ══════════════════════════════════════════════════════════════════════════════
# TREE FORMATTING
# ══════════════════════════════════════════════════════════════════════════════


def format_horizontal_tree(
    tree_paths: list[TreePathLike],
    prompt_len: int,
    max_new_tokens: int,
    width: int = 50,
) -> list[str]:
    """Format tree as horizontal timeline showing token positions.

    Creates an ASCII art visualization showing the tree structure
    with branches at their token positions on a horizontal timeline.

    Args:
        tree_paths: List of tree paths to visualize
        prompt_len: Length of prompt in tokens (for computing relative positions)
        max_new_tokens: Maximum token positions to show on timeline
        width: Character width for the timeline visualization

    Returns:
        List of string lines to print/display.

    Example output:
        0    10   20   30   40   50
        └──────────────────────● [0]
        │      └─────────────● [1]
        │      └───────● [2]
    """
    if not tree_paths:
        return []

    scale = width / max(max_new_tokens, 1)

    def pos_to_col(rel_token_pos: int) -> int:
        return int(rel_token_pos * scale)

    # Build parent->children mapping
    children: dict[int | None, list[TreePathLike]] = {}
    for path in tree_paths:
        parent = path.parent_id
        if parent not in children:
            children[parent] = []
        children[parent].append(path)

    lines: list[str] = []

    # Create ruler line
    prefix = "    "
    ruler = prefix
    step = max(5, max_new_tokens // 6)
    for i in range(0, max_new_tokens + 1, step):
        col = pos_to_col(i)
        label = str(i)
        ruler = ruler.ljust(len(prefix) + col) + label
    lines.append(ruler)

    def get_path_length(path: TreePathLike) -> int:
        return len(path.token_ids) - prompt_len

    def render_path(
        path: TreePathLike,
        row_prefix: str,
        is_last_sibling: bool,
    ) -> None:
        start = path.branch_pos if path.branch_pos is not None else 0
        end = get_path_length(path)
        start_col = pos_to_col(start)
        end_col = pos_to_col(end)

        total_width = len(prefix) + width + 15
        line = list(row_prefix.ljust(total_width))

        line_start = len(prefix) + start_col
        line_end = len(prefix) + end_col

        connector = "└" if is_last_sibling else "├"
        if line_start < len(line):
            line[line_start] = connector
        for i in range(line_start + 1, min(line_end, len(line))):
            line[i] = "─"
        if line_end < len(line):
            line[line_end] = "●"

        label = f" [{path.path_id}]"
        for i, c in enumerate(label):
            if line_end + 1 + i < len(line):
                line[line_end + 1 + i] = c

        lines.append("".join(line).rstrip())

        # Render children recursively
        path_children = children.get(path.path_id, [])
        for i, child in enumerate(path_children):
            child_is_last = i == len(path_children) - 1

            total_width = len(prefix) + width + 15
            new_prefix = list(row_prefix.ljust(total_width))

            if not is_last_sibling:
                vert_col = len(prefix) + start_col
                if vert_col < len(new_prefix):
                    new_prefix[vert_col] = "│"

            branch_col = len(prefix) + pos_to_col(child.branch_pos or 0)
            if branch_col < len(new_prefix):
                new_prefix[branch_col] = "│"

            render_path(child, "".join(new_prefix), child_is_last)

    root_paths = children.get(None, [])
    for i, root in enumerate(root_paths):
        is_last = i == len(root_paths) - 1
        render_path(root, "", is_last)

    return lines


def format_tree_simple(
    tree_paths: list[TreePathLike],
    text_width: int = 40,
) -> list[str]:
    """Format tree as simple list with path details.

    Creates a text list showing each path with its parent relationship
    and continuation text preview.

    Args:
        tree_paths: List of tree paths to display
        text_width: Maximum width for text previews

    Returns:
        List of string lines to print/display.

    Example output:
        [0] "The quick brown fox..."
        [1] <- [0]@15: "jumps over the lazy dog..."
        [2] <- [0]@15: "runs through the forest..."
    """
    if not tree_paths:
        return []

    lines = []
    for path in tree_paths:
        text_preview = preview(oneline(path.continuation), text_width)
        if path.parent_id is None:
            lines.append(f'[{path.path_id}] "{text_preview}"')
        else:
            lines.append(
                f"[{path.path_id}] <- [{path.parent_id}]@{path.branch_pos}: "
                f'"{text_preview}"'
            )

    return lines


# ══════════════════════════════════════════════════════════════════════════════
# FORKING TREE CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════


def create_forking_tree_paths(
    greedy_traj_ids: list[int],
    greedy_continuation: str,
    fork_points: list[tuple[int, list[tuple[list[int], str]]]],
) -> list[SimplePath]:
    """Create tree paths from forking paths generation result.

    Used with the forking paths generation method to construct a
    visualizable tree structure from the greedy trajectory and
    its fork points.

    Args:
        greedy_traj_ids: Token IDs for the greedy (main) trajectory
        greedy_continuation: Text of the greedy continuation
        fork_points: List of (position, continuations) where continuations
                     is a list of (token_ids, text) tuples for each fork

    Returns:
        List of SimplePath objects suitable for tree visualization.

    Example:
        >>> paths = create_forking_tree_paths(
        ...     greedy_traj_ids=[1, 2, 3, 4, 5],
        ...     greedy_continuation="hello world",
        ...     fork_points=[(2, [([1, 2, 6, 7], "hello there")])]
        ... )
        >>> len(paths)
        2  # greedy path + 1 fork
    """
    paths = [
        SimplePath(
            path_id=0,
            parent_id=None,
            branch_pos=None,
            continuation=greedy_continuation,
            _token_ids=greedy_traj_ids,
        )
    ]

    path_id = 1
    for position, continuations in fork_points:
        for traj_ids, cont_text in continuations:
            paths.append(
                SimplePath(
                    path_id=path_id,
                    parent_id=0,
                    branch_pos=position,
                    continuation=cont_text,
                    _token_ids=traj_ids,
                )
            )
            path_id += 1

    return paths
