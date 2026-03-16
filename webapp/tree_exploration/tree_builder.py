"""
Prefix tree builder - creates a trie from sentence prefixes.

Builds a tree where each node represents a divergence point in the prefixes.
Handles any tree shape: chains, wide trees, deep trees, etc.
"""

from __future__ import annotations

from webapp.common.normativity_types import GenerationNode


def _truncate(text: str, max_len: int = 80) -> str:
    """Truncate text for logging."""
    text = text.replace("\n", " ").strip()
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _find_common_prefix(strings: list[str]) -> str:
    """Find longest common prefix among strings."""
    if not strings:
        return ""
    prefix = strings[0]
    for s in strings[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


def _trim_to_word(text: str) -> str:
    """Trim to last complete word boundary."""
    if " " in text:
        return text.rsplit(" ", 1)[0] + " "
    return text


class _TrieNode:
    """Internal trie node for building the tree."""

    def __init__(self, prefix: str, label: str):
        self.prefix = prefix
        self.label = label
        self.children: list[_TrieNode] = []
        self.is_leaf = False


def _build_trie(prefixes: list[str], current_prefix: str = "") -> _TrieNode:
    """Recursively build a trie from prefixes."""
    # Find common prefix among all strings at this level
    remainders = [p[len(current_prefix):] for p in prefixes]
    common = _find_common_prefix(remainders)

    if common:
        common = _trim_to_word(common)

    new_prefix = current_prefix + common
    label = common.strip() if common else (current_prefix.split()[-1] if current_prefix.split() else "root")

    node = _TrieNode(new_prefix, label)

    # Get remainders after common prefix
    remainders = [p[len(new_prefix):] for p in prefixes]

    # Group by first diverging word/segment
    groups: dict[str, list[str]] = {}
    for i, remainder in enumerate(remainders):
        if not remainder:
            # This prefix ends here
            node.is_leaf = True
        else:
            # Get first word as group key
            first_word = remainder.split()[0] if remainder.split() else remainder
            if first_word not in groups:
                groups[first_word] = []
            groups[first_word].append(prefixes[i])

    # Recursively build children
    for group_prefixes in groups.values():
        child = _build_trie(group_prefixes, new_prefix)
        node.children.append(child)

    return node


def _flatten_trie(
    trie_node: _TrieNode,
    parent_id: int | None,
    depth: int,
    nodes: list[GenerationNode],
    node_counter: list[int],
) -> None:
    """Flatten trie to list of GenerationNode in depth-first order."""
    node_id = node_counter[0]
    node_counter[0] += 1

    # Create label for display
    label = trie_node.label if trie_node.label else "root"

    nodes.append(
        GenerationNode(
            node_id=node_id,
            name=f"node_{node_id}",
            prefix=trie_node.prefix,
            label=label,
            parent=parent_id,
            depth=depth,
        )
    )

    # Recursively add children
    for child in trie_node.children:
        _flatten_trie(child, node_id, depth + 1, nodes, node_counter)


def build_prefix_tree(sentences: list[str]) -> list[GenerationNode]:
    """
    Build a prefix tree (trie) from sentence prefixes.

    Creates nodes at each divergence point. Handles any tree structure:
    - Chains (sequential nodes)
    - Wide trees (many branches)
    - Deep trees (many levels)
    - Mixed structures

    Args:
        sentences: List of sentence prefixes

    Returns:
        List of GenerationNode in depth-first order
    """
    print("\n" + "=" * 60)
    print("BUILDING PREFIX TREE")
    print("=" * 60)
    print(f"  Input sentences: {len(sentences)}")

    if not sentences:
        return []

    # Always start with root (empty prefix)
    nodes: list[GenerationNode] = []
    node_counter = [0]

    # Add root
    root = GenerationNode(
        node_id=0,
        name="root",
        prefix="",
        label="root",
        parent=None,
        depth=0,
    )
    nodes.append(root)
    node_counter[0] = 1

    # Build trie from sentences
    trie = _build_trie(sentences, "")

    # If trie has content beyond empty, add its structure
    if trie.prefix or trie.children:
        if trie.prefix:
            # Add trunk node for common prefix
            trunk_id = node_counter[0]
            node_counter[0] += 1
            nodes.append(
                GenerationNode(
                    node_id=trunk_id,
                    name="trunk",
                    prefix=trie.prefix,
                    label=trie.label or trie.prefix.strip(),
                    parent=0,
                    depth=1,
                )
            )
            parent_for_children = trunk_id
            child_depth = 2
        else:
            parent_for_children = 0
            child_depth = 1

        # Add all children
        for child in trie.children:
            _flatten_trie(child, parent_for_children, child_depth, nodes, node_counter)

    print(f"  -> Tree complete: {len(nodes)} total nodes")
    for node in nodes:
        indent = "    " + "  " * node.depth
        parent_str = f"(parent={node.parent})" if node.parent is not None else "(root)"
        print(f"{indent}[{node.node_id}] {node.name}: '{_truncate(node.label, 30)}' {parent_str}")

    return nodes
