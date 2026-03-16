"""Tree exploration - samples at prefix tree nodes."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable

from webapp.common.algorithm_config import AlgorithmEvent, SamplingConfig
from webapp.common.sampling_loop import build_state, run_sampling_loop
from webapp.tree_exploration.tree_builder import build_prefix_tree


def _truncate(text: str, max_len: int = 80) -> str:
    """Truncate text for logging, adding ellipsis if needed."""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


async def explore_prefix_tree(
    prompt: str,
    prefixes: list[str],
    questions: list[str],
    max_rounds: int,
    config: SamplingConfig,
    should_stop: Callable[[], bool],
) -> AsyncIterator[AlgorithmEvent]:
    """Explore prefix tree by sampling at each node."""
    print("\n" + "=" * 60)
    print("STARTING TREE EXPLORATION")
    print("=" * 60)
    print(f"  Prompt: {_truncate(prompt, 100)}")
    print(f"  Prefixes: {len(prefixes)} total")
    for i, p in enumerate(prefixes[:5], 1):
        print(f"    {i}. {_truncate(p, 60)}")
    if len(prefixes) > 5:
        print(f"    ... and {len(prefixes) - 5} more")
    print(f"  Questions: {len(questions)}")
    print(f"  Max rounds: {max_rounds}")

    print("\n  -> Building prefix tree...")
    nodes = build_prefix_tree(prefixes)
    print(f"  -> Tree built with {len(nodes)} nodes")

    state = build_state(prompt, nodes, questions)

    print("\n  -> Starting sampling loop for tree exploration...")
    async for event in run_sampling_loop(
        state, config, max_rounds, should_stop, "tree"
    ):
        yield event

    print("\n" + "=" * 60)
    print("TREE EXPLORATION COMPLETE")
    print("=" * 60)
