"""Dynamics analysis - samples at each word position."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from webapp.common.algorithm_config import AlgorithmEvent, SamplingConfig
from webapp.common.llm_clients import generate_from_llm, get_client, judge_all_questions
from webapp.common.normativity_types import (
    GenerationNode,
    System,
    compute_core_diversity,
    compute_l2_distance,
    compute_l2_norm,
    get_word_positions,
)
from webapp.common.sampling_loop import (
    SamplingState,
    build_state,
    run_sampling_loop,
)
from webapp.common.text_formatting_utils import format_scores, truncate_for_log


def _extract_last_word(text: str) -> str:
    """Extract the last word from text."""
    words = text.split()
    return words[-1] if words else ""


def build_dynamics_state(
    prompt: str, continuation: str, questions: list[str]
) -> SamplingState:
    """Build sampling state from word positions in continuation text."""
    print("\n" + "=" * 60)
    print("BUILDING DYNAMICS STATE")
    print("=" * 60)
    print(f"  Continuation: {truncate_for_log(continuation, 100)}")

    positions = get_word_positions(continuation)
    print(f"  Word positions found: {len(positions)}")

    nodes = [
        GenerationNode(
            node_id=i,
            name=f"pos_{i}",
            prefix=continuation[:pos],
            label=_extract_last_word(continuation[:pos]),
            parent=i - 1 if i > 0 else None,
            depth=i,
        )
        for i, pos in enumerate(positions)
    ]

    # Show first few word positions
    for node in nodes[:5]:
        print(f"    [{node.node_id}] '{node.label}' at char {positions[node.node_id]}")
    if len(nodes) > 5:
        print(f"    ... and {len(nodes) - 5} more positions")

    return build_state(prompt, nodes, questions)


@dataclass
class PrefixSystems:
    """Prefix systems for all nodes in dynamics analysis."""

    systems: dict[int, System]
    initial: System
    final: System


def _compute_dynamics_metrics(
    core: System,
    prefix_system: System,
    initial_prefix: System,
    final_prefix: System,
) -> dict[str, float]:
    """Compute dynamics metrics.

    - pull: L2 norm of core (magnitude of trajectory scores)
    - drift: distance of prefix_system from initial
    - potential: distance of prefix_system to final
    - core_diversity: effective number of structures (exp of entropy of normalized core)
    """
    return {
        "pull": compute_l2_norm(core) if core else 0.0,
        "drift": compute_l2_distance(prefix_system, initial_prefix)
        if prefix_system and initial_prefix
        else 0.0,
        "potential": compute_l2_distance(final_prefix, prefix_system)
        if prefix_system and final_prefix
        else 0.0,
        "core_diversity": compute_core_diversity(core) if core else 1.0,
    }


async def _judge_prefix(
    prefix: str,
    questions: list[str],
    judge_client,
    config: SamplingConfig,
) -> System:
    """Judge a single prefix text."""
    print(f"  -> Judging prefix: '{truncate_for_log(prefix, 50)}'")
    results = await judge_all_questions(
        judge_client,
        config.judge_provider,
        config.judge_model,
        prefix,
        questions,
        config.judge_prompt,
    )
    scores = [r.score for r in results]
    print(f"     Prefix scores: {format_scores(scores)}")
    return scores


async def _judge_prefixes_streaming(
    state: SamplingState,
    config: SamplingConfig,
    position_map: dict[int, int],
) -> AsyncIterator[tuple[int, System, dict]]:
    """Judge prefixes and yield results as they complete."""
    print("\n" + "=" * 60)
    print("JUDGING PREFIX TEXTS (STREAMING)")
    print("=" * 60)
    print(f"  Total prefixes to judge: {len(state.nodes)}")

    judge_client = get_client(config.judge_provider, config.judge_api_key)
    systems: dict[int, System] = {}
    completed = 0

    # Create tasks with node info
    async def judge_with_id(node: GenerationNode) -> tuple[int, System | Exception]:
        try:
            result = await _judge_prefix(node.prefix, state.questions, judge_client, config)
            return (node.node_id, result)
        except Exception as e:
            return (node.node_id, e)

    # Use asyncio.as_completed to yield results as they finish
    tasks = [asyncio.create_task(judge_with_id(node)) for node in state.nodes]

    for coro in asyncio.as_completed(tasks):
        node_id, result = await coro
        completed += 1

        if isinstance(result, Exception):
            print(f"  -> ERROR judging node {node_id}: {str(result)[:80]}")
            continue

        systems[node_id] = result
        node = state.nodes_by_id.get(node_id)

        # Build position data for this node
        first_node_id = state.nodes[0].node_id if state.nodes else 0
        last_node_id = state.nodes[-1].node_id if state.nodes else 0
        initial = systems.get(first_node_id, [])
        final = systems.get(last_node_id, [])

        pos_data = {
            "position": position_map.get(node_id, 0),
            "label": node.label if node else "",
            "core": [],  # No core yet, just prefix
            "prefix_system": result,
            "initial_prefix": initial,
            "final_prefix": final,
            "pull": 0.0,
            "drift": compute_l2_distance(result, initial) if result and initial else 0.0,
            "potential": compute_l2_distance(final, result) if result and final else 0.0,
        }

        yield (node_id, result, pos_data)

    print(f"  -> Prefix judging complete: {len(systems)} success")


async def track_text_dynamics(
    prompt: str,
    continuation: str,
    questions: list[str],
    max_tokens: int,
    max_rounds: int,
    config: SamplingConfig,
    should_stop: Callable[[], bool],
) -> AsyncIterator[AlgorithmEvent]:
    """Track how orientation changes word by word."""
    print("\n" + "=" * 60)
    print("STARTING DYNAMICS ANALYSIS")
    print("=" * 60)
    print(f"  Prompt: {truncate_for_log(prompt, 100)}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Max rounds: {max_rounds}")
    print(f"  Questions: {len(questions)}")

    if not continuation:
        print("\n  -> No continuation provided, generating text...")
        yield AlgorithmEvent("status", {"message": "Generating text..."})
        gen_client = get_client(config.gen_provider, config.gen_api_key)
        gen_result = await generate_from_llm(
            gen_client,
            config.gen_provider,
            config.gen_model,
            prompt,
            "",
            max_tokens,
            config.temperature,
        )
        continuation = gen_result.text
        print(f"  -> Generated: {truncate_for_log(continuation, 100)}")
    else:
        print(f"  Continuation provided: {truncate_for_log(continuation, 100)}")

    yield AlgorithmEvent(
        "continuation", {"text": continuation, "length": len(continuation)}
    )

    state = build_dynamics_state(prompt, continuation, questions)
    position_map = {i: pos for i, pos in enumerate(get_word_positions(continuation))}
    positions_data: dict[int, dict] = {}

    # Send initial started event so UI can show something immediately
    yield AlgorithmEvent("started", {
        "mode": "dynamics",
        "data": {
            "nodes": [{"node_id": n.node_id, "label": n.label, "depth": n.depth, "core": []} for n in state.nodes],
            "questions": questions,
            "total_samples": 0,
            "total_api_calls": 0,
        }
    })

    # Stream prefix judging results
    print("\n  -> Starting prefix judging phase (streaming)...")
    yield AlgorithmEvent("status", {"message": "Judging partial texts..."})

    prefix_systems_dict: dict[int, System] = {}
    first_node_id = state.nodes[0].node_id if state.nodes else 0
    last_node_id = state.nodes[-1].node_id if state.nodes else 0

    async for node_id, prefix_scores, pos_data in _judge_prefixes_streaming(state, config, position_map):
        prefix_systems_dict[node_id] = prefix_scores
        positions_data[node_id] = pos_data

        # Yield position update as each prefix is judged
        yield AlgorithmEvent(
            "position_update",
            {
                "node_id": node_id,
                "label": pos_data["label"],
                "core": [],
                "prefix_system": prefix_scores,
                "position": pos_data["position"],
                "pull": 0.0,
                "drift": pos_data["drift"],
                "potential": pos_data["potential"],
                "all_positions": sorted(positions_data.values(), key=lambda p: p["position"]),
                "progress": len(positions_data) / len(state.nodes) if state.nodes else 0.0,
                "total_api_calls": len(positions_data) * len(questions),
            },
        )

    # Build final prefix_systems for sampling loop
    prefix_systems = PrefixSystems(
        systems=prefix_systems_dict,
        initial=prefix_systems_dict.get(first_node_id, []),
        final=prefix_systems_dict.get(last_node_id, []),
    )

    print(f"  Initial prefix system: {format_scores(prefix_systems.initial)}")
    print(f"  Final prefix system: {format_scores(prefix_systems.final)}")

    # Now run sampling loop for trajectory scores
    print("\n  -> Starting sampling loop for dynamics...")
    yield AlgorithmEvent("status", {"message": "Sampling trajectories..."})

    async for event in run_sampling_loop(
        state, config, max_rounds, should_stop, "dynamics"
    ):
        if event.type == "started":
            # Skip the started event from sampling loop since we already sent one
            continue
        elif event.type == "point_update":
            node_id = event.data.get("node_id")
            core: System = event.data.get("core", [])
            orient_std: System = event.data.get("orient_std", [])
            prefix_system = prefix_systems.systems.get(node_id, [])

            metrics = _compute_dynamics_metrics(
                core, prefix_system, prefix_systems.initial, prefix_systems.final
            )

            # Log dynamics metrics for this position
            print(f"  -> Position {node_id} metrics: pull={metrics['pull']:.3f}, "
                  f"drift={metrics['drift']:.3f}, potential={metrics['potential']:.3f}, "
                  f"diversity={metrics['core_diversity']:.2f}")

            positions_data[node_id] = {
                "position": position_map.get(node_id, 0),
                "label": event.data.get("label", ""),
                "core": core,
                "orient_std": orient_std,
                "prefix_system": prefix_system,
                "initial_prefix": prefix_systems.initial,
                "final_prefix": prefix_systems.final,
                **metrics,
            }

            yield AlgorithmEvent(
                "position_update",
                {
                    **event.data,
                    "prefix_system": prefix_system,
                    **metrics,
                    "position": position_map.get(node_id, 0),
                    "all_positions": sorted(
                        positions_data.values(), key=lambda p: p["position"]
                    ),
                    "progress": 0.5 + 0.5 * (len([p for p in positions_data.values() if p.get("core")]) / len(state.nodes))
                    if state.nodes
                    else 0.0,
                },
            )
        else:
            yield event

    print("\n" + "=" * 60)
    print("DYNAMICS ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Positions analyzed: {len(positions_data)}")
