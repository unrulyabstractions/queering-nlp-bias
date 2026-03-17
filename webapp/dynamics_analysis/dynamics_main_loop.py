"""Dynamics analysis - samples at each word position."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

from webapp.common.algorithm_config import AlgorithmEvent, SamplingConfig
from webapp.common.llm_clients import generate_from_llm, get_client, multi_provider_judge_all_questions
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
from webapp.common.text_formatting_utils import TextComponents, format_scores, truncate_for_log


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


@dataclass
class ConvergenceEntry:
    """Single entry in convergence history: mean and standard deviation at iteration n."""

    mean: System
    std: System


@dataclass
class WelfordAccumulator:
    """Online computation of mean and variance using Welford's algorithm.

    Tracks running mean and M2 (sum of squared deviations) for incremental
    computation of standard deviation. O(1) per update, numerically stable.
    """

    n: int = 0
    mean: list[float] = field(default_factory=list)
    m2: list[float] = field(default_factory=list)

    def update(self, sample: list[float]) -> None:
        """Add a new sample and update running statistics."""
        if self.n == 0:
            n_dims = len(sample)
            self.mean = [0.0] * n_dims
            self.m2 = [0.0] * n_dims

        self.n += 1
        for i in range(len(sample)):
            delta = sample[i] - self.mean[i]
            self.mean[i] += delta / self.n
            delta2 = sample[i] - self.mean[i]
            self.m2[i] += delta * delta2

    def get_std(self) -> list[float]:
        """Return current standard deviation (population std, divided by n)."""
        if self.n == 0:
            return []
        return [(m / self.n) ** 0.5 for m in self.m2]

    def get_entry(self) -> ConvergenceEntry:
        """Return current mean and std as ConvergenceEntry.

        Both are copied: mean explicitly, std implicitly (get_std creates new list).
        """
        return ConvergenceEntry(mean=self.mean.copy(), std=self.get_std())

    @classmethod
    def from_entry(cls, n: int, entry: ConvergenceEntry) -> "WelfordAccumulator":
        """Restore accumulator state from a ConvergenceEntry.

        The m2 value (sum of squared deviations) can be reconstructed from std:
        - variance = std^2
        - variance = m2 / n  (population variance formula used in get_std)
        - Therefore: m2 = variance * n = std^2 * n
        """
        acc = cls(n=n, mean=entry.mean.copy(), m2=[])
        acc.m2 = [(s**2) * n for s in entry.std]
        return acc


def _compute_running_stats_incremental(
    samples: list[System],
) -> list[ConvergenceEntry]:
    """Compute running means and standard deviations incrementally using Welford's algorithm.

    Returns list where entry i contains mean and std of samples[0:i+1].
    O(n) complexity instead of O(n^2).
    """
    if not samples or not samples[0]:
        return []

    acc = WelfordAccumulator()
    running_stats: list[ConvergenceEntry] = []

    for sample in samples:
        acc.update(sample)
        running_stats.append(acc.get_entry())

    return running_stats


# Cache for convergence history to avoid recomputation
_convergence_cache: dict[int, tuple[int, list[ConvergenceEntry]]] = {}


def _get_convergence_for_node(
    node_id: int, normativity_samples: list[System]
) -> list[ConvergenceEntry]:
    """Get convergence history for a single node, using cache when possible."""
    n_samples = len(normativity_samples)
    if not normativity_samples or not normativity_samples[0]:
        return []

    if node_id in _convergence_cache:
        cached_n, cached_history = _convergence_cache[node_id]
        if cached_n == n_samples:
            return cached_history
        # Samples added - extend incrementally using WelfordAccumulator
        if cached_n < n_samples and cached_history:
            acc = WelfordAccumulator.from_entry(cached_n, cached_history[-1])
            extended = cached_history.copy()
            for sample in normativity_samples[cached_n:]:
                acc.update(sample)
                extended.append(acc.get_entry())
            _convergence_cache[node_id] = (n_samples, extended)
            return extended
        # cached_n > n_samples shouldn't happen, but recompute if it does

    # Compute from scratch
    history = _compute_running_stats_incremental(normativity_samples)
    if history:
        _convergence_cache[node_id] = (n_samples, history)
    return history


def _build_convergence_history(
    state: SamplingState,
) -> dict[int, list[dict]]:
    """Build convergence history for all nodes using incremental caching.

    Returns dict[node_id, list[{mean: System, std: System}]] for JSON serialization.
    """
    result: dict[int, list[dict]] = {}
    for node_id, normativity in state.normativities.items():
        entries = _get_convergence_for_node(node_id, normativity.samples)
        result[node_id] = [{"mean": e.mean, "std": e.std} for e in entries]
    return result


async def _judge_prefix(
    prefix: str,
    questions: list[str],
    config: SamplingConfig,
) -> System:
    """Judge a single prefix text using multi-provider judging (scores averaged across judge models)."""
    print(f"  -> Judging prefix: '{truncate_for_log(prefix, 50)}'")
    results = await multi_provider_judge_all_questions(
        config.api_keys,
        config.judge_models,
        prefix,
        questions,
        config.judge_prompt,
        config.judge_temperature,
    )
    # Handle None/error scores: treat as 0.0 but log
    scores = []
    error_count = 0
    for i, r in enumerate(results):
        if r.score is None:
            error_count += 1
            print(f"     ⚠️ JUDGE PARSE ERROR [Q{i}]: treating as 0.0 | raw: {r.raw_response[:50]}...")
            scores.append(0.0)
        else:
            scores.append(r.score)
    errors_str = f" [{error_count} ERR]" if error_count > 0 else ""
    print(f"     Prefix scores: {format_scores(scores)}{errors_str}")
    return scores


async def _judge_prefixes_streaming(
    state: SamplingState,
    config: SamplingConfig,
    position_map: dict[int, int],
    should_stop: Callable[[], bool],
) -> AsyncIterator[tuple[int, System, dict]]:
    """Judge prefixes and yield results as they complete."""
    print("\n" + "=" * 60)
    print("JUDGING PREFIX TEXTS (STREAMING)")
    print("=" * 60)
    print(f"  Total prefixes to judge: {len(state.nodes)}")

    systems: dict[int, System] = {}
    completed = 0

    # Create tasks with node info
    async def judge_with_id(node: GenerationNode) -> tuple[int, System | Exception]:
        try:
            result = await _judge_prefix(node.prefix, state.questions, config)
            return (node.node_id, result)
        except Exception as e:
            return (node.node_id, e)

    # Use asyncio.as_completed to yield results as they finish
    tasks = [asyncio.create_task(judge_with_id(node)) for node in state.nodes]

    for coro in asyncio.as_completed(tasks):
        # Check stop signal before waiting for next result
        if should_stop():
            print("  -> Stop requested, cancelling remaining prefix judging tasks...")
            for task in tasks:
                if not task.done():
                    task.cancel()
            break

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
    prefill: str,
    continuation: str,
    questions: list[str],
    max_rounds: int,
    config: SamplingConfig,
    should_stop: Callable[[], bool],
) -> AsyncIterator[AlgorithmEvent]:
    """Track how orientation changes word by word.

    Args:
        prompt: The initial prompt
        prefill: Optional starting text for generation (model continues from here)
        continuation: If provided, skip generation and use this text directly
    """
    # Clear convergence cache for fresh analysis session
    _convergence_cache.clear()

    print("\n" + "=" * 60)
    print("STARTING DYNAMICS ANALYSIS")
    print("=" * 60)
    print(f"  Prompt: {truncate_for_log(prompt, 100)}")
    print(f"  Prefill: {truncate_for_log(prefill, 50) if prefill else '(none)'}")
    print(f"  Continuation: {truncate_for_log(continuation, 50) if continuation else '(none)'}")
    print(f"  Max tokens: {config.gen_max_tokens}")
    print(f"  Max rounds: {max_rounds}")
    print(f"  Questions: {len(questions)}")

    # Generate or use provided text
    if continuation:
        # Use provided continuation directly, skip generation
        print(f"\n  -> Using provided continuation directly")
        text = TextComponents(prefill="", generated=continuation)
    elif prefill:
        # Generate with prefill
        print(f"\n  -> Generating with prefill...")
        yield AlgorithmEvent("status", {"message": "Generating from prefill..."})
        gen_client = get_client(config.gen_provider, config.gen_api_key)
        gen_result = await generate_from_llm(
            gen_client,
            config.gen_provider,
            config.gen_model,
            prompt,
            prefill,
            config.gen_max_tokens,
            config.gen_temperature,
        )
        text = TextComponents(prefill=prefill, generated=gen_result.text)
        print(f"  -> Generated: {truncate_for_log(text.generated, 100)}")
        print(f"  -> Full text: {truncate_for_log(text.full, 100)}")
    else:
        # Generate from scratch
        print("\n  -> Generating text from scratch...")
        yield AlgorithmEvent("status", {"message": "Generating text..."})
        gen_client = get_client(config.gen_provider, config.gen_api_key)
        gen_result = await generate_from_llm(
            gen_client,
            config.gen_provider,
            config.gen_model,
            prompt,
            "",
            config.gen_max_tokens,
            config.gen_temperature,
        )
        text = TextComponents(prefill="", generated=gen_result.text)
        print(f"  -> Generated: {truncate_for_log(text.full, 100)}")

    yield AlgorithmEvent(
        "continuation", {
            "text": text.full,
            "generated": text.generated,
            "prefill": text.prefill,
            "length": len(text.full)
        }
    )

    state = build_dynamics_state(prompt, text.full, questions)
    position_map = {i: pos for i, pos in enumerate(get_word_positions(text.full))}
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

    async for node_id, prefix_scores, pos_data in _judge_prefixes_streaming(state, config, position_map, should_stop):
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

    # Check if stopped during prefix judging
    if should_stop():
        print("  -> Stopped during prefix judging phase")
        return

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

            # Build convergence history for all positions
            convergence_history = _build_convergence_history(state)

            positions_data[node_id] = {
                "position": position_map.get(node_id, 0),
                "label": event.data.get("label", ""),
                "core": core,
                "orient_std": orient_std,
                "prefix_system": prefix_system,
                "initial_prefix": prefix_systems.initial,
                "final_prefix": prefix_systems.final,
                "convergence_history": convergence_history.get(node_id, []),
                **metrics,
            }

            # Build all_convergence: dict of node_id -> convergence history
            all_convergence = {
                nid: convergence_history.get(nid, [])
                for nid in state.normativities.keys()
            }

            # Build all_trajectories: dict of node_id -> list of trajectory texts
            all_trajectories = {
                nid: state.normativities[nid].trajectories
                for nid in state.normativities.keys()
            }

            # Build all_samples: dict of node_id -> list of score vectors (for showing badges)
            all_samples = {
                nid: state.normativities[nid].samples
                for nid in state.normativities.keys()
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
                    "all_convergence": all_convergence,
                    "all_trajectories": all_trajectories,
                    "all_samples": all_samples,
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
