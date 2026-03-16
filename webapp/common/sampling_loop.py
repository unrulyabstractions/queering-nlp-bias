"""Shared sampling logic for tree and dynamics exploration.

Architecture: Streaming pipeline with single task queue.
- All API calls (generation + judging) share one rate-limited queue
- When generation completes, judge tasks are immediately queued
- Results yield as soon as all judges for a node complete
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from webapp.common.algorithm_config import AlgorithmEvent, SamplingConfig
from webapp.common.llm_clients import generate_from_llm, get_client, llm_judge
from webapp.common.normativity_types import (
    GenerationNode,
    NormativityEstimate,
    System,
    compute_system_means,
)
from webapp.common.rate_limited_executor import ExecutorConfig, RateLimitedExecutor


# ════════════════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════════════════


def _truncate(text: str, max_len: int = 80) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _format_scores(scores: list) -> str:
    if not scores:
        return "[]"
    return "[" + ", ".join(f"{s:.3f}" for s in scores) + "]"


def _log(msg: str) -> None:
    print(f"▓ {msg}")


# ════════════════════════════════════════════════════════════════════════════════
# State
# ════════════════════════════════════════════════════════════════════════════════


@dataclass
class SamplingState:
    """State for multi-node sampling exploration."""

    nodes: list[GenerationNode] = field(default_factory=list)
    normativities: dict[int, NormativityEstimate] = field(default_factory=dict)
    questions: list[str] = field(default_factory=list)
    prompt: str = ""
    total_api_calls: int = 0
    _nodes_by_id: dict[int, GenerationNode] | None = field(default=None, repr=False)

    @property
    def total_samples(self) -> int:
        return sum(n.n_samples for n in self.normativities.values())

    @property
    def min_samples(self) -> int:
        if not self.normativities:
            return 0
        return min(n.n_samples for n in self.normativities.values())

    @property
    def nodes_by_id(self) -> dict[int, GenerationNode]:
        if self._nodes_by_id is None:
            self._nodes_by_id = {n.node_id: n for n in self.nodes}
        return self._nodes_by_id

    def get_node(self, node_id: int) -> GenerationNode | None:
        return self.nodes_by_id.get(node_id)

    def get_orientations_relative_to(self, node_id: int) -> list[System]:
        normativity = self.normativities.get(node_id)
        if not normativity or not normativity.core:
            return []
        max_id = max(self.normativities.keys()) + 1
        result = []
        for nid in range(max_id):
            other = self.normativities.get(nid)
            if other and other.samples:
                orientations = [
                    normativity.get_orientation_for(sample) for sample in other.samples
                ]
                result.append(compute_system_means(orientations))
            else:
                result.append([])
        return result


def build_state(
    prompt: str, nodes: list[GenerationNode], questions: list[str]
) -> SamplingState:
    _log(f"Building state: {len(nodes)} nodes, {len(questions)} questions")
    return SamplingState(
        nodes=nodes,
        normativities={n.node_id: NormativityEstimate(node_id=n.node_id) for n in nodes},
        questions=questions,
        prompt=prompt,
    )


def serialize_node(node: GenerationNode, state: SamplingState) -> dict:
    normativity = state.normativities.get(node.node_id)
    return {
        "node_id": node.node_id,
        "name": node.name,
        "label": node.label,
        "prefix": node.prefix,
        "parent": node.parent,
        "depth": node.depth,
        "core": normativity.core,
        "expected_relative_orientations": state.get_orientations_relative_to(node.node_id),
        "n_samples": normativity.n_samples,
        "trajectories": normativity.trajectories,
        "logprob": normativity.mean_logprob,
    }


def serialize_state(state: SamplingState) -> dict:
    return {
        "nodes": [serialize_node(n, state) for n in state.nodes],
        "questions": state.questions,
        "total_samples": state.total_samples,
        "total_api_calls": state.total_api_calls,
    }


# ════════════════════════════════════════════════════════════════════════════════
# Streaming Pipeline
# ════════════════════════════════════════════════════════════════════════════════


async def run_sampling_loop(
    state: SamplingState,
    config: SamplingConfig,
    max_rounds: int,
    should_stop: Callable[[], bool],
    mode: str,
) -> AsyncIterator[AlgorithmEvent]:
    """Streaming pipeline: generation and judging flow through one queue."""

    _log(f"Starting {mode} | {len(state.nodes)} nodes | max_rounds={max_rounds}")
    yield AlgorithmEvent("started", {"mode": mode, "data": serialize_state(state)})

    # Clients - each uses its own API key
    gen_client = get_client(config.gen_provider, config.gen_api_key)
    judge_client = get_client(config.judge_provider, config.judge_api_key)

    # Separate executors for gen and judge (may be different APIs)
    gen_executor = RateLimitedExecutor(ExecutorConfig(max_concurrent=3, max_retries=5))
    judge_executor = RateLimitedExecutor(ExecutorConfig(max_concurrent=3, max_retries=5))

    # Event queue for yielding results
    event_queue: asyncio.Queue[AlgorithmEvent | None] = asyncio.Queue()

    # Track pending work per node
    pending_judges: dict[int, int] = {}  # node_id -> remaining judge count
    node_scores: dict[int, dict[int, float]] = defaultdict(dict)  # node_id -> {q_idx: score}
    node_texts: dict[int, str] = {}  # node_id -> generated text
    node_logprobs: dict[int, float | None] = {}  # node_id -> logprob

    async def do_generate(node: GenerationNode) -> None:
        """Generate for a node, then queue its judge tasks."""
        try:
            result = await gen_executor.execute(
                ("gen", node.node_id),
                generate_from_llm,
                gen_client,
                config.gen_provider,
                config.gen_model,
                state.prompt,
                node.prefix,
                config.max_tokens,
                config.temperature,
            )

            if not result.success:
                _log(f"✗ Gen [{node.node_id}] failed: {result.error}")
                await event_queue.put(AlgorithmEvent("error", {"message": str(result.error)}))
                return

            gen_output = result.result
            node_texts[node.node_id] = gen_output.text
            node_logprobs[node.node_id] = gen_output.logprob
            _log(f"✓ Gen [{node.node_id}]: {len(gen_output.text)} chars")

            # Immediately queue judge tasks
            pending_judges[node.node_id] = len(state.questions)
            for q_idx, question in enumerate(state.questions):
                asyncio.create_task(do_judge(node.node_id, q_idx, question, gen_output.text))

        except Exception as e:
            _log(f"✗ Gen [{node.node_id}] exception: {e}")
            await event_queue.put(AlgorithmEvent("error", {"message": str(e)}))

    async def do_judge(node_id: int, q_idx: int, question: str, text: str) -> None:
        """Judge one question, check if node is complete."""
        try:
            result = await judge_executor.execute(
                ("judge", node_id, q_idx),
                llm_judge,
                judge_client,
                config.judge_provider,
                config.judge_model,
                text,
                question,
                config.judge_prompt,
            )

            if not result.success:
                _log(f"✗ Judge [{node_id}][Q{q_idx}] failed: {result.error}")
                await event_queue.put(AlgorithmEvent("error", {"message": str(result.error)}))
                pending_judges[node_id] -= 1
                return

            node_scores[node_id][q_idx] = result.result.score
            pending_judges[node_id] -= 1

            # Check if all judges for this node are done
            if pending_judges[node_id] == 0:
                await finalize_node(node_id)

        except Exception as e:
            _log(f"✗ Judge [{node_id}][Q{q_idx}] exception: {e}")
            await event_queue.put(AlgorithmEvent("error", {"message": str(e)}))
            pending_judges[node_id] -= 1

    async def finalize_node(node_id: int) -> None:
        """All judges done for this node - update state and yield event."""
        node = state.get_node(node_id)
        text = node_texts[node_id]
        logprob = node_logprobs.get(node_id)

        # Build ordered scores
        scores = [node_scores[node_id][i] for i in range(len(state.questions))]

        # Update state
        state.normativities[node_id].samples.append(scores)
        state.normativities[node_id].trajectories.append(text)
        if logprob is not None:
            state.normativities[node_id].logprobs.append(logprob)
        state.total_api_calls += 1 + len(state.questions)

        normativity = state.normativities[node_id]
        _log(f"★ Node [{node_id}] complete: {_format_scores(scores)}")

        # Yield event
        await event_queue.put(AlgorithmEvent("point_update", {
            "node_id": node_id,
            "label": node.label,
            "core": normativity.core,
            "orient_std": normativity.orient_std,
            "expected_relative_orientations": state.get_orientations_relative_to(node_id),
            "n_samples": normativity.n_samples,
            "total_samples": state.total_samples,
            "total_api_calls": state.total_api_calls,
            "scores": scores,
            "trajectory": text,
            "logprob": normativity.mean_logprob,
        }))

        # Clear per-round tracking for this node
        del node_scores[node_id]
        del node_texts[node_id]
        if node_id in node_logprobs:
            del node_logprobs[node_id]

    async def run_round() -> int:
        """Run one sampling round. Returns number of successful samples."""
        # Clear tracking
        pending_judges.clear()
        node_scores.clear()
        node_texts.clear()
        node_logprobs.clear()

        # Check stop before starting
        if should_stop():
            return 0

        # Start all generation tasks
        gen_tasks = [asyncio.create_task(do_generate(node)) for node in state.nodes]

        # Wait for all work to complete, but check stop frequently
        done, pending = await asyncio.wait(gen_tasks, timeout=0.1)
        while pending:
            if should_stop():
                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                _log("Stop requested - cancelling generation tasks")
                return 0
            done, pending = await asyncio.wait(pending, timeout=0.1)

        # Wait for any remaining judge tasks, checking stop
        while any(pending_judges.values()):
            if should_stop():
                _log("Stop requested - abandoning judge tasks")
                return 0
            await asyncio.sleep(0.05)

        return len([n for n in state.nodes if state.normativities[n.node_id].n_samples > 0])

    # Consumer: yield events as they arrive
    async def consume_events() -> AsyncIterator[AlgorithmEvent]:
        while True:
            event = await event_queue.get()
            if event is None:
                break
            yield event

    # Run rounds
    round_num = 0
    while not should_stop() and state.min_samples < max_rounds:
        round_num += 1
        _log(f"═══ Round {round_num}/{max_rounds} ═══")

        # Run the round (this populates event_queue)
        round_task = asyncio.create_task(run_round())

        # Yield events as they arrive during the round
        while not round_task.done():
            # Check stop during event consumption
            if should_stop():
                round_task.cancel()
                _log("Stop requested - breaking out of round")
                break
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                yield event
            except asyncio.TimeoutError:
                continue

        # If stopped, don't wait for round completion
        if should_stop():
            break

        await round_task

        # Drain any remaining events
        while not event_queue.empty():
            yield await event_queue.get()

        _log(f"Round {round_num} complete | samples/node: {state.min_samples}")

    _log(f"Sampling complete | total: {state.total_samples}")
    yield AlgorithmEvent("complete", {"mode": mode, "data": serialize_state(state)})
