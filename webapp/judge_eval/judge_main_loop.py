"""Judge evaluation algorithm - scores multiple texts against questions."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

from webapp.common.algorithm_config import AlgorithmEvent, SamplingConfig
from webapp.common.llm_clients import get_client, judge_all_questions


async def run_judge_algorithm(
    texts: list[str],
    questions: list[str],
    config: SamplingConfig,
    should_stop: Callable[[], bool],
    text_offset: int = 0,
) -> AsyncIterator[AlgorithmEvent]:
    """
    Judge evaluation algorithm.

    Yields events:
        - started: Initial state
        - text_scored: After each text is scored
        - complete: When done

    Args:
        text_offset: Starting index for text_idx (for accumulating results across batches)
    """
    client = get_client(config.judge_provider, config.judge_api_key)
    api_calls = 0
    results = []

    yield AlgorithmEvent(
        "started",
        {
            "mode": "judge",
            "data": {"texts": texts, "questions": questions, "total_texts": len(texts)},
        },
    )

    # Process texts in batches
    batch_size = 3
    for batch_start in range(0, len(texts), batch_size):
        if should_stop():
            break

        # Use text_offset for global indexing across multiple judge calls
        batch = list(
            enumerate(texts[batch_start : batch_start + batch_size], start=text_offset + batch_start)
        )

        async def score_text(idx: int, text: str) -> dict:
            judge_results = await judge_all_questions(
                client,
                config.judge_provider,
                config.judge_model,
                text,
                questions,
                config.judge_prompt,
            )
            return {
                "text_idx": idx,
                "text": text[:100] + ("..." if len(text) > 100 else ""),
                "full_text": text,
                "scores": [r.score for r in judge_results],
                "raw_responses": [r.raw_response for r in judge_results],
                "logprobs": [r.logprob for r in judge_results],
            }

        batch_results = await asyncio.gather(
            *[score_text(idx, text) for idx, text in batch]
        )

        for result in batch_results:
            if should_stop():
                break
            api_calls += len(questions)
            results.append(result)

            yield AlgorithmEvent(
                "text_scored",
                {
                    "text_idx": result["text_idx"],
                    "text": result["text"],
                    "scores": result["scores"],
                    "raw_responses": result["raw_responses"],
                    "logprobs": result["logprobs"],
                    "total_api_calls": api_calls,
                    "progress": len(results) / len(texts),
                    "all_results": results,
                },
            )

        await asyncio.sleep(0.01)

    yield AlgorithmEvent(
        "complete",
        {
            "mode": "judge",
            "data": {
                "texts": texts,
                "questions": questions,
                "results": results,
                "total_api_calls": api_calls,
            },
        },
    )
