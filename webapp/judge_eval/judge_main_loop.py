"""Judge evaluation algorithm - scores multiple texts against questions.

Supports multi-judge with multiple providers: when config.judge_models has
multiple models (potentially from different providers), returns per-model
results plus averaged results for comparison.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

from webapp.common.algorithm_config import AlgorithmEvent, JudgeModelSpec, SamplingConfig
from webapp.common.llm_clients import get_client, judge_all_questions


async def run_judge_algorithm(
    texts: list[str],
    questions: list[str],
    config: SamplingConfig,
    should_stop: Callable[[], bool],
    text_offset: int = 0,
) -> AsyncIterator[AlgorithmEvent]:
    """
    Judge evaluation algorithm with multi-judge and multi-provider support.

    Yields events:
        - started: Initial state
        - text_scored: After each text is scored (includes per-model and averaged results)
        - complete: When done

    Args:
        text_offset: Starting index for text_idx (for accumulating results across batches)

    When multiple judge models are configured, results include:
        - results_by_model: {model_key: [result, result, ...]} - individual model results
        - averaged_results: [result, ...] - averaged across all models
    """
    clients: dict[str, object] = {}
    for spec in config.judge_models:
        if spec.provider not in clients:
            clients[spec.provider] = get_client(spec.provider, config.api_keys.get(spec.provider, ""))

    api_calls = 0
    results: list[dict] = []
    model_keys = [f"{m.provider}/{m.model}" for m in config.judge_models]
    results_by_model: dict[str, list[dict]] = {k: [] for k in model_keys}
    judge_models_data = [m.to_dict() for m in config.judge_models]

    yield AlgorithmEvent(
        "started",
        {
            "mode": "judge",
            "data": {
                "texts": texts,
                "questions": questions,
                "total_texts": len(texts),
                "judge_models": judge_models_data,
            },
        },
    )

    async def score_text_with_all_models(idx: int, text: str) -> dict:
        """Score a single text with all judge models concurrently."""
        async def score_with_model(spec: JudgeModelSpec) -> dict:
            client = clients[spec.provider]
            judge_results = await judge_all_questions(
                client,
                spec.provider,
                spec.model,
                text,
                questions,
                config.judge_prompt,
                config.judge_temperature,
                config.judge_max_tokens,
            )
            model_key = f"{spec.provider}/{spec.model}"
            return {
                "text_idx": idx,
                "model": model_key,
                "provider": spec.provider,
                "model_name": spec.model,
                "text": text[:100] + ("..." if len(text) > 100 else ""),
                "full_text": text,
                "scores": [r.score for r in judge_results],
                "raw_responses": [r.raw_response for r in judge_results],
                "logprobs": [r.logprob for r in judge_results],
            }

        model_results = await asyncio.gather(*[
            score_with_model(spec) for spec in config.judge_models
        ])
        return {"idx": idx, "text": text, "model_results": list(model_results)}

    tasks = [
        asyncio.create_task(score_text_with_all_models(text_offset + i, text))
        for i, text in enumerate(texts)
    ]

    for coro in asyncio.as_completed(tasks):
        if should_stop():
            for task in tasks:
                task.cancel()
            break

        result = await coro
        idx = result["idx"]
        text = result["text"]
        model_results = result["model_results"]

        for model_result in model_results:
            model_key = model_result["model"]
            results_by_model[model_key].append(model_result)
            api_calls += len(questions)

        all_scores = [r["scores"] for r in model_results]
        averaged_scores = []
        for q_idx in range(len(questions)):
            valid_scores = [scores[q_idx] for scores in all_scores if scores[q_idx] is not None]
            if valid_scores:
                averaged_scores.append(sum(valid_scores) / len(valid_scores))
            else:
                averaged_scores.append(None)

        if len(config.judge_models) > 1:
            model_label = "averaged"
        else:
            spec = config.judge_models[0]
            model_label = f"{spec.provider}/{spec.model}"

        combined_raw_responses = []
        for q_idx in range(len(questions)):
            q_responses = []
            for r in model_results:
                raw = r["raw_responses"][q_idx] if q_idx < len(r.get("raw_responses", [])) else "N/A"
                q_responses.append(f"[{r['model'].split('/')[-1]}]: {raw}")
            combined_raw_responses.append(" | ".join(q_responses))

        averaged_result = {
            "text_idx": idx,
            "model": model_label,
            "text": text[:100] + ("..." if len(text) > 100 else ""),
            "full_text": text,
            "scores": averaged_scores,
            "raw_responses": combined_raw_responses,
            "logprobs": None,
            "per_model_scores": {r["model"]: r["scores"] for r in model_results},
        }
        results.append(averaged_result)

        yield AlgorithmEvent(
            "text_scored",
            {
                "text_idx": idx,
                "text": averaged_result["text"],
                "scores": averaged_result["scores"],
                "raw_responses": averaged_result["raw_responses"],
                "logprobs": averaged_result["logprobs"],
                "per_model_scores": averaged_result["per_model_scores"],
                "total_api_calls": api_calls,
                "progress": len(results) / len(texts),
                "all_results": results,
                "results_by_model": results_by_model,
            },
        )

    yield AlgorithmEvent(
        "complete",
        {
            "mode": "judge",
            "data": {
                "texts": texts,
                "questions": questions,
                "judge_models": judge_models_data,
                "results": results,
                "results_by_model": results_by_model,
                "total_api_calls": api_calls,
            },
        },
    )
