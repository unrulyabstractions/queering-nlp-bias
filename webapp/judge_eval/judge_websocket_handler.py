"""Judge Evaluation WebSocket handler."""

from __future__ import annotations

import asyncio

from fastapi import WebSocket

from webapp.common.algorithm_config import SamplingConfig
from webapp.common.ui.console_output import log_section, log_timestamped
from webapp.judge_eval.judge_main_loop import run_judge_algorithm


async def run_judge(ws: WebSocket, session: dict, data: dict) -> None:
    """Handle judge evaluation via WebSocket."""
    config = SamplingConfig.from_request(data)

    if error := config.validate_api_keys(need_gen=False, need_judge=True):
        return await ws.send_json({"type": "error", "message": error})

    texts = data.get("texts", [])
    questions = data.get("questions", [])
    text_offset = data.get("text_offset", 0)

    if not texts:
        return await ws.send_json({"type": "error", "message": "Need ≥1 text"})
    if not questions:
        return await ws.send_json({"type": "error", "message": "Need ≥1 question"})

    session.update(running=True, stop=False, mode="judge")
    judge_models_str = ", ".join(f"{m.provider}/{m.model}" for m in config.judge_models)
    log_section(f"Judge ({judge_models_str})")

    try:
        async for event in run_judge_algorithm(
            texts=texts,
            questions=questions,
            config=config,
            should_stop=lambda: session.get("stop", False),
            text_offset=text_offset,
        ):
            if event.type == "text_scored":
                # Handle None scores in logging
                scores_str = [f'{s:.2f}' if s is not None else 'ERR' for s in event.data['scores']]
                log_timestamped(f"  text {event.data['text_idx']}: {scores_str}")
                await ws.send_json({"type": event.type, "data": event.data})
            else:
                await ws.send_json({"type": event.type, **event.data})

        log_section("Judge complete")

    except asyncio.CancelledError:
        log_section("Judge cancelled")

    finally:
        session["running"] = False
