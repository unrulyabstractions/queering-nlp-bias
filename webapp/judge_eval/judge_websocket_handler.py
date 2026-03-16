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

    if not config.judge_api_key:
        return await ws.send_json({"type": "error", "message": "No API key"})

    texts = data.get("texts", [])
    questions = data.get("questions", [])
    text_offset = data.get("text_offset", 0)

    if not texts:
        return await ws.send_json({"type": "error", "message": "Need ≥1 text"})
    if not questions:
        return await ws.send_json({"type": "error", "message": "Need ≥1 question"})

    session.update(running=True, stop=False, mode="judge")
    log_section(f"Judge ({config.judge_provider}/{config.judge_model})")

    try:
        async for event in run_judge_algorithm(
            texts=texts,
            questions=questions,
            config=config,
            should_stop=lambda: session.get("stop", False),
            text_offset=text_offset,
        ):
            if event.type == "text_scored":
                log_timestamped(
                    f"  text {event.data['text_idx']}: {[f'{s:.2f}' for s in event.data['scores']]}"
                )
                await ws.send_json({"type": event.type, "data": event.data})
            else:
                await ws.send_json({"type": event.type, **event.data})

        log_section("Judge complete")

    except asyncio.CancelledError:
        log_section("Judge cancelled")

    finally:
        session["running"] = False
