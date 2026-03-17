"""Dynamics Analysis WebSocket handler."""

from __future__ import annotations

import asyncio

from fastapi import WebSocket

from webapp.common.algorithm_config import SamplingConfig
from webapp.common.ui.console_output import log_section, log_timestamped
from webapp.dynamics_analysis.dynamics_main_loop import track_text_dynamics


async def run_dynamics(ws: WebSocket, session: dict, data: dict) -> None:
    """Handle dynamics analysis via WebSocket."""
    config = SamplingConfig.from_request(data)

    if error := config.validate_api_keys(need_gen=True, need_judge=True):
        return await ws.send_json({"type": "error", "message": error})

    questions = data.get("questions", [])
    if not questions:
        return await ws.send_json({"type": "error", "message": "Need >=1 question"})

    session.update(running=True, stop=False, mode="dynamics")
    log_section(f"Dynamics ({config.gen_provider}/{config.gen_model})")

    try:
        async for event in track_text_dynamics(
            prompt=data.get("prompt", ""),
            continuation=data.get("continuation", ""),
            questions=questions,
            max_tokens=data.get("max_tokens", 300),
            max_rounds=data.get("max_rounds", 10),
            config=config,
            should_stop=lambda: session.get("stop", False),
        ):
            if event.type == "point_update":
                # Legacy point_update from sampling loop - convert to position_update
                scores = event.data.get("scores") or event.data.get("core") or []
                log_timestamped(
                    f"  {event.data.get('label', '?')}: {[f'{s:.2f}' for s in scores]}"
                )
                await ws.send_json({"type": "position_update", "data": event.data})
            elif event.type == "position_update":
                # New position_update from streaming prefix judging
                prefix = event.data.get("prefix_system") or []
                core = event.data.get("core") or []
                scores_str = [f"{s:.2f}" for s in (core if core else prefix)]
                log_timestamped(
                    f"  [{event.data.get('node_id', '?')}] {event.data.get('label', '?')}: {scores_str}"
                )
                await ws.send_json({"type": "position_update", "data": event.data})
            elif event.type == "status":
                await ws.send_json({"type": event.type, "message": event.data["message"]})
            else:
                await ws.send_json({"type": event.type, **event.data})

        log_section("Dynamics complete")

    except asyncio.CancelledError:
        log_section("Dynamics cancelled")

    finally:
        session["running"] = False
