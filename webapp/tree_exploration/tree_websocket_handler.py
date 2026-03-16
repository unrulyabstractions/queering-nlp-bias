"""Tree Exploration WebSocket handler."""

from __future__ import annotations

import asyncio

from fastapi import WebSocket

from webapp.app_settings import DEFAULT_SAMPLES_PER_NODE
from webapp.common.algorithm_config import SamplingConfig
from webapp.common.ui.console_output import log_section, log_timestamped
from webapp.tree_exploration.tree_main_loop import explore_prefix_tree


async def run_tree(ws: WebSocket, session: dict, data: dict) -> None:
    """Handle tree exploration via WebSocket."""
    config = SamplingConfig.from_request(data)

    if not config.gen_api_key and not config.judge_api_key:
        return await ws.send_json({"type": "error", "message": "No API key"})

    prefixes = data.get("prefixes", [])
    print(f"\n▓▓▓ TREE HANDLER: Received {len(prefixes)} prefixes:")
    for i, p in enumerate(prefixes):
        print(f"▓▓▓   [{i}] '{p[:50]}...'")

    if len(prefixes) < 1:
        return await ws.send_json({"type": "error", "message": "Need at least 1 prefix"})

    session.update(running=True, stop=False, mode="tree")
    log_section(f"Tree ({config.gen_provider}/{config.gen_model})")

    try:
        event_count = 0
        async for event in explore_prefix_tree(
            prompt=data.get("prompt", ""),
            prefixes=prefixes,
            questions=data.get("questions", []),
            max_rounds=data.get("max_rounds", DEFAULT_SAMPLES_PER_NODE),
            config=config,
            should_stop=lambda: session.get("stop", False),
        ):
            event_count += 1
            print(f"\n▓▓▓ TREE HANDLER EVENT #{event_count}: type={event.type}")

            if event.type == "point_update":
                node_id = event.data.get("node_id")
                label = event.data.get("label")
                n_samples = event.data.get("n_samples")
                scores = event.data.get("scores", [])
                print(f"▓▓▓   node_id={node_id} label='{label}' n_samples={n_samples}")
                print(f"▓▓▓   scores={[f'{s:.3f}' for s in scores]}")
                print(f"▓▓▓   Sending 'node_update' to WebSocket...")
                log_timestamped(
                    f"  {label}: {[f'{s:.2f}' for s in scores]}"
                )
                await ws.send_json({"type": "node_update", "data": event.data})
                print(f"▓▓▓   Sent successfully!")

            elif event.type == "error":
                print(f"▓▓▓   ERROR: {event.data.get('message', 'unknown')}")
                log_timestamped(f"  ERROR: {event.data['message']}")
                await ws.send_json({"type": "error", "data": event.data})

            elif event.type == "started":
                print(f"▓▓▓   Started event - mode={event.data.get('mode')}")
                print(f"▓▓▓   Initial nodes: {len(event.data.get('data', {}).get('nodes', []))}")
                await ws.send_json({"type": event.type, **event.data})

            elif event.type == "complete":
                print(f"▓▓▓   Complete event - mode={event.data.get('mode')}")
                total = event.data.get("data", {}).get("total_samples", 0)
                print(f"▓▓▓   Total samples: {total}")
                await ws.send_json({"type": event.type, **event.data})

            else:
                print(f"▓▓▓   Other event type: {event.type}")
                await ws.send_json({"type": event.type, **event.data})

        print(f"\n▓▓▓ TREE HANDLER COMPLETE: Sent {event_count} events total")
        log_section("Tree complete")

    except asyncio.CancelledError:
        print("\n▓▓▓ TREE HANDLER CANCELLED")
        log_section("Tree cancelled")

    finally:
        session["running"] = False
