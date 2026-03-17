"""
Web Server for interactive NLP bias exploration.

Three modes: Forking and Localizing Normativity | Dynamics of Meaning | Judge LLM
Run: ./webapp/run.sh or uv run uvicorn webapp.web_server:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from webapp.app_settings import AVAILABLE_MODELS, DEFAULT_SETTINGS, PROVIDER_DISPLAY_NAMES
from webapp.common.ui.html_template import get_html_template
from webapp.common.ui.ui_text_config import APP_TITLE
from webapp.dynamics_analysis.dynamics_websocket_handler import run_dynamics
from webapp.judge_eval.judge_websocket_handler import run_judge
from webapp.tree_exploration.tree_websocket_handler import run_tree

app = FastAPI(title=APP_TITLE)

# Serve static files (images, etc.)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
sessions: dict[str, dict[str, Any]] = {}


@app.get("/config")
async def get_config():
    return JSONResponse(
        {
            "anthropic_key": os.environ.get("ANTHROPIC_API_KEY", ""),
            "openai_key": os.environ.get("OPENAI_API_KEY", ""),
            "models": AVAILABLE_MODELS,
            "provider_names": PROVIDER_DISPLAY_NAMES,
            "defaults": DEFAULT_SETTINGS,
        }
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    return get_html_template()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    sid = str(id(ws))
    sessions[sid] = {"state": None, "running": False, "stop": False, "mode": None, "task": None}

    try:
        while True:
            data = await ws.receive_json()
            action = data.get("action")

            if action == "stop":
                sessions[sid]["stop"] = True
                # Cancel the running task if it exists
                task = sessions[sid].get("task")
                if task and not task.done():
                    task.cancel()
                continue

            # Cancel any existing task before starting a new one
            existing_task = sessions[sid].get("task")
            if existing_task and not existing_task.done():
                sessions[sid]["stop"] = True
                existing_task.cancel()
                try:
                    await existing_task
                except asyncio.CancelledError:
                    pass

            # Reset stop flag for new task
            sessions[sid]["stop"] = False

            if action == "start_tree":
                sessions[sid]["task"] = asyncio.create_task(run_tree(ws, sessions[sid], data))
            elif action == "start_dynamics":
                sessions[sid]["task"] = asyncio.create_task(run_dynamics(ws, sessions[sid], data))
            elif action == "start_judge":
                sessions[sid]["task"] = asyncio.create_task(run_judge(ws, sessions[sid], data))
    except WebSocketDisconnect:
        # Cancel any running task on disconnect
        task = sessions[sid].get("task")
        if task and not task.done():
            task.cancel()
        sessions.pop(sid, None)
