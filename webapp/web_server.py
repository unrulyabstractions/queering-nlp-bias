"""
Queering NLP Bias! - Web Server

Three modes: Forking and Localizing Normativity | Dynamics of Meaning | Judge
Run: ./webapp/run.sh or uv run uvicorn webapp.web_server:app --reload --port 8000
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from webapp.app_settings import AVAILABLE_MODELS, DEFAULT_SETTINGS
from webapp.common.ui.html_template import get_html_template
from webapp.dynamics_analysis.dynamics_websocket_handler import run_dynamics
from webapp.judge_eval.judge_websocket_handler import run_judge
from webapp.tree_exploration.tree_websocket_handler import run_tree

app = FastAPI(title="Queering NLP Bias!")

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
    sessions[sid] = {"state": None, "running": False, "stop": False, "mode": None}

    try:
        while True:
            data = await ws.receive_json()
            action = data.get("action")
            if action == "start_tree":
                await run_tree(ws, sessions[sid], data)
            elif action == "start_dynamics":
                await run_dynamics(ws, sessions[sid], data)
            elif action == "start_judge":
                await run_judge(ws, sessions[sid], data)
            elif action == "stop":
                sessions[sid]["stop"] = True
    except WebSocketDisconnect:
        sessions.pop(sid, None)
