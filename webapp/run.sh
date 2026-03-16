#!/bin/bash
# Run the Normativity Explorer webapp
cd "$(dirname "$0")/.."

# Open Chrome after a short delay
(sleep 2 && open -a "Google Chrome" "http://localhost:8000") &

uv run uvicorn webapp.web_server:app --port 8000 --reload
