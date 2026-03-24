#!/bin/bash
# Run the Normativity Explorer webapp
cd "$(dirname "$0")/.."

# Open Chrome in incognito (ignores cache) after a short delay
(sleep 2 && open -na "Google Chrome" --args --incognito "http://localhost:8001") &

uv run uvicorn webapp.web_server:app --port 8001 --reload
