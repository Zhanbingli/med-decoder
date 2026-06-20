#!/usr/bin/env bash
# Launch the CardioVoice web app: FastAPI backend (:8000) + Vite frontend (:5173).
# Prereqs: conda env `medgemma` active, and `ollama serve` running.
set -e
cd "$(dirname "$0")"

echo "Starting API on http://127.0.0.1:8000 ..."
uvicorn server.main:app --port 8000 &
BACKEND=$!
trap "kill $BACKEND 2>/dev/null" EXIT

cd web
[ -d node_modules ] || npm install
echo "Starting web UI on http://localhost:5173 ..."
npm run dev
