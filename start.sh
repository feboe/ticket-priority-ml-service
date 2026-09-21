#!/bin/sh
set -eu

uvicorn app.api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

cleanup() {
  kill "$API_PID" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

streamlit run app/ui.py --server.address 0.0.0.0 --server.port 8501