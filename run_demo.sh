#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
export DEMO_MODE=1
cd "$ROOT/backend_api"
source "$ROOT/.venv/bin/activate"
exec uvicorn server:app --reload --host 127.0.0.1 --port 8000
