#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/frontend"
source "$ROOT/.venv/bin/activate"
exec streamlit run frontend.py
