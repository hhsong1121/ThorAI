#!/usr/bin/env bash
# Creates a clean export suitable for a NEW public GitHub repository (no weight history).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXPORT_DIR="${1:-$ROOT/../thoracic-cdss-portfolio-export}"

echo "Export directory: $EXPORT_DIR"
rm -rf "$EXPORT_DIR"
mkdir -p "$EXPORT_DIR"

rsync -a \
  --exclude '.git' \
  --exclude '.cursor' \
  --exclude '.venv' \
  --exclude '.venv-1' \
  --exclude 'venv' \
  --exclude '__pycache__' \
  --exclude '.env' \
  --exclude 'backend_api/adapters/*.safetensors' \
  --exclude 'backend_api/weights' \
  --exclude 'database' \
  --exclude 'research_and_training/data' \
  --exclude 'mlx_data' \
  --exclude 'qdrant_db' \
  --exclude '*.csv' \
  --exclude '*.jsonl' \
  --exclude '*.pth' \
  --exclude '*.pt' \
  --exclude '*.pdf' \
  --exclude '.DS_Store' \
  "$ROOT/" "$EXPORT_DIR/"

cd "$EXPORT_DIR"
git init
git add .
git commit -m "$(cat <<'EOF'
Initial public portfolio release with Demo Mode.

Ships application code and training reference scripts without datasets or proprietary weights.
EOF
)"

cat <<MSG

Export ready at: $EXPORT_DIR

Next steps (run manually):

  cd "$EXPORT_DIR"
  gh repo create YOUR_USERNAME/thoracic-cdss --public --source=. --remote=origin --push

Or on GitHub.com: create an empty public repo, then:

  git remote add origin git@github.com:YOUR_USERNAME/thoracic-cdss.git
  git branch -M main
  git push -u origin main

Before pushing, verify no large files:

  find . -type f -size +10M

Capture portfolio screenshots while running ./run_demo.sh and ./run_frontend.sh.

MSG
