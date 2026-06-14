#!/usr/bin/env bash
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
  --exclude '*.csv' \
  --exclude '*.jsonl' \
  --exclude '*.pth' \
  --exclude '*.pt' \
  --exclude '*.safetensors' \
  --exclude '*.pdf' \
  --exclude '.DS_Store' \
  "$ROOT/" "$EXPORT_DIR/"

cd "$EXPORT_DIR"
git init
git add .
git commit -m "$(cat <<'EOF'
Initial public portfolio release (demo mode only).

Ships application code without datasets, weights, or proprietary documents.
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

Before pushing, verify no large files or licensed dataset images:

  find . -type f -size +10M
  find . -path './frontend/demo_assets/sample_cxr.png' -prune -o -name '*.png' -print

Only frontend/demo_assets/sample_cxr.png (synthetic demo) should appear among PNG files.

Capture portfolio screenshots while running ./run_demo.sh and ./run_frontend.sh.

MSG
