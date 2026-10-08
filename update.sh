#!/usr/bin/env bash
# Download → build → sync. Usage: ./update.sh dev|prod [--apply] [--credentials key.json]
set -euo pipefail
cd "$(dirname "$0")"

if [ $# -lt 1 ]; then
  echo "usage: ./update.sh dev|prod [--apply] [--credentials key.json]" >&2
  exit 1
fi
project=$1
shift

python scripts/download_dataset.py
python scripts/build_dataset.py
python scripts/sync_firestore.py --project "$project" "$@"
