#!/usr/bin/env bash
# Refresh the player pool and transfer history: download → build → sync.
# Usage: ./update.sh dev|prod [--apply] [--credentials key.json]
set -euo pipefail
cd "$(dirname "$0")"

if [ $# -lt 1 ]; then
  echo "usage: ./update.sh dev|prod [--apply] [--credentials key.json]" >&2
  exit 1
fi
project=$1
shift

python sources/download_dataset.py
python game_modes_data/transfer_history/build_dataset.py
python game_modes_data/transfer_history/sync_firestore.py --project "$project" "$@"
