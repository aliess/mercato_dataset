#!/usr/bin/env python3
"""
Download transfermarkt-datasets tables from the project's public R2 bucket, the same
files Kaggle publishes (github.com/dcaribou/transfermarkt-datasets). No account needed.

By default it fetches the six tables transfer_history/build_dataset.py needs (~25 MB
gzipped) into sources/dataset/. Starting XI also uses games and game_lineups (~130 MB).

Upstream stopped updating in July 2026 (github.com/dcaribou/transfermarkt-datasets/discussions/383);
build_dataset.py fills the gap from the Transfermarkt API. Re-run this if upstream resumes.

Usage:
    python sources/download_dataset.py
    python sources/download_dataset.py --tables games game_lineups
"""

import argparse
import shutil
import urllib.request
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent / 'dataset'
BASE_URL = 'https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data'
TABLES = ['players', 'player_valuations', 'clubs', 'competitions', 'countries', 'transfers']


def download(tables=TABLES, dataset_dir=DATASET_DIR, only_missing=False):
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for table in tables:
        target = dataset_dir / f'{table}.csv.gz'
        if only_missing and target.exists():
            continue
        partial = target.with_suffix('.gz.part')
        request = urllib.request.Request(f'{BASE_URL}/{table}.csv.gz', headers={'User-Agent': 'mercato-dataset'})
        with urllib.request.urlopen(request, timeout=300) as response, open(partial, 'wb') as out:
            shutil.copyfileobj(response, out)
        partial.replace(target)
        print(f'✓ {target.name} ({target.stat().st_size / 1e6:.1f} MB)')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset-dir', type=Path, default=DATASET_DIR)
    parser.add_argument('--tables', nargs='+', default=TABLES, metavar='TABLE')
    args = parser.parse_args()
    download(args.tables, args.dataset_dir)


if __name__ == '__main__':
    main()
