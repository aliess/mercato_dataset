#!/usr/bin/env python3
"""
Download the transfermarkt-datasets tables build_dataset.py needs (~25 MB gzipped)
from the project's public R2 bucket, the same files Kaggle publishes.

Upstream stopped updating in July 2026 (github.com/dcaribou/transfermarkt-datasets/discussions/383);
build_dataset.py fills the gap from the Transfermarkt API. Re-run this if upstream resumes.

Usage:
    python scripts/download_dataset.py
"""

import argparse
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_URL = 'https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data'
TABLES = ['players', 'player_valuations', 'clubs', 'competitions', 'countries', 'transfers']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset-dir', type=Path, default=ROOT / 'dataset')
    args = parser.parse_args()
    args.dataset_dir.mkdir(parents=True, exist_ok=True)

    for table in TABLES:
        target = args.dataset_dir / f'{table}.csv.gz'
        partial = target.with_suffix('.gz.part')
        request = urllib.request.Request(f'{BASE_URL}/{table}.csv.gz', headers={'User-Agent': 'mercato-dataset'})
        with urllib.request.urlopen(request, timeout=120) as response, open(partial, 'wb') as out:
            shutil.copyfileobj(response, out)
        partial.replace(target)
        print(f'✓ {target.name} ({target.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
