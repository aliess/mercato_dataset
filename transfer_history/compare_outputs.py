#!/usr/bin/env python3
"""
Compare two builds (output/previous/ vs output/) and flag anything that looks wrong.
build_dataset.py runs this automatically; the report is saved to output/compare_report.txt.

Usage:
    python transfer_history/compare_outputs.py [old_dir] [new_dir]
"""

import sys
from datetime import date
from pathlib import Path

import pandas as pd

MODE_DIR = Path(__file__).resolve().parent
PROFILES = 'player_profiles.csv'
TRANSFERS = 'transfer_history.csv'

# A new build that shrinks more than this is probably broken (API down, bad download…).
MAX_PLAYER_DROP = 0.05
MAX_TRANSFER_DROP = 0.05


def latest_clubs(transfers):
    """player_id → (to_team_name, transfer_date) of the most recent transfer."""
    last = transfers.sort_values(['transfer_date', 'to_team_id']).groupby('player_id').tail(1)
    return last.set_index('player_id')[['to_team_id', 'to_team_name', 'transfer_date']]


def names_list(profiles, ids, limit=15):
    names = profiles.set_index('player_id').reindex(sorted(ids))['player_name'].fillna('?').tolist()
    more = f' … +{len(names) - limit}' if len(names) > limit else ''
    return ', '.join(names[:limit]) + more


def compare(old_dir, new_dir):
    """Returns (report lines, warnings)."""
    old_p, new_p = pd.read_csv(old_dir / PROFILES), pd.read_csv(new_dir / PROFILES)
    old_t, new_t = pd.read_csv(old_dir / TRANSFERS), pd.read_csv(new_dir / TRANSFERS)
    lines, warnings = [], []
    out = lines.append

    old_ids, new_ids = set(old_p['player_id']), set(new_p['player_id'])
    added, removed = new_ids - old_ids, old_ids - new_ids
    out(f'Players:   {len(old_ids):,} → {len(new_ids):,}  (+{len(added):,} / -{len(removed):,})')
    out(f'Transfers: {len(old_t):,} → {len(new_t):,}')
    if added:
        out(f'  added:   {names_list(new_p, added)}')
    if removed:
        out(f'  removed: {names_list(old_p, removed)}')

    no_transfers_old = old_ids - set(old_t['player_id'])
    no_transfers_new = new_ids - set(new_t['player_id'])
    out(f'Players without transfers (not playable): {len(no_transfers_old):,} → {len(no_transfers_new):,}')
    if no_transfers_new:
        out(f'  {names_list(new_p, no_transfers_new)}')

    # Players who lost transfers they had before (history should only grow).
    common = old_ids & new_ids
    old_counts = old_t[old_t['player_id'].isin(common)].groupby('player_id').size()
    new_counts = new_t[new_t['player_id'].isin(common)].groupby('player_id').size().reindex(old_counts.index, fill_value=0)
    shrunk = old_counts[new_counts < old_counts]
    out(f'Players with fewer transfers than before: {len(shrunk):,}')
    if len(shrunk):
        worst = (old_counts - new_counts)[shrunk.index].sort_values(ascending=False).head(10)
        names = new_p.set_index('player_id')['player_name']
        out('  ' + ', '.join(f'{names.get(pid, pid)} ({old_counts[pid]}→{new_counts[pid]})' for pid in worst.index))

    # Latest club changes = this window's moves.
    old_last, new_last = latest_clubs(old_t), latest_clubs(new_t)
    both = old_last.index.intersection(new_last.index)
    moved = new_last.loc[both][old_last.loc[both, 'to_team_id'] != new_last.loc[both, 'to_team_id']]
    moved = moved.sort_values('transfer_date', ascending=False)
    out(f'Players whose latest club changed: {len(moved):,}')
    names = new_p.set_index('player_id')['player_name']
    for pid, row in moved.head(25).iterrows():
        out(f'  {row["transfer_date"]}  {names.get(pid, pid)}: {old_last.loc[pid, "to_team_name"]} → {row["to_team_name"]}')
    if len(moved) > 25:
        out(f'  … +{len(moved) - 25} more')

    # Builds leave out moves dated after the build day, so only compare what has happened.
    today = date.today().isoformat()
    old_latest = old_t.loc[old_t['transfer_date'] <= today, 'transfer_date'].max()
    new_latest = new_t.loc[new_t['transfer_date'] <= today, 'transfer_date'].max()
    out(f'Latest transfer date: {old_latest} → {new_latest}')

    # Sanity checks.
    if len(new_ids) < len(old_ids) * (1 - MAX_PLAYER_DROP):
        warnings.append(f'player count dropped {len(old_ids):,} → {len(new_ids):,}')
    if len(new_t) < len(old_t) * (1 - MAX_TRANSFER_DROP):
        warnings.append(f'transfer count dropped {len(old_t):,} → {len(new_t):,}')
    if len(no_transfers_new) > len(no_transfers_old):
        warnings.append(f'{len(no_transfers_new):,} players have no transfers (was {len(no_transfers_old):,})')
    if new_latest < old_latest:
        warnings.append('latest transfer is older than in the previous build')
    if len(shrunk) > 0.05 * max(len(common), 1):
        warnings.append(f'{len(shrunk):,} players lost transfers')
    for column in ('player_name', 'market_value'):
        if new_p[column].isna().any():
            warnings.append(f'{new_p[column].isna().sum():,} players without {column}')
    return lines, warnings


def report(old_dir, new_dir, save_to=None):
    lines, warnings = compare(old_dir, new_dir)
    lines = [f'Comparing {old_dir} → {new_dir}', ''] + lines + ['']
    lines += [f'⚠ {w}' for w in warnings] or ['✓ No problems found']
    text = '\n'.join(lines)
    print(text)
    if save_to:
        Path(save_to).write_text(text + '\n')
    return warnings


def main():
    old_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else MODE_DIR / 'output' / 'previous'
    new_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else MODE_DIR / 'output'
    sys.exit(1 if report(old_dir, new_dir) else 0)


if __name__ == '__main__':
    main()
