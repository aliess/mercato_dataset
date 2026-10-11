#!/usr/bin/env python3
"""
Check where build_lineups.py puts each starter on the pitch (output/xi_lineups.json).

Counts lineups where:
  - a winger, wide midfielder or full-back is in the middle (x = 0.5),
  - a left-sided position is right of centre, or a right-sided one left of centre,
  - a centre-forward or second striker is deeper (larger y) than a winger,
  - a striker sits wide while a winger of that side is more central (out of left-right order).

Usage:
    python game_modes_data/starting_xi/check_positions.py [path]   # exits 1 when anything is found
"""

import json
import sys
from collections import Counter
from pathlib import Path

LEFT = {'Left-Back', 'Left Midfield', 'Left Winger'}
RIGHT = {'Right-Back', 'Right Midfield', 'Right Winger'}
WINGERS = {'Left Winger', 'Right Winger'}
STRIKERS = {'Centre-Forward', 'Second Striker'}


def problems(match):
    found = set()
    slots = match['slots']
    for s in slots:
        if s['position'] in LEFT | RIGHT and s['x'] == 0.5:
            found.add(f'{s["position"]} in the middle')
        if s['position'] in LEFT and s['x'] > 0.5:
            found.add(f'{s["position"]} right of centre')
        if s['position'] in RIGHT and s['x'] < 0.5:
            found.add(f'{s["position"]} left of centre')
    for st in (s for s in slots if s['position'] in STRIKERS):
        for w in (s for s in slots if s['position'] in WINGERS):
            if st['y'] > w['y']:
                found.add(f'{st["position"]} behind a winger')
            if st['y'] == w['y'] and ((w['position'] == 'Left Winger' and w['x'] > st['x'])
                                      or (w['position'] == 'Right Winger' and w['x'] < st['x'])):
                found.add(f'{st["position"]} wider than a winger')
    return found


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'output' / 'xi_lineups.json'
    matches = json.loads(path.read_text())['matches']
    bad, kinds, examples = 0, Counter(), {}
    for m in matches:
        found = problems(m)
        if found:
            bad += 1
            kinds.update(found)
            for k in found:
                examples.setdefault(k, f'{m["id"]} {m["team"]} {m["date"]} {m["formation"]}')
    print(f'{bad:,} of {len(matches):,} lineups with a misplaced position')
    for kind, n in kinds.most_common():
        print(f'  {n:5,} × {kind}   e.g. {examples[kind]}')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
