#!/usr/bin/env python3
"""
Build the Grid Rush answer pool from the Transfermarkt API (see sources/tm_api.py):

  game_modes_data/grid/output/grid_pool.json

The pool only decides whether a typed answer is right; it is much wider than the quiz's
player pool. A player counts for a club when he was in its squad in any season since
FIRST_SEASON, whether or not he played a match. Players of the quiz pool also count for
the clubs in their transfer history, as in the app today.

Clubs: every club that played in one of the top five leagues in any of those seasons,
plus OTHER_CLUB_IDS. Nationality is the player's main one only.

Each player row also carries his position and date of birth, for the hints
(sources/player_facts.py). The build saves the country names it learned to
sources/country_ids.json for the Starting XI build.

Usage:
    python game_modes_data/grid/build_pool.py
    python game_modes_data/grid/build_pool.py --from-cache   # no API calls for data already
        cached, however old: rebuilds the same pool, e.g. after a change to the row's fields
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

MODE_DIR = Path(__file__).resolve().parent
ROOT = MODE_DIR.parents[1]
sys.path.insert(0, str(ROOT))

from sources import fame
from sources.last_updated import record
from sources.player_facts import PlayerFacts, country_names, nation_id, save_country_names
from sources.squads import FIRST_SEASON, pool_clubs, squad_spells, this_season as current_season
from sources.tm_api import TransfermarktAPI

# Clubs and squads are the shared player universe (sources/squads.py). Market value is how the
# game tells a famous pick from an obscure one, and Transfermarkt has only valued players since
# late 2004. So each row's value is the player's fame (sources/fame.py): his peak value, raised
# by his legend value (sources/legends/legends.json) for the well-known players of that era
# (Maradona highest; also Maldini's late-career €3m). A player with no fame is left out.
# This is the raw fame, not fame_now in today's money: the Grid's well-known line (€20m, in the
# app and the functions) and the steals compare raw peaks, so they stay on it.


def main():
    this_season = current_season()
    old = TransfermarktAPI(max_age_hours=24 * 3650)  # finished seasons never change
    live = old if '--from-cache' in sys.argv[1:] else TransfermarktAPI()

    # ── Clubs ──
    club_league = pool_clubs(old, live)  # club id → (league id → seasons in it)
    club_ids = set(club_league)
    in_top5 = sum(1 for leagues in club_league.values() if leagues)
    print(f'{len(club_ids):,} clubs: {in_top5:,} from the top five leagues since {FIRST_SEASON}, '
          f'{len(club_ids) - in_top5:,} others')

    # ── Squads ──
    progress = lambda done, total: print(f'  {done:,}/{total:,}', flush=True)
    print('Fetching squads...', flush=True)
    spells, squad_count = squad_spells(old, live, club_ids, progress)
    print(f'  got {squad_count:,} ({old.requests + live.requests:,} API requests)')

    # Quiz-pool players also count for the clubs in their transfer history.
    transfers = pd.read_csv(ROOT / 'game_modes_data' / 'transfer_history' / 'output' / 'transfer_history.csv',
                            usecols=['player_id', 'transfer_date', 'from_team_id', 'to_team_id'])
    from_history = 0
    for row in transfers.itertuples():
        year = int(row.transfer_date[:4])
        for club_id, season in ((str(row.to_team_id), year), (str(row.from_team_id), year - 1)):
            if club_id in club_ids and club_id not in spells[str(row.player_id)]:
                spells[str(row.player_id)][club_id] = [season, season]
                from_history += 1
    spells = {player_id: clubs for player_id, clubs in spells.items() if clubs}
    print(f'{len(spells):,} players in those squads (+{from_history:,} club spells from transfer histories)')

    # ── Players and clubs ──
    print('Fetching players...', flush=True)
    players = old.players(spells)
    clubs = old.clubs(club_ids)
    # Country names as the quiz pool spells them, so Grid's nationality columns match.
    countries = country_names(players)
    facts = PlayerFacts(players, countries)

    legends = fame.legends()
    unnamed_nations, missing, no_value, raised = Counter(), 0, 0, 0
    rows = []
    for player_id, player_clubs in spells.items():
        player = players.get(player_id)
        if not player or not player.get('name'):
            missing += 1
            continue
        nation = countries.get(nation_id(player))
        if not nation:
            unnamed_nations[nation_id(player)] += 1
        value = fame.fame(player_id, player)
        if not value:
            no_value += 1
            continue
        if value > (fame.peak_value(player) or 0):
            raised += 1
        rows.append([player_id, facts.name(player_id), nation, value,
                     [[int(club_id), *span] for club_id, span in sorted(player_clubs.items(), key=lambda c: c[1])],
                     facts.position(player_id), facts.born(player_id), int(player_id in legends)])
    rows.sort(key=lambda r: (-(r[3] or 0), int(r[0])))

    club_rows = [{
        'id': int(club_id), 'name': club['baseDetails'].get('shortName') or club['name'],
        'country': countries.get(club['baseDetails'].get('countryId')),
        'league': club_league[club_id].most_common(1)[0][0] if club_league[club_id] else None,
        'top5_seasons': sum(club_league[club_id].values()),
        'players': sum(1 for r in rows if any(c[0] == int(club_id) for c in r[4])),
    } for club_id, club in sorted(clubs.items(), key=lambda c: int(c[0]))]

    out = MODE_DIR / 'output' / 'grid_pool.json'
    out.parent.mkdir(exist_ok=True)
    dump = lambda value: json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    out.write_text(
        '{"version":1,"first_season":%d,\n"player_fields":["id","name","nation","peak_value","clubs: [club id, first season, last season]","position","born","legend"],\n'
        '"clubs":[\n%s\n],\n"players":[\n%s\n]}\n' % (FIRST_SEASON, ',\n'.join(map(dump, club_rows)), ',\n'.join(map(dump, rows))))

    save_country_names(countries)
    record(MODE_DIR, 'built', players=len(rows), clubs=len(club_rows),
           seasons=f'{FIRST_SEASON} to {this_season}', source='Transfermarkt API')
    print(f'\n✓ {len(rows):,} players, {len(club_rows):,} clubs → {out} ({out.stat().st_size / 1e6:.1f} MB)')
    print(f'  left out: {no_value:,} players with no market value')
    for column, label in ((5, 'position'), (6, 'born')):
        have = sum(1 for r in rows if r[column])
        print(f'  {label}: {have:,} of {len(rows):,} players ({have / len(rows):.1%})')
    print(f'  {raised:,} of the {len(legends):,} legends got their value from legends.json; '
          f'{sum(r[7] for r in rows):,} legends in the pool')
    absent = set(legends) - {r[0] for r in rows}
    if absent:
        print(f'  ⚠ legends not in any squad: {sorted(absent)}')
    if missing:
        print(f'  left out: {missing:,} players the API returned no record for')
    if unnamed_nations:
        print(f'  {sum(unnamed_nations.values()):,} players have a nationality id we have no name for: '
              f'{dict(unnamed_nations.most_common(10))}')


if __name__ == '__main__':
    main()
