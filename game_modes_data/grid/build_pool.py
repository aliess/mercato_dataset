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

Usage:
    python game_modes_data/grid/build_pool.py
"""

import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import pandas as pd

MODE_DIR = Path(__file__).resolve().parent
ROOT = MODE_DIR.parents[1]
sys.path.insert(0, str(ROOT))

from sources.last_updated import record
from sources.tm_api import TransfermarktAPI

FIRST_SEASON = 1990  # 1990/91; the API has squads back to at least 1975
TOP5_LEAGUES = {'GB1': 'Premier League', 'ES1': 'LaLiga', 'IT1': 'Serie A', 'L1': 'Bundesliga', 'FR1': 'Ligue 1'}
# The 20 clubs outside those leagues that the most quiz-pool players passed through.
OTHER_CLUB_IDS = {
    294: 'Benfica', 610: 'Ajax', 720: 'Porto', 336: 'Sporting CP', 36: 'Fenerbahce', 383: 'PSV',
    141: 'Galatasaray', 114: 'Besiktas', 58: 'Anderlecht', 409: 'Red Bull Salzburg',
    234: 'Feyenoord', 683: 'Olympiacos', 614: 'Flamengo', 209: 'River Plate', 2282: 'Club Brugge',
    449: 'Trabzonspor', 585: 'Sao Paulo', 1075: 'Braga', 189: 'Boca Juniors', 1023: 'Palmeiras',
}


def main():
    this_season = date.today().year if date.today().month >= 7 else date.today().year - 1
    seasons = range(FIRST_SEASON, this_season + 1)
    old = TransfermarktAPI(max_age_hours=24 * 3650)  # finished seasons never change
    live = TransfermarktAPI()

    # ── Clubs ──
    club_league = {}  # club id → (league id, seasons in it)
    for league in TOP5_LEAGUES:
        for season in seasons:
            fixtures = (live if season >= this_season - 1 else old).competition_fixtures(league, season)
            for club_id in (fixtures or {}).get('clubIds') or []:
                club_league.setdefault(str(club_id), Counter())[league] += 1
    club_ids = set(club_league) | {str(i) for i in OTHER_CLUB_IDS}
    print(f'{len(club_ids):,} clubs: {len(club_league):,} from the top five leagues since {FIRST_SEASON}, '
          f'{len(club_ids) - len(club_league):,} others')

    # ── Squads ──
    progress = lambda done, total: print(f'  {done:,}/{total:,}', flush=True)
    past = [(c, s) for c in sorted(club_ids, key=int) for s in seasons if s < this_season - 1]
    recent = [(c, s) for c in sorted(club_ids, key=int) for s in seasons if s >= this_season - 1]
    print(f'Fetching {len(past) + len(recent):,} squads...', flush=True)
    squads = {**old.squads(past, progress), **live.squads(recent)}
    print(f'  got {len(squads):,} ({old.requests + live.requests:,} API requests)')

    spells = defaultdict(dict)  # player id → {club id: [first season, last season]}
    for (club_id, season), player_ids in squads.items():
        for player_id in player_ids:
            span = spells[player_id].setdefault(club_id, [season, season])
            span[0], span[1] = min(span[0], season), max(span[1], season)

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
    countries = pd.read_csv(ROOT / 'sources' / 'dataset' / 'countries.csv.gz').set_index('country_id')['country_name'].to_dict()

    unnamed_nations, missing = Counter(), 0
    rows = []
    for player_id, player_clubs in spells.items():
        player = players.get(player_id)
        if not player or not player.get('name'):
            missing += 1
            continue
        nation_id = ((player.get('nationalityDetails') or {}).get('nationalities') or {}).get('nationalityId')
        nation = countries.get(nation_id)
        if not nation:
            unnamed_nations[nation_id] += 1
        peak = ((player.get('marketValueDetails') or {}).get('highest') or {}).get('value')
        rows.append([player_id, player['name'], nation, peak,
                     [[int(club_id), *span] for club_id, span in sorted(player_clubs.items(), key=lambda c: c[1])]])
    rows.sort(key=lambda r: (-(r[3] or 0), int(r[0])))

    club_rows = [{
        'id': int(club_id), 'name': club['baseDetails'].get('shortName') or club['name'],
        'country': countries.get(club['baseDetails'].get('countryId')),
        'league': club_league[club_id].most_common(1)[0][0] if club_id in club_league else None,
        'top5_seasons': sum(club_league[club_id].values()) if club_id in club_league else 0,
        'players': sum(1 for r in rows if any(c[0] == int(club_id) for c in r[4])),
    } for club_id, club in sorted(clubs.items(), key=lambda c: int(c[0]))]

    out = MODE_DIR / 'output' / 'grid_pool.json'
    out.parent.mkdir(exist_ok=True)
    dump = lambda value: json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    out.write_text(
        '{"version":1,"first_season":%d,\n"player_fields":["id","name","nation","peak_value","clubs: [club id, first season, last season]"],\n'
        '"clubs":[\n%s\n],\n"players":[\n%s\n]}\n' % (FIRST_SEASON, ',\n'.join(map(dump, club_rows)), ',\n'.join(map(dump, rows))))

    record(MODE_DIR, 'built', players=len(rows), clubs=len(club_rows),
           seasons=f'{FIRST_SEASON} to {this_season}', source='Transfermarkt API')
    print(f'\n✓ {len(rows):,} players, {len(club_rows):,} clubs → {out} ({out.stat().st_size / 1e6:.1f} MB)')
    print(f'  without a peak market value: {sum(r[3] is None for r in rows):,}')
    if missing:
        print(f'  left out: {missing:,} players the API returned no record for')
    if unnamed_nations:
        print(f'  {sum(unnamed_nations.values()):,} players have a nationality id we have no name for: '
              f'{dict(unnamed_nations.most_common(10))}')


if __name__ == '__main__':
    main()
