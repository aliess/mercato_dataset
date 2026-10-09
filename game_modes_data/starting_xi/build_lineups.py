#!/usr/bin/env python3
"""
Build the Starting XI lineups from the Transfermarkt API (see sources/tm_api.py):

  game_modes_data/starting_xi/output/xi_lineups.json

Matches: every quarter-final, semi-final and final of the Champions League (since 1992/93),
the World Cup (since 1986) and the Euros (since 1988), plus the matches listed in
famous_matches.json. Each match gives two lineups (home and away).
Only the eleven starters are included; substitutes are never answers.

Difficulty goes by the team and the year (see difficulty()). Team: the biggest clubs and
nations are easiest, other top-five-league clubs and other nations are a step harder,
clubs from other leagues are hardest. Year: 2000–2009 is a step harder and anything
before 2000 two steps, with one step back for finals and famous matches. So nothing
before 2000 is `beginner`: the 1998 World Cup final is `intermediate`, a 1996 quarter-final
is `expert`. Change the constants to re-split.

The file has the layout the app already reads (Resources/Labs/xi_placeholder.json), with
extra fields: difficulty, round, leg, opponent, season, famous.

Usage:
    python game_modes_data/starting_xi/build_lineups.py
"""

import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

MODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODE_DIR.parents[1]))

from sources.last_updated import record
from sources.tm_api import TransfermarktAPI

# (API competition id, label, first season). A season is the year it starts in; the API
# files summer tournaments under the year before (World Cup 2010 → season 2009).
COMPETITIONS = [
    ('CL', 'Champions League', 1992),  # 1992/93 is the first season the API has under this id
    ('FIWC', 'World Cup', 1985),
    ('EURO', 'Euro', 1987),
]
# Rounds taken in full, by the API's round name → the label shown in the app.
ROUNDS = {'Quarter-Finals': 'quarter-final', 'Semi-Finals': 'semi-final', 'Final': 'final'}
# Labels for matches that come from famous_matches.json.
OTHER_ROUNDS = {'Round of 16': 'round of 16', 'intermediate stage': 'knockout play-off',
                'First Round': 'first round', 'Second Round': 'second round'}

# ── Difficulty ──
ELITE_CLUB_IDS = {
    418: 'Real Madrid', 131: 'Barcelona', 27: 'Bayern Munich', 985: 'Man Utd', 31: 'Liverpool',
    631: 'Chelsea', 11: 'Arsenal', 281: 'Man City', 506: 'Juventus', 5: 'AC Milan', 46: 'Inter',
    583: 'PSG',
}
TOP5_COUNTRY_IDS = {189: 'England', 157: 'Spain', 75: 'Italy', 40: 'Germany', 50: 'France'}
TOP5_LEAGUE_CLUB_IDS_ABROAD = {162: 'Monaco'}  # plays in Ligue 1, registered in Monaco
ELITE_NATION_IDS = {26: 'Brazil', 9: 'Argentina', 50: 'France', 40: 'Germany', 157: 'Spain',
                    75: 'Italy', 189: 'England', 122: 'Netherlands', 136: 'Portugal'}

# Pitch depth (0 = own goal) and side (-1 left … 1 right) per Transfermarkt position.
LINE = {
    'Goalkeeper': (0, 0), 'Sweeper': (1, 0), 'Centre-Back': (1, 0), 'Left-Back': (1, -1),
    'Right-Back': (1, 1), 'Defensive Midfield': (2, 0), 'Central Midfield': (3, 0),
    'Left Midfield': (3, -1), 'Right Midfield': (3, 1), 'Attacking Midfield': (4, 0),
    'Left Winger': (5, -1), 'Right Winger': (5, 1), 'Second Striker': (5, 0),
    'Centre-Forward': (5, 0),
}
POSITION_NAMES = {'Sweeper': 'Centre-Back'}  # the app has no sweeper


# Older lineups have to be studied, not remembered: steps harder per era (first year, steps).
ERA_STEPS = [(2010, 0), (2000, 1), (0, 2)]
DIFFICULTIES = ['beginner', 'intermediate', 'expert']


def team_step(club):
    """0 for the biggest teams, 1 for the next group, 2 for the rest."""
    if club['baseDetails'].get('isNationalTeam'):
        return 0 if club['baseDetails'].get('countryId') in ELITE_NATION_IDS else 1
    if int(club['id']) in ELITE_CLUB_IDS:
        return 0
    in_top5 = club['baseDetails'].get('countryId') in TOP5_COUNTRY_IDS or int(club['id']) in TOP5_LEAGUE_CLUB_IDS_ABROAD
    return 1 if in_top5 else 2


def difficulty(club, year, famous):
    era = next(steps for first_year, steps in ERA_STEPS if year >= first_year)
    if famous:  # finals and famous matches are remembered longer
        era = max(era - 1, 0)
    return DIFFICULTIES[min(team_step(club) + era, 2)]


def short_name(club):
    return club['baseDetails'].get('shortName') or club['name']


def score_label(game, home, away):
    """"Milan 3–3 Liverpool (2–3 pens)": the API's score includes the shootout, so the score
    after extra time is taken from the last goal. None when it can't be worked out."""
    score = game['score']
    h, a = score['home'], score['away']
    kind = score.get('additionType')
    suffix = ' (a.e.t.)' if kind == 'after_extra_time' else ''
    if kind == 'after_shootout':
        goals = [x['score'] for x in game.get('actions') or [] if x['type'] == 'GOAL' and x.get('score')]
        played = max(goals, key=lambda s: s['home'] + s['away']) if goals else {'home': 0, 'away': 0}
        pens_h, pens_a = h - played['home'], a - played['away']
        # No check that the match was level: a second leg can go to penalties at 1–0.
        if pens_h == pens_a or min(pens_h, pens_a) < 0:
            return None
        h, a = played['home'], played['away']
        suffix = f' ({pens_h}–{pens_a} pens)'
    return f'{home} {h}–{a} {away}{suffix}'


def place(starters, tactic):
    """Starters → (formation, slots with pitch places).

    Rows come from the match formation ("4-4-2 double 6" → GK, 4, 4, 2), filled by depth;
    without one, players are grouped by position line. Shirts in a row go left to right by
    position. The API doesn't say which centre-back played left, so that order is arbitrary.
    """
    shape = re.match(r'\s*(\d(?:-\d)+)', tactic or '')
    counts = [int(n) for n in shape.group(1).split('-')] if shape else []
    ordered = sorted(starters, key=lambda p: LINE[p['position']])
    if sum(counts) == 10:
        chunks, start = [ordered[:1]], 1
        for n in counts:
            chunks.append(ordered[start:start + n])
            start += n
    else:
        by_depth = {}
        for p in ordered:
            by_depth.setdefault(LINE[p['position']][0], []).append(p)
        chunks = [by_depth[d] for d in sorted(by_depth)]
    slots = []
    for i, chunk in enumerate(chunks):
        row = sorted(chunk, key=lambda p: LINE[p['position']][1])
        y = 0.9 - 0.8 * i / (len(chunks) - 1)
        for j, p in enumerate(row):
            slots.append({
                'player_id': p['id'], 'name': p['name'], 'number': p['number'],
                'position': POSITION_NAMES.get(p['position'], p['position']), 'captain': p['captain'],
                'x': round((j + 1) / (len(row) + 1), 3), 'y': round(y, 3),
            })
    return '-'.join(str(len(c)) for c in chunks[1:]), slots


def main():
    this_season = date.today().year if date.today().month >= 7 else date.today().year - 1
    old = TransfermarktAPI(max_age_hours=24 * 3650)  # finished seasons never change
    live = TransfermarktAPI()

    # ── Which matches ──
    picked = {}  # game id → famous?
    for competition_id, _, first_season in COMPETITIONS:
        for season in range(first_season, this_season + 1):
            api = live if season >= this_season - 1 else old
            fixtures = api.competition_fixtures(competition_id, season)
            for game_day in (fixtures or {}).get('fixtures') or []:
                for game in game_day['games']:
                    group = game['baseDetails'].get('competitionGroup') or {}
                    if group.get('baseName') in ROUNDS and game.get('isFinished'):
                        picked[game['id']] = False
    for entry in json.loads((MODE_DIR / 'famous_matches.json').read_text()):
        picked[str(entry['game_id'])] = True
    print(f'{len(picked):,} matches ({sum(picked.values())} from famous_matches.json)')

    games = old.games(picked)
    clubs = old.clubs({g[side]['clubId'] for g in games.values() for side in ('homeClub', 'awayClub')})
    players = old.players({p['id'] for g in games.values() for side in ('homeClub', 'awayClub')
                           for p in (g[side].get('lineup') or {}).get('players') or []})
    print(f'{len(games):,} match records, {len(clubs):,} clubs, {len(players):,} players '
          f'({old.requests + live.requests:,} API requests)')

    # ── Lineups ──
    labels = {competition_id: label for competition_id, label, _ in COMPETITIONS}
    matches, skipped, skipped_matches = [], Counter(), []
    for game_id in sorted(picked, key=lambda i: games[i]['baseDetails']['date']['dateTimeUTC'] if i in games else ''):
        game = games.get(game_id)
        if not game:
            skipped['match not found'] += 2
            continue
        base = game['baseDetails']
        competition = labels.get(base['competitionId'], base['competition']['name'])
        group = base.get('competitionGroup') or {}
        round_name = ROUNDS.get(group.get('baseName')) or OTHER_ROUNDS.get(group.get('baseName')) \
            or ('group stage' if group.get('isGroupStage') else None)
        home, away = clubs.get(game['homeClub']['clubId']), clubs.get(game['awayClub']['clubId'])
        score = score_label(game, short_name(home), short_name(away)) if home and away else None
        match_date = base['date']['dateTimeUTC'][:10]
        if not (round_name and score):
            skipped['no round name' if not round_name else 'score unclear'] += 2
            skipped_matches.append(f'{match_date} {competition} {game_id}')
            continue
        for side, club, other in (('home', home, away), ('away', away, home)):
            lineup = (game[f'{side}Club'].get('lineup') or {}).get('players') or []
            starters = [{
                'id': p['id'], 'name': (players.get(p['id']) or {}).get('name'),
                'number': p.get('shirtNumber') or None, 'captain': bool(p.get('isCaptain')),
                'position': (p.get('position') or {}).get('name'),
            } for p in lineup]
            if len(starters) != 11:
                skipped[f'{len(starters)} starters'] += 1
                skipped_matches.append(f'{match_date} {competition} {short_name(club)} {game_id}')
                continue
            if any(p['position'] not in LINE for p in starters):
                skipped['unknown position'] += 1
                skipped_matches.append(f'{match_date} {competition} {short_name(club)} {game_id}')
                continue
            if any(not p['name'] for p in starters):
                skipped['player without a name'] += 1
                skipped_matches.append(f'{match_date} {competition} {short_name(club)} {game_id}')
                continue
            if sum(p['position'] == 'Goalkeeper' for p in starters) != 1:
                skipped['not exactly one goalkeeper'] += 1
                skipped_matches.append(f'{match_date} {competition} {short_name(club)} {game_id}')
                continue
            famous = picked[game_id] or round_name == 'final'
            formation, slots = place(starters, (game[f'{side}Club'].get('tactic') or {}).get('tactic'))
            matches.append({
                'id': f'{game_id}-{side}', 'team': short_name(club), 'opponent': short_name(other),
                'competition': f'{competition} {round_name}', 'round': round_name,
                'leg': 1 if group.get('isFirstLeg') else 2 if group.get('isSecondLeg') else None,
                'season': base['season']['display'], 'year': int(match_date[:4]), 'date': match_date,
                'score': score, 'formation': formation,
                'difficulty': difficulty(club, int(match_date[:4]), famous),
                'famous': famous, 'slots': slots,
            })

    out = MODE_DIR / 'output' / 'xi_lineups.json'
    out.parent.mkdir(exist_ok=True)
    lines = ',\n'.join(json.dumps(m, ensure_ascii=False, separators=(',', ':')) for m in matches)
    out.write_text('{"version":2,"matches":[\n' + lines + '\n]}\n')

    record(MODE_DIR, 'built', lineups=len(matches), matches=len({m['id'].split('-')[0] for m in matches}),
           latest_match=matches[-1]['date'], source='Transfermarkt API')
    print(f'\n✓ {len(matches):,} lineups → {out} ({out.stat().st_size / 1e6:.1f} MB)')
    by = lambda key: ', '.join(f'{k} {n:,}' for k, n in Counter(map(key, matches)).most_common())
    print(f'  by competition: {by(lambda m: m["competition"].rsplit(" ", 1)[0] if m["round"] in ROUNDS.values() else "famous extras")}')
    print(f'  by difficulty: {by(lambda m: m["difficulty"])}')
    print(f'  by round: {by(lambda m: m["round"])}')
    print(f'  from {matches[0]["date"]} to {matches[-1]["date"]}')
    if skipped:
        print('  left out: ' + ', '.join(f'{n:,} × {reason}' for reason, n in skipped.most_common()))
        for line in skipped_matches:
            print(f'    {line}')


if __name__ == '__main__':
    main()
