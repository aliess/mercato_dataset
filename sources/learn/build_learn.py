#!/usr/bin/env python3
"""
Build learn_v1.json, the Learn journey (CONTRACTS.md §1, plan footballquiz/docs/plans/learn_journey.md):
worlds, their stops (one per club) and each stop's steps, in path order.

It is built from the club cards in sources/learn/clubs.json and the game files the publisher
builds (game_data_v1.json, xi_v1.json, grid_pool_v1.json, and the lineup star counts of
tiers_v1.json). publish_files.py builds it with the other files; run this script alone to
build just this file into publish/learn_v1.json:

    python sources/learn/build_learn.py                   # approved cards only (what is published)
    python sources/learn/build_learn.py --include-drafts  # draft cards too: local and dev testing only

Only stops whose card is `approved` are built; a stop without one is left out, and with no
stop at all there is no file. `--include-drafts` builds the draft cards too. The file holds no
build time: the same inputs give the same bytes.

World 1, The Giants: each club is one stop, with these steps in order:
  stars      the club's players now (active, current club = this club), the Beginner ones
             first, then by fame (market_value), 6-8 per step
  squad      the rest of them in fame order, 6-8 per step
  nights     the club's lineups in xi_v1.json, famous first, then finals, then by stars and
             year; up to 2 steps of 2-3
  connections  up to 3 pairs of this club with a club of an earlier stop, each with 3+ known
             answers in the Grid pool (peak value €20m+, GridIndex.knownValue in the app),
             most known answers first; none for the first stop
  checkpoint
  pointer    the world where this club's history is taught
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

LEARN_DIR = Path(__file__).resolve().parent
ROOT = LEARN_DIR.parent.parent
CARDS_FILE = LEARN_DIR / 'clubs.json'

CARD_FIELDS = ['club_id', 'short', 'en', 'ar', 'status']
CARD_STATUSES = ('draft', 'approved')
CARD_LENGTH = (220, 320)        # characters of the English card
LESSON = (6, 8)                 # players per stars / squad step
LESSON_FALLBACK = (5, 9)        # one squad step only, when a stop's count can't be split into 6-8
NIGHTS_MAX_STEPS, NIGHTS_PER_STEP = 2, 3
KNOWN_VALUE = 20_000_000        # GridIndex.knownValue in the app
MIN_KNOWN_ANSWERS = 3
MAX_PAIRS = 3
TIERS = ('beginner', 'intermediate', 'expert')
KINDS = ('stars', 'squad', 'history', 'nights', 'connections', 'checkpoint', 'pointer')

# The worlds in map order. `clubs` is the stop order; `history` names the world that teaches a
# club's retired players (World 1 points there instead of teaching them).
WORLDS = [
    {
        'id': 'giants', 'n': 1, 'name': {'en': 'The Giants', 'ar': ''}, 'tier': 'beginner', 'theme': 'giants',
        # Real Madrid, Barcelona, PSG, Man City, Liverpool, Bayern, Arsenal, Chelsea, Man Utd, Tottenham, Atlético
        'clubs': [418, 131, 583, 281, 31, 27, 11, 631, 985, 148, 13],
        'free': 418,
        'history': {27: 'bundesliga', 13: 'laliga', 583: 'ligue1'},
        'history_default': 'giants_history',
    },
]


# ── Club cards ──

def load_cards():
    """{club_id: card} from clubs.json, checked: fields, status, length, unique ids and labels."""
    cards = json.loads(CARDS_FILE.read_text(encoding='utf-8'))
    problems, by_club, shorts = [], {}, Counter(card.get('short') for card in cards)
    for card in cards:
        label = card.get('short') or card.get('club_id')
        if list(card) != CARD_FIELDS:
            problems.append(f'{label}: fields {list(card)}, expected {CARD_FIELDS}')
            continue
        if not isinstance(card['club_id'], int) or card['club_id'] in by_club:
            problems.append(f'{label}: club_id {card["club_id"]!r} is not an integer or is used twice')
        if not re.fullmatch(r'[A-Z]{3}', card['short']) or shorts[card['short']] > 1:
            problems.append(f'{label}: short must be 3 capital letters, used once')
        if card['status'] not in CARD_STATUSES:
            problems.append(f'{label}: status {card["status"]!r}, expected one of {CARD_STATUSES}')
        if not CARD_LENGTH[0] <= len(card['en']) <= CARD_LENGTH[1]:
            problems.append(f'{label}: the English card has {len(card["en"])} characters, '
                            f'expected {CARD_LENGTH[0]}-{CARD_LENGTH[1]}')
        if not isinstance(card['ar'], str):
            problems.append(f'{label}: ar must be text (empty until translated)')
        by_club[card['club_id']] = card
    if problems:
        sys.exit('sources/learn/clubs.json:\n  ' + '\n  '.join(problems))
    return by_club


# ── Splitting players into lessons ──

def split(count, low, high):
    """Sizes of the fewest steps of low..high players that hold `count`, as even as possible and
    the bigger first (20 → [7, 7, 6]); [] for 0; None when it can't be done (17 into 6-8)."""
    if count == 0:
        return []
    steps = -(-count // high)
    if steps * low > count:
        return None
    base, extra = divmod(count, steps)
    return [base + 1] * extra + [base] * (steps - extra)


def splittable(count):
    return split(count, *LESSON) is not None


def plan_lessons(total, beginners):
    """(stars sizes, squad sizes) for a stop with `total` players, the first `beginners` of them
    Beginner. Stars take as close to the Beginner count as an even split allows (a Beginner left
    over leads the squad; a missing one is the best known of the rest), the larger on a tie.
    When `total` can't be split into steps of 6-8, one squad step may have 5 or 9."""
    for squad_range in (LESSON, LESSON_FALLBACK):
        best = None
        for stars in range(LESSON[0], total + 1):
            stars_sizes = split(stars, *LESSON)
            squad_sizes = split(total - stars, *LESSON) if squad_range == LESSON else \
                split(total - stars, *squad_range)
            if stars_sizes is None or squad_sizes is None:
                continue
            if squad_range != LESSON and sum(1 for n in squad_sizes if not LESSON[0] <= n <= LESSON[1]) > 1:
                continue
            key = (abs(stars - beginners), -stars)
            if best is None or key < best[0]:
                best = (key, stars_sizes, squad_sizes)
        if best:
            return best[1], best[2]
    return None


# ── Building ──

def current_club(game_data, player_id):
    """The club of the player's last move (`to_id`), as an integer, or None."""
    moves = game_data['transfers'].get(player_id) or []
    to_id = moves[-1]['to_id'] if moves else None
    return int(to_id) if to_id else None


def club_countries(game_data, grid):
    """Club id → country name as the game data writes it (most common `to_country` of moves to
    the club), else the Grid pool's."""
    seen = {}
    for moves in game_data['transfers'].values():
        for move in moves:
            if move['to_id'] and move['to_country']:
                seen.setdefault(int(move['to_id']), Counter())[move['to_country']] += 1
    countries = {club['id']: club['country'] for club in grid['clubs']}
    countries.update({club_id: names.most_common(1)[0][0] for club_id, names in seen.items()})
    return countries


def grid_members(grid):
    """Club id → {player id: peak value} of everyone in the Grid pool who was in its squad."""
    members = {}
    for row in grid['players']:
        for club_id, _first, _last in row[4]:
            members.setdefault(club_id, {})[row[0]] = row[3]
    return members


def known_answers(members, club_a, club_b):
    both = members.get(club_a, {}).keys() & members.get(club_b, {}).keys()
    return sorted(pid for pid in both if members[club_a][pid] >= KNOWN_VALUE)


def build_learn(game_data, xi, grid, lineup_stars, include_drafts=False):
    """(content, counts, notes); content is None when no stop has a card to publish.
    `lineup_stars`: xi match id → star count (tiers_v1.json's `lineups`)."""
    cards = load_cards()
    statuses = ('approved', 'draft') if include_drafts else ('approved',)
    grid_names = {club['id']: club['name'] for club in grid['clubs']}
    countries = club_countries(game_data, grid)
    members = grid_members(grid)
    by_club = {}
    for player in game_data['players']:
        if player['active']:
            by_club.setdefault(current_club(game_data, player['id']), []).append(player)

    worlds, stops, notes, left_out = [], {}, [], []
    for world in WORLDS:
        stop_ids, passed = [], []
        for club_id in world['clubs']:
            card = cards.get(club_id)
            if club_id not in grid_names:
                sys.exit(f'learn: club {club_id} of world {world["id"]} is not in grid_pool_v1.json')
            if card is None or card['status'] not in statuses:
                left_out.append(card['short'] if card else str(club_id))
                continue
            stop_id = f"{world['id']}_{club_id}"
            steps = []

            def add(kind, **fields):
                n = sum(1 for step in steps if step['kind'] == kind) + 1
                steps.append({'id': f'{stop_id}_{kind}_{n}', 'kind': kind, **fields})

            # Players now: Beginner first, then by fame; ties by id so the order is stable.
            players = sorted(by_club.get(club_id, []),
                             key=lambda p: (p['difficulty'] != 'beginner', -p['market_value'], int(p['id'])))
            beginners = sum(1 for p in players if p['difficulty'] == 'beginner')
            plan = plan_lessons(len(players), beginners)
            if plan is None:
                sys.exit(f"learn: {card['short']} has {len(players)} players, too few for a lesson")
            stars_sizes, squad_sizes = plan
            if any(not LESSON[0] <= n <= LESSON[1] for n in squad_sizes):
                notes.append(f"{card['short']}: {len(players)} players can't be split into steps of "
                             f'{LESSON[0]}-{LESSON[1]}, so one squad step has {squad_sizes}')
            at = 0
            for kind, sizes in (('stars', stars_sizes), ('squad', squad_sizes)):
                for size in sizes:
                    add(kind, players=[p['id'] for p in players[at:at + size]])
                    at += size

            # Famous nights: famous first, then finals, then most stars, then the latest.
            nights = sorted((m for m in xi['matches'] if m['team'] == grid_names[club_id]),
                            key=lambda m: (not m['famous'], m['round'] != 'final',
                                           -lineup_stars[m['id']], -m['year'], m['id']))
            nights = [m['id'] for m in nights[:NIGHTS_MAX_STEPS * NIGHTS_PER_STEP]]
            if nights:
                first = len(nights) if len(nights) <= NIGHTS_PER_STEP else -(-len(nights) // 2)
                for part in (nights[:first], nights[first:]):
                    if part:
                        add('nights', lineups=part)

            # Connections with the clubs already passed in this world.
            pairs = [(len(known_answers(members, club_id, other)), index, other)
                     for index, other in enumerate(passed)]
            pairs = sorted((p for p in pairs if p[0] >= MIN_KNOWN_ANSWERS), key=lambda p: (-p[0], p[1]))
            if pairs:
                add('connections', pairs=[[club_id, other] for _known, _index, other in pairs[:MAX_PAIRS]])

            add('checkpoint')
            add('pointer', world=world['history'].get(club_id, world['history_default']))

            stops[stop_id] = {'club_id': club_id, 'short': card['short'], 'country': countries[club_id],
                              'free': club_id == world['free'], 'card': {'en': card['en'], 'ar': card['ar']},
                              'steps': steps}
            stop_ids.append(stop_id)
            passed.append(club_id)
        if stop_ids:
            worlds.append({key: world[key] for key in ('id', 'n', 'name', 'tier', 'theme')} | {'stops': stop_ids})

    if left_out:
        notes.append(f'{len(left_out)} stops left out, no approved card: {", ".join(left_out)}')
    if include_drafts:
        drafts = [stop['short'] for stop in stops.values() if cards[stop['club_id']]['status'] == 'draft']
        if drafts:
            notes.append(f'built with --include-drafts: {len(drafts)} stops have a draft card ({", ".join(drafts)})')
    if not stops:
        return None, {}, notes
    content = {'version': 1, 'worlds': worlds, 'stops': stops}
    validate(content, game_data, xi, grid)
    steps = [step for stop in stops.values() for step in stop['steps']]
    counts = {'worlds': len(worlds), 'stops': len(stops), 'steps': len(steps),
              'players': sum(len(step.get('players', [])) for step in steps),
              'lineups': sum(len(step.get('lineups', [])) for step in steps),
              'pairs': sum(len(step.get('pairs', [])) for step in steps)}
    return content, counts, notes


# ── The contract's checks (CONTRACTS.md §1, learn_v1.json) ──

def validate(content, game_data, xi, grid):
    """Stops the build when the file breaks the contract: every player, lineup and club known,
    a player in at most one player step, step ids, sizes and fields as agreed."""
    problems = []
    players = {p['id'] for p in game_data['players']}
    lineups = {m['id'] for m in xi['matches']}
    grid_clubs = {club['id'] for club in grid['clubs']}
    members = grid_members(grid)
    if list(content) != ['version', 'worlds', 'stops'] or content['version'] != 1:
        problems.append(f'top level {list(content)}, expected version 1, worlds, stops')
    listed = [stop_id for world in content['worlds'] for stop_id in world['stops']]
    if sorted(listed) != sorted(content['stops']) or len(set(listed)) != len(listed):
        problems.append('every stop must be in exactly one world, and every world stop must exist')
    free = [stop_id for stop_id, stop in content['stops'].items() if stop['free']]
    if len(free) > 1:
        problems.append(f'more than one free stop: {free}')
    taught = Counter()
    for world in content['worlds']:
        if list(world) != ['id', 'n', 'name', 'tier', 'theme', 'stops'] or world['tier'] not in TIERS \
                or list(world['name']) != ['en', 'ar']:
            problems.append(f"world {world.get('id')}: bad fields")
        passed = []
        for stop_id in world['stops']:
            stop = content['stops'].get(stop_id)
            if stop is None:
                continue
            where = f'stop {stop_id}'
            if list(stop) != ['club_id', 'short', 'country', 'free', 'card', 'steps'] \
                    or stop_id != f"{world['id']}_{stop['club_id']}" or stop['club_id'] not in grid_clubs \
                    or not re.fullmatch(r'[A-Z]{3}', stop['short']) or not stop['card'].get('en') \
                    or list(stop['card']) != ['en', 'ar'] or not isinstance(stop['free'], bool):
                problems.append(f'{where}: bad fields, id or club')
            kinds = Counter()
            off_size = []
            stop_players = 0
            for step in stop['steps']:
                kind = step.get('kind')
                kinds[kind] += 1
                if kind not in KINDS or step.get('id') != f'{stop_id}_{kind}_{kinds[kind]}':
                    problems.append(f"{where}: step {step.get('id')} has a bad kind or id")
                    continue
                fields = {'stars': ['players'], 'squad': ['players'], 'history': ['players', 'era'],
                          'nights': ['lineups'], 'connections': ['pairs'], 'checkpoint': [],
                          'pointer': ['world']}[kind]
                if list(step) != ['id', 'kind'] + fields:
                    problems.append(f"{step['id']}: fields {list(step)}")
                    continue
                if 'players' in step:
                    stop_players += len(step['players'])
                    taught.update(step['players'])
                    unknown = [p for p in step['players'] if p not in players]
                    if unknown:
                        problems.append(f"{step['id']}: players not in the game data {unknown}")
                    if not LESSON[0] <= len(step['players']) <= LESSON[1]:
                        off_size.append((step, len(step['players'])))
                if kind == 'nights' and not (1 <= len(step['lineups']) <= 3
                                             and all(i in lineups for i in step['lineups'])):
                    problems.append(f"{step['id']}: needs 1-3 lineups of xi_v1.json")
                if kind == 'connections':
                    for pair in step['pairs']:
                        ok = (len(pair) == 2 and pair[0] == stop['club_id'] and pair[1] in passed
                              and len(known_answers(members, *pair)) >= MIN_KNOWN_ANSWERS)
                        if not ok:
                            problems.append(f"{step['id']}: pair {pair} is not this club with an earlier "
                                            f'stop and {MIN_KNOWN_ANSWERS}+ known answers')
                if kind == 'pointer' and not (isinstance(step['world'], str) and step['world']):
                    problems.append(f"{step['id']}: pointer needs a world id")
            # Off 6-8 only as the one fallback squad step, when the count can't be split.
            if off_size and (len(off_size) > 1 or splittable(stop_players)
                             or off_size[0][0]['kind'] != 'squad'
                             or not LESSON_FALLBACK[0] <= off_size[0][1] <= LESSON_FALLBACK[1]):
                problems.append(f"{where}: player steps of {[n for _s, n in off_size]} (expected "
                                f'{LESSON[0]}-{LESSON[1]})')
            if kinds['checkpoint'] != 1:
                problems.append(f'{where}: needs one checkpoint')
            passed.append(stop['club_id'])
    twice = [p for p, n in taught.items() if n > 1]
    if twice:
        problems.append(f'players in more than one step: {twice[:10]}')
    if problems:
        sys.exit('learn_v1.json breaks the contract, nothing was written:\n  ' + '\n  '.join(problems[:30]))


# ── Alone ──

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--include-drafts', action='store_true',
                        help='Build stops whose card is still a draft (local and dev testing only).')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    import publish_files as publisher
    from sources.last_updated import record

    game_data = publisher.build_game_data()[0]
    xi = publisher.build_xi()[0]
    grid = publisher.build_grid()[0]
    stars = publisher.read_side_output(ROOT / 'game_modes_data' / 'starting_xi' / 'output' / 'xi_stars.json',
                                       'game_modes_data/starting_xi/build_lineups.py')['lineups']
    content, counts, notes = build_learn(game_data, xi, grid, stars, include_drafts=args.include_drafts)
    for note in notes:
        print(f'note: {note}')
    if content is None:
        print('No stop has an approved card: no learn_v1.json (--include-drafts builds the drafts).')
        return
    body = publisher.encode(content)
    out = publisher.PUBLISH_DIR / 'learn_v1.json'
    publisher.PUBLISH_DIR.mkdir(exist_ok=True)
    out.write_bytes(body)
    print(f'{out.relative_to(ROOT)}: {len(body):,} bytes, {len(publisher.compress(body)):,} gzipped; '
          f'version {publisher.sha256(body)[:16]}…')
    print('  ' + ', '.join(f'{value:,} {key}' for key, value in counts.items()))
    cards = load_cards()
    drafts = sum(1 for stop in content['stops'].values() if cards[stop['club_id']]['status'] == 'draft')
    record(LEARN_DIR, 'built', **counts, draft_cards=drafts)


if __name__ == '__main__':
    main()
