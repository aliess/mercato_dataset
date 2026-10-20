#!/usr/bin/env python3
"""
Build learn_v1.json, the Learn journey (CONTRACTS.md §1, plan footballquiz/docs/plans/learn_journey.md):
worlds, their stops and each stop's steps, in path order, plus a hook per taught player and
the look-alike pairs (`twins`).

It is built from the club cards in sources/learn/clubs.json and the game files the publisher
builds (game_data_v1.json, clues_v1.json, xi_v1.json, grid_pool_v1.json, and the lineup star
counts of tiers_v1.json). Routes also need to know which moves were loans and which clubs are
youth or reserve sides: that comes from the Transfermarkt API cache (sources/tm_api.py), which
the transfer-history build filled. publish_files.py builds it with the other files; run this
script alone to build just this file into publish/learn_v1.json:

    python sources/learn/build_learn.py                   # approved cards only (what is published)
    python sources/learn/build_learn.py --include-drafts  # draft and missing cards too: local and dev only

Release builds (no --include-drafts) leave out a club stop whose card is not `approved`, and with
no club stop at all there is no file. --include-drafts builds every club stop: a draft card is
published as it is, a missing one as `card: null`. The file holds no build time: the same inputs
give the same bytes.

Homes (plan §2). Every player has one home: his current club while he plays for a club of the
Grid pool, otherwise the Grid pool club where he spent the most time; the 10 with none go to
`beyond_free`. A home named in WORLDS is that club's stop (today's players in World 1 for the
11 giants, the giants' other players in their history stop); any other home goes to the
`{world}_rest` stop of its league.

Player steps. Today's players (still playing, home = current club) first, best known first
(Beginner, then fame), then the others by era (the decade of their time at the club), legends
first within an era. World 1 splits today's squad as before (stars ≈ the Beginner players).
Elsewhere the list is cut into lessons of 6-8 that cross as few eras as possible and mix today's
players with former ones only when the counts leave no choice: a lesson of today's players only
is `stars` (the first) or `squad`, any other is `history` with its main era. When a count can't
be cut into 6-8, one lesson may have 5 or 9; when even that fails (10 = 5 + 5), every lesson may.

Other steps (per stop, as the stop has material):
  routes       the club's 2-4 most common permanent moves between senior teams (no loans, youth
               or B teams, Without Club or Retired), 3+ moves each, at the club's first stop
  nights       the club's lineups in xi_v1.json not used at an earlier stop, famous first, then
               finals, by stars and year; up to 2 steps of 2-3. Nations: every World Cup and Euro
               lineup of the nation, by tournament, oldest first, 1-3 per step
  connections  up to 3 pairs of this club with a club of an earlier stop (same world first),
               each with 3+ known answers in the Grid pool, a pair taught once
  checkpoint   one per stop; the Summit has one per world before it
  pointer      World 1: the world where this club's history is taught

Dossiers, anchors and trails (item 140, what the app reads since then) are built from these stops
by build_dossiers.py; story dossiers come from sources/learn/dossiers.json.
"""

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

LEARN_DIR = Path(__file__).resolve().parent
ROOT = LEARN_DIR.parent.parent
CARDS_FILE = LEARN_DIR / 'clubs.json'

CARD_FIELDS = ['club_id', 'short', 'en', 'ar', 'status']
CARD_STATUSES = ('draft', 'approved')
CARD_LENGTH = (220, 320)        # characters of the English card
LESSON = (6, 8)                 # players per stars / squad / history step
LESSON_FALLBACK = (5, 9)        # when a stop's count can't be cut into 6-8
NIGHTS_MAX_STEPS, NIGHTS_PER_STEP = 2, 3
KNOWN_VALUE = 20_000_000        # GridIndex.knownValue in the app
MIN_KNOWN_ANSWERS = 3
MAX_PAIRS = 3
ROUTES_PER_STEP = (2, 4)
ROUTE_MIN_MOVES = 3
ROUTE_PLAYERS = (3, 4)
TIERS = ('beginner', 'intermediate', 'expert')
KINDS = ('stars', 'squad', 'routes', 'history', 'nights', 'connections', 'checkpoint', 'pointer')
PLAYER_KINDS = ('stars', 'squad', 'history')
STEP_FIELDS = {'stars': ['players'], 'squad': ['players'], 'history': ['players', 'era'],
               'routes': ['routes'], 'nights': ['lineups'], 'connections': ['pairs'], 'checkpoint': [],
               'pointer': ['world']}
ROUTE_FIELDS = ['from', 'to', 'from_name', 'to_name', 'count', 'players']
HOOK_FACTS = {'top_fee': ['transfer'], 'born_abroad': [], 'one_club': ['club'], 'long_stay': ['club', 'seasons'],
              'many_clubs': ['clubs']}
TWIN_DIFFS = ('club_now', 'born', 'foot', 'height', 'nation', 'prev_club', 'position')
LOAN_TYPES = {'ACTIVE_LOAN_TRANSFER', 'RETURNED_FROM_PREVIOUS_LOAN'}
TODAY_ERA = '2020s'
# Fact hooks, when a player has no clue to lead with
TOP_FEE_HOOK, LONG_STAY_HOOK, MANY_CLUBS_HOOK = 20_000_000, 8, 8
# Twins: a pair's score (see twin_score) must reach this; a player is in one pair at most and a
# lesson shows two pairs at most.
TWIN_MIN_SCORE, TWINS_PER_LESSON, TWIN_MAX_AGE_GAP = 7, 2, 4

GIANTS = [418, 131, 583, 281, 31, 27, 11, 631, 985, 148, 13]
# The worlds in map order (learn_journey.md §4). `clubs` is the order of the club stops; a world
# with a `league` also has a "rest of" stop for its other clubs. `history` (World 1) names the
# world that teaches a giant's former players.
WORLDS = [
    {'id': 'giants', 'n': 1, 'name': {'en': 'The Giants', 'ar': 'العمالقة'}, 'tier': 'beginner',
     # Real Madrid, Barcelona, PSG, Man City, Liverpool, Bayern, Arsenal, Chelsea, Man Utd, Tottenham, Atlético
     'clubs': GIANTS, 'free': 418,
     'history': {27: 'bundesliga', 13: 'laliga', 583: 'ligue1'}, 'history_default': 'giants_history'},
    {'id': 'serie_a', 'n': 2, 'name': {'en': 'Serie A', 'ar': 'الدوري الإيطالي'}, 'league': 'IT1', 'short': 'ITA',
     # Juventus, Milan, Inter, Roma, Napoli, Fiorentina, Atalanta, Lazio, Bologna, Como
     'clubs': [506, 5, 46, 12, 6195, 430, 800, 398, 1025, 1047]},
    {'id': 'bundesliga', 'n': 3, 'name': {'en': 'Bundesliga', 'ar': 'الدوري الألماني'}, 'league': 'L1', 'short': 'GER',
     'rest_name': 'The rest of the Bundesliga',
     # Bayern (history), Dortmund, Leverkusen, Leipzig, Stuttgart, Köln, Hoffenheim, Schalke,
     # Frankfurt, Wolfsburg, Hamburg, Freiburg
     'clubs': [27, 16, 15, 23826, 79, 3, 533, 33, 24, 82, 41, 60]},
    {'id': 'laliga', 'n': 4, 'name': {'en': 'La Liga', 'ar': 'الدوري الإسباني'}, 'league': 'ES1', 'short': 'ESP',
     # Atlético (history), Valencia, Betis, Real Sociedad, Villarreal, Sevilla, Deportivo, Celta,
     # Athletic Club, Getafe
     'clubs': [13, 1049, 150, 681, 1050, 368, 897, 940, 621, 3709]},
    {'id': 'premier', 'n': 5, 'name': {'en': 'Premier League', 'ar': 'الدوري الإنجليزي الممتاز'}, 'league': 'GB1', 'short': 'ENG',
     'rest_name': 'The rest of the Premier League',
     # Crystal Palace, Everton, Newcastle, Aston Villa, Brentford, Brighton, Bournemouth,
     # Nott'm Forest, West Ham, Fulham, Leeds, Sunderland, Wolves
     'clubs': [873, 29, 762, 405, 1148, 1237, 989, 703, 379, 931, 399, 289, 543]},
    {'id': 'ligue1', 'n': 6, 'name': {'en': 'Ligue 1', 'ar': 'الدوري الفرنسي'}, 'league': 'FR1', 'short': 'FRA',
     # PSG (history), Lyon, Monaco, Marseille, Lille, Strasbourg, Lens, Rennes, Nice, Nantes, Bordeaux
     'clubs': [583, 1041, 162, 244, 1082, 667, 826, 273, 417, 995, 40]},
    {'id': 'beyond', 'n': 7, 'name': {'en': 'Beyond the Top Five', 'ar': 'خارج الدوريات الخمس الكبرى'}, 'league': None, 'short': 'WLD',
     # Porto, Benfica, Sporting; Ajax, PSV, Feyenoord; Galatasaray, Fenerbahçe, Beşiktaş,
     # Trabzonspor; Flamengo, São Paulo, River Plate, Palmeiras. (Celtic, named in the plan, is
     # not a Grid pool club, so no player has his home there.)
     'clubs': [720, 294, 336, 610, 383, 234, 141, 36, 114, 449, 614, 585, 209, 1023],
     'rest_name': 'The rest of the world', 'rest_name_ar': 'بقية أندية العالم', 'free_stop': True},
    {'id': 'giants_history', 'n': 8, 'name': {'en': "The Giants' History", 'ar': 'تاريخ العمالقة'}, 'tier': 'intermediate',
     # Real Madrid, Barcelona, Man Utd, Chelsea, Arsenal, Liverpool, Tottenham, Man City
     'clubs': [418, 131, 985, 631, 11, 31, 148, 281]},
    {'id': 'nations', 'n': 9, 'name': {'en': 'Nations and Tournaments', 'ar': 'المنتخبات والبطولات'},
     'nations': [('GER', 'Germany', 'ألمانيا'), ('FRA', 'France', 'فرنسا'), ('ESP', 'Spain', 'إسبانيا'),
                 ('ITA', 'Italy', 'إيطاليا'), ('ENG', 'England', 'إنجلترا'), ('NED', 'Netherlands', 'هولندا'),
                 ('ARG', 'Argentina', 'الأرجنتين'), ('BRA', 'Brazil', 'البرازيل'), ('POR', 'Portugal', 'البرتغال')]},
    {'id': 'summit', 'n': 10, 'name': {'en': 'The Summit', 'ar': 'القمة'}, 'tier': 'expert', 'summit': True},
]
TOURNAMENTS = ('World Cup', 'Euro')


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


def size_rule(count):
    """How a stop of `count` players may be cut: 'lesson' (all 6-8), 'one_off' (one step of 5 or
    9) or 'all_off' (every step 5-9)."""
    if count == 0 or split(count, *LESSON) is not None:
        return 'lesson'
    return 'one_off' if any(split(count - n, *LESSON) is not None for n in LESSON_FALLBACK) else 'all_off'


def sizes_allowed(sizes):
    """The step sizes of a stop follow size_rule for their total."""
    rule = size_rule(sum(sizes))
    off = [n for n in sizes if not LESSON[0] <= n <= LESSON[1]]
    in_fallback = all(LESSON_FALLBACK[0] <= n <= LESSON_FALLBACK[1] for n in sizes)
    return {'lesson': not off, 'one_off': len(off) <= 1 and in_fallback, 'all_off': in_fallback}[rule]


def plan_lessons(total, beginners):
    """World 1: (stars sizes, squad sizes) for a stop with `total` players, the first `beginners`
    of them Beginner. Stars take as close to the Beginner count as an even split allows (a
    Beginner left over leads the squad; a missing one is the best known of the rest), the larger
    on a tie. When `total` can't be split into steps of 6-8, one squad step may have 5 or 9."""
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


def cut_lessons(eras):
    """Cut a stop's ordered players into lessons. `eras` holds each player's era, None for
    today's players (who come first). Returns the sizes: the fewest lessons that mix today's
    players with former ones, then the fewest era changes inside a lesson, then the fewest
    lessons, the bigger first. None when there is nobody."""
    total = len(eras)
    if total == 0:
        return []
    rule = size_rule(total)
    sizes_in = list(range(LESSON[1], LESSON[0] - 1, -1))
    off_sizes = [LESSON_FALLBACK[1], LESSON_FALLBACK[0]]

    def cost(start, size):
        part = eras[start:start + size]
        today = sum(1 for era in part if era is None)
        mixed = 1 if 0 < today < size else 0
        former = {era for era in part if era is not None}
        return (mixed, max(len(former) - 1, 0))

    @lru_cache(maxsize=None)
    def best(start, offs_left):
        if start == total:
            return (0, 0, 0), ()
        found = None
        options = [(n, False) for n in sizes_in] + [(n, True) for n in off_sizes]
        for size, off in options:
            if start + size > total or (off and offs_left == 0):
                continue
            rest = best(start + size, offs_left - (1 if off else 0))
            if rest is None:
                continue
            mixed, eras_crossed = cost(start, size)
            key = (rest[0][0] + mixed, rest[0][1] + eras_crossed, rest[0][2] + 1)
            if found is None or key < found[0]:
                found = (key, (size,) + rest[1])
        return found

    offs = {'lesson': 0, 'one_off': 1, 'all_off': total}[rule]
    result = best(0, offs)
    best.cache_clear()
    return list(result[1]) if result else None


# ── The game data, read for Learn ──

def day_of(millis):
    return datetime.fromtimestamp(millis / 1000, tz=timezone.utc).strftime('%Y-%m-%d')


def year_of(timestamp):
    return datetime.fromtimestamp(timestamp['_seconds'], tz=timezone.utc).year if timestamp else None


def plain(text):
    """Lower case, no accents: for comparing names with clue text."""
    text = unicodedata.normalize('NFKD', text or '')
    return ''.join(c for c in text if not unicodedata.combining(c)).lower().replace('ß', 'ss')


class Moves:
    """The transfers of game_data_v1.json with what Learn needs to know about each: which clubs
    are senior first teams, which moves were loans, each player's spells and his home."""

    def __init__(self, game_data, grid, api_clubs, api_histories, today):
        self.game_data = game_data
        self.grid_ids = {club['id'] for club in grid['clubs']}
        self.today = today
        self.special = {int(i) for i, club in api_clubs.items() if club.get('isSpecialClub')}
        self.reserve = set()
        from sources.club_types import is_reserve_side, AGE_GROUP_NAME
        for club_id, club in api_clubs.items():
            flag = is_reserve_side(club)
            name = club['name'].strip()
            if flag or (flag is None and (AGE_GROUP_NAME.search(name) or re.search(r'\s(B|II)$', name))):
                self.reserve.add(int(club_id))
        self.unknown_clubs = set()
        # A move is a loan when the API files the same move (player, day, clubs) as one.
        loans = set()
        for player_id, history in api_histories.items():
            for move in history:
                if move['typeDetails'].get('type') in LOAN_TYPES:
                    loans.add((player_id, move['details']['date'][:10], move['transferSource']['clubId'],
                               move['transferDestination']['clubId']))
        self.loan_ids = set()
        for player_id, moves in game_data['transfers'].items():
            for move in moves:
                key = (player_id, day_of(move['date']), move['from_id'], move['to_id'])
                if key in loans:
                    self.loan_ids.add(move['id'])
                for club_id in (move['from_id'], move['to_id']):
                    if club_id and club_id not in api_clubs:
                        self.unknown_clubs.add(int(club_id))
        # Club names as the transfers write them (the most common one).
        names = {}
        for moves in game_data['transfers'].values():
            for move in moves:
                for club_id, name in ((move['from_id'], move['from']), (move['to_id'], move['to'])):
                    if club_id and name:
                        names.setdefault(int(club_id), Counter())[name] += 1
        self.names = {club_id: counter.most_common(1)[0][0] for club_id, counter in names.items()}

    def senior(self, club_id):
        return club_id is not None and club_id not in self.special and club_id not in self.reserve \
            and club_id not in self.unknown_clubs

    def moves(self, player_id):
        return self.game_data['transfers'].get(player_id) or []

    def spells(self, player_id):
        """[(club id, start ms, end ms, loan)] of the player's moves, oldest first."""
        moves = self.moves(player_id)
        result = []
        for index, move in enumerate(moves):
            if not move['to_id']:
                continue
            end = moves[index + 1]['date'] if index + 1 < len(moves) else self.today
            result.append((int(move['to_id']), move['date'], end, move['id'] in self.loan_ids))
        return result

    def current_club(self, player_id):
        moves = self.moves(player_id)
        return int(moves[-1]['to_id']) if moves and moves[-1]['to_id'] else None

    def home(self, player):
        """(home club id or None, today): his current club while he plays for a Grid pool club,
        else the Grid pool club where he spent the most time."""
        current = self.current_club(player['id'])
        if player['active'] and current in self.grid_ids:
            return current, True
        time_at = Counter()
        for club_id, start, end, _loan in self.spells(player['id']):
            if club_id in self.grid_ids:
                time_at[club_id] += end - start
        if not time_at:
            return None, False
        return max(time_at.items(), key=lambda item: (item[1], -item[0]))[0], False

    def era(self, player_id, club_id):
        """The decade of the middle of his time at `club_id` (his whole career when None)."""
        spells = [s for s in self.spells(player_id) if club_id is None or s[0] == club_id] \
            or self.spells(player_id)
        if not spells:
            return TODAY_ERA
        total = sum(end - start for _c, start, end, _l in spells)
        half, passed = total / 2, 0
        for _club, start, end, _loan in spells:
            if passed + (end - start) >= half:
                middle = start + (half - passed)
                break
            passed += end - start
        year = datetime.fromtimestamp(middle / 1000, tz=timezone.utc).year
        return f'{year // 10 * 10}s'

    def senior_clubs(self, player_id):
        """Senior clubs he was registered with, in order of first arrival (loans included)."""
        seen = []
        for club_id, _start, _end, _loan in self.spells(player_id):
            if self.senior(club_id) and club_id not in seen:
                seen.append(club_id)
        return seen

    def seasons_at(self, player_id):
        """{club id: whole seasons at that senior club}, rounded, at least 1."""
        days = Counter()
        for club_id, start, end, _loan in self.spells(player_id):
            if self.senior(club_id):
                days[club_id] += (end - start) / 86_400_000
        return {club_id: max(1, round(d / 365.25)) for club_id, d in days.items()}

    def route_moves(self):
        """Every permanent move between two senior teams: (from id, to id, player id)."""
        for player_id, moves in self.game_data['transfers'].items():
            for move in moves:
                if move['id'] in self.loan_ids or not move['from_id'] or not move['to_id']:
                    continue
                from_id, to_id = int(move['from_id']), int(move['to_id'])
                if from_id != to_id and self.senior(from_id) and self.senior(to_id):
                    yield from_id, to_id, player_id


def load_api_facts(game_data):
    """(clubs, histories) from the API cache: the clubs of every move and every player's
    transfer history. The transfer-history build fetched them all; nothing is asked again
    unless the cache was cleared."""
    from sources.tm_api import TransfermarktAPI
    api = TransfermarktAPI(max_age_hours=24 * 365 * 20)
    club_ids = {move[side] for moves in game_data['transfers'].values() for move in moves
                for side in ('from_id', 'to_id') if move[side]}
    clubs = api.clubs(club_ids)
    histories = api.transfer_histories([p['id'] for p in game_data['players']])
    return clubs, histories, api.requests


# ── Building ──

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


def fame_order(player):
    return (player['difficulty'] != 'beginner', -player['market_value'], int(player['id']))


def build_learn(game_data, xi, grid, lineup_stars, clues, include_drafts=False, api_facts=None):
    """(content, counts, notes); content is None when no club stop has a card to publish.
    `lineup_stars`: xi match id → star count (tiers_v1.json's `lineups`); `clues`: clues_v1.json.
    `api_facts`: (clubs, histories) from the API, read from the cache when not given."""
    from sources import fame
    from sources.learn import build_dossiers
    cards = load_cards()
    arabic = build_dossiers.load_arabic()
    statuses = ('approved', 'draft') if include_drafts else ('approved',)
    notes = []
    grid_clubs = {club['id']: club for club in grid['clubs']}
    countries = club_countries(game_data, grid)
    members = grid_members(grid)
    players = {p['id']: p for p in game_data['players']}
    if api_facts is None:
        api_clubs, histories, requests = load_api_facts(game_data)
        if requests:
            notes.append(f'{requests:,} API requests: the cache missed clubs or transfer histories')
    else:
        api_clubs, histories = api_facts
    last_move = max((m['date'] for moves in game_data['transfers'].values() for m in moves), default=0)
    moves = Moves(game_data, grid, api_clubs, histories, today=last_move)
    if moves.unknown_clubs:
        notes.append(f'{len(moves.unknown_clubs):,} clubs of the transfers are not in the API cache; '
                     'their moves are left out of the routes')
    legends = set(fame.legends())

    # ── Each player's stop ──
    world_of_league = {w['league']: w for w in WORLDS if 'league' in w}
    club_stop_worlds = {}  # club id → [world ids with a stop for it]
    for world in WORLDS:
        for club_id in world.get('clubs', []):
            if club_id not in grid_clubs:
                sys.exit(f'learn: club {club_id} of world {world["id"]} is not in grid_pool_v1.json')
            club_stop_worlds.setdefault(club_id, []).append(world['id'])
    giants_world = WORLDS[0]
    stop_players = {}  # stop id → [(player, era or None for today)]
    for player in sorted(game_data['players'], key=lambda p: int(p['id'])):
        home, today = moves.home(player)
        if home is None:
            stop_id = 'beyond_free'
        elif home in GIANTS:
            stop_id = f'giants_{home}' if today else \
                f"{giants_world['history'].get(home, giants_world['history_default'])}_{home}"
        else:
            world = world_of_league[grid_clubs[home]['league']]
            stop_id = f"{world['id']}_{home}" if home in world['clubs'] else f"{world['id']}_rest"
        era = None if today else moves.era(player['id'], home)
        stop_players.setdefault(stop_id, []).append((player, era))

    # ── The stops, in path order ──
    worlds, stops = [], {}
    taught_order = []           # player ids in the order the journey teaches them
    used_lineups, used_pairs = set(), set()
    routes_done = set()         # clubs whose routes are taught
    passed = []                 # (world id, club id) of the club stops built so far
    route_index = build_routes(moves)
    left_out, null_cards = [], []
    stop_lists = {}             # stop id → [(player, era or None for today)], in teaching order

    for world in WORLDS:
        stop_ids = []

        def new_stop(stop_id, head):
            steps = []

            def add(kind, **fields):
                n = sum(1 for step in steps if step['kind'] == kind) + 1
                steps.append({'id': f'{stop_id}_{kind}_{n}', 'kind': kind, **fields})
            return steps, add

        if world.get('summit'):
            steps, add = new_stop('summit_top', None)
            for _ in range(len(worlds)):
                add('checkpoint')
            stops['summit_top'] = {'club_id': None, 'short': 'TOP', 'name': {'en': 'The Summit', 'ar': 'القمة'},
                                   'country': None, 'free': False, 'card': None, 'steps': steps}
            stop_ids.append('summit_top')
        elif 'nations' in world:
            for short, nation, nation_ar in world['nations']:
                stop_id = f'nations_{short}'
                steps, add = new_stop(stop_id, None)
                lineups = [m for m in xi['matches'] if m['team'] == nation
                           and m['competition'].startswith(TOURNAMENTS) and m['id'] not in used_lineups]
                lineups.sort(key=lambda m: (m['year'], m['date'], m['id']))
                by_tournament = {}
                for match in lineups:
                    by_tournament.setdefault((match['year'], match['competition'].split(' ')[0]), []).append(match['id'])
                for ids in by_tournament.values():
                    for size in split(len(ids), 1, NIGHTS_PER_STEP) if len(ids) > 3 else [len(ids)]:
                        add('nights', lineups=ids[:size])
                        ids = ids[size:]
                used_lineups.update(m['id'] for m in lineups)
                if not steps:
                    notes.append(f'{stop_id}: no lineups, stop left out')
                    continue
                add('checkpoint')
                stops[stop_id] = {'club_id': None, 'short': short, 'name': {'en': nation, 'ar': nation_ar},
                                  'country': nation, 'nation': nation, 'free': False, 'card': None, 'steps': steps}
                stop_ids.append(stop_id)
        else:
            plan = [(club_id, f"{world['id']}_{club_id}") for club_id in world['clubs']]
            if 'league' in world:
                plan.append((None, f"{world['id']}_rest"))
            if world.get('free_stop'):
                plan.append((None, 'beyond_free'))
            for club_id, stop_id in plan:
                members_here = stop_players.get(stop_id, [])
                if club_id is not None:
                    card = cards.get(club_id)
                    if card is not None and card['status'] in statuses:
                        card_out = {'en': card['en'], 'ar': card['ar']}
                    elif include_drafts:
                        card_out = None
                        null_cards.append((club_id, stop_id))
                    else:
                        left_out.append(card['short'] if card else str(club_id))
                        continue
                steps, add = new_stop(stop_id, None)

                # Player steps
                today = sorted((p for p, era in members_here if era is None), key=fame_order)
                former = sorted(((p, era) for p, era in members_here if era is not None),
                                key=lambda pe: (pe[1], pe[0]['id'] not in legends, -pe[0]['market_value'],
                                                int(pe[0]['id'])))
                ordered = [(p, None) for p in today] + former
                stop_lists[stop_id] = ordered
                if world['id'] == 'giants':
                    plan_sizes = plan_lessons(len(today), sum(1 for p in today if p['difficulty'] == 'beginner'))
                    if plan_sizes is None:
                        sys.exit(f'learn: {stop_id} has {len(today)} players, too few for a lesson')
                    lessons = [('stars', n) for n in plan_sizes[0]] + [('squad', n) for n in plan_sizes[1]]
                else:
                    sizes = cut_lessons([era for _p, era in ordered])
                    if sizes is None:
                        sys.exit(f'learn: {stop_id} has {len(ordered)} players, too few for a lesson')
                    lessons, at = [], 0
                    for size in sizes:
                        part = ordered[at:at + size]
                        if all(era is None for _p, era in part):
                            lessons.append(('stars' if not lessons else 'squad', size))
                        else:
                            lessons.append(('history', size))
                        at += size
                if any(not LESSON[0] <= n <= LESSON[1] for _k, n in lessons):
                    notes.append(f'{stop_id}: {len(ordered)} players can not be cut into lessons of '
                                 f'{LESSON[0]}-{LESSON[1]}: {[n for _k, n in lessons]}')
                at = 0
                player_steps = []
                for kind, size in lessons:
                    part = ordered[at:at + size]
                    at += size
                    ids = [p['id'] for p, _era in part]
                    if kind == 'history':
                        eras = Counter(era or TODAY_ERA for _p, era in part)
                        era = min(eras, key=lambda e: (-eras[e], e))
                        player_steps.append((kind, {'players': ids, 'era': era}))
                    else:
                        player_steps.append((kind, {'players': ids}))
                for kind, fields in player_steps:
                    if kind != 'history':
                        add(kind, **fields)
                taught_order.extend(pid for kind, fields in player_steps for pid in fields['players'])

                # Routes, at the club's first stop
                if club_id is not None and club_id not in routes_done:
                    routes_done.add(club_id)
                    taught = set(taught_order)
                    picked = []
                    for route in route_index.get(club_id, []):
                        made_it = sorted(route['players'], key=lambda pid: (pid not in taught,
                                                                            -players[pid]['market_value'], int(pid)))
                        picked.append({'from': route['from'], 'to': route['to'],
                                       'from_name': moves.names[route['from']], 'to_name': moves.names[route['to']],
                                       'count': route['count'], 'players': made_it[:ROUTE_PLAYERS[1]]})
                        if len(picked) == ROUTES_PER_STEP[1]:
                            break
                    if len(picked) >= ROUTES_PER_STEP[0]:
                        add('routes', routes=picked)

                for kind, fields in player_steps:
                    if kind == 'history':
                        add(kind, **fields)

                if club_id is not None:
                    # Famous nights not taught at an earlier stop
                    nights = sorted((m for m in xi['matches'] if m['team'] == grid_clubs[club_id]['name']
                                     and m['id'] not in used_lineups),
                                    key=lambda m: (not m['famous'], m['round'] != 'final',
                                                   -lineup_stars[m['id']], -m['year'], m['id']))
                    nights = [m['id'] for m in nights[:NIGHTS_MAX_STEPS * NIGHTS_PER_STEP]]
                    used_lineups.update(nights)
                    if nights:
                        first = len(nights) if len(nights) <= NIGHTS_PER_STEP else -(-len(nights) // 2)
                        for part in (nights[:first], nights[first:]):
                            if part:
                                add('nights', lineups=part)

                    # Connections with clubs of earlier stops: this world first, a pair once
                    candidates = []
                    for index, (other_world, other) in enumerate(passed):
                        pair_key = frozenset((club_id, other))
                        if other == club_id or pair_key in used_pairs or any(c[3] == other for c in candidates):
                            continue
                        known = len(known_answers(members, club_id, other))
                        if known >= MIN_KNOWN_ANSWERS:
                            candidates.append((other_world != world['id'], -known, index, other))
                    candidates.sort()
                    pairs = [[club_id, c[3]] for c in candidates[:MAX_PAIRS]]
                    used_pairs.update(frozenset(pair) for pair in pairs)
                    if pairs:
                        add('connections', pairs=pairs)

                if not any(step['kind'] in PLAYER_KINDS + ('nights',) for step in steps):
                    notes.append(f'{stop_id}: nothing to teach, stop left out')
                    continue
                add('checkpoint')
                if world['id'] == 'giants':
                    add('pointer', world=world['history'].get(club_id, world['history_default']))

                if club_id is not None:
                    if str(club_id) not in arabic['clubs']:
                        sys.exit(f'learn: club {club_id} {grid_clubs[club_id]["name"]} has no Arabic name in '
                                 f'{build_dossiers.ARABIC_FILE.name}')
                    stops[stop_id] = {'club_id': club_id, 'short': cards[club_id]['short'] if club_id in cards
                                      else short_label(grid_clubs[club_id]['name']),
                                      'name': {'en': grid_clubs[club_id]['name'],
                                               'ar': arabic['clubs'][str(club_id)]['ar']},
                                      'country': countries[club_id], 'free': club_id == world.get('free'),
                                      'card': card_out, 'steps': steps}
                    passed.append((world['id'], club_id))
                elif stop_id == 'beyond_free':
                    stops[stop_id] = {'club_id': None, 'short': 'FRE', 'name': {'en': 'Free agents', 'ar': 'اللاعبون الأحرار'},
                                      'country': None, 'free': False, 'card': None, 'steps': steps}
                else:
                    league = world['league']
                    country = Counter(c['country'] for c in grid_clubs.values() if c['league'] == league) \
                        .most_common(1)[0][0] if league else None
                    stops[stop_id] = {'club_id': None, 'short': world['short'],
                                      'name': {'en': world.get('rest_name', f"The rest of {world['name']['en']}"),
                                               'ar': world.get('rest_name_ar', f"بقية أندية {world['name']['ar']}")},
                                      'country': country, 'free': False, 'card': None, 'steps': steps}
                stop_ids.append(stop_id)

        if stop_ids:
            worlds.append({'id': world['id'], 'n': world['n'], 'name': world['name'], 'tier': None,
                           'theme': world['id'], 'stops': stop_ids})

    if not any(stop['club_id'] is not None for stop in stops.values()):
        return None, {}, notes + ['no club stop has an approved card']
    # Without a club stop's card (a release build), its players are not taught: drop the
    # pointers to worlds that are not in the file.
    built_worlds = {w['id'] for w in worlds}
    for stop in stops.values():
        stop['steps'] = [s for s in stop['steps'] if s['kind'] != 'pointer' or s['world'] in built_worlds]

    # Each world's main tier: its players' (or lineups') most common tier
    lineup_tier = {m['id']: m['difficulty'] for m in xi['matches']}
    for world, spec in zip(worlds, [w for w in WORLDS if w['id'] in built_worlds]):
        if spec.get('tier'):
            world['tier'] = spec['tier']
            continue
        tally = Counter()
        for stop_id in world['stops']:
            for step in stops[stop_id]['steps']:
                tally.update(players[p]['difficulty'] for p in step.get('players', []))
                tally.update(lineup_tier[i] for i in step.get('lineups', []))
        world['tier'] = min(TIERS, key=lambda t: (-tally[t], TIERS.index(t)))

    hooks = build_hooks(taught_order, players, clues, game_data, moves, notes)
    twins = build_twins(content_order(worlds, stops), players, moves)
    content = {'version': 1, 'worlds': worlds, 'stops': stops, 'hooks': hooks, 'twins': twins}
    dossiers, anchors, trails = build_dossiers.build(
        content, stop_lists, players, xi, grid, lineup_stars, moves, legends, route_index,
        include_drafts=include_drafts, notes=notes, alt_names=game_data.get('alt_names') or {},
        arabic=arabic)
    for world in worlds:
        world['dossiers'] = dossiers['order'].get(world['id'], [])
    content.update(dossiers=dossiers['by_id'], anchors=anchors, trails=trails)
    validate(content, game_data, xi, grid, clues, moves)

    if left_out:
        notes.append(f'{len(left_out)} club stops left out, no approved card: {", ".join(left_out)}')
    if include_drafts:
        drafts = [stop['short'] for stop in stops.values()
                  if stop['club_id'] is not None and stop['card'] is not None
                  and cards[stop['club_id']]['status'] == 'draft']
        if drafts:
            notes.append(f'built with --include-drafts: {len(drafts)} club stops have a draft card')
        if null_cards:
            notes.append(f'{len(null_cards)} club stops have no card (card: null): '
                         + ', '.join(f"{stops[s]['short']} {c} {grid_clubs[c]['name']}" for c, s in null_cards
                                     if s in stops))
    steps = [step for stop in stops.values() for step in stop['steps']]
    counts = {'worlds': len(worlds), 'stops': len(stops), 'steps': len(steps),
              'players': sum(len(step.get('players', [])) for step in steps),
              'lineups': sum(len(step.get('lineups', [])) for step in steps),
              'pairs': sum(len(step.get('pairs', [])) for step in steps),
              'routes': sum(len(step.get('routes', [])) for step in steps),
              'hooks': len(hooks), 'clue_hooks': sum(1 for h in hooks.values() if 'clue' in h),
              'twins': len(twins)}
    kinds = Counter(d['kind'] for d in content['dossiers'].values())
    counts.update({f'dossiers_{kind}': kinds[kind] for kind in build_dossiers.KINDS},
                  anchored=sum(1 for a in anchors.values() if a), no_anchor=sum(1 for a in anchors.values() if not a),
                  trails=len(trails))
    return content, counts, notes


def short_label(name):
    letters = re.sub(r'[^A-Z]', '', plain(name).upper())
    return (letters + 'XXX')[:3]


def content_order(worlds, stops):
    """[(player id, step id)] in the order the journey teaches them."""
    return [(pid, step['id']) for world in worlds for stop_id in world['stops']
            for step in stops[stop_id]['steps'] for pid in step.get('players', [])]


# ── Routes ──

def build_routes(moves):
    """Club id → its routes, most moves first: [{from, to, count, players}] with count ≥ 3
    permanent senior moves made by 3+ different players."""
    tally, who = Counter(), {}
    for from_id, to_id, player_id in moves.route_moves():
        tally[(from_id, to_id)] += 1
        who.setdefault((from_id, to_id), set()).add(player_id)
    routes = {}
    for (from_id, to_id), count in tally.items():
        if count < ROUTE_MIN_MOVES or len(who[(from_id, to_id)]) < ROUTE_PLAYERS[0]:
            continue
        route = {'from': from_id, 'to': to_id, 'count': count, 'players': sorted(who[(from_id, to_id)], key=int)}
        for club_id in (from_id, to_id):
            routes.setdefault(club_id, []).append(route)
    for club_routes in routes.values():
        club_routes.sort(key=lambda r: (-r['count'], r['from'], r['to']))
    return routes


# ── Hooks ──

MEMORABLE = re.compile(r"\b(record|first|youngest|oldest|only|ever|final|hat-trick|ballon|golden|"
                       r"nickname|called|named|award|world cup|penalty|bicycle|overhead|debut|captain|"
                       r"legend|history|famous|lucky|superstition)\b", re.I)


def name_words(player, alt_names):
    """The words of his names (3+ letters), plain: a clue that holds one gives him away."""
    names = [player.get('player_name'), player.get('name_in_home_country'), player.get('player_slug')]
    names += alt_names.get(player['id'], []) if isinstance(alt_names, dict) else []
    words = set()
    for name in names:
        for word in re.split(r"[^a-z0-9']+", plain((name or '').replace('-', ' '))):
            if len(word) >= 3:
                words.add(word)
    return words


def gives_name_away(text, words):
    found = set(re.split(r"[^a-z0-9']+", plain(text)))
    return bool(found & words)


def pick_clue(pairs, words):
    """Index of the clue to lead with, or None: the one with most memorable words (the signature
    moment, clue 1, on a tie; clue 2 lists clubs and fees, which the reveal shows already), never
    one that holds a word of his name."""
    best = None
    for index, (en, _ar) in enumerate(pairs):
        if gives_name_away(en, words):
            continue
        score = len(MEMORABLE.findall(en)) + {0: 0, 1: 1, 2: -2}[index]
        key = (-score, {1: 0, 0: 1, 2: 2}[index])
        if best is None or key < best[0]:
            best = (key, index)
    return best[1] if best else None


def build_hooks(taught_order, players, clues, game_data, moves, notes):
    alt = game_data.get('alt_names') or {}
    hooks = {}
    facts_used = Counter()
    for player_id in sorted(taught_order, key=int):
        player = players[player_id]
        entry = clues['clues'].get(player_id)
        index = pick_clue(entry['c'], name_words(player, alt)) if entry else None
        if index is not None:
            hooks[player_id] = {'clue': index}
            continue
        hook = fact_hook(player, moves)
        facts_used[hook['fact']] += 1
        hooks[player_id] = hook
    no_clue = sum(facts_used.values())
    if no_clue:
        notes.append(f'{no_clue:,} hooks are facts (no usable clue): '
                     + ', '.join(f'{k} {v}' for k, v in facts_used.most_common()))
    return hooks


def fact_hook(player, moves):
    """One fact from the game data, the most telling first."""
    player_id = player['id']
    clubs = moves.senior_clubs(player_id)
    seasons = moves.seasons_at(player_id)
    fees = [m for m in moves.moves(player_id) if m['fee'] and m['id'] not in moves.loan_ids]
    top = max(fees, key=lambda m: (m['fee'], m['date'])) if fees else None
    longest = max(seasons.items(), key=lambda kv: (kv[1], -kv[0])) if seasons else None
    if len(clubs) == 1:
        return {'fact': 'one_club', 'club': clubs[0]}
    if top and top['fee'] >= TOP_FEE_HOOK:
        return {'fact': 'top_fee', 'transfer': top['id']}
    if longest and longest[1] >= LONG_STAY_HOOK:
        return {'fact': 'long_stay', 'club': longest[0], 'seasons': longest[1]}
    if player.get('country_of_birth') and player.get('citizenship') \
            and player['country_of_birth'] != player['citizenship']:
        return {'fact': 'born_abroad'}
    if len(clubs) >= MANY_CLUBS_HOOK:
        return {'fact': 'many_clubs', 'clubs': len(clubs)}
    if top:
        return {'fact': 'top_fee', 'transfer': top['id']}
    if longest:
        return {'fact': 'long_stay', 'club': longest[0], 'seasons': longest[1]}
    return {'fact': 'many_clubs', 'clubs': len(clubs)}


# ── Twins ──

def twin_score(a, b, facts):
    """How likely two players of the same main position and nation, born 4 years apart at most,
    are to be mixed up; None when they share neither a club nor an era."""
    fa, fb = facts[a['id']], facts[b['id']]
    shared = set(fa['clubs']) & set(fb['clubs'])
    overlap = any(sa[0] == sb[0] and sa[1] < sb[2] and sb[1] < sa[2]
                  for sa in fa['spells'] for sb in fb['spells'] if sa[0] in shared)
    same_era = fa['era'] == fb['era']
    if not shared and not same_era:
        return None
    score = 4 if overlap else 2 if shared else 0
    score += 1 if same_era else 0
    score += 2 if a.get('position') and a.get('position') == b.get('position') else 0
    score += 3 if fa['names'] & fb['names'] else 0
    gap = abs((fa['born'] or 0) - (fb['born'] or 0))
    score += 1 if gap <= 2 else 0
    score += 1 if fa['stop'] == fb['stop'] else 0
    score += 1 if a['difficulty'] == b['difficulty'] else 0
    height_a, height_b = a.get('height'), b.get('height')
    score += 1 if height_a and height_b and abs(height_a - height_b) <= 3 else 0
    return score


def twin_diff(a, b, facts, moves):
    """The one field that tells them apart, or None."""
    fa, fb = facts[a['id']], facts[b['id']]
    if a['active'] and b['active'] and fa['club_now'] and fb['club_now'] and fa['club_now'] != fb['club_now'] \
            and moves.senior(fa['club_now']) and moves.senior(fb['club_now']):
        return 'club_now'
    if a.get('position') and b.get('position') and a['position'] != b['position']:
        return 'position'
    if a.get('foot') and b.get('foot') and a['foot'] != b['foot'] and 'both' not in (a['foot'], b['foot']):
        return 'foot'
    if fa['born'] and fb['born'] and abs(fa['born'] - fb['born']) >= 2:
        return 'born'
    if a.get('height') and b.get('height') and abs(a['height'] - b['height']) >= 5:
        return 'height'
    if fa['prev_club'] and fb['prev_club'] and fa['prev_club'] != fb['prev_club']:
        return 'prev_club'
    if fa['born'] and fb['born'] and fa['born'] != fb['born']:
        return 'born'
    return None


def build_twins(order, players, moves):
    """[[a, b, diff]]: a is taught at or before b. A player is in one pair at most, a lesson shows
    TWINS_PER_LESSON pairs at most, the best scored first."""
    position = {pid: i for i, (pid, _step) in enumerate(order)}
    step_of = dict(order)
    facts = {}
    for pid, step_id in order:
        player = players[pid]
        home_club = None
        spells = moves.spells(pid)
        seasons = moves.seasons_at(pid)
        if seasons:
            home_club = max(seasons.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        prev = None
        for index, spell in enumerate(spells):
            if spell[0] == home_club and index > 0:
                prev = spells[index - 1][0] if moves.senior(spells[index - 1][0]) else None
                break
        facts[pid] = {'clubs': moves.senior_clubs(pid),
                      'spells': [s for s in spells if moves.senior(s[0])],
                      'era': moves.era(pid, home_club), 'born': year_of(player.get('date_of_birth')),
                      'names': {w for w in re.split(r"[^a-z]+", plain(player['player_name'])) if len(w) >= 4},
                      'stop': step_id.rsplit('_', 2)[0], 'club_now': moves.current_club(pid), 'prev_club': prev}
    groups = {}
    for pid, _step in order:
        player = players[pid]
        if player.get('main_position') and player.get('citizenship') and facts[pid]['born']:
            groups.setdefault((player['main_position'], player['citizenship']), []).append(pid)
    candidates = []
    for group in groups.values():
        for i, a_id in enumerate(group):
            for b_id in group[i + 1:]:
                a, b = players[a_id], players[b_id]
                if abs(facts[a_id]['born'] - facts[b_id]['born']) > TWIN_MAX_AGE_GAP:
                    continue
                score = twin_score(a, b, facts)
                if score is None or score < TWIN_MIN_SCORE:
                    continue
                first, second = sorted((a_id, b_id), key=lambda p: position[p])
                diff = twin_diff(players[first], players[second], facts, moves)
                if diff:
                    candidates.append((-score, position[second], position[first], first, second, diff))
    candidates.sort()
    paired, per_lesson, twins = set(), Counter(), []
    for _score, _pb, _pa, first, second, diff in candidates:
        if first in paired or second in paired or per_lesson[step_of[second]] >= TWINS_PER_LESSON:
            continue
        paired.update((first, second))
        per_lesson[step_of[second]] += 1
        twins.append([first, second, diff])
    twins.sort(key=lambda t: (position[t[1]], position[t[0]]))
    return twins


# ── The contract's checks (CONTRACTS.md §1, learn_v1.json) ──

def validate(content, game_data, xi, grid, clues, moves=None):
    """Stops the build when the file breaks the contract: every player, lineup and club known,
    a player in at most one player step, step ids, sizes and fields as agreed, a hook for every
    taught player, twins taught in order, routes with 3+ moves."""
    problems = []
    players = {p['id']: p for p in game_data['players']}
    transfer_ids = {pid: {m['id'] for m in ms} for pid, ms in game_data['transfers'].items()}
    lineups = {m['id'] for m in xi['matches']}
    grid_clubs = {club['id'] for club in grid['clubs']}
    grid_names = {club['id']: club['name'] for club in grid['clubs']}
    members = grid_members(grid)
    if list(content) != ['version', 'worlds', 'stops', 'hooks', 'twins', 'dossiers', 'anchors', 'trails'] \
            or content['version'] != 1:
        problems.append(f'top level {list(content)}, expected version 1, worlds, stops, hooks, twins, '
                        'dossiers, anchors, trails')
    world_ids = [world['id'] for world in content['worlds']]
    listed = [stop_id for world in content['worlds'] for stop_id in world['stops']]
    if sorted(listed) != sorted(content['stops']) or len(set(listed)) != len(listed):
        problems.append('every stop must be in exactly one world, and every world stop must exist')
    free = [stop_id for stop_id, stop in content['stops'].items() if stop['free']]
    if len(free) > 1:
        problems.append(f'more than one free stop: {free}')
    taught = Counter()
    order = []
    passed = []
    for world in content['worlds']:
        if list(world) != ['id', 'n', 'name', 'tier', 'theme', 'stops', 'dossiers'] or world['tier'] not in TIERS \
                or list(world['name']) != ['en', 'ar'] or world['theme'] != world['id']:
            problems.append(f"world {world.get('id')}: bad fields")
        for stop_id in world['stops']:
            stop = content['stops'].get(stop_id)
            if stop is None:
                continue
            where = f'stop {stop_id}'
            card_ok = stop['card'] is None or (list(stop['card']) == ['en', 'ar'] and bool(stop['card']['en']))
            name = stop.get('name')
            name_ok = (isinstance(name, dict) and list(name) == ['en', 'ar']
                       and all(isinstance(name[k], str) and name[k] for k in ('en', 'ar')))
            if not name_ok:
                problems.append(f'{where}: name must be {{en, ar}}, both filled')
            if stop.get('club_id') is not None:
                ok = (list(stop) == ['club_id', 'short', 'name', 'country', 'free', 'card', 'steps']
                      and stop_id == f"{world['id']}_{stop['club_id']}" and stop['club_id'] in grid_clubs
                      and name_ok and name['en'] == grid_names[stop['club_id']])
            else:
                expected = ['club_id', 'short', 'name', 'country'] + (['nation'] if world['id'] == 'nations' else []) \
                    + ['free', 'card', 'steps']
                ok = (list(stop) == expected and list(stop['name']) == ['en', 'ar'] and bool(stop['name']['en'])
                      and stop['free'] is False
                      and (stop_id in (f"{world['id']}_rest", 'beyond_free', 'summit_top')
                           or (world['id'] == 'nations' and stop_id == f"nations_{stop['short']}"
                               and stop['nation'] == stop['country'])))
            if not ok or not card_ok or not re.fullmatch(r'[A-Z]{3}', stop['short']) \
                    or not isinstance(stop['free'], bool):
                problems.append(f'{where}: bad fields, id, card or club')
            kinds = Counter()
            sizes = []
            for step in stop['steps']:
                kind = step.get('kind')
                kinds[kind] += 1
                if kind not in KINDS or step.get('id') != f'{stop_id}_{kind}_{kinds[kind]}':
                    problems.append(f"{where}: step {step.get('id')} has a bad kind or id")
                    continue
                if list(step) != ['id', 'kind'] + STEP_FIELDS[kind]:
                    problems.append(f"{step['id']}: fields {list(step)}")
                    continue
                if 'players' in step:
                    sizes.append(len(step['players']))
                    taught.update(step['players'])
                    order.extend(step['players'])
                    unknown = [p for p in step['players'] if p not in players]
                    if unknown:
                        problems.append(f"{step['id']}: players not in the game data {unknown}")
                if kind == 'history' and not re.fullmatch(r'(19|20)\d0s', step['era']):
                    problems.append(f"{step['id']}: era {step['era']!r}")
                if kind == 'nights' and not (1 <= len(step['lineups']) <= 3
                                             and all(i in lineups for i in step['lineups'])):
                    problems.append(f"{step['id']}: needs 1-3 lineups of xi_v1.json")
                if kind == 'connections':
                    for pair in step['pairs']:
                        ok = (len(pair) == 2 and pair[0] == stop['club_id'] and pair[1] in passed
                              and pair[1] != pair[0] and len(known_answers(members, *pair)) >= MIN_KNOWN_ANSWERS)
                        if not ok:
                            problems.append(f"{step['id']}: pair {pair} is not this club with an earlier "
                                            f'stop and {MIN_KNOWN_ANSWERS}+ known answers')
                if kind == 'routes':
                    routes = step['routes']
                    if not ROUTES_PER_STEP[0] <= len(routes) <= ROUTES_PER_STEP[1] or stop['club_id'] is None:
                        problems.append(f"{step['id']}: {len(routes)} routes at a club stop, expected "
                                        f'{ROUTES_PER_STEP[0]}-{ROUTES_PER_STEP[1]}')
                    for route in routes:
                        ok = (list(route) == ROUTE_FIELDS and isinstance(route['from'], int)
                              and isinstance(route['to'], int) and route['from'] != route['to']
                              and stop['club_id'] in (route['from'], route['to'])
                              and isinstance(route['from_name'], str) and route['from_name']
                              and isinstance(route['to_name'], str) and route['to_name']
                              and isinstance(route['count'], int) and route['count'] >= ROUTE_MIN_MOVES
                              and ROUTE_PLAYERS[0] <= len(route['players']) <= ROUTE_PLAYERS[1]
                              and len(set(route['players'])) == len(route['players'])
                              and all(p in players and any(m['from_id'] == str(route['from'])
                                                           and m['to_id'] == str(route['to'])
                                                           for m in game_data['transfers'].get(p, []))
                                      for p in route['players']))
                        if not ok:
                            problems.append(f"{step['id']}: bad route {route}")
                if kind == 'pointer' and step['world'] not in world_ids:
                    problems.append(f"{step['id']}: pointer to {step['world']!r}, not a world of the file")
            if sizes and not sizes_allowed(sizes):
                problems.append(f'{where}: player steps of {sizes} (expected {LESSON[0]}-{LESSON[1]}, '
                                f'or {LESSON_FALLBACK[0]}-{LESSON_FALLBACK[1]} when the count can not be cut)')
            if stop_id == 'summit_top':
                if not kinds['checkpoint'] or set(kinds) != {'checkpoint'}:
                    problems.append(f'{where}: the Summit has checkpoint steps only')
            elif kinds['checkpoint'] != 1:
                problems.append(f'{where}: needs one checkpoint')
            if world['id'] == 'nations' and set(kinds) - {'nights', 'checkpoint'}:
                problems.append(f'{where}: a nation stop has nights and a checkpoint only')
            if stop.get('club_id') is not None:
                passed.append(stop['club_id'])
    twice = [p for p, n in taught.items() if n > 1]
    if twice:
        problems.append(f'players in more than one step: {twice[:10]}')

    # Hooks: one per taught player, nobody else
    hooks = content.get('hooks', {})
    if set(hooks) != set(taught):
        problems.append(f'hooks: {len(set(taught) - set(hooks)):,} taught players without one, '
                        f'{len(set(hooks) - set(taught)):,} for players not taught')
    alt = game_data.get('alt_names') or {}
    for player_id, hook in hooks.items():
        player = players.get(player_id)
        if player is None:
            continue
        if list(hook) == ['clue']:
            entry = clues['clues'].get(player_id)
            ok = (entry is not None and hook['clue'] in (0, 1, 2)
                  and not gives_name_away(entry['c'][hook['clue']][0], name_words(player, alt)))
        else:
            fact = hook.get('fact')
            ok = fact in HOOK_FACTS and list(hook) == ['fact'] + HOOK_FACTS[fact]
            if ok and fact == 'top_fee':
                ok = hook['transfer'] in transfer_ids.get(player_id, set())
            if ok and fact in ('one_club', 'long_stay'):
                ok = isinstance(hook['club'], int) and any(
                    str(hook['club']) in (m['from_id'], m['to_id']) for m in game_data['transfers'].get(player_id, []))
            if ok and fact == 'long_stay':
                ok = isinstance(hook['seasons'], int) and hook['seasons'] >= 1
            if ok and fact == 'many_clubs':
                ok = isinstance(hook['clubs'], int) and hook['clubs'] >= 1
            if ok and fact == 'born_abroad':
                ok = bool(player.get('country_of_birth')) and player.get('country_of_birth') != player.get('citizenship')
        if not ok:
            problems.append(f'hook of {player_id}: {hook}')

    # Twins: both taught, the first at or before the second, a known diff, a pair once
    position = {pid: i for i, pid in enumerate(order)}
    seen = set()
    for twin in content.get('twins', []):
        ok = (isinstance(twin, list) and len(twin) == 3 and twin[0] in position and twin[1] in position
              and twin[0] != twin[1] and position[twin[0]] <= position[twin[1]] and twin[2] in TWIN_DIFFS
              and frozenset(twin[:2]) not in seen)
        if ok:
            a, b = players[twin[0]], players[twin[1]]
            ok = a.get('main_position') == b.get('main_position') and a.get('citizenship') == b.get('citizenship')
            seen.add(frozenset(twin[:2]))
        if not ok:
            problems.append(f'twin {twin}')
    from sources.learn import build_dossiers
    problems += build_dossiers.validate(content, game_data, xi)
    if problems:
        sys.exit('learn_v1.json breaks the contract, nothing was written:\n  ' + '\n  '.join(problems[:30])
                 + (f'\n  … (+{len(problems) - 30:,})' if len(problems) > 30 else ''))


# ── Alone ──

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--include-drafts', action='store_true',
                        help='Build every club stop, with its draft card or card: null (local and dev testing only).')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    import publish_files as publisher
    from sources.last_updated import record

    game_data = publisher.build_game_data()[0]
    clues = publisher.build_clues(game_data)[0]
    xi = publisher.build_xi()[0]
    grid = publisher.build_grid()[0]
    stars = publisher.read_side_output(ROOT / 'game_modes_data' / 'starting_xi' / 'output' / 'xi_stars.json',
                                       'game_modes_data/starting_xi/build_lineups.py')['lineups']
    content, counts, notes = build_learn(game_data, xi, grid, stars, clues, include_drafts=args.include_drafts)
    for note in notes:
        print(f'note: {note}')
    if content is None:
        print('No club stop has an approved card: no learn_v1.json (--include-drafts builds them all).')
        return
    body = publisher.encode(content)
    out = publisher.PUBLISH_DIR / 'learn_v1.json'
    publisher.PUBLISH_DIR.mkdir(exist_ok=True)
    out.write_bytes(body)
    print(f'{out.relative_to(ROOT)}: {len(body):,} bytes, {len(publisher.compress(body)):,} gzipped; '
          f'version {publisher.sha256(body)[:16]}…')
    print('  ' + ', '.join(f'{value:,} {key}' for key, value in counts.items()))
    cards = load_cards()
    drafts = sum(1 for stop in content['stops'].values()
                 if stop['club_id'] is not None and stop['card'] is not None
                 and cards[stop['club_id']]['status'] == 'draft')
    record(LEARN_DIR, 'built', **counts, draft_cards=drafts,
           no_card=sum(1 for stop in content['stops'].values() if stop['club_id'] is not None and stop['card'] is None))


if __name__ == '__main__':
    main()
