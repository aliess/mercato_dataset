"""
Dossiers, anchors and trails of learn_v1.json (CONTRACTS.md §1, "dossiers on a knowledge map";
plan footballquiz/docs/plans/learn_journey.md §3, §5, §6). build_learn.py calls build() once the
stops are built, and validate() with the rest of the file's checks.

Dossiers (8-15 players, or 1-3 lineups, with a reason to learn them):
  squad   a stop's players of today, best known first; a squad of 16+ is cut into "Today's stars"
          and "Today's squad" (and more parts when it is bigger still)
  era     a stop's former players by decade, legends first. A player may sit in any decade in which
          he spent half a season or more at the club, so a crowded decade can hand players to the
          next one; when a stop has too few of today's players for a squad dossier they join the
          2020s. A small club's decades are too thin for dossiers of 8, so one dossier may span
          several: its title then shows the span ("Napoli · 1980s–2020s") and its id the decade most
          of its players were there. Each stop's split is the cheapest assignment (a small min-cost flow): a player in
          his own decade costs 0, in another decade he was at the club 1, elsewhere more. A stop
          with fewer than 8 players borrows players who spent half a season or more at the club.
          Together, squad and era dossiers hold every player of game_data_v1.json.
  night   one famous match of a club stop's `nights` steps, both XIs
  nation  a nation's best World Cup or Euro run in the `nations` world: up to 3 lineups of one
          tournament (famous first, then the deepest rounds)
  route   a route of a `routes` step that 8+ players took (up to 15 of them, best known first)
  story   written, from sources/learn/dossiers.json; `approved` only unless --include-drafts

Map order inside a world: per stop, its squad, era, route and night dossiers, then the stories
drawn after it: a story follows the club of its world that most of its players played for (Grid
pool spells), else the stop most of them call home; with neither it closes the world.

Anchors: for every player of a dossier, up to 5 teammates of game_data_v1.json, best known first,
with the club and the overlapping seasons of their Grid pool spells (first and last season, as
grid_pool_v1.json writes them: 2016 is 2016/17).

Trails: dossier pairs that share a player or have 3+ anchor links between their players; strength
3 per shared player + 1 per link; at most 4 per dossier, strongest first.
"""

import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

LEARN_DIR = Path(__file__).resolve().parent
STORY_FILE = LEARN_DIR / 'dossiers.json'

KINDS = ('squad', 'era', 'night', 'route', 'nation', 'story')
PLAYER_KINDS = ('squad', 'era', 'route', 'story')
DOSSIER = (8, 15)
LINEUPS = (1, 3)
ROUTE_MIN_PLAYERS = 8
ANCHORS_MAX = 5
TRAILS_MAX = 4
TRAIL_MIN_LINKS = 3
DECADE_MIN_DAYS = 182
TODAY = '2020s'
FIELDS = ['kind', 'world', 'title', 'line', 'mystery', '{list}', 'stop', 'free']
STORY_FIELDS = ['id', 'world', 'status', 'title', 'line', 'mystery', 'players']
STORY_STATUSES = ('draft', 'approved')
STORY_LENGTH = {'title': (6, 40), 'line': (40, 150), 'teaser': (25, 130)}
TIERS = ('beginner', 'intermediate', 'expert')
ROUND_RANK = {'final': 0, 'semi-final': 1, 'quarter-final': 2}
BIG = 10_000                    # cost that makes the first 8 places of a dossier mandatory


# ── Ordering ──

def fame_key(player):
    return (player['difficulty'] != 'beginner', -player['market_value'], int(player['id']))


def era_key(player, legends):
    return (TIERS.index(player['difficulty']), player['id'] not in legends, -player['market_value'],
            int(player['id']))


def even_parts(items, parts):
    """`items` cut into `parts` runs as even as possible, the bigger first."""
    base, extra = divmod(len(items), parts)
    out, at = [], 0
    for index in range(parts):
        size = base + (1 if index < extra else 0)
        out.append(items[at:at + size])
        at += size
    return out


SCORE = re.compile(r'^(?P<home>.+) (?P<hg>\d+)–(?P<ag>\d+) (?P<away>.+?)( \(a\.e\.t\.\))?'
                   r'( \((?P<hp>\d+)–(?P<ap>\d+) pens\))?$')


def won_final(match):
    """True when `match` is a final that the lineup's team won (shootouts counted)."""
    found = SCORE.match(match['score'])
    if match['round'] != 'final' or not found:
        return False
    goals = (int(found['hg']), int(found['ag']))
    if goals[0] == goals[1] and found['hp']:
        goals = (int(found['hp']), int(found['ap']))
    winner = found['home'] if goals[0] > goals[1] else found['away'] if goals[1] > goals[0] else None
    return winner == match['team']


# ── Decades at a club ──

def decade_start_ms(decade):
    return datetime(decade, 1, 1, tzinfo=timezone.utc).timestamp() * 1000


def decade_days(moves, player_id, club_id):
    """{'1990s': days} of his spells at `club_id` (every senior club when None)."""
    days = Counter()
    for club, start, end, _loan in moves.spells(player_id):
        if (club_id is not None and club != club_id) or (club_id is None and not moves.senior(club)):
            continue
        year = datetime.fromtimestamp(start / 1000, tz=timezone.utc).year // 10 * 10
        at = start
        while at < end:
            stop = min(end, decade_start_ms(year + 10))
            days[f'{year}s'] += (stop - at) / 86_400_000
            at, year = stop, year + 10
    return days


def decade_gap(a, b):
    return abs(int(a[:4]) - int(b[:4])) // 10


# ── The cheapest split of a stop into squad and era dossiers ──

def min_cost_flow(nodes, edges, source, sink, need):
    """Successive shortest paths (Bellman-Ford, costs may be negative). `edges`: [(u, v, cap,
    cost)]. Returns (flow, cost, {edge index: flow})."""
    graph = {n: [] for n in nodes}
    to, cap, cost = [], [], []
    for u, v, c, w in edges:
        graph[u].append(len(to)); to.append(v); cap.append(c); cost.append(w)
        graph[v].append(len(to)); to.append(u); cap.append(0); cost.append(-w)
    flow = total = 0
    while flow < need:
        dist, prev = {n: None for n in nodes}, {}
        dist[source] = 0
        for _ in range(len(nodes)):
            changed = False
            for u in nodes:
                if dist[u] is None:
                    continue
                for e in graph[u]:
                    if cap[e] > 0 and (dist[to[e]] is None or dist[u] + cost[e] < dist[to[e]]):
                        dist[to[e]] = dist[u] + cost[e]
                        prev[to[e]] = e
                        changed = True
            if not changed:
                break
        if dist[sink] is None:
            break
        push, v = need - flow, sink
        while v != source:
            e = prev[v]
            push = min(push, cap[e])
            v = to[e ^ 1]
        v = sink
        while v != source:
            e = prev[v]
            cap[e] -= push
            cap[e ^ 1] += push
            v = to[e ^ 1]
        flow += push
        total += push * dist[sink]
    used = {i // 2: cap[i + 1] for i in range(0, len(to), 2)}
    return flow, total, used


def split_stop(entries):
    """entries: [(player id, today, own decade, {decades at the club})] in teaching order.
    Returns (squad player ids, {decade: [player ids]}): squad 0 or 8+ (cut later into parts of
    8-15), each decade 8-15, at the lowest cost."""
    n = len(entries)
    today_count = sum(1 for e in entries if e[1])
    decades = sorted({e[2] for e in entries} | {d for e in entries for d in e[3]})

    def cost(entry, bin_name):
        _pid, today, own, allowed = entry
        if bin_name == 'squad':
            return 0 if today else None
        if today:
            base = 1 if bin_name == TODAY else (2 if bin_name in allowed else None)
        else:
            base = 0 if bin_name == own else (1 if bin_name in allowed else None)
        if base is not None:
            return base
        return 4 * min(decade_gap(bin_name, d) for d in (allowed | {own})) + 2

    types = {}
    for entry in entries:
        types.setdefault((entry[1], entry[2], frozenset(entry[3])), []).append(entry)
    type_keys = sorted(types, key=lambda k: (not k[0], k[1], sorted(k[2])))

    def solve(squad_parts, chosen):
        bins = (['squad'] if squad_parts else []) + list(chosen)
        caps = {b: (DOSSIER[0], DOSSIER[1]) for b in chosen}
        if squad_parts:
            caps['squad'] = (DOSSIER[0] * squad_parts, DOSSIER[1] * squad_parts)
        nodes = ['S', 'T'] + [('t', i) for i in range(len(type_keys))] + [('b', b) for b in bins]
        edges, arcs = [], []
        for i, key in enumerate(type_keys):
            edges.append(('S', ('t', i), len(types[key]), 0))
            for b in bins:
                c = cost(types[key][0], b)
                if c is not None:
                    arcs.append((len(edges), i, b))
                    edges.append((('t', i), ('b', b), len(types[key]), c))
        lower_edges = []
        for b in bins:
            lower_edges.append(len(edges))
            edges.append((('b', b), 'T', caps[b][0], -BIG))
            edges.append((('b', b), 'T', caps[b][1] - caps[b][0], 0))
        flow, total, used = min_cost_flow(nodes, edges, 'S', 'T', n)
        if flow < n or any(used[e] < edges[e][2] for e in lower_edges):
            return None
        total += BIG * sum(edges[e][2] for e in lower_edges)
        return total, {(i, b): used[e] for e, i, b in arcs if used[e]}

    best = None
    low = -(-n // DOSSIER[1])
    for m in range(low, n // DOSSIER[0] + 1):
        if best is not None and m > low + 1:
            break
        for squad_parts in range(0, min(today_count // DOSSIER[0], m) + 1):
            for chosen in combinations(decades, m - squad_parts):
                result = solve(squad_parts, chosen)
                if result is None:
                    continue
                key = (result[0], m, -squad_parts, chosen)
                if best is None or key < best[0]:
                    best = (key, squad_parts, chosen, result[1])
    if best is None:
        return None
    _key, squad_parts, chosen, flows = best
    squad, by_decade = [], {d: [] for d in chosen}
    for i, key in enumerate(type_keys):
        members = [e[0] for e in types[key]]
        for b in (['squad'] if squad_parts else []) + list(chosen):
            take = flows.get((i, b), 0)
            (squad if b == 'squad' else by_decade[b]).extend(members[:take])
            members = members[take:]
    return squad, squad_parts, by_decade


# ── Stories ──

def plain(text):
    import unicodedata
    text = unicodedata.normalize('NFKD', text or '')
    return ''.join(c for c in text if not unicodedata.combining(c)).lower().replace('ß', 'ss')


def name_words(player, alt_names):
    names = [player.get('player_name'), player.get('name_in_home_country'), player.get('player_slug')]
    names += alt_names.get(player['id'], []) if isinstance(alt_names, dict) else []
    return {w for name in names for w in re.split(r'[^a-z0-9]+', plain((name or '').replace('-', ' ')))
            if len(w) >= 3}


def names_in(text, words):
    """Words of a player's names that `text` spells out ("Mbappé's" counts)."""
    return sorted(set(re.split(r'[^a-z0-9]+', plain(text))) & words)


def load_stories(players, world_ids, alt_names):
    """The story dossiers of dossiers.json, checked: fields, ids, status, lengths, 8-15 players of
    the game data, the mystery among them and his name nowhere in his teaser."""
    if not STORY_FILE.exists():
        return []
    stories = json.loads(STORY_FILE.read_text(encoding='utf-8'))
    problems, seen = [], set()
    for story in stories:
        label = story.get('id', '?')
        if list(story) != STORY_FIELDS:
            problems.append(f'{label}: fields {list(story)}, expected {STORY_FIELDS}')
            continue
        if not re.fullmatch(r'story_[a-z0-9_]+', story['id']) or story['id'] in seen:
            problems.append(f'{label}: id must be story_<slug>, used once')
        seen.add(story['id'])
        if story['status'] not in STORY_STATUSES:
            problems.append(f"{label}: status {story['status']!r}")
        if story['world'] not in world_ids or story['world'] == 'summit':
            problems.append(f"{label}: world {story['world']!r} is not a world of the map (nor the Summit)")
        mystery = story['mystery']
        texts = {'title': story['title'], 'line': story['line'],
                 'teaser': mystery.get('teaser') if isinstance(mystery, dict) else None}
        for field, text in texts.items():
            if not isinstance(text, dict) or list(text) != ['en', 'ar'] or not isinstance(text['en'], str):
                problems.append(f'{label}: {field} must be {{en, ar}}')
                continue
            low, high = STORY_LENGTH[field]
            if not low <= len(text['en']) <= high:
                problems.append(f"{label}: {field} has {len(text['en'])} characters, expected {low}-{high}")
            if text['ar'] is not None and not (isinstance(text['ar'], str) and text['ar']):
                problems.append(f'{label}: {field}.ar must be text or null')
        if not isinstance(mystery, dict) or list(mystery) != ['player', 'teaser']:
            problems.append(f'{label}: mystery must be {{player, teaser}}')
            continue
        ids = story['players']
        if not DOSSIER[0] <= len(ids) <= DOSSIER[1] or len(set(ids)) != len(ids):
            problems.append(f'{label}: {len(ids)} players (unique {len(set(ids))}), expected {DOSSIER[0]}-{DOSSIER[1]}')
        unknown = [p for p in ids if p not in players]
        if unknown:
            problems.append(f'{label}: players not in game_data_v1.json {unknown}')
        if mystery['player'] not in ids:
            problems.append(f"{label}: mystery {mystery['player']} is not one of its players")
        elif isinstance(texts['teaser'], dict) and isinstance(texts['teaser'].get('en'), str):
            spelled = names_in(texts['teaser']['en'], name_words(players[mystery['player']], alt_names))
            if spelled:
                problems.append(f'{label}: the teaser spells out the mystery player ({", ".join(spelled)})')
    if problems:
        sys.exit(f'{STORY_FILE.relative_to(LEARN_DIR.parent.parent)}:\n  ' + '\n  '.join(problems))
    return stories


# ── Building ──

def build(content, stop_lists, players, xi, grid, lineup_stars, moves, legends, route_index,
          include_drafts=False, notes=None, alt_names=None):
    """({'by_id': dossiers, 'order': {world id: [dossier id]}}, anchors, trails)."""
    notes = notes if notes is not None else []
    worlds, stops, hooks = content['worlds'], content['stops'], content['hooks']
    grid_names = {club['id']: club['name'] for club in grid['clubs']}
    matches = {m['id']: m for m in xi['matches']}
    by_id, order = {}, {}
    placed_at = {}              # dossier id → stop id it is drawn after

    def mystery_of(ids):
        """The best known player whose hook is a clue (the most telling lead), else the first."""
        for pid in ids:
            if 'clue' in hooks.get(pid, {}):
                return pid
        return ids[0]

    def lineup_players(ids):
        found = []
        for lineup in ids:
            for slot in matches[lineup]['slots']:
                pid = slot.get('player_id')
                if pid in players and pid not in found:
                    found.append(pid)
        return sorted(found, key=lambda p: fame_key(players[p]))

    def add(dossier_id, world_id, stop_id, **fields):
        by_id[dossier_id] = fields
        order.setdefault(world_id, []).append(dossier_id)
        placed_at[dossier_id] = stop_id

    # Players who spent half a season or more at a club, for stops that are too small
    club_regulars = {}
    for pid in sorted(players, key=int):
        for club, days in Counter({c: sum(decade_days(moves, pid, c).values())
                                   for c in {s[0] for s in moves.spells(pid)}}).items():
            if days >= DECADE_MIN_DAYS:
                club_regulars.setdefault(club, []).append((pid, days))

    for world in worlds:
        wid = world['id']
        order.setdefault(wid, [])
        for stop_id in world['stops']:
            stop = stops[stop_id]
            club_id = stop['club_id']
            stop_name = grid_names[club_id] if club_id is not None else stop['name']['en']

            if stop_id in stop_lists and stop_lists[stop_id]:
                ordered = list(stop_lists[stop_id])
                entries, in_stop = [], {p['id'] for p, _era in ordered}

                def entry(player, today, own_era):
                    club = club_id if club_id is not None else moves.home(player)[0]
                    days = decade_days(moves, player['id'], club)
                    allowed = {d for d, n in days.items() if n >= DECADE_MIN_DAYS}
                    return (player['id'], today, TODAY if today else own_era, allowed)

                for player, era in ordered:
                    entries.append(entry(player, era is None, era))
                if len(entries) < DOSSIER[0] and club_id is not None:
                    borrowed = sorted((c for c in club_regulars.get(club_id, []) if c[0] not in in_stop),
                                      key=lambda c: (-c[1], fame_key(players[c[0]])))
                    for pid, _days in borrowed[:DOSSIER[0] - len(entries)]:
                        entries.append(entry(players[pid], False, moves.era(pid, club_id)))
                    notes.append(f'{stop_id}: {len(ordered)} players, {len(entries) - len(ordered)} borrowed '
                                 'from those who spent half a season or more there')
                result = split_stop(entries)
                if result is None:
                    sys.exit(f'learn: {stop_id} ({len(entries)} players) can not be cut into dossiers of '
                             f'{DOSSIER[0]}-{DOSSIER[1]}')
                squad, parts, by_decade = result
                squad = sorted(squad, key=lambda p: fame_key(players[p]))
                for n, part in enumerate(even_parts(squad, parts) if parts else [], start=1):
                    subject = "Today's squad" if parts == 1 else "Today's stars" if n == 1 else \
                        "Today's squad" if parts == 2 else f"Today's squad, part {n - 1}"
                    add(f'squad_{stop_id}_{n}', wid, stop_id, kind='squad', world=wid,
                        title={'en': f'{stop_name} · {subject}', 'ar': None}, line=None,
                        mystery={'player': mystery_of(part), 'teaser': None}, players=part,
                        stop=stop_id, free=bool(stop['free']))
                # Each member's decade: the dossier's when he was at the club then, else his own
                # closest to it. The id takes the most common one (the bin's when that is taken);
                # the title shows the span, so a small club's mixed dossier reads "1980s–2020s".
                by_pid = {e[0]: e for e in entries}
                labels = {}
                for decade in sorted(by_decade):
                    own = []
                    for pid in by_decade[decade]:
                        _p, _today, era, allowed = by_pid[pid]
                        own.append(decade if decade in allowed or decade == era else
                                   min(allowed | {era}, key=lambda d: (decade_gap(d, decade), d)))
                    tally = Counter(own)
                    main = min(tally, key=lambda d: (-tally[d], d != decade, d))
                    labels[decade] = (main, min(own), max(own))
                taken = Counter(main for main, _a, _b in labels.values())
                for decade in sorted(by_decade):
                    main, first, last = labels[decade]
                    if taken[main] > 1 or (main != decade and main in by_decade):
                        main = decade
                    span = first if first == last else f'{first}–{last}'
                    ids = sorted(by_decade[decade], key=lambda p: era_key(players[p], legends))
                    add(f'era_{stop_id}_{main}', wid, stop_id, kind='era', world=wid,
                        title={'en': f'{stop_name} · {span}', 'ar': None}, line=None,
                        mystery={'player': mystery_of(ids), 'teaser': None}, players=ids,
                        stop=stop_id, free=False)

            for step in stop['steps']:
                if step['kind'] == 'routes':
                    for route in step['routes']:
                        dossier_id = f"route_{route['from']}_{route['to']}"
                        full = next(r for r in route_index[route['from']]
                                    if (r['from'], r['to']) == (route['from'], route['to']))
                        if dossier_id in by_id or len(full['players']) < ROUTE_MIN_PLAYERS:
                            continue
                        ids = sorted(full['players'], key=lambda p: fame_key(players[p]))[:DOSSIER[1]]
                        add(dossier_id, wid, stop_id, kind='route', world=wid,
                            title={'en': f"{route['from_name']} → {route['to_name']} · "
                                         f"{len(full['players'])} made the move", 'ar': None},
                            line=None, mystery={'player': mystery_of(ids), 'teaser': None}, players=ids,
                            stop=None, free=False)
                elif step['kind'] == 'nights' and wid != 'nations':
                    for lineup in step['lineups']:
                        match_key = lineup.rsplit('-', 1)[0]
                        sides = [lineup] + [i for i in (f'{match_key}-home', f'{match_key}-away')
                                            if i != lineup and i in matches]
                        if any(f'night_{i}' in by_id for i in sides):
                            continue
                        match = matches[lineup]
                        known = lineup_players(sides)
                        if not known:
                            notes.append(f'night {lineup}: no player of the game data, left out')
                            continue
                        leg = {1: ', first leg', 2: ', second leg'}.get(match['leg'], '')
                        add(f'night_{lineup}', wid, stop_id, kind='night', world=wid,
                            title={'en': f"{match['score']} · {match['competition']}{leg} {match['year']}",
                                   'ar': None},
                            line=None, mystery={'player': mystery_of(known), 'teaser': None}, lineups=sides,
                            stop=None, free=False)

            if wid == 'nations':
                lineups = [i for step in stop['steps'] if step['kind'] == 'nights' for i in step['lineups']]
                for code, prefix in (('wc', 'World Cup'), ('euro', 'Euro')):
                    runs = {}
                    for i in lineups:
                        if matches[i]['competition'].startswith(prefix):
                            runs.setdefault(matches[i]['year'], []).append(i)
                    if not runs:
                        continue

                    def run_key(item):
                        year, ids = item
                        return (not any(won_final(matches[i]) for i in ids), -sum(matches[i]['famous'] for i in ids),
                                min(ROUND_RANK.get(matches[i]['round'], 3) for i in ids), -len(ids),
                                -year, -sum(lineup_stars.get(i, 0) for i in ids))
                    year, ids = min(runs.items(), key=run_key)
                    ids = sorted(ids, key=lambda i: (not matches[i]['famous'],
                                                     ROUND_RANK.get(matches[i]['round'], 3)))[:LINEUPS[1]]
                    ids = sorted(ids, key=lambda i: (matches[i]['date'], i))
                    known = lineup_players(ids)
                    name = 'Euro 2020' if prefix == 'Euro' and year == 2021 else f'{prefix} {year}'
                    add(f"nation_{stop['short']}_{code}", wid, stop_id, kind='nation', world=wid,
                        title={'en': f"{stop['nation']} · {name}", 'ar': None}, line=None,
                        mystery={'player': mystery_of(known), 'teaser': None}, lineups=ids,
                        stop=None, free=False)

    # Stories, after the stop where most of their players are taught
    world_ids = [w['id'] for w in worlds]
    stories = load_stories(players, set(world_ids), alt_names or {})
    statuses = ('approved', 'draft') if include_drafts else ('approved',)
    clubs_of = {row[0]: {club for club, _a, _b in row[4]} for row in grid['players']}
    home_stop = {}
    for dossier_id, d in by_id.items():
        if d['kind'] in ('squad', 'era'):
            for pid in d['players']:
                home_stop.setdefault(pid, d['stop'])
    drafts = 0
    for story in stories:
        if story['status'] not in statuses:
            continue
        drafts += story['status'] == 'draft'
        world_stops = next(w['stops'] for w in worlds if w['id'] == story['world'])
        here = Counter()
        for stop_id in world_stops:
            club = stops[stop_id]['club_id']
            if club is not None:
                here[stop_id] = sum(1 for p in story['players'] if club in clubs_of.get(p, ()))
        if not here or max(here.values()) == 0:
            here = Counter(home_stop[p] for p in story['players'] if home_stop.get(p) in world_stops)
        rank = {s: i for i, s in enumerate(world_stops)}
        stop_id = min(here, key=lambda s: (-here[s], rank[s])) if here and max(here.values()) else None
        by_id[story['id']] = {'kind': 'story', 'world': story['world'], 'title': story['title'],
                              'line': story['line'], 'mystery': story['mystery'], 'players': story['players'],
                              'stop': None, 'free': False}
        placed_at[story['id']] = stop_id
        world_order = order[story['world']]
        if stop_id is None:
            world_order.append(story['id'])
        else:
            last = max(i for i, d in enumerate(world_order) if placed_at[d] == stop_id)
            world_order.insert(last + 1, story['id'])
    if drafts:
        notes.append(f'built with --include-drafts: {drafts} story dossiers are drafts')
    for wid in world_ids:
        order.setdefault(wid, [])
    position = {d: i for i, d in enumerate(d for wid in world_ids for d in order[wid])}
    by_id = {d: by_id[d] for d in sorted(by_id, key=lambda d: position[d])}

    # Anchors
    in_dossiers = set()
    for d in by_id.values():
        in_dossiers.update(d.get('players', []))
        in_dossiers.add(d['mystery']['player'])
    anchors = build_anchors(sorted(in_dossiers, key=int), players, grid)

    # Trails
    members = {d_id: set(d['players']) if 'players' in d else set(lineup_players(d['lineups']))
               for d_id, d in by_id.items()}
    trails = build_trails(members, anchors, position)
    return {'by_id': by_id, 'order': order}, anchors, trails


def build_anchors(ids, players, grid):
    spells = {row[0]: row[4] for row in grid['players'] if row[0] in players}
    at_club = {}
    for pid, rows in spells.items():
        for club, first, last in rows:
            at_club.setdefault(club, []).append((pid, first, last))
    anchors = {}
    for pid in ids:
        best = {}
        for club, first, last in spells.get(pid, []):
            for other, o_first, o_last in at_club[club]:
                if other == pid:
                    continue
                start, end = max(first, o_first), min(last, o_last)
                if start > end:
                    continue
                key = (end - start, end, -club)
                if other not in best or key > best[other][0]:
                    best[other] = (key, [other, club, start, end])
        top = sorted(best, key=lambda o: fame_key(players[o]))[:ANCHORS_MAX]
        anchors[pid] = [best[o][1] for o in top]
    return anchors


def build_trails(members, anchors, position):
    of_player = {}
    for d_id in sorted(members, key=lambda d: position[d]):
        for pid in members[d_id]:
            of_player.setdefault(pid, []).append(d_id)
    shared = Counter()
    for pid, ds in of_player.items():
        for a, b in combinations(ds, 2):
            shared[(a, b)] += 1
    links = {}
    pairs = {tuple(sorted((pid, a[0]), key=int)) for pid, rows in anchors.items() for a in rows}
    for p, q in pairs:
        for a in of_player.get(p, []):
            for b in of_player.get(q, []):
                if a == b:
                    continue
                key = (a, b) if position[a] < position[b] else (b, a)
                links.setdefault(key, set()).add((p, q))
    scored = []
    for key in set(shared) | set(links):
        n_shared, n_links = shared.get(key, 0), len(links.get(key, ()))
        if n_shared or n_links >= TRAIL_MIN_LINKS:
            scored.append((-(3 * n_shared + n_links), position[key[0]], position[key[1]], key))
    scored.sort()
    # First every dossier's strongest trail, then the rest, strongest first
    per, taken = Counter(), set()
    for first_pass in (True, False):
        for _s, _pa, _pb, key in scored:
            a, b = key
            if key in taken or per[a] >= TRAILS_MAX or per[b] >= TRAILS_MAX:
                continue
            if first_pass and per[a] and per[b]:
                continue
            per[a] += 1
            per[b] += 1
            taken.add(key)
    return [list(key) for _s, _pa, _pb, key in scored if key in taken]


# ── The contract's checks ──

def validate(content, game_data, xi):
    problems = []
    players = {p['id']: p for p in game_data['players']}
    matches = {m['id']: m for m in xi['matches']}
    worlds = content['worlds']
    dossiers = content.get('dossiers', {})
    listed = [d for w in worlds for d in w.get('dossiers', [])]
    if sorted(listed) != sorted(dossiers) or len(set(listed)) != len(listed):
        problems.append('every dossier must be listed in exactly one world, and every listed dossier must exist')
    position = {d: i for i, d in enumerate(listed)}
    world_of = {d: w['id'] for w in worlds for d in w.get('dossiers', [])}
    if any(w['id'] == 'summit' and w.get('dossiers') for w in worlds):
        problems.append("the Summit's dossier list must be empty")
    patterns = {'squad': r'squad_(.+)_\d+', 'era': r'era_(.+)_(19|20)\d0s', 'night': r'night_(.+)',
                'route': r'route_\d+_\d+', 'nation': r'nation_[A-Z]{3}_(wc|euro)', 'story': r'story_[a-z0-9_]+'}
    covered = set()
    for d_id, d in dossiers.items():
        kind = d.get('kind')
        if kind not in KINDS or not re.fullmatch(patterns[kind], d_id):
            problems.append(f'{d_id}: kind {kind!r} or id')
            continue
        listing = 'players' if kind in PLAYER_KINDS else 'lineups'
        expected = [f if f != '{list}' else listing for f in FIELDS]
        if list(d) != expected:
            problems.append(f'{d_id}: fields {list(d)}, expected {expected}')
            continue
        if d['world'] != world_of.get(d_id):
            problems.append(f"{d_id}: world {d['world']} but listed in {world_of.get(d_id)}")
        if list(d['title']) != ['en', 'ar'] or not d['title']['en']:
            problems.append(f'{d_id}: title')
        if kind == 'story' and (not isinstance(d['line'], dict) or not d['line'].get('en')):
            problems.append(f'{d_id}: a story needs a line')
        if d['line'] is not None and list(d['line']) != ['en', 'ar']:
            problems.append(f'{d_id}: line')
        if kind in PLAYER_KINDS:
            ids = d['players']
            ok = DOSSIER[0] <= len(ids) <= DOSSIER[1] and len(set(ids)) == len(ids) and all(p in players for p in ids)
            pool = ids
        else:
            ids = d['lineups']
            ok = LINEUPS[0] <= len(ids) <= LINEUPS[1] and len(set(ids)) == len(ids) and all(i in matches for i in ids)
            pool = [s.get('player_id') for i in ids if i in matches for s in matches[i]['slots']]
        if not ok:
            problems.append(f'{d_id}: {len(ids)} {listing}, expected '
                            f'{DOSSIER if listing == "players" else LINEUPS} known ones, each once')
        mystery = d['mystery']
        if list(mystery) != ['player', 'teaser'] or mystery['player'] not in pool or mystery['player'] not in players:
            problems.append(f'{d_id}: mystery {mystery}')
        elif (kind == 'story') != (mystery['teaser'] is not None):
            problems.append(f'{d_id}: a teaser is written for story dossiers only')
        if kind in ('squad', 'era'):
            if d['stop'] not in content['stops']:
                problems.append(f"{d_id}: stop {d['stop']!r}")
            covered.update(ids)
        elif d['stop'] is not None:
            problems.append(f'{d_id}: stop must be null')
        if d['free'] is not (kind == 'squad' and d['stop'] in content['stops'] and content['stops'][d['stop']]['free']):
            problems.append(f'{d_id}: free must be true on the free stop\'s squad dossiers only')
    missing = set(players) - covered
    if missing:
        problems.append(f'{len(missing):,} players are in no squad or era dossier: {sorted(missing, key=int)[:10]}')
    anchors = content.get('anchors', {})
    needed = {p for d in dossiers.values() for p in d.get('players', [])} | \
        {d['mystery']['player'] for d in dossiers.values()}
    if set(anchors) != needed:
        problems.append(f'anchors: {len(needed - set(anchors)):,} dossier players without an entry, '
                        f'{len(set(anchors) - needed):,} entries for others')
    for pid, rows in anchors.items():
        ok = len(rows) <= ANCHORS_MAX and len({r[0] for r in rows}) == len(rows) and all(
            len(r) == 4 and r[0] in players and r[0] != pid and isinstance(r[1], int)
            and isinstance(r[2], int) and isinstance(r[3], int) and r[2] <= r[3] for r in rows)
        if not ok:
            problems.append(f'anchors of {pid}: {rows}')
    per, seen = Counter(), set()
    for trail in content.get('trails', []):
        ok = (isinstance(trail, list) and len(trail) == 2 and all(t in position for t in trail)
              and position[trail[0]] < position[trail[1]] and frozenset(trail) not in seen)
        if ok:
            seen.add(frozenset(trail))
            per.update(trail)
        else:
            problems.append(f'trail {trail}')
    over = [d for d, n in per.items() if n > TRAILS_MAX]
    if over:
        problems.append(f'more than {TRAILS_MAX} trails: {over[:5]}')
    return problems
