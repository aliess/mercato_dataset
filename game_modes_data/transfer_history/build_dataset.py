#!/usr/bin/env python3
"""
Build the two files the app's game data is made from (publish_files.py turns them into
Storage cache/game_data_v1.json):

  game_modes_data/transfer_history/output/player_profiles.csv    → the file's `players`
  game_modes_data/transfer_history/output/transfer_history.csv   → the file's `transfers`

and, for the admin panel's cache/tiers_v1.json, why each player has his tier:

  game_modes_data/transfer_history/output/player_tiers.json      → the file's `players`, `rules`

Players are picked from the one player list every mode shares (sources/squads.py: everyone in
a pool club's squad since 1990/91), plus every legend (sources/legends/legends.json) and
EXTRA_PLAYER_IDS. Profiles, transfer histories and market values all come from Transfermarkt's
JSON API (sources/tm_api.py). Each player's fame, tier and whether he is still playing come
from sources/fame.py and are written as the columns market_value, difficulty and active.

Legends with fewer than MIN_CLUBS clubs (Maldini, Totti) are kept for Player Story only
(story_only); other players with fewer are left out.

The transfermarkt-datasets tables (sources/dataset/) are still read for the club names and
countries they know (so names stay as in earlier imports), for the transfers of a player whose
API request fails, and for country names (sources/player_facts.py).

Usage:
    python game_modes_data/transfer_history/build_dataset.py
    python game_modes_data/transfer_history/build_dataset.py --from-cache   # reuse cached API
        responses however old (no new requests for anything already cached)
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

MODE_DIR = Path(__file__).resolve().parent
ROOT = MODE_DIR.parent.parent
sys.path.insert(0, str(ROOT))

import compare_outputs
from sources import fame
from sources.club_types import is_reserve_side
from sources.country_names import canonical
from sources.last_updated import record
from sources.names import latin_name, player_name
from sources.player_facts import POSITION_NAMES, POSITIONS, country_names
from sources.squads import pool_clubs, squad_spells
from sources.tm_api import DEFAULT_CACHE_DIR, TransfermarktAPI

# ── Player selection ──
# Who is in the game data at all goes by the raw fame (fame.fame: Transfermarkt peak, raised by
# the legend value), not by fame now in today's money: the tiers use that, the pool doesn't.
# Stars: fame (sources/fame.py) at or above this (Jamie Vardy peaked at exactly 20M).
STAR_MIN_VALUE = 20_000_000
# Lesser players are kept if they played for one of these clubs and reached this fame.
PRESTIGIOUS_MIN_VALUE = 10_000_000
PRESTIGIOUS_CLUB_IDS = {
    11: 'Arsenal',
    5: 'AC Milan',
    418: 'Real Madrid',
    131: 'Barcelona',
    631: 'Chelsea',
}
# Players who were in Transfer Path before it was picked from the squads (2026-10-16) but were
# never in a pool club's squad. They stay while they still qualify.
EXTRA_PLAYER_IDS = {
    102740: 'Mario Fernandes', 69683: 'Alan Dzagoev', 489893: 'Yuri Alberto',
    515744: 'Eduard Spertsyan', 15650: 'Igor Denisov', 289059: 'Mykola Matvienko',
    403315: 'Claudinho', 580461: 'Kevin Viveros', 1004741: 'Pedro',
}

# Players whose history shows fewer first-team clubs than this are left out of Transfer Path:
# a one-club player (Koke, Iñaki Williams) has no transfer path to guess. A legend with fewer
# stays, for Player Story only (story_only).
MIN_CLUBS = 2
# Not clubs: they stay in a history but don't count towards MIN_CLUBS.
SPECIAL_CLUB_NAMES = {'Without Club', 'Retired', 'Career break', 'Unknown', 'Disqualification', 'Own Youth'}

# Transfers TO youth and reserve sides are dropped. The API says which clubs those are
# (see is_reserve_side). These name endings are only the fallback for clubs the API didn't
# return; on their own they also catch real clubs (Willem II, Esbjerg fB).
YOUTH_SUFFIXES = ('YTH', 'Youth', 'You', 'U19', 'U17', 'Yth', 'U20', 'U21', 'U18',
                  'U16', 'U23', 'U22', 'U24', 'II', 'Yth.', 'B')

# ── Output columns (the app decodes these names from game_data_v1.json; see ../CONTRACTS.md) ──
PROFILES_COLUMNS = [
    'player_id', 'player_slug', 'player_name', 'player_image_url',
    'name_in_home_country', 'date_of_birth', 'place_of_birth', 'country_of_birth',
    'height', 'citizenship', 'is_eu', 'position', 'main_position', 'foot',
    'current_club_id', 'current_club_name', 'joined', 'contract_expires',
    'outfitter', 'social_media_url', 'player_agent_id', 'player_agent_name',
    'contract_option', 'date_of_last_contract_extension', 'on_loan_from_club_id',
    'on_loan_from_club_name', 'contract_there_expires', 'second_club_url',
    'second_club_name', 'third_club_url', 'third_club_name', 'fourth_club_url',
    'fourth_club_name', 'date_of_death', 'market_value',
    # From sources/fame.py (CONTRACTS.md §1): tier, still playing, Player Story only.
    'difficulty', 'active', 'story_only',
]
TRANSFERS_COLUMNS = [
    'player_id', 'season_name', 'transfer_date', 'from_team_id', 'from_team_name',
    'to_team_id', 'to_team_name', 'transfer_type', 'value_at_transfer',
    'transfer_fee', 'from_team_country', 'to_team_country',
]
# The API's position group → the dataset's main position, as the file always had it.
MAIN_POSITIONS = {'GOALKEEPER': 'Goalkeeper', 'DEFENDER': 'Defender', 'MIDFIELDER': 'Midfield',
                  'FORWARD': 'Attack'}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset-dir', type=Path, default=ROOT / 'sources' / 'dataset')
    parser.add_argument('--output-dir', type=Path, default=MODE_DIR / 'output')
    parser.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument('--from-cache', action='store_true',
                        help='Reuse cached API responses however old.')
    parser.add_argument('--max-age-hours', type=float, default=24,
                        help='Reuse cached API responses younger than this (default 24).')
    parser.add_argument('--workers', type=int, default=4, help='Parallel API requests (default 4).')
    return parser.parse_args()


def read_table(dataset_dir, name, **kwargs):
    """Read dataset/<name>.csv.gz (from download_dataset.py), else dataset/<name>.csv (Kaggle zip)."""
    for path in (dataset_dir / f'{name}.csv.gz', dataset_dir / f'{name}.csv'):
        if path.exists():
            return pd.read_csv(path, **kwargs)
    sys.exit(f'Missing {name}.csv in {dataset_dir}. Run sources/download_dataset.py first.')


def is_youth_team(name):
    return isinstance(name, str) and name.strip().endswith(YOUTH_SUFFIXES)


# ── Dataset tables (club names and countries only) ──

class Dataset:
    def __init__(self, dataset_dir):
        print(f'Loading dataset tables from {dataset_dir}')
        self.clubs = read_table(dataset_dir, 'clubs', usecols=['club_id', 'name', 'domestic_competition_id'])
        self.competitions = read_table(dataset_dir, 'competitions')
        self.countries = read_table(dataset_dir, 'countries', usecols=['country_id', 'country_name'])
        self.transfers = read_table(dataset_dir, 'transfers')

        # club_id → country, via the club's domestic league (same as before).
        league_country = self.competitions.set_index('competition_id')['country_name']
        self.club_country = (self.clubs.set_index('club_id')['domestic_competition_id']
                             .map(league_country).dropna().to_dict())
        self.country_by_id = self.countries.set_index('country_id')['country_name'].to_dict()
        self.club_official_name = self.clubs.set_index('club_id')['name'].to_dict()

    def transfers_in_output_schema(self, player_ids):
        t = self.transfers[self.transfers['player_id'].isin(player_ids)]
        return pd.DataFrame({
            'player_id': t['player_id'],
            'season_name': t['transfer_season'],
            'transfer_date': t['transfer_date'],
            'from_team_id': t['from_club_id'],
            'from_team_name': t['from_club_name'],
            'to_team_id': t['to_club_id'],
            'to_team_name': t['to_club_name'],
            'transfer_type': None,
            'value_at_transfer': t['market_value_in_eur'],
            'transfer_fee': t['transfer_fee'],
            'from_team_country': None,
            'to_team_country': None,
        }, columns=TRANSFERS_COLUMNS)


# ── Transfermarkt API ──

FREE_TYPES = {'ACTIVE_LOAN_TRANSFER', 'RETURNED_FROM_PREVIOUS_LOAN'}


def transfer_fee(transfer):
    """Fee in EUR like the dataset: 0 for free moves and loans without a fee, empty when unknown."""
    fee = transfer['details'].get('fee') or {}
    if fee.get('value') is not None:
        return fee['value']
    label = ((fee.get('compact') or {}).get('content') or '').lower()
    if transfer['typeDetails'].get('type') in FREE_TYPES or 'free' in label or 'loan' in label:
        return 0
    return None


def fetch_transfers(api, dataset, player_ids):
    """Transfer histories from the API, in the output schema.

    Returns (transfers DataFrame, {club id: API club}, {player id: date of his latest move up
    to today}, ids whose request failed). Moves dated after today (pending) are left out.
    """
    print('Fetching transfer histories...')
    histories = api.transfer_histories(
        player_ids, progress=lambda done, total: print(f'  {done:,}/{total:,}'))
    failed = [int(player_id) for player_id, _ in api.failed]
    print(f'  got {len(histories):,} histories ({api.requests:,} API requests, {api.cache_hits:,} cached)')
    if failed:
        print(f'  ⚠ {len(failed):,} players failed and keep their dataset transfers, e.g. {api.failed[0]}')

    club_ids = set()
    for history in histories.values():
        for t in history:
            club_ids.add(t['transferSource']['clubId'])
            club_ids.add(t['transferDestination']['clubId'])
    clubs = api.clubs(club_ids)
    print(f'  resolved {len(clubs):,}/{len(club_ids):,} clubs')

    # Keep the dataset's club names ("Besiktas", "Stade Rennais") so names stay consistent with
    # earlier imports; clubs the dataset never saw get Transfermarkt's short name ("Man City").
    dataset_names = pd.concat([
        dataset.transfers[['from_club_id', 'from_club_name']].set_axis(['club_id', 'name'], axis=1),
        dataset.transfers[['to_club_id', 'to_club_name']].set_axis(['club_id', 'name'], axis=1),
    ]).dropna().groupby('club_id')['name'].agg(lambda names: names.mode().iloc[0]).to_dict()

    def short_name(club_id):
        name = dataset_names.get(int(club_id))
        if name:
            return name
        club = clubs.get(str(club_id))
        return (club['baseDetails'].get('shortName') or club['name']) if club else None

    def club_country(club_id, fallback_country_id):
        club_id = int(club_id)
        if club_id in dataset.club_country:
            return dataset.club_country[club_id]
        club = clubs.get(str(club_id))
        country_id = club['baseDetails'].get('countryId') if club else None
        return dataset.country_by_id.get(country_id) or dataset.country_by_id.get(fallback_country_id)

    today = date.today().isoformat()
    rows, last_move = [], {}
    skipped_future = skipped_unknown = 0
    for player_id, history in histories.items():
        for t in history:
            details = t['details']
            transfer_date = (details.get('date') or '')[:10]
            if not transfer_date:
                continue
            if transfer_date > today:
                skipped_future += 1
                continue
            last_move[int(player_id)] = max(last_move.get(int(player_id), ''), transfer_date)
            source, destination = t['transferSource'], t['transferDestination']
            to_name = short_name(destination['clubId'])
            if not to_name:
                skipped_unknown += 1
                continue
            value = (details.get('marketValue') or {}).get('value')
            rows.append({
                'player_id': int(player_id),
                'season_name': (details.get('season') or {}).get('display'),
                'transfer_date': transfer_date,
                'from_team_id': int(source['clubId']),
                'from_team_name': short_name(source['clubId']),
                'to_team_id': int(destination['clubId']),
                'to_team_name': to_name,
                'transfer_type': None,
                'value_at_transfer': value,
                'transfer_fee': transfer_fee(t),
                'from_team_country': club_country(source['clubId'], source.get('countryId')),
                'to_team_country': club_country(destination['clubId'], destination.get('countryId')),
            })
    if skipped_future:
        print(f'  skipped {skipped_future:,} transfers dated after today (pending moves)')
    if skipped_unknown:
        print(f'  skipped {skipped_unknown:,} transfers to unknown clubs')
    transfers = pd.DataFrame(rows, columns=TRANSFERS_COLUMNS)
    if failed:
        transfers = pd.concat([transfers, dataset.transfers_in_output_schema(failed)], ignore_index=True)
    return transfers, clubs, last_move, failed


RETIRED_CLUB_ID = 123
DECEASED_CLUB_ID = 4023  # Transfermarkt files players who have died under a club named "---"


def current_club(profile):
    """His current club id. A player who has died gets Retired's id, like any other former player."""
    current = next((a for a in profile.get('clubAssignments') or [] if a.get('type') == 'current'), None)
    if not current:
        return None
    club_id = int(current['clubId'])
    return RETIRED_CLUB_ID if club_id == DECEASED_CLUB_ID else club_id


def profile_row(player_id, p, countries, clubs, dataset):
    """One player_profiles.csv row from his API profile (fame columns are added later)."""
    attributes = p.get('attributes') or {}
    life = p.get('lifeDates') or {}
    birth = p.get('birthPlaceDetails') or {}
    club_id = current_club(p)
    club = clubs.get(str(club_id)) if club_id is not None else None
    official = dataset.club_official_name.get(club_id) if club_id is not None else None
    if not official and club:
        # First teams carry their legal name ("Derby County Football Club") like the dataset;
        # Retired / Without Club / youth sides keep their own name (the parent's is localized).
        is_first_team = club['baseDetails'].get('clubTypeId') == 1 and not club.get('isSpecialClub')
        superior = (club['baseDetails'].get('superiorClub') or {}).get('name')
        official = superior if is_first_team and superior else club['name']
    position = (attributes.get('position') or {}).get('name')
    position = POSITION_NAMES.get(position, position)
    height = attributes.get('height')
    slug = (p.get('relativeUrl') or '').strip('/').split('/')[0] or None
    row = dict.fromkeys(PROFILES_COLUMNS)
    row.update({
        'player_id': int(player_id),
        'player_slug': slug,
        'player_name': player_name(player_id, p.get('name')),
        'player_image_url': p.get('portraitUrl'),
        # A guessed date of birth (the API marks it unknown, often 1 January) is left out.
        'date_of_birth': None if life.get('isDateOfBirthUnknown') else life.get('dateOfBirth'),
        'place_of_birth': (birth.get('placeOfBirth') or '').strip() or None,
        'country_of_birth': canonical(countries.get(birth.get('countryOfBirthId'))),
        'height': round(height * 100) if height else None,
        'citizenship': canonical(countries.get(((p.get('nationalityDetails') or {}).get('nationalities') or {})
                                               .get('nationalityId'))),
        'position': position if position in POSITIONS else None,
        'main_position': MAIN_POSITIONS.get(attributes.get('positionGroup')),
        'foot': (attributes.get('preferredFoot') or {}).get('name'),
        'current_club_id': club_id,
        'current_club_name': official,
        'contract_expires': attributes.get('contractUntil'),
        'player_agent_name': (attributes.get('consultantAgency') or {}).get('name'),
        'date_of_death': life.get('dateOfDeath'),
    })
    return row


def write_player_tiers(path, why):
    """Why each player has his tier, with the lines this build used (publish_files.py makes
    tiers_v1.json from it). One player per line, by id."""
    head = {'rules': fame.player_rules(),
            'growth': fame.growth_table(entry['peak_year'] for entry in why.values() if 'peak_year' in entry),
            'pool': {'star_min_value': STAR_MIN_VALUE, 'prestigious_min_value': PRESTIGIOUS_MIN_VALUE,
                     'prestigious_club_ids': sorted(PRESTIGIOUS_CLUB_IDS), 'min_clubs': MIN_CLUBS}}
    compact = lambda value: json.dumps(value, separators=(',', ':'))
    players = ',\n'.join(f'{compact(pid)}:{compact(entry)}' for pid, entry in sorted(why.items(), key=lambda i: int(i[0])))
    path.write_text(compact(head)[:-1] + ',"players":{\n' + players + '\n}}\n')


# ── Main ──

def main():
    args = parse_args()
    dataset = Dataset(args.dataset_dir)
    max_age = 24 * 3650 if args.from_cache else args.max_age_hours
    api = TransfermarktAPI(args.cache_dir, max_age_hours=max_age, workers=args.workers)
    old = TransfermarktAPI(args.cache_dir, max_age_hours=24 * 3650, workers=args.workers)  # finished seasons

    # ── Candidates: the shared player list, the legends and the extra players ──
    print('Reading the squads of the pool clubs (sources/squads.py)...')
    spells, squad_count = squad_spells(old, api, set(pool_clubs(old, api)))
    universe = set(spells) | set(fame.legends()) | {str(i) for i in EXTRA_PLAYER_IDS}
    print(f'  {len(spells):,} players in {squad_count:,} squads; {len(universe):,} with the legends and extras')
    # Peak values only rise, so the cached profiles are enough to find who could reach the bar.
    known = old.players(universe)
    candidate_ids = sorted(int(pid) for pid in universe if fame.fame(pid, known.get(pid)) >= PRESTIGIOUS_MIN_VALUE)
    print(f'{len(candidate_ids):,} candidates with fame of €{PRESTIGIOUS_MIN_VALUE / 1e6:.0f}M or more')

    print(f'Fetching {len(candidate_ids):,} player profiles...')
    profiles = {int(pid): p for pid, p in api.players(candidate_ids).items()}
    print(f'  got {len(profiles):,}')
    candidate_ids = [pid for pid in candidate_ids if pid in profiles]
    player_fame = {pid: fame.fame(pid, profiles[pid]) for pid in candidate_ids}

    transfers, clubs, last_move, failed = fetch_transfers(api, dataset, candidate_ids)
    clubs.update(api.clubs({current_club(p) for p in profiles.values()} - {None} - {int(c) for c in clubs}))
    reserve_sides = {int(club_id): is_reserve_side(club) for club_id, club in clubs.items()}
    special_clubs = {int(club_id) for club_id, club in clubs.items() if club.get('isSpecialClub')}

    # ── Select players ──
    prestigious_ids = {int(pid) for pid, player_clubs in spells.items()
                       if any(int(club_id) in PRESTIGIOUS_CLUB_IDS for club_id in player_clubs)}
    prestigious_ids |= set(transfers.loc[transfers['from_team_id'].isin(PRESTIGIOUS_CLUB_IDS)
                                         | transfers['to_team_id'].isin(PRESTIGIOUS_CLUB_IDS), 'player_id'])
    legend_ids = {int(pid) for pid in fame.legends()}
    stars = {pid for pid in candidate_ids if player_fame[pid] >= STAR_MIN_VALUE}
    prestigious = {pid for pid in candidate_ids if pid not in stars and pid in prestigious_ids}
    selected = stars | prestigious
    print(f'\nSelected {len(selected):,} players: {len(stars):,} with fame of €{STAR_MIN_VALUE / 1e6:.0f}M or more '
          f'({len(stars & legend_ids):,} legends), {len(prestigious):,} more played for '
          f'{", ".join(PRESTIGIOUS_CLUB_IDS.values())}')
    missing_legends = legend_ids - selected
    if missing_legends:
        print(f'  ⚠ legends the API has no profile for: {sorted(missing_legends)}')

    # ── Transfers ──
    def to_reserve_side(ids, names):
        by_name = names.map(is_youth_team)
        return ids.map(lambda club_id: reserve_sides.get(club_id)).combine(
            by_name, lambda flag, name: name if flag is None or pd.isna(flag) else flag).astype(bool)

    out = transfers[transfers['player_id'].isin(selected)]
    to_reserve = to_reserve_side(out['to_team_id'], out['to_team_name'])
    print(f'  dropped {to_reserve.sum():,} transfers to youth and reserve sides')
    out = out[~to_reserve].copy()

    # First-team clubs per player: every destination, and where a move started from when
    # that is a first team too.
    def real_clubs(ids, names):
        special = ids.isin(special_clubs) | names.isin(SPECIAL_CLUB_NAMES)
        return ~special & ~to_reserve_side(ids, names) & names.notna()

    club_rows = pd.concat([
        out.loc[real_clubs(out[f'{side}_team_id'], out[f'{side}_team_name']), ['player_id', f'{side}_team_id']]
           .set_axis(['player_id', 'club_id'], axis=1)
        for side in ('from', 'to')])
    club_count = club_rows.groupby('player_id')['club_id'].nunique()
    few_clubs = {pid for pid in selected if club_count.get(pid, 0) < MIN_CLUBS}
    story_only = few_clubs & legend_ids
    left_out = few_clubs - legend_ids
    by_fame = lambda ids: sorted(ids, key=lambda pid: (-player_fame[pid], pid))
    name_of = lambda pid: latin_name(profiles[pid].get('name'))
    if left_out:
        print(f'  left out {len(left_out):,} players with fewer than {MIN_CLUBS} clubs in their history, '
              f'e.g. {", ".join(map(name_of, by_fame(left_out)[:12]))}')
    if story_only:
        print(f'  {len(story_only):,} legends with fewer than {MIN_CLUBS} clubs are Player Story only: '
              f'{", ".join(map(name_of, by_fame(story_only)))}')
    selected -= left_out
    out = out[out['player_id'].isin(selected)]

    # ── Fame, tier, still playing ──
    print(f'Fetching {len(selected):,} market value histories...')
    histories = {int(pid): h for pid, h in api.market_value_histories(sorted(selected)).items()}
    if api.failed:
        print(f'  ⚠ {len(api.failed):,} failed; their value now is the profile\'s current value only')
    today = date.today()

    # ── Profiles ──
    countries = country_names({str(pid): p for pid, p in profiles.items()})
    rows, why = [], {}
    for pid in sorted(selected):
        row = profile_row(pid, profiles[pid], countries, clubs, dataset)
        club = clubs.get(str(row['current_club_id'])) if row['current_club_id'] is not None else None
        status = fame.playing_status(club['name'] if club else None, row['date_of_death'], last_move.get(pid), today)
        active = status != 'stopped'
        now = int(fame.value_now(histories.get(pid), profiles[pid]))
        # The tier goes by fame in today's money; market_value stays the raw fame (the pool,
        # the name suggestions and the Daily Transfer's €20m Expert floor read it).
        fame_now, peak_year = fame.fame_now(pid, profiles[pid], histories.get(pid))
        # Beginner also needs a big club: his current one, or a squad there since 2024/25
        # (the same squads as the Grid), unless he is an all-time star.
        big_club = fame.beginner_club(row['current_club_id'], spells.get(str(pid)) or spells.get(pid))
        reason = fame.tier_reason(pid, fame_now, now, active, big_club)
        row['market_value'] = player_fame[pid]
        row['difficulty'] = fame.REASON_TIERS[reason]
        row['active'] = active
        row['story_only'] = True if pid in story_only else None
        rows.append(row)
        # tiers_v1.json's player entry (CONTRACTS.md §1): value now, fame now and the year of
        # the peak it comes from, the rule, the big club that allows Beginner, legend, grace.
        why[str(pid)] = {'now': now, 'fame_now': fame_now,
                         **({'peak_year': peak_year} if peak_year is not None else {}),
                         'why': reason,
                         **({'big_club': big_club} if big_club is not None else {}),
                         **({'legend': True} if pid in legend_ids else {}),
                         **({'grace': True} if status == 'grace' else {})}
    profiles_out = pd.DataFrame(rows, columns=PROFILES_COLUMNS)
    profiles_out['current_club_id'] = profiles_out['current_club_id'].astype('Int64')
    profiles_out['height'] = profiles_out['height'].astype('Int64')
    profiles_out = profiles_out.sort_values(['market_value', 'player_id'], ascending=[False, True])

    for side in ('from', 'to'):
        missing = out[f'{side}_team_country'].isna()
        out.loc[missing, f'{side}_team_country'] = out.loc[missing, f'{side}_team_id'].map(dataset.club_country)
        out[f'{side}_team_country'] = out[f'{side}_team_country'].map(canonical)
    out = out.sort_values(['player_id', 'transfer_date', 'to_team_id'], ascending=[True, False, True])

    without = selected - set(out['player_id']) - story_only
    if without:
        print(f'  ⚠ {len(without):,} selected players have no transfers and will not appear in quizzes')
    for column in ('position', 'citizenship', 'date_of_birth'):
        empty = profiles_out[column].isna().sum()
        if empty:
            print(f'  {empty:,} players without {column}')

    # Keep the last build in output/previous/ so the new one can be compared against it.
    output_files = ('player_profiles.csv', 'transfer_history.csv')
    previous_dir = args.output_dir / 'previous'
    has_previous_build = all((args.output_dir / name).exists() for name in output_files)
    if has_previous_build:
        previous_dir.mkdir(parents=True, exist_ok=True)
        for name in output_files:
            (args.output_dir / name).replace(previous_dir / name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    profiles_out[PROFILES_COLUMNS].to_csv(args.output_dir / 'player_profiles.csv', index=False)
    out[TRANSFERS_COLUMNS].to_csv(args.output_dir / 'transfer_history.csv', index=False)
    write_player_tiers(args.output_dir / 'player_tiers.json', why)
    tiers = profiles_out.groupby('difficulty').size().reindex(fame.TIERS, fill_value=0)
    path_tiers = profiles_out[profiles_out['story_only'] != True].groupby('difficulty').size().reindex(fame.TIERS, fill_value=0)  # noqa: E712
    print(f'\n✓ {len(profiles_out):,} players  → {args.output_dir / "player_profiles.csv"}')
    print(f'  tiers: ' + ', '.join(f'{t} {n:,}' for t, n in tiers.items())
          + f'; Transfer Path (without the {len(story_only):,} story-only): '
          + ', '.join(f'{t} {n:,}' for t, n in path_tiers.items()))
    print(f'  {int(profiles_out["active"].sum()):,} still playing; '
          f'{len(selected & legend_ids):,} legends ({len(story_only):,} story only)')
    print(f'✓ {len(out):,} transfers → {args.output_dir / "transfer_history.csv"}')
    if args.output_dir == MODE_DIR / 'output':
        record(MODE_DIR, 'built', players=len(profiles_out), transfers=len(out),
               latest_transfer=out['transfer_date'].max(),
               beginner=int(tiers['beginner']), intermediate=int(tiers['intermediate']),
               expert=int(tiers['expert']), story_only=len(story_only),
               source='Transfermarkt API' + (f' ({len(failed):,} players from the dataset tables)' if failed else ''))
    country_share = (out['to_team_country'].notna().mean() * 100) if len(out) else 0
    print(f'  {country_share:.1f}% of transfers have a destination country')

    if all((previous_dir / name).exists() for name in output_files):
        print()
        warnings = compare_outputs.report(previous_dir, args.output_dir,
                                          save_to=args.output_dir / 'compare_report.txt')
        if warnings:
            sys.exit('\nThe new build looks wrong (see above). Fix it before publishing.')


if __name__ == '__main__':
    main()
