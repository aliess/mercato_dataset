#!/usr/bin/env python3
"""
Build the two files the app's Firestore is filled from:

  output/player_profiles.csv    → collection `player_profiles_and_value` (doc id = player_id)
  output/transfer_history.csv   → collection `transfer_history_filtered`

Players are picked from the transfermarkt-datasets tables (dataset/). Unless --offline is
given, every candidate is then refreshed from Transfermarkt's JSON API (see tm_api.py):
highest market value, current club, and the full transfer history. That keeps the data
current even though the upstream dataset stopped updating in July 2026, and fills in
players whose transfers the dataset never had (e.g. Hazard, Bale, Aguero).

Usage:
    python scripts/build_dataset.py              # dataset + live Transfermarkt refresh
    python scripts/build_dataset.py --offline    # dataset only (old behaviour)
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from unidecode import unidecode

import compare_outputs
from tm_api import TransfermarktAPI

ROOT = Path(__file__).resolve().parent.parent

# ── Player selection ──
# Stars: highest-ever market value above this.
STAR_MIN_VALUE = 20_000_000
# Lesser players are kept if they played for one of these clubs and peaked at or above this.
PRESTIGIOUS_MIN_VALUE = 10_000_000
PRESTIGIOUS_CLUB_IDS = {
    11: 'Arsenal',
    5: 'AC Milan',
    418: 'Real Madrid',
    131: 'Barcelona',
    631: 'Chelsea',
}

# Transfers TO teams whose name ends like this are youth/reserve moves and are dropped.
YOUTH_SUFFIXES = ('YTH', 'Youth', 'You', 'U19', 'U17', 'Yth', 'U20', 'U21', 'U18',
                  'U16', 'U23', 'U22', 'U24', 'II', 'Yth.', 'B')

# ── Output columns (must match what the app / Firestore import expect) ──
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
]
TRANSFERS_COLUMNS = [
    'player_id', 'season_name', 'transfer_date', 'from_team_id', 'from_team_name',
    'to_team_id', 'to_team_name', 'transfer_type', 'value_at_transfer',
    'transfer_fee', 'from_team_country', 'to_team_country',
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset-dir', type=Path, default=ROOT / 'dataset')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'output')
    parser.add_argument('--cache-dir', type=Path, default=ROOT / 'cache' / 'tm_api')
    parser.add_argument('--offline', action='store_true',
                        help='Use only the dataset tables; skip the Transfermarkt API.')
    parser.add_argument('--max-age-hours', type=float, default=24,
                        help='Reuse cached API responses younger than this (default 24).')
    parser.add_argument('--workers', type=int, default=4, help='Parallel API requests (default 4).')
    return parser.parse_args()


def read_table(dataset_dir, name, **kwargs):
    """Read dataset/<name>.csv.gz (from download_dataset.py), else dataset/<name>.csv (Kaggle zip)."""
    for path in (dataset_dir / f'{name}.csv.gz', dataset_dir / f'{name}.csv'):
        if path.exists():
            return pd.read_csv(path, **kwargs)
    sys.exit(f'Missing {name}.csv in {dataset_dir}. Run scripts/download_dataset.py first.')


def is_youth_team(name):
    return isinstance(name, str) and name.strip().endswith(YOUTH_SUFFIXES)


def ascii_name(name):
    return unidecode(name) if isinstance(name, str) else name


# ── Dataset tables ──

class Dataset:
    def __init__(self, dataset_dir):
        print(f'Loading dataset from {dataset_dir}')
        self.players = read_table(dataset_dir, 'players', low_memory=False)
        self.valuations = read_table(
            dataset_dir, 'player_valuations',
            usecols=['player_id', 'date', 'market_value_in_eur', 'current_club_id'])
        self.clubs = read_table(dataset_dir, 'clubs', usecols=['club_id', 'name', 'domestic_competition_id'])
        self.competitions = read_table(dataset_dir, 'competitions')
        self.countries = read_table(dataset_dir, 'countries', usecols=['country_id', 'country_name'])
        self.transfers = read_table(dataset_dir, 'transfers')
        print(f'  {len(self.players):,} players, {len(self.valuations):,} valuations '
              f'(latest {self.valuations["date"].max()}), {len(self.transfers):,} transfers')

        # club_id → country, via the club's domestic league (same as before).
        league_country = self.competitions.set_index('competition_id')['country_name']
        self.club_country = (self.clubs.set_index('club_id')['domestic_competition_id']
                             .map(league_country).dropna().to_dict())
        self.country_by_id = self.countries.set_index('country_id')['country_name'].to_dict()
        self.club_official_name = self.clubs.set_index('club_id')['name'].to_dict()

    def highest_values(self):
        """Highest-ever market value per player, from valuations and the players table."""
        from_valuations = self.valuations.groupby('player_id')['market_value_in_eur'].max()
        from_players = self.players.set_index('player_id')['highest_market_value_in_eur']
        return pd.concat([from_valuations, from_players], axis=1).max(axis=1).dropna()

    def transfers_in_output_schema(self):
        t = self.transfers
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
        })

    def profiles_in_output_schema(self, player_ids):
        p = self.players[self.players['player_id'].isin(player_ids)]
        df = pd.DataFrame({column: None for column in PROFILES_COLUMNS}, index=p.index)
        df['player_id'] = p['player_id']
        df['player_slug'] = p['player_code']
        df['player_name'] = p['name'].map(ascii_name)
        df['player_image_url'] = p['image_url']
        df['date_of_birth'] = p['date_of_birth']
        df['place_of_birth'] = p['city_of_birth']
        df['country_of_birth'] = p['country_of_birth']
        df['height'] = p['height_in_cm']
        df['citizenship'] = p['country_of_citizenship']
        df['position'] = p['sub_position']
        df['main_position'] = p['position']
        df['foot'] = p['foot']
        df['current_club_id'] = p['current_club_id']
        df['current_club_name'] = p['current_club_name']
        df['contract_expires'] = p['contract_expiration_date']
        df['player_agent_name'] = p['agent_name']
        return df.reset_index(drop=True)


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


def fetch_live_data(api, dataset, candidate_ids):
    """Refresh candidates from the Transfermarkt API.

    Returns (profiles by player id, transfers DataFrame in output schema).
    """
    print(f'\nFetching {len(candidate_ids):,} player profiles from Transfermarkt...')
    profiles = api.players(candidate_ids)
    print(f'  got {len(profiles):,}')

    print(f'Fetching transfer histories...')
    histories = api.transfer_histories(
        candidate_ids, progress=lambda done, total: print(f'  {done:,}/{total:,}'))
    print(f'  got {len(histories):,} histories ({api.requests:,} API requests, {api.cache_hits:,} cached)')
    if api.failed:
        print(f'  ⚠ {len(api.failed):,} players failed and keep their dataset transfers, e.g. {api.failed[0]}')

    club_ids = {str(a['clubId']) for p in profiles.values() for a in p.get('clubAssignments') or []
                if a.get('type') == 'current'}
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
    rows = []
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
            source, destination = t['transferSource'], t['transferDestination']
            to_name = short_name(destination['clubId'])
            if not to_name:
                skipped_unknown += 1
                continue
            fee = (details.get('fee') or {})
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

    live_profiles = {}
    for player_id, p in profiles.items():
        current = next((a for a in p.get('clubAssignments') or [] if a.get('type') == 'current'), None)
        club_id = int(current['clubId']) if current else None
        club = clubs.get(str(club_id)) if club_id is not None else None
        official = dataset.club_official_name.get(club_id) if club_id is not None else None
        if not official and club:
            # First teams carry their legal name ("Derby County Football Club") like the dataset;
            # Retired / Without Club / youth sides keep their own name (the parent's is localized).
            is_first_team = club['baseDetails'].get('clubTypeId') == 1 and not club.get('isSpecialClub')
            superior = (club['baseDetails'].get('superiorClub') or {}).get('name')
            official = superior if is_first_team and superior else club['name']
        highest = ((p.get('marketValueDetails') or {}).get('highest') or {}).get('value')
        live_profiles[int(player_id)] = {
            'current_club_id': club_id,
            'current_club_name': official,
            'contract_expires': (p.get('attributes') or {}).get('contractUntil'),
            'date_of_death': (p.get('lifeDates') or {}).get('dateOfDeath'),
            'highest_value': highest,
        }
    return live_profiles, pd.DataFrame(rows, columns=TRANSFERS_COLUMNS)


# ── Main ──

def main():
    args = parse_args()
    dataset = Dataset(args.dataset_dir)

    highest = dataset.highest_values()
    known_ids = set(dataset.players['player_id'])
    candidate_ids = sorted(pid for pid, value in highest.items()
                           if value >= PRESTIGIOUS_MIN_VALUE and pid in known_ids)
    print(f'\n{len(candidate_ids):,} candidates peaked at €{PRESTIGIOUS_MIN_VALUE / 1e6:.0f}M or more')

    transfers = dataset.transfers_in_output_schema()
    live_profiles = {}
    if not args.offline:
        api = TransfermarktAPI(args.cache_dir, max_age_hours=args.max_age_hours, workers=args.workers)
        live_profiles, live_transfers = fetch_live_data(api, dataset, candidate_ids)
        # The API history replaces the dataset's for every player it returned.
        live_ids = set(live_transfers['player_id'])
        transfers = pd.concat([transfers[~transfers['player_id'].isin(live_ids)], live_transfers],
                              ignore_index=True)
        for player_id, live in live_profiles.items():
            if live['highest_value']:
                highest[player_id] = max(highest.get(player_id, 0), live['highest_value'])

    # ── Select players ──
    clubs_played_for = pd.concat([
        transfers[['player_id', 'from_team_id']].set_axis(['player_id', 'club_id'], axis=1),
        transfers[['player_id', 'to_team_id']].set_axis(['player_id', 'club_id'], axis=1),
        dataset.valuations[['player_id', 'current_club_id']].set_axis(['player_id', 'club_id'], axis=1),
    ])
    prestigious_ids = set(clubs_played_for[clubs_played_for['club_id'].isin(PRESTIGIOUS_CLUB_IDS)]['player_id'])

    stars = {pid for pid in candidate_ids if highest[pid] > STAR_MIN_VALUE}
    prestigious = {pid for pid in candidate_ids
                   if pid not in stars and highest[pid] >= PRESTIGIOUS_MIN_VALUE and pid in prestigious_ids}
    selected = stars | prestigious
    print(f'\nSelected {len(selected):,} players: {len(stars):,} peaked above €{STAR_MIN_VALUE / 1e6:.0f}M, '
          f'{len(prestigious):,} more played for {", ".join(PRESTIGIOUS_CLUB_IDS.values())}')

    # ── Profiles ──
    profiles = dataset.profiles_in_output_schema(selected)
    profiles['market_value'] = profiles['player_id'].map(highest).astype('int64')
    for column in ('current_club_id', 'current_club_name', 'contract_expires', 'date_of_death'):
        live = profiles['player_id'].map(lambda pid: live_profiles.get(pid, {}).get(column))
        profiles[column] = live.where(live.notna(), profiles[column])
    profiles['current_club_id'] = profiles['current_club_id'].astype('Int64')
    profiles = profiles.sort_values(['market_value', 'player_id'], ascending=[False, True])

    # ── Transfers ──
    out = transfers[transfers['player_id'].isin(selected) & ~transfers['to_team_name'].map(is_youth_team)].copy()
    for side in ('from', 'to'):
        missing = out[f'{side}_team_country'].isna()
        out.loc[missing, f'{side}_team_country'] = out.loc[missing, f'{side}_team_id'].map(dataset.club_country)
    out = out.sort_values(['player_id', 'transfer_date', 'to_team_id'], ascending=[True, False, True])

    without = selected - set(out['player_id'])
    if without:
        print(f'  ⚠ {len(without):,} selected players have no transfers and will not appear in quizzes')

    # Keep the last build in output/previous/ so the new one can be compared against it.
    output_files = ('player_profiles.csv', 'transfer_history.csv')
    previous_dir = args.output_dir / 'previous'
    has_previous_build = all((args.output_dir / name).exists() for name in output_files)
    if has_previous_build:
        previous_dir.mkdir(parents=True, exist_ok=True)
        for name in output_files:
            (args.output_dir / name).replace(previous_dir / name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    profiles[PROFILES_COLUMNS].to_csv(args.output_dir / 'player_profiles.csv', index=False)
    out[TRANSFERS_COLUMNS].to_csv(args.output_dir / 'transfer_history.csv', index=False)
    print(f'\n✓ {len(profiles):,} players  → {args.output_dir / "player_profiles.csv"}')
    print(f'✓ {len(out):,} transfers → {args.output_dir / "transfer_history.csv"}')
    country_share = (out['to_team_country'].notna().mean() * 100) if len(out) else 0
    print(f'  {country_share:.1f}% of transfers have a destination country')

    if all((previous_dir / name).exists() for name in output_files):
        print()
        warnings = compare_outputs.report(previous_dir, args.output_dir,
                                          save_to=args.output_dir / 'compare_report.txt')
        if warnings:
            sys.exit('\nThe new build looks wrong (see above). Fix it before syncing to Firestore.')


if __name__ == '__main__':
    main()
