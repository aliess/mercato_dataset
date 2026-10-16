"""
Name, nationality, position and date of birth of any player: the name so every game file
spells a player the same way (the app builds one suggestion list from all of them), the rest
for the hints (CONTRACTS.md §3.2).

Each fact is taken from the first source that has it:

1. the game data (game_modes_data/transfer_history/output/player_profiles.csv), so a player
   who is also in Transfer Path shows the same citizenship and position there and here;
2. the player's Transfermarkt API profile, which the Grid and Starting XI builds already
   fetch (current, covers everyone);
3. the dataset table (sources/dataset/players.csv.gz), when the API leaves a field empty.

Names come from the game data only, then the API: a player in Transfer Path must be spelled
the same in Grid and Starting XI. (The game data's names are the API's since 2026-10-16, so
both give the same spelling; the game data comes first so a later name fix there wins.) The
dataset table is not used for names: it is frozen
and staler than the API ("Noah Korczowski" for "Noah van Lent"). Every name goes through
latin_name().

Positions use the game data's 13 names ("Centre-Back", "Right Winger", ...). The API also
has "Sweeper", filed here as "Centre-Back" (Starting XI does the same).
Country names come from country_names(): the API only gives ids.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from sources.country_names import canonical
from sources.names import latin_name

ROOT = Path(__file__).resolve().parent.parent
GAME_PROFILES = ROOT / 'game_modes_data' / 'transfer_history' / 'output' / 'player_profiles.csv'
DATASET_PLAYERS = ROOT / 'sources' / 'dataset' / 'players.csv.gz'
DATASET_COUNTRIES = ROOT / 'sources' / 'dataset' / 'countries.csv.gz'
COUNTRY_IDS = ROOT / 'sources' / 'country_ids.json'  # written by game_modes_data/grid/build_pool.py

POSITIONS = {'Goalkeeper', 'Centre-Back', 'Left-Back', 'Right-Back', 'Defensive Midfield',
             'Central Midfield', 'Left Midfield', 'Right Midfield', 'Attacking Midfield',
             'Left Winger', 'Right Winger', 'Second Striker', 'Centre-Forward'}
POSITION_NAMES = {'Sweeper': 'Centre-Back'}


def nation_id(profile):
    """The API profile's first (main) nationality id, or None."""
    return (((profile or {}).get('nationalityDetails') or {}).get('nationalities') or {}).get('nationalityId')


def country_names(api_players):
    """Transfermarkt country id → country name, spelled as the game data spells citizenship.

    The API has no country names. They are learned from players the dataset names: the
    citizenship most players with that nationality id have. Ids none of `api_players` can
    name come from COUNTRY_IDS (what the last Grid build learned from its ~45,000 players;
    the Starting XI's 2,500 players alone leave Liberia or Chad unnamed), then from the
    dataset's countries table (124 countries).
    """
    countries = pd.read_csv(DATASET_COUNTRIES).set_index('country_id')['country_name'].to_dict()
    if COUNTRY_IDS.exists():
        countries.update({int(k): v for k, v in json.loads(COUNTRY_IDS.read_text()).items()})
    citizenship = pd.read_csv(DATASET_PLAYERS, usecols=['player_id', 'country_of_citizenship']).dropna()
    votes = defaultdict(Counter)
    for row in citizenship.itertuples():
        found = nation_id(api_players.get(str(row.player_id)))
        if found:
            votes[found][row.country_of_citizenship] += 1
    countries.update({country_id: names.most_common(1)[0][0] for country_id, names in votes.items()})
    return {country_id: canonical(name) for country_id, name in countries.items()}


def save_country_names(countries):
    """Write the id → name map for the builds with fewer players (see country_names)."""
    COUNTRY_IDS.write_text(json.dumps({str(k): v for k, v in sorted(countries.items())},
                                      ensure_ascii=False, indent=0) + '\n')


def _position(name):
    name = POSITION_NAMES.get(name, name)
    return name if name in POSITIONS else None


def _day(value):
    """'1987-06-24 00:00:00' or '1987-06-24' → '1987-06-24'; anything else → None."""
    if isinstance(value, str) and len(value) >= 10 and value[4] == '-' and value[7] == '-':
        return value[:10]
    return None


class PlayerFacts:
    def __init__(self, api_players, countries=None):
        """`api_players`: {player id (str): API profile}, as TransfermarktAPI.players returns."""
        self.api = api_players
        self.countries = countries if countries is not None else country_names(api_players)
        game = pd.read_csv(GAME_PROFILES, usecols=['player_id', 'player_name', 'citizenship', 'position', 'date_of_birth'])
        self.game = {str(r.player_id): r for r in game.itertuples()}
        dataset = pd.read_csv(DATASET_PLAYERS, usecols=['player_id', 'sub_position', 'date_of_birth'])
        self.dataset = {str(r.player_id): r for r in dataset.itertuples()}

    def name(self, player_id):
        """The player's name as game_data_v1.json spells it, else the API's; Latin letters."""
        player_id = str(player_id)
        game = self.game.get(player_id)
        for name in (game.player_name if game is not None else None,
                     (self.api.get(player_id) or {}).get('name')):
            if isinstance(name, str) and name.strip():
                return latin_name(name)
        return None

    def nation(self, player_id):
        game = self.game.get(str(player_id))
        if game is not None and isinstance(game.citizenship, str):
            return canonical(game.citizenship)
        return self.countries.get(nation_id(self.api.get(str(player_id))))

    def position(self, player_id):
        player_id = str(player_id)
        game = self.game.get(player_id)
        api = ((self.api.get(player_id) or {}).get('attributes') or {}).get('position') or {}
        dataset = self.dataset.get(player_id)
        for name in (game.position if game is not None else None, api.get('name'),
                     dataset.sub_position if dataset is not None else None):
            if isinstance(name, str) and _position(name):
                return _position(name)
        return None

    def born(self, player_id):
        player_id = str(player_id)
        game = self.game.get(player_id)
        life = (self.api.get(player_id) or {}).get('lifeDates') or {}
        dataset = self.dataset.get(player_id)
        # The API marks a guessed date (often 1 January) as unknown; such a date is left out.
        api_day = None if life.get('isDateOfBirthUnknown') else _day(life.get('dateOfBirth'))
        for day in (_day(game.date_of_birth) if game is not None else None, api_day,
                    _day(dataset.date_of_birth) if dataset is not None else None):
            if day:
                return day
        return None
