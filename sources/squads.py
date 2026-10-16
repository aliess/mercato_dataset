"""
The one list of players every mode starts from: everyone who was in the first-team squad of a
pool club in any season since FIRST_SEASON, whether or not he played a match.

Pool clubs: every club that played in one of the top five leagues in any of those seasons, plus
OTHER_CLUB_IDS. Grid Rush answers come from these squads (game_modes_data/grid/build_pool.py),
and the Transfer Path pool is picked from the same players plus the legends
(game_modes_data/transfer_history/build_dataset.py).

Finished seasons never change, so they are read from the cache however old it is (`old`);
the last two seasons come from `live`.
"""

from collections import Counter, defaultdict
from datetime import date

FIRST_SEASON = 1990  # 1990/91; the API has squads back to at least 1975
TOP5_LEAGUES = {'GB1': 'Premier League', 'ES1': 'LaLiga', 'IT1': 'Serie A', 'L1': 'Bundesliga', 'FR1': 'Ligue 1'}
# The 20 clubs outside those leagues that the most quiz-pool players passed through.
OTHER_CLUB_IDS = {
    294: 'Benfica', 610: 'Ajax', 720: 'Porto', 336: 'Sporting CP', 36: 'Fenerbahce', 383: 'PSV',
    141: 'Galatasaray', 114: 'Besiktas', 58: 'Anderlecht', 409: 'Red Bull Salzburg',
    234: 'Feyenoord', 683: 'Olympiacos', 614: 'Flamengo', 209: 'River Plate', 2282: 'Club Brugge',
    449: 'Trabzonspor', 585: 'Sao Paulo', 1075: 'Braga', 189: 'Boca Juniors', 1023: 'Palmeiras',
}


def this_season(today=None):
    """The season running now (2026 = 2026/27 from July on)."""
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def seasons():
    return range(FIRST_SEASON, this_season() + 1)


def pool_clubs(old, live):
    """{club id (str): Counter(top-five league id → seasons in it)}; OTHER_CLUB_IDS have an
    empty Counter."""
    current = this_season()
    club_league = {}
    for league in TOP5_LEAGUES:
        for season in seasons():
            fixtures = (live if season >= current - 1 else old).competition_fixtures(league, season)
            for club_id in (fixtures or {}).get('clubIds') or []:
                club_league.setdefault(str(club_id), Counter())[league] += 1
    for club_id in OTHER_CLUB_IDS:
        club_league.setdefault(str(club_id), Counter())
    return club_league


def squad_spells(old, live, club_ids, progress=None):
    """{player id: {club id: [first season, last season]}} from every squad of `club_ids`."""
    current = this_season()
    past = [(c, s) for c in sorted(club_ids, key=int) for s in seasons() if s < current - 1]
    recent = [(c, s) for c in sorted(club_ids, key=int) for s in seasons() if s >= current - 1]
    squads = {**old.squads(past, progress), **live.squads(recent)}
    spells = defaultdict(dict)
    for (club_id, season), player_ids in squads.items():
        for player_id in player_ids:
            span = spells[player_id].setdefault(club_id, [season, season])
            span[0], span[1] = min(span[0], season), max(span[1], season)
    return spells, len(squads)
