"""
How famous a player is, and the difficulty tiers that follow from it, for every mode
(plan §8 item 115, CONTRACTS.md §5 "Tiers"). Every build takes these from here.

- fame(player) = the higher of his Transfermarkt peak value and his legend value. Transfermarkt
  has valued players only since late 2004, so legends.json gives the well-known older ones an
  approximate present-day value (see legend_values). This raw fame picks who is in the game
  data at all (build_dataset.py), is the game data's market_value and the Grid pool's value.
- fame_now(player) = the same in today's money: peak value × growth(year of the peak), the
  higher of that and the legend value (legend values are present-day amounts already, never
  adjusted). It sets every tier: Transfer Path, Player Story and the Starting XI stars.
- value_now(player) = his highest Transfermarkt value since VALUE_NOW_FROM.
- still_playing(...) = not retired, alive, and when without a club, moved in the last year.
- Transfer Path and Player Story: player_tier(...), and tier_reason() for which rule set it
  (the `why` of tiers_v1.json; player_tier is read off it, so the two never disagree). Beginner
  also needs a big club (beginner_club(): current club or a recent squad there) or an
  all-time star; otherwise the player is Intermediate.
- Starting XI: a starter is a star when his fame now is STAR_VALUE or more (no match-year
  factor on top: his value is already in today's money); lineup_tier(year, stars).
- rules(): these constants as the admin panel sees them (tiers_v1.json, CONTRACTS.md §1).

The app and the functions read the tier from the files (`difficulty` in game_data_v1.json and
xi_v1.json) and hold no value lines of their own.
"""

import json
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd

SOURCES = Path(__file__).resolve().parent
LEGENDS_FILE = SOURCES / 'legends' / 'legends.json'
VALUATIONS_FILE = SOURCES / 'dataset' / 'player_valuations.csv.gz'

TIERS = ('beginner', 'intermediate', 'expert')

# ── Transfer Path and Player Story ──
VALUE_NOW_FROM = '2023-07-01'
BEGINNER_VALUE_NOW = 60_000_000         # still playing and worth €60m+ since VALUE_NOW_FROM,
BEGINNER_FAME = 80_000_000              # or fame €80m+ ...
BEGINNER_FAME_VALUE_NOW = 15_000_000    # ... and still worth €15m+ since VALUE_NOW_FROM
INTERMEDIATE_FAME = 40_000_000          # otherwise: intermediate from €40m fame, expert below
# "fame" in these lines is fame_now(): the peak in today's money (growth of its year).
# Beginner also needs one of: his current club is one of these, or he was in one of their squads
# (sources/squads.py, the Grid's spells) in BEGINNER_CLUB_LAST_SEASON or later, or he is one of
# ALL_TIME_STAR_IDS. A player who meets the value lines without either is intermediate.
# Real Madrid, Barcelona, Atlético, Man City, Liverpool, Arsenal, Chelsea, Man Utd, Tottenham, PSG, Bayern
BEGINNER_CLUB_IDS = (418, 131, 13, 281, 31, 11, 631, 985, 148, 583, 27)
BEGINNER_CLUB_LAST_SEASON = 2024        # a squad there in 2024/25 or later counts
# Messi, Cristiano Ronaldo, Neymar, Benzema
ALL_TIME_STAR_IDS = (28003, 8198, 68290, 18922)

# ── Still playing ──
RETIRED = 'Retired'
NO_CLUB = {'Without Club', 'Career break', 'Unknown'}  # still playing if he moved in the last year
NO_CLUB_GRACE = timedelta(days=365)

# ── Starting XI ──
STAR_VALUE = 40_000_000       # a star: €40m+ fame now (fame_now, today's money)
GROWTH_TOP = 300              # the index is the mean value of the year's 300 most valuable players
GROWTH_CAP = 2.5              # no year is scaled up by more than this
GROWTH_MIN_PLAYERS = 3_000    # a year with fewer valued players (before 2007) takes the cap
BEGINNER_LINEUP_FROM = 2017   # beginner: match year 2017 on ...
BEGINNER_LINEUP_STARS = 10    # ... with 10+ stars
INTERMEDIATE_LINEUP_STARS = 5  # intermediate: 5+ stars, any year; expert: the rest


# ── Fame ──

def legend_values(legends):
    """Each legend's value. A tier in legends.json is a ceiling: its legends are spread
    evenly from just under it down to the next tier, the one with the most Wikipedia
    languages highest. Each value is a round amount less one euro, so it is never equal to
    a real Transfermarkt value: a legend tied with a current player is the rarer one."""
    tiers = sorted({legend['value'] for legend in legends}, reverse=True)
    values = {}
    for i, tier in enumerate(tiers):
        floor = tiers[i + 1] if i + 1 < len(tiers) else tier - (tiers[i - 1] - tier)
        members = sorted((legend for legend in legends if legend['value'] == tier),
                         key=lambda legend: (-legend['wikipedia_languages'], int(legend['id'])))
        for rank, legend in enumerate(members):
            values[legend['id']] = round((tier - rank * (tier - floor) / len(members)) / 10_000) * 10_000 - 1
    return values


@lru_cache(maxsize=1)
def legends():
    """{player id (str): legend value} for every player in legends.json."""
    return legend_values(json.loads(LEGENDS_FILE.read_text()))


def is_legend(player_id):
    return str(player_id) in legends()


def peak_value(profile):
    """The highest value Transfermarkt ever gave the player (API profile), or None."""
    return (((profile or {}).get('marketValueDetails') or {}).get('highest') or {}).get('value')


def fame(player_id, profile):
    """max(Transfermarkt peak value, legend value); 0 when he has neither. The raw fame: who is
    in the game data, market_value, the Grid pool. Tiers use fame_now()."""
    return max(peak_value(profile) or 0, legends().get(str(player_id), 0))


def peak_year(player_id, profile, history=None):
    """The year the player reached his peak value: the date Transfermarkt gives with it
    (API profile), else the date of the highest point of his value history, else of his
    highest dataset valuation; None when nothing has a date."""
    determined = (((profile or {}).get('marketValueDetails') or {}).get('highest') or {}).get('determined')
    if determined:
        return int(determined[:4])
    if history:
        return int(max(history, key=lambda p: (p['value'], p['date']))['date'][:4])
    return dataset_peak_years().get(int(player_id))


def fame_now(player_id, profile, history=None):
    """(fame in today's money, peak year or None): max(peak value × growth(peak year), legend
    value). A peak with no date at all is taken as it is."""
    peak = peak_value(profile) or 0
    year = peak_year(player_id, profile, history) if peak else None
    adjusted = peak * growth(year) if year is not None else peak
    return int(round(max(adjusted, legends().get(str(player_id), 0)))), year


def value_now(history, profile=None):
    """The highest value since VALUE_NOW_FROM, from the market value history
    (TransfermarktAPI.market_value_histories) and the profile's current value; 0 when none."""
    points = [p['value'] for p in history or [] if p['date'] >= VALUE_NOW_FROM]
    current = ((profile or {}).get('marketValueDetails') or {}).get('current') or {}
    if current.get('value') and (current.get('determined') or '') >= VALUE_NOW_FROM:
        points.append(current['value'])
    return max(points, default=0)


def playing_status(current_club, date_of_death, last_move, today=None):
    """'playing', 'grace' (without a club or on a career break, but his last move, an ISO
    date, is less than NO_CLUB_GRACE before `today`) or 'stopped' (retired, dead, or without a
    club for longer)."""
    if date_of_death or current_club == RETIRED:
        return 'stopped'
    if current_club in NO_CLUB:
        today = today or date.today()
        recent = bool(last_move) and date.fromisoformat(last_move[:10]) >= today - NO_CLUB_GRACE
        return 'grace' if recent else 'stopped'
    return 'playing'


def still_playing(current_club, date_of_death, last_move, today=None):
    """Not retired and alive. A player without a club (or on a career break) counts only when
    his last move (ISO date) is less than a year before `today`."""
    return playing_status(current_club, date_of_death, last_move, today) != 'stopped'


# The rule that set a player's tier → the tier.
REASON_TIERS = {
    'beginner_value_now': 'beginner',   # still playing, value now ≥ BEGINNER_VALUE_NOW, a big club
    'beginner_fame': 'beginner',        # still playing, fame ≥ BEGINNER_FAME, value now ≥ BEGINNER_FAME_VALUE_NOW, a big club
    'beginner_star': 'beginner',        # one of those value lines, no big club, one of ALL_TIME_STAR_IDS
    'intermediate_club': 'intermediate',  # one of those value lines, but no big club and not an all-time star
    'intermediate_fame': 'intermediate',  # fame ≥ INTERMEDIATE_FAME
    'expert': 'expert',                 # the rest
}


def beginner_club(current_club_id, spells):
    """The big club that lets the player be Beginner, or None: his current club when it is one
    of BEGINNER_CLUB_IDS, else the one he was last in the squad of, when that squad is of
    BEGINNER_CLUB_LAST_SEASON or later. `spells` = {club id: [first season, last season]}, his
    entry of sources/squads.squad_spells (the Grid's squads)."""
    if current_club_id is not None and int(current_club_id) in BEGINNER_CLUB_IDS:
        return int(current_club_id)
    recent = [(span[1], int(club_id)) for club_id, span in (spells or {}).items()
              if int(club_id) in BEGINNER_CLUB_IDS and span[1] >= BEGINNER_CLUB_LAST_SEASON]
    return max(recent)[1] if recent else None


def tier_reason(player_id, player_fame, player_value_now, active, big_club):
    """Which rule sets the Transfer Path and Player Story tier: a key of REASON_TIERS.
    `big_club` is beginner_club()'s answer."""
    value_line = player_value_now >= BEGINNER_VALUE_NOW
    fame_line = player_fame >= BEGINNER_FAME and player_value_now >= BEGINNER_FAME_VALUE_NOW
    if active and (value_line or fame_line):
        if big_club is not None:
            return 'beginner_value_now' if value_line else 'beginner_fame'
        return 'beginner_star' if int(player_id) in ALL_TIME_STAR_IDS else 'intermediate_club'
    return 'intermediate_fame' if player_fame >= INTERMEDIATE_FAME else 'expert'


def player_tier(player_id, player_fame, player_value_now, active, big_club):
    """Transfer Path and Player Story tier."""
    return REASON_TIERS[tier_reason(player_id, player_fame, player_value_now, active, big_club)]


# ── Today's money (Starting XI) ──

@lru_cache(maxsize=1)
def _valuations():
    valuations = pd.read_csv(VALUATIONS_FILE, usecols=['player_id', 'date', 'market_value_in_eur'])
    valuations['year'] = valuations['date'].str[:4].astype(int)
    return valuations


@lru_cache(maxsize=1)
def dataset_peak_years():
    """{player id: year of his highest dataset valuation} (peak_year's last fallback)."""
    v = _valuations().sort_values(['market_value_in_eur', 'date'])
    return v.groupby('player_id')['year'].last().to_dict()


@lru_cache(maxsize=1)
def growth_index():
    """({year: mean of the year's GROWTH_TOP highest player values}, latest full year).

    Each player counts once a year, at his highest value that year. Values come from the
    transfermarkt-datasets valuations (every valuation since 2004, frozen in June 2026); the
    latest full year is the last one with valuations in December.
    """
    valuations = _valuations()
    per_player = valuations.groupby(['year', 'player_id'])['market_value_in_eur'].max()
    index = {}
    for year, values in per_player.groupby(level='year'):
        if len(values) >= GROWTH_MIN_PLAYERS:
            index[int(year)] = float(values.nlargest(GROWTH_TOP).mean())
    december = valuations.loc[valuations['date'].str[5:7] == '12', 'year']
    latest_full = int(december.max())
    return index, latest_full


def growth(year):
    """How much a value of `year` is worth in today's money: index(latest full year) /
    index(year), at most GROWTH_CAP; 1 from the latest full year on; the cap for years with
    too few valuations."""
    index, latest_full = growth_index()
    if year >= latest_full:
        return 1.0
    if year not in index:
        return GROWTH_CAP
    return min(GROWTH_CAP, index[latest_full] / index[year])


def is_star(player_id, profile):
    """A Starting XI star: fame now (today's money, from the peak's year) ≥ STAR_VALUE. The
    match year plays no part: the value is already in today's money."""
    return fame_now(player_id, profile)[0] >= STAR_VALUE


def lineup_tier(year, stars):
    """Starting XI tier from the match year and the number of stars among the eleven."""
    if year >= BEGINNER_LINEUP_FROM and stars >= BEGINNER_LINEUP_STARS:
        return 'beginner'
    return 'intermediate' if stars >= INTERMEDIATE_LINEUP_STARS else 'expert'


def growth_table(years):
    """{latest_full_year, by_year: {"<year>": growth(year)}} for these years (tiers_v1.json)."""
    return {'latest_full_year': growth_index()[1],
            'by_year': {str(year): growth(year) for year in sorted(set(years))}}


# ── The lines, as tiers_v1.json shows them ──

def player_rules():
    """The constants player_tier(), beginner_club() and still_playing() use (amounts in euros)."""
    return {
        'value_now_from': VALUE_NOW_FROM,
        'beginner_value_now': BEGINNER_VALUE_NOW,
        'beginner_fame': BEGINNER_FAME,
        'beginner_fame_value_now': BEGINNER_FAME_VALUE_NOW,
        'intermediate_fame': INTERMEDIATE_FAME,
        'beginner_club_ids': list(BEGINNER_CLUB_IDS),
        'beginner_club_last_season': BEGINNER_CLUB_LAST_SEASON,
        'all_time_star_ids': list(ALL_TIME_STAR_IDS),
        'no_club_grace_days': NO_CLUB_GRACE.days,
    }


def lineup_rules():
    """The constants is_star(), growth() and lineup_tier() use."""
    return {
        'star_value': STAR_VALUE,
        'growth_top': GROWTH_TOP,
        'growth_cap': GROWTH_CAP,
        'growth_min_players': GROWTH_MIN_PLAYERS,
        'beginner_lineup_from': BEGINNER_LINEUP_FROM,
        'beginner_lineup_stars': BEGINNER_LINEUP_STARS,
        'intermediate_lineup_stars': INTERMEDIATE_LINEUP_STARS,
    }


def rules():
    """Every tier line of this module, in the order tiers_v1.json lists them."""
    return {**player_rules(), **lineup_rules()}
