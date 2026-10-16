"""
How famous a player is, and the difficulty tiers that follow from it, for every mode
(plan §8 item 115, CONTRACTS.md §5 "Tiers"). Every build takes these from here.

- fame(player) = the higher of his Transfermarkt peak value and his legend value. Transfermarkt
  has valued players only since late 2004, so legends.json gives the well-known older ones an
  approximate present-day value (see legend_values).
- value_now(player) = his highest Transfermarkt value since VALUE_NOW_FROM.
- still_playing(...) = not retired, alive, and when without a club, moved in the last year.
- Transfer Path and Player Story: player_tier(fame, value now, still playing).
- Starting XI: a starter is a star when his fame in today's money (growth()) is
  STAR_VALUE or more; lineup_tier(year, stars).

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
BEGINNER_VALUE_NOW = 50_000_000         # still playing and worth €50m+ since VALUE_NOW_FROM,
BEGINNER_FAME = 80_000_000              # or fame €80m+ ...
BEGINNER_FAME_VALUE_NOW = 15_000_000    # ... and still worth €15m+ since VALUE_NOW_FROM
INTERMEDIATE_FAME = 40_000_000          # otherwise: intermediate from €40m fame, expert below

# ── Still playing ──
RETIRED = 'Retired'
NO_CLUB = {'Without Club', 'Career break', 'Unknown'}  # still playing if he moved in the last year
NO_CLUB_GRACE = timedelta(days=365)

# ── Starting XI ──
STAR_VALUE = 40_000_000       # a star: €40m+ fame in today's money
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
    """max(Transfermarkt peak value, legend value); 0 when he has neither."""
    return max(peak_value(profile) or 0, legends().get(str(player_id), 0))


def value_now(history, profile=None):
    """The highest value since VALUE_NOW_FROM, from the market value history
    (TransfermarktAPI.market_value_histories) and the profile's current value; 0 when none."""
    points = [p['value'] for p in history or [] if p['date'] >= VALUE_NOW_FROM]
    current = ((profile or {}).get('marketValueDetails') or {}).get('current') or {}
    if current.get('value') and (current.get('determined') or '') >= VALUE_NOW_FROM:
        points.append(current['value'])
    return max(points, default=0)


def still_playing(current_club, date_of_death, last_move, today=None):
    """Not retired and alive. A player without a club (or on a career break) counts only when
    his last move (ISO date) is less than a year before `today`."""
    if date_of_death or current_club == RETIRED:
        return False
    if current_club in NO_CLUB:
        today = today or date.today()
        return bool(last_move) and date.fromisoformat(last_move[:10]) >= today - NO_CLUB_GRACE
    return True


def player_tier(player_fame, player_value_now, active):
    """Transfer Path and Player Story tier."""
    if active and (player_value_now >= BEGINNER_VALUE_NOW
                   or (player_fame >= BEGINNER_FAME and player_value_now >= BEGINNER_FAME_VALUE_NOW)):
        return 'beginner'
    return 'intermediate' if player_fame >= INTERMEDIATE_FAME else 'expert'


# ── Today's money (Starting XI) ──

@lru_cache(maxsize=1)
def growth_index():
    """({year: mean of the year's GROWTH_TOP highest player values}, latest full year).

    Each player counts once a year, at his highest value that year. Values come from the
    transfermarkt-datasets valuations (every valuation since 2004, frozen in June 2026); the
    latest full year is the last one with valuations in December.
    """
    valuations = pd.read_csv(VALUATIONS_FILE, usecols=['player_id', 'date', 'market_value_in_eur'])
    valuations['year'] = valuations['date'].str[:4].astype(int)
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


def todays_money(player_id, profile, year):
    """A starter's fame in today's money for a match in `year`. Legends are not adjusted:
    their legend value is already a present-day amount."""
    value = fame(player_id, profile)
    return value if is_legend(player_id) else value * growth(year)


def is_star(player_id, profile, year):
    return todays_money(player_id, profile, year) >= STAR_VALUE


def lineup_tier(year, stars):
    """Starting XI tier from the match year and the number of stars among the eleven."""
    if year >= BEGINNER_LINEUP_FROM and stars >= BEGINNER_LINEUP_STARS:
        return 'beginner'
    return 'intermediate' if stars >= INTERMEDIATE_LINEUP_STARS else 'expert'
