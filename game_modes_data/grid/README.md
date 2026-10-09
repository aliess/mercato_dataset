# Grid Rush

3×3 board: rows are clubs, columns are clubs or nationalities; each square is answered by a
player who fits both.

```bash
python game_modes_data/grid/build_pool.py     # → output/grid_pool.json (kept in git)
```

## The answer pool

`output/grid_pool.json` only decides whether a typed answer is right. It is much wider than
the quiz's player pool: 46,872 players at 268 clubs.

- **A player counts for a club** when he was in its squad in any season since 1990/91, whether
  or not he played a match. Quiz-pool players also count for the clubs in their transfer
  history, as in the app today.
- **Clubs:** every club that played in the Premier League, LaLiga, Serie A, Bundesliga or
  Ligue 1 in any of those seasons (248), plus the 20 clubs outside those leagues that the most
  quiz-pool players passed through (`OTHER_CLUB_IDS` in `build_pool.py`): Benfica, Ajax, Porto,
  Sporting, Fenerbahce, PSV, Galatasaray, Besiktas, Anderlecht, Salzburg, Feyenoord, Olympiacos,
  Flamengo, River Plate, Club Brugge, Trabzonspor, Sao Paulo, Braga, Boca Juniors, Palmeiras.
  A club's squads are taken for every season, also those it spent in a lower division.
- **Nationality:** the player's main one only.

### Who is left out

Market value is the only fame signal the API has, and Transfermarkt only started valuing
players in late 2004. So players without a value are handled in two groups:

| Group | Rule | Players |
|---|---|---|
| In a squad in 2005/06 or later, never valued | Left out: fringe squad members (third keepers, registered youth players) | 4,749 out |
| Career ended before values existed | Kept only when notable (below) | 2,006 kept, 10,384 out |

An earlier player is **notable** when he did at least one of these:

- started a World Cup match (1986–2002), a Euro match (1988–2004) or a Champions League match
  (1992/93–2004/05), in any round;
- was in a top-five-league squad for five seasons or more.

That keeps Maradona, van Basten, Baggio, Baresi, Cantona, Lineker, Gascoigne and Le Tissier
(10 seasons at Southampton, no big match), and drops lower-division and short-stay players.
The constants are at the top of `build_pool.py`.

Everything comes from the Transfermarkt API (`sources/tm_api.py`): one squad request per club
season (about 9,400), about 1,800 old World Cup, Euro and Champions League matches, then the
player records. The first build takes about 40 minutes;
finished seasons are cached for good, so a rebuild only fetches the last two seasons.

## File layout

```json
{"version": 1, "first_season": 1990,
 "clubs": [{"id": 418, "name": "Real Madrid", "country": "Spain", "league": "ES1",
            "top5_seasons": 37, "players": 374}],
 "players": [["3111", "Zinédine Zidane", "France", 25000000,
              [[895, 1990, 1991], [40, 1992, 1995], [506, 1996, 2000], [418, 2001, 2005]]]]}
```

A player is `[id, name, nation, peak market value, clubs]`; each club is
`[club id, first season, last season]` (2001 = 2001/02). The seasons are the first and last
he was in the squad, not proof he was there every season in between.

## Known limits

- **4.0 MB** (1.1 MB gzipped). The app should download it only when Grid is opened.
- **917 names are shared** by two or more players (12 called Fernando, 8 called Rodri), so
  the app has to match a typed name against everyone with that name.
- **2,006 kept players have no market value** (the notable ones from before late 2004), and
  players who retired soon after have a low late-career value, not their peak (Zidane €25m,
  Maldini €3m). So "rarer player" steals can't be ranked fairly for anyone from that era.
- The API files the English top flight under the Premier League only from 1992/93, so 1990/91
  and 1991/92 don't count towards the five seasons, and a club that was only in the old First
  Division in those two seasons is not in the club list.
- 101 players have no nationality name (53 have none in the API; the rest are countries no
  dataset player has).
- Country names follow the quiz pool. Some have no flag in the app yet (Cote d'Ivoire,
  Bosnia-Herzegovina, Northern Ireland, Korea, South, Israel…).
- Squads before 1990/91 are not fetched (`FIRST_SEASON`): Maradona shows Napoli only for 1990.
- The app does not read this file yet; it still builds boards from the quiz pool.
