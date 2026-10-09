# Grid Rush

3×3 board: rows are clubs, columns are clubs or nationalities; each square is answered by a
player who fits both.

```bash
python game_modes_data/grid/build_pool.py     # → output/grid_pool.json (kept in git)
```

## The answer pool

`output/grid_pool.json` only decides whether a typed answer is right. It is much wider than
the quiz's player pool: 62,005 players at 268 clubs.

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

Everything comes from the Transfermarkt API (`sources/tm_api.py`): one squad request per club
season (about 9,400), then the player records. The first build takes about 40 minutes;
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

- **5.0 MB** (1.4 MB gzipped). The app should download it only when Grid is opened.
- **1,378 names are shared** by two or more players (12 called Fernando, 8 called Rodri), so
  the app has to match a typed name against everyone with that name.
- **17,139 players have no market value** (careers before about 2004, or never valued), so
  "rarer player" steals can't be ranked for them.
- 101 players have no nationality name (53 have none in the API; the rest are countries no
  dataset player has).
- Country names follow the quiz pool. Some have no flag in the app yet (Cote d'Ivoire,
  Bosnia-Herzegovina, Northern Ireland, Korea, South, Israel…).
- Squads before 1990/91 are not fetched (`FIRST_SEASON`): Maradona shows Napoli only for 1990.
- The app does not read this file yet; it still builds boards from the quiz pool.
