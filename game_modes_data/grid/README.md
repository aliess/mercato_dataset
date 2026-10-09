# Grid Rush

3×3 board: rows are clubs, columns are clubs or nationalities; each square is answered by a
player who fits both.

```bash
python game_modes_data/grid/build_pool.py     # → output/grid_pool.json (kept in git)
```

## The answer pool

`output/grid_pool.json` only decides whether a typed answer is right. It is much wider than
the quiz's player pool: 45,048 players at 268 clubs. Every player has a market value.

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

### Market values and legends

The game rewards obscure picks, and market value is how it tells a famous player from an
obscure one. Transfermarkt has only valued players since late 2004, so:

- **A player with no market value is left out** (16,957): fringe squad members who were never
  valued, and everyone whose career ended before values existed.
- **`legends.json` brings the well-known ones back** with an approximate present-day value,
  picked by hand: 333 players. 183 of them had no value (Maradona, van Basten, Baggio,
  Baresi); the other 150 had a low late-career value that would have made them look obscure
  (Maldini €3m, Cafu €3m, Zidane €25m), and get the higher one.

| Value | Who | Examples |
|---|---|---|
| €250m | The most famous player in the pool | Maradona |
| €200m | | Ronaldo, Zidane |
| €180m | | Maldini, Beckham, Henry, Romário, van Basten, Baggio, Matthäus, Gullit, Baresi |
| €150m | Superstars | Cantona, Schmeichel, Weah, Figo, Raúl, Batistuta, Bergkamp, Buffon, Totti |
| €120m | Stars | Blanc, Papin, Redondo, Suker, Seedorf, Davids, Zola, Hagi |
| €100m | Well known | Kohler, Zenga, Pearce, Poyet, Kanu, Yorke, Mendieta |
| €80m | Known to keen fans | Bakero, Kirsten, Dahlin, Hendry, Onopko |

For comparison, the highest real values are Haaland and Yamal at €220m. To add a legend or
change a value, edit `legends.json` (Transfermarkt player id, name, value) and rebuild; a
player without a value who is not in that file stays out.

Everything comes from the Transfermarkt API (`sources/tm_api.py`): one squad request per club
season (about 9,400), then the player records. The first build takes about 40 minutes;
finished seasons are cached for good, so a rebuild only fetches the last two seasons.

## File layout

```json
{"version": 1, "first_season": 1990,
 "clubs": [{"id": 418, "name": "Real Madrid", "country": "Spain", "league": "ES1",
            "top5_seasons": 37, "players": 374}],
 "players": [["3111", "Zinedine Zidane", "France", 25000000,
              [[895, 1990, 1991], [40, 1992, 1995], [506, 1996, 2000], [418, 2001, 2005]]]]}
```

A player is `[id, name, nation, peak market value, clubs]`; each club is
`[club id, first season, last season]` (2001 = 2001/02). The seasons are the first and last
he was in the squad, not proof he was there every season in between.

## Known limits

- **3.8 MB** (1.1 MB gzipped). The app should download it only when Grid is opened.
- **899 names are shared** by two or more players (12 called Fernando, 8 called Rodri), so
  the app has to match a typed name against everyone with that name.
- **Legend values are judgment calls**, not data. Players of that era who are not in
  `legends.json` but have a Transfermarkt value keep it, however low.
- **27,000 players are valued under €1m.** They are real squad members and the obscure
  answers the game rewards, but most are not names anyone would type.
- The API files the English top flight under the Premier League only from 1992/93, so a club
  that was only in the old First Division in 1990/91 or 1991/92 is not in the club list.
- 101 players have no nationality name (53 have none in the API; the rest are countries no
  dataset player has).
- Country names follow the quiz pool. Some have no flag in the app yet (Cote d'Ivoire,
  Bosnia-Herzegovina, Northern Ireland, Korea, South, Israel…).
- Squads before 1990/91 are not fetched (`FIRST_SEASON`): Maradona shows Napoli only for 1990.
- `output/grid_pool.json` is published as Storage `cache/grid_pool_v1.json` by `publish_files.py`
  (run from the repo root; uploading needs the user's yes). The publisher checks that every
  player row has the five fields and that every club id in it is in `clubs`. A player's `nation`
  can be null (31 players).
