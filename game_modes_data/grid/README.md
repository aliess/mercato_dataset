# Grid Rush

3×3 board: rows are clubs, columns are clubs or nationalities; each square is answered by a
player who fits both. Still a Labs prototype, and it has no data of its own yet.

## What it uses today

Boards are built on the phone from the game data file the app already downloads
(`cache/game_data_v1.json`, made from `game_modes_data/transfer_history/output/`). A player "played for" a club
when the club appears in his transfer history; nationality is his `citizenship`.

That limits answers to the ~1,900 players in the pool (peaked at €10M or more), and to clubs
a player transferred to or from.

## Planned answer pool

A wider pool than the quiz's ~1,900 players, used only to decide whether a typed answer is
right. Decided so far:

- **Clubs covered:** every club of the top five leagues, plus the 20 biggest clubs from other
  leagues.
- **Second nationality counts**: the API has it (`nationalityDetails.nationalities`); it has no
  country names, so the ids need a small lookup table.
- **Still open: what "played for" means.** The API has squads by season but no appearance
  counts. Squad membership is one cheap call per club season; "actually played" has to be
  worked out from every match lineup of every club season, which is far more requests.

| Need | Call |
|---|---|
| Everyone who was in a club's squad in a season | `api.squad(club_id, 2005)` (2005 = 2005/06), back to at least 1975 |
| A club's matches in a season, then who started them | `api.club_fixtures(club_id, 2005)` → `api.games([…])` |
| Names, both nationalities, positions | `api.players([…])` |
| Tell first teams from youth/reserve sides | `api.clubs([…])` → `clubTypeId` (see `is_reserve_side` in `transfer_history/build_dataset.py`; the flag alone is not enough) |

Other open questions: how far back to go, and how steals work for players from before market
values existed (about 2004).
