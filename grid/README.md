# Grid Rush

3×3 board: rows are clubs, columns are clubs or nationalities; each square is answered by a
player who fits both. Still a Labs prototype, and it has no data of its own yet.

## What it uses today

Boards are built on the phone from the game data file the app already downloads
(`cache/game_data_v1.json`, made from `transfer_history/output/`). A player "played for" a club
when the club appears in his transfer history; nationality is his `citizenship`.

That limits answers to the ~1,900 players in the pool (peaked at €10M or more), and to clubs
a player transferred to or from.

## Where more data could come from

The Transfermarkt API (`sources/tm_api.py`) has every first-team squad by season, back to at
least 1975:

| Need | Call |
|---|---|
| Everyone who was in a club's squad in a season | `api.squad(club_id, 2005)` (2005 = 2005/06) |
| Names, citizenship, positions for those ids | `api.players([…])` |
| Tell first teams from youth/reserve sides | `api.clubs([…])` → `baseDetails.clubTypeId == 1`, `mainClubId` |

Open questions before building it: which clubs and seasons to cover, whether players outside
the current pool become valid answers, and whether trophies or teammates become categories.
