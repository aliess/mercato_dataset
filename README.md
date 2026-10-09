# Mercato dataset

The data behind the Mercato app: where it comes from, how it is built for each game mode, and
how to refresh it.

## Where the data comes from

| Source | What we take from it | Access | Code |
|---|---|---|---|
| **Transfermarkt JSON API**<br>`https://tmapi-alpha.transfermarkt.technology` | Live data: full transfer histories, highest market value, current club. Also has squads by season and match lineups (not used yet). | No key, no login | `sources/tm_api.py` |
| **transfermarkt-datasets**<br>[github.com/dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets), public R2 bucket (the same files as Kaggle) | The list of players to consider, and their profile fields (birth, position, foot, height…). Frozen since July 2026. | No key, no login | `sources/download_dataset.py` |

Clue text (Three Clues) is written by hand in this repo, against facts exported from the two
sources above.

### About the Transfermarkt API

- It is the API behind Transfermarkt's own apps. It is **unofficial and undocumented**: endpoints
  can change or disappear without notice, so every response is cached and each build is compared
  with the last one.
- **No username or password is needed.** Opening the base URL in a browser redirects to `/doc`,
  the API's documentation page, and only that page asks for a login. The data endpoints answer
  plain GET requests, e.g. open
  `https://tmapi-alpha.transfermarkt.technology/players?ids[]=28003` in a browser.
- Why not transfermarkt.com itself: since July 2026 the website serves an AWS WAF "Human
  Verification" challenge to scrapers. That is what stopped transfermarkt-datasets
  ([discussion #383](https://github.com/dcaribou/transfermarkt-datasets/discussions/383)).
  The dataset's last valuations are from 2026-06-12.

Endpoints checked by hand (last on 2026-10-09). All ids are Transfermarkt ids.

| Endpoint | Returns | Used by |
|---|---|---|
| `players?ids[]=…` (up to 500 ids) | Profile, current club, market value history | transfer history |
| `clubs?ids[]=…` | Name, short name, country, `clubTypeId` (1 = first team), `mainClubId` | transfer history |
| `transfer/history/player/{id}` | Every transfer, loan, return and retirement, with fee and date | transfer history |
| `club/{id}/squad?season=2005` | First-team squad of 2005/06, back to at least 1975. `seasonId=` is ignored (returns the current squad) | planned: grid |
| `games?ids[]=…`, `game/{id}` | Match with both starting lineups, formation, score and every goal; old and national-team matches too. The score includes shootout penalties (`additionType`) | starting XI |
| `competition/{id}/fixtures?season=2004` | Every match of a competition season, with game ids and round names. Ids: `CL`, `FIWC` (World Cup), `EURO`. Summer tournaments are filed under the year before (World Cup 2010 → 2009) | starting XI |
| `club/{id}/fixtures?season=2004` | A club's matches that season | — |
| `competitions?ids[]=CL`, `competition/CL` | Competition details | — |
| `player/{id}/market-value-history` | Answers (200); content not looked at yet | — |

There is no search endpoint and no `countries` endpoint (404). Retired players work by id
(Zidane 3111, Henry 3207).

## Layout

```
sources/            where the data comes from; shared by every mode
  tm_api.py           Transfermarkt API client (cached, throttled, retries)
  download_dataset.py transfermarkt-datasets tables → sources/dataset/
  dataset/, cache/    downloaded tables and API responses (not in git)
game_modes_data/    one folder per game mode: its scripts and its data
  transfer_history/   the player pool and every player's transfers; all other modes build on it
  clues/              Three Clues: hand-written clue text per player
  starting_xi/        Starting XI: starting lineups of Champions League, World Cup and Euro knockout matches
  grid/               Grid Rush: no data of its own yet (uses the transfer history)
update.sh           refresh the transfer history in one command
```

| Game mode | Folder | Status | Ends up in |
|---|---|---|---|
| Transfer history (main quiz, daily, multiplayer) | `game_modes_data/transfer_history/` | Live | Firestore `player_profiles_and_value`, `transfer_history_filtered` → Storage `cache/game_data_v1.json` |
| Three Clues | `game_modes_data/clues/` ([README](game_modes_data/clues/README.md)) | Live | Firestore `player_clues` → Storage `cache/clues_v1.json` |
| Starting XI | `game_modes_data/starting_xi/` ([README](game_modes_data/starting_xi/README.md)) | Data built; app still uses a placeholder | `game_modes_data/starting_xi/output/xi_lineups.json` (not in the app yet) |
| Grid Rush | `game_modes_data/grid/` ([README](game_modes_data/grid/README.md)) | No data yet | Built on the phone from `game_data_v1.json` |

## Setup

```bash
pip install -r requirements.txt          # once (pyenv env: football-dataset-env)
```

Firestore writes need a service-account key per project: Firebase console → Project settings →
Service accounts → *Generate new private key*. Save it in this folder under its downloaded name
(`<project-id>-firebase-adminsdk-….json`); the scripts find it by project, refuse a key that
belongs to another project, and git ignores it. `--credentials key.json`,
`GOOGLE_APPLICATION_CREDENTIALS` and `gcloud auth application-default login` also work for
`sync_firestore.py`.

The sync and clue scripts also use the built Cloud Functions next to this repo
(`cd ../footballquiz_firebase/functions && npm install && npm run build`).

Projects: `dev` = `football-quiz-32eb9`, `prod` = `mercato-6e710`.

## Refresh the transfer history

Run from this folder. Do it after each transfer window, or whenever the data should be current.

```bash
./update.sh dev                          # download + build + show what would change in dev
./update.sh dev --apply                  # …and write it, then rebuild the game data file
./update.sh prod --apply
```

Or step by step:

| Step | Command | What it does |
|---|---|---|
| 1 | `python sources/download_dataset.py` | Fetches the 6 tables needed (~25 MB) into `sources/dataset/` |
| 2 | `python game_modes_data/transfer_history/build_dataset.py` | Picks players, refreshes them from the Transfermarkt API, writes `game_modes_data/transfer_history/output/`, compares with the last build |
| 3 | `python game_modes_data/transfer_history/sync_firestore.py --project dev` | Shows what would change in Firestore (dry run) |
| 4 | `python game_modes_data/transfer_history/sync_firestore.py --project dev --apply` | Backs up Firestore, writes only changed docs, rebuilds the game data file |

A full refresh is about 3,500 API requests and takes a few minutes. Responses are cached in
`sources/cache/tm_api/` for 24 h (`--max-age-hours`), so a rerun the same day is instant.
`build_dataset.py --offline` skips the API and uses only the dataset tables.

After new players are added, give them clues: see [game_modes_data/clues/README.md](game_modes_data/clues/README.md).

### Checking a new build before it goes live

- Each build moves the last one to `game_modes_data/transfer_history/output/previous/` and compares them:
  players added/removed, players with no transfers, players who lost transfers, club changes,
  latest transfer date. The report is saved to `game_modes_data/transfer_history/output/compare_report.txt`. If
  something looks broken (counts drop more than 5%, unplayable players increase, data got
  older), the build exits with an error. Re-run the comparison any time with
  `python game_modes_data/transfer_history/compare_outputs.py`.
- Each `sync_firestore.py --apply` first saves both Firestore collections to
  `game_modes_data/transfer_history/backups/<project>/<time>/`. To undo a sync:
  `python game_modes_data/transfer_history/sync_firestore.py --project prod --restore game_modes_data/transfer_history/backups/mercato-6e710/<time> --apply`
- The sync stops if it would delete more than 20% of a collection (`--allow-shrink` overrides).

### Which players are included

- **Stars**: highest-ever market value of €20M or more.
- **Big-club players**: peaked at €10M–20M and played for Arsenal, AC Milan, Real Madrid,
  Barcelona or Chelsea (first team, by club id).

- **At least two clubs**: players whose history shows a single first-team club (Lamine Yamal,
  Saka, Totti) are left out, because there is no transfer path to guess. This also takes them
  out of Three Clues, since both modes share the player pool.

Thresholds and clubs are constants at the top of `game_modes_data/transfer_history/build_dataset.py`.
Candidates come from the dataset tables, so a player who first reached €10M after June 2026 is
not picked up until the dataset updates again or the candidate list gets another source.

### How the two sources are combined

For every candidate (anyone who peaked at €10M or more in the dataset), the API replaces the
dataset's transfer history, highest market value, current club, contract end and date of death.
The dataset still supplies the profile fields and the club names and countries it knows, so
names stay the same as in earlier imports. Players whose API request fails keep their dataset
transfers.

### Output

`game_modes_data/transfer_history/output/player_profiles.csv` → `player_profiles_and_value` (doc id = `player_id`).
`game_modes_data/transfer_history/output/transfer_history.csv` → `transfer_history_filtered`. Transfers to
youth and reserve sides (Castilla, Barcelona B, U19s, FC Liefering) and moves dated after today
are left out. The API's club type decides what a youth or reserve side is, not the name, so
Willem II and Esbjerg fB stay. Column layout is unchanged from the original import.
