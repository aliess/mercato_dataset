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
| `club/{id}/squad?season=2005` | First-team squad of 2005/06, back to at least 1975. `seasonId=` is ignored (returns the current squad) | grid |
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
publish_files.py    builds the JSON files the app downloads and uploads them to Firebase Storage
publish/            the files it built (not in git)
game_modes_data/    one folder per game mode: its scripts and its data
  transfer_history/   the player pool and every player's transfers; all other modes build on it
  clues/              Three Clues: hand-written clue text per player
  starting_xi/        Starting XI: starting lineups of Champions League, World Cup and Euro knockout matches
  grid/               Grid Rush: everyone who was in a club's squad, 268 clubs since 1990/91
update.sh           refresh the transfer history and run the publisher in one command
```

| Game mode | Folder | Built file | Published as (Storage) |
|---|---|---|---|
| Transfer history (main quiz, daily, multiplayer) | `game_modes_data/transfer_history/` | `output/player_profiles.csv`, `output/transfer_history.csv` | `cache/game_data_v1.json` |
| Three Clues | `game_modes_data/clues/` ([README](game_modes_data/clues/README.md)) | `player_clues.json` | `cache/clues_v1.json` |
| Starting XI | `game_modes_data/starting_xi/` ([README](game_modes_data/starting_xi/README.md)) | `output/xi_lineups.json` | `cache/xi_v1.json` |
| Grid Rush | `game_modes_data/grid/` ([README](game_modes_data/grid/README.md)) | `output/grid_pool.json` | `cache/grid_pool_v1.json` |

Game data does not pass through Firestore. [Publishing](#publishing-to-the-app) below says how
the files get to the app; `game_modes_data/STATUS.md` says what has been uploaded where.

### Player names

Every mode writes player names in plain Latin letters through `sources/names.py`, so a name
can be typed on any keyboard and is spelled the same in every mode: "Pascal Groß" → "Pascal
Gross", "Kenan Yıldız" → "Kenan Yildiz", "Martin Ødegaard" → "Martin Odegaard". Club names keep
their accents.

### When was each set last updated?

See [game_modes_data/STATUS.md](game_modes_data/STATUS.md): one table for all modes, rewritten by the
scripts on every build and upload.

It is made from the `LAST_UPDATED.json` in every mode folder, also kept in git: `built` is when the files here were last built (with counts and the newest match or
transfer in them), and `synced` is when they were last uploaded to each Firebase project. If `built` is
newer than `synced`, the app is behind the files. For an exact answer run the publisher's dry run.

## Setup

```bash
pip install -r requirements.txt          # once (pyenv env: football-dataset-env)
```

Uploading needs a service-account key per project: Firebase console → Project settings →
Service accounts → *Generate new private key*. Save it in this folder under its downloaded name
(`<project-id>-firebase-adminsdk-….json`); the publisher finds it by project, refuses a key that
belongs to another project, and git ignores it. `--credentials key.json` and
`GOOGLE_APPLICATION_CREDENTIALS` also work. Building and dry runs need no key.

The publisher and the clue checker run the clue validator from the built Cloud Functions next
to this repo (`cd ../footballquiz_firebase/functions && npm install && npm run build`, needs node).

Projects: `dev` = `football-quiz-32eb9`, `prod` = `mercato-6e710`.

## Publishing to the app

`publish_files.py` builds five files into `publish/` and uploads them to the project's default
bucket (`<project-id>.firebasestorage.app`) under `cache/`. The app downloads them from there.
The shapes are in `../CONTRACTS.md` §1.

```bash
python publish_files.py --project dev            # dry run: build, compare with what is live
python publish_files.py --project dev --apply    # upload what changed (ask the user first)
python publish_files.py --project prod --apply
```

| File | Built from |
|---|---|
| `game_data_v1.json` | `transfer_history/output/player_profiles.csv` and `transfer_history.csv` |
| `clues_v1.json` | `clues/player_clues.json`: approved sets whose player is in the game data and that pass the validator |
| `xi_v1.json` | `starting_xi/output/xi_lineups.json` |
| `grid_pool_v1.json` | `grid/output/grid_pool.json` |
| `manifest.json` | version (SHA-256 of the bytes), size, gzip size and last change of the four files |

- **The dry run** downloads the live `manifest.json` by public HTTP and says per file whether it
  is unchanged, changed or new, with sizes and counts. For a file that would change it also
  downloads the live copy and lists what differs (players added and removed, transfers, clue
  sets); `--no-diff` skips that. It uploads nothing and never touches Firestore.
- **`--apply`** uploads only the files whose version changed, gzip-compressed (`contentEncoding:
  gzip`, `cacheControl: public, max-age=300`), reads each one back and checks it, and uploads
  `manifest.json` last (`cacheControl: no-cache`). When nothing changed it uploads nothing.
- **The files are deterministic**: no build time inside, fixed order, so the same inputs give the
  same version and the app does not re-download. Keep it that way when changing a builder.
- **Safety stop**: a file that would be more than 20% smaller than the live one stops the run
  (`--allow-shrink` overrides).
- **To undo an upload**: check out the commit whose data was live before, and publish again.
  Storage keeps no old versions.
- How values are written in `game_data_v1.json` (the app depends on it): ids and `market_value`
  are integers, `height` a whole number, dates are `{"_seconds": …, "_nanoseconds": 0}` at
  midnight UTC, empty cells are left out. A transfer's `id` is
  `<player_id>_<yyyymmdd>_<from id>_<to id>`. Two moves on the same day are put in the order
  that connects (the move that starts where the player was comes first).
- A clue set's `v` is the first 52 bits of the SHA-256 of its six strings: it changes only when
  the text changes.

The Firestore collections that used to hold this data (`player_profiles_and_value`,
`transfer_history_filtered`, `player_clues`) are no longer read or written by anything here.
Old backups of them stay in `game_modes_data/transfer_history/backups/` (not in git).

## Refresh the transfer history

Run from this folder. Do it after each transfer window, or whenever the data should be current.

```bash
./update.sh dev                          # download + build + show what would change in dev
./update.sh dev --apply                  # …and upload it (ask the user first)
./update.sh prod --apply
```

Or step by step:

| Step | Command | What it does |
|---|---|---|
| 1 | `python sources/download_dataset.py` | Fetches the 6 tables needed (~25 MB) into `sources/dataset/` |
| 2 | `python game_modes_data/transfer_history/build_dataset.py` | Picks players, refreshes them from the Transfermarkt API, writes `game_modes_data/transfer_history/output/`, compares with the last build |
| 3 | `python publish_files.py --project dev` | Builds the files in `publish/` and shows what would change in dev (dry run) |
| 4 | `python publish_files.py --project dev --apply` | Uploads the files that changed, then the manifest |

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
- The publisher's dry run then compares the new files with what is live in a project, and stops
  if a file would shrink by more than 20% (see [Publishing](#publishing-to-the-app)).

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

`game_modes_data/transfer_history/output/player_profiles.csv` → `players` in `game_data_v1.json`
(column names become field names). `game_modes_data/transfer_history/output/transfer_history.csv`
→ `transfers`, grouped by player, oldest first. Transfers to
youth and reserve sides (Castilla, Barcelona B, U19s, FC Liefering) and moves dated after today
are left out. The API's club type decides what a youth or reserve side is, not the name, so
Willem II and Esbjerg fB stay. Column layout is unchanged from the original import.
