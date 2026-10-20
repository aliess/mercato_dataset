# Mercato dataset

The data behind the Mercato app: where it comes from, how it is built for each game mode, and
how to refresh it.

## Where the data comes from

| Source | What we take from it | Access | Code |
|---|---|---|---|
| **Transfermarkt JSON API**<br>`https://tmapi-alpha.transfermarkt.technology` | Everything about players: profiles, full transfer histories, market values, current club; squads by season (who is in the game), match lineups. | No key, no login | `sources/tm_api.py` |
| **transfermarkt-datasets**<br>[github.com/dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets), public R2 bucket (the same files as Kaggle) | No longer the list of players. Still read for: every valuation since 2004 (the growth index that puts old Starting XI values in today's money), the club names and countries it knows (so club names stay as in earlier imports), the transfers of a player whose API request fails, and country names. Frozen since July 2026. | No key, no login | `sources/download_dataset.py` |

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
| `players?ids[]=…` (up to 500 ids) | Profile, current club, current and highest market value | every mode |
| `clubs?ids[]=…` | Name, short name, country, `clubTypeId` (1 = first team), `mainClubId` | transfer history |
| `transfer/history/player/{id}` | Every transfer, loan, return and retirement, with fee and date | transfer history |
| `club/{id}/squad?season=2005` | First-team squad of 2005/06, back to at least 1975. `seasonId=` is ignored (returns the current squad) | grid |
| `games?ids[]=…`, `game/{id}` | Match with both starting lineups, formation, score and every goal; old and national-team matches too. The score includes shootout penalties (`additionType`) | starting XI |
| `competition/{id}/fixtures?season=2004` | Every match of a competition season, with game ids and round names. Ids: `CL`, `FIWC` (World Cup), `EURO`. Summer tournaments are filed under the year before (World Cup 2010 → 2009) | starting XI |
| `club/{id}/fixtures?season=2004` | A club's matches that season | — |
| `competitions?ids[]=CL`, `competition/CL` | Competition details | — |
| `player/{id}/market-value-history` | Every value a player was given: `history[]` with `marketValue.value` and `determined` (date) | transfer history (value now, for the tiers) |

There is no search endpoint and no `countries` endpoint (404). Retired players work by id
(Zidane 3111, Henry 3207). A player who has died has the current club "---" (id 4023);
`build_dataset.py` writes him as Retired (id 123), like any other former player.

## Layout

```
sources/            where the data comes from; shared by every mode
  tm_api.py           Transfermarkt API client (cached, throttled, retries)
  club_types.py       which API clubs are youth or reserve sides (transfer history, Learn routes)
  squads.py           the pool clubs and their squads since 1990/91: the one player list
  fame.py             fame, value now, still playing, and the tiers of every mode
  legends/            legends.json (hand-picked older players and their value) and legend_fame.py
  learn/              Learn: clubs.json (club cards) and dossiers.json (story dossiers), both draft →
                      approved; build_learn.py and build_dossiers.py (learn_v1.json)
  download_dataset.py transfermarkt-datasets tables → sources/dataset/
  player_facts.py     a player's nationality, position and date of birth (for the hints)
  country_ids.json    Transfermarkt country id → name, written by the Grid build
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
| Why each tier (admin panel only) | `transfer_history/`, `starting_xi/` | `output/player_tiers.json`, `output/xi_stars.json` | `cache/tiers_v1.json` |
| Learn (the journey map) | `sources/learn/` | `clubs.json` + the files above | `cache/learn_v1.json` |

Game data does not pass through Firestore. [Publishing](#publishing-to-the-app) below says how
the files get to the app; `game_modes_data/STATUS.md` says what has been uploaded where.

### Fame and difficulty tiers

`sources/fame.py` is the one place that says how famous a player is and which tier he is in
(plan §8 item 115, `../CONTRACTS.md` §5). Every build takes its values from it, and the files
carry the result: the app and the server hold no value lines.

- **Fame** = the higher of the player's Transfermarkt peak value and his legend value.
  Transfermarkt has valued players only since late 2004, so `sources/legends/legends.json`
  gives 333 well-known older players an approximate present-day value (Maradona highest,
  €250m). A legend's value is a tier ceiling spread down by Wikipedia fame
  (`sources/legends/legend_fame.py`); see the Grid README. This raw fame picks who is in the
  pool (€20m / €10m at the five clubs), is `market_value` in `game_data_v1.json` and the Grid
  pool's value.
- **Fame now** (`fame_now`) = the same in today's money, and it sets every tier: peak value ×
  growth(year of the peak), or the legend value when that is higher (legends are not adjusted:
  their value is already today's). The peak year is the date Transfermarkt gives with the peak
  (`marketValueDetails.highest.determined`), else the highest point of the value history, else
  the highest dataset valuation. Growth = the mean of the 300 highest player values in the
  latest full year (2025) ÷ the same in the peak year, at most ×2.5 (2012 and earlier hit the
  cap; 2016 is ×2.02, 2017 ×1.72, 2018 ×1.14, 2023 ×1.08; 2025 on ×1.0). The values are the
  dataset's valuations (`player_valuations`), complete up to June 2026.
- **Value now** = the highest Transfermarkt value since 1 July 2023 (the API's market value
  history plus the current value).
- **Still playing** (`active`) = not retired and alive; a player without a club (or on a career
  break) counts only if his last move was less than a year ago.
- **Transfer Path and Player Story**: Beginner = still playing and (value now €60m+, or fame
  now €80m+ with value now €15m+), and a big club: his current club is one of the 11
  (`BEGINNER_CLUB_IDS`: Real Madrid, Barcelona, Atlético, Man City, Liverpool, Arsenal, Chelsea,
  Man Utd, Tottenham, PSG, Bayern), or he was in one of their squads in 2024/25 or later (the
  Grid's squads, `sources/squads.py`), or he is one of the four all-time stars
  (`ALL_TIME_STAR_IDS`: Messi, Cristiano Ronaldo, Neymar, Benzema). A player who meets the
  value lines without a big club is Intermediate (`why` = `intermediate_club`: Lautaro,
  Osimhen, Barella...). Intermediate = otherwise fame now €40m+ (every legend lands here);
  Expert = fame now under €40m. Written as `difficulty` on each player of
  `game_data_v1.json`; `tiers_v1.json` gives each player's `fame_now`, `peak_year`, `big_club` and the rule.
- **Starting XI**: a starter is a star when his fame now is €40m+; the match year adds nothing
  on top. Beginner = 2017 or later with 10+ stars, Intermediate = 5+ stars, Expert = the rest.

### Player names

Every mode writes player names in plain Latin letters through `sources/names.py`, so a name
can be typed on any keyboard and is spelled the same in every mode: "Pascal Groß" → "Pascal
Gross", "Kenan Yıldız" → "Kenan Yildiz", "Martin Ødegaard" → "Martin Odegaard". Club names keep
their accents.

A player has one spelling across all game files: the app builds one suggestion list from them.
Grid and Starting XI take the name from `PlayerFacts.name()` (`sources/player_facts.py`): the
game data's `player_name` when the player is in Transfer Path, else the API's name. Since
2026-10-16 the game data's names are the API's too (its profiles come from the API), so the
API's spelling is the one everywhere, except the few in `NAME_OVERRIDES` (`sources/names.py`)
where players know another name: Savinho (API "Savio"), Flaco Lopez ("Jose Manuel Lopez"),
Marc-Andre ter Stegen ("Marc ter Stegen"), Rafael Borre ("Rafael Santos Borre"). Other names a
player is known by go in `ALT_NAMES` (same file) and reach the app as `alt_names` in
`game_data_v1.json`: they are accepted as an answer next to his name in every mode, but never
shown. Only players the user names are in it; it starts with the API's names of the four
overrides, so "Savio" still counts for Savinho. The dataset table is not used for names: it is staler than the API. After a
transfer-history build adds players, rebuild Grid and Starting XI so they pick up the spelling.

### Player facts for the hints

Grid Rush rows and Starting XI slots carry a player's position, date of birth and (in Starting
XI) nationality for the hints. `sources/player_facts.py` takes each from the game data when
the player is in it, so every mode agrees, else from the API profile the build already
fetched, else from the dataset. Positions use the game data's 13 names.

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

`publish_files.py` builds six files into `publish/` and uploads them to the project's default
bucket (`<project-id>.firebasestorage.app`) under `cache/`. The app downloads them from there,
all but `tiers_v1.json`, which only the admin panel's `tiers_report` reads.
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
| `tiers_v1.json` | `transfer_history/output/player_tiers.json` and `starting_xi/output/xi_stars.json`: each player's value now and the `sources/fame.py` rule that set his tier, each lineup's star count, the lines and the growth factors those builds used |
| `learn_v1.json` | `sources/learn/clubs.json`, the files above and the API cache (loans, club types), by `sources/learn/build_learn.py`: the Learn worlds, stops, steps, hooks and twins. Only club stops with an `approved` card; with none, no file |
| `manifest.json` | version (SHA-256 of the bytes), size, gzip size and last change of the data files |

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
- **Learn** (`learn_v1.json`, plan `footballquiz/docs/plans/learn_journey.md`): all ten
  worlds. A club stop is built only when its card in `sources/learn/clubs.json` is `approved`
  (cards are written as `draft`, the user reviews them, the arabic agent adds `ar`).
  `--include-drafts` (dev only; the publisher refuses it for prod) builds every club stop, with
  its draft card or `card: null`. `python sources/learn/build_learn.py [--include-drafts]`
  builds just this file into `publish/`. The rules are in the script's docstring: every player
  has one home (current Grid pool club while he plays, else the pool club where he spent the
  most time; 10 have none and go to `beyond_free`); lessons of 6–8, with 5 or 9 only when a
  count can't be cut (10 → 5 + 5); routes count permanent moves between senior teams, so the
  build reads the loan flag and the club types from the API cache (`sources/club_types.py`,
  shared with the transfer-history build), and needs no request while that cache is there.
  Hooks lead with clue 1 of the published set unless another clue has more memorable words or
  it holds a word of his name; a player without a usable clue gets a fact. Celtic is named in
  the plan but is not a Grid pool club, so it has no stop.
  Since item 140 the app reads **dossiers, anchors and trails** (`sources/learn/build_dossiers.py`,
  rules in its docstring). Squad and era dossiers (8–15 players) are cut from each stop's
  players; a former player may sit in any decade he spent half a season at the club, a small
  club's era dossier spans several decades and says so in its title, and a stop of fewer than 8
  borrows players who spent half a season there. Night, nation and route dossiers come from the
  stops' steps. Story dossiers are written in `sources/learn/dossiers.json` (`draft` until the
  user approves; only `approved` ones are published without `--include-drafts`); the build
  checks their lengths, players and that a teaser never spells out the mystery player's name.
  Each story has a `theme` the app groups the Stories page by (item 145): `history` (teams and
  nights that made history), `sellers` (clubs that sell high), `rivals` (both sides of a derby
  or a move between two clubs) or `players` (players with something in common); the build
  refuses any other value and passes it to the story's dossier in `learn_v1.json`.
  Their player lists were drafted from data queries (squads from the API cache, moves, fees,
  XIs): a season squad from the API also lists players loaned out at the start of that season,
  so a "that season's team" list keeps only players at the club 120+ days of it.
- **To undo an upload**: check out the commit whose data was live before, and publish again.
- **`tiers_v1.json` must explain the data**: the publisher stops, writing nothing, when a player
  of the game data or a lineup has no entry, when a player's `why` (recomputed from his fame,
  value now and `active`) does not give his `difficulty`, when `lineup_tier(year, stars)` does
  not give a lineup's, or when `sources/fame.py`'s lines differ from the ones the build used.
  Rebuild the mode it names; `build_dataset.py --from-cache` and `build_lineups.py --from-cache`
  do it from the cache with no API calls.
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
`build_dataset.py --from-cache` reuses cached API responses however old (no requests for
anything already cached).

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

Candidates are the one player list every mode shares (`sources/squads.py`): everyone who was in
the squad of a pool club (top five leagues plus 20 other clubs) in any season since 1990/91,
the same list Grid Rush answers come from. Added to it: every legend, and the few players of
earlier builds who were never in a pool club's squad (`EXTRA_PLAYER_IDS`). From those:

- **Stars**: fame of €20M or more.
- **Big-club players**: fame of €10M–20M and in the squad of, or a transfer to or from,
  Arsenal, AC Milan, Real Madrid, Barcelona or Chelsea (by club id).
- **At least two clubs**: players whose history shows a single first-team club (Lamine Yamal,
  Saka) are left out, because there is no transfer path to guess. A **legend** with a single
  club (Maldini, Totti, Baresi, Giggs, Puyol) stays with `story_only: true`: Player Story can
  use him, Transfer Path skips him.

Thresholds and clubs are constants at the top of `game_modes_data/transfer_history/build_dataset.py`.
A new player is picked up once he is in a pool club's squad and reaches the bar.

### Where each field comes from

Every profile field comes from the player's API profile: name (in Latin letters), slug, image,
date and place of birth, height, citizenship (named through `country_names()` in
`sources/player_facts.py`), position, foot, agent, current club, contract end, date of death.
A date of birth the API marks as unknown is left out. Transfers come from the API's transfer
history; club names and countries the dataset knows are kept so they stay as in earlier
imports. A player whose transfer history request fails keeps his dataset transfers.
`market_value`, `difficulty`, `active` and `story_only` come from `sources/fame.py`.

### Output

`game_modes_data/transfer_history/output/player_profiles.csv` → `players` in `game_data_v1.json`
(column names become field names). `game_modes_data/transfer_history/output/transfer_history.csv`
→ `transfers`, grouped by player, oldest first. Transfers to
youth and reserve sides (Castilla, Barcelona B, U19s, FC Liefering) and moves dated after today
are left out. The API's club type decides what a youth or reserve side is, not the name, so
Willem II and Esbjerg fB stay. Column layout is the original import's, plus `difficulty`,
`active` and `story_only` at the end (from `sources/fame.py`).
