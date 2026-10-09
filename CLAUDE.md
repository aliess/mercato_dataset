# Mercato dataset

Builds the game data for every mode from the Transfermarkt JSON API and the frozen
transfermarkt-datasets tables. The backend is `../footballquiz_firebase`, the app is
`../footballquiz`. What this project hands to them is in `../CONTRACTS.md`.

Read [README.md](README.md) first: sources, endpoint table, layout, refresh steps, which
players are included. The `mercato-data` skill (`../.claude/skills/mercato-data/`) has the
short version and the caveats. Each mode folder under `game_modes_data/` has its own README.

## Hard rule

**Never upload anything to Storage (or write to Firestore), dev or prod, without the user saying
yes for that run.** That covers `publish_files.py … --apply` and `./update.sh … --apply`.
General permission to deploy does not cover it. Build locally, show the comparison report and
the publisher's dry-run output, and wait.

Game data does not pass through Firestore any more: `publish_files.py` builds five JSON files in
`publish/` and uploads them to Storage `cache/`. Nothing in this project reads or writes
Firestore. The old collections are still there until the user deletes them; leave them alone.
A dry run costs a few public HTTP downloads and nothing else.

## Commands

Python 3.11, pyenv env `football-dataset-env`; `pip install -r requirements.txt`.

- `python sources/download_dataset.py`: fetch the dataset tables.
- `python game_modes_data/transfer_history/build_dataset.py`: build and compare with the last
  build. Exits with an error when the result looks broken; read the report, don't override it.
- `python publish_files.py --project dev|prod`: build the five files into `publish/` and compare
  them with what is live (dry run). `--apply` uploads the files that changed.
- `./update.sh dev`: download, build, then that dry run against dev.
- `python game_modes_data/starting_xi/build_lineups.py`, `python game_modes_data/grid/build_pool.py`.
- Clues: `game_modes_data/clues/README.md`.

The publisher and the clue checker run the compiled clue validator next door:
`cd ../footballquiz_firebase/functions && npm install && npm run build`.

## Conventions

- All API access goes through `sources/tm_api.py` (cache, throttle, retries). No second client.
  Keep the cache on and the default 4 workers.
- The API is unofficial. If an endpoint fails, test it by hand with curl before changing code,
  and say that it may have changed.
- There is no search endpoint. When players are missing or need adding, ask the user whether
  they will give the list or want a search; never guess a list or start a large crawl unasked.
- Output CSV columns become the field names in `game_data_v1.json`, which the app and the
  functions decode by name. Do not rename or drop a column without a contract change. The same
  goes for the fields of the lineup, grid and clue files.
- The published files must stay deterministic: no build time inside them, stable order. The
  manifest's version is the SHA-256 of the bytes, and the app re-downloads when it changes.
- `game_modes_data/STATUS.md` and every `LAST_UPDATED.json` are written by the scripts. Never
  edit them by hand.
- The two `*-firebase-adminsdk-*.json` files are service-account keys, used only by
  `publish_files.py --apply`. They are git-ignored; never commit, print or copy them.
- One spelling per country, through `sources/country_names.py`.
- When you learn something new about the API or the data, update `README.md` and the
  `mercato-data` skill.
