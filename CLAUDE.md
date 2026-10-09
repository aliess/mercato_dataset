# Mercato dataset

Builds the game data for every mode from the Transfermarkt JSON API and the frozen
transfermarkt-datasets tables. The backend is `../footballquiz_firebase`, the app is
`../footballquiz`. What this project hands to them is in `../CONTRACTS.md`.

Read [README.md](README.md) first: sources, endpoint table, layout, refresh steps, which
players are included. The `mercato-data` skill (`../.claude/skills/mercato-data/`) has the
short version and the caveats. Each mode folder under `game_modes_data/` has its own README.

## Hard rule

**Never run anything that writes to Firestore or Storage, dev or prod, without the user saying
yes for that run.** That covers `./update.sh … --apply`, `sync_firestore.py --apply`,
`upload_clues.js --apply`, `--restore … --apply` and `--rebuild-only`. General permission to
deploy does not cover it. Build locally, show the comparison report and the dry-run diff, and
wait.

A dry run of the sync still reads both collections (about 17k reads). Do not run one more
often than needed.

## Commands

Python 3.11, pyenv env `football-dataset-env`; `pip install -r requirements.txt`.

- `python sources/download_dataset.py`: fetch the dataset tables.
- `python game_modes_data/transfer_history/build_dataset.py`: build and compare with the last
  build. Exits with an error when the result looks broken; read the report, don't override it.
- `./update.sh dev`: download, build, dry-run diff against dev.
- `python game_modes_data/starting_xi/build_lineups.py`, `python game_modes_data/grid/build_pool.py`.
- Clues: `game_modes_data/clues/README.md`.

The sync and clue scripts need the compiled functions next door:
`cd ../footballquiz_firebase/functions && npm install && npm run build`.

## Conventions

- All API access goes through `sources/tm_api.py` (cache, throttle, retries). No second client.
  Keep the cache on and the default 4 workers.
- The API is unofficial. If an endpoint fails, test it by hand with curl before changing code,
  and say that it may have changed.
- There is no search endpoint. When players are missing or need adding, ask the user whether
  they will give the list or want a search; never guess a list or start a large crawl unasked.
- Output CSV columns become Firestore field names that the functions and the app read. Do not
  rename or drop a column without a contract change.
- `game_modes_data/STATUS.md` and every `LAST_UPDATED.json` are written by the scripts. Never
  edit them by hand.
- The two `*-firebase-adminsdk-*.json` files are service-account keys. They are git-ignored;
  never commit, print or copy them.
- One spelling per country, through `sources/country_names.py`.
- When you learn something new about the API or the data, update `README.md` and the
  `mercato-data` skill.
