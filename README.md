# Mercato dataset

Builds the player and transfer data behind the Mercato app and syncs it into Firestore.

```
transfermarkt-datasets (R2/Kaggle) ─┐
                                    ├─ build_dataset.py ─→ output/*.csv ─→ sync_firestore.py ─→ Firestore
Transfermarkt JSON API (live) ──────┘                                                          └→ adminRebuildGameData
                                                                                                  → cache/game_data_v1.json (app)
```

## Update the data

```bash
pip install -r requirements.txt          # once (pyenv env: football-dataset-env)

./update.sh dev                          # download + build + dry-run diff against dev
./update.sh dev --apply                  # …and write it, then rebuild game data
./update.sh prod --apply
```

Or step by step:

| Step | Command | What it does |
|---|---|---|
| 1 | `python scripts/download_dataset.py` | Fetches the 6 tables needed (~25 MB) into `dataset/` |
| 2 | `python scripts/build_dataset.py` | Picks players, refreshes them from the Transfermarkt API, writes `output/`, compares with the last build |
| 3 | `python scripts/sync_firestore.py --project dev` | Shows what would change in Firestore (dry run) |
| 4 | `python scripts/sync_firestore.py --project dev --apply` | Backs up Firestore, writes only changed docs, rebuilds the game data file |

`build_dataset.py --offline` skips the API and uses only the dataset tables.

### Checking a new build before it goes live

- Each build moves the last one to `output/previous/` and compares them: players added/removed,
  players with no transfers, players who lost transfers, club changes, latest transfer date.
  The report is saved to `output/compare_report.txt`. If something looks broken (counts drop more
  than 5%, unplayable players increase, data got older), the build exits with an error.
  Re-run the comparison any time with `python scripts/compare_outputs.py`.
- Each `sync_firestore.py --apply` first saves both Firestore collections to
  `backups/<project>/<time>/`. To undo a sync:
  `python scripts/sync_firestore.py --project prod --restore backups/mercato-6e710/<time> --apply`

### Firestore credentials

`sync_firestore.py` needs one of:
- `--credentials key.json`: Firebase console → Project settings → Service accounts → *Generate new private key* (one per project; never commit it, `*-firebase-adminsdk-*.json` is git-ignored). The script refuses a key that belongs to a different project than `--project`.
- `GOOGLE_APPLICATION_CREDENTIALS=key.json`
- `gcloud auth application-default login`

The game data rebuild uses the admin key in `../footballquiz_firebase/ADMIN_API_KEY`.

## Why the API refresh

The [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) pipeline stopped on
2026-07-10 ([discussion #383](https://github.com/dcaribou/transfermarkt-datasets/discussions/383)):
transfermarkt.com now serves an AWS WAF "Human Verification" challenge to scrapers. Transfermarkt's
JSON API (`tmapi-alpha.transfermarkt.technology`, used by their apps) still answers, so
`build_dataset.py` uses it to fetch, for every candidate player:

- the full transfer history (including loans, returns and retirements), which replaces the dataset's;
- the highest market value and current club.

Responses are cached in `cache/tm_api/` for 24 h (`--max-age-hours`), so reruns are fast. A full
refresh is ~5,000 requests and takes about 2 minutes.

The dataset tables are still used to choose candidates (anyone who peaked at €10M+) and for profile
fields (birth, position, foot…). If upstream resumes, re-run `download_dataset.py`.

## Which players are included

- **Stars**: highest-ever market value above €20M.
- **Big-club players**: peaked at €10M–20M and played for Arsenal, AC Milan, Real Madrid, Barcelona
  or Chelsea (first team, by club id).

Thresholds and clubs are constants at the top of `scripts/build_dataset.py`.

## Output

`output/player_profiles.csv` → `player_profiles_and_value` (doc id = `player_id`).
`output/transfer_history.csv` → `transfer_history_filtered`. Transfers to youth/reserve sides
(names ending in U19, U21, B, II…) and moves dated after today are left out. Column layout is
unchanged from the original import.
