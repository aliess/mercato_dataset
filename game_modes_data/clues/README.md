# Three Clues content

Clue text for the Three Clues mode, written and reviewed here, then published to Storage
`cache/clues_v1.json` by `publish_files.py` (see `footballquiz/docs/plans/three_clues_mode_plan.md` §3 and §8.2).
It does not pass through Firestore.

| File | What it is |
|---|---|
| `facts.jsonl` | Player facts to write against (`python game_modes_data/clues/export_clue_facts.py`) |
| `batches/batch_NN.json` | The written clues, about 30 players per file |
| `player_clues.json` | Every clue set that passes the server validator, keyed by player ID: what the publisher reads |
| `publishable_clues.js` | Used by `publish_files.py`: picks the approved sets whose player is in the game data and runs the validator again |
| `clue_report.json` | Errors (left out) and warnings (kept) from the validator |
| `review.csv` | The same clues as a spreadsheet, ranked by market value, with each player's difficulty |

```bash
python game_modes_data/clues/export_clue_facts.py                      # every player, with its difficulty
node game_modes_data/clues/check_clues.js                              # validate + merge
python publish_files.py --project dev                                  # dry run: what would change in the app's clue file
```

Every doc starts as `status: "draft"`. Change it to `"approved"` once reviewed; only approved
docs are published to the app. Uploading (`publish_files.py … --apply`) needs the user's yes.
In the published file each set carries `v`, a number made from its text, so it changes only
when the text changes.

## Batch format

```json
[{"player_id": "85314", "name": "Oscar",
  "clue1_en": "…", "clue1_ar": "…",
  "clue2_en": "…", "clue2_ar": "…",
  "clue3_en": "…", "clue3_ar": "…",
  "notes": "verify: the facts that don't come from facts.jsonl"}]
```

`name` is only for reading the file; it isn't published.

## Writing rules

- **Voice:** first person ("I"), like the player telling a short story about himself.
- **Order:** the three clues follow the career in time order. Clue 1 is the early or little-known
  part, clue 3 the famous moves.
- **Plain sentences:** one or two short sentences per clue. Simple time links ("A year later",
  "In 2017") are fine. No dramatic lines ("Then the East called", "shocked Europe").
- **Clue 1 (3 points):** hard. Two lesser-known facts (youth team, first club, early record,
  off-pitch event). No famous club, teammate or record holder by name: hint instead ("a record
  that had stood since 1958").
- **Clue 2 (2 points):** a well-known moment or record from the middle of the career. Avoid naming
  the country when it gives the answer away.
- **Clue 3 (1 point):** a casual fan gets it: the big clubs, the big transfer, the nationality.
- **Never:** the player's name, any part of it, nicknames, or a place that contains the name
  (Pato was born in Pato Branco).
- **Facts:** only facts that can be checked. Club moves, years and fees come from `facts.jsonl`
  (fees in euros). Anything else goes in `notes` as `verify: …`.
- **Length:** 10–160 characters per clue, English and Arabic.
- **Arabic:** the same story in natural Modern Standard Arabic, not word-for-word.

### Example (Oscar)

1. As a teenager, I went to court to leave my first club. A year later, I scored a hat-trick in the U-20 World Cup final.
2. I scored twice against Juventus on my Champions League debut. Two years later, I scored my country's only goal in a 7–1 World Cup defeat.
3. In 2017, I left Chelsea and moved to China for €60m.
