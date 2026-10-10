# Data status

When each game mode's data was last built, and when it was last uploaded to Firebase Storage.
Written by the build and publish scripts from each mode's `LAST_UPDATED.json`; don't edit by hand.

| Game mode | Last built | What the files hold | Source | Dev (`football-quiz-32eb9`) | Prod (`mercato-6e710`) |
|---|---|---|---|---|---|
| Transfer history | 2026-10-09 | 1,825 players; 14,220 transfers; latest transfer 2026-10-08; dataset latest valuation 2026-06-12 | Transfermarkt API | 2026-10-09 (1,825 players; 14,220 transfers) | 2026-10-09 (1,825 players; 14,220 transfers) |
| Three Clues | 2026-10-10 | 1,635 clue sets; 1,825 players in pool | written by hand against facts.jsonl | 2026-10-10 (1,635 clue sets) | 2026-10-10 (1,635 clue sets) |
| Starting XI | 2026-10-09 | 1,120 lineups; 560 matches; latest match 2026-07-19 | Transfermarkt API | 2026-10-10 (1,120 lineups; 560 matches) | 2026-10-10 (1,120 lineups; 560 matches) |
| Grid Rush | 2026-10-09 | 45,048 players; 268 clubs; seasons 1990 to 2026 | Transfermarkt API | 2026-10-09 (45,048 players; 268 clubs) | 2026-10-09 (45,048 players; 268 clubs) |

"Behind the files" means the files here were rebuilt after the last upload, or hold different
counts: the app is still getting the older data. "never" means the mode has not been
uploaded to that project yet. `python publish_files.py --project dev|prod` compares exactly.
