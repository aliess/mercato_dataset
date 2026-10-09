# Data status

When each game mode's data was last built, and when it last went to Firebase.
Written by the build and sync scripts from each mode's `LAST_UPDATED.json`; don't edit by hand.

| Game mode | Last built | What the files hold | Source | Dev (`football-quiz-32eb9`) | Prod (`mercato-6e710`) |
|---|---|---|---|---|---|
| Transfer history | 2026-10-09 | 1,825 players; 14,220 transfers; latest transfer 2026-10-08; dataset latest valuation 2026-06-12 | Transfermarkt API | 2026-10-09 (1,897 players; 14,299 transfers) — **behind the files** | 2026-10-08 (1,897 players; 14,773 transfers) — **behind the files** |
| Three Clues | 2026-10-09 | 1,635 clue sets; 1,825 players in pool | written by hand against facts.jsonl | 2026-10-08 (1,698 clue sets) — **behind the files** | 2026-10-08 (1,698 clue sets) — **behind the files** |
| Starting XI | 2026-10-09 | 1,120 lineups; 560 matches; latest match 2026-07-19 | Transfermarkt API | never | never |
| Grid Rush | 2026-10-09 | 45,048 players; 268 clubs; seasons 1990 to 2026 | Transfermarkt API | never | never |

"Behind the files" means the files here were rebuilt after the last sync, or hold different
counts: the app is still serving the older data. "never" means the mode is not in the app's
backend yet.
