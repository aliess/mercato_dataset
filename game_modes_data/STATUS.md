# Data status

When each game mode's data was last built, and when it was last uploaded to Firebase Storage.
Written by the build and publish scripts from each mode's `LAST_UPDATED.json`; don't edit by hand.

| Game mode | Last built | What the files hold | Source | Dev (`football-quiz-32eb9`) | Prod (`mercato-6e710`) |
|---|---|---|---|---|---|
| Transfer history | 2026-10-18 | 2,211 players; 17,483 transfers; latest transfer 2026-10-14; 141 beginner; 1,042 intermediate; 1,028 expert; 13 story only | Transfermarkt API | 2026-10-18 (2,211 players; 17,483 transfers) | 2026-10-18 (2,211 players; 17,483 transfers) |
| Three Clues | 2026-10-18 | 1,954 clue sets; 2,211 players in pool | written by hand against facts.jsonl | 2026-10-18 (1,954 clue sets) | 2026-10-18 (1,954 clue sets) |
| Starting XI | 2026-10-20 | 1,120 lineups; 560 matches; latest match 2026-07-19 | Transfermarkt API | 2026-10-20 (1,120 lineups; 560 matches) | 2026-10-20 (1,120 lineups; 560 matches) |
| Grid Rush | 2026-10-16 | 45,057 players; 268 clubs; seasons 1990 to 2026 | Transfermarkt API | 2026-10-16 (45,057 players; 268 clubs) | 2026-10-16 (45,057 players; 268 clubs) |
| Learn | 2026-10-22 | 10 worlds; 106 stops; 781 steps; 2,211 players; 451 lineups; 258 pairs; 189 routes; 2,211 hooks; 1,954 clue hooks; 396 twins; 79 dossiers squad; 140 dossiers era; 192 dossiers night; 8 dossiers route; 16 dossiers nation; 40 dossiers story; 2,197 anchored; 14 no anchor; 827 trails; 89 draft cards; 0 no card | — | 2026-10-22 (106 stops; 2,211 players) | never |

"Behind the files" means the files here were rebuilt after the last upload, or hold different
counts: the app is still getting the older data. "never" means the mode has not been
uploaded to that project yet. `python publish_files.py --project dev|prod` compares exactly.
