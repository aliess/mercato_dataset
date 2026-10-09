# Starting XI

Name the eleven starters of a famous match. Still a Labs prototype: the app ships a
placeholder set, and the real data set is not built yet.

## What exists

`export_xi_placeholder.py` writes the 18 Champions League final lineups (2014–2025) bundled
in the app as `footballquiz/footballquiz/Resources/Labs/xi_placeholder.json`:

```bash
python starting_xi/export_xi_placeholder.py ../footballquiz/footballquiz/Resources/Labs/xi_placeholder.json
```

Source: the `games` and `game_lineups` tables of transfermarkt-datasets (~130 MB, downloaded
into `sources/dataset/` on first run). Limits of that source: club matches only (no
national-team finals), nothing before July 2013, and no updates since July 2026.

## Where the real data will come from

The Transfermarkt API (`sources/tm_api.py`) has full lineups for old and national-team
matches (2005 Champions League final and 2010 World Cup final checked):

| Need | Call |
|---|---|
| The matches of a competition season, with game ids | `api.competition_fixtures('CL', 2004)` |
| Starters, shirt numbers, captain, formation, score | `api.games([game_id, …])` → `homeClub.lineup.players`, `tactic`, `score` |
| Player names for the ids in a lineup | `api.players([…])` |

Open questions before building it: which matches to include, and how the app receives them
(a Storage JSON file like the other modes, not Firestore docs per match).
