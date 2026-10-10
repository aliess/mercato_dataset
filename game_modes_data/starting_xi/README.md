# Starting XI

Name the eleven starters of a famous match. Only the players who started count; substitutes
and the rest of the squad are never answers.

```bash
python game_modes_data/starting_xi/build_lineups.py     # → output/xi_lineups.json (kept in git)
```

Everything comes from the Transfermarkt API (`sources/tm_api.py`): the fixture list of each
competition season gives the match ids, the match record gives the starters, shirt numbers,
captain, formation and score, and the player records give the names. A full build is about
150 requests; finished seasons are cached for good, so a rerun only fetches the latest ones.

## Which matches

| Competition | Rounds | From |
|---|---|---|
| Champions League | Quarter-finals, semi-finals, finals | 1992/93 (first season the API has) |
| World Cup | Quarter-finals, semi-finals, finals | 1986 |
| Euros | Quarter-finals, semi-finals, finals | 1988 |
| Famous matches from other rounds | The ids in `famous_matches.json` | — |

Each match gives two lineups, one per team: 1,120 lineups from 561 matches in the current build.
To add a famous match, add its Transfermarkt match id (the number at the end of the match
report URL) to `famous_matches.json` and rebuild. To add a competition, add a row to
`COMPETITIONS` in `build_lineups.py`.

## Difficulty

By team and by year; the lists and years are constants at the top of `build_lineups.py`. The
round does not matter.

| Level | Clubs | National teams |
|---|---|---|
| `beginner` | The big clubs, 2012 or later | The top five nations, 2012 or later |
| `intermediate` | The big clubs, 2000–2011 | The top five nations, 2000–2011; the next fifteen nations, 2000 or later |
| `expert` | Every other club, and any club before 2000 | Every other nation, and any nation before 2000 |

- **Big clubs:** Real Madrid, Barcelona, Atlético, Sevilla, Valencia; Man Utd, Liverpool,
  Chelsea, Arsenal, Man City; Juventus, AC Milan, Inter, Roma, Napoli; Bayern, Dortmund; PSG.
- **Top five nations:** Brazil, Argentina, France, Germany, Spain.
- **Next fifteen nations:** Italy, England, Netherlands, Portugal, Croatia, Belgium, Czechia,
  Denmark, Türkiye, Sweden, Greece, Russia, Uruguay, Switzerland, Morocco.

Examples: Bayern and PSG in the 2020 final are `beginner`; Liverpool in the 2005 final and
Portugal in the Euro 2016 final are `intermediate`; France in the 1998 World Cup final, Man
Utd in the 1999 final and Porto in the 2004 final are `expert`.

The current build has 381 `beginner`, 368 `intermediate` and 371 `expert` lineups.

## File layout

`output/xi_lineups.json` is published unchanged in content as Storage `cache/xi_v1.json` by
`publish_files.py` (run from the repo root; uploading needs the user's yes). The publisher
refuses a lineup without exactly 11 slots or with other fields than the ones below, so adding
or renaming a field means changing `publish_files.py` and `../CONTRACTS.md` too.

Same layout the app reads for the bundled placeholder (`Resources/Labs/xi_placeholder.json`),
with extra fields the app ignores until it uses them:

```json
{"id": "31195-away", "team": "Liverpool", "opponent": "AC Milan",
 "competition": "Champions League final", "round": "final", "leg": null,
 "season": "04/05", "year": 2005, "date": "2005-05-25",
 "score": "AC Milan 3–3 Liverpool (2–3 pens)", "formation": "4-4-1-1",
 "difficulty": "beginner", "famous": true,
 "slots": [{"player_id": "…", "name": "Jerzy Dudek", "number": 1, "position": "Goalkeeper",
            "captain": false, "x": 0.5, "y": 0.9}]}
```

## Known limits

- The API doesn't say which of two centre-backs played left or right, so shirts with the same
  position are in arbitrary order within their row.
- The API's score includes shootout penalties. The builder works out the real score from the
  goals, and shows the shootout separately: "(2–3 pens)".
- 9 lineups have no captain marked and 1 shirt has no number.
- One match is left out because the API has no positions for it (Legia–Panathinaikos 1996).
- Team names are today's ("Germany" for West Germany in 1986).
- The app has to read `cache/xi_v1.json` instead of its bundled 18-lineup placeholder, and show
  the new competition labels ("Champions League semi-final", "World Cup final"…). In the file
  `leg` and a slot's `number` can be null, and the top-level `version` is 2.
