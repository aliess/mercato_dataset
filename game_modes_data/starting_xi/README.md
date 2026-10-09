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

By team, as a first guess; the lists are constants at the top of `build_lineups.py`.

| Difficulty | Clubs | National teams |
|---|---|---|
| `beginner` | Real Madrid, Barcelona, Bayern, Man Utd, Liverpool, Chelsea, Arsenal, Man City, Juventus, AC Milan, Inter, PSG | Brazil, Argentina, France, Germany, Spain, Italy, England, Netherlands, Portugal |
| `intermediate` | Other clubs from the top five leagues | Every other nation |
| `expert` | Clubs from other leagues | — |

The year is not part of it yet: Bayern's 1999 lineup is `beginner` like their 2020 one. Each
lineup carries `year`, `round` and `famous` (finals and the extra matches), so the split can
be refined without refetching.

## File layout

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
- The app still bundles the 18-lineup placeholder. It has to switch to this file, and to show
  the new competition labels ("Champions League semi-final", "World Cup final"…).
