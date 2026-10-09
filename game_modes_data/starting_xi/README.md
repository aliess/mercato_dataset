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

By team and by year; the lists and steps are constants at the top of `build_lineups.py`.

1. **Team** sets the starting level:

| Level | Clubs | National teams |
|---|---|---|
| `beginner` | Real Madrid, Barcelona, Bayern, Man Utd, Liverpool, Chelsea, Arsenal, Man City, Juventus, AC Milan, Inter, PSG | Brazil, Argentina, France, Germany, Spain, Italy, England, Netherlands, Portugal |
| `intermediate` | Other clubs from the top five leagues | Every other nation |
| `expert` | Clubs from other leagues | — |

2. **Year** makes it harder, because older lineups have to be studied, not remembered:

| Match year | Quarter-finals and semi-finals | Finals and famous matches |
|---|---|---|
| 2010 or later | no change | no change |
| 2000–2009 | one level harder | no change |
| Before 2000 | two levels harder (always `expert`) | one level harder |

So nothing before 2000 is `beginner`. Examples: France in the 1998 World Cup final is
`intermediate`; Dortmund in the 1997 final is `expert`; Real Madrid in a 1996 quarter-final is
`expert`; Barcelona in a 2008 semi-final is `intermediate`; Liverpool in the 2005 final is
`beginner`.

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
