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
| `beginner` | The eleven most famous clubs, 2017 or later | The six most famous nations, 2017 or later |
| `intermediate` | The big clubs from 2000 on, when not `beginner` | The top five nations and the next fifteen from 2000 on, when not `beginner` |
| `expert` | Every other club, and any club before 2000 | Every other nation, and any nation before 2000 |

The year is the year the match was played, so "2017 or later" is the 2016/17 knockout rounds
on: the last ten seasons. Euro 2016 is out; the 2018, 2022 and 2026 World Cups and the 2021
and 2024 Euros are in.

- **Most famous clubs** (`BEGINNER_CLUB_IDS`): Real Madrid, Barcelona, Bayern, Man City,
  Liverpool, PSG, Chelsea, Arsenal, Atlético, Man Utd, Juventus.
- **Most famous nations** (`BEGINNER_NATION_IDS`): Brazil, Argentina, France, Germany, Spain,
  England.
- **Big clubs** (`BIG_CLUB_IDS`): Real Madrid, Barcelona, Atlético, Sevilla, Valencia; Man Utd,
  Liverpool, Chelsea, Arsenal, Man City; Juventus, AC Milan, Inter, Roma, Napoli; Bayern,
  Dortmund; PSG.
- **Top five nations** (`TOP_NATION_IDS`): Brazil, Argentina, France, Germany, Spain.
- **Next fifteen nations** (`KNOWN_NATION_IDS`): Italy, England, Netherlands, Portugal, Croatia,
  Belgium, Czechia, Denmark, Türkiye, Sweden, Greece, Russia, Uruguay, Switzerland, Morocco.

Every `beginner` team is also on one of the three lists below it, so a lineup that is not
`beginner` falls to `intermediate`, never to `expert`. To make Beginner wider or narrower,
change the two `BEGINNER_` lists or `RECENT_FROM`; the other lists also decide Intermediate.

Examples: Bayern and PSG in the 2020 final and England in the Euro 2024 final are `beginner`;
Dortmund in the 2024 final, Real Madrid in the 2014 final, Liverpool in the 2005 final and
Portugal in the Euro 2016 final are `intermediate`; France in the 1998 World Cup final, Man
Utd in the 1999 final and Porto in the 2004 final are `expert`.

The current build has 227 `beginner`, 522 `intermediate` and 371 `expert` lineups.

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

## Pitch places

`x` (0 left … 1 right) and `y` (0.9 goalkeeper … 0.1 front row) come from the match formation:
the starters are sorted by how deep their position plays (centre-forward deepest in the
order, then second striker, then wingers, so a 4-2-3-1 gives LW AM RW behind the CF and a
4-3-3 gives LW CF RW in one row), central players before wide ones at the same depth, and
cut into the formation's rows. Each row goes left to right by position side; a wide player
alone in a row stays on his wing (x 0.2 or 0.8). After a build, run

```bash
python game_modes_data/starting_xi/check_positions.py   # exits 1 if a wide player is central or on the wrong side
```

## Known limits

- When the API's positions don't fit the formation (one wide midfielder in a five-man back
  line's 3, say), a row can have players on one side only; 14 lineups are like that.

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
