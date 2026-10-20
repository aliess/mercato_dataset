# Starting XI

Name the eleven starters of a famous match. Only the players who started count; substitutes
and the rest of the squad are never answers.

```bash
python game_modes_data/starting_xi/build_lineups.py     # → output/xi_lineups.json (kept in git)
```

Everything comes from the Transfermarkt API (`sources/tm_api.py`): the fixture list of each
competition season gives the match ids, the match record gives the starters, shirt numbers,
captain, formation and score, and the player records give the names and the fame behind the
difficulty. A full build is about
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

By how many of the eleven starters are stars (`sources/fame.py`, plan §8 item 115). The team
and the round do not matter.

- A starter is a **star** when his fame now is €40m or more (`fame.fame_now`): his
  Transfermarkt peak value (from his current API profile) × the growth of Transfermarkt values
  since the year of that peak, at most ×2.5, or his legend value from
  `sources/legends/legends.json` when that is higher (legends are not adjusted). The match year
  adds nothing: the value is already in today's money. Rules and growth table in
  `../../README.md` "Fame and difficulty tiers".
- `beginner`: match year 2017 or later with 10+ stars.
- `intermediate`: 5+ stars, any year.
- `expert`: the rest.

The build also writes `output/xi_stars.json` (each lineup's star count and the lines); the
publisher turns it into the `lineups` of the admin panel's `cache/tiers_v1.json`, which the app never downloads.

Examples: Liverpool in the 2026 quarter-final, England at Euro 2024 and Portugal at the 2022
World Cup are `beginner`; Belgium 2014–2021, the Netherlands, Croatia 2018 and 2022, Portugal
at Euro 2016 and Switzerland 2026 (5 stars) are `intermediate`; Russia, Morocco and Greece are
`expert`.

The current build has 158 `beginner`, 717 `intermediate` and 245 `expert` lineups; the Daily
XI pool (famous and not `expert`) is 143: the matches of `famous_matches.json`, every final, and
the Beginner semi-finals (one leg per tie, the second). The build prints how many lineups have each number
of stars. A re-split must keep every lineup id: the dailies already written reference them.

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
            "captain": false, "x": 0.5, "y": 0.9, "nation": "Poland", "born": "1973-03-23"}]}
```

`nation` and `born` are for the hints (`../../../CONTRACTS.md` §3.2), from
`sources/player_facts.py`; both can be null:

- **`nation`**: the player's citizenship, first country, spelled as the game data's
  `citizenship`. The game data's value for players who are in it, else the API profile's main
  nationality, named through the same map as Grid (`sources/country_ids.json`, written by the
  Grid build). Not his team's country: a slot can be a club side's foreign player.
- **`born`**: date of birth, `YYYY-MM-DD` (game data, else API, else the dataset). The app
  works out the age at the match `date`.

In the current build 12,320 of 12,320 slots have a `nation` and 12,313 a `born` (Magath,
Riedl and Ernst have no date in any source). The player records are already fetched for the
names, so these cost no extra requests. After a change to the slot fields only, rebuild with
`build_lineups.py --from-cache` (uses the cache however old, gives the same lineups).

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
