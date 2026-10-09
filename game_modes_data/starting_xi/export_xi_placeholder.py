#!/usr/bin/env python3
"""
Builds the Labs Starting XI placeholder lineups bundled in the app
(footballquiz/footballquiz/Resources/Labs/xi_placeholder.json) from the
transfermarkt-datasets match tables (games, game_lineups; ~130 MB, not kept in git).

Usage:
    python game_modes_data/starting_xi/export_xi_placeholder.py <output.json>

The tables are downloaded into sources/dataset/ when missing. Lineups exist for club
matches only (national-team finals have none). Pitch places come from the match
formation; shirts are placed left to right by position.
"""
import csv, gzip, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sources.download_dataset import DATASET_DIR, download

D, OUT = str(DATASET_DIR), sys.argv[1]
download(("games", "game_lineups"), only_missing=True)
# game_id, side ("home"/"away"), competition label, score label (regulation + pens)
PICKS = [
    ("3183105", "away", "Champions League final", 2019, "Tottenham 0–2 Liverpool"),
    ("3183105", "home", "Champions League final", 2019, "Tottenham 0–2 Liverpool"),
    ("2853591", "away", "Champions League final", 2017, "Juventus 1–4 Real Madrid"),
    ("2567502", "away", "Champions League final", 2015, "Juventus 1–3 Barcelona"),
    ("2567502", "home", "Champions League final", 2015, "Juventus 1–3 Barcelona"),
    ("2456475", "home", "Champions League final", 2014, "Real Madrid 4–1 Atlético (a.e.t.)"),
    ("3038747", "away", "Champions League final", 2018, "Real Madrid 3–1 Liverpool"),
    ("4077814", "home", "Champions League final", 2023, "Man City 1–0 Inter"),
    ("4077814", "away", "Champions League final", 2023, "Man City 1–0 Inter"),
    ("4547655", "home", "Champions League final", 2025, "PSG 5–0 Inter"),
    ("4343512", "away", "Champions League final", 2024, "Dortmund 0–2 Real Madrid"),
    ("2690027", "home", "Champions League final", 2016, "Real Madrid 1–1 Atlético (5–3 pens)"),
    ("3812391", "away", "Champions League final", 2022, "Liverpool 0–1 Real Madrid"),
    ("3812391", "home", "Champions League final", 2022, "Liverpool 0–1 Real Madrid"),
    ("3038747", "home", "Champions League final", 2018, "Real Madrid 3–1 Liverpool"),
    ("2456475", "away", "Champions League final", 2014, "Real Madrid 4–1 Atlético (a.e.t.)"),
    ("2853591", "home", "Champions League final", 2017, "Juventus 1–4 Real Madrid"),
    ("4343512", "home", "Champions League final", 2024, "Dortmund 0–2 Real Madrid"),
]
# Pitch depth (0 = own goal) and side (-1 left … 1 right) per Transfermarkt position.
LINE = {
    "Goalkeeper": (0, 0), "Centre-Back": (1, 0), "Left-Back": (1, -1), "Right-Back": (1, 1),
    "Defensive Midfield": (2, 0), "Central Midfield": (3, 0), "Left Midfield": (3, -1),
    "Right Midfield": (3, 1), "Attacking Midfield": (4, 0), "Left Winger": (5, -1),
    "Right Winger": (5, 1), "Second Striker": (5, 0), "Centre-Forward": (5, 0),
}
games = {r["game_id"]: r for r in csv.DictReader(gzip.open(D + "/games.csv.gz", "rt"))}
ids = {p[0] for p in PICKS}
lineups = {}
for r in csv.DictReader(gzip.open(D + "/game_lineups.csv.gz", "rt")):
    if r["game_id"] in ids and r["type"] == "starting_lineup":
        lineups.setdefault((r["game_id"], r["club_id"]), []).append(r)

matches = []
for game_id, side, comp, year, score in PICKS:
    g = games[game_id]
    club_id, team = g[f"{side}_club_id"], g[f"{side}_club_name"]
    players = lineups.get((game_id, club_id), [])
    if len(players) != 11 or any(p["position"] not in LINE for p in players):
        print("skipped", year, team, len(players), {p["position"] for p in players} - set(LINE))
        continue
    # Rows from the match formation ("4-3-3 Attacking" → GK, 4, 3, 3), filled by depth;
    # without one, players are grouped by position line.
    # Only the leading "4-4-2" part: "4-4-2 double 6" names its midfield shape after it.
    shape = re.match(r"\s*(\d(?:-\d)+)", g[f"{side}_club_formation"])
    counts = [int(n) for n in shape.group(1).split("-")] if shape else []
    ordered = sorted(players, key=lambda p: LINE[p["position"]])
    if sum(counts) == 10:
        chunks, start = [ordered[:1]], 1
        for n in counts:
            chunks.append(ordered[start:start + n])
            start += n
    else:
        by_depth = {}
        for p in ordered:
            by_depth.setdefault(LINE[p["position"]][0], []).append(p)
        chunks = [by_depth[d] for d in sorted(by_depth)]
    slots = []
    for i, chunk in enumerate(chunks):
        row = sorted(((LINE[p["position"]][1], p) for p in chunk), key=lambda x: x[0])
        y = 0.9 - 0.8 * i / (len(chunks) - 1)
        for j, (_, p) in enumerate(row):
            x = (j + 1) / (len(row) + 1)
            slots.append({
                "player_id": p["player_id"], "name": p["player_name"],
                "number": int(p["number"]) if p["number"].isdigit() else None,
                "position": p["position"], "captain": p["team_captain"] == "1",
                "x": round(x, 3), "y": round(y, 3),
            })
    formation = "-".join(str(len(c)) for c in chunks[1:])
    short = {"Liverpool FC": "Liverpool", "Tottenham Hotspur": "Tottenham", "FC Barcelona": "Barcelona",
             "Juventus FC": "Juventus", "Manchester City": "Man City", "Inter Milan": "Inter",
             "Paris Saint-Germain": "PSG", "Atlético de Madrid": "Atlético Madrid"}.get(team, team)
    matches.append({
        "id": f"{game_id}-{side}", "team": short, "competition": comp, "year": year,
        "score": score, "formation": formation, "date": g["date"], "slots": slots,
    })
json.dump({"version": 1, "placeholder": True, "matches": matches}, open(OUT, "w"), ensure_ascii=False, indent=1)
for m in matches:
    print(m["year"], m["team"], m["formation"], ", ".join(f"{s['number']} {s['name']}" for s in m["slots"]))
