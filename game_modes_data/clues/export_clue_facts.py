"""Exports the facts behind Three Clues, most famous players first.

Writes game_modes_data/clues/facts.jsonl: one player per line with the profile fields and the
club path from game_modes_data/transfer_history/output/ (build that first), for writing clues against.
`market_value` is the player's fame and `difficulty` his tier, both from sources/fame.py as the
game data has them; `story_only` marks a one-club legend (Player Story only).

    python game_modes_data/clues/export_clue_facts.py               # every player
    python game_modes_data/clues/export_clue_facts.py --top 500     # only the top 500 by fame
    python game_modes_data/clues/export_clue_facts.py --legends-todo
        # also writes legends_todo.jsonl: the legends (sources/legends/legends.json) with no
        # approved clue set in player_clues.json, the same facts, to write their clues against
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

CLUES = Path(__file__).resolve().parent
sys.path.insert(0, str(CLUES.parents[1]))

from sources.fame import is_legend  # noqa: E402

OUTPUT = CLUES.parent / "transfer_history" / "output"

SKIP_CLUBS = {"Without Club", "Retired", "Career break", "Unknown"}


def fee_text(value):
    if not value:
        return None
    fee = float(value)
    if fee <= 0:
        return "free"
    return f"€{fee / 1e6:g}m"


def career(rows):
    """Club path, oldest first: [{club, country, from, fee, loan}]."""
    path = []
    for r in sorted(rows, key=lambda r: r["transfer_date"]):
        to = r["to_team_name"]
        if to in SKIP_CLUBS:
            continue
        path.append({
            "club": to,
            "country": r["to_team_country"] or None,
            "from": r["transfer_date"][:4],
            "fee": fee_text(r["transfer_fee"]),
            "loan": "loan" in (r["transfer_type"] or "").lower(),
        })
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--legends-todo", action="store_true")
    args = parser.parse_args()

    transfers = defaultdict(list)
    with open(OUTPUT / "transfer_history.csv") as f:
        for r in csv.DictReader(f):
            transfers[r["player_id"]].append(r)

    with open(OUTPUT / "player_profiles.csv") as f:
        players = [p for p in csv.DictReader(f) if p["market_value"]]
    players.sort(key=lambda p: -float(p["market_value"]))

    picked = players[:args.top] if args.top else players

    facts = []
    for rank, p in enumerate(picked, 1):
        facts.append({
            "rank": rank,
            "player_id": p["player_id"],
            "name": p["player_name"],
            "native_name": p["name_in_home_country"] or None,
            "born": p["date_of_birth"][:10] or None,
            "birthplace": ", ".join(x.strip() for x in (p["place_of_birth"], p["country_of_birth"]) if x.strip()) or None,
            "citizenship": p["citizenship"] or None,
            "position": p["position"] or None,
            "foot": p["foot"] or None,
            "height": p["height"] or None,
            "current_club": p["current_club_name"] or None,
            "market_value": float(p["market_value"]),
            "difficulty": p["difficulty"],
            "active": p["active"] == "True",
            "legend": is_legend(p["player_id"]),
            "story_only": p["story_only"] == "True",
            "career": career(transfers[p["player_id"]]),
        })
    write(CLUES / "facts.jsonl", facts)

    if args.legends_todo:
        clues = json.loads((CLUES / "player_clues.json").read_text())
        approved = {pid for pid, doc in clues.items() if doc.get("status") == "approved"}
        todo = [f for f in facts if f["legend"] and f["player_id"] not in approved]
        write(CLUES / "legends_todo.jsonl", todo)
        print(f"  {sum(f['story_only'] for f in todo)} of them story only (one club)")


def write(path, facts):
    with open(path, "w") as out:
        for fact in facts:
            out.write(json.dumps(fact, ensure_ascii=False) + "\n")
    print(f"Wrote {len(facts)} players to {path}")


if __name__ == "__main__":
    main()
