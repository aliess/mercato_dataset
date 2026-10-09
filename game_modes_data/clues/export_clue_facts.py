"""Exports the facts behind Three Clues, most valuable players first.

Writes game_modes_data/clues/facts.jsonl: one player per line with the profile fields and the
club path from game_modes_data/transfer_history/output/ (build that first), for writing clues against.

    python game_modes_data/clues/export_clue_facts.py            # every player
    python game_modes_data/clues/export_clue_facts.py --top 500  # only the top 500 by market value
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

CLUES = Path(__file__).resolve().parent
OUTPUT = CLUES.parent / "transfer_history" / "output"

SKIP_CLUBS = {"Without Club", "Retired", "Career break", "Unknown"}


def fee_text(value):
    if not value:
        return None
    fee = float(value)
    if fee <= 0:
        return "free"
    return f"€{fee / 1e6:g}m"


def difficulty(market_value):
    """Same thresholds as the app (Difficulty.swift) and the server (difficultyForValue)."""
    if market_value >= 100_000_000:
        return "beginner"
    return "intermediate" if market_value >= 40_000_000 else "expert"


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
    args = parser.parse_args()

    transfers = defaultdict(list)
    with open(OUTPUT / "transfer_history.csv") as f:
        for r in csv.DictReader(f):
            transfers[r["player_id"]].append(r)

    with open(OUTPUT / "player_profiles.csv") as f:
        players = [p for p in csv.DictReader(f) if p["market_value"]]
    players.sort(key=lambda p: -float(p["market_value"]))

    picked = players[:args.top] if args.top else players

    CLUES.mkdir(exist_ok=True)
    with open(CLUES / "facts.jsonl", "w") as out:
        for rank, p in enumerate(picked, 1):
            out.write(json.dumps({
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
                "difficulty": difficulty(float(p["market_value"])),
                "career": career(transfers[p["player_id"]]),
            }, ensure_ascii=False) + "\n")
    print(f"Wrote {len(picked)} players to {CLUES / 'facts.jsonl'}")


if __name__ == "__main__":
    main()
