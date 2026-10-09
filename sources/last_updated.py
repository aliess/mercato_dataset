"""
Each game mode folder keeps a LAST_UPDATED.json: when its data was last built, from what,
how much, and when it last went to each Firebase project. The build and sync scripts
write it; it is kept in git so the dates travel with the data.
"""

import json
from datetime import date
from pathlib import Path

FILE_NAME = 'LAST_UPDATED.json'


def record(mode_dir, section, **facts):
    """Set LAST_UPDATED.json[section] = {date: today, **facts}; other sections are kept.

    A dotted section ("synced.mercato-6e710") is nested.
    """
    path = Path(mode_dir) / FILE_NAME
    data = json.loads(path.read_text()) if path.exists() else {}
    target = data
    *parents, last = section.split('.')
    for key in parents:
        target = target.setdefault(key, {})
    target[last] = {'date': date.today().isoformat(), **facts}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
