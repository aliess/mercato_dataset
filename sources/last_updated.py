"""
Each game mode folder keeps a LAST_UPDATED.json: when its data was last built, from what,
how much, and when it was last uploaded to each Firebase project (`synced`, written by
publish_files.py --apply). The build and publish scripts write it; it is kept in git so the dates travel with the data.

game_modes_data/STATUS.md is the same information as one table, rewritten on every change.
Don't edit either by hand. To rewrite the table: python sources/last_updated.py
"""

import json
from datetime import date
from pathlib import Path

FILE_NAME = 'LAST_UPDATED.json'
MODES_DIR = Path(__file__).resolve().parent.parent / 'game_modes_data'
PROJECTS = {'football-quiz-32eb9': 'Dev', 'mercato-6e710': 'Prod'}
MODES = {'transfer_history': 'Transfer history', 'clues': 'Three Clues',
         'starting_xi': 'Starting XI', 'grid': 'Grid Rush'}


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
    render_status()


def describe(facts):
    """{"players": 1825, "latest_transfer": "2026-10-08"} → "1,825 players; latest transfer 2026-10-08"."""
    parts = []
    for key, value in facts.items():
        if key in ('date', 'source', 'transfers_from'):
            continue
        label = key.replace('_', ' ')
        parts.append(f'{value:,} {label}' if isinstance(value, int) else f'{label} {value}')
    return '; '.join(parts)


def synced_cell(built, synced):
    if not synced:
        return 'never'
    text = synced['date']
    if describe(synced):
        text += f' ({describe(synced)})'
    # Behind when the files were built later, or hold different counts than what was uploaded.
    differs = any(built.get(key) != value for key, value in synced.items() if key != 'date' and key in built)
    if built and (differs or built.get('date', '') > synced['date']):
        text += ' — **behind the files**'
    return text


def render_status():
    lines = [
        '# Data status',
        '',
        'When each game mode\'s data was last built, and when it was last uploaded to Firebase Storage.',
        'Written by the build and publish scripts from each mode\'s `LAST_UPDATED.json`; don\'t edit by hand.',
        '',
        '| Game mode | Last built | What the files hold | Source | Dev (`football-quiz-32eb9`) | Prod (`mercato-6e710`) |',
        '|---|---|---|---|---|---|',
    ]
    for folder, name in MODES.items():
        path = MODES_DIR / folder / FILE_NAME
        data = json.loads(path.read_text()) if path.exists() else {}
        built, synced = data.get('built') or {}, data.get('synced') or {}
        cells = [name, built.get('date', 'not built'), describe(built) or '—',
                 built.get('source') or built.get('transfers_from') or '—']
        cells += [synced_cell(built, synced.get(project)) for project in PROJECTS]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '"Behind the files" means the files here were rebuilt after the last upload, or hold different',
              'counts: the app is still getting the older data. "never" means the mode has not been',
              'uploaded to that project yet. `python publish_files.py --project dev|prod` compares exactly.', '']
    (MODES_DIR / 'STATUS.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    render_status()
