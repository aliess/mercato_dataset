#!/usr/bin/env python3
"""
Build the game files the app downloads and upload them to Firebase Storage (cache/).
Game data never passes through Firestore: this script does not touch it.

  cache/game_data_v1.json   game_modes_data/transfer_history/output/player_profiles.csv, transfer_history.csv
  cache/clues_v1.json       game_modes_data/clues/player_clues.json (approved sets that pass the validator)
  cache/xi_v1.json          game_modes_data/starting_xi/output/xi_lineups.json
  cache/grid_pool_v1.json   game_modes_data/grid/output/grid_pool.json
  cache/manifest.json       version (SHA-256), size and last change of the four files

The files hold no build time, so the same inputs always give the same bytes and the same
version. Shapes are in ../CONTRACTS.md §1.

Usage (from this folder):
    python publish_files.py --project dev            # dry run: build into publish/, compare with what is live
    python publish_files.py --project dev --apply    # upload the files that changed, then the manifest
    python publish_files.py --project prod --apply

A dry run makes public HTTP GETs only (the live manifest, and the live copy of each file that
would change, to describe the difference; --no-diff skips those). It writes nothing remotely.

--apply needs a service-account key: --credentials key.json, or the
<project-id>-firebase-adminsdk-*.json in this folder, or GOOGLE_APPLICATION_CREDENTIALS.
The clue file needs node and the compiled functions next door
(cd ../footballquiz_firebase/functions && npm install && npm run build).
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sources.last_updated import record  # noqa: E402
from sources.tiers import TIERS, difficulty_for_value  # noqa: E402

PROJECTS = {'dev': 'football-quiz-32eb9', 'prod': 'mercato-6e710'}
MODES = ROOT / 'game_modes_data'
PUBLISH_DIR = ROOT / 'publish'
PREFIX = 'cache/'
MANIFEST = 'manifest.json'
DATA_CACHE_CONTROL = 'public, max-age=300'
MANIFEST_CACHE_CONTROL = 'no-cache'
SHRINK_LIMIT = 0.8  # stop when a file would lose more than 20% of its size

# How the CSV cells are typed in game_data_v1.json. The app decodes them this way
# (footballquiz/Models/Player.swift), so these stay as they were when Firestore held the data.
INT_FIELDS = {'player_id', 'current_club_id', 'market_value', 'player_agent_id', 'on_loan_from_club_id'}
NUMBER_FIELDS = {'height'}
DATE_FIELDS = {'date_of_birth', 'joined', 'contract_expires', 'date_of_last_contract_extension',
               'contract_there_expires', 'date_of_death'}

XI_MATCH_FIELDS = ['id', 'team', 'opponent', 'competition', 'round', 'leg', 'season', 'year', 'date',
                   'score', 'formation', 'difficulty', 'famous', 'slots']
XI_SLOT_FIELDS = ['player_id', 'name', 'number', 'position', 'captain', 'x', 'y']
GRID_CLUB_FIELDS = ['id', 'name', 'country', 'league', 'top5_seasons', 'players']


def bucket_name(project_id):
    """The project's default bucket, the one in the app's GoogleService-Info plists."""
    return f'{project_id}.firebasestorage.app'


# ── Values ──

def number(text):
    """'184.0' → 184, '1.5' → 1.5: whole numbers are written without a decimal point, as
    JavaScript wrote them, because the app decodes height as an Int."""
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f'not a finite number: {text!r}')
    return int(value) if value == int(value) else value


def epoch_seconds(text):
    """'1998-12-20 00:00:00' → seconds since 1970 at midnight UTC of that day."""
    day = datetime.strptime(text[:10], '%Y-%m-%d').replace(tzinfo=timezone.utc)
    return int(day.timestamp())


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def encode(content):
    """The exact bytes that are published and hashed: compact JSON, UTF-8, keys in the order built."""
    return json.dumps(content, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def compress(body):
    """Gzip with no time or file name in the header, so the same bytes compress the same way."""
    return gzip.compress(body, compresslevel=9, mtime=0)


def sha256(body):
    return hashlib.sha256(body).hexdigest()


# ── game_data_v1.json ──

def player_record(row):
    """One profile row → the object the app decodes. Empty cells are left out."""
    player = {'id': str(int(float(row['player_id'])))}
    for field, cell in row.items():
        if cell is None or cell == '':
            continue
        if field in INT_FIELDS:
            player[field] = int(float(cell))
        elif field in NUMBER_FIELDS:
            player[field] = number(cell)
        elif field in DATE_FIELDS:
            # The shape a Firestore Timestamp had in this file; the app reads `_seconds`.
            player[field] = {'_seconds': epoch_seconds(cell), '_nanoseconds': 0}
        else:
            player[field] = cell
    return player


def order_same_day(moves, came_from):
    """Two moves on one day (a loan ends and the player is sold): put first the one that starts
    where the player was. `came_from` is the club he was at before that day."""
    remaining, ordered, at = list(moves), [], came_from
    while remaining:
        chosen = next((m for m in remaining if at is not None and m['from_id'] == at), None)
        if chosen is None:  # start with a move that no other move of the day leads into
            chosen = next((m for m in remaining
                           if not any(o is not m and o['to_id'] == m['from_id'] for o in remaining)),
                          remaining[0])
        ordered.append(chosen)
        remaining.remove(chosen)
        at = chosen['to_id']
    return ordered


def build_game_data():
    output = MODES / 'transfer_history' / 'output'
    players = [player_record(row) for row in read_csv(output / 'player_profiles.csv')]
    players.sort(key=lambda p: p['id'])  # by id as text, the order the file always had
    if len({p['id'] for p in players}) != len(players):
        sys.exit('player_profiles.csv has the same player_id twice.')

    by_player, used_ids, dropped = {}, set(), 0
    for row in read_csv(output / 'transfer_history.csv'):
        if not row['player_id'] or not row['transfer_date'] or not row['to_team_name']:
            dropped += 1
            continue
        player_id = str(int(float(row['player_id'])))
        day = row['transfer_date'][:10]
        from_id = str(int(float(row['from_team_id']))) if row['from_team_id'] else None
        to_id = str(int(float(row['to_team_id']))) if row['to_team_id'] else None
        transfer_id = f"{player_id}_{day.replace('-', '')}_{from_id or ''}_{to_id or ''}"
        while transfer_id in used_ids:  # the same move twice on one day
            transfer_id += '_'
        used_ids.add(transfer_id)
        by_player.setdefault(player_id, []).append({
            'id': transfer_id,
            'date': epoch_seconds(day) * 1000,
            'season': row['season_name'] or None,
            'from_id': from_id,
            'to_id': to_id,
            'from': row['from_team_name'] or None,
            'to': row['to_team_name'],
            'fee': number(row['transfer_fee']) if row['transfer_fee'] else None,
            'value': number(row['value_at_transfer']) if row['value_at_transfer'] else None,
            'from_country': row['from_team_country'] or None,
            'to_country': row['to_team_country'] or None,
        })

    transfers = {}
    for player_id in sorted(by_player, key=int):
        moves = sorted(by_player[player_id], key=lambda m: m['date'])  # stable: oldest first
        ordered, start = [], 0
        while start < len(moves):
            end = start
            while end < len(moves) and moves[end]['date'] == moves[start]['date']:
                end += 1
            same_day = moves[start:end]
            if len(same_day) > 1:
                same_day = order_same_day(same_day, ordered[-1]['to_id'] if ordered else None)
            ordered += same_day
            start = end
        transfers[player_id] = ordered

    transfer_count = sum(len(moves) for moves in transfers.values())
    content = {'version': 1, 'player_count': len(players), 'transfer_count': transfer_count,
               'players': players, 'transfers': transfers}
    if not players or not transfer_count:
        sys.exit('The game data would be empty. Build the transfer history first.')
    for transfer in (t for moves in transfers.values() for t in moves):
        if not transfer['to'] or not isinstance(transfer['date'], int):
            sys.exit(f'Bad transfer row: {transfer}')
    known = {p['id'] for p in players}
    counts = {'players': len(players), 'transfers': transfer_count}
    notes = []
    if dropped:
        notes.append(f'{dropped:,} transfer rows dropped (no player, date or destination)')
    orphans = [pid for pid in transfers if pid not in known]
    if orphans:
        notes.append(f'{len(orphans):,} players have transfers but no profile')
    without = sum(1 for p in players if p['id'] not in transfers)
    if without:
        notes.append(f'{without:,} players have no transfers')
    return content, counts, notes


# ── clues_v1.json ──

def clue_version(pairs):
    """An integer below 2^53 from the set's six strings: it changes only when the text does."""
    return int(sha256(encode(pairs))[:13], 16)


def build_clues(game_data):
    subjects = [{key: player.get(key) for key in
                 ('id', 'player_name', 'name_in_home_country', 'current_club_name', 'market_value')}
                for player in game_data['players']]
    script = MODES / 'clues' / 'publishable_clues.js'
    try:
        result = subprocess.run(['node', str(script)], input=json.dumps(subjects), capture_output=True,
                                text=True, encoding='utf-8')
    except FileNotFoundError:
        sys.exit('node is not installed; the clue file needs it to run the validator.')
    if result.returncode != 0:
        sys.exit(f'Clue build failed:\n{result.stderr.strip()[-2000:]}')
    picked = json.loads(result.stdout)

    clues = {}
    for player_id in sorted(picked['clues'], key=int):
        pairs = picked['clues'][player_id]
        if len(pairs) != 3 or any(len(pair) != 2 or not all(isinstance(s, str) and s for s in pair)
                                  for pair in pairs):
            sys.exit(f'Clue set {player_id} does not have three [en, ar] pairs.')
        clues[player_id] = {'v': clue_version(pairs), 'c': pairs}
    if not clues:
        sys.exit('The clue file would be empty.')
    # Tiers by the dataset's own lines (sources/tiers.py), so the file does not depend on which
    # version of the functions is compiled next door.
    values = {str(player['id']): player.get('market_value') for player in game_data['players']}
    by_difficulty = dict.fromkeys(TIERS, 0)
    for player_id in clues:
        by_difficulty[difficulty_for_value(values[player_id])] += 1
    content = {'version': 1, 'count': len(clues), 'by_difficulty': by_difficulty, 'clues': clues}
    counts = {'clue_sets': len(clues), **by_difficulty}
    notes = [f'{count:,} sets left out: {reason.replace("_", " ")}'
             for reason, count in picked['skipped'].items() if count]
    warned = sum(1 for problem in picked['problems'] if problem['published'])
    if warned:
        notes.append(f'{warned:,} published sets have validator warnings')
    theirs = {tier: picked['by_difficulty'][tier] for tier in TIERS}
    if theirs != by_difficulty:
        notes.append(f'the compiled functions next door count the tiers as {theirs}: '
                     'their difficultyForValue has other lines than sources/tiers.py')
    return content, counts, notes


# ── xi_v1.json and grid_pool_v1.json ──

def build_xi():
    content = json.loads((MODES / 'starting_xi' / 'output' / 'xi_lineups.json').read_text(encoding='utf-8'))
    if list(content) != ['version', 'matches'] or not content['matches']:
        sys.exit('xi_lineups.json: expected {version, matches}.')
    for match in content['matches']:
        if list(match) != XI_MATCH_FIELDS:
            sys.exit(f"xi_lineups.json: lineup {match.get('id')} has fields {list(match)}")
        if len(match['slots']) != 11 or any(list(slot) != XI_SLOT_FIELDS for slot in match['slots']):
            sys.exit(f"xi_lineups.json: lineup {match['id']} does not have 11 complete slots.")
    if len({match['id'] for match in content['matches']}) != len(content['matches']):
        sys.exit('xi_lineups.json has the same lineup id twice.')
    matches = {match['id'].rsplit('-', 1)[0] for match in content['matches']}
    return content, {'lineups': len(content['matches']), 'matches': len(matches)}, []


def build_grid():
    content = json.loads((MODES / 'grid' / 'output' / 'grid_pool.json').read_text(encoding='utf-8'))
    if list(content) != ['version', 'first_season', 'player_fields', 'clubs', 'players']:
        sys.exit(f'grid_pool.json: unexpected top-level fields {list(content)}')
    if not content['clubs'] or not content['players']:
        sys.exit('grid_pool.json is empty.')
    for club in content['clubs']:
        if list(club) != GRID_CLUB_FIELDS:
            sys.exit(f'grid_pool.json: club {club} has unexpected fields.')
    club_ids = {club['id'] for club in content['clubs']}
    for row in content['players']:
        ok = (len(row) == 5 and isinstance(row[0], str) and isinstance(row[1], str)
              and isinstance(row[3], int) and isinstance(row[4], list) and row[4]
              and all(len(spell) == 3 and spell[0] in club_ids for spell in row[4]))
        if not ok:
            sys.exit(f'grid_pool.json: bad player row {row}')
    return content, {'players': len(content['players']), 'clubs': len(content['clubs'])}, []


# file name → (mode folder for LAST_UPDATED.json, which counts go there)
FILES = {
    'game_data_v1.json': ('transfer_history', ('players', 'transfers')),
    'clues_v1.json': ('clues', ('clue_sets',)),
    'xi_v1.json': ('starting_xi', ('lineups', 'matches')),
    'grid_pool_v1.json': ('grid', ('players', 'clubs')),
}


def build_all():
    """{file name: {content, body, gzip, version, counts, notes}} for the four data files."""
    game_data, game_counts, game_notes = build_game_data()
    built = {
        'game_data_v1.json': (game_data, game_counts, game_notes),
        'clues_v1.json': build_clues(game_data),
        'xi_v1.json': build_xi(),
        'grid_pool_v1.json': build_grid(),
    }
    files = {}
    for name, (content, counts, notes) in built.items():
        body = encode(content)
        files[name] = {'content': content, 'body': body, 'gzip': compress(body), 'version': sha256(body),
                       'counts': counts, 'notes': notes}
    return files


# ── What is live ──

def public_url(bucket, name):
    return f'https://firebasestorage.googleapis.com/v0/b/{bucket}/o/{PREFIX.replace("/", "%2F")}{name}?alt=media'


def public_get(bucket, name):
    """The live file's bytes by plain HTTP (cache/ is public read), or None when it is not there.
    Any other failure stops the run: a network error must not look like 'nothing is live'."""
    request = urllib.request.Request(public_url(bucket, name), headers={'Cache-Control': 'no-cache'})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        sys.exit(f'Could not read {name} from {bucket}: HTTP {error.code}')
    except OSError as error:
        sys.exit(f'Could not read {name} from {bucket}: {error}')
    return gzip.decompress(body) if body[:2] == b'\x1f\x8b' else body


def live_manifest(bucket):
    body = public_get(bucket, MANIFEST)
    if body is None:
        return None
    try:
        manifest = json.loads(body)
        assert isinstance(manifest['files'], dict)
    except (ValueError, KeyError, AssertionError):
        sys.exit(f'The live {MANIFEST} in {bucket} is not readable. Look at it before publishing.')
    return manifest


def names(ids, name_of, limit=12):
    listed = ', '.join(name_of(i) for i in ids[:limit])
    return listed + (f', … (+{len(ids) - limit:,})' if len(ids) > limit else '')


def describe_difference(name, live, new):
    """Lines saying what differs in content between the live file and the new one."""
    lines = []
    if name == 'game_data_v1.json':
        old_players = {p['id']: p for p in live['players']}
        new_players = {p['id']: p for p in new['players']}
        added = sorted(set(new_players) - set(old_players), key=lambda i: (-new_players[i].get('market_value', 0), int(i)))
        removed = sorted(set(old_players) - set(new_players), key=lambda i: (-old_players[i].get('market_value', 0), int(i)))
        both = set(old_players) & set(new_players)
        changed = [i for i in both if old_players[i] != new_players[i]]
        moved = [i for i in changed if old_players[i].get('current_club_id') != new_players[i].get('current_club_id')]
        lines.append(f"players: {len(old_players):,} live → {len(new_players):,}; {len(added):,} added, "
                     f"{len(removed):,} removed, {len(changed):,} with a changed profile "
                     f"({len(moved):,} changed club)")
        if added:
            lines.append('  added: ' + names(added, lambda i: new_players[i].get('player_name', i)))
        if removed:
            lines.append('  removed: ' + names(removed, lambda i: old_players[i].get('player_name', i)))

        def moves(data, player_id):  # without ids: old files carry Firestore document ids
            return [{k: v for k, v in t.items() if k != 'id'} for t in data['transfers'].get(player_id, [])]
        different = [i for i in both if moves(live, i) != moves(new, i)]
        reordered = [i for i in different
                     if sorted(map(json.dumps, moves(live, i))) == sorted(map(json.dumps, moves(new, i)))]
        lines.append(f"transfers: {live.get('transfer_count', 0):,} live → {new['transfer_count']:,}; "
                     f"{len(different):,} of the {len(both):,} players in both have a different list "
                     f"({len(reordered):,} of them only in the order of same-day moves)")
    elif name == 'clues_v1.json':
        old, new_sets = live['clues'], new['clues']
        added = [i for i in new_sets if i not in old]
        removed = [i for i in old if i not in new_sets]
        changed = [i for i in new_sets if i in old and old[i]['c'] != new_sets[i]['c']]
        test = sum(1 for entry in old.values() if any('[TEST]' in pair[0] for pair in entry['c']))
        lines.append(f"clue sets: {len(old):,} live → {len(new_sets):,}; {len(added):,} added, "
                     f"{len(removed):,} removed, {len(changed):,} with changed text"
                     + (f'; {test:,} live sets are [TEST] mock clues' if test else ''))
        lines.append(f"  by difficulty: {live.get('by_difficulty')} live → {new['by_difficulty']}")
    elif name == 'xi_v1.json':
        old = {m['id']: m for m in live['matches']}
        new_matches = {m['id']: m for m in new['matches']}
        changed = sum(1 for i in new_matches if i in old and old[i] != new_matches[i])
        lines.append(f"lineups: {len(old):,} live → {len(new_matches):,}; "
                     f"{len(set(new_matches) - set(old)):,} added, {len(set(old) - set(new_matches)):,} removed, "
                     f"{changed:,} changed")
    elif name == 'grid_pool_v1.json':
        old = {row[0]: row for row in live['players']}
        new_rows = {row[0]: row for row in new['players']}
        changed = sum(1 for i in new_rows if i in old and old[i] != new_rows[i])
        lines.append(f"players: {len(old):,} live → {len(new_rows):,}; "
                     f"{len(set(new_rows) - set(old)):,} added, {len(set(old) - set(new_rows)):,} removed, "
                     f"{changed:,} changed; clubs: {len(live['clubs']):,} live → {len(new['clubs']):,}")
    return lines


# ── Upload ──

def find_credentials(project_id, given):
    """Path of the service-account key for this project, or None to use the environment's default."""
    path = given
    if not path and not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        path = next(iter(sorted(ROOT.glob(f'{project_id}-firebase-adminsdk-*.json'))), None)
    if path:
        key_project = json.loads(Path(path).read_text()).get('project_id')
        if key_project != project_id:
            sys.exit(f'{Path(path).name} is for project {key_project}, not {project_id}.')
    return path


def storage_bucket(project_id, credentials_path):
    try:
        from google.cloud import storage
    except ImportError:
        sys.exit('google-cloud-storage is not installed: pip install -r requirements.txt')
    if credentials_path:
        client = storage.Client.from_service_account_json(str(credentials_path), project=project_id)
    else:
        client = storage.Client(project=project_id)
    return client.bucket(bucket_name(project_id))


def upload_data_file(bucket, name, gzipped_body):
    """A data file: stored gzip-compressed, served as JSON, cached for five minutes."""
    blob = bucket.blob(PREFIX + name)
    blob.content_encoding = 'gzip'
    blob.cache_control = DATA_CACHE_CONTROL
    blob.upload_from_string(gzipped_body, content_type='application/json')
    return blob


def upload_manifest(bucket, body):
    """The manifest: last, uncompressed, never cached."""
    blob = bucket.blob(PREFIX + MANIFEST)
    blob.cache_control = MANIFEST_CACHE_CONTROL
    blob.upload_from_string(body, content_type='application/json')
    return blob


def check_stored(bucket, name, version):
    """Read the object back with the key (no cache in between) and compare it with what was built."""
    blob = bucket.get_blob(PREFIX + name)
    if blob is None:
        sys.exit(f'{name} is not in the bucket after the upload. The manifest was not changed.')
    # Read the metadata before downloading: the download replaces these with response headers.
    metadata = (blob.content_encoding, blob.content_type, blob.cache_control)
    stored = blob.download_as_bytes(raw_download=True)
    if metadata != ('gzip', 'application/json', DATA_CACHE_CONTROL) \
            or sha256(gzip.decompress(stored)) != version:
        sys.exit(f'{name} in the bucket is not what was built. The manifest was not changed.')


# ── Main ──

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project', required=True, choices=sorted(PROJECTS), help='dev or prod')
    parser.add_argument('--apply', action='store_true', help='Upload (default is a dry run).')
    parser.add_argument('--credentials', type=Path, help='Service account JSON key (only for --apply).')
    parser.add_argument('--no-diff', action='store_true',
                        help="Don't download the live files to describe what changes.")
    parser.add_argument('--allow-shrink', action='store_true',
                        help='Allow a file to lose more than 20%% of its size (safety stop otherwise).')
    return parser.parse_args()


def main():
    args = parse_args()
    project_id = PROJECTS[args.project]
    bucket_id = bucket_name(project_id)

    files = build_all()
    PUBLISH_DIR.mkdir(exist_ok=True)
    for name, built in files.items():
        (PUBLISH_DIR / name).write_bytes(built['body'])

    live = live_manifest(bucket_id)
    live_files = (live or {}).get('files', {})
    now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    manifest = {'generated_at': now, 'files': {}}
    shrunk = []
    print(f"{'Dry run' if not args.apply else 'Publishing'}: {args.project} ({project_id}), bucket {bucket_id}")
    print(f'Live {MANIFEST}: ' + (f"generated {live.get('generated_at')}" if live else 'none yet'))
    for name, built in files.items():
        before = live_files.get(name)
        built['status'] = 'new' if not before else \
            'unchanged' if before.get('version') == built['version'] else 'changed'
        manifest['files'][name] = {
            'version': built['version'], 'bytes': len(built['body']), 'gzip_bytes': len(built['gzip']),
            'updated_at': before.get('updated_at', now) if built['status'] == 'unchanged' else now,
        }
        counts = ', '.join(f'{value:,} {key.replace("_", " ")}' for key, value in built['counts'].items())
        print(f"\n{name}: {built['status'].upper()}")
        print(f"  {len(built['body']):,} bytes, {len(built['gzip']):,} gzipped; version {built['version'][:16]}…")
        print(f'  {counts}')
        for note in built['notes']:
            print(f'  note: {note}')
        if before and len(built['body']) < SHRINK_LIMIT * before.get('bytes', 0):
            shrunk.append(name)
            print(f"  ⚠ much smaller than the live file ({before['bytes']:,} bytes)")
        if built['status'] != 'unchanged' and not args.no_diff:
            body = public_get(bucket_id, name)
            if body is None:
                print('  not live yet')
            else:
                try:
                    lines = describe_difference(name, json.loads(body), built['content'])
                except (ValueError, KeyError, TypeError, AttributeError) as error:
                    lines = [f'the live file could not be compared ({type(error).__name__}: {error})']
                for line in lines:
                    print(f'  {line}')
    manifest_body = json.dumps(manifest, indent=1).encode('utf-8')
    (PUBLISH_DIR / MANIFEST).write_bytes(manifest_body)

    to_upload = [name for name, built in files.items() if built['status'] != 'unchanged']
    print(f"\nBuilt in {PUBLISH_DIR.relative_to(ROOT)}/. "
          + (f"To upload: {', '.join(to_upload)}, then {MANIFEST}." if to_upload else 'Nothing to upload.'))
    if shrunk and not args.allow_shrink:
        sys.exit(f"Stopping: {', '.join(shrunk)} would lose more than 20% of its size. "
                 'Check the build, or pass --allow-shrink.')
    if not args.apply:
        print('Dry run: nothing was uploaded. Re-run with --apply to upload (ask the user first).')
        return
    if not to_upload:
        return

    bucket = storage_bucket(project_id, find_credentials(project_id, args.credentials))
    for name in to_upload:
        upload_data_file(bucket, name, files[name]['gzip'])
        check_stored(bucket, name, files[name]['version'])
        print(f'  ✓ uploaded {PREFIX}{name}')
    upload_manifest(bucket, manifest_body)
    print(f'  ✓ uploaded {PREFIX}{MANIFEST}')

    # What a phone sees: the same public URLs the app uses.
    for name in to_upload:
        seen = public_get(bucket_id, name)
        if seen is None or sha256(seen) != files[name]['version']:
            print(f'  ⚠ the public URL of {name} does not serve the new bytes yet '
                  '(it may be cached for up to 5 minutes; check again).')
    for name in to_upload:
        mode, keys = FILES[name]
        record(MODES / mode, f'synced.{project_id}', **{key: files[name]['counts'][key] for key in keys})
    print('\n✓ Done')


if __name__ == '__main__':
    main()
