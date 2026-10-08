#!/usr/bin/env python3
"""
Sync output/player_profiles.csv and output/transfer_history.csv into Firestore, writing only
what changed, then rebuild the game data file the app downloads (Storage cache/game_data_v1.json)
by running the functions' buildGameData() locally (needs node + footballquiz_firebase/functions).

  player_profiles_and_value   doc id = player_id
  transfer_history_filtered   matched on (player_id, transfer_date, from_team_id, to_team_id);
                              new docs get the id "<player_id>_<date>_<from>_<to>"

Field types match the original import: ids/market_value as integers, height/fees/values as
doubles, dates as timestamps, empty cells left out.

Credentials (any one):
  --credentials path/to/service-account.json   (Firebase console → Project settings →
                                                Service accounts → Generate new private key)
  GOOGLE_APPLICATION_CREDENTIALS=path/to/key.json
  gcloud auth application-default login

Every --apply first saves both collections to backups/<project>/<time>/.

Usage:
    python scripts/sync_firestore.py --project dev                  # dry run: show the diff
    python scripts/sync_firestore.py --project dev --apply          # backup, write, rebuild game data
    python scripts/sync_firestore.py --project prod --apply
    python scripts/sync_firestore.py --project prod --restore backups/mercato-6e710/<time> --apply
"""

import argparse
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROJECTS = {'dev': 'football-quiz-32eb9', 'prod': 'mercato-6e710'}
FUNCTIONS_DIR = ROOT.parent / 'footballquiz_firebase' / 'functions'

PLAYERS_COLLECTION = 'player_profiles_and_value'
TRANSFERS_COLLECTION = 'transfer_history_filtered'

INT_FIELDS = {'player_id', 'current_club_id', 'market_value', 'player_agent_id',
              'on_loan_from_club_id', 'from_team_id', 'to_team_id'}
FLOAT_FIELDS = {'height', 'value_at_transfer', 'transfer_fee'}
DATE_FIELDS = {'date_of_birth', 'joined', 'contract_expires', 'date_of_last_contract_extension',
               'contract_there_expires', 'date_of_death', 'transfer_date'}
TRANSFER_KEY = ('player_id', 'transfer_date', 'from_team_id', 'to_team_id')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project', required=True, help='dev, prod, or a Firebase project id')
    parser.add_argument('--apply', action='store_true', help='Write changes (default is a dry run).')
    parser.add_argument('--credentials', type=Path, help='Service account JSON key.')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'output')
    parser.add_argument('--skip-rebuild', action='store_true', help="Don't rebuild the game data file.")
    parser.add_argument('--allow-shrink', action='store_true',
                        help='Allow deleting more than 20%% of a collection (safety stop otherwise).')
    parser.add_argument('--rebuild-only', action='store_true', help='Only rebuild the game data file.')
    parser.add_argument('--restore', type=Path, metavar='BACKUP_DIR',
                        help='Put both collections back exactly as in a backup made by --apply.')
    return parser.parse_args()


# ── Values ──

def to_firestore(field, value):
    """CSV cell → Firestore value, or None to leave the field out."""
    if value is None or (isinstance(value, float) and math.isnan(value)) or value == '':
        return None
    if field in INT_FIELDS:
        return int(float(value))
    if field in FLOAT_FIELDS:
        return float(value)
    if field in DATE_FIELDS:
        return datetime.strptime(str(value)[:10], '%Y-%m-%d').replace(tzinfo=timezone.utc)
    return str(value)


def comparable(value):
    """Normalize a Firestore or CSV value so equal data compares equal."""
    if isinstance(value, datetime):
        return value.strftime('%Y-%m-%d')
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return value


def comparable_doc(doc):
    return {k: comparable(v) for k, v in doc.items()}


def csv_docs(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    docs = []
    for record in df.to_dict('records'):
        doc = {field: to_firestore(field, value) for field, value in record.items()}
        docs.append({k: v for k, v in doc.items() if v is not None})
    return docs


def transfer_key(doc):
    return tuple(comparable(doc.get(field)) for field in TRANSFER_KEY)


def transfer_doc_id(doc):
    return f"{doc['player_id']}_{doc['transfer_date']:%Y%m%d}_{doc['from_team_id']}_{doc['to_team_id']}"


# ── Diff ──

def diff_players(target_docs, existing):
    """existing: {doc_id: data}. Returns (sets {doc_id: data}, deletes [doc_id])."""
    target = {str(doc['player_id']): doc for doc in target_docs}
    sets = {doc_id: doc for doc_id, doc in target.items()
            if doc_id not in existing or comparable_doc(existing[doc_id]) != comparable_doc(doc)}
    deletes = [doc_id for doc_id in existing if doc_id not in target]
    return sets, deletes


def diff_transfers(target_docs, existing):
    """Match on TRANSFER_KEY so existing random doc ids are kept."""
    existing_by_key = defaultdict(list)
    for doc_id, data in existing.items():
        existing_by_key[transfer_key(data)].append(doc_id)

    sets, used = {}, set()
    for doc in target_docs:
        candidates = [d for d in existing_by_key.get(transfer_key(doc), []) if d not in used]
        if candidates:
            doc_id = candidates[0]
            used.add(doc_id)
            if comparable_doc(existing[doc_id]) != comparable_doc(doc):
                sets[doc_id] = doc
        else:
            doc_id = transfer_doc_id(doc)
            while doc_id in sets or doc_id in existing:  # same move twice on one day
                doc_id += '_'
            sets[doc_id] = doc
    deletes = [doc_id for doc_id in existing if doc_id not in used]
    return sets, deletes


# ── Firestore ──

def connect(project_id, credentials_path):
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
    except ImportError:
        sys.exit('firebase-admin is not installed: pip install -r requirements.txt')
    try:
        cred = credentials.Certificate(str(credentials_path)) if credentials_path \
            else credentials.ApplicationDefault()
        app = firebase_admin.initialize_app(cred, {'projectId': project_id})
        return firestore.client(app)
    except Exception as error:
        sys.exit(f'Could not get Firestore credentials ({error}).\n'
                 'Pass --credentials <service-account.json> or run `gcloud auth application-default login`.')


def read_collection(db, name):
    return {snap.id: snap.to_dict() for snap in db.collection(name).stream()}


def write(db, collection, sets, deletes):
    ref = db.collection(collection)
    operations = [('set', doc_id, data) for doc_id, data in sets.items()] + \
                 [('delete', doc_id, None) for doc_id in deletes]
    for start in range(0, len(operations), 450):
        batch = db.batch()
        for op, doc_id, data in operations[start:start + 450]:
            if op == 'set':
                batch.set(ref.document(doc_id), data)
            else:
                batch.delete(ref.document(doc_id))
        batch.commit()
        print(f'    {min(start + 450, len(operations)):,}/{len(operations):,}')


def rebuild_game_data(project_id, credentials_path):
    """Run the Cloud Function's own buildGameData() locally (same code as adminRebuildGameData),
    so no admin key is needed. Uses the compiled functions in footballquiz_firebase/functions/lib."""
    env = dict(os.environ,
               GCLOUD_PROJECT=project_id,
               FIREBASE_CONFIG=json.dumps({'projectId': project_id,
                                           'storageBucket': f'{project_id}.firebasestorage.app'}))
    if credentials_path:
        env['GOOGLE_APPLICATION_CREDENTIALS'] = str(Path(credentials_path).resolve())
    script = ("require('./lib/gameData').buildGameData()"
              ".then(r => { console.log(JSON.stringify(r)); process.exit(0); })"
              ".catch(e => { console.error(e); process.exit(1); })")
    result = subprocess.run(['node', '-e', script], cwd=FUNCTIONS_DIR, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f'  ⚠ game data rebuild failed (Firestore is already updated; re-run with --rebuild-only):\n'
                 f'{result.stderr.strip()[-2000:]}')
    print(f'  ✓ game data rebuilt: {result.stdout.strip().splitlines()[-1]}')


# ── Backups ──

def encode(value):
    return {'__timestamp__': value.isoformat()} if isinstance(value, datetime) else value


def decode(value):
    if isinstance(value, dict) and '__timestamp__' in value:
        return datetime.fromisoformat(value['__timestamp__'])
    return value


def backup(project_id, collections):
    """Save {collection: {doc_id: data}} to backups/<project>/<time>/ and return that folder."""
    folder = ROOT / 'backups' / project_id / datetime.now().strftime('%Y-%m-%d_%H%M%S')
    folder.mkdir(parents=True)
    for name, docs in collections.items():
        data = {doc_id: {k: encode(v) for k, v in doc.items()} for doc_id, doc in docs.items()}
        (folder / f'{name}.json').write_text(json.dumps(data, ensure_ascii=False))
    return folder


def restore(db, folder, existing):
    for name in (PLAYERS_COLLECTION, TRANSFERS_COLLECTION):
        saved = json.loads((folder / f'{name}.json').read_text())
        saved = {doc_id: {k: decode(v) for k, v in doc.items()} for doc_id, doc in saved.items()}
        sets = {doc_id: doc for doc_id, doc in saved.items()
                if doc_id not in existing[name] or comparable_doc(existing[name][doc_id]) != comparable_doc(doc)}
        deletes = [doc_id for doc_id in existing[name] if doc_id not in saved]
        print(f'{name}: restoring {len(sets):,} docs, deleting {len(deletes):,}')
        write(db, name, sets, deletes)


# ── Main ──

def describe_player_changes(sets, existing):
    moved = []
    for doc_id, doc in sets.items():
        before = existing.get(doc_id)
        if before and comparable(before.get('current_club_id')) != comparable(doc.get('current_club_id')):
            moved.append(f"{doc.get('player_name')}: {before.get('current_club_name')} → {doc.get('current_club_name')}")
    return moved


def main():
    args = parse_args()
    project_id = PROJECTS.get(args.project, args.project)
    if args.credentials:
        key_project = json.loads(args.credentials.read_text()).get('project_id')
        if key_project != project_id:
            sys.exit(f'{args.credentials.name} is for project {key_project}, not {project_id}.')
    if args.rebuild_only:
        rebuild_game_data(project_id, args.credentials)
        return
    if args.restore:
        db = connect(project_id, args.credentials)
        existing = {name: read_collection(db, name) for name in (PLAYERS_COLLECTION, TRANSFERS_COLLECTION)}
        if not args.apply:
            print(f'Dry run — would restore {project_id} from {args.restore}. Add --apply.')
            return
        restore(db, args.restore, existing)
        if not args.skip_rebuild:
            rebuild_game_data(project_id, args.credentials)
        return

    players = csv_docs(args.output_dir / 'player_profiles.csv')
    transfers = csv_docs(args.output_dir / 'transfer_history.csv')
    print(f'Target: {len(players):,} players, {len(transfers):,} transfers → project {project_id}')

    db = connect(project_id, args.credentials)
    print('Reading Firestore...')
    existing_players = read_collection(db, PLAYERS_COLLECTION)
    existing_transfers = read_collection(db, TRANSFERS_COLLECTION)
    print(f'  {len(existing_players):,} players, {len(existing_transfers):,} transfers in Firestore')

    player_sets, player_deletes = diff_players(players, existing_players)
    transfer_sets, transfer_deletes = diff_transfers(transfers, existing_transfers)
    new_players = sum(1 for d in player_sets if d not in existing_players)
    new_transfers = sum(1 for d in transfer_sets if d not in existing_transfers)

    print(f'\nPlayers:   {new_players:,} new, {len(player_sets) - new_players:,} updated, {len(player_deletes):,} removed')
    print(f'Transfers: {new_transfers:,} new, {len(transfer_sets) - new_transfers:,} updated, {len(transfer_deletes):,} removed')
    moved = describe_player_changes(player_sets, existing_players)
    if moved:
        print(f'\nCurrent club changed for {len(moved):,} players, e.g.:')
        for line in moved[:15]:
            print(f'  {line}')

    for name, deletes, existing in ((PLAYERS_COLLECTION, player_deletes, existing_players),
                                    (TRANSFERS_COLLECTION, transfer_deletes, existing_transfers)):
        if existing and len(deletes) > 0.2 * len(existing) and not args.allow_shrink:
            sys.exit(f'\nStopping: would delete {len(deletes):,} of {len(existing):,} docs in {name}. '
                     'Check the output files, or pass --allow-shrink.')

    if not args.apply:
        print('\nDry run — nothing written. Re-run with --apply to write.')
        return

    folder = backup(project_id, {PLAYERS_COLLECTION: existing_players, TRANSFERS_COLLECTION: existing_transfers})
    print(f'\nBacked up current Firestore data to {folder.relative_to(ROOT)}')
    print(f'  (undo with: python scripts/sync_firestore.py --project {args.project} --restore {folder.relative_to(ROOT)} --apply)')

    print('\nWriting players...')
    write(db, PLAYERS_COLLECTION, player_sets, player_deletes)
    print('Writing transfers...')
    write(db, TRANSFERS_COLLECTION, transfer_sets, transfer_deletes)
    if not args.skip_rebuild:
        print('Rebuilding game data...')
        rebuild_game_data(project_id, args.credentials)
    print('\n✓ Done')


if __name__ == '__main__':
    main()
