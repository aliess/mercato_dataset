"""
Small client for Transfermarkt's JSON API (tmapi-alpha.transfermarkt.technology).

The transfermarkt.com pages sit behind an AWS WAF "Human Verification" challenge,
which is why the upstream transfermarkt-datasets scraper stopped in July 2026.
This JSON API (used by Transfermarkt's own apps) is not behind the challenge.

No key or login is needed: the data endpoints answer plain GET requests. Only the
API's documentation page asks for a username and password (the base URL redirects to
/doc, which is what a browser shows). The API is unofficial and undocumented, so
endpoints can change without notice; README.md lists the ones checked by hand.

Responses are cached on disk (sources/cache/tm_api/) so reruns are fast and gentle on the API.
"""

import hashlib
import http.client
import json
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE_URL = 'https://tmapi-alpha.transfermarkt.technology'
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / 'cache' / 'tm_api'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/129.0 Safari/537.36',
    'Accept': 'application/json',
}
BATCH_SIZE = 200  # ids per players?/clubs? request (500 works, 200 keeps URLs short)


class TransfermarktAPI:
    def __init__(self, cache_dir=DEFAULT_CACHE_DIR, max_age_hours=24.0, workers=4, min_interval=0.05):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age = max_age_hours * 3600
        self.workers = workers
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._last_request = 0.0
        self.requests = 0
        self.cache_hits = 0
        self.failed = []

    # ── low level ──

    def _cache_path(self, path):
        key = hashlib.sha1(path.encode()).hexdigest()
        return self.cache_dir / key[:2] / f'{key}.json'

    def _throttle(self):
        with self._lock:
            wait = self._last_request + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()

    def get(self, path, retries=6):
        """GET `path` and return the response's `data`, or None when not found."""
        cache_file = self._cache_path(path)
        if cache_file.exists() and time.time() - cache_file.stat().st_mtime < self.max_age:
            self.cache_hits += 1
            return json.loads(cache_file.read_text())['data']

        url = f'{BASE_URL}/{path}'
        for attempt in range(retries):
            self._throttle()
            try:
                request = urllib.request.Request(url, headers=HEADERS)
                with urllib.request.urlopen(request, timeout=60) as response:
                    body = json.load(response)
                self.requests += 1
                break
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    return None
                if error.code not in (403, 405, 429, 500, 502, 503, 504) or attempt == retries - 1:
                    raise
            except (urllib.error.URLError, http.client.HTTPException, OSError, json.JSONDecodeError):
                if attempt == retries - 1:
                    raise
            time.sleep(min(2 ** attempt, 30) + random.random())

        if not body.get('success'):
            return None
        cache_file.parent.mkdir(exist_ok=True)
        cache_file.write_text(json.dumps({'path': path, 'data': body['data']}))
        return body['data']

    def _get_batched(self, endpoint, ids):
        ids = sorted({str(i) for i in ids})
        chunks = [ids[i:i + BATCH_SIZE] for i in range(0, len(ids), BATCH_SIZE)]
        paths = [f'{endpoint}?' + '&'.join(f'ids[]={urllib.parse.quote(i)}' for i in chunk)
                 for chunk in chunks]
        result = {}
        with ThreadPoolExecutor(self.workers) as pool:
            for data in pool.map(self.get, paths):
                for item in data or []:
                    result[str(item['id'])] = item
        return result

    # ── endpoints ──

    def players(self, player_ids):
        """Player profiles by id: name, birth, position, current club, market values."""
        return self._get_batched('players', player_ids)

    def clubs(self, club_ids):
        """Clubs by id: name, shortName, country, superior (official) club name."""
        return self._get_batched('clubs', club_ids)

    def transfer_histories(self, player_ids, progress=None):
        """Full transfer history (completed + pending) by player id.

        Players whose request kept failing are left out and listed in `self.failed`.
        """
        player_ids = [str(i) for i in player_ids]
        result = {}
        self.failed = []

        def fetch(player_id):
            try:
                return player_id, self.get(f'transfer/history/player/{player_id}')
            except Exception as error:  # one bad player must not stop the run
                self.failed.append((player_id, repr(error)))
                return player_id, None

        with ThreadPoolExecutor(self.workers) as pool:
            for done, (player_id, data) in enumerate(pool.map(fetch, player_ids), 1):
                if data is not None:
                    history = data.get('history') or {}
                    result[player_id] = (history.get('terminated') or []) + (history.get('pending') or [])
                if progress and (done % 250 == 0 or done == len(player_ids)):
                    progress(done, len(player_ids))
        return result

    # Not used by a build yet; these are the endpoints Starting XI and Grid need.
    # Old seasons never change, so pass a large max_age_hours when fetching them.

    def squad(self, club_id, season):
        """First-team squad of a season (season=2005 is 2005/06): playerId, shirtNumber, isCaptain.

        The parameter must be `season`; `seasonId` is ignored and returns the current squad.
        """
        return self.get(f'club/{club_id}/squad?season={season}')

    def games(self, game_ids):
        """Matches by id: both clubs' starting lineups, formation (`tactic`) and score."""
        return self._get_batched('games', game_ids)

    def competition_fixtures(self, competition_id, season):
        """Every match (with game ids) of a competition season, e.g. ('CL', 2004)."""
        return self.get(f'competition/{competition_id}/fixtures?season={season}')

    def club_fixtures(self, club_id, season):
        """A club's matches in a season."""
        return self.get(f'club/{club_id}/fixtures?season={season}')
