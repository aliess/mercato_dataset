#!/usr/bin/env python3
"""
Store how famous each legend is in legends.json: the number of Wikipedia languages that
have an article on him (Wikidata sitelinks, matched by Transfermarkt player id, P2446).

sources/fame.py uses it to order the legends within their value tier. Run it again after
adding legends; it makes one public request to the Wikidata query service.

Usage:
    python sources/legends/legend_fame.py
"""

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

LEGENDS_FILE = Path(__file__).resolve().parent / 'legends.json'
QUERY_URL = 'https://query.wikidata.org/sparql?format=json&query='


def main():
    legends = json.loads(LEGENDS_FILE.read_text())
    ids = ' '.join(f'"{legend["id"]}"' for legend in legends)
    query = f'SELECT ?tm ?links WHERE {{ VALUES ?tm {{ {ids} }} ?item wdt:P2446 ?tm; wikibase:sitelinks ?links. }}'
    request = urllib.request.Request(QUERY_URL + urllib.parse.quote(query),
                                     headers={'User-Agent': 'mercato-dataset/1.0 (football quiz data build)'})
    rows = json.load(urllib.request.urlopen(request, timeout=60))['results']['bindings']
    languages = {}
    for row in rows:  # a Transfermarkt id on two Wikidata items keeps the larger count
        player_id = row['tm']['value']
        languages[player_id] = max(languages.get(player_id, 0), int(row['links']['value']))

    missing = [legend['name'] for legend in legends if legend['id'] not in languages]
    if missing:
        sys.exit(f'No Wikidata item for {missing}: add the Transfermarkt id (P2446) on Wikidata, '
                 'or give the legend "wikipedia_languages" by hand.')
    for legend in legends:
        legend['wikipedia_languages'] = languages[legend['id']]
    LEGENDS_FILE.write_text('[\n' + ',\n'.join(' ' + json.dumps(legend, ensure_ascii=False) for legend in legends) + '\n]\n')
    print(f'✓ {len(legends)} legends, {min(languages.values())} to {max(languages.values())} Wikipedia languages')


if __name__ == '__main__':
    main()
