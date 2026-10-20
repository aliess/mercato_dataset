"""
Tests for the story themes of build_dossiers.py (CONTRACTS.md §1, plan §8 item 145): the source
check refuses a missing or unknown theme, the contract check refuses an unknown one (and a theme
on a dossier that is not a story), and the built learn_v1.json carries each story's theme.

    cd mercato_dataset && python -m unittest sources.learn.test_build_dossiers -v

The last test reads publish/learn_v1.json and is skipped when it has not been built.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sources.learn import build_dossiers as bd  # noqa: E402

PLAYER_IDS = [str(i) for i in range(1, 9)]
PLAYERS = {pid: {'id': pid, 'player_name': f'Player Number{pid}', 'player_slug': f'player-number{pid}'}
           for pid in PLAYER_IDS}
TEASER = {'en': 'Signed for €10m in 2010 and sold on two years later.', 'ar': None}
LINE = {'en': 'A line of story that is long enough to pass the length check.', 'ar': None}


def source_story(**fields):
    """A story as dossiers.json writes it (STORY_FIELDS order)."""
    story = {'id': 'story_test', 'world': 'w1', 'theme': 'history', 'status': 'draft',
             'title': {'en': 'A test story', 'ar': None}, 'line': LINE,
             'mystery': {'player': '1', 'teaser': TEASER}, 'players': PLAYER_IDS}
    story.update(fields)
    return story


def built_story(**fields):
    """A story dossier as learn_v1.json writes it."""
    dossier = {'kind': 'story', 'world': 'w1', 'title': {'en': 'A test story', 'ar': None}, 'line': LINE,
               'mystery': {'player': '1', 'teaser': TEASER}, 'players': PLAYER_IDS, 'stop': None, 'free': False,
               'theme': 'history'}
    dossier.update(fields)
    return dossier


class LoadStoriesTheme(unittest.TestCase):
    """The source check in load_stories()."""

    def load(self, stories):
        with tempfile.TemporaryDirectory() as tmp:
            learn_dir = Path(tmp) / 'sources' / 'learn'
            learn_dir.mkdir(parents=True)
            story_file = learn_dir / 'dossiers.json'
            story_file.write_text(json.dumps(stories, ensure_ascii=False), encoding='utf-8')
            with mock.patch.object(bd, 'LEARN_DIR', learn_dir), mock.patch.object(bd, 'STORY_FILE', story_file):
                return bd.load_stories(PLAYERS, {'w1', 'summit'}, {})

    def test_each_theme_is_accepted(self):
        for theme in bd.STORY_THEMES:
            with self.subTest(theme=theme):
                self.assertEqual(self.load([source_story(theme=theme)])[0]['theme'], theme)

    def test_unknown_theme_is_refused(self):
        for theme in ('legends', 'History', '', None, ['history']):
            with self.subTest(theme=theme), self.assertRaises(SystemExit) as stop:
                self.load([source_story(theme=theme)])
            self.assertIn('theme', str(stop.exception.code))

    def test_missing_theme_is_refused(self):
        story = source_story()
        del story['theme']
        with self.assertRaises(SystemExit) as stop:
            self.load([story])
        self.assertIn('expected', str(stop.exception.code))

    def test_every_written_story_has_a_known_theme(self):
        stories = json.loads(bd.STORY_FILE.read_text(encoding='utf-8'))
        self.assertTrue(stories)
        for story in stories:
            with self.subTest(story=story['id']):
                self.assertEqual(list(story), bd.STORY_FIELDS)
                self.assertIn(story['theme'], bd.STORY_THEMES)


class ValidateTheme(unittest.TestCase):
    """The contract check in validate(): `theme` is optional on a story, one of STORY_THEMES."""

    def problems(self, dossier_id, dossier):
        content = {'worlds': [{'id': 'w1', 'dossiers': [dossier_id]}], 'stops': {},
                   'dossiers': {dossier_id: dossier}, 'anchors': {pid: [] for pid in PLAYER_IDS}, 'trails': []}
        found = bd.validate(content, {'players': list(PLAYERS.values())}, {'matches': []})
        return [p for p in found if p.startswith(dossier_id)]

    def test_known_theme_passes(self):
        for theme in bd.STORY_THEMES:
            with self.subTest(theme=theme):
                self.assertEqual(self.problems('story_test', built_story(theme=theme)), [])

    def test_unknown_theme_is_refused(self):
        problems = self.problems('story_test', built_story(theme='derbies'))
        self.assertEqual(len(problems), 1)
        self.assertIn("theme 'derbies'", problems[0])

    def test_story_without_theme_passes(self):
        dossier = built_story()
        del dossier['theme']
        self.assertEqual(self.problems('story_test', dossier), [])

    def test_theme_on_another_kind_is_refused(self):
        dossier = built_story(kind='route', title={'en': 'A → B', 'ar': 'أ ← ب'}, line=None,
                              mystery={'player': '1', 'teaser': None}, theme='rivals')
        problems = self.problems('route_1_2', dossier)
        self.assertEqual(len(problems), 1)
        self.assertIn('fields', problems[0])


LEARN_FILE = ROOT / 'publish' / 'learn_v1.json'


@unittest.skipUnless(LEARN_FILE.exists(), 'publish/learn_v1.json is not built')
class BuiltFileTheme(unittest.TestCase):
    """The built file: every story dossier carries the theme dossiers.json gives it."""

    def test_stories_carry_their_theme(self):
        dossiers = json.loads(LEARN_FILE.read_text(encoding='utf-8'))['dossiers']
        source = {s['id']: s for s in json.loads(bd.STORY_FILE.read_text(encoding='utf-8'))}
        for d_id, dossier in dossiers.items():
            with self.subTest(dossier=d_id):
                if dossier['kind'] == 'story':
                    self.assertEqual(dossier.get('theme'), source[d_id]['theme'])
                else:
                    self.assertNotIn('theme', dossier)


if __name__ == '__main__':
    unittest.main()
