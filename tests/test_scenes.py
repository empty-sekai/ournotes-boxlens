"""List-screen geometry, coverage accounting and dataset verification for scene generation."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bdon_vision.scenes import _box_fraction, list_layout, verify


class ListLayoutTests(unittest.TestCase):
    def test_member_rows_are_centred_on_a_4_by_3_screen(self):
        layout = list_layout('member', 1441, 1080)
        self.assertAlmostEqual(layout['scale'], 1441/1920)
        self.assertEqual(layout['columns'], 6)
        self.assertAlmostEqual(layout['origin'][0], 1441/2-1479*1441/1920/2)
        self.assertAlmostEqual(layout['origin'][1], 180*1441/1920)
        self.assertAlmostEqual(layout['pitch'][0], 251*1441/1920)
        self.assertAlmostEqual(layout['pitch'][1], 321*1441/1920)

    def test_snap_rows_hold_four_cells(self):
        layout = list_layout('snap', 1920, 1080)
        self.assertEqual(layout['columns'], 4)
        self.assertAlmostEqual(layout['origin'][0], (1920-1412)/2)
        self.assertEqual(layout['tile'], (326., 184.))

    def test_wide_screens_keep_height_scale_and_fixed_columns(self):
        layout = list_layout('member', 2340, 1080)
        self.assertAlmostEqual(layout['scale'], 1.)
        self.assertAlmostEqual(layout['origin'][0], (2340-1479)/2)

    def test_ui_scale_multiplies_canvas_scale(self):
        self.assertAlmostEqual(list_layout('snap', 1920, 1080, .9)['scale'], .9)


class CoverageTests(unittest.TestCase):
    def test_visible_box_is_uncovered(self):
        mask = np.zeros((100, 200), bool)
        self.assertEqual(_box_fraction(mask, (10, 10, 50, 20), 200, 100), 0.)

    def test_mask_and_screen_edges_count_as_covered(self):
        mask = np.zeros((100, 200), bool)
        mask[:, :20] = True
        self.assertAlmostEqual(_box_fraction(mask, (10, 10, 40, 10), 200, 100), .25)
        self.assertAlmostEqual(_box_fraction(mask, (180, 90, 40, 20), 200, 100), .75)


class VerifyTests(unittest.TestCase):
    def test_hash_mismatch_and_stray_images_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'000000.png').write_bytes(b'first')
            row = {'file': '000000.png', 'sha256': hashlib.sha256(b'first').hexdigest(), 'cards': []}
            (root/'truth.json').write_text(json.dumps({'complete': True, 'count': 1, 'screenshots': [row]}), encoding='utf-8')
            self.assertTrue(verify(root)['ok'])
            (root/'000001.png').write_bytes(b'stray')
            self.assertFalse(verify(root)['ok'])
            (root/'000001.png').unlink()
            (root/'000000.png').write_bytes(b'changed')
            self.assertEqual(verify(root)['hash_failed'], 1)


if __name__ == '__main__':
    unittest.main()
