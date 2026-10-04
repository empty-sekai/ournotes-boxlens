"""Detector pose error against truth tiles and its effect on field and rank crops, on synthetic rectangles."""
import unittest

import numpy as np

from bdon_vision.kind_fields import LOGICAL, misread_sweep
from bdon_vision.kind_pose import normalise, pair, paired_tiles, pose_error, summary


class PairTests(unittest.TestCase):
    def test_greedy_one_to_one_pairing(self):
        truth = [[0, 0, 100, 100], [200, 0, 100, 100]]
        detections = [[205, 0, 100, 100], [2, 0, 100, 100], [100, 0, 100, 100], [600, 0, 50, 50]]
        self.assertEqual(pair(truth, detections), {0: 1, 1: 0})
        self.assertEqual(pair(truth, [[60, 0, 100, 100]]), {})

    def test_paired_tiles_counts_misses_kind_confusions_and_ignored_cards(self):
        row = {'cards': [{'kind': 'member', 'bbox': [0, 0, 224, 294]},
                         {'kind': 'snap', 'bbox': [300, 0, 326, 184]},
                         {'kind': 'member', 'bbox': [0, 400, 224, 294]}],
               'ignored_cards': [{'kind': 'member', 'bbox': [700, 0, 224, 294]}]}
        found = [{'kind': 'member', 'bbox': [2, 1, 224, 294]}, {'kind': 'member', 'bbox': [301, 0, 326, 184]},
                 {'kind': 'member', 'bbox': [702, 0, 224, 294]}, {'kind': 'snap', 'bbox': [1500, 900, 326, 184]}]
        tiles, counts = paired_tiles(row, found)
        self.assertEqual(tiles, [[2, 1, 224, 294], None, None])
        self.assertEqual(counts['member:cards'], 2)
        self.assertEqual(counts['member:missed'], 1)
        self.assertEqual(counts['snap:kind_confused'], 1)
        self.assertEqual(counts['unpaired_detections'], 1)

    def test_layout_tile_key(self):
        row = {'cards': [{'kind': 'member', 'bbox': [0, 0, 10, 10], 'bbox_layout': [0, 0, 224, 294]}]}
        tiles, _ = paired_tiles(row, [{'kind': 'member', 'bbox': [1, 0, 224, 294]}], tile_key='bbox_layout')
        self.assertEqual(tiles, [[1, 0, 224, 294]])


class PoseErrorTests(unittest.TestCase):
    def test_identity_has_no_error(self):
        error = pose_error('snap', [10., 20., 163., 92.], [10., 20., 163., 92.])
        self.assertTrue(all(abs(v) < 1e-12 for v in error.values()))

    def test_shift_in_logical_units_and_crop_pixels(self):
        s = .5
        truth = [100., 50., 224*s, 294*s]
        error = pose_error('member', truth, [100.+2*s, 50.-3*s, 224*s, 294*s])
        for key, value in {'centre_dx': 2, 'centre_dy': -3, 'width': 0, 'height': 0, 'scale': 0, 'field_dx': 2,
                           'field_dy': -3, 'field_reach_x': 2, 'field_reach_y': 3, 'rank_dx': 1, 'rank_dy': -1.5,
                           'rank_reach': 1.5}.items():
            self.assertAlmostEqual(error[key], value, msg=key)

    def test_scale_about_the_tile_origin_moves_the_crops(self):
        error = pose_error('member', [0., 0., 224., 294.], [0., 0., 224*1.1, 294*1.1])
        self.assertAlmostEqual(error['scale'], .1)
        self.assertAlmostEqual(error['field_dx'], 66*.1)
        self.assertAlmostEqual(error['field_dy'], (294-28)*.1)
        self.assertAlmostEqual(error['field_reach_x'], 66*.1+72*.1)
        self.assertAlmostEqual(error['rank_dx'], 194.5*.1/2)
        self.assertAlmostEqual(error['rank_dy'], 267.2*.1/2)

    def test_normalise_keeps_centre_and_logical_aspect(self):
        tile = [10., 20., 230., 280.]
        x, y, w, h = normalise('member', tile)
        self.assertAlmostEqual(x+w/2, 10+115)
        self.assertAlmostEqual(y+h/2, 20+140)
        self.assertAlmostEqual(w/h, LOGICAL['member'][0]/LOGICAL['member'][1])
        self.assertAlmostEqual(w/224, (230/224+280/294)/2)

    def test_summary_uses_absolute_quantiles_and_signed_mean(self):
        report = summary([-4., 1., 2., 3.])
        self.assertEqual(report['n'], 4)
        self.assertEqual(report['mean'], .5)
        self.assertEqual(report['max'], 4.)
        self.assertEqual(report['p50'], 2.5)
        self.assertEqual(summary([]), {'n': 0})


class UnreadCardTests(unittest.TestCase):
    def test_unread_cards_never_emit_a_value(self):
        from bdon_vision.evaluate_kind_fields import scatter
        read = np.array([[.001, .999, 0.], [.999, .001, 0.]])
        full = scatter(read, [0, 2], 3)
        self.assertEqual(full.shape, (3, 3))
        self.assertTrue(np.array_equal(full[1], [1., 0., 0.]))
        sweep = misread_sweep(full, np.array([1, 2, 0]), limit=2)
        self.assertEqual(sweep['at_zero_misread_threshold'], {'misreads': 0, 'recall': .5})
        self.assertIsNone(scatter(None, [], 3))


if __name__ == '__main__':
    unittest.main()
