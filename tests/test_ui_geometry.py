"""Native Snap aspect-fill placement must preserve full-art reference boxes."""
import unittest

import numpy as np

from bdon_vision.engine import Engine


class UIFrameGeometryTests(unittest.TestCase):
    def test_snap_full_art_reference_maps_to_native_window(self):
        engine = Engine.__new__(Engine)
        engine.art = [np.zeros((288, 512, 3), dtype=np.uint8)]
        item = {'kind': 'snap', '_index': 0, 'bbox': [10., 20., 314., 176.625]}
        self.assertEqual(engine.ui_bbox(item), [10., 22.3125, 314., 172.])
        self.assertEqual(item['bbox'], [10., 20., 314., 176.625])

    def test_member_canonical_reference_is_not_cropped_again(self):
        engine = Engine.__new__(Engine)
        engine.art = [np.zeros((282, 212, 3), dtype=np.uint8)]
        item = {'kind': 'member', '_index': 0, 'bbox': [10., 20., 212., 282.]}
        self.assertEqual(engine.ui_bbox(item), item['bbox'])


if __name__ == '__main__':
    unittest.main()
