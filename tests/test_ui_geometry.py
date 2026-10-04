"""Tile, artwork-window and reference geometry of the native list layout."""
import unittest

from PIL import Image

from bdon_vision.kind_geometry import art_window, center_crop_box, reference_image


class UIFrameGeometryTests(unittest.TestCase):
    def test_artwork_window_is_inset_by_six_logical_units(self):
        self.assertEqual(art_window('member', [10., 20., 224., 294.]), (16., 26., 212., 282.))
        self.assertEqual(art_window('snap', [0., 0., 163., 92.]), (3., 3., 157., 86.))

    def test_snap_reference_takes_the_centre_of_the_full_artwork(self):
        left, top, right, bottom = center_crop_box(512, 288, 314/172)
        self.assertEqual((left, right), (0., 512.))
        self.assertAlmostEqual(top, (288-512*172/314)/2)
        self.assertAlmostEqual(bottom, 288-top)
        self.assertEqual(reference_image('snap', Image.new('RGB', (512, 288))).size, (224, 128))

    def test_member_reference_fills_the_portrait_window(self):
        self.assertEqual(reference_image('member', Image.new('RGB', (256, 256))).size, (128, 160))
        self.assertEqual(reference_image('member', Image.new('RGB', (212, 282))).size, (128, 160))


if __name__ == '__main__':
    unittest.main()
