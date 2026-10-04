"""Per-kind field/rank crop geometry, labels and acceptance rules on synthetic inputs."""
import unittest

import numpy as np

from bdon_vision.kind_fields import (FIELD_STORED, MARGIN, RANK_INPUT, RANK_SIDE, RANK_STORED, crop_field, crop_rank,
                                     decide_field, decide_rank, field_inside, field_label, field_region,
                                     latent_field, misread_sweep, rank_icon_inside, rank_label, rank_region)


def one_hot(classes, index, confidence=.999):
    p = np.full(classes, (1-confidence)/(classes-1))
    p[index] = confidence
    return p


class GeometryTests(unittest.TestCase):
    def test_field_region_follows_tile_scale(self):
        left, top, w, h = field_region('member', [100., 50., 112., 147.])
        s = 112/224
        self.assertAlmostEqual(left, 100-6*s)
        self.assertAlmostEqual(top, 50+147-56*s)
        self.assertAlmostEqual(w, 144*s)
        self.assertAlmostEqual(h, 56*s)
        left, top, w, h = field_region('snap', [10., 20., 652., 368.])
        self.assertEqual((left, top, w, h), (10-12., 20+368-112., 288., 112.))

    def test_rank_region_is_centred_on_icon(self):
        left, top, w, h = rank_region('member', [0., 0., 224., 294.])
        self.assertEqual((left+w/2, top+h/2, w, h), (194.5, 267.2, 96., 96.))
        left, top, w, h = rank_region('snap', [0., 0., 163., 92.])
        self.assertAlmostEqual(left+w/2, 306.1/2)
        self.assertAlmostEqual(top+h/2, 153.8/2)
        self.assertAlmostEqual(w, 48.)

    def test_crop_reads_the_expected_pixels(self):
        image = np.zeros((400, 600, 3), np.uint8)
        tile = [100., 40., 224., 294.]
        left, top, w, h = (round(v) for v in field_region('member', tile))
        image[top:top+h, left:left+w] = (10, 20, 30)
        crop = crop_field(image, 'member', tile)
        self.assertEqual(crop.shape, (56, 144, 3))
        # BGR input, RGB output.
        self.assertTrue((crop[1:-1, 1:-1] == (30, 20, 10)).all())
        rank = crop_rank(image, 'member', tile)
        self.assertEqual(rank.shape, (RANK_INPUT, RANK_INPUT, 3))

    def test_stored_margin_centre_equals_deployed_crop(self):
        # Smooth content: warpAffine quantizes source positions to 1/32 pixel.
        yy, xx = np.mgrid[:500, :700]
        image = np.stack([127+100*np.sin(xx/7.)*np.cos(yy/9.), 127+90*np.cos(xx/11.), 127+80*np.sin(yy/5.)], -1)
        image = image.astype(np.uint8)
        for kind, tile in [('member', [123.4, 56.7, 151.3, 198.6]), ('snap', [80.2, 90.1, 260.8, 147.2])]:
            stored = crop_field(image, kind, tile, MARGIN)
            self.assertEqual(stored.shape, (FIELD_STORED[1], FIELD_STORED[0], 3))
            direct = crop_field(image, kind, tile)
            centre = stored[MARGIN:MARGIN+56, MARGIN:MARGIN+144]
            self.assertLessEqual(np.abs(centre.astype(int)-direct.astype(int)).max(), 2)
            stored = crop_rank(image, kind, tile, MARGIN)
            self.assertEqual(stored.shape, (RANK_STORED, RANK_STORED, 3))
            m = MARGIN*RANK_INPUT//RANK_SIDE
            direct = crop_rank(image, kind, tile)
            centre = stored[m:m+RANK_INPUT, m:m+RANK_INPUT]
            self.assertLessEqual(np.abs(centre.astype(int)-direct.astype(int)).max(), 2)

    def test_outside_checks(self):
        self.assertTrue(field_inside('member', [100., 40., 224., 294.], 600, 400))
        self.assertFalse(field_inside('member', [100., 120., 224., 294.], 600, 400))
        self.assertFalse(field_inside('member', [5., 40., 224., 294.], 600, 400))
        self.assertTrue(rank_icon_inside('snap', [10., 10., 326., 184.], 400, 300))
        self.assertFalse(rank_icon_inside('snap', [80., 10., 326., 184.], 400, 300))


class LabelTests(unittest.TestCase):
    def test_field_labels(self):
        self.assertEqual(field_label('member', {'display_mode': 'level', 'level': 37}), 37)
        self.assertIsNone(field_label('member', {'display_mode': 'level', 'level': None}))
        self.assertEqual(field_label('member', {'display_mode': 'training', 'awake_count': 4}), 104)
        self.assertIsNone(field_label('snap', {'display_mode': 'training', 'awake_count': 4}))
        self.assertEqual(field_label('snap', {'display_mode': 'visual', 'field_visible': True}), 0)
        occluded = {'display_mode': 'level', 'level': None, 'field_visible': False, 'latent_state': {'level': 37}}
        self.assertEqual(field_label('member', occluded), 0)

    def test_rank_labels_never_use_latent_state(self):
        self.assertEqual(rank_label({'card_rank': 3, 'display_mode': 'level'}), 3)
        hidden = {'card_rank': None, 'display_mode': 'hide', 'latent_state': {'card_rank': 4}}
        self.assertEqual(rank_label(hidden), 0)


class MisreadSweepTests(unittest.TestCase):
    def test_zero_misread_threshold_and_recall(self):
        p = np.array([one_hot(6, 3, .9999), one_hot(6, 2, .997), one_hot(6, 4, .996), one_hot(6, 5, .999)])
        truth = np.array([3, 2, 1, 0])
        latent = np.array([3, 2, 1, 5])
        sweep = misread_sweep(p, truth, latent, margin=.5, default=.995)
        # Only the 0.996 read of a rank-1 icon is wrong; the hidden read matches the card state.
        self.assertAlmostEqual(sweep['highest_misread_confidence'], .996)
        self.assertGreater(sweep['zero_misread_threshold'], .996)
        self.assertEqual(sweep['at_zero_misread_threshold']['misreads'], 0)
        self.assertAlmostEqual(sweep['at_zero_misread_threshold']['recall'], 2/3)
        self.assertEqual(sweep['at_default_threshold']['misreads'], 1)
        unknown_state = misread_sweep(p, truth, None, margin=.5)
        self.assertAlmostEqual(unknown_state['highest_misread_confidence'], .999)

    def test_latent_value_is_analysis_only(self):
        card = {'display_mode': 'level', 'level': None, 'field_visible': False, 'latent_state': {'level': 12}}
        self.assertEqual(field_label('member', card), 0)
        self.assertEqual(latent_field('member', card), 12)


class DecisionTests(unittest.TestCase):
    def test_member_field_decisions(self):
        self.assertEqual(decide_field('member', one_hot(106, 42))['level'], 42)
        training = decide_field('member', one_hot(106, 103))
        self.assertEqual((training['awake_count'], training['display_mode']), (3, 'training'))
        other = decide_field('member', one_hot(106, 0))
        self.assertEqual((other['level'], other['display_mode']), (None, 'other'))
        self.assertIsNone(decide_field('member', one_hot(106, 42, .99))['level'])

    def test_snap_field_decisions(self):
        self.assertEqual(decide_field('snap', one_hot(101, 100))['level'], 100)
        # A shared 106-class model may emit a training class; Snap has none.
        unsupported = decide_field('snap', one_hot(106, 102))
        self.assertIsNone(unsupported['level'])
        self.assertIsNone(unsupported['awake_count'])

    def test_margin_rule(self):
        p = np.zeros(101)
        p[5], p[6] = .996, .004
        self.assertEqual(decide_field('snap', p)['level'], 5)
        p = np.zeros(6)
        p[2], p[3] = .7, .3
        self.assertIsNone(decide_rank(p)['value'])

    def test_rank_decisions(self):
        self.assertEqual(decide_rank(one_hot(6, 4))['value'], 4)
        none = decide_rank(one_hot(6, 0))
        self.assertIsNone(none['value'])
        self.assertEqual(none['reason'], 'no_visible_rank_icon')


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch not installed')
class ModelTests(unittest.TestCase):
    def test_shapes(self):
        from bdon_vision.train_kind_fields import field_model
        from bdon_vision.train_kind_rank import RankNet
        self.assertEqual(tuple(field_model('member')(torch.zeros(2, 3, 56, 144)).shape), (2, 106))
        self.assertEqual(tuple(field_model('snap')(torch.zeros(2, 3, 56, 144)).shape), (2, 101))
        self.assertEqual(tuple(RankNet().eval()(torch.zeros(3, 3, 48, 48)).shape), (3, 6))

    def test_snap_head_initialised_from_leading_rows(self):
        import tempfile
        from pathlib import Path
        from bdon_vision.train_kind_fields import field_model, initialise_from
        member = field_model('member')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'shared.pt'
            torch.save(member.state_dict(), path)
            snap = initialise_from(field_model('snap'), path)
        self.assertTrue(torch.equal(snap.head.weight, member.head.weight[:101]))
        self.assertTrue(torch.equal(snap.layers[0].weight, member.layers[0].weight))

    def test_unjittered_crop_is_the_stored_centre(self):
        from bdon_vision.train_kind_fields import center_crop, jitter_crop
        x = torch.rand(2, 3, FIELD_STORED[1], FIELD_STORED[0])
        same = jitter_crop(x, 144, 56, MARGIN, 0., 0.)
        self.assertLess(float((same-center_crop(x, 144, 56, MARGIN)).abs().max()), 1e-5)

    def test_anisotropic_jitter_shift(self):
        from bdon_vision.train_kind_fields import center_crop, jitter_crop, jitter_reach
        h, w = FIELD_STORED[1], FIELD_STORED[0]
        columns = torch.arange(w, dtype=torch.float32).view(1, 1, 1, w).expand(64, 1, h, w)
        rows = torch.arange(h, dtype=torch.float32).view(1, 1, h, 1).expand(64, 1, h, w)
        x = torch.cat([columns, rows], 1)
        torch.manual_seed(0)
        moved = jitter_crop(x, 144, 56, MARGIN, (3., 0.), 0.)-center_crop(x, 144, 56, MARGIN)
        dx = moved[:, 0, 20:36, 20:124]
        self.assertLess(float((dx-dx[:, :1, :1]).abs().max()), 1e-3)
        self.assertLessEqual(float(dx.abs().max()), 3.+1e-4)
        self.assertGreater(float(dx.abs().max()), 1.)
        self.assertLess(float(moved[:, 1].abs().max()), 1e-3)
        reach = jitter_reach(144, 56, (4., 7.), .035)
        self.assertAlmostEqual(reach[0], 4+.035*71.5)
        self.assertAlmostEqual(reach[1], 7+.035*27.5)

    def test_onnx_export_has_variable_batch(self):
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            self.skipTest('ONNX Runtime not installed')
        import tempfile
        from pathlib import Path
        from bdon_vision.train_kind_fields import export_onnx, verify_onnx
        from bdon_vision.train_kind_rank import RankNet
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'rank.onnx'
            model = RankNet()
            export_onnx(model, path, (1, 3, 48, 48))
            report = verify_onnx(model, path, np.random.default_rng(0).random((70, 3, 48, 48), dtype=np.float32))
        self.assertLess(report['max_abs_logit_diff'], 1e-4)
        self.assertIn('70', report['batches'])


if __name__ == '__main__':
    unittest.main()
