"""Per-kind encoder geometry, crop banks and open-set metrics on synthetic inputs."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from bdon_vision import kind_geometry as geo
from bdon_vision.kind_metrics import retrieval_table, select_thresholds, summarize

HAS_TORCH = importlib.util.find_spec('torch') is not None and importlib.util.find_spec('torchvision') is not None


class GeometryTests(unittest.TestCase):
    def test_art_window_insets_six_logical_units(self):
        self.assertEqual(geo.art_window('member', [10, 20, 224, 294]), (16., 26., 212., 282.))
        x, y, w, h = geo.art_window('snap', [0, 0, 652, 368])
        self.assertAlmostEqual(x, 12.)
        self.assertAlmostEqual(y, 12.)
        self.assertAlmostEqual(w, 628.)
        self.assertAlmostEqual(h, 344.)

    def test_snap_reference_uses_centered_window_aspect(self):
        left, top, right, bottom = geo.center_crop_box(512, 288, 314 / 172)
        self.assertEqual((left, right), (0., 512.))
        self.assertAlmostEqual(bottom - top, 512 * 172 / 314)
        self.assertAlmostEqual(top, 288 - bottom)

    def test_reference_sizes_and_center_content(self):
        snap = np.zeros((288, 512, 3), np.uint8)
        snap[:, :] = (0, 0, 255)
        snap[4:284] = (255, 0, 0)
        out = np.asarray(geo.reference_image('snap', Image.fromarray(snap)))
        self.assertEqual(out.shape, (128, 224, 3))
        self.assertTrue((out[2:-2, :, 0] > 200).all())
        square = Image.new('RGB', (300, 300), (0, 255, 0))
        member = geo.reference_image('member', square)
        self.assertEqual(member.size, (128, 160))

    def test_member_fitted_canonical_is_not_refitted(self):
        canonical = np.zeros((282, 212, 3), np.uint8)
        canonical[:, :106] = 255
        out = np.asarray(geo.reference_image('member', Image.fromarray(canonical)))
        self.assertGreater(out[:, :60].mean(), 250)
        self.assertLess(out[:, 68:].mean(), 5)

    def test_query_crop_samples_the_artwork_window(self):
        image = np.zeros((400, 600, 3), np.uint8)
        image[:, :] = (255, 0, 0)  # BGR blue frame and background
        bbox = [100, 50, 326, 184]
        x, y, w, h = geo.art_window('snap', bbox)
        image[int(y):int(y + h), int(x):int(x + w)] = (0, 0, 255)  # BGR red art
        crop = geo.query_crop(image, 'snap', bbox)
        self.assertEqual(crop.shape, (128, 224, 3))
        self.assertGreater(crop[2:-2, 2:-2, 0].mean(), 250)  # RGB red
        wide = geo.query_crop(image, 'snap', bbox, margin=.125)
        self.assertEqual(wide.shape, (160, 280, 3))
        self.assertGreater(wide[0, :, 2].mean(), 200)  # context reaches the blue frame

    def test_area_resampling_matches_bilinear_on_flat_regions(self):
        image = np.zeros((800, 1200, 3), np.uint8)
        bbox = [100, 100, 652, 368]  # twice the logical snap tile size
        x, y, w, h = geo.art_window('snap', bbox)
        image[int(y):int(y + h), int(x):int(x + w)] = (0, 0, 255)
        linear = geo.query_crop(image, 'snap', bbox).astype(int)
        area = geo.query_crop(image, 'snap', bbox, resample='area').astype(int)
        self.assertEqual(area.shape, linear.shape)
        self.assertLessEqual(np.abs(linear[2:-2, 2:-2] - area[2:-2, 2:-2]).max(), 2)

    def test_overlay_mask_covers_corners_only(self):
        for kind in geo.KINDS:
            h, w = geo.INPUT[kind]
            mask = geo.overlay_mask(kind, h, w)
            self.assertTrue(mask[0, 0] and mask[-1, 0] and mask[-1, -1])
            self.assertFalse(mask[h // 2, w // 2])
            self.assertLess(mask.mean(), .3)

    def test_cross_kind_windows_keep_the_target_aspect(self):
        fx, fy = geo.cross_window('member', 'snap')
        self.assertEqual(fy, 1.)
        self.assertAlmostEqual(314 * fx / (172 * fy), 212 / 282)
        fx, fy = geo.cross_window('snap', 'member')
        self.assertEqual(fx, 1.)
        self.assertAlmostEqual(212 * fx / (282 * fy), 314 / 172)

    def test_cross_kind_reference_takes_the_center_of_the_window(self):
        snap = np.zeros((288, 512, 3), np.uint8)
        snap[:, :] = (0, 255, 0)
        snap[:, 150:362] = (255, 0, 0)
        out = np.asarray(geo.cross_reference_image('member', 'snap', Image.fromarray(snap)))
        self.assertEqual(out.shape, (160, 128, 3))
        self.assertGreater(out[:, 4:-4, 0].mean(), 250)
        member = geo.cross_reference_image('snap', 'member', Image.new('RGB', (300, 300), (0, 0, 255)))
        self.assertEqual(member.size, (224, 128))


class MetricTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        gallery = rng.normal(size=(6, 16))
        self.gallery = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)
        labels = np.repeat(np.arange(6), 5)
        queries = self.gallery[labels] + rng.normal(scale=.05, size=(30, 16))
        foreign = rng.normal(size=(4, 16))
        queries = np.concatenate([queries, foreign])
        self.queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        self.labels = np.concatenate([labels, [-1] * 4])
        self.held = np.array([False, False, False, False, True, True])

    def test_groups_and_leave_true_out(self):
        table = retrieval_table(self.queries, self.gallery, self.labels, self.held)
        self.assertEqual(list(np.unique(table['group'])), ['foreign', 'seen', 'unseen'])
        self.assertTrue(table['correct'][:30].all())
        self.assertTrue((table['loo_pred'][:30] != self.labels[:30]).all())
        self.assertTrue((table['loo_s1'][:30] < table['s1'][:30]).all())

    def test_selected_thresholds_respect_open_set_limits(self):
        table = retrieval_table(self.queries, self.gallery, self.labels, self.held)
        choice = select_thresholds(table, max_false_accept=0.)
        self.assertTrue(choice['feasible'])
        summary = summarize(table, choice['similarity'], choice['margin'])
        self.assertEqual(summary['unseen']['open_set_false_accept'], 0.)
        self.assertEqual(summary['foreign']['false_accept'], 0.)
        self.assertEqual(summary['known']['accepted_precision'], 1.)
        self.assertEqual(summary['known']['accepted_correct_rate'], 1.)

    def test_identical_cards_cannot_be_accepted(self):
        gallery = np.concatenate([self.gallery[:1], self.gallery[:1], self.gallery[1:]])
        labels = np.zeros(5, int)
        table = retrieval_table(np.repeat(gallery[:1], 5, 0), gallery, labels, np.zeros(len(gallery), bool))
        summary = summarize(table, .5, .01)
        self.assertEqual(summary['known']['accept_rate'], 0.)


class CropBankTests(unittest.TestCase):
    def test_bank_labels_foreign_and_partial_tiles(self):
        from bdon_vision.kind_corpus import build, load_bank
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data, scenes, out = tmp / 'data', tmp / 'scenes', tmp / 'banks'
            data.mkdir()
            scenes.mkdir()
            cards = [{'kind': 'snap', 'id': i, 'asset_id': i, 'file': f'snap-{i}.png'} for i in (1, 7)]
            cards.append({'kind': 'member', 'id': 1, 'asset_id': 1, 'file': 'member-1.png'})
            (data / 'catalog.json').write_text(json.dumps({'cards': cards}), encoding='utf-8')
            image = np.full((300, 800, 3), 40, np.uint8)
            image[20:204, 20:346] = (0, 0, 200)
            image[20:204, 400:726] = (0, 200, 0)
            cv2.imwrite(str(scenes / 'a.png'), image)
            truth = {'complete': True, 'profile': 'train', 'screenshots': [{
                'file': 'a.png', 'locale': 'ja',
                'cards': [{'kind': 'snap', 'id': 7, 'bbox': [20, 20, 326, 184]},
                          {'kind': 'snap', 'id': 1, 'bbox': [700, 200, 326, 184]}],
                'foreign_cards': [{'kind': 'snap', 'bbox': [400, 20, 326, 184], 'foreign_method': 'mix'}]}]}
            (scenes / 'truth.json').write_text(json.dumps(truth), encoding='utf-8')
            manifest = build(data, [scenes], out, 'snap', workers=1)
            self.assertEqual(manifest['count'], 2)
            self.assertEqual(manifest['foreign'], 1)
            self.assertEqual(manifest['skipped'], {'window_outside_image': 1})
            arrays, labels, meta, _ = load_bank(out, cards, 'snap', ('art', 'context', 'baseline'))
            self.assertEqual(list(labels), [1, -1])
            self.assertEqual(arrays['art'].shape[1:], (128, 224, 3))
            self.assertEqual(arrays['baseline'].shape[1:], (160, 128, 3))
            self.assertGreater(arrays['art'][0, ..., 0].mean(), 190)
            self.assertGreater(arrays['art'][1, ..., 1].mean(), 190)
            self.assertEqual(list(meta['foreign']), [False, True])
            self.assertEqual(list(meta['foreign_method']), ['', 'mix'])

    def test_stride_counts_only_screens_of_the_kind_and_arrays_are_optional(self):
        from bdon_vision.kind_corpus import build, load_bank
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data, scenes, out = tmp / 'data', tmp / 'scenes', tmp / 'banks'
            data.mkdir()
            scenes.mkdir()
            cards = [{'kind': 'snap', 'id': 1, 'asset_id': 1, 'file': 'snap-1.png'},
                     {'kind': 'member', 'id': 1, 'asset_id': 1, 'file': 'member-1.png'}]
            (data / 'catalog.json').write_text(json.dumps({'cards': cards}), encoding='utf-8')
            cv2.imwrite(str(scenes / 'a.png'), np.full((300, 400, 3), 90, np.uint8))
            rows = []
            for i in range(6):  # member and snap screens alternate
                kind = 'member' if i % 2 == 0 else 'snap'
                bbox = [10, 10, 224, 294] if kind == 'member' else [10, 10, 326, 184]
                rows.append({'file': 'a.png', 'cards': [{'kind': kind, 'id': 1, 'bbox': bbox}]})
            (scenes / 'truth.json').write_text(json.dumps({'complete': True, 'profile': 'train', 'screenshots': rows}),
                                               encoding='utf-8')
            manifest = build(data, [scenes], out, 'snap', workers=1, arrays=('context',), stride=2)
            self.assertEqual(manifest['count'], 2)
            self.assertEqual(manifest['arrays'], ['context'])
            arrays, labels, meta, _ = load_bank(out, cards, 'snap', ('context',))
            self.assertEqual(arrays['context'].shape[1:], (160, 280, 3))
            self.assertEqual(list(meta['screenshot']), [1, 5])
            self.assertFalse((out / 'snap' / 'art.npy').exists())


@unittest.skipUnless(HAS_TORCH, 'torch not installed')
class EncoderTests(unittest.TestCase):
    def test_kind_encoders_share_the_shared_encoder_state_layout(self):
        import torch
        from bdon_vision.kind_encoder import build_model, pooled_grid
        from bdon_vision.train import CardEncoder
        state = CardEncoder().state_dict()
        for kind in geo.KINDS:
            self.assertEqual(pooled_grid(kind), (2, 2))
            model = build_model(kind)
            model.load_state_dict(state)
            out = model.eval()(torch.rand(3, 3, *geo.INPUT[kind]))
            self.assertEqual(tuple(out.shape), (3, 128))
            self.assertTrue(torch.allclose(out.norm(dim=1), torch.ones(3), atol=1e-5))


if __name__ == '__main__':
    unittest.main()
