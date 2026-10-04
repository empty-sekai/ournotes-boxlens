"""Card-frame locator: target encoding, decoding and metrics on synthetic inputs."""
import importlib.util
import unittest

import numpy as np

from bdon_vision.locator import CLASSES, STRIDE, box_iou, decode, encode_targets, find_peaks, letterbox
from bdon_vision.locator_eval import Accumulator, grid_ignore_regions, match


def outputs_from_targets(target):
    """Ideal network outputs for the given dense targets."""
    heat = target['heatmap']
    _, h, w = heat.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    box = target['box']
    size = np.stack([box[2], box[3]])
    offset = np.stack([box[0] / STRIDE - xs, box[1] / STRIDE - ys])
    return heat, size, offset


class LetterboxTests(unittest.TestCase):
    def test_long_edge_and_padding(self):
        image = np.zeros((290, 500, 3), np.uint8)
        canvas, scale = letterbox(image, 640)
        self.assertEqual(canvas.shape, (384, 640, 3))
        self.assertAlmostEqual(scale, 640 / 500)
        self.assertTrue((canvas[384 - 1, 0] == 128).all())


class TargetRoundTripTests(unittest.TestCase):
    def test_decode_recovers_encoded_tiles(self):
        boxes = np.array([[37.5, 40.25, 74, 97], [140, 44, 74, 97], [20, 200, 110, 62], [150.5, 210, 110, 62]], np.float32)
        labels = np.array([0, 0, 1, 1])
        target = encode_targets(boxes, labels, np.zeros((0, 4)), (320, 288))
        self.assertEqual(target['heatmap'].shape, (len(CLASSES), 40, 36))
        self.assertEqual(int((target['heatmap'] == 1).sum()), 4)
        for label, (x, y, w, h) in zip(labels, boxes):
            self.assertAlmostEqual(target['weight'][(target['box'][0] == x + w / 2) & (target['box'][1] == y + h / 2)].sum(), 1., places=5)
        detections = decode(*outputs_from_targets(target), scale=2., threshold=.5)
        self.assertEqual(len(detections), 4)
        recovered = sorted((d['kind'], *np.round(d['bbox'], 4)) for d in detections)
        expected = sorted((CLASSES[k], *np.round(b / 2, 4)) for k, b in zip(labels, boxes))
        for a, b in zip(recovered, expected):
            self.assertEqual(a[0], b[0])
            np.testing.assert_allclose(a[1:], b[1:], atol=1e-3)

    def test_ignore_region_masks_background_only(self):
        target = encode_targets(np.array([[0, 0, 64, 80]]), [0], np.array([[0, 0, 160, 96]]), (96, 160))
        mask, heat = target['loss_mask'], target['heatmap'].max(axis=0)
        self.assertTrue((mask[heat > 0] == 1).all())
        self.assertEqual(mask[:, 10:].sum(), 0)

    def test_overlapping_detections_are_suppressed_across_kinds(self):
        heat = np.zeros((2, 6, 6), np.float32)
        heat[0, 2, 2], heat[1, 2, 4] = .9, .6
        size = np.full((2, 6, 6), 40., np.float32)
        offset = np.full((2, 6, 6), .5, np.float32)
        detections = decode(heat, size, offset, threshold=.3)
        self.assertEqual([d['kind'] for d in detections], ['member'])

    def test_visibility_and_aspect_filters(self):
        heat = np.zeros((2, 8, 8), np.float32)
        heat[0, 0, 0], heat[0, 4, 4], heat[1, 4, 1] = .9, .8, .7
        size = np.stack([np.full((8, 8), 30., np.float32), np.full((8, 8), 40., np.float32)])
        offset = np.full((2, 8, 8), .5, np.float32)
        kept = decode(heat, size, offset, threshold=.3, image_size=(64, 64))
        self.assertEqual([round(d['score'], 1) for d in kept], [.8, .7])
        kept = decode(heat, size, offset, threshold=.3, image_size=(64, 64), aspect_tolerance=.2)
        self.assertEqual([d['kind'] for d in kept], ['member'])

    def test_peaks_are_local_maxima(self):
        heat = np.zeros((1, 5, 5), np.float32)
        heat[0, 1, 1], heat[0, 1, 2], heat[0, 4, 4] = .8, .7, .5
        cls, rows, cols, scores = find_peaks(heat, .4)
        self.assertEqual(list(zip(rows.tolist(), cols.tolist())), [(1, 1), (4, 4)])


class MetricTests(unittest.TestCase):
    def grid(self, rows, cols, w=74, h=97, x0=30, y0=20):
        return np.array([[x0 + c * w * 1.11, y0 + r * h * 1.10, w, h] for r in range(rows) for c in range(cols)], np.float32)

    def test_grid_ignore_regions_cover_partial_row(self):
        boxes = self.grid(3, 4, x0=5, y0=5)
        regions = grid_ignore_regions(boxes, 330, 360)
        self.assertEqual(len(regions), 4)
        self.assertTrue(all(abs(r[1] - (5 + 3 * 97 * 1.10)) < 1e-3 for r in regions))

    def test_perfect_predictions_and_ignored_false_positive(self):
        boxes = self.grid(2, 3)
        scene = {'boxes': boxes, 'labels': np.zeros(len(boxes), np.int64), 'foreign': np.zeros(len(boxes), bool),
                 'ignored_rows': [{'kind': 'member', 'bbox': [30, 250, 74, 97]}], 'source_resolution': [400, 300]}
        detections = [{'kind': 'member', 'score': .9, 'bbox': b.tolist()} for b in boxes]
        detections.append({'kind': 'member', 'score': .8, 'bbox': [31, 251, 74, 97]})
        detections.append({'kind': 'snap', 'score': .7, 'bbox': [300, 200, 50, 30]})
        acc = Accumulator()
        acc.add(detections, scene, 400, 300)
        overall = acc.summary(.5)['overall']
        self.assertEqual(overall['iou0.5']['recall'], 1.)
        self.assertEqual(overall['iou0.5']['false_positives'], 1)
        self.assertEqual(acc.summary(.75)['overall']['iou0.5']['false_positives'], 0)
        self.assertEqual(overall['iou0.75']['kind_accuracy'], 1.)

    def test_wrong_kind_counts_as_localized_but_not_kind_correct(self):
        boxes = self.grid(1, 2)
        rows, truth = match([{'kind': 'snap', 'score': .9, 'bbox': boxes[0].tolist()}], boxes, np.zeros(2, np.int64), np.zeros((0, 4)))
        self.assertEqual(rows[0][.5], 'tp_wrong_kind')
        self.assertEqual(truth[.5][0], (.9, False))
        self.assertIsNone(truth[.5][1])

    def test_box_iou(self):
        self.assertAlmostEqual(float(box_iou([[0, 0, 10, 10]], [[5, 0, 10, 10]])[0, 0]), 50 / 150)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('torchvision'), 'torch not installed')
class ModelShapeTests(unittest.TestCase):
    def test_output_grid_is_stride_eight(self):
        import torch
        from bdon_vision.locator_model import ExportedLocator, LocatorNet
        model = ExportedLocator(LocatorNet()).eval()
        with torch.no_grad():
            heat, size, offset = model(torch.rand(1, 3, 96, 160))
        self.assertEqual(tuple(heat.shape), (1, len(CLASSES), 12, 20))
        self.assertEqual(tuple(size.shape), (1, 2, 12, 20))
        self.assertTrue(bool(((heat >= 0) & (heat <= 1)).all()))
        self.assertTrue(bool((size > 0).all()))


if __name__ == '__main__':
    unittest.main()
