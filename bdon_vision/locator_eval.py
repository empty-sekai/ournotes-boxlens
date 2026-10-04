"""Detection metrics for card-frame locators and scene truth loading.

Scene truth follows the ``bdon-synthetic/2`` layout: each screenshot lists
``cards`` (tile ``bbox`` = [x, y, w, h], ``kind``, ``visible_fraction``),
optional ``foreign_cards`` (tiles whose artwork is outside the reference
gallery; counted as positives), ``ignored_cards`` and
``ignored_foreign_cards`` (less than half visible; neither positive nor
negative) and optional ``ui_elements`` (non-card interface rectangles, used
to attribute false positives).
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .locator import CLASSES, box_iou

WIDTH_BINS = (0, 40, 56, 72, 96, 128, 192, 100000)


def _cluster(values, tolerance):
    groups = []
    for v in sorted(values):
        if groups and abs(v - groups[-1][-1]) < tolerance:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [float(np.median(g)) for g in groups]


def grid_ignore_regions(boxes, width, height):
    """Grid cells that are 0-50% visible, inferred from a regular card grid.

    Used when truth lists partially visible cards without rectangles. Cells
    that could hold such a card are excluded from both loss and metrics.
    """
    if not len(boxes):
        return [[0., 0., float(width), float(height)]]
    b = np.asarray(boxes, np.float64)
    w, h = np.median(b[:, 2]), np.median(b[:, 3])
    xs, ys = _cluster(b[:, 0], .3 * w), _cluster(b[:, 1], .3 * h)
    px = min(np.diff(xs)) if len(xs) > 1 else 1.11 * w
    py = min(np.diff(ys)) if len(ys) > 1 else 1.10 * h
    cols = np.arange(np.floor((-w - xs[0]) / px), np.ceil((width - xs[0]) / px) + 1) * px + xs[0]
    rows = np.arange(np.floor((-h - ys[0]) / py), np.ceil((height - ys[0]) / py) + 1) * py + ys[0]
    out = []
    for y in rows:
        for x in cols:
            vis = max(0, min(width, x + w) - max(0, x)) * max(0, min(height, y + h) - max(0, y)) / (w * h)
            if 0 < vis < .5:
                out.append([float(x), float(y), float(w), float(h)])
    return out


def load_scenes(directories):
    """Read every ``truth.json`` under the given chunk directories."""
    scenes = []
    for directory in directories:
        directory = Path(directory)
        truth = json.loads((directory / 'truth.json').read_text(encoding='utf-8'))
        if not truth.get('complete', False):
            raise ValueError(f'incomplete truth: {directory}')
        for row in truth['screenshots']:
            acquisition = row.get('acquisition', {})
            cards = [dict(c, foreign=False) for c in row.get('cards', [])]
            cards += [dict(c, foreign=True) for c in row.get('foreign_cards', [])]
            size = acquisition.get('output_resolution')
            scenes.append({'path': str(directory / row['file']), 'chunk': directory.name,
                           'boxes': np.asarray([c['bbox'] for c in cards], np.float32).reshape(-1, 4),
                           'labels': np.asarray([CLASSES.index(c['kind']) for c in cards], np.int64),
                           'foreign': np.asarray([c['foreign'] for c in cards], bool),
                           'visible': np.asarray([c.get('visible_fraction', 1.) for c in cards], np.float32),
                           'ignored_rows': row.get('ignored_cards', []) + row.get('ignored_foreign_cards', []),
                           'ui': np.asarray([e['bbox'] for e in row.get('ui_elements', [])], np.float32).reshape(-1, 4),
                           'negative': bool(row.get('negative', False)), 'size': size,
                           'source_resolution': acquisition.get('source_resolution', size),
                           'profile': acquisition.get('profile', truth.get('profile'))})
    return scenes


def resolve_ignore(scene, width, height):
    rows = scene['ignored_rows']
    boxes = [r['bbox'] for r in rows if 'bbox' in r]
    if any('bbox' not in r for r in rows):
        boxes += grid_ignore_regions(scene['boxes'], width, height)
    return np.asarray(boxes, np.float32).reshape(-1, 4)


def _cover(pred, regions):
    """Largest fraction of each predicted box covered by one of ``regions``."""
    if not len(regions) or not len(pred):
        return np.zeros(len(pred))
    x0 = np.maximum(pred[:, None, 0], regions[None, :, 0])
    y0 = np.maximum(pred[:, None, 1], regions[None, :, 1])
    x1 = np.minimum(pred[:, None, 0] + pred[:, None, 2], regions[None, :, 0] + regions[None, :, 2])
    y1 = np.minimum(pred[:, None, 1] + pred[:, None, 3], regions[None, :, 1] + regions[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    return inter.max(axis=1) / np.maximum(pred[:, 2] * pred[:, 3], 1e-9)


def match(detections, boxes, labels, ignore, ui=()):
    """Greedy score-ordered, class-agnostic matching at IoU 0.5 and 0.75.

    Returns per-detection rows and per-truth rows. Because matching visits
    detections in score order, the result at any higher score threshold is the
    restriction of this result.
    """
    order = sorted(range(len(detections)), key=lambda i: -detections[i]['score'])
    pred = np.asarray([detections[i]['bbox'] for i in order], np.float64).reshape(-1, 4)
    kinds = [CLASSES.index(detections[i]['kind']) for i in order]
    iou = box_iou(pred, boxes) if len(boxes) else np.zeros((len(pred), 0))
    cover = _cover(pred, np.asarray(ignore, np.float64).reshape(-1, 4))
    # A box mostly inside a UI element, or one overlapping a UI element.
    on_ui = np.zeros(len(pred), bool)
    ui = np.asarray(ui, np.float64).reshape(-1, 4)
    if len(ui) and len(pred):
        on_ui = (_cover(pred, ui) >= .3) | (box_iou(pred, ui).max(axis=1) >= .3)
    rows = []
    truth = {t: [None] * len(boxes) for t in (.5, .75)}
    for p in range(len(pred)):
        row = {'score': detections[order[p]]['score'], 'kind': kinds[p], 'on_ui': bool(on_ui[p])}
        for t in (.5, .75):
            free = [g for g in range(len(boxes)) if truth[t][g] is None and iou[p, g] >= t]
            if free:
                g = max(free, key=lambda g: iou[p, g])
                truth[t][g] = (row['score'], kinds[p] == int(labels[g]))
                row[t] = 'tp' if kinds[p] == int(labels[g]) else 'tp_wrong_kind'
            elif cover[p] >= .5:
                row[t] = 'ignored'
            else:
                row[t] = 'fp'
        rows.append(row)
    return rows, truth


class Accumulator:
    """Collects matching results and reports metrics at a score threshold."""

    def __init__(self):
        self.preds, self.truth = [], []

    def add(self, detections, scene, width, height, input_scale=1.):
        boxes, labels = scene['boxes'], scene['labels']
        rows, truth = match(detections, boxes, labels, resolve_ignore(scene, width, height), scene.get('ui', ()))
        resolution = f'{width}x{height}'
        long_edge = max(width, height)
        image_keys = ['overall', f'output_long_edge:{int(long_edge // 480 * 480)}',
                      f'source_resolution:{scene["source_resolution"]}']
        if scene.get('negative'):
            image_keys = image_keys + ['screen:card_free']
        for row in rows:
            # Predictions also count under their predicted kind for per-kind precision.
            row['keys'] = image_keys + [f'kind:{CLASSES[row["kind"]]}']
            self.preds.append(row)
        for g in range(len(boxes)):
            w_input = float(boxes[g][2]) * input_scale
            bin_index = int(np.searchsorted(WIDTH_BINS, w_input, side='right')) - 1
            keys = image_keys + [f'kind:{CLASSES[int(labels[g])]}',
                                 f'input_width:{WIDTH_BINS[bin_index]}-{WIDTH_BINS[bin_index + 1]}',
                                 'artwork:' + ('foreign' if scene['foreign'][g] else 'gallery')]
            self.truth.append({'keys': keys, .5: truth[.5][g], .75: truth[.75][g]})
        return rows, truth, resolution

    def summary(self, threshold, strata=True):
        out = {}
        groups = defaultdict(lambda: defaultdict(int))
        for row in self.preds:
            if row['score'] < threshold:
                continue
            for key in row['keys']:
                for t in (.5, .75):
                    groups[key][f'{row[t]}@{t}'] += 1
                    groups[key][f'fp_on_ui@{t}'] += int(row[t] == 'fp' and row.get('on_ui', False))
        for row in self.truth:
            for key in row['keys']:
                groups[key]['truth'] += 1
                for t in (.5, .75):
                    m = row[t]
                    if m is not None and m[0] >= threshold:
                        groups[key][f'recalled@{t}'] += 1
                        groups[key][f'kind_correct@{t}'] += int(m[1])
        for key, g in groups.items():
            item = {'truth': g['truth']}
            for t in (.5, .75):
                tp = g[f'tp@{t}'] + g[f'tp_wrong_kind@{t}']
                fp = g[f'fp@{t}']
                item[f'iou{t}'] = {'recall': g[f'recalled@{t}'] / max(g['truth'], 1),
                                   'precision': tp / max(tp + fp, 1), 'false_positives': fp,
                                   'false_positives_on_ui': g[f'fp_on_ui@{t}'],
                                   'kind_accuracy': g[f'kind_correct@{t}'] / max(g[f'recalled@{t}'], 1),
                                   'recalled': g[f'recalled@{t}'], 'kind_correct': g[f'kind_correct@{t}']}
            out[key] = item
        if not strata:
            return out.get('overall')
        return dict(sorted(out.items()))

    def average_precision(self, t):
        rows = sorted(self.preds, key=lambda r: -r['score'])
        positives = len(self.truth)
        tp = np.cumsum([r[t] in ('tp', 'tp_wrong_kind') for r in rows if r[t] != 'ignored'])
        fp = np.cumsum([r[t] == 'fp' for r in rows if r[t] != 'ignored'])
        if not len(tp) or not positives:
            return 0.
        recall = tp / positives
        precision = tp / np.maximum(tp + fp, 1)
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        grid = np.linspace(0, 1, 101)
        idx = np.searchsorted(recall, grid, side='left')
        return float(np.mean([precision[i] if i < len(precision) else 0. for i in idx]))

    def best_threshold(self, t=.5):
        best = (-1, .3)
        for threshold in np.arange(.1, .91, .02):
            s = self.summary(float(threshold), strata=False)[f'iou{t}']
            f1 = 2 * s['recall'] * s['precision'] / max(s['recall'] + s['precision'], 1e-9)
            best = max(best, (f1, round(float(threshold), 2)))
        return best[1]
