"""Card-pose error of a card-frame detector and its effect on field and rank crops.

``detect`` runs a card-frame locator (``bdon_vision.locator.Locator``) over
scene sets and stores the detected tile rectangles per image. ``errors`` pairs
detections with truth tiles (IoU >= 0.5, greedy by IoU) and reports per card
kind:

* tile centre offset and width / height error in logical units (truth pixels
  divided by the truth tile scale ``s``);
* the induced shift of the parameter-field crop centre in model input pixels
  (one pixel per logical unit) and of the rank crop centre (one pixel per two
  logical units), the relative crop scale error, and the largest displacement
  of any crop pixel (centre shift plus scale error times half the crop size).

Quantiles are reported for absolute values (p50, p90, p99, max) together with
the signed mean. Two tile conventions are measured: ``raw`` uses the detected
rectangle as is (scale from its width, field anchored to its bottom edge);
``aspect`` keeps the detected centre and rebuilds the tile with the logical
aspect from the mean of the width and height scales.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .assets import read, write
from .kind_fields import (FIELD_SIZE, KINDS, LOGICAL, RANK_INPUT, RANK_SIDE, field_region, rank_region, scene_cards,
                          tile_scale)

CONVENTIONS = ('raw', 'aspect')
METRICS = ('centre_dx', 'centre_dy', 'width', 'height', 'scale', 'field_dx', 'field_dy', 'field_reach_x',
           'field_reach_y', 'rank_dx', 'rank_dy', 'rank_reach')


def iou(a, b):
    ix = max(0., min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0]))
    iy = max(0., min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    inter = ix*iy
    return inter/max(1e-9, a[2]*a[3]+b[2]*b[3]-inter)


def pair(truth, detections, min_iou=.5):
    """Greedy one-to-one pairing by IoU; returns {truth_index: detection_index}."""
    candidates = sorted(((iou(t, d), i, j) for i, t in enumerate(truth) for j, d in enumerate(detections)), reverse=True)
    out, used = {}, set()
    for value, i, j in candidates:
        if value < min_iou:
            break
        if i in out or j in used:
            continue
        out[i] = j
        used.add(j)
    return out


def normalise(kind, tile):
    """Tile with the logical aspect, same centre, scale = mean of the width and height scales."""
    x, y, w, h = tile
    lw, lh = LOGICAL[kind]
    s = (w/lw+h/lh)/2
    return [x+w/2-lw*s/2, y+h/2-lh*s/2, lw*s, lh*s]


def convention(kind, tile, name):
    if name == 'raw':
        return list(tile)
    if name == 'aspect':
        return normalise(kind, tile)
    raise ValueError(f'Unknown tile convention: {name}')


def pose_error(kind, truth, tile):
    """Errors of ``tile`` against ``truth`` (both [x, y, w, h] tile rectangles) in logical units / input pixels."""
    s = tile_scale(kind, truth)
    scale = tile_scale(kind, tile)/s-1

    def centre(region):
        left, top, w, h = region
        return left+w/2, top+h/2
    fx, fy = centre(field_region(kind, tile))
    tx, ty = centre(field_region(kind, truth))
    rx, ry = centre(rank_region(kind, tile))
    qx, qy = centre(rank_region(kind, truth))
    per_pixel = RANK_SIDE/RANK_INPUT
    field_dx, field_dy = (fx-tx)/s, (fy-ty)/s
    rank_dx, rank_dy = (rx-qx)/s/per_pixel, (ry-qy)/s/per_pixel
    return {'centre_dx': (tile[0]+tile[2]/2-truth[0]-truth[2]/2)/s, 'centre_dy': (tile[1]+tile[3]/2-truth[1]-truth[3]/2)/s,
            'width': (tile[2]-truth[2])/s, 'height': (tile[3]-truth[3])/s, 'scale': scale,
            'field_dx': field_dx, 'field_dy': field_dy,
            'field_reach_x': abs(field_dx)+abs(scale)*FIELD_SIZE[0]/2,
            'field_reach_y': abs(field_dy)+abs(scale)*FIELD_SIZE[1]/2,
            'rank_dx': rank_dx, 'rank_dy': rank_dy,
            'rank_reach': max(abs(rank_dx), abs(rank_dy))+abs(scale)*RANK_INPUT/2}


def summary(values):
    v = np.asarray(values, np.float64)
    if not len(v):
        return {'n': 0}
    a = np.abs(v)
    return {'n': int(len(v)), 'mean': round(float(v.mean()), 4), 'p50': round(float(np.quantile(a, .5)), 4),
            'p90': round(float(np.quantile(a, .9)), 4), 'p99': round(float(np.quantile(a, .99)), 4),
            'max': round(float(a.max()), 4)}


# ---------------------------------------------------------------- detection

LOCATOR = {}


def _init_detector(model, long_edge, threshold):
    from .locator import Locator
    cv2.setNumThreads(1)
    LOCATOR['locator'] = Locator(model, long_edge, threads=1, threshold=threshold)


def _detect(job):
    path, name = job
    image = cv2.imread(str(Path(path)/name), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot decode {Path(path)/name}')
    out = LOCATOR['locator'](cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return [{'kind': d['kind'], 'score': round(d['score'], 5), 'bbox': [round(float(v), 3) for v in d['bbox']]} for d in out]


def detect(datasets, model, output, workers=8, long_edge=640, threshold=.4):
    """Locator tile rectangles for every image of the given scene sets."""
    report = {'schema': 'ournotes-boxlens.locator-detections/1',
              'locator': {'path': str(model), 'sha256': hashlib.sha256(Path(model).read_bytes()).hexdigest(),
                          'long_edge': long_edge, 'threshold': threshold}, 'datasets': {}}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_detector, initargs=(str(model), long_edge, threshold)) as pool:
        for path in map(Path, datasets):
            document = read(path/'truth.json')
            names = [row['file'] for row in document['screenshots']]
            images = {}
            for done, (name, found) in enumerate(zip(names, pool.map(_detect, [(str(path), n) for n in names], chunksize=4))):
                images[name] = found
                if (done+1) % 500 == 0:
                    print(json.dumps({'dataset': str(path), 'images': done+1, 'total': len(names)}), flush=True)
            report['datasets'][str(path)] = {'truth_sha256': hashlib.sha256((path/'truth.json').read_bytes()).hexdigest(),
                                             'images': images}
    write(output, report)
    return report


def detections_for(detections, path):
    """Per-image detections of one scene set, checked against its truth file."""
    entry = detections['datasets'].get(str(path))
    if entry is None:
        raise ValueError(f'No detections for {path}')
    if entry['truth_sha256'] != hashlib.sha256((Path(path)/'truth.json').read_bytes()).hexdigest():
        raise ValueError(f'Detections belong to another truth file: {path}')
    return entry['images']


def paired_tiles(row, found, tile_key='bbox', min_iou=.5):
    """Detected tile per card of ``scene_cards(row)`` (None when missed or detected as the other kind).

    Also returns counts of missed cards, kind confusions and detections paired
    with no card (cards ignored by the truth, e.g. mostly outside the image,
    absorb detections without counting).
    """
    cards = [card for card, _ in scene_cards(row)]
    truth = [[float(v) for v in card[tile_key]] for card in cards]
    ignored = [[float(v) for v in card[tile_key]] for card in row.get('ignored_cards', []) if card.get(tile_key)]
    boxes = [d['bbox'] for d in found]
    matched = pair(truth+ignored, boxes, min_iou)
    tiles, counts = [], Counter()
    for i, card in enumerate(cards):
        j = matched.get(i)
        counts[card['kind']+':cards'] += 1
        if j is None:
            counts[card['kind']+':missed'] += 1
            tiles.append(None)
        elif found[j]['kind'] != card['kind']:
            counts[card['kind']+':kind_confused'] += 1
            tiles.append(None)
        else:
            tiles.append(list(found[j]['bbox']))
    counts['unpaired_detections'] += len(found)-len(matched)
    return tiles, counts


# ---------------------------------------------------------------- error report

def width_bin(width):
    return 'lt100' if width < 100 else 'lt160' if width < 160 else 'ge160'


def errors(datasets, detections, output, tile_key='bbox', min_iou=.5, worst=20):
    detections = read(detections) if not isinstance(detections, dict) else detections
    report = {'schema': 'ournotes-boxlens.locator-pose-error/1', 'locator': detections['locator'], 'tile_key': tile_key,
              'min_iou': min_iou, 'units': {'centre_dx centre_dy width height': 'logical units (truth tile scale)',
                                             'scale': 'detected / truth tile scale - 1',
                                             'field_*': 'field-crop input pixels (1 px = 1 logical unit)',
                                             'rank_*': 'rank-crop input pixels (1 px = 2 logical units)'},
              'datasets': {}}
    for path in map(Path, datasets):
        images = detections_for(detections, path)
        document = read(path/'truth.json')
        values = {c: {k: defaultdict(list) for k in KINDS} for c in CONVENTIONS}
        binned = {k: defaultdict(lambda: defaultdict(list)) for k in KINDS}
        counts, examples = Counter(), {k: [] for k in KINDS}
        for row in document['screenshots']:
            tiles, scene_counts = paired_tiles(row, images[row['file']], tile_key, min_iou)
            counts.update(scene_counts)
            for index, ((card, _), tile) in enumerate(zip(scene_cards(row), tiles)):
                if tile is None:
                    continue
                kind = card['kind']
                truth = [float(v) for v in card[tile_key]]
                for name in CONVENTIONS:
                    error = pose_error(kind, truth, convention(kind, tile, name))
                    for metric in METRICS:
                        values[name][kind][metric].append(error[metric])
                    if name == 'raw':
                        for metric in ('centre_dx', 'centre_dy', 'scale', 'field_reach_x', 'field_reach_y', 'rank_reach'):
                            binned[kind][width_bin(truth[2])][metric].append(error[metric])
                        examples[kind].append((max(error['field_reach_x'], error['field_reach_y']),
                                               {'file': row['file'], 'card': index, 'truth': [round(v, 2) for v in truth],
                                                'detected': tile, **{m: round(error[m], 3) for m in METRICS}}))
        entry = {'images': len(document['screenshots']), 'counts': dict(sorted(counts.items())), 'kinds': {}}
        for kind in KINDS:
            entry['kinds'][kind] = {
                'pairs': len(values['raw'][kind]['scale']),
                'conventions': {name: {m: summary(values[name][kind][m]) for m in METRICS} for name in CONVENTIONS},
                'raw_by_truth_width_px': {b: {m: summary(v) for m, v in sorted(d.items())} for b, d in sorted(binned[kind].items())},
                'largest_field_displacement': [e for _, e in sorted(examples[kind], key=lambda e: -e[0])[:worst]]}
        report['datasets'][str(path)] = entry
        print(json.dumps({'dataset': str(path), 'counts': entry['counts'],
                          **{k: {m: entry['kinds'][k]['conventions']['raw'][m].get('p99') for m in METRICS} for k in KINDS}}), flush=True)
    write(output, report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='action', required=True)
    d = sub.add_parser('detect', help='Run a locator ONNX model over scene sets')
    d.add_argument('--datasets', nargs='+', type=Path, required=True)
    d.add_argument('--locator', type=Path, required=True)
    d.add_argument('--long-edge', type=int, default=640)
    d.add_argument('--threshold', type=float, default=.4)
    d.add_argument('--workers', type=int, default=8)
    d.add_argument('--output', type=Path, required=True)
    e = sub.add_parser('errors', help='Pose error of stored detections against truth tiles')
    e.add_argument('--datasets', nargs='+', type=Path, required=True)
    e.add_argument('--detections', type=Path, required=True)
    e.add_argument('--tile-key', default='bbox', help='Truth tile field of each card (bbox or bbox_layout)')
    e.add_argument('--min-iou', type=float, default=.5)
    e.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.action == 'detect':
        detect(a.datasets, a.locator, a.output, a.workers, a.long_edge, a.threshold)
    else:
        errors(a.datasets, a.detections, a.output, a.tile_key, a.min_iou)
