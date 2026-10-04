"""Per-kind field and card-rank crops, labels and acceptance rules.

Member cards and Snap cards use separate readers. Every crop is defined
relative to the card tile rectangle ``[x, y, w, h]`` (output-image pixels) and
the tile scale ``s = w / logical_width`` (member 224, Snap 326):

* parameter field: left ``x - 6s``, top ``y + h - 56s``, size ``144s x 56s``,
  resampled to 144x56;
* rank icon: square of side ``96s`` centred on the icon centre
  ``(x + cx*s, y + cy*s)`` with ``(cx, cy)`` = member (194.5, 267.2) or
  Snap (306.1, 153.8), resampled to 48x48.

Sampling follows ``cv2.warpAffine`` with ``WARP_INVERSE_MAP``: output pixel
``(i, j)`` reads the bilinear source position ``(left + i*sx, top + j*sy)``
with integer pixel centres, ``sx = width / out_width`` and
``sy = height / out_height``. Pixels outside the image are zero.

Crop banks are stored with a context margin (8 logical units on every side)
so training can jitter the crop; the centre of a stored crop is identical to
the deployed crop.
"""
import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .assets import read, write
from .corpus import sample

KINDS = ('member', 'snap')
LOGICAL = {'member': (224, 294), 'snap': (326, 184)}
RANK_CENTER = {'member': (194.5, 267.2), 'snap': (306.1, 153.8)}
FIELD_SIZE = (144, 56)          # logical units == model input pixels (W, H)
RANK_SIDE = 96                  # logical units of the square rank crop
RANK_INPUT = 48                 # model input side in pixels
MARGIN = 8                      # stored context margin in logical units
FIELD_STORED = (FIELD_SIZE[0]+2*MARGIN, FIELD_SIZE[1]+2*MARGIN)
RANK_STORED = (RANK_SIDE+2*MARGIN)*RANK_INPUT//RANK_SIDE
FIELD_CLASSES = {'member': 106, 'snap': 101}
RANK_CLASSES = 6
FIELD_CONFIDENCE = .995
FIELD_MARGIN = .5
RANK_CONFIDENCE = .995
RANK_MARGIN = .5
LOCALES = ['ja', 'en', 'zh-Hant', 'zh-Hans', 'ko']
MODES = ['level', 'training', 'total', 'performance', 'technic', 'visual', 'hide']


def tile_scale(kind, bbox):
    return bbox[2]/LOGICAL[kind][0]


def field_region(kind, bbox, margin=0.):
    """Return (left, top, width, height) of the parameter-field crop."""
    x, y, w, h = bbox
    s = tile_scale(kind, bbox)
    return (x-(6+margin)*s, y+h-(56+margin)*s, (FIELD_SIZE[0]+2*margin)*s, (FIELD_SIZE[1]+2*margin)*s)


def rank_region(kind, bbox, margin=0.):
    """Return (left, top, width, height) of the square rank-icon crop."""
    x, y, w, h = bbox
    s = tile_scale(kind, bbox)
    cx, cy = RANK_CENTER[kind]
    half = (RANK_SIDE/2+margin)*s
    return (x+cx*s-half, y+cy*s-half, 2*half, 2*half)


def field_inside(kind, bbox, width, height, border=2.):
    left, top, w, h = field_region(kind, bbox)
    return left >= border and top >= border and left+w <= width-border and top+h <= height-border


def rank_icon_inside(kind, bbox, width, height, border=2.):
    """The full 64-unit icon (aspect 102:95) lies inside the image."""
    x, y = bbox[:2]
    s = tile_scale(kind, bbox)
    cx, cy = RANK_CENTER[kind]
    half_w, half_h = 32*s, 32*95/102*s
    return (x+cx*s-half_w >= border and y+cy*s-half_h >= border
            and x+cx*s+half_w <= width-border and y+cy*s+half_h <= height-border)


def crop_field(image, kind, bbox, margin=0.):
    """RGB uint8 crop of shape (56+2m, 144+2m, 3) from a BGR image."""
    size = (round(FIELD_SIZE[0]+2*margin), round(FIELD_SIZE[1]+2*margin))
    return sample(image, *field_region(kind, bbox, margin), size)


def crop_rank(image, kind, bbox, margin=0.):
    """RGB uint8 square rank crop; side 48 at margin 0."""
    side = round((RANK_SIDE+2*margin)*RANK_INPUT/RANK_SIDE)
    return sample(image, *rank_region(kind, bbox, margin), (side, side))


def to_input(crops):
    """uint8 NHWC RGB -> float32 NCHW in [0, 1]."""
    return np.ascontiguousarray(np.asarray(crops, np.float32).transpose(0, 3, 1, 2)/255.)


def field_label(kind, card):
    """Field class for a crop lying inside the image, or None when unusable.

    0 other/hidden parameter or an unreadable (occluded) field, 1..100 level,
    101..105 member training count. Values of invisible fields are never
    targets: an occluded level field is class 0, not its latent level.
    """
    mode = card.get('display_mode', 'level')
    if not card.get('field_visible', True):
        return 0
    if mode == 'level':
        return card.get('level')
    if mode == 'training':
        if kind != 'member' or card.get('awake_count') is None:
            return None
        return 100+card['awake_count']
    return 0


def rank_label(card):
    """1..5 for a fully visible icon, otherwise 0 (hidden, absent or cut)."""
    return card['card_rank'] if card.get('card_rank') is not None else 0


def latent_field(kind, card):
    """Value the field would show without occlusion, for error analysis only (-1 unknown).

    Never a training target. It separates a value read through a partial
    occlusion that matches the card state from a genuine misread.
    """
    latent = card.get('latent_state')
    if latent is None:
        return -1
    mode = card.get('display_mode', 'level')
    if mode == 'level':
        return latent['level']
    if mode == 'training':
        return 100+latent['awake_count'] if kind == 'member' else 0
    return 0


def latent_rank(card):
    """Rank icon the card state would show (0 in the hidden view), for error analysis only (-1 unknown)."""
    latent = card.get('latent_state')
    if latent is None:
        return -1
    return 0 if card.get('display_mode') == 'hide' else latent['card_rank']


def softmax(logits):
    logits = np.asarray(logits, np.float64)
    p = np.exp(logits-logits.max(axis=-1, keepdims=True))
    return p/p.sum(axis=-1, keepdims=True)


def accept(probabilities, confidence, margin):
    """Top class, its probability, and whether the acceptance rule holds."""
    p = np.asarray(probabilities)
    order = np.sort(p, axis=-1)
    top = p.argmax(axis=-1)
    best, second = order[..., -1], order[..., -2]
    return top, best, (best >= confidence) & (best-second >= margin)


def misread_sweep(probabilities, truth, latent=None, limit=None, margin=FIELD_MARGIN, default=FIELD_CONFIDENCE,
                  grid=(.9, .95, .98, .99, .995, .998, .999, .9995, .9999)):
    """Misreads and recall as the confidence threshold rises (margin rule fixed).

    A misread is an emitted value that differs from the visible target, or a
    value emitted for a hidden target that differs from the card state
    (``latent``; -1 or missing means unknown, so any value is a misread).
    ``limit`` is the largest class that emits a value (Snap 100 even for a
    106-class model). Returns the lowest threshold with no misreads on these
    crops and the recall of correct values there.
    """
    p = np.asarray(probabilities, np.float64)
    truth = np.asarray(truth)
    latent = np.full(len(truth), -1) if latent is None else np.asarray(latent)
    order = np.sort(p, axis=-1)
    top, best = p.argmax(axis=-1), order[:, -1]
    limit = p.shape[1]-1 if limit is None else limit
    value = np.where((best-order[:, -2] >= margin) & (top > 0) & (top <= limit), top, 0)
    misread = (value > 0) & (((truth > 0) & (value != truth)) | ((truth == 0) & (value != latent)))
    visible = max(1, int((truth > 0).sum()))

    def at(tau):
        emitted = (value > 0) & (best >= tau)
        return {'misreads': int((emitted & misread).sum()),
                'recall': float((emitted & (truth > 0) & (value == truth)).sum()/visible)}
    worst = float(best[misread].max()) if misread.any() else None
    zero = 0. if worst is None else float(np.nextafter(worst, 2.))
    return {'visible_targets': int((truth > 0).sum()), 'highest_misread_confidence': worst,
            'zero_misread_threshold': zero, 'zero_misread_achievable': zero <= 1.,
            'at_zero_misread_threshold': at(zero), 'at_default_threshold': {'threshold': default, **at(default)},
            'by_threshold': {str(t): at(t) for t in grid}}


def decide_field(kind, probabilities, confidence=FIELD_CONFIDENCE, margin=FIELD_MARGIN):
    """Field decision for one crop: dict(level, awake_count, display_mode, confidence, reason)."""
    top, best, ok = accept(probabilities, confidence, margin)
    top, best, ok = int(top), float(best), bool(ok)
    out = {'level': None, 'awake_count': None, 'display_mode': 'unknown', 'confidence': round(best, 4)}
    if not ok:
        return {**out, 'reason': 'ambiguous_parameter'}
    if top == 0:
        return {**out, 'display_mode': 'other', 'reason': 'other_or_hidden_parameter'}
    if top <= 100:
        return {**out, 'level': top, 'display_mode': 'level'}
    if kind == 'member' and top <= 105:
        return {**out, 'awake_count': top-100, 'display_mode': 'training'}
    return {**out, 'reason': 'unsupported_parameter'}


def decide_rank(probabilities, confidence=RANK_CONFIDENCE, margin=RANK_MARGIN):
    """Rank decision for one crop: dict(value, confidence, reason)."""
    top, best, ok = accept(probabilities, confidence, margin)
    top, best, ok = int(top), float(best), bool(ok)
    if not ok:
        return {'value': None, 'confidence': round(best, 4), 'reason': 'ambiguous_icon'}
    if top == 0:
        return {'value': None, 'confidence': round(best, 4), 'reason': 'no_visible_rank_icon'}
    return {'value': top, 'confidence': round(best, 4)}


# ---------------------------------------------------------------- crop banks

def scene_cards(row):
    """Catalog cards and foreign-art cards of one screenshot with a foreign flag."""
    for card in row.get('cards', []):
        yield card, 0
    for card in row.get('foreign_cards', []):
        yield card, 1


def _scene_job(job):
    path, row, dataset_index, scene_index = job
    raw = (Path(path)/row['file']).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if row.get('sha256') and row['sha256'] != digest:
        raise ValueError(f"Image hash mismatch: {row['file']}")
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode {Path(path)/row['file']}")
    ih, iw = image.shape[:2]
    acquisition = row.get('acquisition', {})
    out = {kind: {'field': [], 'rank': []} for kind in KINDS}
    skipped = Counter()
    for card_index, (card, foreign) in enumerate(scene_cards(row)):
        kind = card['kind']
        bbox = [float(v) for v in card['bbox']]
        meta = {'dataset': dataset_index, 'scene': scene_index, 'card': card_index,
                'id': int(card.get('id', -1) if card.get('id') is not None else -1), 'foreign': foreign,
                'locale': LOCALES.index(row.get('locale', 'zh-Hans')),
                'mode': MODES.index(card.get('display_mode', 'level')),
                'width': bbox[2], 'source_resolution': 'x'.join(map(str, acquisition.get('source_resolution', [iw, ih]))),
                'profile': acquisition.get('profile', 'unknown'),
                'occluded': int(not card.get('field_visible', True)), 'latent': latent_field(kind, card)}
        label = field_label(kind, card)
        left, top, w, h = field_region(kind, bbox)
        if label is None:
            skipped[kind+':field_no_target'] += 1
        elif left < 0 or top < 0 or left+w > iw or top+h > ih:
            skipped[kind+':field_cropped'] += 1
        else:
            out[kind]['field'].append((crop_field(image, kind, bbox, MARGIN), label, meta))
        rank = rank_label(card)
        meta = {**meta, 'icon_inside': int(rank_icon_inside(kind, bbox, iw, ih)), 'latent': latent_rank(card),
                'occluded': int(card.get('rank_visible') is False and rank_icon_inside(kind, bbox, iw, ih))}
        out[kind]['rank'].append((crop_rank(image, kind, bbox, MARGIN), rank, meta))
    return out, skipped, digest


META_INT = ['dataset', 'scene', 'card', 'id', 'foreign', 'locale', 'mode', 'icon_inside', 'occluded', 'latent']


def build(datasets, output, workers=8):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    jobs, fingerprint, names = [], hashlib.sha256(), []
    for d, path in enumerate(map(Path, datasets)):
        document = read(path/'truth.json')
        if not document.get('complete', False):
            raise ValueError(f'Incomplete dataset: {path}')
        fingerprint.update((path/'truth.json').read_bytes())
        names.append(str(path))
        jobs += [(str(path), row, d, i) for i, row in enumerate(document['screenshots'])]
    banks = {kind: {'field': [], 'rank': []} for kind in KINDS}
    skipped = Counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, (out, skip, digest) in enumerate(pool.map(_scene_job, jobs, chunksize=4)):
            fingerprint.update(bytes.fromhex(digest))
            skipped.update(skip)
            for kind in KINDS:
                for part in ('field', 'rank'):
                    banks[kind][part] += out[kind][part]
            if (done+1) % 500 == 0:
                print(json.dumps({'scenes': done+1, 'total': len(jobs)}), flush=True)
    summary = {}
    for kind in KINDS:
        for part in ('field', 'rank'):
            rows = banks[kind][part]
            prefix = output/f'{kind}-{part}'
            shape = (FIELD_STORED[1], FIELD_STORED[0], 3) if part == 'field' else (RANK_STORED, RANK_STORED, 3)
            images = np.stack([r[0] for r in rows]) if rows else np.zeros((0, *shape), np.uint8)
            labels = np.array([r[1] for r in rows], np.int16)
            np.save(prefix.with_name(prefix.name+'-images.npy'), images)
            np.save(prefix.with_name(prefix.name+'-labels.npy'), labels)
            meta = {k: np.array([r[2].get(k, -1) for r in rows], np.int32) for k in META_INT}
            meta['width'] = np.array([r[2]['width'] for r in rows], np.float32)
            for key in ('source_resolution', 'profile'):
                vocabulary = sorted({r[2][key] for r in rows})
                meta[key+'_vocabulary'] = np.array(vocabulary)
                meta[key] = np.array([vocabulary.index(r[2][key]) for r in rows], np.int32)
            np.savez(prefix.with_name(prefix.name+'-meta.npz'), **meta)
            summary[f'{kind}-{part}'] = {'count': len(rows), 'labels': {str(k): int(v) for k, v in sorted(Counter(labels.tolist()).items())}}
    manifest = {'schema': 'ournotes-boxlens.kind-crop-bank/1', 'datasets': names, 'scenes': len(jobs),
                'source_fingerprint': fingerprint.hexdigest(), 'banks': summary, 'skipped': dict(skipped),
                'locales': LOCALES, 'modes': MODES, 'margin_logical': MARGIN,
                'field_stored_wh': list(FIELD_STORED), 'rank_stored_side': RANK_STORED, 'complete': True}
    write(output/'manifest.json', manifest)
    print(json.dumps({'scenes': len(jobs), 'banks': {k: v['count'] for k, v in summary.items()}, 'skipped': dict(skipped)}), flush=True)
    return manifest


def load_bank(path, kind, part):
    """(images uint8 NHWC, labels int64, meta dict) of one stored bank."""
    path = Path(path)
    if not read(path/'manifest.json').get('complete'):
        raise ValueError(f'Incomplete crop bank: {path}')
    images = np.load(path/f'{kind}-{part}-images.npy', mmap_mode='r')
    labels = np.load(path/f'{kind}-{part}-labels.npy').astype(np.int64)
    with np.load(path/f'{kind}-{part}-meta.npz') as meta:
        return images, labels, {k: meta[k] for k in meta.files}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Build per-kind field and rank crop banks from synthetic scene sets')
    p.add_argument('--datasets', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    build(a.datasets, a.output, a.workers)
