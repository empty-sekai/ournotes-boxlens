"""Per-kind artwork crop banks from scene truth files.

Every tile listed in a screenshot's ``cards`` becomes a query crop labelled
with its catalog index. Tiles listed in ``foreign_cards`` (artwork outside the
catalog) are stored with label -1 and a ``foreign`` flag; they are evaluation
material for open-set rejection and never identity supervision.

Bank layout (one directory per kind; ``--arrays`` selects the crop arrays)::

    art.npy        uint8 [N, H, W, 3]    exact artwork window at encoder input size
    context.npy    uint8 [N, Hc, Wc, 3]  window expanded by CONTEXT_MARGIN per side
    baseline.npy   uint8 [N, 160, 128, 3] snap only: window squashed to the shared
                                          portrait input of the single-encoder model
    labels.npy     int32 [N]             catalog index, -1 for foreign artwork
    meta.npz       per-crop source, visibility and acquisition fields
    manifest.json  catalog order, dataset fingerprints and counts
"""
import argparse
import hashlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from numpy.lib.format import open_memmap

from .assets import read, write
from .kind_geometry import INPUT, KINDS, art_window, inside_fraction, query_crop, sample_rect

CONTEXT_MARGIN = .125
ARRAYS = ('art', 'context', 'baseline')
BASELINE_SIZE = (128, 160)  # (w, h) of the shared portrait encoder input
LOCALES = ['ja', 'en', 'zh-Hant', 'zh-Hans', 'ko']
MIN_WINDOW_INSIDE = .6


def context_shape(kind):
    h, w = INPUT[kind]
    return round(h * (1 + 2 * CONTEXT_MARGIN)), round(w * (1 + 2 * CONTEXT_MARGIN))


def _tiles(row, kind):
    for index, card in enumerate(row.get('cards', [])):
        if card.get('kind', row.get('kind')) == kind:
            yield index, card, False
    for index, card in enumerate(row.get('foreign_cards', [])):
        if card.get('kind', row.get('kind')) == kind:
            yield index, card, True


def _crop_screenshot(job):
    path, row, kind, verify, arrays, bbox_key, resample = job
    raw = (path / row['file']).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if verify and row.get('sha256') and row['sha256'] != digest:
        raise ValueError(f"Image hash mismatch: {path / row['file']}")
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode {path / row['file']}")
    ih, iw = image.shape[:2]
    out = []
    for index, card, foreign in _tiles(row, kind):
        bbox = [float(v) for v in card.get(bbox_key) or card['bbox']]
        window = art_window(kind, bbox)
        window_inside = inside_fraction(window, (iw, ih))
        if window_inside < MIN_WINDOW_INSIDE:
            out.append(None)
            continue
        item = {
            'art': query_crop(image, kind, bbox, resample=resample) if 'art' in arrays else None,
            'context': query_crop(image, kind, bbox, CONTEXT_MARGIN) if 'context' in arrays else None,
            'foreign': foreign, 'id': card.get('id'), 'card_index': index,
            'window_inside': window_inside, 'tile_inside': inside_fraction(bbox, (iw, ih)),
            'visible_fraction': float(card.get('visible_fraction', 1.)),
            'occluded_fraction': float(card.get('occluded_fraction', 0.)),
            'foreign_method': str(card.get('foreign_method', '')) if foreign else '',
            'tile_width': bbox[2], 'display_mode': card.get('display_mode', row.get('display_mode', '')),
            'bbox': bbox,
        }
        if kind == 'snap' and 'baseline' in arrays:
            item['baseline'] = sample_rect(image, window, BASELINE_SIZE, resample=resample)
        out.append(item)
    return digest, out


def build(data, datasets, output, kind, workers=8, allow_incomplete=False, arrays=ARRAYS, stride=1, bbox_key='bbox',
          resample='linear'):
    """Write the crop bank of one kind. ``arrays`` selects which crop arrays to
    store; ``stride`` keeps every n-th screenshot that shows this kind;
    ``bbox_key`` names an alternative tile rectangle field (falls back to bbox);
    ``resample`` selects how evaluation crops are shrunk (training context is
    always bilinear)."""
    output = Path(output) / kind
    output.mkdir(parents=True, exist_ok=True)
    catalog = read(Path(data) / 'catalog.json')['cards']
    lookup = {(c['kind'], c['id']): i for i, c in enumerate(catalog)}
    manifests = []
    fingerprint = hashlib.sha256()
    for path in datasets:
        path = Path(path)
        document = read(path / 'truth.json')
        if not document.get('complete', False) and not allow_incomplete:
            raise ValueError(f'Incomplete dataset: {path}')
        fingerprint.update((path / 'truth.json').read_bytes())
        manifests.append((path, document))
    selected = [(d, s) for d, (_, document) in enumerate(manifests)
                for s, row in enumerate(document['screenshots']) if any(True for _ in _tiles(row, kind))][::stride]
    jobs = [(manifests[d][0], manifests[d][1]['screenshots'][s], kind, True, tuple(arrays), bbox_key, resample) for d, s in selected]
    capacity = sum(1 for job in jobs for _ in _tiles(job[1], kind))
    h, w = INPUT[kind]
    ch, cw = context_shape(kind)
    capacity = max(capacity, 1)
    art = open_memmap(output / 'art.npy', mode='w+', dtype='uint8', shape=(capacity, h, w, 3)) if 'art' in arrays else None
    context = open_memmap(output / 'context.npy', mode='w+', dtype='uint8',
                          shape=(capacity, ch, cw, 3)) if 'context' in arrays else None
    baseline = open_memmap(output / 'baseline.npy', mode='w+', dtype='uint8',
                           shape=(capacity, BASELINE_SIZE[1], BASELINE_SIZE[0], 3)) if kind == 'snap' and 'baseline' in arrays else None
    labels = np.full(capacity, -1, np.int32)
    columns = {name: [] for name in ['dataset', 'screenshot', 'card_index', 'foreign', 'window_inside',
                                     'tile_inside', 'visible_fraction', 'occluded_fraction', 'tile_width', 'locale',
                                     'mode', 'profile', 'foreign_method']}
    bboxes = []
    files = []
    count = 0
    skipped = Counter()
    unknown_ids = Counter()
    profiles = Counter()
    modes = []
    dataset_of_job = [d for d, _ in selected]
    screenshot_of_job = [s for _, s in selected]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for j, (digest, items) in enumerate(pool.map(_crop_screenshot, jobs, chunksize=4)):
            fingerprint.update(bytes.fromhex(digest))
            row = jobs[j][1]
            document = manifests[dataset_of_job[j]][1]
            for item in items:
                if item is None:
                    skipped['window_outside_image'] += 1
                    continue
                if not item['foreign']:
                    key = (kind, item['id'])
                    if key not in lookup:
                        unknown_ids[str(item['id'])] += 1
                        continue
                    labels[count] = lookup[key]
                if art is not None:
                    art[count] = item['art']
                if context is not None:
                    context[count] = item['context']
                if baseline is not None:
                    baseline[count] = item['baseline']
                mode = item['display_mode']
                if mode not in modes:
                    modes.append(mode)
                profile = document.get('profile', 'real')
                profiles[profile] += 1
                for name, value in [('dataset', dataset_of_job[j]), ('screenshot', screenshot_of_job[j]),
                                    ('card_index', item['card_index']), ('foreign', item['foreign']),
                                    ('window_inside', item['window_inside']), ('tile_inside', item['tile_inside']),
                                    ('visible_fraction', item['visible_fraction']),
                                    ('occluded_fraction', item['occluded_fraction']), ('tile_width', item['tile_width']),
                                    ('locale', LOCALES.index(row['locale']) if row.get('locale') in LOCALES else -1),
                                    ('mode', modes.index(mode)), ('profile', profile),
                                    ('foreign_method', item['foreign_method'])]:
                    columns[name].append(value)
                bboxes.append(item['bbox'])
                files.append(f"{dataset_of_job[j]}:{row['file']}")
                count += 1
            if (j + 1) % 200 == 0:
                print({'kind': kind, 'screenshots': j + 1, 'of': len(jobs), 'crops': count}, flush=True)
    for array in (art, context, baseline):
        if array is not None:
            array.flush()
    np.save(output / 'labels.npy', labels[:count])
    np.savez(output / 'meta.npz', bbox=np.array(bboxes, np.float32).reshape(-1, 4), source=np.array(files),
             **{k: np.array(v) for k, v in columns.items()})
    kind_cards = [[c['kind'], c['id']] for c in catalog]
    manifest = {
        'schema': 'bdon-kind-crop-bank/1', 'kind': kind, 'input_hw': list(INPUT[kind]),
        'context_hw': list(context_shape(kind)), 'context_margin': CONTEXT_MARGIN,
        'source_fingerprint': fingerprint.hexdigest(), 'catalog_ids': kind_cards,
        'capacity': capacity, 'count': count, 'arrays': [a for a in ARRAYS if a in arrays and (a != 'baseline' or kind == 'snap')],
        'screenshot_stride': stride, 'bbox_key': bbox_key, 'resample': resample, 'known': int((labels[:count] >= 0).sum()),
        'foreign': int((labels[:count] < 0).sum()), 'screenshots': len(jobs), 'profiles': dict(profiles),
        'display_modes': modes, 'locale_names': LOCALES, 'skipped': dict(skipped),
        'ids_outside_catalog': dict(unknown_ids), 'datasets': [str(p) for p, _ in manifests], 'complete': True,
    }
    write(output / 'manifest.json', manifest)
    print(manifest, flush=True)
    return manifest


def load_bank(path, catalog, kind, arrays=('art',)):
    """Load a bank as numpy arrays plus kind-local labels (-1 foreign)."""
    path = Path(path) / kind
    manifest = read(path / 'manifest.json')
    if not manifest.get('complete'):
        raise ValueError(f'Incomplete crop bank: {path}')
    if manifest['catalog_ids'] != [[c['kind'], c['id']] for c in catalog]:
        raise ValueError(f'Catalog order differs from crop bank {path}; rebuild it')
    count = manifest['count']
    kind_index = {i: k for k, i in enumerate(i for i, c in enumerate(catalog) if c['kind'] == kind)}
    labels = np.load(path / 'labels.npy')
    local = np.array([kind_index[int(v)] if v >= 0 else -1 for v in labels], np.int64)
    out = {name: np.load(path / f'{name}.npy', mmap_mode='r')[:count] for name in arrays}
    meta = dict(np.load(path / 'meta.npz', allow_pickle=False))
    return out, local, meta, manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--datasets', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--kinds', nargs='+', choices=KINDS, default=list(KINDS))
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--allow-incomplete', action='store_true',
                   help='accept truth files without a complete flag (hand-labelled regression sets)')
    p.add_argument('--arrays', nargs='+', choices=ARRAYS, default=list(ARRAYS),
                   help='crop arrays to store (training needs context; evaluation needs art, plus baseline for snap)')
    p.add_argument('--stride', type=int, default=1, help='keep every n-th screenshot that shows the kind')
    p.add_argument('--bbox-key', default='bbox', help='card field holding the tile rectangle')
    p.add_argument('--resample', choices=['linear', 'area'], default='linear', help='shrinking method of art crops')
    a = p.parse_args()
    for kind in a.kinds:
        build(a.data, a.datasets, a.output, kind, a.workers, a.allow_incomplete, a.arrays, a.stride, a.bbox_key, a.resample)
