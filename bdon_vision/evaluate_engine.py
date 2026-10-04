"""End-to-end evaluation of ``Engine.scan`` on scene sets with tile truth.

Every truth card (``cards`` and ``foreign_cards`` of each screenshot in
``truth.json``) is paired one-to-one with the scan output tile of the highest
IoU at or above 0.5; ``cards`` and ``unidentified`` outputs both take part.
Outputs that pair with no truth card and overlap no ignored tile count as
extra detections.

Per paired card the report scores:

* identity: a gallery card must come back with its own ID (correct), another
  ID (wrong) or unidentified (unknown); a foreign card (artwork outside the
  gallery) must come back unidentified, otherwise it is a false accept;
* level, member training count and card rank: a visible target is correct,
  wrong or unknown; a value emitted for a hidden target is a misread unless it
  equals the card state. A target is visible when the list shows it and its
  crop lies fully inside the image (the field box, or the whole rank icon).

The module needs only the scan output contract, so the same file scores any
engine version that implements ``Engine(data).scan(image)``.
"""
import argparse
import hashlib
import json
import os
import platform
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .assets import read, write

LOGICAL = {'member': (224, 294), 'snap': (326, 184)}
RANK_CENTER = {'member': (194.5, 267.2), 'snap': (306.1, 153.8)}
FIELDS = ('level', 'awake_count', 'card_rank')
WORKER = {}


def iou(a, b):
    ix = max(0., min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0]))
    iy = max(0., min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    inter = ix*iy
    return inter/max(1e-9, a[2]*a[3]+b[2]*b[3]-inter)


def field_inside(kind, tile, width, height, border=2.):
    x, y, w, h = tile
    s = w/LOGICAL[kind][0]
    left, top = x-6*s, y+h-56*s
    return left >= border and top >= border and left+144*s <= width-border and top+56*s <= height-border


def rank_inside(kind, tile, width, height, border=2.):
    x, y, w, _ = tile
    s = w/LOGICAL[kind][0]
    cx, cy = x+RANK_CENTER[kind][0]*s, y+RANK_CENTER[kind][1]*s
    hw, hh = 32*s, 32*95/102*s
    return cx-hw >= border and cy-hh >= border and cx+hw <= width-border and cy+hh <= height-border


def targets(card, width, height):
    """{field: (visible value or None, card-state value or None)} for one truth card."""
    kind, mode = card['kind'], card.get('display_mode', 'level')
    latent = card.get('latent_state') or {}
    shown = card.get('field_visible', True) and field_inside(kind, card['bbox'], width, height)
    out = {'level': (card.get('level') if shown and mode == 'level' else None, latent.get('level')),
           'card_rank': (card.get('card_rank') if rank_inside(kind, card['bbox'], width, height) else None,
                         0 if mode == 'hide' else latent.get('card_rank'))}
    if kind == 'member':
        out['awake_count'] = (card.get('awake_count') if shown and mode == 'training' else None, latent.get('awake_count'))
    return out


def pair(truth, outputs):
    """Greedy one-to-one pairing by IoU >= 0.5; returns {truth index: output index}."""
    candidates = sorted(((iou(t['bbox'], o['bbox']), i, j) for i, t in enumerate(truth) for j, o in enumerate(outputs)),
                        reverse=True)
    used_t, used_o, out = set(), set(), {}
    for score, i, j in candidates:
        if score < .5:
            break
        if i in used_t or j in used_o:
            continue
        used_t.add(i)
        used_o.add(j)
        out[i] = j
    return out


def strata(row, card, foreign, width):
    acquisition = row.get('acquisition', {})
    keys = ['overall', 'kind:'+card['kind'], 'profile:'+acquisition.get('profile', 'unknown'),
            'locale:'+row.get('locale', 'unknown'), 'mode:'+card.get('display_mode', 'level'),
            'source_resolution:'+'x'.join(map(str, acquisition.get('source_resolution', ['unknown']))),
            'card_width_bin:'+str(int(card['bbox'][2]//60)*60), 'foreign:'+str(foreign)]
    if not foreign:
        keys.append('identity:'+('heldout' if card['id'] % 7 == 0 else 'seen'))
    if card.get('occluded_fraction', 0) > 0:
        keys.append('occluded')
    if card.get('visible_fraction', 1.) < 1:
        keys.append('partly_outside')
    return keys


def score_screen(row, scan):
    height, width = scan['height'], scan['width']
    truth = [(c, False) for c in row.get('cards', [])] + [(c, True) for c in row.get('foreign_cards', [])]
    ignored = row.get('ignored_cards', []) + row.get('ignored_foreign_cards', [])
    outputs = scan['cards'] + scan.get('unidentified', [])
    pairs = pair([c for c, _ in truth], outputs)
    counts = defaultdict(Counter)
    errors = []
    for i, (card, foreign) in enumerate(truth):
        keys = strata(row, card, foreign, width)
        found = outputs[pairs[i]] if i in pairs else None
        result = Counter({'cards': 1, 'located': int(found is not None)})
        if found is not None:
            result['kind_correct'] += int(found['kind'] == card['kind'])
            predicted = found.get('id')
            if foreign:
                result['foreign_rejected' if predicted is None else 'foreign_false_accept'] += 1
            elif predicted is None:
                result['identity_unknown'] += 1
            else:
                correct = predicted == card['id'] and found['kind'] == card['kind']
                result['identity_correct' if correct else 'identity_wrong'] += 1
                if not correct:
                    errors.append({'file': row['file'], 'card': i, 'error': 'identity', 'expected': [card['kind'], card['id']],
                                   'actual': [found['kind'], predicted]})
            for name, (visible, latent) in targets(card, width, height).items():
                value = found[name]['value'] if name in found else None
                if visible is not None:
                    result[name+'_visible'] += 1
                    result[name+'_correct'] += int(value == visible)
                    result[name+'_unknown'] += int(value is None)
                    if value is not None and value != visible:
                        result[name+'_wrong'] += 1
                        errors.append({'file': row['file'], 'card': i, 'error': name, 'expected': visible, 'actual': value})
                elif value is not None:
                    match = value == latent
                    result[name+'_hidden_emitted'] += 1
                    result[name+('_hidden_latent_match' if match else '_hidden_misread')] += 1
                    if not match:
                        errors.append({'file': row['file'], 'card': i, 'error': name+'_hidden', 'expected': None,
                                       'latent': latent, 'actual': value})
        elif not foreign:
            for name, (visible, _) in targets(card, width, height).items():
                result[name+'_visible'] += int(visible is not None)
                result[name+'_unknown'] += int(visible is not None)
        for key in keys:
            counts[key].update(result)
    paired = set(pairs.values())
    extra = [o for j, o in enumerate(outputs) if j not in paired and not any(iou(o['bbox'], g['bbox']) >= .5 for g in ignored)]
    screen = Counter({'screens': 1, 'negative_screens': int(bool(row.get('negative'))),
                      'extra_identified': sum(o.get('id') is not None for o in extra),
                      'extra_unidentified': sum(o.get('id') is None for o in extra)})
    for key in ['overall', 'profile:'+row.get('acquisition', {}).get('profile', 'unknown'), 'locale:'+row.get('locale', 'unknown')]:
        counts[key].update(screen)
    return counts, errors


def summarize(counter):
    out = dict(counter)
    known = counter['cards']-counter['foreign_rejected']-counter['foreign_false_accept']
    accepted = counter['identity_correct']+counter['identity_wrong']
    out['identity_recall'] = round(counter['identity_correct']/known, 6) if known else None
    out['identity_precision'] = round(counter['identity_correct']/accepted, 6) if accepted else None
    for name in FIELDS:
        total = counter[name+'_visible']
        out[name+'_coverage'] = round(counter[name+'_correct']/total, 6) if total else None
        out[name+'_misreads'] = counter[name+'_wrong']+counter[name+'_hidden_misread']
    return out


def _init(data, threads):
    cv2.setNumThreads(threads)
    from .engine import Engine
    WORKER['engine'] = Engine(data, threads=threads)


def _scan(job):
    path, row = job
    image = cv2.imread(str(Path(path)/row['file']), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode {Path(path)/row['file']}")
    scan = WORKER['engine'].scan(image, row['file'])
    counts, errors = score_screen(row, scan)
    return {k: dict(v) for k, v in counts.items()}, errors, scan['elapsed_ms'], scan


def evaluate(data, sets, output, workers=8, threads=1, keep_scans=False):
    report = {'schema': 'ournotes-boxlens.engine-evaluation/1', 'data': str(data), 'sets': {},
              'platform': platform.platform(), 'workers': workers, 'threads_per_worker': threads,
              'matching': 'one-to-one tile pairing at IoU >= 0.5; cards and unidentified outputs both pair',
              'scope': __doc__.strip().split('\n\n')[0]}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(str(data), threads)) as pool:
        report['provenance'] = None
        for name, path in sets:
            path = Path(path)
            document = read(path/'truth.json')
            if not document.get('complete', False):
                raise ValueError(f'Incomplete dataset: {path}')
            jobs = [(str(path), row) for row in document['screenshots']]
            counts, errors, timings, scans = defaultdict(Counter), [], [], []
            for done, (screen, screen_errors, elapsed, scan) in enumerate(pool.map(_scan, jobs, chunksize=2)):
                for key, value in screen.items():
                    counts[key].update(value)
                errors += screen_errors
                timings.append(elapsed)
                report['provenance'] = scan.get('recognition_provenance', report['provenance'])
                if keep_scans:
                    scans.append(scan)
                if (done+1) % 250 == 0:
                    print(json.dumps({'set': name, 'screens': done+1, 'total': len(jobs)}), flush=True)
            report['sets'][name] = {
                'path': str(path), 'truth_sha256': hashlib.sha256((path/'truth.json').read_bytes()).hexdigest(),
                'screens': len(jobs), 'latency_ms': {'median': float(np.median(timings)), 'p95': float(np.percentile(timings, 95))},
                'strata': {key: summarize(value) for key, value in sorted(counts.items())},
                'errors': errors[:500], 'error_count': len(errors)}
            if keep_scans:
                write(Path(output).with_name(Path(output).stem+f'-{name}-scans.json'), scans)
            overall = report['sets'][name]['strata'].get('overall', {})
            print(json.dumps({'set': name, **{k: overall.get(k) for k in ('cards', 'located', 'identity_recall', 'identity_precision',
                              'foreign_false_accept', 'level_coverage', 'level_misreads', 'awake_count_coverage',
                              'awake_count_misreads', 'card_rank_coverage', 'card_rank_misreads', 'extra_identified',
                              'extra_unidentified')}}), flush=True)
    write(output, report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--set', action='append', required=True, metavar='NAME=DIR')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--keep-scans', action='store_true')
    a = p.parse_args()
    evaluate(a.data, [s.split('=', 1) for s in a.set], a.output, a.workers, a.threads, a.keep_scans)
