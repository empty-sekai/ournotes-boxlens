"""Development evaluation of the per-kind field and rank readers.

Compares the per-kind field readers with a shared baseline field model on
identical crops and identical acceptance rules, and reports the per-kind rank
classifiers.

Scenes are read directly from ``truth.json`` and images; crops use the
deployed geometry in ``kind_fields``. The card pose is the truth tile, a
deterministically perturbed truth tile, or the tile of a stored detection
(``kind_pose detect``) paired with the truth tile; cards the detector misses
or assigns to the other kind count as unread. Results are stratified by kind,
locale, display mode, source resolution, acquisition profile, card width,
foreign artwork and held-out identity. Only development data belongs here.
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
from .kind_fields import (FIELD_CONFIDENCE, KINDS, RANK_CONFIDENCE, MARGIN, RANK_INPUT, RANK_SIDE, crop_field, crop_rank, decide_field,
                          decide_rank, field_inside, field_label, latent_field, latent_rank, load_bank, misread_sweep,
                          rank_icon_inside, rank_label, scene_cards, softmax, to_input)
from .kind_pose import detections_for, paired_tiles

STORED_RANK_MARGIN = MARGIN*RANK_INPUT//RANK_SIDE


def session(path, threads=8):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])


def run(sess, crops, batch=512):
    if len(crops) == 0:
        return np.zeros((0, sess.get_outputs()[0].shape[1] or 1), np.float64)
    out = [softmax(sess.run(None, {sess.get_inputs()[0].name: to_input(crops[i:i+batch])})[0])
           for i in range(0, len(crops), batch)]
    return np.concatenate(out)


def field_value(kind, decision):
    if decision['level'] is not None:
        return decision['level']
    if decision['awake_count'] is not None:
        return 100+decision['awake_count']
    return 0


def tally(counter, truth, value, prefix='', latent=-1):
    """truth/value: 0 = no value; counts correct, wrong, unknown, false visible.

    ``latent`` (card state, analysis only; -1 unknown) splits values emitted
    for hidden targets into those matching the card state and misreads.
    """
    if truth > 0:
        counter[prefix+'visible_total'] += 1
        counter[prefix+'correct'] += int(value == truth)
        counter[prefix+'wrong'] += int(value > 0 and value != truth)
        counter[prefix+'unknown'] += int(value == 0)
    else:
        counter[prefix+'hidden_total'] += 1
        counter[prefix+'false_visible'] += int(value > 0)
        counter[prefix+'false_visible_latent_match'] += int(value > 0 and value == latent)


def finish(counter):
    out = dict(counter)
    for prefix in sorted({k[:-len('visible_total')] for k in counter if k.endswith('visible_total')}):
        total = counter[prefix+'visible_total']
        out[prefix+'coverage'] = round(counter[prefix+'correct']/total, 6) if total else None
        out[prefix+'misreads'] = (counter[prefix+'wrong']+counter[prefix+'false_visible']
                                  - counter[prefix+'false_visible_latent_match'])
    return out


WORKER = {}


def _init_worker(jitter=0., tile_key='bbox', detections=None):
    cv2.setNumThreads(1)
    WORKER['jitter'] = jitter
    WORKER['tile_key'] = tile_key
    WORKER['detections'] = detections


def perturb(tile, jitter, key):
    """Deterministic pose error: shift by up to ``jitter`` of the size, scale by up to ``jitter``."""
    if not jitter:
        return tile
    rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:12], 16))
    dx, dy, ds = rng.uniform(-jitter, jitter, 3)
    x, y, w, h = tile
    return [x+dx*w, y+dy*h, w*(1+ds), h*(1+ds)]


def _scene(job):
    path, row, dataset = job
    image = cv2.imread(str(Path(path)/row['file']))
    if image is None:
        raise ValueError(f"Cannot decode {Path(path)/row['file']}")
    ih, iw = image.shape[:2]
    tile_key = WORKER.get('tile_key', 'bbox')
    detections = WORKER.get('detections')
    located = paired_tiles(row, detections[dataset][row['file']], tile_key)[0] if detections is not None else None
    acquisition = row.get('acquisition', {})
    rows = []
    for index, (card, foreign) in enumerate(scene_cards(row)):
        kind = card['kind']
        reference = perturb([float(v) for v in card[tile_key]], WORKER.get('jitter', 0.), f"{dataset}/{row['file']}/{index}")
        tile = reference if located is None else located[index]
        f_label = field_label(kind, card)
        f_inside = field_inside(kind, reference, iw, ih)
        r_inside = rank_icon_inside(kind, reference, iw, ih)
        f_read = tile is not None and f_label is not None and f_inside and field_inside(kind, tile, iw, ih)
        r_read = tile is not None and r_inside and rank_icon_inside(kind, tile, iw, ih)
        item = {'dataset': dataset, 'file': row['file'], 'card': index, 'kind': kind, 'id': card.get('id'),
                'foreign': foreign, 'locale': row.get('locale', 'unknown'), 'mode': card.get('display_mode', 'level'),
                'profile': acquisition.get('profile', 'unknown'),
                'source_resolution': 'x'.join(map(str, acquisition.get('source_resolution', [iw, ih]))),
                'width': reference[2], 'field_label': f_label, 'field_inside': f_inside, 'located': tile is not None,
                'field_latent': latent_field(kind, card), 'rank_latent': latent_rank(card),
                'field_occluded': not card.get('field_visible', True), 'rank_occluded': card.get('rank_visible') is False,
                'rank_label': rank_label(card), 'rank_inside': r_inside,
                'field_crop': crop_field(image, kind, tile) if f_read else None,
                'rank_crop': crop_rank(image, kind, tile) if r_read else None}
        rows.append(item)
    return rows


def strata_keys(item):
    keys = ['overall', 'locale:'+item['locale'], 'mode:'+item['mode'], 'profile:'+item['profile'],
            'source_resolution:'+item['source_resolution'], 'card_width_bin:'+str(int(item['width']//30)*30),
            'foreign:'+str(item['foreign'])]
    if item['field_occluded'] and item['field_inside']:
        keys.append('field_occluded')
    if item['id'] is not None and not item['foreign']:
        keys.append('identity:'+('heldout' if item['id'] % 7 == 0 else 'seen'))
    return keys


def scatter(probabilities, index, total):
    """Full probability table with unread cards set to certain class 0 (no value at any threshold)."""
    if probabilities is None:
        return None
    out = np.zeros((total, probabilities.shape[1]), np.float64)
    out[:, 0] = 1.
    out[index] = probabilities
    return out


def evaluate_scenes(datasets, models, baseline, output, workers=8, jitter=0., tile_key='bbox', detections=None):
    """``jitter``: deterministic relative pose error applied to every truth tile (0 = exact).
    ``tile_key``: truth tile field of each card. ``detections``: stored detections (``kind_pose detect``)
    whose paired tiles replace the truth tiles; the evaluated population stays defined by the truth tiles."""
    jobs = []
    fingerprint = hashlib.sha256()
    stored = read(detections) if detections else None
    located = {} if stored else None
    for d, path in enumerate(map(Path, datasets)):
        document = read(path/'truth.json')
        if not document.get('complete', False):
            raise ValueError(f'Incomplete dataset: {path}')
        fingerprint.update((path/'truth.json').read_bytes())
        jobs += [(str(path), row, d) for row in document['screenshots']]
        if stored:
            located[d] = detections_for(stored, path)
    pose = {'jitter': jitter, 'tile_key': tile_key,
            'detections': {'path': str(detections), 'sha256': hashlib.sha256(Path(detections).read_bytes()).hexdigest(),
                           'locator': stored['locator']} if stored else None}
    items = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(jitter, tile_key, located)) as pool:
        for done, rows in enumerate(pool.map(_scene, jobs, chunksize=4)):
            items += rows
            if (done+1) % 500 == 0:
                print(json.dumps({'scenes': done+1, 'total': len(jobs)}), flush=True)
    sessions = {name: session(path) for name, path in models.items()}
    thresholds = acceptance(models)
    base = session(baseline) if baseline else None
    report = {'schema': 'ournotes-boxlens.kind-field-evaluation/1', 'datasets': [str(p) for p in datasets],
              'truth_fingerprint': fingerprint.hexdigest(), 'scenes': len(jobs),
              'models': {k: {'path': str(v), 'sha256': hashlib.sha256(Path(v).read_bytes()).hexdigest(),
                             'confidence_threshold': thresholds[k]} for k, v in models.items()},
              'baseline_fields': {'path': str(baseline), 'sha256': hashlib.sha256(Path(baseline).read_bytes()).hexdigest()} if baseline else None,
              'pose_jitter': jitter, 'pose': pose, 'kinds': {}}
    for kind in KINDS:
        fields = [i for i in items if i['kind'] == kind and i['field_label'] is not None and i['field_inside']]
        ranks = [i for i in items if i['kind'] == kind and i['rank_inside']]
        strata = defaultdict(Counter)
        read_fields = [n for n, i in enumerate(fields) if i['field_crop'] is not None]
        field_crops = (np.stack([fields[n]['field_crop'] for n in read_fields]) if read_fields
                       else np.zeros((0, 56, 144, 3), np.uint8))
        new_p = scatter(run(sessions[f'{kind}-fields'], field_crops), read_fields, len(fields)) if f'{kind}-fields' in sessions else None
        base_p = scatter(run(base, field_crops), read_fields, len(fields)) if base else None
        for n, item in enumerate(fields):
            truth = item['field_label']
            for key in strata_keys(item):
                if new_p is not None:
                    tally(strata[key], truth, field_value(kind, decide_field(kind, new_p[n], thresholds[f'{kind}-fields'])), 'field_new_',
                          item['field_latent'])
                if base_p is not None:
                    tally(strata[key], truth, field_value(kind, decide_field(kind, base_p[n])), 'field_baseline_',
                          item['field_latent'])
        read_ranks = [n for n, i in enumerate(ranks) if i['rank_crop'] is not None]
        rank_crops = (np.stack([ranks[n]['rank_crop'] for n in read_ranks]) if read_ranks
                      else np.zeros((0, 48, 48, 3), np.uint8))
        rank_p = scatter(run(sessions[f'{kind}-rank'], rank_crops), read_ranks, len(ranks)) if f'{kind}-rank' in sessions else None
        examples = []
        for n, item in enumerate(ranks):
            truth = item['rank_label']
            new_value = (decide_rank(rank_p[n], thresholds[f'{kind}-rank'])['value'] or 0) if rank_p is not None else None
            for key in strata_keys(item):
                latent = item['rank_latent']
                if new_value is not None:
                    tally(strata[key], truth, new_value, 'rank_new_', latent)
            if new_value is not None and new_value > 0 and new_value != truth and len(examples) < 50:
                examples.append({k: item[k] for k in ('dataset', 'file', 'card', 'id', 'mode', 'width', 'rank_label')} | {'predicted': new_value})
        limit = 105 if kind == 'member' else 100
        f_truth = np.array([i['field_label'] for i in fields])
        f_latent = np.array([i['field_latent'] for i in fields])
        r_truth = np.array([i['rank_label'] for i in ranks])
        r_latent = np.array([i['rank_latent'] for i in ranks])
        sweeps = {}
        if new_p is not None and len(fields):
            sweeps['field_new'] = misread_sweep(new_p, f_truth, f_latent, limit, default=thresholds[f'{kind}-fields'])
        if base_p is not None and len(fields):
            sweeps['field_baseline'] = misread_sweep(base_p, f_truth, f_latent, limit)
        if rank_p is not None and len(ranks):
            sweeps['rank_new'] = misread_sweep(rank_p, r_truth, r_latent, margin=.5, default=thresholds[f'{kind}-rank'])
        report['kinds'][kind] = {'field_crops': len(fields), 'rank_crops': len(ranks), 'zero_misread': sweeps,
                                 'field_cards_not_read_for_pose': len(fields)-len(read_fields),
                                 'rank_cards_not_read_for_pose': len(ranks)-len(read_ranks),
                                 'rank_cards_outside_image': sum(1 for i in items if i['kind'] == kind and not i['rank_inside']),
                                 'field_cards_outside_image': sum(1 for i in items if i['kind'] == kind and i['field_label'] is not None and not i['field_inside']),
                                 'strata': {k: finish(v) for k, v in sorted(strata.items())}, 'rank_error_examples': examples}
        print(json.dumps({'kind': kind, 'overall': report['kinds'][kind]['strata'].get('overall')}), flush=True)
    report['scope'] = (('Card pose from stored detections paired with truth tiles; cards missed or detected as the '
                        'other kind, or whose detected crop leaves the image, count as unread. ' if stored else
                        'Truth-derived card pose for every method (oracle localization). ') +
                       'Field targets: visible level/training values, 0 = other or hidden parameter. Rank targets: '
                       'visible icon 1..5, 0 = no icon; only icons fully inside the image are scored.')
    write(output, report)
    return report


def evaluate_bank(bank, kind, models, baseline, output):
    """Crop-bank evaluation (e.g. acquisition-degraded validation crops)."""
    report = {'bank': str(bank), 'kind': kind}
    images, labels, meta = load_bank(bank, kind, 'field')
    crops = np.asarray(images)[:, MARGIN:MARGIN+56, MARGIN:MARGIN+144]
    thresholds = acceptance(models)
    for name, path in [('new', models.get(f'{kind}-fields')), ('baseline', baseline)]:
        if path is None:
            continue
        threshold = thresholds[f'{kind}-fields'] if name == 'new' else FIELD_CONFIDENCE
        report['field_'+name+'_threshold'] = threshold
        p = run(session(path), crops)
        counter = Counter()
        latent = meta.get('latent', np.full(len(labels), -1))
        for prob, truth, state in zip(p, labels, latent):
            tally(counter, int(truth), field_value(kind, decide_field(kind, prob, threshold)), latent=int(state))
        report['field_'+name] = finish(counter)
        report['field_'+name+'_zero_misread'] = misread_sweep(p, labels, latent, 105 if kind == 'member' else 100,
                                                              default=threshold)
    if models.get(f'{kind}-rank'):
        images, labels, meta = load_bank(bank, kind, 'rank')
        m = STORED_RANK_MARGIN
        crops = np.asarray(images)[:, m:m+RANK_INPUT, m:m+RANK_INPUT]
        p = run(session(models[f'{kind}-rank']), crops)
        counter = Counter()
        latent = meta.get('latent', np.full(len(labels), -1))
        for prob, truth, state in zip(p, labels, latent):
            tally(counter, int(truth), decide_rank(prob, thresholds[f'{kind}-rank'])['value'] or 0, latent=int(state))
        report['rank_new'] = finish(counter)
        report['rank_new_threshold'] = thresholds[f'{kind}-rank']
        report['rank_new_zero_misread'] = misread_sweep(p, labels, latent, margin=.5, default=thresholds[f'{kind}-rank'])
    write(output, report)
    print(json.dumps(report), flush=True)
    return report


def acceptance(models):
    """Per-model confidence threshold from ``selection.json`` next to the ONNX file, else the default."""
    out = {}
    for name, path in models.items():
        selection = Path(path).parent/'selection.json'
        default = FIELD_CONFIDENCE if name.endswith('fields') else RANK_CONFIDENCE
        out[name] = read(selection).get('acceptance', {}).get('confidence', default) if selection.exists() else default
    return out


def parse_models(values):
    models = {}
    for value in values or []:
        name, path = value.split('=', 1)
        if name not in [f'{k}-{p}' for k in KINDS for p in ('fields', 'rank')]:
            raise ValueError(f'Unknown model name: {name}')
        models[name] = Path(path)
    return models


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    s = sub.add_parser('scenes')
    s.add_argument('--datasets', nargs='+', type=Path, required=True)
    s.add_argument('--model', action='append', help='NAME=PATH, NAME in member-fields|snap-fields|member-rank|snap-rank')
    s.add_argument('--baseline', type=Path, help='Shared field model ONNX for comparison')
    s.add_argument('--output', type=Path, required=True)
    s.add_argument('--workers', type=int, default=8)
    s.add_argument('--pose-jitter', type=float, default=0., help='Relative pose error applied to truth tiles')
    s.add_argument('--tile-key', default='bbox', help='Truth tile field of each card (bbox or bbox_layout)')
    s.add_argument('--detections', type=Path, help='Stored detections whose paired tiles replace the truth tiles')
    b = sub.add_parser('bank')
    b.add_argument('--bank', type=Path, required=True)
    b.add_argument('--kind', choices=KINDS, required=True)
    b.add_argument('--model', action='append')
    b.add_argument('--baseline', type=Path)
    b.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.action == 'scenes':
        evaluate_scenes(a.datasets, parse_models(a.model), a.baseline, a.output, a.workers, a.pose_jitter,
                        a.tile_key, a.detections)
    else:
        evaluate_bank(a.bank, a.kind, parse_models(a.model), a.baseline, a.output)
