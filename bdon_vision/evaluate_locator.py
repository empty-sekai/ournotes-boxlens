"""Evaluate card-frame locators on scene and real sets.

Writes one JSON report with recall / precision at IoU 0.5 and 0.75, card kind
accuracy, strata by card width and resolution, and per-image results for
real screenshots. The score threshold is chosen on the first scene set
(normally validation) and then applied unchanged to every set.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .locator import CLASSES, Locator, decode, letterbox, to_tensor
from .locator_eval import Accumulator, load_scenes, match
from .train_locator import expand, write_json


def load_real(directory):
    """Real screenshots with approximate tile rectangles (``screenshots[].cards[].bbox``)."""
    directory = Path(directory)
    truth = json.loads((directory / 'truth.json').read_text(encoding='utf-8'))
    scenes = []
    for row in truth['screenshots']:
        cards = row.get('cards', []) + [dict(c, foreign=True) for c in row.get('foreign_cards', [])]
        scenes.append({'path': str(directory / row['file']), 'chunk': directory.name,
                       'boxes': np.asarray([c['bbox'] for c in cards], np.float32).reshape(-1, 4),
                       'labels': np.asarray([CLASSES.index(c['kind']) for c in cards], np.int64),
                       'ids': [c.get('id') for c in cards],
                       'foreign': np.asarray([bool(c.get('foreign')) for c in cards], bool),
                       'ignored_rows': row.get('ignored_cards', []), 'source_resolution': None, 'profile': 'real'})
    return scenes


def read_rgb(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'unreadable image: {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class TorchPredictor:
    def __init__(self, weights, width=96):
        import torch
        from .locator_model import LocatorNet
        self.torch = torch
        self.model = LocatorNet(width=width)
        self.model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        self.model = self.model.cuda().eval()

    def raw(self, image, long_edge):
        torch = self.torch
        canvas, scale = letterbox(image, long_edge)
        with torch.inference_mode():
            heat, size, offset = self.model(torch.from_numpy(to_tensor(canvas)).cuda())
            heat = torch.sigmoid(heat)[0].cpu().numpy()
            size = (torch.exp(size.clamp(max=8)) * 8)[0].cpu().numpy()
            offset = offset[0].cpu().numpy()
        return heat, size, offset, scale


class OnnxPredictor:
    def __init__(self, model, threads=4):
        self.locators = {}
        self.model, self.threads = model, threads

    def raw(self, image, long_edge):
        if long_edge not in self.locators:
            self.locators[long_edge] = Locator(self.model, long_edge, self.threads)
        return self.locators[long_edge].raw(image)


def evaluate_sets(predictor, sets, sizes, threshold=None, real=None, overlay=None, aspect_tolerance=None):
    report = {'sizes': {}}
    for size in sizes:
        accs, per_image, timings = {}, {}, []
        for name, scenes in sets:
            acc = Accumulator()
            for scene in scenes:
                image = read_rgb(scene['path'])
                h, w = image.shape[:2]
                started = time.perf_counter()
                heat, wh, offset, scale = predictor.raw(image, size)
                timings.append(time.perf_counter() - started)
                acc.add(decode(heat, wh, offset, scale, .05, image_size=(w, h), aspect_tolerance=aspect_tolerance),
                        scene, w, h, scale)
            accs[name] = acc
        chosen = threshold if threshold is not None else accs[sets[0][0]].best_threshold(.5)
        entry = {'threshold': chosen, 'threshold_source': 'fixed' if threshold is not None else f'best F1@0.5 on {sets[0][0]}',
                 'aspect_tolerance': aspect_tolerance, 'sets': {}}
        for name, acc in accs.items():
            entry['sets'][name] = {'ap50': acc.average_precision(.5), 'ap75': acc.average_precision(.75),
                                   'strata': acc.summary(chosen)}
        if real:
            entry['real'] = evaluate_real(predictor, real, size, chosen, overlay, aspect_tolerance)
        report['sizes'][str(size)] = entry
        print(json.dumps({'size': size, 'threshold': chosen,
                          **{n: {'ap50': round(e['ap50'], 4), 'ap75': round(e['ap75'], 4),
                                 'overall': e['strata'].get('overall')} for n, e in entry['sets'].items()}}), flush=True)
    return report


def _top_bar(box, height):
    return bool(box[1] + box[3] / 2 < .1 * height)


def evaluate_real(predictor, scenes, size, threshold, overlay=None, aspect_tolerance=None):
    rows = []
    for scene in scenes:
        image = read_rgb(scene['path'])
        h, w = image.shape[:2]
        heat, wh, offset, scale = predictor.raw(image, size)
        detections = decode(heat, wh, offset, scale, threshold, image_size=(w, h), aspect_tolerance=aspect_tolerance)
        rows.append(real_row(scene, image, detections, overlay, f'{size}'))
    return rows


def real_row(scene, image, detections, overlay=None, tag=''):
    h, w = image.shape[:2]
    result, truth = match(detections, scene['boxes'], scene['labels'], np.zeros((0, 4)))
    order = sorted(range(len(detections)), key=lambda i: -detections[i]['score'])
    false_positives = [dict(detections[order[i]], top_bar=_top_bar(detections[order[i]]['bbox'], h))
                       for i, r in enumerate(result) if r[.5] == 'fp']
    cards = []
    for g in range(len(scene['boxes'])):
        m = truth[.5][g]
        cards.append({'kind': CLASSES[int(scene['labels'][g])], 'id': scene['ids'][g] if 'ids' in scene else None,
                      'found': m is not None, 'kind_correct': bool(m and m[1]),
                      'found_iou75': truth[.75][g] is not None, 'score': m[0] if m else None})
    if overlay:
        canvas = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for b in scene['boxes']:
            x, y, bw, bh = map(int, b)
            cv2.rectangle(canvas, (x, y), (x + bw, y + bh), (0, 200, 0), 2)
        for d in detections:
            x, y, bw, bh = map(int, d['bbox'])
            color = (255, 80, 0) if d['kind'] == 'member' else (0, 80, 255)
            cv2.rectangle(canvas, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(canvas, f"{d['kind'][0]}{d['score']:.2f}", (x + 3, y + 18), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
        Path(overlay).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(Path(overlay) / f'{tag}-{Path(scene["path"]).stem}.jpg'), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {'file': Path(scene['path']).name, 'truth': len(scene['boxes']), 'detections': len(detections),
            'found': sum(c['found'] for c in cards), 'kind_correct': sum(c['kind_correct'] for c in cards),
            'false_positives': len(false_positives), 'false_positives_in_top_bar': sum(f['top_bar'] for f in false_positives),
            'false_positive_boxes': false_positives, 'cards': cards}


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--model', help='Locator state dict (.pt, GPU) or exported .onnx (CPU)')
    p.add_argument('--width', type=int, default=96)
    p.add_argument('--set', action='append', default=[], metavar='NAME=GLOB',
                   help='Scene set; the first one selects the score threshold')
    p.add_argument('--real', nargs='*', default=[], help='Directories with real screenshots and truth.json')
    p.add_argument('--sizes', type=int, nargs='+', default=[640])
    p.add_argument('--threshold', type=float)
    p.add_argument('--aspect-tolerance', type=float, help='Drop tiles whose aspect differs from the class by more')
    p.add_argument('--threads', type=int, default=4, help='onnxruntime CPU threads')
    p.add_argument('--overlay', help='Directory for annotated real screenshots')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    sets = []
    for spec in a.set:
        name, pattern = spec.split('=', 1)
        sets.append((name, load_scenes(expand(pattern.split(',')))))
    real = [s for d in a.real for s in load_real(d)]
    report = {'sets': {name: len(scenes) for name, scenes in sets}, 'real_images': len(real)}
    if a.model:
        predictor = OnnxPredictor(a.model, a.threads) if a.model.endswith('.onnx') else TorchPredictor(a.model, a.width)
        report['locator'] = evaluate_sets(predictor, sets, a.sizes, a.threshold, real, a.overlay, a.aspect_tolerance)
    write_json(a.output, report)


if __name__ == '__main__':
    main()
