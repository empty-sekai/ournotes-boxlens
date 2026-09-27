"""Evaluate opaque overlays and viewport fades without inventing readability labels.

Uses saved development scenes, never modifies source images or frozen tests.
Opaque-covered values must remain unknown. Faded values have uncertain human
readability, so their output rate and latent agreement are descriptive only.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from bdon_vision.assets import read, write
from bdon_vision.engine import Engine
from bdon_vision.synthetic import bbox_iou

FIELDS = ('level', 'card_rank', 'awake_count')


def modify(image, cards, effect):
    result = image.copy()
    regions = []
    for card in cards:
        x, y, w, h = card['bbox']
        # Cover the complete lower UI band, including parameter and rank glyphs.
        left, right = max(0, int(x-w*.05)), min(image.shape[1], int(np.ceil(x+w*1.05)))
        top, bottom = max(0, int(y+h*.68)), min(image.shape[0], int(np.ceil(y+h*1.03)))
        if right <= left or bottom <= top:
            continue
        if effect == 'opaque':
            alpha = np.ones((bottom-top, 1, 1), np.float32)
        else:
            alpha = np.linspace(0, .98, bottom-top, dtype=np.float32)[:, None, None]
        color = np.array([220, 225, 229], np.float32).reshape(1, 1, 3)
        region = result[top:bottom, left:right].astype(np.float32)
        result[top:bottom, left:right] = np.rint(region*(1-alpha)+color*alpha).astype(np.uint8)
        regions.append([left, top, right-left, bottom-top])
    return result, regions


def run(data, datasets, output):
    output.mkdir(parents=True, exist_ok=False)
    engine = Engine(data, threads=2)
    groups = defaultdict(Counter)
    details, identities, effects = [], set(), ('opaque', 'fade')
    catalog = {(c['kind'], c['id']): c for c in engine.cards}
    for dataset in datasets:
        truth = read(dataset/'truth.json')
        for row_index, row in enumerate(truth['screenshots']):
            original = cv2.imread(str(dataset/row['file']))
            if original is None:
                raise ValueError(f"Unreadable image: {dataset/row['file']}")
            for effect in effects:
                scene, regions = modify(original, row['cards'], effect)
                name = f'{dataset.parent.name}-{dataset.name}-{row_index:06d}-{effect}'
                scan = engine.scan(scene, name)
                errors = []
                for card in row['cards']:
                    identity = card['kind'], card['id']
                    identities.add(identity)
                    meta = catalog[identity]
                    mode = card.get('display_mode', row.get('display_mode', 'unrecorded'))
                    keys = [effect, f"{effect}/rarity/{card['kind']}/{meta['rarity']}",
                            f"{effect}/type/{meta['card_type']}", f"{effect}/locale/{row.get('locale','unrecorded')}",
                            f'{effect}/mode/{mode}', f"{effect}/profile/{truth.get('profile','unrecorded')}"]
                    found = [c for c in scan['cards'] if (c['kind'], c['id']) == identity
                             and bbox_iou(c['bbox'], card['bbox']) >= .5]
                    found = max(found, key=lambda c: bbox_iou(c['bbox'], card['bbox'])) if found else None
                    counts = Counter(expected_cards=1, identified=int(found is not None))
                    for field in FIELDS:
                        value = found[field]['value'] if found else None
                        if effect == 'opaque':
                            counts[field+'_hidden_targets'] += 1
                            counts[field+'_hidden_emitted'] += int(value is not None)
                            if value is not None:
                                errors.append({'card': list(identity), 'field': field, 'value': value,
                                               'error': 'value_emitted_under_opaque_overlay'})
                        else:
                            expected = card.get(field)
                            if expected is not None:
                                counts[field+'_uncertain_readability_targets'] += 1
                                counts[field+'_emitted'] += int(value is not None)
                                counts[field+'_latent_agreement'] += int(value == expected)
                                counts[field+'_latent_disagreement'] += int(value is not None and value != expected)
                            elif value is not None:
                                counts[field+'_emitted_without_visible_source_truth'] += 1
                    for key in keys:
                        groups[key].update(counts)
                # Keep every scan and failure, plus a compact contact-sheet source.
                detail = {'source_dataset': str(dataset), 'source_file': row['file'], 'effect': effect,
                          'source_sha256': hashlib.sha256((dataset/row['file']).read_bytes()).hexdigest(),
                          'modified_pixels_sha256': hashlib.sha256(scene.tobytes()).hexdigest(),
                          'overlay_regions': regions, 'errors': errors, 'scan': scan}
                details.append(detail)
                if row_index < 2 or errors:
                    cv2.imwrite(str(output/(name+'.jpg')), scene, [cv2.IMWRITE_JPEG_QUALITY, 94])
            if row_index % 10 == 0:
                print(json.dumps({'dataset': str(dataset), 'source_processed': row_index+1,
                                  'source_total': len(truth['screenshots'])}), flush=True)
        write(output/'progress.json', {'complete': False, 'processed': len(details), 'groups': dict(groups)})
    report = {'schema': 'ournotes-boxlens.occlusion-diagnostic/1', 'complete': True,
              'source_screenshots': len(details)//2, 'modified_screenshots': len(details),
              'unique_source_identities': len(identities), 'groups': dict(groups), 'details': details,
              'recognition_provenance': getattr(engine, 'provenance', {
                  'files': {name: hashlib.sha256((data/name).read_bytes()).hexdigest()
                            for name in ['catalog.json', 'index.npz', 'models/encoder.onnx', 'models/fields.onnx']},
                  'source': 'frozen evaluation implementation; model and catalog hashes recorded'}),
              'limits': ['Development diagnostic, not a new independent final test.',
                         'Opaque values are fully removed; any emitted value is a hidden-field claim.',
                         'Fade readability is uncertain: latent agreement is not human-readable accuracy.',
                         'Per-card lower-band overlays approximate occlusion; not an exact full-game viewport.',
                         'Identity recall is measured; this diagnostic does not score all extra predictions.']}
    write(output/'report.json', report)
    print(json.dumps({k: report[k] for k in ('modified_screenshots', 'unique_source_identities')}
                     | {'groups': {e: dict(groups[e]) for e in effects}}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--datasets', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.data, a.datasets, a.output)
