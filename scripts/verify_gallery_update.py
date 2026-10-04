"""Integration check: a card missing from the catalog is reported as unidentified,
and becomes identified once its catalog entry and artwork are added. No model
file changes."""
import argparse
import hashlib
import shutil
from pathlib import Path

import cv2
import numpy as np

from bdon_vision.assets import read, write
from bdon_vision.engine import Engine
from bdon_vision.kind_encoder import reference_source


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(data, output, member_image=None, snap_image=None, member_id=7, snap_id=70):
    data = Path(data)
    output = Path(output)
    target = output / 'gallery-fixture'
    paths = {'member': Path(member_image) if member_image else data / 'examples/member.jpg',
             'snap': Path(snap_image) if snap_image else data / 'examples/snap.jpg'}
    images = {kind: cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR) if path.is_file() else None
              for kind, path in paths.items()}
    if any(image is None for image in images.values()):
        raise ValueError('Provide local --member-image and --snap-image; game screenshots are not distributed.')
    cards = read(data / 'catalog.json')['cards']
    held = {('member', member_id), ('snap', snap_id)}
    if not held.issubset({(c['kind'], c['id']) for c in cards}):
        raise ValueError('Selected test identities must exist in the supplied catalog')
    target.mkdir(parents=True, exist_ok=False)
    shutil.copytree(data / 'models', target / 'models', ignore=shutil.ignore_patterns('*-gallery.npz'))
    for card in cards:
        path = reference_source(data, card)
        dest = target / path.relative_to(data)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    models = sorted(p for p in (target / 'models').iterdir() if p.suffix in ('.onnx', '.json'))
    hashes = lambda: {p.name: digest(p) for p in models}
    before_hashes = hashes()

    def observe(engine):
        out = {}
        for kind, image in images.items():
            scan = engine.scan(image, kind)
            out[kind] = {'identified': sorted([c['kind'], c['id']] for c in scan['cards']),
                         'unidentified': [[c['kind'], c['bbox']] for c in scan['unidentified']]}
        return out

    write(target / 'catalog.json', {'cards': [c for c in cards if (c['kind'], c['id']) not in held]})
    before = observe(Engine(target))
    write(target / 'catalog.json', {'cards': cards})
    after = observe(Engine(target))
    rejected = all([kind, i] not in before[kind]['identified'] and before[kind]['unidentified'] for kind, i in held)
    restored = all([kind, i] in after[kind]['identified'] for kind, i in held)
    report = {'removed_ids': sorted(held), 'old_catalog_size': len(cards) - len(held), 'new_catalog_size': len(cards),
              'model_sha256_before': before_hashes, 'model_sha256_after': hashes(),
              'model_unchanged': before_hashes == hashes(), 'missing_cards_unidentified': rejected,
              'added_cards_identified': restored, 'before': before, 'after': after,
              'scope': 'User-provided local test images; incremental update integration, not an independent accuracy estimate'}
    write(output / 'gallery-update.json', report)
    print(report)
    if not report['model_unchanged'] or not rejected or not restored:
        raise RuntimeError('Gallery update regression')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--member-image', type=Path)
    p.add_argument('--snap-image', type=Path)
    p.add_argument('--member-id', type=int, default=7)
    p.add_argument('--snap-id', type=int, default=70)
    a = p.parse_args()
    verify(a.data, a.output, a.member_image, a.snap_image, a.member_id, a.snap_id)
