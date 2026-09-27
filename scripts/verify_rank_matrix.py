"""Exhaustive owned-card/rank coverage on complete, clear native list screenshots.

Resources and generated images stay local; neither belongs in the public repo.
"""
import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from bdon_vision.assets import native_module, read, write
from bdon_vision.synthetic import evaluate, bbox_iou

RESOLUTIONS = [(1280, 720), (1440, 1080), (1920, 1080), (2560, 1440)]
LOCALES = [('ja', '_japanese'), ('en', '_english'), ('zh-Hant', '_traditionalChinese'),
           ('zh-Hans', '_simplifiedChinese'), ('ko', '_korean')]


def generate(data, output):
    output.mkdir(parents=True, exist_ok=False)
    native = native_module(data)
    renderers = {name: native.CardRenderer(column) for name, column in LOCALES}
    catalog = read(data/'catalog.json')['cards']
    tables = {kind: read(data/'master'/filename)['_allData'] for kind, filename in
              [('member', 'MasterMemberCardRank.json'), ('snap', 'MasterSupportCardRank.json')]}
    expected = {(c['kind'], c['id'], r['_rank'], w, h)
                for c in catalog for r in tables[c['kind']] if r['_group'] == c['rank_group']
                and 1 <= r['_rank'] <= 5 for w, h in RESOLUTIONS}
    seen, records, counts = set(), [], Counter()
    for width, height in RESOLUTIONS:
        for rank in range(1, 6):
            for kind in ('member', 'snap'):
                pool = [c for c in catalog if c['kind'] == kind and
                        any(r['_group'] == c['rank_group'] and r['_rank'] == rank for r in tables[kind])]
                for start in range(0, len(pool), 12):
                    locale = LOCALES[len(records) % len(LOCALES)][0]
                    scene = Image.new('RGBA', (width, height), (210, 217, 220, 255))
                    lw, lh, columns = (224, 294, 6) if kind == 'member' else (326, 184, 4)
                    scale = width * (.116 if kind == 'member' else .171) / lw
                    cw, ch = round(lw*scale), round(lh*scale)
                    left, top = round(width*.10), round(height*.13)
                    cards = []
                    for i, card in enumerate(pool[start:start+12]):
                        x = left+(i % columns)*round(cw*1.11)
                        y = top+(i//columns)*round(ch*1.10)
                        assert x > 5 and y > 5 and x+cw+10 < width and y+ch+10 < height
                        state = native.CardState(asset_id=card['asset_id'], rarity=card['rarity'],
                                                 card_type=card['card_type'], level=1, rank=rank,
                                                 awake_count=1, param='level')
                        tile = renderers[locale].render(state, kind, scale=1.)
                        # Native type/rank icons extend outside the card frame.
                        # Preserve the renderer padding instead of cropping them.
                        tile = tile.resize((round((lw+64)*scale), round((lh+64)*scale)), Image.Resampling.BILINEAR)
                        scene.alpha_composite(tile, (x-round(32*scale), y-round(32*scale)))
                        cards.append({'kind': kind, 'id': card['id'], 'level': 1, 'card_rank': rank,
                                      'awake_count': None, 'bbox': [x, y, cw, ch], 'field_visible': True,
                                      'visible_fraction': 1., 'display_mode': 'level',
                                      'latent_state': {'level': 1, 'card_rank': rank, 'awake_count': 1}})
                        seen.add((kind, card['id'], rank, width, height))
                        counts[f"{kind}/rarity{card['rarity']}/rank{rank}"] += 1
                    name = f'{len(records):06d}.jpg'
                    scene.convert('RGB').save(output/name, quality=95)
                    records.append({'file': name, 'cards': cards, 'ignored_cards': [], 'locale': locale,
                                    'box_id': f'clear-owned-rank-{rank}',
                                    'sha256': hashlib.sha256((output/name).read_bytes()).hexdigest(),
                                    'acquisition': {'profile': 'clear-rank-matrix', 'source_resolution': [width, height],
                                                    'compression': [{'codec': 'JPEG', 'quality': 95}]}})
                    if len(records) % 20 == 0:
                        print({'generated_screenshots': len(records)}, flush=True)
    assert seen == expected, 'Incomplete owned-card/rank/resolution matrix'
    write(output/'truth.json', {'schema': 'bdon-synthetic/2', 'complete': True, 'screenshots': records,
                               'profile': 'clear-rank-matrix', 'composition': 'native-padding-preserved-v2',
                               'matrix_card_instances': len(seen),
                               'counts': dict(counts), 'catalog_sha256': hashlib.sha256((data/'catalog.json').read_bytes()).hexdigest(),
                               'scope': 'Every catalog identity at every master-defined owned rank at four resolutions; '
                                        'complete visible lists, JPEG95, no blur or occlusion. Rank0 is not an owned rank.'})
    print({'screenshots': len(records), 'card_instances': len(seen), 'complete': True}, flush=True)


def summarize(data, output):
    truth = read(output/'dataset/truth.json')
    scans = read(output/'evaluation/observations.json')
    catalog = {(c['kind'], c['id']): c for c in read(data/'catalog.json')['cards']}
    groups = defaultdict(Counter)
    for row, scan in zip(truth['screenshots'], scans):
        assert row['file'] == scan['source']
        for card in row['cards']:
            meta = catalog[card['kind'], card['id']]
            candidates = [c for c in scan['cards'] if (c['kind'], c['id']) == (card['kind'], card['id'])
                          and bbox_iou(c['bbox'], card['bbox']) >= .5]
            found = candidates[0] if candidates else None
            value = found['card_rank']['value'] if found else None
            result = Counter(expected=1, identified=int(found is not None), correct=int(value == card['card_rank']),
                             wrong=int(value is not None and value != card['card_rank']),
                             unknown=int(found is not None and value is None), missed=int(found is None))
            keys = ['overall', f"{card['kind']}/rarity{meta['rarity']}/rank{card['card_rank']}",
                    f"resolution/{row['acquisition']['source_resolution']}", f"locale/{row['locale']}"]
            for key in keys:
                groups[key].update(result)
    report = {'schema': 'ournotes-boxlens.clear-rank-matrix/1', 'composition': truth['composition'],
              'screenshots': len(scans), 'groups': dict(groups), 'level_fixed_to': 1,
              'scope': truth['scope'], 'catalog_sha256': truth['catalog_sha256'],
              'truth_sha256': hashlib.sha256((output/'dataset/truth.json').read_bytes()).hexdigest(),
              'recognition_provenance': scans[0].get('recognition_provenance')}
    write(output/'rank-matrix-summary.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    generate(a.data, a.output/'dataset')
    evaluate(a.data, a.output/'dataset', a.output/'evaluation')
    summarize(a.data, a.output)
