"""Count observed rarity/type/identity coverage from saved screenshot truth.

Counts are evidence of presence, not recognition accuracy or unseen UI coverage.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

RARITIES = {2: 'R', 3: 'SR', 4: 'SSR', 10: 'EX', 20: 'BD'}
TYPES = {1: 'Red', 2: 'Blue', 3: 'Green', 4: 'Yellow', 5: 'Purple'}


def summarize(cards):
    rows = list(cards)
    return {
        'count': len(rows),
        'by_kind_rarity': dict(sorted(Counter(
            f"{c['kind']}/{RARITIES.get(c['rarity'], str(c['rarity']))}" for c in rows).items())),
        'by_kind_type': dict(sorted(Counter(
            f"{c['kind']}/{TYPES.get(c['card_type'], str(c['card_type']))}" for c in rows).items())),
    }


def audit(catalog_path, datasets):
    catalog = json.loads(catalog_path.read_text(encoding='utf8'))['cards']
    lookup = {(c['kind'], c['id']): c for c in catalog}
    result = {
        'schema': 'ournotes-boxlens.coverage/1',
        'catalog_sha256': hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        'catalog': summarize(catalog),
        'rarity_enum': RARITIES, 'card_type_enum': TYPES,
        'enum_evidence': {'rarity_type_index': 34458, 'card_type_type_index': 19592,
                          'client': '1.0.1-25', 'metadata_hash_verified': True},
        'absent_enum_rarities': [name for code, name in RARITIES.items()
                                if code not in {c['rarity'] for c in catalog}],
        'datasets': {},
        'scope': 'Observed identities in saved truth; presence is not accuracy. '
                 'Catalog snapshot only; no assertion of all current game cards or all UI styles.',
    }
    for dataset in datasets:
        truth = json.loads(dataset.read_text(encoding='utf8'))
        seen, observations, modes, visibility = {}, [], Counter(), Counter()
        for row in truth['screenshots']:
            for card in row['cards']:
                key = (card['kind'], card['id'])
                observed = lookup[key]
                seen[key] = observed
                observations.append(observed)
                modes[card.get('display_mode', row.get('display_mode', 'unrecorded'))] += 1
                visibility['partial_card' if card.get('visible_fraction', 1) < 1 else 'whole_card'] += 1
        result['datasets'][str(dataset)] = {
            'truth_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
            'screenshots': len(truth['screenshots']),
            'unique': summarize(seen.values()), 'observations': summarize(observations),
            'missing_catalog_ids': [list(k) for k in sorted(lookup.keys() - seen.keys())],
            'modes': dict(modes), 'geometric_visibility': dict(visibility),
        }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog', type=Path, required=True)
    p.add_argument('--datasets', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.catalog, a.datasets)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'catalog': result['catalog'], 'datasets': {
        k: v['unique'] for k, v in result['datasets'].items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
