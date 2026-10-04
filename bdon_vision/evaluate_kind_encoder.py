"""Compare per-kind encoders with the shared encoder on identical crop banks.

Thresholds are always chosen on the development split named by
``--select-on`` (separately for each model) and then applied unchanged to
every other bank. The shared encoder sees its own input format: member crops
are identical; snap windows and the 512x288 snap artwork are squashed to its
160x128 portrait input.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .assets import read, write
from .kind_corpus import load_bank
from .kind_encoder import kind_cards, load_model, references
from .kind_metrics import open_set_curve, retrieval_table, select_thresholds, summarize
from .train_kind_encoder import embed

SHARED_THRESHOLDS = {'similarity': .72, 'margin': .06}


def shared_model(path):
    from .train import CardEncoder
    model = CardEncoder()
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    return model.eval()


def shared_references(data, cards):
    rows = []
    for card in cards:
        with Image.open(Path(data) / card.get('match_file', card['file'])) as image:
            rows.append(np.asarray(image.convert('RGB').resize((128, 160), Image.Resampling.BILINEAR), np.float32))
    return np.stack(rows).transpose(0, 3, 1, 2) / 255.


def stratified(table, meta, mask, t, m):
    """Summaries per tile width, visibility, occlusion and foreign-art method."""
    def sub(sel):
        return summarize({k: v[sel] for k, v in table.items()}, t, m)
    out = {}
    width = meta['tile_width'][mask]
    bins = np.digitize(width, [80, 120, 170, 240])
    for b, name in enumerate(['<80px', '80-120px', '120-170px', '170-240px', '>=240px']):
        if (bins == b).any():
            out[f'tile_width {name}'] = sub(bins == b)
    partial = meta['window_inside'][mask] < .999
    for flag, name in [(False, 'fully inside'), (True, 'partly outside')]:
        if (partial == flag).any():
            out[name] = sub(partial == flag)
    if 'occluded_fraction' in meta:
        occluded = meta['occluded_fraction'][mask] > .05
        for flag, name in [(False, 'not occluded'), (True, 'occluded > 5%')]:
            if (occluded == flag).any():
                out[name] = sub(occluded == flag)
    if 'foreign_method' in meta:
        method = meta['foreign_method'][mask]
        for name in sorted(set(method.tolist()) - {''}):
            out[f'foreign {name}'] = sub(method == name)
    return out


def per_crop(table, meta, mask, cards):
    rows = []
    files = meta['source'][mask]
    boxes = meta['bbox'][mask]
    for i in range(len(table['labels'])):
        label = int(table['labels'][i])
        rows.append({'file': str(files[i]).split(':', 1)[1], 'bbox': [round(float(v), 1) for v in boxes[i]],
                     'truth': cards[label]['id'] if label >= 0 else None,
                     'top1': cards[int(table['pred'][i])]['id'], 'similarity': round(float(table['s1'][i]), 4),
                     'margin': round(float(table['s1'][i] - table['s2'][i]), 4),
                     'true_similarity': None if label < 0 else round(float(table['true_score'][i]), 4),
                     'without_true_top1': cards[int(table['loo_pred'][i])]['id'],
                     'without_true_similarity': round(float(table['loo_s1'][i]), 4),
                     'without_true_margin': round(float(table['loo_s1'][i] - table['loo_s2'][i]), 4)})
    return rows


def evaluate(args):
    catalog = read(args.data / 'catalog.json')['cards']
    cards = kind_cards(catalog, args.kind)
    held = np.array([c['id'] % 7 == 0 for c in cards])
    models = {}
    if args.model:
        models['kind'] = (load_model(args.kind, args.model).to(args.device), torch.from_numpy(references(args.data, cards, args.kind)),
                          'art', hashlib.sha256(args.model.read_bytes()).hexdigest())
    if args.shared:
        crops = 'art' if args.kind == 'member' else 'baseline'
        models['shared'] = (shared_model(args.shared).to(args.device), torch.from_numpy(shared_references(args.data, cards)).float(),
                            crops, hashlib.sha256(args.shared.read_bytes()).hexdigest())
    banks = dict(item.split('=', 1) for item in args.banks)
    report = {'kind': args.kind, 'select_on': args.select_on, 'banks': {}, 'models': {}}
    for name, (model, refs, crops, digest) in models.items():
        gallery = embed(model, refs.float())
        tables = {}
        for bank_name, path in banks.items():
            arrays, labels, meta, manifest = load_bank(path, catalog, args.kind, (crops,))
            mask = np.ones(len(labels), bool) if bank_name in args.all_crops else meta['window_inside'] >= .999
            queries = embed(model, np.ascontiguousarray(arrays[crops][mask]))
            tables[bank_name] = (retrieval_table(queries, gallery, labels[mask], held), meta, mask, manifest)
        choice = select_thresholds(tables[args.select_on][0])
        result = {'sha256': digest, 'input': crops, 'selected_thresholds': choice,
                  'open_set_curve_on_selection_split': [{k: v for k, v in row.items() if k != 'summary'} | (
                      {'known_accepted_correct': row['summary']['known']['accepted_correct_rate']} if 'summary' in row else {})
                      for row in open_set_curve(tables[args.select_on][0])], 'banks': {}}
        for bank_name, (table, meta, mask, manifest) in tables.items():
            entry = {'crops': int(mask.sum()), 'fingerprint': manifest['source_fingerprint'],
                     'top1_only': summarize(table, 2., 1.)}
            if choice['feasible']:
                entry['at_selected'] = summarize(table, choice['similarity'], choice['margin'])
                entry['strata_at_selected'] = stratified(table, meta, mask, choice['similarity'], choice['margin'])
            if name == 'shared':
                entry['at_shared_runtime_thresholds'] = summarize(table, SHARED_THRESHOLDS['similarity'],
                                                                  SHARED_THRESHOLDS['margin'])
            if bank_name in args.per_crop:
                entry['per_crop'] = per_crop(table, meta, mask, cards)
            result['banks'][bank_name] = entry
            report['banks'][bank_name] = str(path)
        report['models'][name] = result
    write(args.output, report)
    brief = {name: {b: {'seen_top1': r['top1_only']['seen']['top1'], 'unseen_top1': r['top1_only']['unseen']['top1'],
                        'sel': res['selected_thresholds'].get('similarity'),
                        'known_acc_correct': r.get('at_selected', {}).get('known', {}).get('accepted_correct_rate'),
                        'unseen_open_far': r.get('at_selected', {}).get('unseen', {}).get('open_set_false_accept')}
                    for b, r in res['banks'].items()} for name, res in report['models'].items()}
    print(json.dumps(brief, indent=1), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=['member', 'snap'], required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--model', type=Path, help='per-kind encoder state dict')
    p.add_argument('--shared', type=Path, help='shared 160x128 encoder state dict')
    p.add_argument('--banks', nargs='+', required=True, help='name=crop-bank-root')
    p.add_argument('--select-on', default='validation')
    p.add_argument('--per-crop', nargs='*', default=[], help='bank names to list crop by crop')
    p.add_argument('--all-crops', nargs='*', default=[], help='bank names that keep partly visible windows')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    evaluate(p.parse_args())


if __name__ == '__main__':
    main()
