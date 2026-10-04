"""Checkpoint selection for a finished per-kind training run.

Every saved ``step-*.pt`` is evaluated on the development crop banks. Two
rules are available:

* ``zero-misread`` (default): for each checkpoint find the lowest confidence
  threshold (margin rule fixed, never below the default threshold) at which
  no development crop is misread, then pick the checkpoint with the highest
  recall of normal and stress scene targets at that threshold (ties: recall
  on degraded crops). The chosen threshold becomes the model's acceptance
  threshold.
* ``lexicographic``: the training-time rule of ``train_kind_fields`` at the
  default threshold.

The chosen checkpoint is copied to ``selected.pt`` and exported to ONNX with
a consistency check.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from .assets import write
from .kind_fields import FIELD_CONFIDENCE, FIELD_MARGIN, FIELD_SIZE, MARGIN, RANK_CONFIDENCE, RANK_INPUT
from .train_kind_fields import (SELECTION_RULE, ZERO_MISREAD_RULE, GpuBank, center_crop, export_onnx, field_model, field_outcomes,
                                file_sha256, parse_dev, predict, selection_key, verify_onnx, zero_misread,
                                zero_misread_key)
from .train_kind_rank import STORED_MARGIN, RankNet, inside_flags, rank_outcomes


def reselect(kind, part, run, dev_sets, device='cpu', threads=8, rule='zero-misread'):
    torch.set_num_threads(threads)
    run = Path(run)
    roles = {name: role for name, (_, role) in dev_sets.items()}
    dev = {name: GpuBank(paths, kind, 'field' if part == 'fields' else 'rank', device) for name, (paths, _) in dev_sets.items()}
    make = (lambda: field_model(kind)) if part == 'fields' else RankNet
    out_w, out_h, margin = (*FIELD_SIZE, MARGIN) if part == 'fields' else (RANK_INPUT, RANK_INPUT, STORED_MARGIN)
    curve = []
    for path in sorted(run.glob('step-*.pt')):
        model = make()
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        model = model.to(device).eval()
        row = {'step': int(path.stem.split('-')[1]), 'sets': {}}
        kept = {}
        for name, bank in dev.items():
            probabilities = predict(model, bank, out_w, out_h, margin)
            labels = bank.y.cpu().numpy()
            if part == 'fields':
                row['sets'][name] = field_outcomes(kind, probabilities, labels, latent=bank.latent)[1]
                keep = np.ones(len(labels), bool)
            else:
                keep = inside_flags(bank).astype(bool)
                row['sets'][name] = rank_outcomes(probabilities, labels, keep, latent=bank.latent)[1]
            kept[name] = (probabilities[keep], labels[keep], bank.latent[keep])
        limit = (105 if kind == 'member' else 100) if part == 'fields' else 5
        default = FIELD_CONFIDENCE if part == 'fields' else RANK_CONFIDENCE
        row['zero_misread'] = zero_misread({n: v[0] for n, v in kept.items()}, {n: v[1] for n, v in kept.items()},
                                           {n: v[2] for n, v in kept.items()}, roles, limit, FIELD_MARGIN, default)
        curve.append(row)
        print(json.dumps({'step': row['step'], **{k: (v['correct'], v['errors'], v['false_visible']) for k, v in row['sets'].items()},
                          'zero_misread': row['zero_misread']}), flush=True)
    if rule == 'zero-misread':
        best = min(curve, key=zero_misread_key)
    else:
        best = min(curve, key=lambda row: selection_key(row, roles))
    shutil.copy2(run/f"step-{best['step']:06d}.pt", run/'selected.pt')
    model = make()
    model.load_state_dict(torch.load(run/'selected.pt', map_location='cpu', weights_only=True))
    onnx_path = run/f'{kind}-{part}.onnx'
    export_onnx(model, onnx_path, (1, 3, out_h, out_w))
    probe = next(iter(dev.values()))
    samples = center_crop(probe.x[:256].float().div(255), out_w, out_h, margin).cpu().numpy().astype(np.float32)
    consistency = verify_onnx(model, onnx_path, samples)
    write(run/'learning-curve-reselected.json', {'checkpoints': curve, 'scope': 'development sets only'})
    rule_text = ZERO_MISREAD_RULE if rule == 'zero-misread' else SELECTION_RULE
    write(run/'selection.json', {'selected_step': best['step'], 'selected': best, 'roles': roles, 'rule': rule, 'rule_text': rule_text,
                                 'acceptance': {'confidence': (best['zero_misread']['threshold'] if rule == 'zero-misread'
                                                               and best['zero_misread']['achievable'] else default),
                                                'margin': FIELD_MARGIN},
                                 'dev_sets': {k: [str(p) for p in v[0]] for k, v in dev_sets.items()},
                                 'onnx': str(onnx_path), 'onnx_sha256': file_sha256(onnx_path),
                                 'pt_sha256': file_sha256(run/'selected.pt'), 'onnx_consistency': consistency,
                                 'reselected': True})
    print(json.dumps({'selected_step': best['step'], 'onnx_max_abs_diff': consistency['max_abs_logit_diff']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=['member', 'snap'], required=True)
    p.add_argument('--part', choices=['fields', 'rank'], required=True)
    p.add_argument('--run', type=Path, required=True, help='Training output directory with step-*.pt')
    p.add_argument('--dev', nargs='+', required=True, help='NAME:ROLE:PATH[,PATH] development sets (never the final test)')
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--rule', choices=['zero-misread', 'lexicographic'], default='zero-misread')
    a = p.parse_args()
    reselect(a.kind, a.part, a.run, parse_dev(a.dev), a.device, a.threads, a.rule)
