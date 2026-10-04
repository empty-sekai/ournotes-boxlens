"""Train a per-kind card-rank icon classifier.

Six classes: 0 = no readable owned-rank icon (hidden view, absent or cut),
1..5 = card rank. A prediction is accepted only when its softmax probability
and the gap to the runner-up both pass the thresholds in ``kind_fields``;
otherwise the rank is unknown. Checkpoint selection uses the same
zero-misread rule as the field reader, counting only crops whose icon lies
fully inside the image because deployed inference rejects the others by
geometry before the model runs.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .assets import write
from .kind_fields import MARGIN, RANK_CLASSES, RANK_CONFIDENCE, RANK_INPUT, RANK_MARGIN, RANK_SIDE, accept
from .train import seed_all
from .train_kind_fields import (ZERO_MISREAD_RULE, GpuBank, center_crop, export_onnx, file_sha256, jitter_crop, jitter_reach,
                                outcome_counts, parse_dev, photometric, predict, verify_onnx, zero_misread,
                                zero_misread_key)

STORED_MARGIN = MARGIN*RANK_INPUT//RANK_SIDE


class RankNet(nn.Module):
    """Compact CNN, input 3x48x48 RGB in [0, 1], output 6 logits."""

    def __init__(self):
        super().__init__()

        def block(a, b):
            return [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU()]
        self.layers = nn.Sequential(*block(3, 32), *block(32, 32), nn.MaxPool2d(2),
                                    *block(32, 64), nn.MaxPool2d(2),
                                    *block(64, 96), nn.MaxPool2d(2),
                                    nn.Conv2d(96, 128, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2))
        self.dropout = nn.Dropout(.2)
        self.head = nn.Linear(128*3*3, RANK_CLASSES)

    def forward(self, x):
        return self.head(self.dropout(self.layers(x).flatten(1)))


def rank_outcomes(probabilities, labels, inside=None, confidence=RANK_CONFIDENCE, margin=RANK_MARGIN, latent=None):
    """Emitted rank (0 = none) and counts over crops whose icon is inside the image."""
    top, _, ok = accept(probabilities, confidence, margin)
    value = np.where(ok & (top > 0), top, 0)
    counts = outcome_counts(value, labels, inside, latent)
    keep = np.ones(len(value), bool) if inside is None else np.asarray(inside).astype(bool)
    counts['outside_emitted'] = int((~keep & (value > 0)).sum())
    return value, counts


def inside_flags(bank):
    return np.concatenate([m['icon_inside'] for m in bank.meta]) if bank.meta else np.zeros(0, int)


def train(kind, train_banks, hard_banks, dev_sets, output, steps, batch, eval_every, lr, hard_fraction, device='cuda', threads=8,
          jitter_shift=(2.5, 2.5), jitter_scale=.07):
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('GPU not available')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seed_all(92017 if kind == 'member' else 92033)
    torch.set_num_threads(threads)
    side = RANK_INPUT
    scenes = GpuBank(train_banks, kind, 'rank', device)
    hard = GpuBank(hard_banks, kind, 'rank', device) if hard_banks else None
    dev = {name: GpuBank(paths, kind, 'rank', device) for name, (paths, _) in dev_sets.items()}
    inside = {name: inside_flags(bank) for name, bank in dev.items()}
    roles = {name: role for name, (_, role) in dev_sets.items()}
    model = RankNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps, pct_start=.05)
    started = time.perf_counter()
    curve = []
    config = {'kind': kind, 'train_banks': [str(p) for p in train_banks], 'hard_banks': [str(p) for p in hard_banks],
              'dev_sets': {k: {'banks': [str(p) for p in v[0]], 'role': v[1]} for k, v in dev_sets.items()},
              'steps': steps, 'batch': batch, 'eval_every': eval_every, 'lr': lr, 'hard_fraction': hard_fraction,
              'jitter': {'shift_px': list(jitter_shift), 'scale': jitter_scale, 'stored_margin_px': STORED_MARGIN,
                         'reach_px': [round(v, 3) for v in jitter_reach(side, side, jitter_shift, jitter_scale)]},
              'train_counts': {'scenes': len(scenes), 'hard': len(hard) if hard else 0},
              'dev_counts': {k: len(v) for k, v in dev.items()},
              'input': {'side': side, 'logical_side': RANK_SIDE},
              'acceptance': {'confidence': RANK_CONFIDENCE, 'margin': RANK_MARGIN},
              'selection_rule': ZERO_MISREAD_RULE+'; icon-inside crops only'}
    write(output/'config.json', config)

    def checkpoint(step, loss):
        row = {'step': step, 'train_loss': loss, 'elapsed_s': round(time.perf_counter()-started, 1), 'sets': {}}
        kept = {}
        for name, bank in dev.items():
            probabilities = predict(model, bank, side, side, STORED_MARGIN)
            labels = bank.y.cpu().numpy()
            row['sets'][name] = rank_outcomes(probabilities, labels, inside[name], latent=bank.latent)[1]
            keep = inside[name].astype(bool)
            kept[name] = (probabilities[keep], labels[keep], bank.latent[keep])
        row['zero_misread'] = zero_misread({n: v[0] for n, v in kept.items()}, {n: v[1] for n, v in kept.items()},
                                           {n: v[2] for n, v in kept.items()}, roles, 5, RANK_MARGIN, RANK_CONFIDENCE)
        curve.append(row)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, output/f'step-{step:06d}.pt')
        write(output/'learning-curve.json', {'checkpoints': curve, 'scope': 'development sets only'})
        print(json.dumps({'checkpoint': step, **{k: (v['correct'], v['errors'], v['visible_total']) for k, v in row['sets'].items()},
                          'zero_misread': [row['zero_misread']['threshold'], row['zero_misread']['recall']]}), flush=True)
        model.train()

    model.train()
    for step in range(steps):
        n_hard = int(batch*hard_fraction) if hard else 0
        parts = [scenes.batch(torch.randint(len(scenes), (batch-n_hard,), device=device))]
        if n_hard:
            parts.append(hard.batch(torch.randint(len(hard), (n_hard,), device=device)))
        x = torch.cat([p[0] for p in parts])
        y = torch.cat([p[1] for p in parts])
        x = photometric(jitter_crop(x, side, side, STORED_MARGIN, jitter_shift, jitter_scale))
        loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        if step == 0 or (step+1) % 100 == 0:
            write(output/'progress.json', {'step': step+1, 'steps': steps, 'loss': float(loss.detach()),
                                          'lr': schedule.get_last_lr()[0], 'elapsed_s': round(time.perf_counter()-started, 1),
                                          'gpu_allocated_gib': round(torch.cuda.memory_allocated()/2**30, 2) if device.startswith('cuda') else None})
        if (step+1) % eval_every == 0 or step+1 == steps:
            checkpoint(step+1, float(loss.detach()))
    best = min(curve, key=zero_misread_key)
    shutil.copy2(output/f"step-{best['step']:06d}.pt", output/'selected.pt')
    model = RankNet()
    model.load_state_dict(torch.load(output/'selected.pt', map_location='cpu', weights_only=True))
    onnx_path = output/f'{kind}-rank.onnx'
    export_onnx(model, onnx_path, (1, 3, side, side))
    probe = next(iter(dev.values()))
    samples = center_crop(probe.x[:256].float().div(255), side, side, STORED_MARGIN).cpu().numpy().astype(np.float32)
    consistency = verify_onnx(model, onnx_path, samples)
    write(output/'selection.json', {'selected_step': best['step'], 'selected': best, 'roles': roles,
                                    'rule': config['selection_rule'],
                                    'acceptance': {'confidence': (best['zero_misread']['threshold'] if best['zero_misread']['achievable']
                                                                  else RANK_CONFIDENCE), 'margin': RANK_MARGIN}, 'onnx': str(onnx_path), 'onnx_sha256': file_sha256(onnx_path),
                                    'pt_sha256': file_sha256(output/'selected.pt'), 'onnx_consistency': consistency,
                                    'elapsed_s': round(time.perf_counter()-started, 1)})
    print(json.dumps({'selected_step': best['step'], 'onnx_max_abs_diff': consistency['max_abs_logit_diff']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=['member', 'snap'], required=True)
    p.add_argument('--train', nargs='+', type=Path, required=True)
    p.add_argument('--hard', nargs='*', type=Path, default=[])
    p.add_argument('--dev', nargs='+', required=True, help='NAME:ROLE:PATH[,PATH] development sets (never the final test)')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--hard-fraction', type=float, default=.4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--jitter-shift', nargs=2, type=float, default=[2.5, 2.5], metavar=('X', 'Y'),
                   help='Largest training crop shift in input pixels (uniform)')
    p.add_argument('--jitter-scale', type=float, default=.07, help='Largest relative training crop scale change (uniform)')
    a = p.parse_args()
    train(a.kind, a.train, a.hard, parse_dev(a.dev), a.output, a.steps, a.batch, a.eval_every, a.lr, a.hard_fraction, a.device, a.threads,
          tuple(a.jitter_shift), a.jitter_scale)
