"""Train a per-kind parameter-field reader (member 106 classes, Snap 101 classes).

Initialised from the shared 106-class field model; the Snap head keeps rows
0..100. Development sets drive checkpoint selection: for every checkpoint the
lowest confidence threshold (not below 0.995, margin 0.5) without any misread
on the development crops is found, and the checkpoint with the highest recall
on normal and stress scenes at its own threshold wins. A misread is a wrong
visible value or a value emitted for a hidden target that differs from the
card state. The chosen threshold is the model's acceptance threshold.
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .assets import write
from .kind_fields import (FIELD_CLASSES, FIELD_CONFIDENCE, FIELD_MARGIN, FIELD_SIZE, FIELD_STORED, MARGIN,
                          accept, load_bank, misread_sweep)
from .train import LevelNet, seed_all


def field_model(kind):
    model = LevelNet()
    model.head = nn.Linear(96*3*8, FIELD_CLASSES[kind])
    return model


def initialise_from(model, path):
    """Copy every shared layer and the leading rows of the classifier head."""
    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict({k: v for k, v in state.items() if not k.startswith('head.')}, strict=False)
    with torch.no_grad():
        n = min(model.head.out_features, state['head.weight'].shape[0])
        model.head.weight[:n].copy_(state['head.weight'][:n])
        model.head.bias[:n].copy_(state['head.bias'][:n])
    return model


class GpuBank:
    """uint8 crops resident on the GPU (NCHW) with integer labels."""

    def __init__(self, paths, kind, part, device='cuda'):
        images, labels, metas = [], [], []
        for path in paths:
            x, y, m = load_bank(path, kind, part)
            images.append(np.asarray(x))
            labels.append(y)
            metas.append(m)
        self.paths = [str(p) for p in paths]
        x = np.concatenate(images) if images else np.zeros((0,), np.uint8)
        self.x = torch.from_numpy(x).to(device).permute(0, 3, 1, 2).contiguous()
        self.y = torch.from_numpy(np.concatenate(labels)).to(device)
        self.meta = metas
        self.latent = np.concatenate([m['latent'] if 'latent' in m else np.full(len(y), -1, np.int32)
                                      for m, y in zip(metas, labels)]) if metas else np.zeros(0, np.int32)

    def __len__(self):
        return len(self.y)

    def batch(self, indices):
        return self.x[indices].float().div_(255), self.y[indices]


def jitter_crop(x, out_w, out_h, margin, shift, scale):
    """Random shift/scale crop of the stored margin bank; identity is the deployed crop.

    ``shift``: largest centre shift in output pixels, one value or (x, y);
    ``scale``: largest relative scale change. Both are drawn uniformly.
    """
    n, _, h, w = x.shape
    dev = x.device
    shift_x, shift_y = (shift, shift) if np.isscalar(shift) else shift
    dx = torch.empty(n, 1, 1, device=dev).uniform_(-shift_x, shift_x)
    dy = torch.empty(n, 1, 1, device=dev).uniform_(-shift_y, shift_y)
    sc = torch.empty(n, 1, 1, device=dev).uniform_(1-scale, 1+scale)
    i = torch.arange(out_w, device=dev, dtype=torch.float32).view(1, 1, -1)
    j = torch.arange(out_h, device=dev, dtype=torch.float32).view(1, -1, 1)
    cx, cy = (out_w-1)/2, (out_h-1)/2
    xs = margin+cx+dx+(i-cx)*sc
    ys = margin+cy+dy+(j-cy)*sc
    grid = torch.stack([((2*xs+1)/w-1).expand(n, out_h, out_w), ((2*ys+1)/h-1).expand(n, out_h, out_w)], dim=-1)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


def jitter_reach(out_w, out_h, shift, scale):
    """Largest displacement (x, y) of any output pixel under ``jitter_crop``; compare with the stored margin."""
    shift_x, shift_y = (shift, shift) if np.isscalar(shift) else shift
    return shift_x+scale*(out_w-1)/2, shift_y+scale*(out_h-1)/2


def photometric(x, strength=1.):
    n = x.shape[0]
    dev = x.device
    brightness = torch.empty(n, 1, 1, 1, device=dev).uniform_(1-.10*strength, 1+.10*strength)
    contrast = torch.empty(n, 1, 1, 1, device=dev).uniform_(1-.12*strength, 1+.12*strength)
    gain = torch.empty(n, 3, 1, 1, device=dev).uniform_(1-.04*strength, 1+.04*strength)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x-mean)*contrast+mean)*brightness*gain
    blur = torch.empty(n, 1, 1, 1, device=dev).uniform_(0, .6*strength)*(torch.rand(n, 1, 1, 1, device=dev) < .4)
    x = x*(1-blur)+F.avg_pool2d(x, 3, 1, 1, count_include_pad=False)*blur
    x = x+torch.randn_like(x)*.01*strength
    return x.clamp_(0, 1)


def train_mode(model, freeze_bn=False):
    """Training mode; optionally keep the initial BatchNorm statistics fixed."""
    model.train()
    if freeze_bn:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


def center_crop(x, out_w, out_h, margin):
    return x[:, :, margin:margin+out_h, margin:margin+out_w]


def predict(model, bank, out_w, out_h, margin, batch=1024):
    model.eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(bank), batch):
            x = bank.x[start:start+batch].float().div_(255)
            parts.append(model(center_crop(x, out_w, out_h, margin)).float().softmax(1).cpu().numpy())
    return np.concatenate(parts) if parts else np.zeros((0, 1), np.float32)


def outcome_counts(value, labels, keep=None, latent=None):
    """Counts for emitted values (0 = none) against visible targets (label > 0) and hidden targets (0).

    ``false_visible`` is any value emitted for a hidden target. ``latent`` (-1
    unknown), used only for analysis, splits those into values matching the
    card state behind a partial occlusion and genuine misreads. ``errors`` =
    wrong visible values + hidden-target values not matching the card state.
    """
    labels = np.asarray(labels)
    keep = np.ones(len(labels), bool) if keep is None else np.asarray(keep).astype(bool)
    latent = np.full(len(labels), -1) if latent is None else np.asarray(latent)
    visible = (labels > 0) & keep
    hidden = (labels == 0) & keep
    read_hidden = hidden & (value > 0)
    counts = {'visible_total': int(visible.sum()), 'hidden_total': int(hidden.sum()),
              'correct': int((visible & (value == labels)).sum()),
              'wrong': int((visible & (value > 0) & (value != labels)).sum()),
              'unknown': int((visible & (value == 0)).sum()),
              'false_visible': int(read_hidden.sum()),
              'false_visible_latent_match': int((read_hidden & (value == latent)).sum()),
              'false_visible_latent_mismatch': int((read_hidden & (value != latent)).sum())}
    counts['errors'] = counts['wrong']+counts['false_visible_latent_mismatch']
    counts['coverage'] = counts['correct']/counts['visible_total'] if counts['visible_total'] else None
    return counts


def field_outcomes(kind, probabilities, labels, confidence=FIELD_CONFIDENCE, margin=FIELD_MARGIN, latent=None):
    """Per-crop emitted value (0 = no value) and outcome counts.

    For Snap, an accepted 101..105 class (possible only with the shared
    106-class model) is an unsupported parameter and emits no value.
    """
    top, _, ok = accept(probabilities, confidence, margin)
    limit = 105 if kind == 'member' else 100
    value = np.where(ok & (top > 0) & (top <= limit), top, 0)
    return value, outcome_counts(value, labels, latent=latent)


SELECTION_RULE = ('min errors on normal sets, then min errors on stress and hard sets, then max correct on '
                  'normal and stress sets, then max correct on hard sets; errors = wrong visible values + values on '
                  'hidden targets that differ from the card state (any hidden-target value when the state is unknown)')


def selection_key(row, roles):
    def total(key, wanted):
        return sum(row['sets'][name][key] for name, role in roles.items() if role in wanted and name in row['sets'])
    return (total('errors', ('normal',)), total('errors', ('stress', 'hard')),
            -total('correct', ('normal', 'stress')), -total('correct', ('hard',)), row['step'])


ZERO_MISREAD_RULE = ('per checkpoint: lowest confidence threshold (not below the default, margin fixed) with no misread on '
                     'any development crop; choose the highest recall of normal and stress scene targets at that '
                     'threshold, then the highest recall on degraded crops; checkpoints where no threshold removes '
                     'every misread rank after all others, by fewest misreads at the default threshold, then recall')


def zero_misread(probabilities, truth, latent, roles, limit, margin, default):
    """Joint zero-misread threshold over all sets, with recall per role group there and at the default."""
    names = list(probabilities)

    def joined(chosen, key):
        return np.concatenate([key[n] for n in chosen])
    sweep = misread_sweep(joined(names, probabilities), joined(names, truth), joined(names, latent), limit, margin, default)
    threshold = max(default, sweep['zero_misread_threshold'])
    out = {'threshold': threshold, 'achievable': threshold <= 1.,
           'highest_misread_confidence': sweep['highest_misread_confidence'],
           'misreads_at_default': sweep['at_default_threshold']['misreads'], 'recall': {}, 'recall_at_default': {},
           'per_set': {}}
    for group, wanted in [('scenes', ('normal', 'stress')), ('hard', ('hard',))]:
        chosen = [n for n in names if roles[n] in wanted]
        if not chosen:
            continue
        part = misread_sweep(joined(chosen, probabilities), joined(chosen, truth), joined(chosen, latent), limit, margin,
                             default, grid=(threshold,) if threshold <= 1. else ())
        out['recall_at_default'][group] = part['at_default_threshold']['recall']
        out['recall'][group] = part['by_threshold'][str(threshold)]['recall'] if threshold <= 1. else 0.
    for name in names:
        alone = misread_sweep(probabilities[name], truth[name], latent[name], limit, margin, default, grid=())
        out['per_set'][name] = {'zero_misread_threshold': max(default, alone['zero_misread_threshold']),
                                'achievable': alone['zero_misread_achievable'],
                                'recall_at_zero_misread_threshold': alone['at_zero_misread_threshold']['recall']
                                if alone['zero_misread_threshold'] > default else alone['at_default_threshold']['recall'],
                                'misreads_at_default': alone['at_default_threshold']['misreads']}
    return out


def zero_misread_key(row):
    z = row['zero_misread']
    if z['achievable']:
        return (0, -z['recall'].get('scenes', 0.), -z['recall'].get('hard', 0.), row['step'])
    return (1, z['misreads_at_default'], -z['recall_at_default'].get('scenes', 0.), row['step'])


def export_onnx(model, path, shape):
    model = model.eval().cpu()
    torch.onnx.export(model, torch.zeros(*shape), str(path), input_names=['image'], output_names=['logits'],
                      dynamic_axes={'image': {0: 'batch'}, 'logits': {0: 'batch'}}, opset_version=17, dynamo=False)


def verify_onnx(model, path, samples):
    """Compare ONNX Runtime CPU against PyTorch on real crops and several batch sizes."""
    import onnxruntime as ort
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    model = model.eval().cpu()
    report = {'batches': {}}
    worst = 0.
    for size in (1, 7, 64, len(samples)):
        x = samples[:size]
        with torch.inference_mode():
            reference = model(torch.from_numpy(x)).numpy()
        got = session.run(None, {'image': x})[0]
        diff = float(np.abs(reference-got).max())
        worst = max(worst, diff)
        report['batches'][str(len(x))] = {'max_abs_logit_diff': diff,
                                          'argmax_agreement': float((reference.argmax(1) == got.argmax(1)).mean())}
    report['max_abs_logit_diff'] = worst
    report['opset'] = 17
    report['inputs'] = [(i.name, i.shape) for i in session.get_inputs()]
    report['outputs'] = [(o.name, o.shape) for o in session.get_outputs()]
    return report


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train(kind, train_banks, hard_banks, dev_sets, output, initial, steps, batch, eval_every, lr, hard_fraction, device='cuda', threads=8,
          freeze_bn=False, jitter_shift=(2.5, 2.5), jitter_scale=.04):
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('GPU not available')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seed_all(91019 if kind == 'member' else 91031)
    torch.set_num_threads(threads)
    out_w, out_h = FIELD_SIZE
    scenes = GpuBank(train_banks, kind, 'field', device)
    hard = GpuBank(hard_banks, kind, 'field', device) if hard_banks else None
    dev = {name: GpuBank(paths, kind, 'field', device) for name, (paths, _) in dev_sets.items()}
    roles = {name: role for name, (_, role) in dev_sets.items()}
    model = field_model(kind)
    if initial:
        initialise_from(model, initial)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps, pct_start=.05,
                                                   div_factor=10, final_div_factor=100)
    started = time.perf_counter()
    curve = []
    config = {'kind': kind, 'train_banks': [str(p) for p in train_banks], 'hard_banks': [str(p) for p in hard_banks],
              'dev_sets': {k: {'banks': [str(p) for p in v[0]], 'role': v[1]} for k, v in dev_sets.items()},
              'initial': str(initial) if initial else None, 'initial_sha256': file_sha256(initial) if initial else None,
              'steps': steps, 'batch': batch, 'eval_every': eval_every, 'lr': lr, 'hard_fraction': hard_fraction, 'freeze_bn': freeze_bn,
              'jitter': {'shift_px': list(jitter_shift), 'scale': jitter_scale, 'stored_margin_px': MARGIN,
                         'reach_px': [round(v, 3) for v in jitter_reach(out_w, out_h, jitter_shift, jitter_scale)]},
              'train_counts': {'scenes': len(scenes), 'hard': len(hard) if hard else 0},
              'dev_counts': {k: len(v) for k, v in dev.items()},
              'acceptance': {'confidence': FIELD_CONFIDENCE, 'margin': FIELD_MARGIN},
              'schedule': 'linear warmup 5% then cosine decay', 'selection_rule': ZERO_MISREAD_RULE}
    write(output/'config.json', config)

    def checkpoint(step, loss):
        row = {'step': step, 'train_loss': loss, 'elapsed_s': round(time.perf_counter()-started, 1), 'sets': {}}
        kept = {}
        for name, bank in dev.items():
            probabilities = predict(model, bank, out_w, out_h, MARGIN)
            labels = bank.y.cpu().numpy()
            row['sets'][name] = field_outcomes(kind, probabilities, labels, latent=bank.latent)[1]
            kept[name] = (probabilities, labels, bank.latent)
        row['zero_misread'] = zero_misread({n: v[0] for n, v in kept.items()}, {n: v[1] for n, v in kept.items()},
                                           {n: v[2] for n, v in kept.items()}, roles, 105 if kind == 'member' else 100,
                                           FIELD_MARGIN, FIELD_CONFIDENCE)
        curve.append(row)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, output/f'step-{step:06d}.pt')
        write(output/'learning-curve.json', {'checkpoints': curve, 'scope': 'development sets only'})
        print(json.dumps({'checkpoint': step, **{k: (v['correct'], v['errors'], v['visible_total']) for k, v in row['sets'].items()},
                          'zero_misread': [row['zero_misread']['threshold'], row['zero_misread']['recall']]}), flush=True)
        train_mode(model, freeze_bn)

    checkpoint(0, None)
    train_mode(model, freeze_bn)
    for step in range(steps):
        n_hard = int(batch*hard_fraction) if hard else 0
        parts = [scenes.batch(torch.randint(len(scenes), (batch-n_hard,), device=device))]
        if n_hard:
            parts.append(hard.batch(torch.randint(len(hard), (n_hard,), device=device)))
        x = torch.cat([p[0] for p in parts])
        y = torch.cat([p[1] for p in parts])
        x = photometric(jitter_crop(x, out_w, out_h, MARGIN, jitter_shift, jitter_scale))
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
    model = field_model(kind)
    model.load_state_dict(torch.load(output/'selected.pt', map_location='cpu', weights_only=True))
    onnx_path = output/f'{kind}-fields.onnx'
    export_onnx(model, onnx_path, (1, 3, out_h, out_w))
    probe = next(iter(dev.values()))
    samples = center_crop(probe.x[:256].float().div(255), out_w, out_h, MARGIN).cpu().numpy().astype(np.float32)
    consistency = verify_onnx(model, onnx_path, samples)
    write(output/'selection.json', {'selected_step': best['step'], 'selected': best, 'roles': roles,
                                    'rule': ZERO_MISREAD_RULE,
                                    'acceptance': {'confidence': (best['zero_misread']['threshold'] if best['zero_misread']['achievable']
                                                                  else FIELD_CONFIDENCE), 'margin': FIELD_MARGIN}, 'onnx': str(onnx_path), 'onnx_sha256': file_sha256(onnx_path),
                                    'pt_sha256': file_sha256(output/'selected.pt'), 'onnx_consistency': consistency,
                                    'elapsed_s': round(time.perf_counter()-started, 1)})
    print(json.dumps({'selected_step': best['step'], 'onnx_max_abs_diff': consistency['max_abs_logit_diff']}), flush=True)


def parse_dev(values):
    """NAME:ROLE:PATH[,PATH...] with ROLE normal|stress|hard|monitor (monitor sets do not affect selection)."""
    sets = {}
    for value in values or []:
        name, role, paths = value.split(':', 2)
        if role not in ('normal', 'stress', 'hard', 'monitor'):
            raise ValueError(f'Unknown development role: {role}')
        sets[name] = ([Path(p) for p in paths.split(',')], role)
    return sets


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=['member', 'snap'], required=True)
    p.add_argument('--train', nargs='+', type=Path, required=True, help='Scene crop banks')
    p.add_argument('--hard', nargs='*', type=Path, default=[], help='Acquisition-degraded crop banks')
    p.add_argument('--dev', nargs='+', required=True, help='NAME:ROLE:PATH[,PATH] development sets (never the final test)')
    p.add_argument('--initial', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--hard-fraction', type=float, default=.4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--freeze-bn', action='store_true', help='Keep the initial BatchNorm running statistics')
    p.add_argument('--jitter-shift', nargs=2, type=float, default=[2.5, 2.5], metavar=('X', 'Y'),
                   help='Largest training crop shift in input pixels (uniform)')
    p.add_argument('--jitter-scale', type=float, default=.04, help='Largest relative training crop scale change (uniform)')
    a = p.parse_args()
    train(a.kind, a.train, a.hard, parse_dev(a.dev), a.output, a.initial, a.steps, a.batch, a.eval_every, a.lr, a.hard_fraction, a.device, a.threads,
          a.freeze_bn, tuple(a.jitter_shift), a.jitter_scale)
