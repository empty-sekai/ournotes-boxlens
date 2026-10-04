"""Train a per-kind open-set artwork encoder on scene crop banks.

Queries are artwork-window crops from complete synthetic scenes; references
are catalog artwork prepared with ``kind_geometry.reference_image``. Only
training identities (``id % 7 != 0``) enter either branch. Held-out
identities and foreign artwork are used only by the development evaluation,
where held-out cards are added to the gallery as plain vectors.

Objective per step (training references are encoded with gradients, or
without gradients and refreshed periodically with ``--ref-mode frozen``):

* margin softmax over training references plus a fixed "unknown" logit:
  the true card must beat every other card and the acceptance level;
* leave-true-out softmax: with the true card removed, the unknown logit must
  beat every remaining card (open-set rejection);
* cosine consistency between each query and its reference;
* optional cosine distillation towards the initial model's embeddings, which
  keeps held-out identities from drifting while training identities adapt.

Other cards of the same character get an additive logit bonus in both
softmax terms, so the closest confusers are kept further apart. BatchNorm
statistics stay frozen so query and reference branches share one forward.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .assets import read, write
from .kind_corpus import CONTEXT_MARGIN, load_bank
from .kind_encoder import build_model, freeze_batchnorm, kind_cards, reference_source, references
from .kind_geometry import INPUT, OVERLAYS, cross_reference_image, cross_window
from .kind_metrics import open_set_curve, retrieval_table, select_thresholds, summarize


def seed_all(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _uniform(n, low, high, device):
    return torch.empty(n, device=device).uniform_(low, high)


def warp(x, out_hw, base, shift, scale, aspect):
    """Batched crop of the central ``base`` fraction of each image.

    ``base`` is a fraction of the input size, either one number or an
    ``(x, y)`` pair. ``shift`` is the maximum translation as a fraction of the
    cropped window, ``scale`` and ``aspect`` the relative size and aspect jitter.
    """
    n = x.shape[0]
    bx, by = (base, base) if isinstance(base, (int, float)) else base
    s = _uniform(n, 1 - scale, 1 + scale, x.device)
    a = _uniform(n, -aspect, aspect, x.device)
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0] = bx * s * (1 + a)
    theta[:, 1, 1] = by * s * (1 - a)
    theta[:, :, 2] = (torch.rand(n, 2, device=x.device) * 2 - 1) * 2 * shift * torch.tensor([bx, by], device=x.device)
    grid = F.affine_grid(theta, (n, x.shape[1], *out_hw), align_corners=False)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)


def photometric(x, strength):
    n = x.shape[0]
    dev = x.device
    gray = x.mean(1, keepdim=True)
    sat = _uniform(n, 1 - .25 * strength, 1 + .2 * strength, dev).view(n, 1, 1, 1)
    x = gray + (x - gray) * sat
    contrast = _uniform(n, 1 - .2 * strength, 1 + .2 * strength, dev).view(n, 1, 1, 1)
    bright = _uniform(n, 1 - .2 * strength, 1 + .15 * strength, dev).view(n, 1, 1, 1)
    x = ((x - .5) * contrast + .5) * bright
    tint = 1 + (torch.rand(n, 3, 1, 1, device=dev) * 2 - 1) * .04 * strength
    x = (x * tint).clamp(0, 1)
    gamma = torch.exp(_uniform(n, -.15 * strength, .15 * strength, dev)).view(n, 1, 1, 1)
    return x.clamp(1e-4, 1) ** gamma


def resolution(x, low, prob, chunks=4):
    """Downscale-then-upscale random chunks of the batch (small or blurry tiles)."""
    n, _, h, w = x.shape
    out = []
    for part in x.chunk(chunks):
        if torch.rand(()) < prob:
            f = float(torch.empty(()).uniform_(low, 1.))
            small = F.interpolate(part, size=(max(8, round(h * f)), max(8, round(w * f))), mode='area')
            part = F.interpolate(small, size=(h, w), mode='bilinear', align_corners=False)
        out.append(part)
    return torch.cat(out)


def occlude(x, kind, overlay_prob, erase_prob, edge_prob):
    """Display-layer occlusion: UI overlay zones, random rectangles and cut edges."""
    n, _, h, w = x.shape
    dev = x.device
    donor = x.roll(1, 0)
    x = x.clone()
    for x0, y0, x1, y1 in OVERLAYS[kind]:
        pick = torch.rand(n, device=dev)
        r0, r1 = int(y0 * h), int(math.ceil(y1 * h))
        c0, c1 = int(x0 * w), int(math.ceil(x1 * w))
        solid = pick < overlay_prob * .5
        patch = (pick >= overlay_prob * .5) & (pick < overlay_prob)
        if solid.any():
            x[solid, :, r0:r1, c0:c1] = torch.rand(int(solid.sum()), 3, 1, 1, device=dev)
        if patch.any():
            x[patch, :, r0:r1, c0:c1] = donor[patch, :, r0:r1, c0:c1]
    for i in (torch.rand(n) < erase_prob).nonzero().flatten().tolist():
        area = float(torch.empty(()).uniform_(.03, .14))
        ratio = math.exp(float(torch.empty(()).uniform_(-1., 1.)))
        eh = min(h, max(2, int(round(math.sqrt(area * h * w / ratio)))))
        ew = min(w, max(2, int(round(math.sqrt(area * h * w * ratio)))))
        top = int(torch.randint(0, h - eh + 1, ()))
        left = int(torch.randint(0, w - ew + 1, ()))
        if torch.rand(()) < .5:
            x[i, :, top:top + eh, left:left + ew] = torch.rand(3, 1, 1, device=dev)
        else:
            x[i, :, top:top + eh, left:left + ew] = torch.rand(3, eh, ew, device=dev)
    for i in (torch.rand(n) < edge_prob).nonzero().flatten().tolist():
        side = int(torch.randint(0, 4, ()))
        f = float(torch.empty(()).uniform_(.05, .3))
        fill = torch.rand(3, 1, 1, device=dev)
        if side == 0:
            x[i, :, :int(h * f)] = fill
        elif side == 1:
            x[i, :, h - int(h * f):] = fill
        elif side == 2:
            x[i, :, :, :int(w * f)] = fill
        else:
            x[i, :, :, w - int(w * f):] = fill
    return x


def augment_queries(ctx, kind, cfg, window=(1., 1.)):
    """Augmented encoder queries from context crops.

    ``window`` is the central fraction (x, y) of the artwork window to keep;
    other-kind crops use the central part matching this kind's aspect ratio.
    """
    h, w = INPUT[kind]
    base = tuple(f / (1 + 2 * CONTEXT_MARGIN) for f in window)
    x = warp(ctx, (h, w), base, cfg['shift'], cfg['scale'], cfg['aspect'])
    x = photometric(x, cfg['color'])
    x = resolution(x, cfg['min_resolution'], cfg['resolution_prob'])
    x = occlude(x, kind, cfg['overlay_prob'], cfg['erase_prob'], cfg['edge_prob'])
    x = x + torch.randn_like(x) * float(torch.empty(()).uniform_(0, cfg['noise']))
    return x.clamp(0, 1)


def augment_references(ref, kind):
    h, w = INPUT[kind]
    x = warp(ref, (h, w), 1., .02, .03, .02)
    x = photometric(x, .3)
    x = resolution(x, .5, .3, chunks=2)
    return x.clamp(0, 1)


@torch.no_grad()
def embed(model, array, batch=512):
    """Embed uint8 NHWC (numpy or tensor) or float NCHW tensors in eval mode."""
    device = next(model.parameters()).device
    was = model.training
    model.eval()
    out = []
    for start in range(0, len(array), batch):
        part = array[start:start + batch]
        if isinstance(part, np.ndarray):
            part = torch.from_numpy(np.ascontiguousarray(part))
        if part.dtype == torch.uint8:
            part = part.to(device).permute(0, 3, 1, 2).float().div_(255)
        out.append(model(part.to(device)).float().cpu())
    model.train(was)
    if was:
        freeze_batchnorm(model)
    return torch.cat(out).numpy()


def evaluate_bank(model, bank, gallery_inputs, held, thresholds=None):
    """Retrieval summary with thresholds chosen on this bank; if ``thresholds``
    (chosen elsewhere) are given, also report the bank at those thresholds."""
    gallery = embed(model, gallery_inputs)
    queries = embed(model, bank['art'])
    table = retrieval_table(queries, gallery, bank['labels'], held)
    choice = select_thresholds(table)
    result = {'selected_thresholds': choice, 'top1_only': summarize(table, 2., 1.)}
    if choice['feasible']:
        result['at_selected'] = summarize(table, choice['similarity'], choice['margin'])
    if thresholds and thresholds.get('feasible'):
        at = summarize(table, thresholds['similarity'], thresholds['margin'])
        parts = [at[g]['accepted_correct_rate'] for g in ('seen', 'unseen') if at[g]['count']]
        result['at_validation_thresholds'] = at
        result['score_at_validation_thresholds'] = float(np.mean(parts)) if parts else 0.
    return result, table


class QueryPool:
    """Training queries of one source kind plus the matching reference inputs.

    The own kind uses whole artwork windows. With ``--cross-kind`` the training
    identities of the other kind become extra identities: queries keep the
    central part of their artwork window with this kind's aspect ratio and the
    references are cut the same way, so identities and degradations both grow
    without touching held-out cards.
    """

    def __init__(self, args, catalog, kind, source_kind, device, offset=0):
        self.kind, self.source_kind, self.device = kind, source_kind, device
        self.cards = kind_cards(catalog, source_kind)
        held = np.array([c['id'] % 7 == 0 for c in self.cards])
        self.train_ids = np.flatnonzero(~held)
        contexts, labels, self.fingerprints = [], [], []
        self.dropped = {'held_out_identity': 0, 'foreign': 0, 'window_partly_outside': 0}
        for path in args.train:
            arrays, local, meta, manifest = load_bank(path, catalog, source_kind, ('context',))
            if set(manifest['profiles']) - {'train'}:
                raise ValueError(f'Training bank {path} contains non-train profiles {manifest["profiles"]}')
            keep = local >= 0
            self.dropped['foreign'] += int((~keep).sum())
            held_q = keep & held[np.maximum(local, 0)]
            self.dropped['held_out_identity'] += int(held_q.sum())
            keep &= ~held_q
            partial = meta['window_inside'] < args.min_inside
            self.dropped['window_partly_outside'] += int((keep & partial).sum())
            keep &= ~partial
            idx = np.flatnonzero(keep)
            contexts.append(torch.from_numpy(np.ascontiguousarray(arrays['context'][idx])))
            labels.append(local[idx])
            self.fingerprints.append(manifest['source_fingerprint'])
        self.context = torch.cat(contexts)
        self.label = torch.from_numpy(np.concatenate(labels))
        if held[self.label.numpy()].any():
            raise ValueError('Held-out identity in training queries')
        self.on_gpu = device.type == 'cpu' or self.context.numel() < args.gpu_bank_bytes
        self.context = self.context.to(device) if self.on_gpu else self.context.pin_memory()
        self.order = torch.argsort(self.label)
        self.counts = torch.bincount(self.label, minlength=len(self.cards))
        self.offsets = torch.cumsum(self.counts, 0) - self.counts
        self.present = torch.tensor([int(k) for k in self.train_ids if self.counts[k] > 0])
        self.position = torch.full((len(self.cards),), -1, dtype=torch.long)
        self.position[torch.from_numpy(self.train_ids)] = torch.arange(len(self.train_ids)) + offset
        if source_kind == kind:
            self.window = (1., 1.)
            inputs = references(args.data, self.cards, kind)[self.train_ids]
        else:
            self.window = cross_window(kind, source_kind)
            rows = []
            for k in self.train_ids:
                with Image.open(reference_source(args.data, self.cards[k])) as image:
                    rows.append(np.asarray(cross_reference_image(kind, source_kind, image), np.float32) / 255.)
            inputs = np.stack(rows).transpose(0, 3, 1, 2)
        self.ref_inputs = torch.from_numpy(np.ascontiguousarray(inputs)).float().to(device)

    def sample(self, n, cfg):
        """``n`` identity-balanced augmented queries and their reference positions."""
        ident = self.present[torch.randint(len(self.present), (n,))]
        offset = (torch.rand(n) * self.counts[ident]).long().clamp_max(self.counts[ident] - 1)
        rows = self.order[self.offsets[ident] + offset]
        batch = self.context[rows.to(self.context.device)] if self.on_gpu else self.context[rows]
        batch = batch.to(self.device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
        return augment_queries(batch, self.kind, cfg, self.window), self.position[ident].to(self.device)


def train(args):
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    kind = args.kind
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    torch.set_num_threads(8)
    catalog = read(args.data / 'catalog.json')['cards']
    cards = kind_cards(catalog, kind)
    held = np.array([c['id'] % 7 == 0 for c in cards])
    train_ids = np.flatnonzero(~held)
    ref_inputs = torch.from_numpy(references(args.data, cards, kind)).float().to(device)

    pools = [QueryPool(args, catalog, kind, kind, device)]
    if args.cross_kind:
        other = 'snap' if kind == 'member' else 'member'
        pools.append(QueryPool(args, catalog, kind, other, device, offset=len(pools[0].train_ids)))
    own = pools[0]
    missing = [int(cards[k]['id']) for k in train_ids if own.counts[k] == 0]

    dev_sets = {}
    for name, path in [('validation', args.validation), ('stress', args.stress)]:
        if path is None:
            continue
        arrays, local, meta, manifest = load_bank(path, catalog, kind, ('art',))
        keep = meta['window_inside'] >= args.min_inside
        dev_sets[name] = {'art': np.ascontiguousarray(arrays['art'][keep]), 'labels': local[keep],
                          'fingerprint': manifest['source_fingerprint'], 'profiles': manifest['profiles']}

    model = build_model(kind)
    state = torch.load(args.initial, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).train()
    freeze_batchnorm(model)
    teacher = None
    if args.kd_weight:
        teacher = build_model(kind)
        teacher.load_state_dict(state)
        teacher = teacher.to(device).eval().requires_grad_(False)
    head = list(model.projection.parameters())
    backbone = [p for n, p in model.named_parameters() if not n.startswith('projection.')]
    optimizer = torch.optim.AdamW([{'params': backbone, 'lr': args.lr}, {'params': head, 'lr': args.head_lr}],
                                  weight_decay=args.weight_decay)
    base_lrs = [args.lr, args.head_lr]
    cfg = {'shift': args.shift, 'scale': args.scale_jitter, 'aspect': .03, 'color': 1., 'min_resolution': args.min_resolution,
           'resolution_prob': .5, 'overlay_prob': args.overlay_prob, 'erase_prob': args.erase_prob,
           'edge_prob': args.edge_prob, 'noise': .02}
    config = {k: (str(v) if isinstance(v, Path) else ([str(p) for p in v] if isinstance(v, list) else v))
              for k, v in vars(args).items()}
    config.update({'augmentation': cfg, 'train_queries': {p.source_kind: int(len(p.label)) for p in pools},
                   'dropped_queries': {p.source_kind: p.dropped for p in pools},
                   'reference_identities': {p.source_kind: [int(p.cards[k]['id']) for k in p.train_ids] for p in pools},
                   'train_identities': [int(cards[k]['id']) for k in train_ids],
                   'held_out_identities': [int(c['id']) for c in cards if c['id'] % 7 == 0],
                   'train_identities_without_queries': missing,
                   'train_fingerprints': {p.source_kind: p.fingerprints for p in pools},
                   'initial_sha256': hashlib.sha256(args.initial.read_bytes()).hexdigest(),
                   'bank_on_gpu': all(p.on_gpu for p in pools)})
    write(out / 'config.json', config)
    print(json.dumps({k: config[k] for k in ['train_queries', 'dropped_queries', 'bank_on_gpu']}), flush=True)

    all_refs = torch.cat([p.ref_inputs for p in pools])
    names = [p.cards[k].get('name') for p in pools for k in p.train_ids]
    same_name = torch.tensor([[a is not None and a == b for b in names] for a in names], device=device)
    shares = [1.] if len(pools) == 1 else [args.own_fraction, 1 - args.own_fraction]
    sizes = [round(args.batch * f) for f in shares]
    sizes[-1] = args.batch - sum(sizes[:-1])
    curve = []
    started = time.perf_counter()

    def checkpoint(step, loss_value):
        row = {'step': step, 'train_loss': loss_value, 'elapsed_s': round(time.perf_counter() - started, 1)}
        choice = None
        for name, bank in dev_sets.items():
            result, _ = evaluate_bank(model, bank, ref_inputs, held, choice)
            row[name] = result
            if name == 'validation':
                choice = result['selected_thresholds']
        curve.append(row)
        write(out / 'learning-curve.json', {'checkpoints': curve})
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / f'step-{step:06d}.pt')
        brief = {name: {'select': row[name]['selected_thresholds'],
                        'score_at_validation_thresholds': row[name].get('score_at_validation_thresholds'),
                        'top1_seen': row[name]['top1_only']['seen']['top1'],
                        'top1_unseen': row[name]['top1_only']['unseen']['top1']} for name in dev_sets}
        print(json.dumps({'checkpoint': step, **brief}), flush=True)

    if args.eval_every:
        checkpoint(0, None)
    for step in range(args.steps):
        warm = min(1., (step + 1) / args.warmup)
        cosine = .05 + .95 * .5 * (1 + math.cos(math.pi * step / args.steps))
        for group, lr in zip(optimizer.param_groups, base_lrs):
            group['lr'] = lr * warm * cosine
        parts = [p.sample(n, cfg) for p, n in zip(pools, sizes) if n]
        queries = torch.cat([q for q, _ in parts])
        target = torch.cat([t for _, t in parts])
        if args.ref_mode == 'grad':
            ref_batch = augment_references(all_refs, kind)
            emb = model(torch.cat([queries, ref_batch]))
            q, r = emb[:len(queries)], emb[len(queries):]
        else:
            if step % args.ref_every == 0:
                with torch.no_grad():
                    frozen_refs = model(all_refs)
            q, r = model(queries), frozen_refs
        sims = q @ r.T
        onehot = F.one_hot(target, len(all_refs)).bool()
        unknown = torch.full((len(q), 1), args.unknown_level, device=device)
        # Cards of the same character are the closest confusers; their logits
        # get an extra bonus so the loss keeps a wider gap to them.
        hard = (same_name[target] & ~onehot).float() * args.same_name_margin
        logits = torch.cat([sims - args.margin * onehot.float() + hard, unknown], 1) / args.temperature
        loss_id = F.cross_entropy(logits, target)
        loo = torch.cat([(sims + hard).masked_fill(onehot, -1e4), unknown], 1) / args.temperature
        loss_open = F.cross_entropy(loo, torch.full_like(target, len(all_refs)))
        loss_cos = (1 - sims[onehot]).mean()
        loss = loss_id + args.open_weight * loss_open + args.cos_weight * loss_cos
        loss_kd = torch.zeros((), device=device)
        if teacher is not None:
            # Stay close to the initial embedding so identities that never
            # appear in training keep their place in the embedding space.
            with torch.no_grad():
                q0 = teacher(queries)
            loss_kd = (1 - (q * q0).sum(1)).mean()
            if args.ref_mode == 'grad':
                with torch.no_grad():
                    r0 = teacher(ref_batch)
                loss_kd = loss_kd + (1 - (r * r0).sum(1)).mean()
            loss = loss + args.kd_weight * loss_kd
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0:
            status = {'step': step + 1, 'steps': args.steps, 'loss': loss.item(), 'identity': loss_id.item(),
                      'open': loss_open.item(), 'cos': loss_cos.item(), 'kd': loss_kd.item(),
                      'elapsed_s': round(time.perf_counter() - started, 1),
                      'gpu_max_allocated_gib': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if device.type == 'cuda' else 0}
            write(out / 'progress.json', status)
            print(json.dumps(status), flush=True)
        if args.eval_every and ((step + 1) % args.eval_every == 0 or step + 1 == args.steps):
            checkpoint(step + 1, loss.item())
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / 'last.pt')
    if curve:
        best = select_checkpoint(curve)
        write(out / 'selection.json', best)
        print(json.dumps({'selected': best}), flush=True)
    write(out / 'progress.json', {'step': args.steps, 'steps': args.steps, 'done': True,
                                  'elapsed_s': round(time.perf_counter() - started, 1)})


def select_checkpoint(curve):
    """Highest validation score (mean of seen and unseen accepted-correct rates
    at the open-set-constrained thresholds chosen on validation), rounded to
    0.001; ties go to the higher stress score at those same thresholds, then
    to the earlier step."""
    def key(row):
        stress = row.get('stress', {}).get('score_at_validation_thresholds', 0.)
        return round(row['validation']['selected_thresholds']['score'], 3), round(stress, 4), -row['step']
    row = max(curve, key=key)
    return {'rule': 'max validation score = mean(seen, unseen) accepted-correct rate at thresholds with '
                    'unseen leave-true-out and foreign false accepts <= 1% and accepted precision >= 99.5%; '
                    'ties broken by stress score at the validation thresholds, then earlier step',
            'step': row['step'], 'validation_score': row['validation']['selected_thresholds']['score'],
            'stress_score': row.get('stress', {}).get('score_at_validation_thresholds'),
            'thresholds': row['validation']['selected_thresholds']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=['member', 'snap'], required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--train', type=Path, nargs='+', required=True, help='crop bank roots (train profile only)')
    p.add_argument('--validation', type=Path, required=True)
    p.add_argument('--stress', type=Path)
    p.add_argument('--initial', type=Path, required=True, help='state dict of the shared encoder')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--lr', type=float, default=1.5e-4)
    p.add_argument('--head-lr', type=float, default=4e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--temperature', type=float, default=.08)
    p.add_argument('--margin', type=float, default=.1)
    p.add_argument('--unknown-level', type=float, default=.55)
    p.add_argument('--open-weight', type=float, default=.5)
    p.add_argument('--cos-weight', type=float, default=.15)
    p.add_argument('--same-name-margin', type=float, default=.05)
    p.add_argument('--kd-weight', type=float, default=0.,
                   help='cosine distillation to the initial model on query (and reference) embeddings')
    p.add_argument('--ref-mode', choices=['grad', 'frozen'], default='grad',
                   help='encode training references with gradients every step, or without gradients every --ref-every steps')
    p.add_argument('--ref-every', type=int, default=20)
    p.add_argument('--cross-kind', action='store_true',
                   help='add the training identities of the other kind, cut to this aspect ratio')
    p.add_argument('--own-fraction', type=float, default=.6, help='share of each batch from the own kind')
    p.add_argument('--shift', type=float, default=.05)
    p.add_argument('--scale-jitter', type=float, default=.07)
    p.add_argument('--min-resolution', type=float, default=.35)
    p.add_argument('--overlay-prob', type=float, default=.5)
    p.add_argument('--erase-prob', type=float, default=.3)
    p.add_argument('--edge-prob', type=float, default=.1)
    p.add_argument('--min-inside', type=float, default=.999)
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--gpu-bank-bytes', type=float, default=4e9)
    p.add_argument('--seed', type=int, default=52817)
    p.add_argument('--device', default='cuda')
    train(p.parse_args())


if __name__ == '__main__':
    main()
