"""Train the card-frame locator on synthetic scene sets.

Example::

    python -m bdon_vision.train_locator --train 'scenes/train-*' \
        --validation 'scenes/validation-*' --output runs/locator --steps 12000

Scenes are decoded once into memory (long edge capped), then sampled with
scale, crop, two-scene mosaic, color, blur, noise and JPEG augmentation.
"""
import argparse
import glob
import json
import math
import random
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from .locator import CLASSES, PAD_VALUE, STRIDE, decode, encode_targets, letterbox
from .locator_eval import Accumulator, load_scenes, resolve_ignore
from .locator_model import LocatorNet, export_onnx

CACHE_EDGE = 1280


def _decode_scene(args):
    path, edge, jpeg = args
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'unreadable image: {path}')
    h, w = image.shape[:2]
    scale = min(1., edge / max(h, w))
    if scale < 1:
        image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    if jpeg:
        # Large sets are held as high-quality JPEG bytes to bound memory.
        return cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, jpeg])[1], (w, h), scale
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (w, h), scale


def scene_image(scene):
    """Cached RGB image of a scene (long edge capped, see ``cache_scale``)."""
    image = scene['image']
    if image.ndim == 1:
        return cv2.cvtColor(cv2.imdecode(image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return image


def cache_scenes(scenes, edge=CACHE_EDGE, workers=8, jpeg=0):
    with Pool(workers) as pool:
        decoded = pool.map(_decode_scene, [(s['path'], edge, jpeg) for s in scenes], chunksize=8)
    for scene, (image, (w, h), scale) in zip(scenes, decoded):
        scene['image'], scene['width'], scene['height'], scene['cache_scale'] = image, w, h, scale
        scene['ignore'] = resolve_ignore(scene, w, h)
    return scenes


def _visible(boxes, rect):
    x0 = np.maximum(boxes[:, 0], rect[0])
    y0 = np.maximum(boxes[:, 1], rect[1])
    x1 = np.minimum(boxes[:, 0] + boxes[:, 2], rect[0] + rect[2])
    y1 = np.minimum(boxes[:, 1] + boxes[:, 3], rect[1] + rect[3])
    return np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None) / np.maximum(boxes[:, 2] * boxes[:, 3], 1e-9)


class SceneSampler(torch.utils.data.Dataset):
    """Random augmented square views; every item is independently sampled."""

    def __init__(self, scenes, size=512, length=10 ** 7, edge_range=(448, 832), mosaic=.3, seed=0):
        self.scenes, self.size, self.length = scenes, size, length
        self.edge_range, self.mosaic, self.seed = edge_range, mosaic, seed

    def __len__(self):
        return self.length

    def view(self, rng):
        scene = self.scenes[rng.randrange(len(self.scenes))]
        image = scene_image(scene)
        s = self.size
        long_edge = rng.uniform(*self.edge_range)
        r = rng.random()
        if r < .3:
            long_edge *= rng.uniform(1., 2.6)
        elif r < .4:
            long_edge *= rng.uniform(.65, 1.)
        scale = long_edge / max(scene['width'], scene['height'])
        k = scale / scene['cache_scale']
        nh, nw = max(1, round(image.shape[0] * k)), max(1, round(image.shape[1] * k))
        # Crop before resizing when enlarging, to keep the work bounded.
        ox = rng.uniform(0, max(0., nw - s)) if nw > s else -rng.uniform(0, s - nw) * (rng.random() < .5)
        oy = rng.uniform(0, max(0., nh - s)) if nh > s else -rng.uniform(0, s - nh) * (rng.random() < .5)
        src = (max(0, int(ox / k)), max(0, int(oy / k)))
        src_end = (min(image.shape[1], int(math.ceil((ox + s) / k)) + 1), min(image.shape[0], int(math.ceil((oy + s) / k)) + 1))
        patch = image[src[1]:src_end[1], src[0]:src_end[0]]
        pw, ph = max(1, round(patch.shape[1] * k)), max(1, round(patch.shape[0] * k))
        # Mixed resampling keeps the model robust to browser canvas scaling.
        interp = rng.choice([cv2.INTER_AREA, cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC] if k < 1 else [cv2.INTER_LINEAR, cv2.INTER_CUBIC])
        patch = cv2.resize(patch, (pw, ph), interpolation=interp)
        canvas = np.full((s, s, 3), PAD_VALUE, np.uint8)
        dx, dy = round(src[0] * k - ox), round(src[1] * k - oy)
        x0, y0 = max(0, dx), max(0, dy)
        x1, y1 = min(s, dx + pw), min(s, dy + ph)
        if x1 > x0 and y1 > y0:
            canvas[y0:y1, x0:x1] = patch[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
        shift = np.array([-ox, -oy, 0, 0], np.float32)
        boxes = scene['boxes'] * scale + shift
        ignore = scene['ignore'] * scale + shift
        content = [-ox, -oy, scene['width'] * scale, scene['height'] * scale]
        # Share of each tile that is unoccluded within the screenshot bounds.
        occlusion = np.minimum(scene['visible'] / np.maximum(_visible(boxes, content), 1e-6), 1.) if len(boxes) else scene['visible']
        return canvas, boxes, scene['labels'], ignore, content, occlusion

    def split(self, a, b, rng):
        """Join two views along a vertical or horizontal seam."""
        s = self.size
        cut = round(rng.uniform(.3, .7) * s)
        vertical = rng.random() < .5
        canvas = a[0].copy()
        regions = ([0, 0, cut, s], [cut, 0, s - cut, s]) if vertical else ([0, 0, s, cut], [0, cut, s, s - cut])
        rx, ry, rw, rh = regions[1]
        canvas[ry:ry + rh, rx:rx + rw] = b[0][ry:ry + rh, rx:rx + rw]
        parts = []
        for view, region in zip((a, b), regions):
            _, boxes, labels, ignore, content, occlusion = view
            cx0, cy0 = max(content[0], region[0]), max(content[1], region[1])
            cx1 = min(content[0] + content[2], region[0] + region[2])
            cy1 = min(content[1] + content[3], region[1] + region[3])
            parts.append((boxes, labels, ignore, [cx0, cy0, max(0, cx1 - cx0), max(0, cy1 - cy0)], occlusion))
        return canvas, parts

    def photometric(self, image, rng):
        x = image.astype(np.float32)
        if rng.random() < .8:
            x = (x - 128) * rng.uniform(.75, 1.3) + 128 + rng.uniform(-30, 30)
        if rng.random() < .5:
            gray = x.mean(axis=2, keepdims=True)
            x = gray + (x - gray) * rng.uniform(.5, 1.4)
        if rng.random() < .3:
            x = x * np.array([rng.uniform(.88, 1.12) for _ in range(3)], np.float32)
        x = np.clip(x, 0, 255)
        if rng.random() < .25:
            x = cv2.GaussianBlur(x, (0, 0), rng.uniform(.3, 1.3))
        if rng.random() < .25:
            x = x + np.random.default_rng(rng.randrange(2 ** 31)).normal(0, rng.uniform(2, 8), x.shape).astype(np.float32)
        x = np.clip(x, 0, 255).astype(np.uint8)
        if rng.random() < .6:
            ok, buf = cv2.imencode('.jpg', x, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(28, 95)])
            x = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        return x

    def __getitem__(self, index):
        rng = random.Random(self.seed * 1000003 + index)
        first = self.view(rng)
        if rng.random() < self.mosaic:
            canvas, parts = self.split(first, self.view(rng), rng)
        else:
            canvas = first[0]
            parts = [first[1:]]
        boxes, labels, ignore = [], [], []
        for b, l, ig, content, occlusion in parts:
            vis = _visible(b, content) * occlusion if len(b) else np.zeros(0)
            keep = vis >= .5
            boxes.append(b[keep])
            labels.append(l[keep])
            ignore.append(b[(vis > 0) & ~keep])
            if len(ig):
                ignore.append(ig[_visible(ig, content) > 0])
        canvas = self.photometric(canvas, rng)
        target = encode_targets(np.concatenate(boxes), np.concatenate(labels), np.concatenate(ignore), canvas.shape[:2])
        x = torch.from_numpy(canvas).permute(2, 0, 1).contiguous()
        return (x, torch.from_numpy(target['heatmap']), torch.from_numpy(target['loss_mask']),
                torch.from_numpy(target['box']), torch.from_numpy(target['weight']))


def focal_loss(logits, target, mask):
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = (1 - pos) * mask[:, None]
    pos_loss = -torch.log(p) * (1 - p) ** 2 * pos
    neg_loss = -torch.log(1 - p) * p ** 2 * (1 - target) ** 4 * neg
    return (pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1)


def regression_loss(size, offset, box, weight):
    n, _, h, w = size.shape
    ys = torch.arange(h, device=size.device, dtype=torch.float32).view(1, h, 1)
    xs = torch.arange(w, device=size.device, dtype=torch.float32).view(1, 1, w)
    valid = weight > 0
    if not valid.any():
        return size.sum() * 0, size.sum() * 0
    tw, th = box[:, 2].clamp(min=1), box[:, 3].clamp(min=1)
    tox, toy = box[:, 0] / STRIDE - xs, box[:, 1] / STRIDE - ys
    tlw, tlh = torch.log(tw / STRIDE), torch.log(th / STRIDE)
    l1 = ((size[:, 0] - tlw).abs() + (size[:, 1] - tlh).abs() + (offset[:, 0] - tox).abs() + (offset[:, 1] - toy).abs())
    pw, ph = torch.exp(size[:, 0].clamp(max=8)) * STRIDE, torch.exp(size[:, 1].clamp(max=8)) * STRIDE
    pcx, pcy = (xs + offset[:, 0]) * STRIDE, (ys + offset[:, 1]) * STRIDE
    px0, py0, px1, py1 = pcx - pw / 2, pcy - ph / 2, pcx + pw / 2, pcy + ph / 2
    tx0, ty0, tx1, ty1 = box[:, 0] - tw / 2, box[:, 1] - th / 2, box[:, 0] + tw / 2, box[:, 1] + th / 2
    inter = (torch.min(px1, tx1) - torch.max(px0, tx0)).clamp(min=0) * (torch.min(py1, ty1) - torch.max(py0, ty0)).clamp(min=0)
    union = pw * ph + tw * th - inter
    hull = (torch.max(px1, tx1) - torch.min(px0, tx0)) * (torch.max(py1, ty1) - torch.min(py0, ty0))
    giou = inter / union - (hull - union) / hull.clamp(min=1e-6)
    objects = weight.sum().clamp(min=1)
    return (l1 * weight)[valid].sum() / objects, ((1 - giou) * weight)[valid].sum() / objects


def predict(model, image_rgb, long_edge, threshold=.05, device='cuda'):
    canvas, scale = letterbox(image_rgb, long_edge)
    x = torch.from_numpy(canvas).permute(2, 0, 1)[None].to(device).float().div_(255)
    with torch.inference_mode():
        heat, size, offset = model(x)
        heat = torch.sigmoid(heat.float())[0].cpu().numpy()
        size = (torch.exp(size.float().clamp(max=8)) * STRIDE)[0].cpu().numpy()
        offset = offset.float()[0].cpu().numpy()
    return decode(heat, size, offset, scale, threshold, image_size=image_rgb.shape[1::-1]), scale


def evaluate_model(model, scenes, long_edge, device='cuda'):
    was_training = model.training
    model.eval()
    acc = Accumulator()
    for scene in scenes:
        detections, _ = predict(model, scene_image(scene), long_edge, device=device)
        for d in detections:
            d['bbox'] = [v / scene['cache_scale'] for v in d['bbox']]
        acc.add(detections, scene, scene['width'], scene['height'], long_edge / max(scene['width'], scene['height']))
    model.train(was_training)
    return acc


def expand(patterns):
    out = []
    for pattern in patterns:
        found = sorted(p for p in glob.glob(pattern) if Path(p, 'truth.json').exists())
        if not found:
            raise ValueError(f'no scene chunks match {pattern}')
        out += found
    return out


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'not serializable: {type(value).__name__}')


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=_plain), encoding='utf-8')
    tmp.replace(path)


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    train_scenes = cache_scenes(load_scenes(expand(args.train)), workers=args.workers, jpeg=args.cache_jpeg)
    valid_scenes = load_scenes(expand(args.validation)) if args.validation else []
    if args.validation_limit and len(valid_scenes) > args.validation_limit:
        step = len(valid_scenes) / args.validation_limit
        valid_scenes = [valid_scenes[int(i * step)] for i in range(args.validation_limit)]
    valid_scenes = cache_scenes(valid_scenes, workers=args.workers, jpeg=args.cache_jpeg) if valid_scenes else []
    print(json.dumps({'train_scenes': len(train_scenes), 'validation_scenes': len(valid_scenes),
                      'cache_s': round(time.time() - started, 1)}), flush=True)
    sampler = SceneSampler(train_scenes, size=args.crop, length=args.steps * args.batch, seed=args.seed)
    device = torch.device(args.device)
    cuda = device.type == 'cuda'
    if args.threads:
        torch.set_num_threads(args.threads)
    loader = torch.utils.data.DataLoader(sampler, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                                         pin_memory=cuda, drop_last=True, persistent_workers=True, prefetch_factor=4)
    model = LocatorNet(width=args.width, backbone_weights=args.backbone).to(device).to(memory_format=torch.channels_last)
    if args.initial:
        model.load_state_dict(torch.load(args.initial, map_location='cpu', weights_only=True))
    backbone = [p for n, p in model.named_parameters() if n.startswith(('stem', 'c3', 'c4', 'c5'))]
    rest = [p for n, p in model.named_parameters() if not n.startswith(('stem', 'c3', 'c4', 'c5'))]
    optimizer = torch.optim.AdamW([{'params': backbone, 'lr': args.lr * args.backbone_lr},
                                   {'params': rest, 'lr': args.lr}], weight_decay=args.weight_decay)
    base = [g['lr'] for g in optimizer.param_groups]
    history, best = [], None
    t0 = time.time()
    for step, (x, heat, mask, box, weight) in enumerate(loader, 1):
        frac = step / args.steps
        factor = min(1., step / args.warmup) * (.5 * (1 + math.cos(math.pi * frac)) * .98 + .02)
        for g, lr in zip(optimizer.param_groups, base):
            g['lr'] = lr * factor
        x = x.to(device, non_blocking=True).float().div_(255).contiguous(memory_format=torch.channels_last)
        heat, mask = heat.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        box, weight = box.to(device, non_blocking=True), weight.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cuda and args.bf16):
            logits, size, offset = model(x)
        logits, size, offset = logits.float(), size.float(), offset.float()
        lh = focal_loss(logits, heat, mask)
        l1, giou = regression_loss(size, offset, box, weight)
        loss = lh + args.l1_weight * l1 + args.giou_weight * giou
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
        if step % 50 == 0 or step == 1:
            status = {'step': step, 'steps': args.steps, 'loss': round(float(loss.detach()), 4),
                      'heat': round(float(lh.detach()), 4), 'l1': round(float(l1.detach()), 4),
                      'giou': round(float(giou.detach()), 4), 'lr': optimizer.param_groups[1]['lr'],
                      'elapsed_s': round(time.time() - t0, 1),
                      'gpu_max_gib': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if cuda else None}
            write_json(out / 'progress.json', status)
            print(json.dumps(status), flush=True)
        if valid_scenes and (step % args.eval_every == 0 or step == args.steps):
            acc = evaluate_model(model, valid_scenes, args.eval_edge, device)
            s = acc.summary(.3, strata=False)
            ap50, ap75 = acc.average_precision(.5), acc.average_precision(.75)
            score = ap50 + ap75 + s['iou0.75']['recall'] * s['iou0.75']['precision'] + s['iou0.5']['recall'] * s['iou0.5']['precision']
            row = {'step': step, 'ap50': ap50, 'ap75': ap75, 'at_0.3': s, 'score': score}
            history.append(row)
            write_json(out / 'validation-history.json', history)
            print(json.dumps({'validation': row}), flush=True)
            torch.save(model.state_dict(), out / 'last.pt')
            # Later checkpoints win ties: they were trained at a lower learning rate.
            if best is None or score >= best['score']:
                best = row
                torch.save(model.state_dict(), out / 'best.pt')
        if (out / 'STOP').exists():
            print(json.dumps({'stopped_at': step}), flush=True)
            break
    if not valid_scenes:
        torch.save(model.state_dict(), out / 'best.pt')
    torch.save(model.state_dict(), out / 'last.pt')
    model.load_state_dict(torch.load(out / 'best.pt', map_location='cpu', weights_only=True))
    export_onnx(model.float().to(memory_format=torch.contiguous_format), out / 'locator.onnx')
    write_json(out / 'training.json', {'args': vars(args), 'best': best, 'train_scenes': len(train_scenes),
                                       'validation_scenes': len(valid_scenes), 'elapsed_s': round(time.time() - started, 1),
                                       'classes': list(CLASSES), 'stride': STRIDE})
    print(json.dumps({'done': True, 'best': best}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--train', nargs='+', required=True, help='Glob(s) of scene chunk directories')
    p.add_argument('--validation', nargs='*', default=[])
    p.add_argument('--validation-limit', type=int, default=0, help='Evenly subsample validation scenes')
    p.add_argument('--output', required=True)
    p.add_argument('--backbone', help='torchvision MobileNetV3-Large ImageNet state dict')
    p.add_argument('--initial', help='Locator state dict to continue from')
    p.add_argument('--steps', type=int, default=12000)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--crop', type=int, default=512)
    p.add_argument('--width', type=int, default=96)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--backbone-lr', type=float, default=.5, help='Backbone learning-rate multiplier')
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--l1-weight', type=float, default=.5)
    p.add_argument('--giou-weight', type=float, default=2.)
    p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--eval-edge', type=int, default=640)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--cache-jpeg', type=int, default=0, help='Hold cached scenes as JPEG of this quality (0 = raw)')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--device', default='cuda')
    p.add_argument('--threads', type=int, default=0, help='CPU threads for PyTorch (0 = default)')
    p.add_argument('--no-bf16', dest='bf16', action='store_false')
    train(p.parse_args())


if __name__ == '__main__':
    main()
