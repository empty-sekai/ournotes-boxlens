"""Acquisition-degraded field and rank crops rendered per card kind.

Each sample renders one native card tile (artwork, frame, rank icon) with one
localized parameter text layer, composites it on a list-like background, and
then imitates screen capture on the whole tile: it is shrunk to the screen
scale, blurred, padded to a random codec-block phase, compressed one to four
times with JPEG or WebP, and only then sampled with the deployed crop
geometry. Field and rank crops come from the same degraded tile. Optionally
a share of samples gets a UI-like panel over the field or the rank icon; a
value whose glyph or icon pixels are at least 2 % covered becomes class 0,
while panels that only overlap empty parts of the crop keep the value.
"""
import argparse
import hashlib
import io
import json
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from .assets import native_module, read, write
from .kind_fields import (FIELD_STORED, LOCALES, LOGICAL, MARGIN, META_INT, MODES, RANK_CENTER, RANK_INPUT, RANK_SIDE,
                          RANK_STORED, crop_field, crop_rank)

LOCALE_COLUMNS = {'ja': '_japanese', 'en': '_english', 'zh-Hant': '_traditionalChinese',
                  'zh-Hans': '_simplifiedChinese', 'ko': '_korean'}
PROFILES = {
    'train': dict(ratios=[.26, .33, .41, .49, .61, .77, .93, 1.09, 1.4], qualities=[34, 46, 58, 73, 87, 95], blur=(.05, .95)),
    'validation': dict(ratios=[.29, .37, .45, .55, .69, .85, 1.01, 1.25], qualities=[38, 50, 64, 79, 91], blur=(.12, 1.10)),
}
PAD = 32
OCCLUSION_RATE = 0.
OCCLUDED = .02  # a value whose glyph or icon pixels are covered at least this much is unreadable
STATE = {}


def _init(data, kind, split, occlusion_rate=OCCLUSION_RATE):
    native = native_module(data)
    STATE.update(native=native, kind=kind, split=split, occlusion_rate=occlusion_rate,
                 renderers={loc: native.CardRenderer(col) for loc, col in LOCALE_COLUMNS.items()})
    catalog = [c for c in read(Path(data)/'catalog.json')['cards'] if c['kind'] == kind]
    if split == 'train':
        catalog = [c for c in catalog if c['id'] % 7 != 0]
    STATE['pool'] = catalog


@lru_cache(maxsize=1024)
def _base(card_index, icon):
    """Card tile without parameter text; icon None hides the rank icon."""
    card = STATE['pool'][card_index]
    native = STATE['native']
    state = native.CardState(asset_id=card['asset_id'], rarity=card['rarity'], card_type=card['card_type'],
                             param='hide' if icon is None else 'level', level=None, rank=icon)
    return STATE['renderers']['zh-Hans'].render(state, STATE['kind'], scale=1)


@lru_cache(maxsize=4096)
def _text(locale, mode, value):
    native = STATE['native']
    kind = STATE['kind']
    power = (value, value+17, value+31) if kind == 'member' else (value, value+7, value+11)
    state = native.CardState(asset_id=1, rarity=2, card_type=1, param=mode,
                             level=value if mode == 'level' else 1, awake_count=value if mode == 'training' else 1,
                             power=power)
    return STATE['renderers'][locale].render(state, kind, scale=1, only='text')


@lru_cache(maxsize=1024)
def _icon_mask(card_index, icon):
    """Pixels where the rank icon changes the tile."""
    if icon is None:
        return None
    shown = np.asarray(_base(card_index, icon)).astype(np.int16)
    hidden = np.asarray(_base(card_index, None)).astype(np.int16)
    return np.abs(shown-hidden).max(axis=2) > 24


def _occlude(canvas, rng, boxes):
    """Draw a UI-like panel over part of one target box; return the opaque-enough panel mask."""
    width, height = canvas.size
    left, top, w, h = boxes[rng.randrange(len(boxes))]
    side = rng.choice(['top', 'bottom', 'left', 'right'])
    depth = rng.uniform(.15, .9)
    if side in ('top', 'bottom'):
        x0, x1 = left-rng.uniform(0, 90), left+w+rng.uniform(0, 90)
        if rng.random() < .3:
            x0, x1 = (x0, left+w*rng.uniform(.3, .8)) if rng.random() < .5 else (left+w*rng.uniform(.2, .7), x1)
        y0, y1 = (top-rng.uniform(5, 60), top+h*depth) if side == 'top' else (top+h*(1-depth), top+h+rng.uniform(5, 60))
    else:
        y0, y1 = top-rng.uniform(0, 40), top+h+rng.uniform(0, 40)
        x0, x1 = (left-rng.uniform(5, 60), left+w*depth) if side == 'left' else (left+w*(1-depth), left+w+rng.uniform(5, 60))
    dark = rng.random() < .7
    color = tuple(rng.randrange(15, 80) for _ in range(3)) if dark else tuple(rng.randrange(200, 256) for _ in range(3))
    overlay = Image.new('RGBA', (width, height))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle((x0, y0, x1, y1), radius=rng.uniform(0, 18), fill=color+(rng.randrange(175, 250),))
    if rng.random() < .6:
        try:
            font = ImageFont.load_default(size=rng.randint(14, 30))
        except TypeError:
            font = ImageFont.load_default()
        text = ''.join(rng.choice('ABCDEFGHJKLMNPRSTUVWXYZ0123456789 ') for _ in range(rng.randint(2, 9)))
        draw.text((x0+rng.uniform(0, max(1, x1-x0-40)), y0+rng.uniform(0, max(1, y1-y0-20))), text, font=font,
                  fill=(255, 255, 255, 255) if dark else (40, 40, 60, 255))
    canvas.alpha_composite(overlay)
    return np.asarray(overlay.getchannel('A')) >= 128


def _covered(mask, target):
    """Share of target pixels under the panel (0 when the target is empty)."""
    if target is None or not target.any():
        return 0.
    return float((mask & target).sum())/float(target.sum())


def _sample(rng):
    kind = STATE['kind']
    profile = PROFILES[STATE['split']]
    lw, lh = LOGICAL[kind]
    card_index = rng.randrange(len(STATE['pool']))
    locale = rng.choice(LOCALES)
    roll = rng.random()
    if kind == 'member' and roll < .2:
        mode, value = 'training', rng.randint(1, 5)
        label = 100+value
    elif roll < .6:
        mode, value = 'level', rng.randint(1, 100)
        label = value
    else:
        others = ['hide', 'performance', 'technic', 'visual']+(['total'] if kind == 'member' else [])
        mode = rng.choice(others)
        value = rng.randrange(64)*293+137 if kind == 'member' else rng.randrange(64)*17+37
        label = 0
    if mode == 'hide':
        icon, rank = None, 0
    else:
        icon = 0 if rng.random() < .05 else rng.randint(1, 5)
        rank = icon
    width, height = lw+2*PAD, lh+2*PAD
    background = tuple(rng.randrange(90, 240) for _ in range(3))+(255,)
    canvas = Image.new('RGBA', (width, height), background)
    canvas.alpha_composite(_base(card_index, icon))
    glyphs = None
    if mode != 'hide':
        text = _text(locale, mode, value)
        offset = (rng.randint(-2, 2), rng.randint(-2, 2))
        canvas.alpha_composite(text, offset)
        alpha = np.asarray(text.getchannel('A')) >= 128
        ox, oy = offset
        glyphs = np.zeros_like(alpha)
        source = alpha[max(0, -oy):height-max(0, oy), max(0, -ox):width-max(0, ox)]
        glyphs[max(0, oy):height+min(0, oy), max(0, ox):width+min(0, ox)] = source
    if label == 0 and rng.random() < .1:
        # A partly erased status value is still not a level or training count.
        cut = rng.randrange(PAD+20, PAD+130)
        canvas.paste(_base(card_index, icon).crop((cut, 0, width, height)), (cut, 0),
                     _base(card_index, icon).crop((cut, 0, width, height)))
    occluded, rank_occluded, field_latent, rank_latent = 0, 0, label, rank
    if rng.random() < STATE['occlusion_rate']:
        cx, cy = RANK_CENTER[kind]
        field_box = (PAD-6, PAD+lh-56, 144, 56)
        icon_box = (PAD+cx-32, PAD+cy-32*95/102, 64, 64*95/102)
        mask = _occlude(canvas, rng, [field_box, icon_box])
        # The value is unreadable only when the panel covers its own glyphs or icon,
        # not when it merely overlaps empty parts of the crop.
        if label > 0 and _covered(mask, glyphs) >= OCCLUDED:
            label, occluded = 0, 1
        if rank > 0 and _covered(mask, _icon_mask(card_index, icon)) >= OCCLUDED:
            rank, rank_occluded = 0, 1
    ratio = rng.choice(profile['ratios'])*rng.uniform(.94, 1.06)
    tile = ImageEnhance.Brightness(canvas.convert('RGB')).enhance(rng.uniform(.88, 1.12))
    small_size = (max(8, round(width*ratio)), max(8, round(height*ratio)))
    tile = tile.resize(small_size, rng.choice([Image.Resampling.BILINEAR, Image.Resampling.BICUBIC, Image.Resampling.LANCZOS]))
    tile = tile.filter(ImageFilter.GaussianBlur(rng.uniform(*profile['blur'])))
    px, py = rng.randrange(16), rng.randrange(16)
    tile = Image.fromarray(np.pad(np.asarray(tile), ((py, 16-py), (px, 16-px), (0, 0)), mode='edge'))
    for _ in range(rng.choice([1, 2, 3, 4])):
        buffer = io.BytesIO()
        tile.save(buffer, format=rng.choice(['JPEG', 'WEBP']), quality=rng.choice(profile['qualities']))
        tile = Image.open(io.BytesIO(buffer.getvalue())).convert('RGB')
    bgr = np.ascontiguousarray(np.asarray(tile)[:, :, ::-1])
    scale_x, scale_y = small_size[0]/width, small_size[1]/height
    bbox = [px+PAD*scale_x, py+PAD*scale_y, lw*scale_x, lh*scale_y]
    field = crop_field(bgr, kind, bbox, MARGIN)
    rank_crop = crop_rank(bgr, kind, bbox, MARGIN)
    meta = {'id': STATE['pool'][card_index]['id'], 'foreign': 0, 'locale': LOCALES.index(locale),
            'mode': MODES.index(mode), 'icon_inside': 1, 'occluded': occluded,
            'rank_occluded': rank_occluded, 'width': bbox[2]}
    return field, label, rank_crop, rank, meta, field_latent, rank_latent


def _chunk(job):
    seed, start, count = job
    rng = random.Random(seed)
    rows = []
    for i in range(count):
        field, label, rank_crop, rank, meta, field_latent, rank_latent = _sample(rng)
        rows.append((field, label, rank_crop, rank, {**meta, 'dataset': 0, 'scene': start+i, 'card': 0},
                     field_latent, rank_latent))
    return rows


def generate(data, kind, output, count, seed, split, workers=8, chunk=500, occlusion_rate=OCCLUSION_RATE):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    jobs = [(seed*1000003+i, i, min(chunk, count-i)) for i in range(0, count, chunk)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(str(data), kind, split, occlusion_rate)) as pool:
        for part in pool.map(_chunk, jobs):
            rows += part
            print(json.dumps({'generated': len(rows), 'total': count, 'kind': kind, 'split': split}), flush=True)
    summary = {}
    for part, image_index, label_index, shape in [('field', 0, 1, (FIELD_STORED[1], FIELD_STORED[0], 3)),
                                                   ('rank', 2, 3, (RANK_STORED, RANK_STORED, 3))]:
        images = np.stack([r[image_index] for r in rows])
        assert images.shape[1:] == shape, images.shape
        labels = np.array([r[label_index] for r in rows], np.int16)
        prefix = f'{kind}-{part}'
        np.save(output/f'{prefix}-images.npy', images)
        np.save(output/f'{prefix}-labels.npy', labels)
        meta = {k: np.array([r[4].get(k, -1) for r in rows], np.int32) for k in META_INT}
        meta['latent'] = np.array([r[5 if part == 'field' else 6] for r in rows], np.int32)
        if part == 'rank':
            meta['occluded'] = np.array([r[4]['rank_occluded'] for r in rows], np.int32)
        meta['width'] = np.array([r[4]['width'] for r in rows], np.float32)
        meta['source_resolution_vocabulary'] = np.array(['rendered-tile'])
        meta['source_resolution'] = np.zeros(len(rows), np.int32)
        meta['profile_vocabulary'] = np.array(['hard-'+split])
        meta['profile'] = np.zeros(len(rows), np.int32)
        np.savez(output/f'{prefix}-meta.npz', **meta)
        summary[prefix] = {'count': len(rows), 'labels': {str(k): int(v) for k, v in sorted(Counter(labels.tolist()).items())}}
    write(output/'manifest.json', {'schema': 'ournotes-boxlens.kind-crop-bank/1', 'source': 'rendered-acquisition',
                                   'kind': kind, 'split': split, 'seed': seed, 'samples': count, 'profile': PROFILES[split],
                                   'catalog_sha256': hashlib.sha256((Path(data)/'catalog.json').read_bytes()).hexdigest(),
                                   'banks': summary, 'margin_logical': MARGIN, 'field_stored_wh': list(FIELD_STORED),
                                   'rank_stored_side': RANK_STORED, 'rank_logical_side': RANK_SIDE, 'rank_input': RANK_INPUT,
                                   'occlusion_rate': occlusion_rate, 'occluded_threshold': OCCLUDED,
                                   'complete': True})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--kind', choices=['member', 'snap'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=60000)
    p.add_argument('--seed', type=int, default=130001)
    p.add_argument('--split', choices=list(PROFILES), default='train')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--occlusion-rate', type=float, default=OCCLUSION_RATE,
                   help='Share of samples with a UI-like panel over the field or icon')
    a = p.parse_args()
    generate(a.data, a.kind, a.output, a.count, a.seed, a.split, a.workers, occlusion_rate=a.occlusion_rate)
