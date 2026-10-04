"""List-screen scene generator (v2) with a parallel, resumable dataset writer.

Native card tiles from ``assets.native_module`` are placed with the list widget
geometry: a 1920x1080 reference canvas scaled by ``min(W/1920, H/1080)``,
rows of six member or four snap cells centred horizontally, 180 reference
units of top padding, and a vertical scroll offset. Scenes add blurred
game-art backgrounds with haze, a header (title, sort and view buttons),
light UI distractors, out-of-gallery tiles (``foreign_cards``) and card-free
screens. The output is a ``bdon-synthetic/2`` dataset: the ``cards`` and
``ignored_cards`` lists keep their meaning and every new key is additive.
"""
import argparse
import colorsys
import hashlib
import io
import json
import math
import multiprocessing
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from .assets import read, write, native_module
from .scenarios import PROFILES, VERSION, acquire

GENERATOR = 'scenes-v2.1'
LOCALE_COLUMNS = {'ja': '_japanese', 'en': '_english', 'zh-Hant': '_traditionalChinese',
                  'zh-Hans': '_simplifiedChinese', 'ko': '_korean'}
MODES = ('level', 'training', 'total', 'performance', 'technic', 'visual', 'hide')
MEMBER_ONLY_MODES = ('training', 'total')
RARITIES = {'member': (2, 3, 4, 10, 20), 'snap': (2, 3, 4, 10)}
# Reference-canvas geometry of the list widgets and their row prefabs.
GRID = {'member': dict(tile=(224, 294), columns=6, spacing=27),
        'snap': dict(tile=(326, 184), columns=4, spacing=36)}
REFERENCE = (1920, 1080)
LIST_PADDING_TOP, LIST_PADDING_BOTTOM = 180, 60
RANK_CENTER = {'member': (194.5, 267.2), 'snap': (306.1, 153.8)}
TITLE_TEXT = {'member': 'ui_title_member_card_list', 'snap': 'ui_title_support_card_list'}
SORT_KEYS = {'member': ['AcquiredAt', 'Release', 'TotalPower', 'Rarity', 'Level', 'Character', 'Awakening',
                        'MemberCardRank', 'Performance', 'Technique', 'Visual'],
             'snap': ['AcquiredAt', 'Release', 'TotalPower', 'Rarity', 'Level', 'SupportCardRank',
                      'Performance', 'Technique', 'Visual']}
FOREIGN_METHODS = ('cross_kind', 'flip_recolor', 'mix', 'procedural')
DISTRACTOR_ICONS = ['IconSetting', 'IconInfo', 'IconHelp', 'IconRanking', 'IconSearchOutLine', 'IconFriend',
                    'IconGood', 'IconReward', 'IconLog', 'IconFavorite', 'IconPlay', 'IconTag', 'IconDiary']
SCHEMA_EXTENSION = {
    'screenshot': ['kind', 'display_mode', 'negative', 'screen_resolution', 'layout', 'background',
                   'ui_elements', 'foreign_cards', 'ignored_foreign_cards'],
    'card': ['rarity', 'card_type', 'rank_visible', 'selected', 'event_bonus', 'badge', 'occluded_fraction'],
    'ignored_card': ['bbox', 'visible_fraction'],
    'foreign_card': 'card fields without id, plus foreign_method and foreign_sources',
}


def scene_backgrounds(data, profile):
    """Optional scene backdrops; every fifth one is kept out of the training profile."""
    def number(path):
        return int(''.join(ch for ch in path.stem if ch.isdigit()) or 0)
    paths = sorted((Path(data)/'backgrounds/adv-stage').glob('*.webp'))
    return [p for p in paths if profile != 'train' or number(p) % 5 != 0]


def heldout(card):
    """Identities kept out of encoder training (members and snaps separately)."""
    return card['id'] % 7 == 0


def list_layout(kind, width, height, ui_scale=1.):
    """Native list geometry in output pixels for scroll offset 0."""
    base = min(width/REFERENCE[0], height/REFERENCE[1])
    s = base*ui_scale
    grid = GRID[kind]
    tw, th = grid['tile']
    row = grid['columns']*tw+(grid['columns']-1)*grid['spacing']
    return dict(scale=s, base_scale=base, columns=grid['columns'],
                origin=(width/2-row*s/2, LIST_PADDING_TOP*s),
                pitch=((tw+grid['spacing'])*s, (th+grid['spacing'])*s), tile=(tw*s, th*s))


def _box_fraction(mask, box, width, height):
    """Covered fraction of an in-screen box; out-of-screen parts count as covered."""
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return 1.
    x0, y0, x1, y1 = max(0, math.floor(x)), max(0, math.floor(y)), min(width, math.ceil(x+w)), min(height, math.ceil(y+h))
    inside = max(0, x1-x0)*max(0, y1-y0)
    covered = int(mask[y0:y1, x0:x1].sum()) if inside else 0
    return 1.-(inside-covered)/max(1., w*h)


def _composite(base, image, x, y):
    """Alpha-composite with clipping at every edge of the destination."""
    left, top = max(0, -x), max(0, -y)
    right, bottom = min(image.width, base.width-x), min(image.height, base.height-y)
    if right > left and bottom > top:
        base.alpha_composite(image, (x+left, y+top), (left, top, right, bottom))


def _hue_shift(image, degrees, saturation=1.):
    hsv = np.array(image.convert('RGB').convert('HSV'), dtype=np.int16)
    hsv[..., 0] = (hsv[..., 0]+round(degrees/360*256)) % 256
    hsv[..., 1] = np.clip(hsv[..., 1]*saturation, 0, 255)
    return Image.fromarray(hsv.astype(np.uint8), 'HSV').convert('RGB')


def _color(rng, light=None):
    h = rng.random()
    s = rng.uniform(.25, .85)
    v = rng.uniform(.75, 1.) if light else rng.uniform(.25, .7) if light is False else rng.uniform(.35, 1.)
    return tuple(round(c*255) for c in colorsys.hsv_to_rgb(h, s, v))


def _crop_aspect(image, aspect, rng, zoom=1.):
    """Random crop with the requested aspect, optionally zoomed in."""
    w, h = image.size
    cw, ch = (w, w/aspect) if w/h < aspect else (h*aspect, h)
    cw, ch = cw/zoom, ch/zoom
    x, y = rng.uniform(0, w-cw), rng.uniform(0, h-ch)
    return image.crop((round(x), round(y), round(x+cw), round(y+ch)))


def _figure(size, rng):
    """Bust of a stylised character on a transparent layer."""
    w, h = size
    im = Image.new('RGBA', size)
    d = ImageDraw.Draw(im, 'RGBA')
    r = rng.uniform(.15, .3)*min(w, h)
    cx, cy = rng.uniform(.25, .75)*w, rng.uniform(.28, .55)*h
    hair, cloth, accent, eye = _color(rng), _color(rng), _color(rng, True), _color(rng)
    skin = (rng.randint(244, 255), rng.randint(212, 238), rng.randint(196, 224))
    line = tuple(max(0, c-110) for c in hair)
    lw = max(1, round(r*.035))
    length = rng.uniform(.5, 3.2)
    if rng.random() < .3:
        for sgn in (-1, 1):
            d.polygon([(cx+sgn*.45*r, cy-1.1*r), (cx+sgn*1.15*r, cy-1.9*r), (cx+sgn*1.05*r, cy-.6*r)], fill=hair, outline=line)
    d.ellipse((cx-1.2*r, cy-1.3*r, cx+1.2*r, cy+1.0*r), fill=hair, outline=line, width=lw)
    d.polygon([(cx-1.2*r, cy-.1*r), (cx-(1.3+rng.uniform(0, .35))*r, cy+length*r), (cx-.5*r, cy+(length+.25)*r),
               (cx+.5*r, cy+(length+.25)*r), (cx+(1.3+rng.uniform(0, .35))*r, cy+length*r), (cx+1.2*r, cy-.1*r)], fill=hair)
    d.rectangle((cx-.28*r, cy+.7*r, cx+.28*r, cy+1.5*r), fill=tuple(round(c*.9) for c in skin))
    d.polygon([(cx-2.1*r, h+5), (cx-1.55*r, cy+1.6*r), (cx-.32*r, cy+1.35*r), (cx+.32*r, cy+1.35*r), (cx+1.55*r, cy+1.6*r),
               (cx+2.1*r, h+5)], fill=cloth, outline=line, width=lw)
    d.polygon([(cx-.5*r, cy+1.35*r), (cx, cy+rng.uniform(1.8, 2.3)*r), (cx+.5*r, cy+1.35*r)], fill=accent, outline=line, width=lw)
    d.ellipse((cx-.85*r, cy-.85*r, cx+.85*r, cy+.75*r), fill=skin)
    d.polygon([(cx-.83*r, cy+.05*r), (cx-.35*r, cy+.85*r), (cx, cy+1.0*r), (cx+.35*r, cy+.85*r), (cx+.83*r, cy+.05*r)], fill=skin)
    d.chord((cx-1.15*r, cy-1.35*r, cx+1.15*r, cy+.1*r), 180, 360, fill=hair)
    edges = np.linspace(cx-1.05*r, cx+1.05*r, rng.randint(4, 8)+1)
    for left, right in zip(edges[:-1], edges[1:]):
        tip = (left+right)/2+rng.uniform(-.12, .12)*r
        d.polygon([(left, cy-.65*r), (right, cy-.65*r), (tip, cy+rng.uniform(-.3, .15)*r)], fill=hair)
    for sgn in (-1, 1):
        d.polygon([(cx+sgn*.72*r, cy-.55*r), (cx+sgn*.98*r, cy-.45*r), (cx+sgn*(.92+rng.uniform(0, .25))*r, cy+rng.uniform(.6, 1.5)*r),
                   (cx+sgn*.7*r, cy+.35*r)], fill=hair)
    ey, ew, eh, gap = cy+rng.uniform(.05, .2)*r, rng.uniform(.17, .25)*r, rng.uniform(.22, .33)*r, rng.uniform(.33, .42)*r
    dark = tuple(max(0, c-80) for c in eye)
    for sgn in (-1, 1):
        ex = cx+sgn*gap
        d.ellipse((ex-ew, ey-eh, ex+ew, ey+eh), fill=(255, 255, 255))
        d.ellipse((ex-ew*.85, ey-eh*.8, ex+ew*.85, ey+eh), fill=eye)
        d.ellipse((ex-ew*.45, ey-eh*.15, ex+ew*.45, ey+eh*.65), fill=dark)
        d.ellipse((ex-ew*.6, ey-eh*.65, ex-ew*.05, ey-eh*.1), fill=(255, 255, 255, 235))
        d.arc((ex-ew*1.2, ey-eh*1.15, ex+ew*1.2, ey+eh*1.05), 195, 345, fill=line, width=max(1, round(r*.07)))
        d.line([(ex-ew*.9, ey-eh*1.6), (ex+ew*.8, ey-eh*1.75+sgn*.02*r)], fill=line, width=max(1, round(r*.035)))
        d.ellipse((ex-ew, ey+eh*1.15, ex+ew, ey+eh*1.55), fill=(255, 130, 150, 80))
    my, mw = cy+.62*r, rng.uniform(.07, .2)*r
    if rng.random() < .5:
        d.chord((cx-mw, my-mw*.5, cx+mw, my+mw), 0, 180, fill=(196, 70, 92), outline=line, width=lw)
    else:
        d.arc((cx-mw, my-mw, cx+mw, my+mw*.5), 20, 160, fill=line, width=lw)
    if rng.random() < .45:
        sgn = rng.choice((-1, 1))
        bx, by, br = cx+sgn*.8*r, cy-.95*r, rng.uniform(.18, .32)*r
        d.polygon([(bx, by), (bx-2*br, by-br), (bx-2*br, by+br)], fill=accent, outline=line)
        d.polygon([(bx, by), (bx+2*br, by-br), (bx+2*br, by+br)], fill=accent, outline=line)
        d.ellipse((bx-br*.5, by-br*.5, bx+br*.5, by+br*.5), fill=accent, outline=line)
    return im.rotate(rng.uniform(-18, 18), resample=Image.Resampling.BICUBIC, center=(cx, cy))


def _procedural_art(size, rng):
    """Illustration-like tile: painted backdrop, light effects and stylised figures."""
    w, h = size
    k = 2
    yy, xx = np.mgrid[0:h*k, 0:w*k].astype(np.float32)
    angle = rng.uniform(0, 2*math.pi)
    t = np.clip((xx/(w*k)-.5)*math.cos(angle)+(yy/(h*k)-.5)*math.sin(angle)+.5, 0, 1)[..., None]
    a, b = np.array(_color(rng)), np.array(_color(rng))
    im = Image.fromarray(np.uint8(a*(1-t)+b*t)).convert('RGBA')
    d = ImageDraw.Draw(im, 'RGBA')
    if rng.random() < .3:
        base = h*k*rng.uniform(.55, .85)
        x = -10
        while x < w*k:
            bw = rng.uniform(.06, .18)*w*k
            top = base-rng.uniform(.05, .45)*h*k
            d.rectangle((x, top, x+bw, h*k), fill=(*_color(rng, False), 255))
            for _ in range(rng.randint(0, 10)):
                wx, wy = rng.uniform(x, x+bw), rng.uniform(top, h*k)
                d.rectangle((wx, wy, wx+4*k, wy+5*k), fill=(255, 240, 180, 200))
            x += bw+rng.uniform(0, 8)*k
    for _ in range(rng.randint(0, 4)):
        points = [(rng.uniform(-.2, 1.2)*w*k, rng.uniform(-.2, 1.2)*h*k) for _ in range(rng.randint(3, 5))]
        d.polygon(points, fill=(*_color(rng), rng.randint(40, 160)))
    if rng.random() < .4:
        for _ in range(rng.randint(2, 6)):
            x0 = rng.uniform(0, w*k)
            spread = rng.uniform(.05, .25)*w*k
            d.polygon([(x0, -5), (x0+rng.uniform(-.6, .6)*w*k-spread, h*k), (x0+rng.uniform(-.6, .6)*w*k+spread, h*k)],
                      fill=(255, 255, rng.randint(200, 255), rng.randint(25, 80)))
    for _ in range(rng.randint(3, 16)):
        r = rng.uniform(.02, .16)*min(w, h)*k
        cx, cy = rng.uniform(0, w*k), rng.uniform(0, h*k)
        d.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(*_color(rng, True), rng.randint(30, 120)))
    figures = rng.choices([0, 1, 2], weights=(.2, .65, .15))[0]
    for _ in range(figures):
        im.alpha_composite(_figure(im.size, rng))
    d = ImageDraw.Draw(im, 'RGBA')
    for _ in range(rng.randint(0, 12)):
        cx, cy, r = rng.uniform(0, w*k), rng.uniform(0, h*k), rng.uniform(2, 9)*k
        d.polygon([(cx, cy-r), (cx+r*.25, cy-r*.25), (cx+r, cy), (cx+r*.25, cy+r*.25), (cx, cy+r),
                   (cx-r*.25, cy+r*.25), (cx-r, cy), (cx-r*.25, cy-r*.25)], fill=(255, 255, 255, rng.randint(120, 230)))
    im = im.convert('RGB').resize((w, h), Image.Resampling.LANCZOS)
    if rng.random() < .6:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(.3, 1.2)))
    return ImageEnhance.Sharpness(im).enhance(rng.uniform(1., 2.2))


class SceneContext:
    """Per-process resources: renderers, catalogue, fonts, sprites and art."""

    def __init__(self, data, locales):
        self.data = Path(data)
        self.native = native_module(self.data)
        self.renderers = {locale: self.native.CardRenderer(LOCALE_COLUMNS[locale]) for locale in locales}
        self.catalog = read(self.data/'catalog.json')['cards']
        self.support_ranks = read(self.data/'master/MasterSupportCardRank.json')['_allData']
        self.member_limits = read(self.data/'master/MasterMemberCardLevelLimit.json')['_allData']
        self.texts = {row['_id']: row for row in read(self.data/'master/MasterText.json')['_allData']}
        self.ui = self.data/'native/game-ui'
        def short(value):
            value = str(value or '')
            return 2 <= len(value) <= 16 and '\n' not in value and '{' not in value and sum(c.isalnum() for c in value) >= 2
        self.short_texts = sorted(key for key, row in self.texts.items() if key.startswith('ui_') and all(
            short(row.get(column)) for column in LOCALE_COLUMNS.values()))
        self.sources = [c for c in self.catalog if not heldout(c)]
        self.stage_pools = {}
        self.art_cache = {}
        self.overrides = {}
        original = self.native.picture

        def picture(path):
            image = self.overrides.get(Path(path).name)
            return image if image is not None else original(path)
        self.native.picture = picture

    def text(self, key, locale):
        row = self.texts.get(key, {})
        return str(row.get(LOCALE_COLUMNS[locale]) or row.get('_english') or key)

    @lru_cache(maxsize=64)
    def font(self, locale, size):
        name = 'Pretendard-SemiBold.font' if locale == 'ko' else 'FZLTH_GB18030L2_R.font'
        return ImageFont.truetype(str(self.ui/'fonts'/name), max(4, round(size)))

    @lru_cache(maxsize=64)
    def sprite(self, name):
        return Image.open(self.ui/'sprites'/f'{name}.png').convert('RGBA')

    def art(self, card, variant=None):
        """Gallery artwork as RGB: snap landscape, member square or portrait."""
        if card['kind'] == 'snap':
            name = f"snap-{card['asset_id']}.webp"
        else:
            name = f"member-{card['asset_id']}{'' if variant == 'portrait' else '-square'}.webp"
        if name not in self.art_cache:
            with Image.open(self.data/'native/assets'/name) as image:
                self.art_cache[name] = image.convert('RGB')
        return self.art_cache[name]

    # Card states ---------------------------------------------------------------------------
    def card_state(self, kind, rarity, rank_group, rng):
        rank, awake = rng.randint(1, 5), rng.randint(1, 5)
        if kind == 'snap':
            choices = [r for r in self.support_ranks if r['_group'] == rank_group and r['_rank'] == rank]
            if not choices:
                choices = [r for r in self.support_ranks if r['_group'] == rank_group]
                rank = choices[0]['_rank']
            limit = choices[0]['_limitLevel']
        else:
            limits = [r['_limitLevel'] for r in self.member_limits if r['_rarity'] == rarity and r['_awakeCount'] == awake]
            limit = limits[0] if limits else 50
        level = rng.randint(1, limit)
        power = tuple(rng.randint(500, 18000) if kind == 'member' else rng.randint(10, 1500) for _ in range(3))
        return dict(level=level, rank=rank, awake=awake, power=power)

    # Foreign artwork -----------------------------------------------------------------------
    def foreign_art(self, kind, rng):
        """Out-of-gallery artwork built only from training-visible identities."""
        method = rng.choices(FOREIGN_METHODS, weights=(.32, .26, .2, .22))[0]
        size = (384, 384) if kind == 'member' else (512, 288)
        aspect = size[0]/size[1]
        other = 'snap' if kind == 'member' else 'member'
        sources = []
        if method == 'cross_kind':
            card = rng.choice([c for c in self.sources if c['kind'] == other])
            source = self.art(card, 'portrait' if other == 'member' and rng.random() < .5 else None)
            image = _crop_aspect(source, aspect, rng, rng.uniform(1., 1.3)).resize(size, Image.Resampling.LANCZOS)
            sources.append(card)
        elif method == 'flip_recolor':
            card = rng.choice([c for c in self.sources if c['kind'] == kind])
            image = _crop_aspect(ImageOps.mirror(self.art(card)), aspect, rng, rng.uniform(1.2, 1.7))
            image = _hue_shift(image.resize(size, Image.Resampling.LANCZOS), rng.uniform(60, 300), rng.uniform(.6, 1.4))
            if rng.random() < .2:
                image = Image.merge('RGB', [image.getchannel(i) for i in rng.sample(range(3), 3)])
            sources.append(card)
        elif method == 'mix':
            cards = rng.sample(self.sources, 2)
            a, b = [_crop_aspect(self.art(c), aspect, rng, rng.uniform(1., 1.5)).resize(size, Image.Resampling.LANCZOS)
                    for c in cards]
            style = rng.choice(['split', 'blend', 'strips'])
            if style == 'blend':
                image = Image.blend(a, b, rng.uniform(.35, .65))
            else:
                yy, xx = np.mgrid[0:size[1], 0:size[0]].astype(np.float32)
                if style == 'split':
                    angle = rng.uniform(0, math.pi)
                    t = (xx-size[0]/2)*math.cos(angle)+(yy-size[1]/2)*math.sin(angle)
                    mask = np.clip(t/rng.uniform(4, 40)+.5, 0, 1)
                else:
                    mask = (np.floor(xx/size[0]*rng.randint(2, 4)) % 2).astype(np.float32)
                image = Image.composite(b, a, Image.fromarray(np.uint8(mask*255)))
            if rng.random() < .5:
                image = _hue_shift(image, rng.uniform(-40, 40), rng.uniform(.8, 1.2))
            sources.extend(cards)
        else:
            image = _procedural_art(size, rng)
        return image.convert('RGBA'), method, [{'kind': c['kind'], 'id': c['id']} for c in sources]

    # Backgrounds ---------------------------------------------------------------------------
    @lru_cache(maxsize=16)
    def stage(self, path):
        with Image.open(path) as image:
            return image.convert('RGB')

    def stage_pool(self, profile):
        if profile not in self.stage_pools:
            self.stage_pools[profile] = scene_backgrounds(self.data, profile)
        return self.stage_pools[profile]

    @staticmethod
    def _haze(image, rng):
        """Washed-out list backdrop: desaturate, lift mid-tones, add a white haze gradient."""
        image = ImageEnhance.Color(image).enhance(rng.uniform(.5, .85))
        gamma = rng.uniform(.55, .85)
        lut = [round(255*(i/255)**gamma) for i in range(256)]*3
        image = image.point(lut)
        top, bottom = rng.uniform(.08, .28), rng.uniform(.18, .48)
        fog = np.linspace(top, bottom, image.height, dtype=np.float32)[:, None, None]
        tint = np.array(rng.choice([(255, 255, 255), (250, 248, 255), (255, 250, 244)]), np.float32)
        return np.asarray(image, np.float32)*(1-fog)+tint*fog, {'gamma': round(gamma, 3), 'haze': [round(top, 3), round(bottom, 3)]}

    def background(self, width, height, rng, nrng, profile):
        stages = self.stage_pool(profile)
        roll = rng.random()
        small = (max(32, width//8), max(18, height//8))
        info = {}
        if stages and roll < .55:
            path = rng.choice(stages)
            crop = _crop_aspect(self.stage(path), width/height, rng, rng.uniform(1., 1.8))
            if rng.random() < .5:
                crop = ImageOps.mirror(crop)
            sigma = rng.uniform(.8, 2.)
            low = crop.resize(small, Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(sigma))
            arr, tone = self._haze(low.resize((width, height), Image.Resampling.BICUBIC), rng)
            info = {'type': 'stage', 'source': path.stem, 'blur': round(sigma*8, 1), **tone}
        elif roll < (.75 if stages else .72):
            snap = rng.random() < .85
            card = rng.choice([c for c in self.sources if c['kind'] == ('snap' if snap else 'member')])
            source = self.art(card)
            crop = _crop_aspect(source, width/height, rng, rng.uniform(1., 2.4) if snap else rng.uniform(1.6, 3.))
            if rng.random() < .5:
                crop = ImageOps.mirror(crop)
            sigma = rng.uniform(1.3, 3.2) if snap else rng.uniform(2.4, 4.)
            low = crop.resize(small, Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(sigma))
            arr, tone = self._haze(low.resize((width, height), Image.Resampling.BICUBIC), rng)
            info = {'type': 'blurred_art', 'source': {'kind': card['kind'], 'id': card['id']}, 'blur': round(sigma*8, 1), **tone}
        elif roll < (.88 if stages else .9):
            noise = nrng.integers(100, 235, (max(1, height//16), max(1, width//16), 3), dtype=np.uint8)
            low = Image.fromarray(noise).resize(small, Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(1.5))
            arr = np.asarray(low.resize((width, height), Image.Resampling.BICUBIC), np.float32)
            info = {'type': 'noise'}
        else:
            yy = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
            a, b = [np.array(colorsys.hsv_to_rgb(rng.random(), rng.uniform(.05, .4), rng.uniform(.75, 1.)), np.float32)*255
                    for _ in range(2)]
            arr = np.broadcast_to(a*(1-yy)+b*yy, (height, width, 3)).copy()
            pattern = self.sprite('Pattern1_tiling')
            tile = pattern.resize((max(8, round(88*height/1080)),)*2)
            layer = Image.new('RGBA', (width, height))
            for y in range(0, height, tile.height):
                for x in range(0, width, tile.width):
                    layer.alpha_composite(tile, (x, y))
            alpha = np.asarray(layer.getchannel('A'), np.float32)[..., None]/255*rng.uniform(.08, .25)
            arr = arr*(1-alpha)+255*alpha
            info = {'type': 'gradient_pattern'}
        if rng.random() < .5:
            grain = rng.uniform(1., 4.)
            arr = arr+nrng.normal(0, grain, (height, width, 1)).astype(np.float32)
            info['grain'] = round(grain, 2)
        return Image.fromarray(np.uint8(np.clip(arr, 0, 255))).convert('RGBA'), info

    # Header and UI ---------------------------------------------------------------------------
    def header(self, size, s, kind, locale, rng, inset):
        """Title capsule plus sort/view buttons, drawn 2x supersampled."""
        width, height = size
        k = s*2
        band = min(height, math.ceil(118*s)+2)
        layer = Image.new('RGBA', (width*2, band*2))
        shadow = Image.new('RGBA', layer.size)
        d, ds = ImageDraw.Draw(layer), ImageDraw.Draw(shadow)

        def p(*values):
            return [v*k for v in values]

        def jitter(color, amount=10):
            return tuple(max(0, min(255, c+rng.randint(-amount, amount))) for c in color)+((255,) if len(color) == 3 else ())
        blue, edge, plum = jitter((72, 86, 148)), jitter((38, 42, 88), 6), jitter((56, 45, 61))
        light = jitter((126, 134, 200))
        left = inset+rng.uniform(-3, 3)
        right = width/s-inset+rng.uniform(-3, 3)
        dy = rng.uniform(-3, 3)
        elements = []

        def add(kind_name, box):
            elements.append({'type': kind_name, 'bbox': [box[0]*s, box[1]*s, (box[2]-box[0])*s, (box[3]-box[1])*s]})

        def icon(name, cx, cy, size_):
            image = self.sprite(name)
            image = ImageOps.contain(image, (max(1, round(size_*k)),)*2, Image.Resampling.LANCZOS)
            _composite(layer, image, round(cx*k-image.width/2), round(cy*k-image.height/2))

        def circle(cx, cy, r, name):
            ds.ellipse(p(cx-r, cy-r+5, cx+r, cy+r+5), fill=(0, 0, 0, 90))
            frame = self.sprite('FrameNormalButton_H80').resize((round(2*r*k),)*2, Image.Resampling.LANCZOS)
            _composite(layer, frame, round((cx-r)*k), round((cy-r)*k))
            icon(name, cx, cy, r*.95)
        title = self.text(TITLE_TEXT[kind], locale)
        font = self.font(locale, 32*k)
        y1, y2 = 21+dy, 97+dy
        x1 = left+10.7
        x2 = max(left+553, left+346+font.getlength(title)/k+60)
        ds.rounded_rectangle(p(x1, y1+5, x2, y2+5), radius=(y2-y1)/2*k, fill=(0, 0, 0, 80))
        d.rounded_rectangle(p(x1, y1, x2, y2), radius=(y2-y1)/2*k, fill=plum, outline=jitter((104, 92, 122)),
                            width=max(1, round(2*k)))
        d.rounded_rectangle(p(x1+1, y1+1, left+156, y2-1), radius=30*k, fill=blue, outline=light,
                            width=max(1, round(1.5*k)), corners=(True, False, False, True))
        d.rounded_rectangle(p(left+177, y1+1, left+317, y2-1), radius=5*k, fill=blue, outline=light, width=max(1, round(1.5*k)))
        icon('IconBack2', left+83, (y1+y2)/2, 46)
        icon('IconHome', left+247, (y1+y2)/2, 50)
        d.text(((left+346)*k, (y1+y2)/2*k), title, font=font, fill=(255, 255, 255, 255), anchor='lm',
               stroke_width=max(0, round(.7*k)), stroke_fill=(255, 255, 255, 255))
        add('title', (x1, y1, x2, y2))
        cy = 59+dy
        if rng.random() < .93:
            circle(right-682, cy, 38.6, 'IconFilter')
            add('button', (right-682-38.6, cy-38.6, right-682+38.6, cy+38.6))
        if rng.random() < .93:
            label = self.text('ui_sort_key_'+rng.choice(SORT_KEYS[kind]), locale)
            box = (right-608, cy-37, right-348, cy+37)
            ds.rounded_rectangle(p(box[0], box[1]+5, box[2], box[3]+5), radius=37*k, fill=(0, 0, 0, 90))
            d.rounded_rectangle(p(*box), radius=37*k, fill=blue, outline=edge, width=max(1, round(3*k)))
            d.rounded_rectangle(p(box[0]+5, box[1]+4, box[2]-5, cy), radius=30*k, fill=(255, 255, 255, 18))
            sort_font = self.font(locale, 28*k)
            while sort_font.getlength(label) > (box[2]-box[0]-30)*k and sort_font.size > 8:
                sort_font = self.font(locale, sort_font.size*.9)
            d.text(((box[0]+box[2])/2*k, cy*k), label, font=sort_font, fill=(255, 255, 255, 255), anchor='mm',
                   stroke_width=max(0, round(.5*k)), stroke_fill=(255, 255, 255, 255))
            add('sort_button', box)
        if rng.random() < .93:
            circle(right-284, cy, 38.6, rng.choice(['IconDesc_Mask', 'IconAsc_Mask']))
            add('button', (right-284-38.6, cy-38.6, right-284+38.6, cy+38.6))
        if rng.random() < .93:
            circle(right-182, cy, 38.6, 'IconSwitch')
            add('button', (right-182-38.6, cy-38.6, right-182+38.6, cy+38.6))
        if rng.random() < .93:
            box = (right-119, 12+dy, right-21, 107+dy)
            ds.rounded_rectangle(p(box[0]+2, box[1]+6, box[2]+2, box[3]+6), radius=6*k, fill=(0, 0, 0, 100))
            d.rounded_rectangle(p(*box), radius=6*k, fill=jitter((66, 78, 150)), outline=edge, width=max(1, round(3*k)))
            icon('IconMenu', (box[0]+box[2])/2, (box[1]+box[3])/2, 52)
            add('button', box)
        shadow = shadow.filter(ImageFilter.GaussianBlur(4*k))
        shadow.alpha_composite(layer)
        out = shadow.convert('RGBa').resize((width, band), Image.Resampling.LANCZOS).convert('RGBA')
        solid = layer.convert('RGBa').resize((width, band), Image.Resampling.LANCZOS).convert('RGBA')
        return out, np.asarray(solid.getchannel('A')) >= 128, elements

    def distractors(self, overlay, size, s, rng, avoid, locale):
        """Small UI items around the grid; recorded as UI elements."""
        width, height = size
        d = ImageDraw.Draw(overlay)
        elements = []
        gx0, gx1 = avoid
        for _ in range(rng.choice([0, 0, 1, 1, 2, 3]) if rng.random() < .35 else 0):
            size_ = rng.uniform(40, 90)*s
            sides = [r for r in [(0, gx0), (gx1, width)] if r[1]-r[0] > 1.6*size_]
            if not sides or height-size_*1.3 <= 130*s:
                break
            a, b = rng.choice(sides)
            x, y = rng.uniform(a+.25*size_, b-1.25*size_), rng.uniform(130*s, height-1.3*size_)
            image = ImageOps.contain(self.sprite(rng.choice(DISTRACTOR_ICONS)), (round(size_),)*2, Image.Resampling.LANCZOS)
            if rng.random() < .5:
                frame = self.sprite('FrameNormalButton_H80').resize((round(size_*1.4),)*2, Image.Resampling.LANCZOS)
                _composite(overlay, frame, round(x-size_*.2), round(y-size_*.2))
                elements.append({'type': 'icon_button', 'bbox': [x-size_*.2, y-size_*.2, size_*1.4, size_*1.4]})
            else:
                elements.append({'type': 'icon', 'bbox': [x, y, image.width, image.height]})
            _composite(overlay, image, round(x), round(y))
        if rng.random() < .06:
            text = self.text(rng.choice(self.short_texts), locale)
            font = self.font(locale, 30*s)
            w, h = font.getlength(text)+80*s, 76*s
            x, y = width/2-w/2, rng.choice([height*.82, height*.5, 140*s])
            d.rounded_rectangle((x, y, x+w, y+h), radius=h/2, fill=(24, 22, 40, rng.randint(170, 225)))
            d.text((width/2, y+h/2), text, font=font, fill=(255, 255, 255, 255), anchor='mm')
            elements.append({'type': 'toast', 'bbox': [x, y, w, h]})
        if rng.random() < .08:
            font = self.font('en', 22*s)
            clock = f'{rng.randint(0, 23)}:{rng.randint(0, 59):02d}'
            d.text((28*s, 6*s), clock, font=font, fill=(255, 255, 255, 235))
            bx = width-90*s
            d.rounded_rectangle((bx, 9*s, bx+46*s, 29*s), radius=5*s, outline=(255, 255, 255, 235), width=max(1, round(2*s)))
            d.rectangle((bx+4*s, 13*s, bx+4*s+38*s*rng.uniform(.2, 1), 25*s), fill=(255, 255, 255, 235))
            elements.append({'type': 'status_bar', 'bbox': [0, 0, width, 34*s]})
        if rng.random() < .1:
            r = 44*s
            x, y = width-rng.uniform(70, 160)*s, height-rng.uniform(70, 160)*s
            frame = self.sprite('FrameNormalButton_H80').resize((round(2*r),)*2, Image.Resampling.LANCZOS)
            _composite(overlay, frame, round(x-r), round(y-r))
            icon = ImageOps.contain(self.sprite(rng.choice(DISTRACTOR_ICONS)), (round(r),)*2, Image.Resampling.LANCZOS)
            _composite(overlay, icon, round(x-icon.width/2), round(y-icon.height/2))
            elements.append({'type': 'floating_button', 'bbox': [x-r, y-r, 2*r, 2*r]})
        return elements

    # Scene ---------------------------------------------------------------------------------
    def scene(self, index, config):
        profile, seed = config['profile'], config['seed']
        rng = random.Random(f'{GENERATOR}:{profile}:{seed}:{index}')
        nrng = np.random.default_rng([seed, index, 2])
        self.overrides.clear()
        kind = 'member' if index % 2 == 0 else 'snap'
        locale = rng.choice(config['locales'])
        modes = [m for m in config['modes'] if kind == 'member' or m not in MEMBER_ONLY_MODES]
        mode = rng.choice(modes)
        renderer = self.renderers[locale]
        width, height = rng.choice(PROFILES[profile]['resolutions'])
        ui_scale = rng.uniform(*PROFILES[profile]['ui_scale'])
        geometry = list_layout(kind, width, height, ui_scale)
        s = geometry['scale']
        tw, th = GRID[kind]['tile']
        cols, spacing = GRID[kind]['columns'], GRID[kind]['spacing']
        pitch_x = (tw+spacing+rng.uniform(-.8, .8))*s
        pitch_y = (th+spacing+rng.uniform(-.8, .8))*s
        x0 = geometry['origin'][0]+rng.uniform(-10, 10)*s
        top = (LIST_PADDING_TOP+rng.uniform(-6, 6))*s
        inset = rng.uniform(0, 90) if width/height > 2.05 and rng.random() < .7 else 0.
        scene, background = self.background(width, height, rng, nrng, profile)
        negative = rng.random() < config['negative_rate']
        pool = [c for c in self.catalog if c['kind'] == kind and (profile != 'train' or not heldout(c))]
        rng.shuffle(pool)
        rows_visible = max(1, math.ceil((height-top)/pitch_y))
        capacity = cols*rows_visible
        roll = rng.random()
        if roll < .45:
            amount = rng.randint(min(capacity, len(pool)), len(pool))
        elif roll < .85:
            amount = rng.randint(1, min(capacity, len(pool)))
        else:
            amount = rng.randint(1, min(6, len(pool)))
        foreign_rate = rng.uniform(.1, .32) if rng.random() < config['foreign_screen_fraction'] else 0.
        box = []
        real = iter(pool)
        while len(box) < amount and not negative:
            if rng.random() < foreign_rate:
                box.append(None)
            else:
                card = next(real, None)
                if card is None:
                    break
                box.append(card)
        rows = math.ceil(len(box)/cols)
        content = LIST_PADDING_TOP+rows*th+max(0, rows-1)*spacing+LIST_PADDING_BOTTOM
        max_scroll = max(0., content-height/s)
        scroll = 0. if max_scroll <= 0 or rng.random() < .4 else rng.uniform(0, max_scroll)
        resample = rng.choice([Image.Resampling.BILINEAR, Image.Resampling.BICUBIC, Image.Resampling.LANCZOS])
        render_scale = 1. if s <= 1.15 else min(2., round(s*4)/4)
        selection = rng.random()
        event = rng.random() < .1
        badges = rng.random() < .1
        type_bonus = rng.random() < .08
        placed = []
        cw, ch = round(tw*s), round(th*s)
        box_index = index//4
        for slot, card in enumerate(box):
            x = round(x0+(slot % cols)*pitch_x)
            y = round(top-scroll*s+(slot//cols)*pitch_y)
            if y+ch+40*s < 0 or y-40*s > height or x+cw < 0 or x > width:
                continue
            if card is None:
                art, method, sources = self.foreign_art(kind, rng)
                asset = 900000+slot
                for name in ([f'member-{asset}.webp', f'member-{asset}-square.webp'] if kind == 'member' else [f'snap-{asset}.webp']):
                    self.overrides[name] = art
                rarity = rng.choice(RARITIES[kind])
                card_type = rng.randint(1, 5)
                template = rng.choice([c for c in self.catalog if c['kind'] == kind])
                state = self.card_state(kind, rarity, template['rank_group'], rng)
                identity = {'foreign_method': method, 'foreign_sources': sources}
            else:
                asset, rarity, card_type = card['asset_id'], card['rarity'], card['card_type']
                # Four neighbouring scenes share one player box: states stay consistent.
                state_rng = random.Random(f'state:{profile}:{seed}:{box_index}:{kind}:{card["id"]}')
                state = self.card_state(kind, rarity, card['rank_group'], state_rng)
                identity = {'id': card['id']}
            placed.append(dict(identity=identity, x=x, y=y, rarity=rarity, card_type=card_type, state=state, asset=asset,
                               event_bonus=rng.choice([5, 10, 15, 20, 30, 50]) if event and rng.random() < .5 else 0,
                               badge=badges and rng.random() < .15, type_bonus=type_bonus and rng.random() < .4))
        on_screen = [i for i, p in enumerate(placed) if 0 <= p['y'] and p['y']+ch <= height]
        selected = set()
        if selection < .07 and placed:
            selected = {rng.choice(on_screen or list(range(len(placed))))}
        elif selection < .13:
            selected = {i for i in range(len(placed)) if rng.random() < .35}
        for i, p in enumerate(placed):
            st = p['state']
            state = self.native.CardState(asset_id=p['asset'], rarity=p['rarity'], card_type=p['card_type'],
                                          level=st['level'], rank=st['rank'], awake_count=st['awake'], param=mode,
                                          power=st['power'], selected=i in selected, event_bonus=p['event_bonus'],
                                          badge=p['badge'], type_bonus=p['type_bonus'])
            tile = renderer.render(state, kind, scale=render_scale)
            tile = tile.resize((round((tw+64)*s), round((th+64)*s)), resample)
            _composite(scene, tile, p['x']-round(32*s), p['y']-round(32*s))
            p['selected'] = i in selected
        overlay = Image.new('RGBA', (width, height))
        ui_elements = []
        scrollable = max_scroll > 0 and (kind == 'member' or rng.random() < .3)
        if scrollable and rng.random() < .9:
            draw = ImageDraw.Draw(overlay)
            right = width/s-inset-50
            y_top, y_bottom = 222*s, (height/s-142)*s
            if y_bottom > y_top+20*s:
                draw.rounded_rectangle(((right-6)*s, y_top, (right-2)*s, y_bottom), radius=2*s, fill=(255, 255, 255, 70))
                length = max(.08, min(1., (height/s)/content))*(y_bottom-y_top)
                start = y_top+(scroll/max_scroll)*(y_bottom-y_top-length)
                draw.rounded_rectangle(((right-6)*s, start, (right-2)*s, start+length), radius=2*s, fill=(255, 255, 255, 235))
                ui_elements.append({'type': 'scrollbar', 'bbox': [(right-6)*s, y_top, 4*s, y_bottom-y_top]})
        grid_left = x0-40*s
        grid_right = x0+(cols*tw+(cols-1)*spacing)*s+40*s
        ui_elements += self.distractors(overlay, (width, height), s, rng, (grid_left, grid_right), locale)
        if negative and rng.random() < .55:
            draw = ImageDraw.Draw(overlay)
            font = self.font(locale, 44*s)
            text = self.text('ui_not_exist', locale)
            draw.text((width/2+2*s, height/2+3*s), text, font=font, fill=(40, 36, 70, 140), anchor='mm')
            draw.text((width/2, height/2), text, font=font, fill=(255, 255, 255, 255), anchor='mm')
            box_ = draw.textbbox((width/2, height/2), text, font=font, anchor='mm')
            ui_elements.append({'type': 'empty_message', 'bbox': [box_[0], box_[1], box_[2]-box_[0], box_[3]-box_[1]]})
        occluder = np.asarray(overlay.getchannel('A')) >= 128
        scene.alpha_composite(overlay)
        header = rng.random() < .93
        if header:
            layer, solid, elements = self.header((width, height), s, kind, locale, rng, inset)
            scene.alpha_composite(layer, (0, 0))
            occluder = occluder.copy()
            occluder[:solid.shape[0]] |= solid
            ui_elements += elements
        if rng.random() < .04:
            veil = Image.new('RGBA', (width, height), rng.choice([(0, 0, 0), (255, 255, 255)])+(rng.randint(30, 70),))
            scene.alpha_composite(veil)
            background['veil'] = True
        cards, ignored, foreign, ignored_foreign = [], [], [], []
        rank_cx, rank_cy = RANK_CENTER[kind]
        for p in placed:
            x, y = p['x'], p['y']
            inside = max(0, min(height, y+ch)-max(0, y))*max(0, min(width, x+cw)-max(0, x))
            covered = _box_fraction(occluder, (x, y, cw, ch), width, height)
            visible = max(0., 1.-covered)
            occluded = max(0., covered-(1-inside/(cw*ch)))
            st = p['state']
            field_box = (x-6*s, y+ch-56*s, 144*s, 56*s)
            field_visible = (y+ch <= height and y+ch-56*s >= 0 and x-6*s >= 0 and x+138*s <= width
                             and _box_fraction(occluder, field_box, width, height) < .02)
            rank_box = (x+(rank_cx-32)*s, y+(rank_cy-32*95/102)*s, 64*s, 64*95/102*s)
            rank_visible = (x+(rank_cx-32)*s >= 2 and x+(rank_cx+32)*s <= width-2 and y+(rank_cy-32*95/102)*s >= 2
                            and y+(rank_cy+32*95/102)*s <= height-2 and _box_fraction(occluder, rank_box, width, height) < .02)
            entry = {'kind': kind, **({'id': p['identity']['id']} if 'id' in p['identity'] else {}),
                     'level': st['level'] if field_visible and mode == 'level' else None,
                     'awake_count': st['awake'] if field_visible and mode == 'training' else None,
                     'card_rank': st['rank'] if rank_visible and mode != 'hide' else None,
                     'bbox': [x, y, cw, ch], 'display_mode': mode, 'field_visible': field_visible,
                     'visible_fraction': visible,
                     'latent_state': {'level': st['level'], 'card_rank': st['rank'], 'awake_count': st['awake']},
                     'rarity': p['rarity'], 'card_type': p['card_type'], 'rank_visible': rank_visible,
                     'selected': p['selected'], 'event_bonus': p['event_bonus'], 'badge': p['badge'],
                     'occluded_fraction': round(occluded, 4)}
            if 'id' not in p['identity']:
                entry.update(p['identity'])
            if visible >= .5:
                (cards if 'id' in entry else foreign).append(entry)
            elif visible > 0:
                small = {'kind': kind, **({'id': entry['id']} if 'id' in entry else {}),
                         'reason': 'less_than_half_visible', 'bbox': [x, y, cw, ch], 'visible_fraction': visible}
                if 'id' in entry:
                    ignored.append(small)
                else:
                    small.update(p['identity'])
                    ignored_foreign.append(small)
        scene, acquisition = acquire(scene, rng, profile)
        sx, sy = acquisition['scale_x'], acquisition['scale_y']

        def scale_box(box):
            return [v*(sx if i % 2 == 0 else sy) for i, v in enumerate(box)]
        for entry in cards+ignored+foreign+ignored_foreign+ui_elements:
            entry['bbox'] = scale_box(entry['bbox'])
        output = Path(config['output'])
        name = f'{index:06d}.' + ('webp' if config['image_format'] == 'webp' else 'png')
        buffer = io.BytesIO()
        if config['image_format'] == 'webp':
            scene.save(buffer, format='WEBP', lossless=True, quality=config['webp_effort'], method=config['webp_method'])
        else:
            scene.save(buffer, format='PNG', compress_level=config['png_level'])
        body = buffer.getvalue()
        temporary = output/(name+f'.{os.getpid()}.tmp')
        temporary.write_bytes(body)
        os.replace(temporary, output/name)
        self.overrides.clear()
        return {'file': name, 'cards': cards, 'ignored_cards': ignored,
                'quality': min(stage['quality'] for stage in acquisition['compression']),
                'sha256': hashlib.sha256(body).hexdigest(), 'box_id': f'{profile}-{box_index}', 'locale': locale,
                'acquisition': acquisition, 'kind': kind, 'display_mode': mode, 'negative': negative,
                'screen_resolution': [width, height],
                'layout': {'canvas_scale': geometry['base_scale'], 'ui_scale': ui_scale, 'scale': s, 'columns': cols,
                           'origin': [x0*sx, (top-scroll*s)*sy], 'pitch': [pitch_x*sx, pitch_y*sy],
                           'tile': [cw*sx, ch*sy], 'scroll': scroll, 'max_scroll': max_scroll,
                           'box_size': len(box), 'safe_inset': inset, 'header': header},
                'background': background, 'ui_elements': ui_elements,
                'foreign_cards': foreign, 'ignored_foreign_cards': ignored_foreign}


# Parallel dataset writer -------------------------------------------------------------------
_CONTEXT = None


def _initialize(data, locales):
    global _CONTEXT
    os.environ['OMP_NUM_THREADS'] = '1'
    _CONTEXT = SceneContext(data, locales)


def _part_valid(output, part, start, end, check_files=True):
    try:
        document = read(part)
    except (OSError, ValueError):
        return None
    records = document.get('records', [])
    if document.get('start') != start or len(records) != end-start:
        return None
    for record in records if check_files else []:
        path = output/record['file']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            return None
    return records


def _chunk(job):
    start, end, config = job
    output = Path(config['output'])
    part = output/'.parts'/f'{start:07d}.json'
    if part.exists() and _part_valid(output, part, start, end) is not None:
        return start, end-start, True
    records = [_CONTEXT.scene(i, config) for i in range(start, end)]
    write(part, {'start': start, 'end': end, 'records': records})
    return start, end-start, False


def generate(data, output, count, seed, profile, modes=MODES, locales=tuple(LOCALE_COLUMNS), workers=1, chunk=16,
             image_format='webp', negative_rate=.05, foreign_screen_fraction=.6, webp_effort=60, webp_method=4,
             png_level=6):
    data, output = Path(data), Path(output)
    if profile not in PROFILES:
        raise ValueError(f'Unknown profile: {profile}')
    output.mkdir(parents=True, exist_ok=True)
    (output/'.parts').mkdir(exist_ok=True)
    native_module(data)  # applies the renderer patch layer once, before workers start
    config = dict(output=str(output), seed=seed, profile=profile, modes=list(modes), locales=list(locales),
                  image_format=image_format, negative_rate=negative_rate,
                  foreign_screen_fraction=foreign_screen_fraction, webp_effort=webp_effort,
                  webp_method=webp_method, png_level=png_level)
    jobs = [(start, min(count, start+chunk), config) for start in range(0, count, chunk)]
    started, done = time.time(), 0

    def progress(final=False):
        write(output/'progress.json', {'generated': done, 'total': count, 'elapsed_s': round(time.time()-started, 1),
                                       'complete': final})
    if workers <= 1:
        _initialize(data, locales)
        results = map(_chunk, jobs)
        for _, amount, _ in results:
            done += amount
            progress()
    else:
        method = 'fork' if 'fork' in multiprocessing.get_all_start_methods() else 'spawn'
        with multiprocessing.get_context(method).Pool(workers, _initialize, (str(data), tuple(locales))) as pool:
            for _, amount, _ in pool.imap_unordered(_chunk, jobs):
                done += amount
                progress()
                print(json.dumps({'generated': done, 'total': count}), flush=True)
    records = []
    for start, end, _ in jobs:
        part_records = _part_valid(output, output/'.parts'/f'{start:07d}.json', start, end, check_files=False)
        if part_records is None:
            raise RuntimeError(f'Chunk {start}-{end} is missing or inconsistent')
        records.extend(part_records)
    font_manifest = data/'native/card-fonts/manifest.json'
    write(output/'truth.json', {
        'schema': 'bdon-synthetic/2', 'generator': GENERATOR, 'seed': seed, 'profile': profile,
        'acquisition_version': VERSION, 'composition': 'list-layout-v2', 'screenshots': records, 'complete': True,
        'count': len(records), 'display_modes': list(modes), 'locales': list(locales), 'image_format': image_format,
        'identity_pool': 'heldout-excluded' if profile == 'train' else 'all',
        'background_pool': {'scene_backgrounds': len(scene_backgrounds(data, profile)),
                            'rule': 'training profile skips scene backgrounds whose number % 5 == 0'},
        'heldout_rule': 'id % 7 == 0', 'negative_rate': negative_rate,
        'foreign_screen_fraction': foreign_screen_fraction, 'schema_extension': SCHEMA_EXTENSION,
        'catalog_sha256': hashlib.sha256((data/'catalog.json').read_bytes()).hexdigest(),
        'renderer_sha256': hashlib.sha256((data/'native/render_game_components.py').read_bytes()).hexdigest(),
        'font_manifest_sha256': hashlib.sha256(font_manifest.read_bytes()).hexdigest() if font_manifest.exists() else None,
        'scope': 'native card UI on generated list screens; header and backgrounds approximate; SDF rasterization approximate'})
    shutil.rmtree(output/'.parts', ignore_errors=True)
    progress(True)
    print(json.dumps({'complete': True, 'screenshots': len(records), 'elapsed_s': round(time.time()-started, 1)}), flush=True)


def verify(dataset, threads=16):
    """Check completeness, per-image SHA-256 and count of a written dataset."""
    dataset = Path(dataset)
    document = read(dataset/'truth.json')
    rows = document['screenshots']

    def check(row):
        path = dataset/row['file']
        return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
    with ThreadPoolExecutor(threads) as pool:
        results = list(pool.map(check, rows))
    files = {row['file'] for row in rows}
    stray = sorted(p.name for p in dataset.iterdir() if p.is_file() and p.suffix in ('.png', '.webp') and p.name not in files)
    report = {'dataset': str(dataset), 'complete': bool(document.get('complete')), 'screenshots': len(rows),
              'declared_count': document.get('count'), 'hash_ok': sum(results), 'hash_failed': len(results)-sum(results),
              'duplicate_files': len(rows)-len(files), 'stray_images': len(stray),
              'ok': bool(document.get('complete')) and all(results) and len(rows) == len(files) and not stray}
    return report


def statistics(dataset):
    """Counts by kind, rarity, display mode and language; foreign and held-out tallies."""
    dataset = Path(dataset)
    document = read(dataset/'truth.json')
    out = defaultdict(Counter)
    heldout_ids = defaultdict(Counter)
    for row in document['screenshots']:
        out['screens']['total'] += 1
        out['screens_by_kind'][row.get('kind')] += 1
        out['screens_by_locale'][row['locale']] += 1
        out['screens_by_mode'][row.get('display_mode')] += 1
        out['screen_resolution']['x'.join(map(str, row.get('screen_resolution', [])))] += 1
        out['screens']['negative'] += int(bool(row.get('negative')))
        out['screens']['with_foreign'] += int(bool(row.get('foreign_cards')))
        out['screens']['with_header'] += int(bool(row.get('layout', {}).get('header')))
        out['background'][row.get('background', {}).get('type')] += 1
        for element in row.get('ui_elements', []):
            out['ui_elements'][element['type']] += 1
        for card in row['cards']:
            key = card['kind']
            out['cards_by_kind'][key] += 1
            out['cards_by_rarity'][f"{key}:{card.get('rarity')}"] += 1
            out['cards_by_mode'][f"{key}:{card['display_mode']}"] += 1
            out['cards_by_locale'][f"{key}:{row['locale']}"] += 1
            out['visible_fields'][f"{key}:level"] += card['level'] is not None
            out['visible_fields'][f"{key}:awake_count"] += card['awake_count'] is not None
            out['visible_fields'][f"{key}:card_rank"] += card['card_rank'] is not None
            if card['id'] % 7 == 0:
                out['heldout_cards'][key] += 1
                heldout_ids[key][card['id']] += 1
        for card in row.get('foreign_cards', []):
            out['foreign_by_kind'][card['kind']] += 1
            out['foreign_by_method'][card['foreign_method']] += 1
            out['foreign_by_rarity'][f"{card['kind']}:{card['rarity']}"] += 1
        out['ignored']['cards'] += len(row.get('ignored_cards', []))
        out['ignored']['foreign'] += len(row.get('ignored_foreign_cards', []))
    ids = defaultdict(set)
    for row in document['screenshots']:
        for card in row['cards']:
            ids[card['kind']].add(card['id'])
    out['identities'] = Counter({k: len(v) for k, v in ids.items()})
    total = sum(out['cards_by_kind'].values())+sum(out['foreign_by_kind'].values())
    files = [dataset/row['file'] for row in document['screenshots']]
    report = {k: dict(sorted(v.items(), key=lambda kv: str(kv[0]))) for k, v in out.items()}
    report['heldout_ids'] = {k: dict(sorted(v.items())) for k, v in heldout_ids.items()}
    report['foreign_fraction'] = sum(out['foreign_by_kind'].values())/max(1, total)
    report['image_bytes'] = sum(p.stat().st_size for p in files if p.exists())
    report['profile'] = document['profile']
    report['complete'] = document['complete']
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('generate')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--profile', choices=list(PROFILES), required=True)
    p.add_argument('--modes', default=','.join(MODES))
    p.add_argument('--locales', default=','.join(LOCALE_COLUMNS))
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--chunk', type=int, default=16)
    p.add_argument('--format', choices=['webp', 'png'], default='webp')
    p.add_argument('--negative-rate', type=float, default=.05)
    p.add_argument('--foreign-screen-fraction', type=float, default=.6)
    p = sub.add_parser('verify')
    p.add_argument('dataset', type=Path, nargs='+')
    p = sub.add_parser('stats')
    p.add_argument('dataset', type=Path)
    p.add_argument('--output', type=Path)
    a = parser.parse_args()
    if a.action == 'generate':
        modes = tuple(a.modes.split(','))
        locales = tuple(a.locales.split(','))
        if not set(modes) <= set(MODES) or not any(m not in MEMBER_ONLY_MODES for m in modes):
            parser.error('Unsupported display modes')
        if not set(locales) <= set(LOCALE_COLUMNS):
            parser.error('Unsupported locale')
        generate(a.data, a.output, a.count, a.seed, a.profile, modes, locales, a.workers, a.chunk, a.format,
                 a.negative_rate, a.foreign_screen_fraction)
    elif a.action == 'verify':
        reports = [verify(d) for d in a.dataset]
        for report in reports:
            print(json.dumps(report), flush=True)
        if not all(r['ok'] for r in reports):
            raise SystemExit(1)
    else:
        report = statistics(a.dataset)
        if a.output:
            write(a.output, report)
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
