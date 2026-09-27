import hashlib
import importlib.util
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def native_module(data):
    source = Path(data) / 'native/render_game_components.py'
    # Apply screenshot-calibrated aspect-fill only to our packaged copy.
    # The original reverse-engineering artifact is deliberately not modified.
    code = source.read_text(encoding='utf-8')
    old = "if variant in ('snap','formation_snap'):\n                    clip="
    new = "if variant in ('member','snap','formation_snap'):\n                    clip="
    if old in code:
        source.write_text(code.replace(old, new), encoding='utf-8')
    spec = importlib.util.spec_from_file_location('bdon_native_renderer', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from .font import NativeFont
    module.GameText = lambda: NativeFont(data)
    return module


def prepare(data):
    data = Path(data)
    records = []
    for kind, table in [('member', 'MasterMemberCard'), ('snap', 'MasterSupportCard')]:
        for card in read(data / 'master' / (table + '.json'))['_allData']:
            asset = card['_assetID']
            records.append(dict(kind=kind, id=card['_id'], asset_id=asset,
                rarity=card['_rarity'], card_type=card['_cardType'],
                name=card['_nameTextID'].replace('Character_Name_', '').replace('Snap_Name_', ''),
                level_group=card.get('_memberCardLevelGroup', card.get('_supportCardLevelGroup')),
                rank_group=card.get('_memberCardRankGroup', card.get('_supportCardRankGroup')),
                file=f'native/assets/{kind}-{asset}.webp'))

    def fetch(row):
        f = data / row['file']
        kind, asset = row['kind'], row['asset_id']
        directory, stem = ('MemberCard', 'member_thumbnail') if kind == 'member' else ('SupportCard', 'snap_thumbnail')
        url = f'https://assets.bdon.moe/zh-Hans/{directory}/{asset}/{stem}/{stem}.webp'
        if not f.exists():
            request = urllib.request.Request(url, headers={'Referer': 'https://bdon.moe/', 'User-Agent': 'BDON-box-vision/0.1'})
            with urllib.request.urlopen(request, timeout=45) as response:
                body = response.read()
            f.write_bytes(body)
        with Image.open(f) as im:
            im.verify()
        row['sha256'] = hashlib.sha256(f.read_bytes()).hexdigest()
        row['source'] = url
        if kind == 'member':
            square = data / f'native/assets/member-{asset}-square.webp'
            square_url = f'https://assets.bdon.moe/zh-Hans/MemberCard/{asset}/member_thumbnail/square.webp'
            if not square.exists():
                req = urllib.request.Request(square_url, headers={'Referer':'https://bdon.moe/', 'User-Agent':'BDON-box-vision/0.1'})
                with urllib.request.urlopen(req, timeout=45) as response:
                    square.write_bytes(response.read())
            with Image.open(square) as im:
                im.verify()
            canonical = data / f'canonical/member-{asset}.png'
            canonical.parent.mkdir(exist_ok=True)
            # The captured list fills a portrait mask with the square sprite.
            # Preserve aspect and crop its sides; stretching shifts field crops.
            # This runtime calibration differs from the serialized fitter mode.
            with Image.open(square) as im:
                ImageOps.fit(im.convert('RGB'),(212,282),Image.Resampling.LANCZOS).save(canonical)
            row['match_file'] = canonical.relative_to(data).as_posix()
        else:
            row['match_file'] = row['file']
        return row

    failures, completed = [], []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [(row, pool.submit(fetch, row)) for row in records]
        for row, future in futures:
            try:
                completed.append(future.result())
            except Exception as e:
                failures.append({'kind': row['kind'], 'id': row['id'], 'error': type(e).__name__ + ': ' + str(e)})
    write(data / 'catalog.json', {'schema': 'bdon-box-catalog/1', 'cards': completed, 'unavailable': failures})
    print(json.dumps({'catalog': len(completed), 'unavailable': failures}), flush=True)
    if not completed:
        raise RuntimeError('No artwork available')
    build_index(data)


def build_index(data):
    data = Path(data)
    cv2.setNumThreads(2)
    sift = cv2.SIFT_create(nfeatures=260, contrastThreshold=.018, edgeThreshold=12)
    descriptors, points, owners = [], [], []
    cards = read(data / 'catalog.json')['cards']
    for i, card in enumerate(cards):
        im = cv2.imread(str(data / card.get('match_file', card['file'])))
        # Match at list-thumbnail resolution; original coordinates are retained.
        h, w = im.shape[:2]
        scales = [min(1., 220 / w)]
        for scale in scales:
            gray = cv2.cvtColor(cv2.resize(im, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
            mask = np.full(gray.shape, 255, np.uint8)
            mh, mw = mask.shape
            mask[int(mh*.82):, :int(mw*.4)] = 0
            mask[int(mh*.72):, int(mw*.76):] = 0
            mask[:int(mh*.18), :int(mw*.15)] = 0
            kp, desc = sift.detectAndCompute(gray, mask)
            if desc is None:
                continue
            # RootSIFT improves matching under screenshot compression.
            desc = np.sqrt(desc / (desc.sum(axis=1, keepdims=True) + 1e-8))
            descriptors.append(desc)
            points.extend([(k.pt[0] / scale, k.pt[1] / scale) for k in kp])
            owners.extend([i] * len(kp))
        card['width'], card['height'] = w, h
    np.savez_compressed(data / 'index.npz', descriptors=np.vstack(descriptors),
                        points=np.array(points, np.float32), owners=np.array(owners, np.int32))
    write(data / 'catalog.json', {'schema': 'bdon-box-catalog/1', 'cards': cards,
                               'unavailable': read(data / 'catalog.json').get('unavailable', [])})
    print(json.dumps({'index_descriptors': len(points)}), flush=True)
