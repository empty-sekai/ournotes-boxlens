import hashlib
import importlib.util
import json
import sys
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temporary,path)


def native_module(data):
    source = Path(data) / 'native/render_game_components.py'
    # Apply screenshot-calibrated aspect-fill only to our packaged copy.
    # The original reverse-engineering artifact is deliberately not modified.
    original = source.read_text(encoding='utf-8');code=original
    old = "if variant in ('snap','formation_snap'):\n                    clip="
    new = "if variant in ('member','snap','formation_snap'):\n                    clip="
    if old in code:
        code=code.replace(old,new)
    code=code.replace('            if layer:\n                flip=',
                      "            if layer and only != 'text':\n                flip=")
    code=code.replace("preferred=self.text.text(content,text_comp['m_fontSize']*psx,shadow=False)",
        "preferred=self.text.text(content,text_comp['m_fontSize']*psx,shadow=False,font_asset=text_comp['m_fontAsset']['name'],character_spacing=text_comp.get('m_characterSpacing',0))")
    code=code.replace("tile=self.text.text(value,font_size,rgba(text.get('m_fontColor',{})))",
        "tile=self.text.text(value,font_size,rgba(text.get('m_fontColor',{})),font_asset=text['m_fontAsset']['name'],character_spacing=text.get('m_characterSpacing',0))")
    code=code.replace("{'total':'综合力','performance':'Pfm.','technic':'Tec.','visual':'Vis.'}",
                      "{'total':self.total_label,'performance':'Pfm','technic':'Tec','visual':'Vis'}")
    old_line="text_values[ref('_statusValueText')]=str({'total':sum(state.power),'performance':state.power[0],'technic':state.power[1],'visual':state.power[2]}.get(state.param,''))"
    new_line=old_line+"\n            if variant in ('snap','formation_snap') and state.param in ('performance','technic','visual'):\n                text_values[ref('_statusValueText')]=format({'performance':state.power[0],'technic':state.power[1],'visual':state.power[2]}[state.param]/100.,'.2f')+'%'"
    if new_line not in code:code=code.replace(old_line,new_line)
    if code!=original:
        temporary=source.with_name(source.name+f'.{os.getpid()}.tmp')
        temporary.write_text(code,encoding='utf-8');os.replace(temporary,source)
    spec = importlib.util.spec_from_file_location('bdon_native_renderer', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from .font import NativeFont
    module.GameText = lambda: NativeFont(data)
    base=module.CardRenderer
    localized={row['_id']:row for row in read(Path(data)/'master/MasterText.json')['_allData']}
    class CardRenderer(base):
        def __init__(self,locale='_simplifiedChinese'):
            super().__init__()
            self.locale=locale
            self.total_label=localized['ui_profile_info_total_power'][locale]
            mapping={'_japanese':'ja','_english':'en','_traditionalChinese':'zh-Hant','_simplifiedChinese':'zh-Hans','_korean':'ko'}
            self.text.locale=mapping.get(locale,locale)

        @lru_cache(maxsize=32)
        def nodes(self,variant):
            nodes,kind=super().nodes(variant)
            for node in nodes:
                local=next((c for c in node.get('components',[]) if c.get('class')=='LocalizeText'),None)
                text=next((c for c in node.get('components',[]) if c.get('class')=='TextMeshProUGUI'),None)
                if local and text and local.get('_localizeEnabled'):
                    value=localized.get(local.get('_masterTextID'),{}).get(self.locale)
                    if value:text['m_text']=value
            return nodes,kind
    module.CardRenderer=CardRenderer
    return module


def prepare(data):
    data = Path(data)
    records = []
    texts={r['_id']:r for r in read(data/'master/MasterText.json')['_allData']}
    member_limits=read(data/'master/MasterMemberCardLevelLimit.json')['_allData']
    snap_ranks=read(data/'master/MasterSupportCardRank.json')['_allData']
    for kind, table in [('member', 'MasterMemberCard'), ('snap', 'MasterSupportCard')]:
        for card in read(data / 'master' / (table + '.json'))['_allData']:
            asset = card['_assetID']
            limits={str(r['_rank']):r['_limitLevel'] for r in snap_ranks if r['_group']==card.get('_supportCardRankGroup')} if kind=='snap' else {}
            maximum=max(limits.values(),default=0) if kind=='snap' else max((r['_limitLevel'] for r in member_limits if r['_rarity']==card['_rarity']),default=0)
            records.append(dict(kind=kind, id=card['_id'], asset_id=asset,max_level=maximum or None,rank_level_limits=limits,
                rarity=card['_rarity'], card_type=card['_cardType'],
                name=texts.get(card['_nameTextID'],{}).get('_simplifiedChinese') or card['_nameTextID'],
                title=texts.get(card.get('_subtitleTextID',card.get('_descriptionTextID')),{}).get('_simplifiedChinese',''),
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
    if (data/'models/encoder.onnx').exists():
        from .inference import Gallery
        Gallery(data,read(data/'catalog.json')['cards'])
        print(json.dumps({'embedding_gallery':'updated'}),flush=True)


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
