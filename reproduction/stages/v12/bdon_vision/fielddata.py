"""Native text-layer composites for multilingual level/cultivation/mode training."""
import argparse
import io
import random
import hashlib
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from PIL import Image,ImageFilter,ImageEnhance

from .assets import read,write,native_module

LOCALES={'ja':'_japanese','en':'_english','zh-Hant':'_traditionalChinese','zh-Hans':'_simplifiedChinese','ko':'_korean'}


def generate(data,output,count=106000,seed=81101,split='train',acquisition='legacy'):
    data,output=Path(data),Path(output);output.mkdir(parents=True,exist_ok=True)
    if not (data/'native/card-fonts/manifest.json').exists():raise ValueError('Actual localized font assets required')
    native=native_module(data);renderers={loc:native.CardRenderer(col) for loc,col in LOCALES.items()}
    catalog=read(data/'catalog.json')['cards'];rng=random.Random(seed)
    names=list(LOCALES);pools={kind:[c for c in catalog if c['kind']==kind] for kind in ('member','snap')}
    backgrounds={}
    for card in catalog:
        kind=card['kind'];height=294 if kind=='member' else 184
        tile=renderers['zh-Hans'].render(native.CardState(asset_id=card['asset_id'],rarity=card['rarity'],
            card_type=card['card_type'],param='hide',rank=None),kind,scale=1)
        backgrounds[(kind,card['id'])]=tile.crop((26,32+height-56,170,32+height))

    @lru_cache(maxsize=4096)
    def layer(locale,kind,mode,value):
        height=294 if kind=='member' else 184
        power=(value,value+17,value+31) if kind=='member' else (value,value+7,value+11)
        tile=renderers[locale].render(native.CardState(asset_id=1,rarity=2,card_type=1,
            param=mode,level=value if mode=='level' else 1,awake_count=value if mode=='training' else 1,
            power=power),kind,scale=1,only='text')
        return tile.crop((26,32+height-56,170,32+height))

    images=open_memmap(output/'fields.npy',mode='w+',dtype='uint8',shape=(count,56,144,3))
    labels=open_memmap(output/'field-labels.npy',mode='w+',dtype='int32',shape=(count,))
    locales=open_memmap(output/'locales.npy',mode='w+',dtype='uint8',shape=(count,))
    values=Counter();mode_counts=Counter();locale_counts=Counter()
    for j in range(count):
        locale=rng.choice(names);kind=rng.choice(['member','snap'])
        if j%5==0:
            modes=['hide','performance','technic','visual']+(['total'] if kind=='member' else [])
            mode=rng.choice(modes);value=(rng.randrange(64)*293+137) if kind=='member' else (rng.randrange(64)*17+37);label=0
        elif j%5==1:
            kind='member';mode='training';value=(j//5)%5+1;label=100+value
        else:mode='level';value=rng.randint(1,100);label=value
        card=rng.choice(pools[kind]);patch=Image.new('RGBA',(144,56),tuple(rng.randrange(100,230) for _ in range(3))+(255,))
        patch.alpha_composite(backgrounds[(kind,card['id'])])
        text=layer(locale,kind,mode,value)
        # Registration errors from card localization, not arbitrary font swapping.
        patch.alpha_composite(text,(rng.randint(-3,3),rng.randint(-3,3)))
        if label==0 and j%11==0:
            # Partial glyphs are not license to infer hidden values.
            cut=rng.randrange(35,110);patch.paste(backgrounds[(kind,card['id'])].crop((cut,0,144,56)),(cut,0))
        if acquisition=='screen':
            # Compression happens on the received low-resolution pixels BEFORE
            # the recognizer enlarges the crop. Legacy preprocessing compressed
            # an already enlarged patch, which understates hard screenshot loss.
            ratios=[.26,.33,.41,.49,.61,.77,.93,1.09] if split=='train' else [.29,.37,.45,.55,.69,.85,1.01]
            qualities=[34,46,58,73,87,95] if split=='train' else [38,50,64,79,91]
            small=rng.choice(ratios)
            patch=ImageEnhance.Brightness(patch.convert('RGB')).enhance(rng.uniform(.88,1.12))
            patch=patch.resize((round(144*small),round(56*small)),rng.choice([Image.Resampling.BILINEAR,Image.Resampling.BICUBIC,Image.Resampling.LANCZOS]))
            patch=patch.filter(ImageFilter.GaussianBlur(rng.uniform(.05,.95) if split=='train' else rng.uniform(.12,1.10)))
            width,height=patch.size;px=rng.randrange(16);py=rng.randrange(16)
            # Random codec-block alignment approximates cropping from a larger
            # screenshot instead of always aligning the text crop to block zero.
            patch=Image.fromarray(np.pad(np.asarray(patch),((py,16-py),(px,16-px),(0,0)),mode='edge'))
            for _ in range(rng.choice([1,2,3,4])):
                buf=io.BytesIO();patch.save(buf,format=rng.choice(['JPEG','WEBP']),quality=rng.choice(qualities))
                patch=Image.open(io.BytesIO(buf.getvalue())).convert('RGB')
            patch=patch.crop((px,py,px+width,py+height)).resize((144,56),Image.Resampling.BILINEAR)
        else:
            small=rng.choice([.40,.52,.65,.80,1.,1.18] if split=='train' else [.34,.46,.59,.73,.91,1.08])
            patch=patch.resize((round(144*small),round(56*small)),Image.Resampling.BILINEAR).resize((144,56),Image.Resampling.BILINEAR)
            if rng.random()<.35:patch=patch.filter(ImageFilter.GaussianBlur(rng.uniform(.1,.5 if split=='train' else .75)))
            patch=ImageEnhance.Brightness(patch.convert('RGB')).enhance(rng.uniform(.88,1.12))
            for _ in range(rng.choice([1,2] if split=='train' else [2,3])):
                buf=io.BytesIO();patch.save(buf,format=rng.choice(['JPEG','WEBP']),
                    quality=rng.choice([48,62,76,89,97] if split=='train' else [40,56,70,83,93]))
                patch=Image.open(io.BytesIO(buf.getvalue())).convert('RGB')
        images[j]=np.asarray(patch);labels[j]=label;locales[j]=names.index(locale)
        values[str(label)]+=1;mode_counts[mode]+=1;locale_counts[locale]+=1
        if (j+1)%5000==0:print({'generated':j+1,'total':count,'split':split},flush=True)
    for a in (images,labels,locales):a.flush()
    metadata={'schema':'bdon-native-field-corpus/1','field_count':count,'field_counts':dict(values),
        'modes':dict(mode_counts),'locales':dict(locale_counts),'locale_names':names,'split':split,
        'seed':seed,'acquisition':acquisition,'complete':True,'label_encoding':{'0':'no level/cultivation value','1..100':'level','101..105':'awake_count1..5'},
        'font_manifest_sha256':hashlib.sha256((data/'native/card-fonts/manifest.json').read_bytes()).hexdigest(),
        'provenance':'native prefab text geometry; actual localized glyphs, material and master text; raster AA approximate'}
    write(output/'manifest.json',metadata);print(metadata,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=106000);p.add_argument('--seed',type=int,default=81101)
    p.add_argument('--split',choices=['train','validation'],default='train')
    p.add_argument('--acquisition',choices=['legacy','screen'],default='legacy')
    a=p.parse_args();generate(a.data,a.output,a.count,a.seed,a.split,a.acquisition)
