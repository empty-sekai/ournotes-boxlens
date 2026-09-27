"""Native-prefab synthetic screenshot generation and end-to-end CPU evaluation."""
import argparse
import io
import json
import random
import time
import os
import platform
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

from .assets import read,write,native_module
from .scenarios import PROFILES,acquire,VERSION


def generate(data,output,count=24,seed=20260927,profile='dev'):
    data,output=Path(data),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    rng=random.Random(seed)
    native=native_module(data)
    renderer=native.CardRenderer()
    catalog=read(data/'catalog.json')['cards']
    support_ranks=read(data/'master/MasterSupportCardRank.json')['_allData']
    member_limits=read(data/'master/MasterMemberCardLevelLimit.json')['_allData']
    records=[]
    for number in range(count):
        if profile!='dev':rng=random.Random(seed+number*100003)
        kind='member' if number%2==0 else 'snap'
        width,height=rng.choice(PROFILES[profile]['resolutions'] if profile!='dev' else [(1440,1080),(1280,720),(1920,1080),(1024,768),(1600,900)])
        bg=np.zeros((max(1,height//16),max(1,width//16),3),np.uint8)
        noise=np.random.default_rng(seed+number).integers(100,235,bg.shape,dtype=np.uint8)
        scene=Image.fromarray(noise).resize((width,height),Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(12)).convert('RGBA')
        cols=6 if kind=='member' else 4
        logical_w,logical_h=(224,294) if kind=='member' else (326,184)
        scale=width*(.116 if kind=='member' else .171)/logical_w
        if profile!='dev':scale*=rng.uniform(*PROFILES[profile]['ui_scale'])
        cw,ch=round(logical_w*scale),round(logical_h*scale)
        left=round(width*rng.uniform(.085,.12));top=round(height*.125)
        pitchx=round(cw*1.11);pitchy=round(ch*1.10)
        if number%5==4:
            top-=round(ch*.28)
        pool=[c for c in catalog if c['kind']==kind]
        if profile=='train':pool=[c for c in pool if c['id']%7!=0]
        rng.shuffle(pool)
        capacity=min(len(pool),cols*(int((height-top)/pitchy)+1))
        amount=capacity if number%3 else rng.randint(1,capacity)
        truth=[]
        ignored=[]
        for idx,c in enumerate(pool[:amount]):
            x=left+(idx%cols)*pitchx;y=top+(idx//cols)*pitchy
            # Four neighboring scenes share the same underlying player box.
            # Reordering and overlap must not fabricate progression conflicts.
            state_rng=rng if profile=='dev' else random.Random(seed*1000003+(number//4)*1009+c['id']*17+(0 if kind=='member' else 1))
            rank=state_rng.randint(1,5);awake=state_rng.randint(1,5)
            if kind=='snap':
                choices=[r for r in support_ranks if r['_group']==c['rank_group'] and r['_rank']==rank]
                if not choices:
                    choices=[r for r in support_ranks if r['_group']==c['rank_group']]
                    if not choices:
                        continue
                    rank=choices[0]['_rank']
                limit=choices[0]['_limitLevel']
            else:
                choices=[r['_limitLevel'] for r in member_limits if r['_rarity']==c['rarity'] and r['_awakeCount']==awake]
                limit=choices[0] if choices else 50
            level=state_rng.randint(1,limit)
            state=native.CardState(asset_id=c['asset_id'],rarity=c['rarity'],card_type=c['card_type'],
                                   level=level,rank=rank,awake_count=awake,selected=number%7==6)
            # Render at original resolution before framebuffer scaling.
            tile=renderer.render(state,kind,scale=1.)
            tile=tile.crop((32,32,32+logical_w,32+logical_h)).resize((cw,ch),Image.Resampling.BILINEAR)
            scene.alpha_composite(tile,(x,y))
            visible=max(0,min(height,y+ch)-max(0,y))*max(0,min(width,x+cw)-max(0,x))/(cw*ch)
            if visible>=.5:
                level_visible=y+ch<=height and y+ch-56*scale>=0 and x-6*scale>=0 and x+138*scale<=width
                rank_visible=y+ch<=height and y+ch-64*scale>=0 and x+cw<=width and x+cw-64*scale>=0
                truth.append({'kind':kind,'id':c['id'],'level':level if level_visible else None,
                              'card_rank':rank if rank_visible else None,'bbox':[x,y,cw,ch],
                              'visible_fraction':visible,'latent_state':{'level':level,'card_rank':rank,'awake_count':awake}})
            elif visible>0:ignored.append({'kind':kind,'id':c['id'],'reason':'less_than_half_visible'})
        quality=rng.choice([65,78,88,95])
        if profile=='dev':
            out=output/f'{number:06d}.jpg';scene.convert('RGB').save(out,quality=quality)
            acquisition={'profile':'dev','quality':quality}
        else:
            scene,acquisition=acquire(scene,rng,profile)
            for card in truth:
                card['bbox']=[v*acquisition['scale_x' if i%2==0 else 'scale_y'] for i,v in enumerate(card['bbox'])]
            out=output/f'{number:06d}.png';scene.save(out)
        records.append({'file':out.name,'cards':truth,'ignored_cards':ignored,'quality':quality,
                        'box_id':f'{profile}-{number//4}','acquisition':acquisition})
        if number%4==0:
            print(json.dumps({'generated':number+1,'total':count}),flush=True)
        if number%32==31 or number==count-1:
            write(output/'truth.json',{'schema':'bdon-synthetic/2','seed':seed,'profile':profile,
                'acquisition_version':VERSION,'screenshots':records,'complete':number==count-1,
                'scope':'native card UI with generated backgrounds; calibrated aspect-fill; SDF rasterization approximate'})


def evaluate(data,dataset,output):
    from .engine import Engine,merge
    affinity=None
    if hasattr(os,'sched_getaffinity'):
        affinity=sorted(os.sched_getaffinity(0))[:2]
        os.sched_setaffinity(0,affinity)
    engine=Engine(data,threads=2)
    truth=read(Path(dataset)/'truth.json')
    total=matched=false_positive=level_correct=rank_correct=level_total=rank_total=0
    timings=[];details=[];scans=[];strata=defaultdict(lambda:defaultdict(int))
    for row in truth['screenshots']:
        result=engine.scan(cv2.imread(str(Path(dataset)/row['file'])),row['file'])
        scans.append(result);timings.append(result['elapsed_ms'])
        expected={(c['kind'],c['id']):c for c in row['cards']}
        actual={(c['kind'],c['id']):c for c in result['cards']}
        total+=len(expected);matched+=len(expected.keys()&actual.keys())
        ignored={(c['kind'],c['id']) for c in row.get('ignored_cards',[])}
        extras=list(actual.keys()-expected.keys()-ignored);false_positive+=len(extras)
        errors=[]
        acquisition=row.get('acquisition',{})
        keys=['overall','profile:'+acquisition.get('profile','legacy'),
            'resolution:'+str(acquisition.get('output_resolution',[result['width'],result['height']])),
            'compression_rounds:'+str(len(acquisition.get('compression',[])))]
        for key in keys:
            strata[key]['screenshots']+=1;strata[key]['extra_predictions']+=len(extras)
        for key,card in expected.items():
            found=actual.get(key)
            card_keys=keys+['kind:'+key[0],'identity_split:'+('heldout' if key[1]%7==0 else 'seen')]
            for group in card_keys:
                strata[group]['expected_cards']+=1;strata[group]['identified']+=int(found is not None)
            if not found:
                errors.append({'card':key,'error':'missing'})
            for name in ('level','card_rank'):
                value=card[name]
                if value is None:
                    continue
                correct=bool(found and found[name]['value']==value)
                for group in card_keys:
                    strata[group][name+'_total']+=1
                    strata[group][name+'_correct']+=int(correct)
                    strata[group][name+'_wrong']+=int(found is not None and found[name]['value'] is not None and not correct)
                    strata[group][name+'_unknown']+=int(found is not None and found[name]['value'] is None)
                if name=='level':level_total+=1;level_correct+=correct
                else:rank_total+=1;rank_correct+=correct
                if not correct:
                    errors.append({'card':key,'field':name,'expected':value,'actual':found[name] if found else None})
        details.append({'file':row['file'],'box_id':row.get('box_id'),'acquisition':acquisition,
                        'elapsed_ms':result['elapsed_ms'],'errors':errors,'extra_ids':extras})
        print(json.dumps({'evaluated':len(details),'total':len(truth['screenshots']),'ms':result['elapsed_ms']}),flush=True)
    report={'screenshots':len(details),'expected_cards':total,'identified':matched,'extra_predictions':false_positive,
            'level':{'correct':int(level_correct),'total':level_total},
            'card_rank':{'correct':int(rank_correct),'total':rank_total},
            'latency_ms':{'median':float(np.median(timings)),'p95':float(np.percentile(timings,95))},
            'opencv_threads':2,'cpu_affinity':affinity,'platform':platform.platform(),
            'strata':dict(strata),'details':details}
    write(Path(output)/'evaluation.json',report)
    write(Path(output)/'observations.json',scans)
    print(json.dumps({k:v for k,v in report.items() if k!='details'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','evaluate'])
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset',type=Path);p.add_argument('--count',type=int,default=24);p.add_argument('--seed',type=int,default=20260927)
    p.add_argument('--profile',choices=['dev',*PROFILES],default='dev')
    a=p.parse_args()
    if a.action=='generate':generate(a.data,a.output,a.count,a.seed,a.profile)
    else:evaluate(a.data,a.dataset,a.output)
