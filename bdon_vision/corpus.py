"""Streaming crop banks from complete synthetic scenes with exact metadata labels."""
import argparse
import hashlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from numpy.lib.format import open_memmap

from .assets import read,write


def sample(image,left,top,width,height,size):
    matrix=np.array([[width/size[0],0,left],[0,height/size[1],top]],np.float32)
    return cv2.warpAffine(image,matrix,size,flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)[:,:,::-1]


def build(data,datasets,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    catalog=read(Path(data)/'catalog.json')['cards']
    lookup={(c['kind'],c['id']):i for i,c in enumerate(catalog)}
    manifests=[];fingerprint=hashlib.sha256()
    for path in datasets:
        document=read(Path(path)/'truth.json')
        if not document.get('complete',False):raise ValueError(f'Incomplete dataset: {path}')
        fingerprint.update((Path(path)/'truth.json').read_bytes())
        manifests.append((Path(path),document))
    capacity=sum(len(row['cards']) for _,doc in manifests for row in doc['screenshots'])
    images=open_memmap(output/'art.npy',mode='w+',dtype='uint8',shape=(capacity,160,128,3))
    labels=open_memmap(output/'art-labels.npy',mode='w+',dtype='int32',shape=(capacity,))
    fields=open_memmap(output/'fields.npy',mode='w+',dtype='uint8',shape=(capacity,56,144,3))
    values=open_memmap(output/'field-labels.npy',mode='w+',dtype='int32',shape=(capacity,))
    locale_names=['ja','en','zh-Hant','zh-Hans','ko']
    locales=open_memmap(output/'locales.npy',mode='w+',dtype='uint8',shape=(capacity,))
    arts=digits=scenes=0;counts=Counter();profiles=Counter();skipped=Counter()
    for path,document in manifests:
        for row in document['screenshots']:
            raw=(path/row['file']).read_bytes();digest=hashlib.sha256(raw).hexdigest()
            if row.get('sha256') and row['sha256']!=digest:raise ValueError(f"Image hash mismatch: {row['file']}")
            fingerprint.update(bytes.fromhex(digest))
            im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            if im is None:raise ValueError(f"Cannot decode {path/row['file']}")
            ih,iw=im.shape[:2]
            for card in row['cards']:
                x,y,w,h=card['bbox'];member=card['kind']=='member'
                sx=w/(224 if member else 326);sy=h/(294 if member else 184)
                if min(x,y)>=0 and x+w<=iw and y+h<=ih:
                    images[arts]=sample(im,x+6*sx,y+6*sy,w-12*sx,h-12*sy,(128,160))
                    labels[arts]=lookup[(card['kind'],card['id'])];arts+=1
                else:skipped['cropped_art']+=1
                left=x-6*sx;top=y+h-56*sy
                if min(left,top)<0 or left+144*sx>iw or top+56*sy>ih:
                    skipped['cropped_field']+=1;continue
                mode=card.get('display_mode','level')
                if mode=='level':value=card['level']
                elif mode=='training':value=100+card['awake_count'] if card.get('awake_count') is not None else None
                else:value=0
                if value is None:skipped['unknown_field']+=1;continue
                fields[digits]=sample(im,left,top,144*sx,56*sy,(144,56));values[digits]=value
                locales[digits]=locale_names.index(row.get('locale','zh-Hans'))
                counts[str(value)]+=1;profiles[document['profile']]+=1;digits+=1
            scenes+=1
            if scenes%64==0:print({'scenes':scenes,'art_crops':arts,'field_crops':digits},flush=True)
    for a in (images,labels,fields,values,locales):a.flush()
    metadata={'schema':'bdon-scene-corpus/1','source_fingerprint':fingerprint.hexdigest(),
        'catalog_ids':[[c['kind'],c['id']] for c in catalog],'capacity':capacity,'art_count':arts,
        'field_count':digits,'scenes':scenes,'field_counts':dict(counts),'profiles':dict(profiles),
        'locale_names':locale_names,
        'skipped':dict(skipped),'datasets':[str(p) for p,_ in manifests],'complete':True}
    write(output/'manifest.json',metadata);print(metadata,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--datasets',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.data,a.datasets,a.output)
