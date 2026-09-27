"""Remove holdout cards from the gallery and measure false known-card claims."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from .assets import read,write
from .engine import Engine
from .synthetic import bbox_iou


def evaluate(data,dataset,output):
    catalog=read(Path(data)/'catalog.json')['cards']
    excluded={(c['kind'],c['id']) for c in catalog if c['id']%7==0}
    engine=Engine(data,2,excluded_identities=excluded)
    truth=read(Path(dataset)/'truth.json');unknown_total=0;false_known=[];background_claims=[];known_total=known_correct=0
    for n,row in enumerate(truth['screenshots']):
        image=cv2.imread(str(Path(dataset)/row['file']));found=engine.identify(image);found=engine.fill_grid(image,found)
        used=set()
        for card in row['cards']:
            candidates=[(i,p) for i,p in enumerate(found) if bbox_iou(card['bbox'],p['bbox'])>=.5]
            identity=(card['kind'],card['id'])
            if identity in excluded:unknown_total+=1
            else:known_total+=1
            if not candidates:continue
            i,p=max(candidates,key=lambda v:bbox_iou(card['bbox'],v[1]['bbox']));used.add(i)
            actual=(p['kind'],p['id'])
            if identity in excluded:false_known.append({'file':row['file'],'unknown':identity,'predicted':actual,'score':p['identity_confidence']})
            elif actual==identity:known_correct+=1
        for i,p in enumerate(found):
            if i not in used:background_claims.append({'file':row['file'],'predicted':[p['kind'],p['id']]})
        if (n+1)%32==0:print(json.dumps({'evaluated':n+1,'unknown_total':unknown_total,'false_known':len(false_known)}),flush=True)
    blank_results=[]
    rng=np.random.default_rng(8901)
    for value in [0,128,255]:
        image=np.full((720,1280,3),value,np.uint8);blank_results.append(len(engine.identify(image)))
    for _ in range(5):
        image=rng.integers(0,256,(40,70,3),dtype=np.uint8);image=cv2.resize(image,(1280,720),interpolation=cv2.INTER_CUBIC)
        blank_results.append(len(engine.identify(image)))
    report={'excluded_ids':sorted(excluded),'unknown_visible_instances':unknown_total,
        'false_known_claims':len(false_known),'false_known_examples':false_known,
        'known_visible_instances':known_total,'known_correct':known_correct,
        'unmatched_predictions':background_claims,'blank_scene_predictions':blank_results,
        'scope':'same-artwork holdout removed from gallery; identity+IoU matching; no false-field claims inferred from this test'}
    write(Path(output)/'open-set.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();evaluate(a.data,a.dataset,a.output)
