"""Separate field-reader errors from missing or inaccurate localization, using development data."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path

import cv2

from .assets import read,write
from .ocr import NumberReader,FIELD_CONFIDENCE
from .synthetic import bbox_iou


def encoded(values):
    if values['awake_count']['value'] is not None:return 100+values['awake_count']['value']
    return values['level']['value']


def count_result(counts,prefix,truth,value):
    if truth==0:
        counts[prefix+'_hidden_total']+=1;counts[prefix+'_false_visible']+=int(value is not None)
    else:
        counts[prefix+'_visible_total']+=1
        counts[prefix+'_correct']+=int(value==truth)
        counts[prefix+'_unknown']+=int(value is None)
        counts[prefix+'_wrong']+=int(value is not None and value!=truth)


def diagnose(data,dataset,observations,output):
    reader=NumberReader(data,2);truth=read(dataset/'truth.json')
    scans={r['source']:r for r in read(observations)};totals=Counter();groups=defaultdict(Counter);examples=[]
    for row in truth['screenshots']:
        image=cv2.imread(str(dataset/row['file']));items=[];targets=[];actual=[]
        for card in row['cards']:
            mode=card.get('display_mode','level')
            if mode=='level':target=card.get('level')
            elif mode=='training':target=100+card['awake_count'] if card.get('awake_count') is not None else None
            else:target=0 if card.get('field_visible') else None
            if target is None:continue
            x,y,w,h=card['bbox'];member=card['kind']=='member';sx=w/(224 if member else 326);sy=h/(294 if member else 184)
            items.append({'kind':card['kind'],'bbox':[x+6*sx,y+6*sy,w-12*sx,h-12*sy]});targets.append((card,target))
            candidates=[c for c in scans[row['file']]['cards'] if c['kind']==card['kind'] and c['id']==card['id'] and bbox_iou(c['bbox'],card['bbox'])>=.5]
            actual.append(max(candidates,key=lambda c:bbox_iou(c['bbox'],card['bbox'])) if candidates else None)
        oracle=reader.read_fields(image,items)
        predicted=iter(reader.read_fields(image,[c for c in actual if c is not None]))
        for (card,target),ideal,found in zip(targets,oracle,actual):
            value=encoded(next(predicted)) if found else None;oracle_value=encoded(ideal)
            keys=['overall','kind:'+card['kind'],'locale:'+row.get('locale','zh-Hans'),
                'card_width_bin:'+str(int(card['bbox'][2]//30)*30)]
            for key in keys:
                counts=totals if key=='overall' else groups[key]
                count_result(counts,'oracle',target,oracle_value)
                if found:
                    count_result(counts,'detected_pose',target,value)
                    counts['oracle_correct_but_pose_lost']+=int(target>0 and oracle_value==target and value!=target)
                else:
                    counts['localization_missing']+=1
                    counts['oracle_correct_on_missing_card']+=int(target>0 and oracle_value==target)
            if target>0 and oracle_value==target and value!=target:
                examples.append({'file':row['file'],'kind':card['kind'],'id':card['id'],'truth':target,
                    'oracle':oracle_value,'detected_pose':value,'bbox':found['bbox'] if found else None})
    report={'schema':'ournotes-boxlens.field-diagnosis/1','threshold':FIELD_CONFIDENCE,'counts':dict(totals),
        'strata':{k:dict(v) for k,v in groups.items()},'localization_loss_examples':examples,
        'scope':'Development screenshots only. Oracle artwork rectangle derived from native tile truth; detected poses come from saved matching identity+IoU results. Conditional pose counts exclude cards not localized.'}
    write(output,report);print(report['counts'])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['data','dataset','observations','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();diagnose(a.data,a.dataset,a.observations,a.output)
