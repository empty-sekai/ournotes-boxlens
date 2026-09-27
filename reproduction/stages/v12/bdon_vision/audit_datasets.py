"""Verify synthetic split isolation, ID holdout, and masterdata progression limits."""
import argparse
import hashlib
from collections import Counter
from pathlib import Path

from .assets import read,write


def audit(data,datasets,output):
    data=Path(data)
    catalog={(c['kind'],c['id']):c for c in read(data/'catalog.json')['cards']}
    ml=read(data/'master/MasterMemberCardLevelLimit.json')['_allData']
    sr=read(data/'master/MasterSupportCardRank.json')['_allData']
    mr=read(data/'master/MasterMemberCardRank.json')['_allData']
    seen={};reports=[];errors=[]
    for path in map(Path,datasets):
        document=read(path/'truth.json');profile=document['profile']
        if not document.get('complete'):errors.append({'dataset':str(path),'error':'incomplete'})
        counts=Counter();locales=Counter();identities=set()
        for row in document['screenshots']:
            counts['screenshots']+=1;locales[row.get('locale','zh-Hans')]+=1
            image=path/row['file'];digest=None
            if image.exists():
                digest=hashlib.sha256(image.read_bytes()).hexdigest()
                if row.get('sha256') and digest!=row['sha256']:
                    errors.append({'dataset':str(path),'file':row['file'],'error':'image_hash_mismatch'})
                if not row.get('sha256'):counts['legacy_hash_computed_from_image']+=1
            else:errors.append({'dataset':str(path),'file':row['file'],'error':'image_unavailable_for_hash_audit'})
            if digest and digest in seen:
                errors.append({'dataset':str(path),'file':row['file'],'error':'duplicate_image','previous':seen[digest]})
            elif digest:seen[digest]=[str(path),row['file']]
            for card in row['cards']:
                key=(card['kind'],card['id']);c=catalog[key];counts['cards']+=1;identities.add(key)
                if profile=='train' and c['id']%7==0:errors.append({'dataset':str(path),'error':'heldout_identity_in_training','card':key})
                state=card.get('latent_state')
                if not state:errors.append({'dataset':str(path),'error':'missing_latent_state','card':key});continue
                rank=state['card_rank'];level=state['level'];awake=state['awake_count']
                if key[0]=='member':
                    limits=[r['_limitLevel'] for r in ml if r['_rarity']==c['rarity'] and r['_awakeCount']==awake]
                    ranks=[r['_rank'] for r in mr if r['_group']==c['rank_group']]
                else:
                    limits=[r['_limitLevel'] for r in sr if r['_group']==c['rank_group'] and r['_rank']==rank]
                    ranks=[r['_rank'] for r in sr if r['_group']==c['rank_group']]
                if not limits or not 1<=level<=max(limits) or rank not in ranks:
                    errors.append({'dataset':str(path),'file':row['file'],'error':'illegal_progression','card':key,'state':state})
                mode=card.get('display_mode','level');counts['mode:'+mode]+=1
                if card.get('level') is not None and (mode!='level' or card['level']!=level):
                    errors.append({'dataset':str(path),'file':row['file'],'error':'bad_visible_level','card':key})
                if card.get('awake_count') is not None and (mode!='training' or card['awake_count']!=awake or key[0]!='member'):
                    errors.append({'dataset':str(path),'file':row['file'],'error':'bad_visible_training','card':key})
                if mode=='hide' and card.get('card_rank') is not None:
                    errors.append({'dataset':str(path),'file':row['file'],'error':'rank_in_hidden_mode','card':key})
        reports.append({'dataset':str(path),'profile':profile,'seed':document['seed'],
            'manifest_sha256':hashlib.sha256((path/'truth.json').read_bytes()).hexdigest(),
            'counts':dict(counts),'locales':dict(locales),'identities':len(identities)})
    report={'schema':'bdon-data-audit/1','passed':not errors,'datasets':reports,'errors':errors,
        'scope':'Manifest hashes, exact screenshot duplicates, training ID exclusions, progression and visible labels. Does not certify pixel parity or exact randomized power values.'}
    write(output,report);print({'passed':report['passed'],'datasets':len(reports),'errors':len(errors)})
    if errors:raise RuntimeError('Dataset audit failed; inspect the saved report')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('datasets',nargs='+',type=Path);a=p.parse_args();audit(a.data,a.datasets,a.output)
