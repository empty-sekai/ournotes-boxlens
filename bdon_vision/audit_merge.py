"""Audit complete player boxes from cached predictions and independently labeled views."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path

from .assets import read,write
from .engine import merge

FIELDS=('level','card_rank','awake_count')


def audit_box(rows,scans):
    expected=defaultdict(lambda:{key:set() for key in FIELDS})
    ignored=set()
    for row in rows:
        ignored.update((c['kind'],c['id']) for c in row.get('ignored_cards',[]))
        for card in row['cards']:
            target=expected[(card['kind'],card['id'])]
            for key in FIELDS:
                if card.get(key) is not None:target[key].add(card[key])
    box=merge(scans);repeat=merge(scans+scans)
    actual={(c['kind'],c['id']):c for c in box['cards']}
    counts={key:Counter() for key in FIELDS};errors=[]
    for identity,values in expected.items():
        found=actual.get(identity)
        for key,truth in values.items():
            metrics=counts[key]
            value=found[key]['value'] if found else None
            conflicts={int(v) for v in found.get('conflicts',{}).get(key,{})} if found else set()
            if len(truth)==0:
                metrics['not_observed']+=1
                if value is not None or conflicts:
                    metrics['fabricated_field']+=1
                    errors.append({'card':identity,'field':key,'error':'unobserved_field_claim','value':value,'conflicts':sorted(conflicts)})
            elif len(truth)==1:
                metrics['observable']+=1
                expected_value=next(iter(truth))
                if value==expected_value:metrics['correct']+=1
                elif value is not None:
                    metrics['wrong']+=1
                    errors.append({'card':identity,'field':key,'error':'wrong_value','expected':expected_value,'value':value})
                elif conflicts:
                    metrics['spurious_conflict']+=1
                    errors.append({'card':identity,'field':key,'error':'spurious_conflict','expected':expected_value,'conflicts':sorted(conflicts)})
                else:metrics['unknown']+=1
            else:
                metrics['expected_conflict']+=1
                if value is None and conflicts==truth:metrics['conflict_preserved']+=1
                else:
                    metrics['conflict_lost']+=1
                    errors.append({'card':identity,'field':key,'error':'conflict_lost','expected':sorted(truth),'value':value,'conflicts':sorted(conflicts)})
    return {'expected_unique':len(expected),'identified_unique':len(expected.keys()&actual.keys()),
        'extra_ids':sorted(actual.keys()-expected.keys()-ignored),
        'duplicate_upload_invariant':box['cards']==repeat['cards'],
        'fields':{k:dict(v) for k,v in counts.items()},'errors':errors,
        'conflicting_cards':sum(bool(c['conflicts']) for c in box['cards'])}


def audit(dataset,observations,output):
    truth=read(Path(dataset)/'truth.json');scans=read(observations)
    by_source={s['source']:s for s in scans}
    if len(by_source)!=len(scans):raise ValueError('Repeated source names: need unambiguous scan-to-truth association')
    groups=defaultdict(list)
    for row in truth['screenshots']:
        if row['file'] not in by_source:raise ValueError(f"Missing scan: {row['file']}")
        groups[row.get('box_id',row['file'])].append(row)
    results=[];totals={k:Counter() for k in FIELDS}
    for box_id,rows in groups.items():
        result=audit_box(rows,[by_source[row['file']] for row in rows]);result['box_id']=box_id
        results.append(result)
        for key in FIELDS:totals[key].update(result['fields'][key])
    report={'schema':'bdon-box-audit/1','boxes':len(results),'fields':{k:dict(v) for k,v in totals.items()},
        'all_duplicate_upload_invariant':all(r['duplicate_upload_invariant'] for r in results),
        'details':results,'scope':'Only visible truth fields across supplied views; no latent-state inference'}
    write(output,report)
    print({k:v for k,v in report.items() if k!='details'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--observations',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.dataset,a.observations,a.output)
