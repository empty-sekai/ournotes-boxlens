"""Turn raw evaluation evidence into explicit public metrics without hiding abstentions."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from bdon_vision.assets import read,write


def ratio(n,d):return n/d if d else None


def summarize_counts(counts):
    expected=counts.get('expected_cards',0);identified=counts.get('identified',0);extra=counts.get('extra_predictions',0)
    fields={}
    for field in ['level','card_rank','awake_count']:
        total=counts.get(field+'_total',0);correct=counts.get(field+'_correct',0);wrong=counts.get(field+'_wrong',0);unknown=counts.get(field+'_unknown',0)
        fields[field]={'visible_truth':total,'correct':correct,'wrong':wrong,'unknown_after_localization':unknown,
            'not_localized':total-correct-wrong-unknown,'end_to_end_accuracy':ratio(correct,total),
            'accepted_precision':ratio(correct,correct+wrong),'coverage':ratio(correct+wrong,total)}
    return {'identity':{'expected':expected,'correct':identified,'extra':extra,'missed':expected-identified,
            'recall':ratio(identified,expected),'precision_on_scored_regions':ratio(identified,identified+extra)},
        'fields':fields,'hidden_field_predictions':counts.get('hidden_field_predictions',0)}


def publish(evaluations,output):
    output.mkdir(parents=True,exist_ok=True);runs={};lines=['# End-to-end evaluation metrics','',
        'Counts include abstentions and missed cards. Accepted precision is conditional on producing a value; it is not end-to-end accuracy. Ignored heavily cropped regions follow the raw evaluator policy.','',
        '| Run | Screenshots | ID correct / truth | Extras | Hidden field claims |','|---|---:|---:|---:|---:|']
    for name,path in evaluations:
        raw=read(path);summary=summarize_counts(raw['strata']['overall'])
        hidden=Counter(e['field'] for row in raw['details'] for e in row['errors'] if e.get('error')=='field_not_visible')
        boxes=raw.get('merged_boxes',[]);merged={key:Counter() for key in ['level','card_rank','awake_count']}
        for box in boxes:
            for key in merged:merged[key].update(box.get('fields',{}).get(key,{}))
        entry={'screenshots':raw['screenshots'],**summary,'hidden_claims_by_field':dict(hidden),
            'latency_ms':raw['latency_ms'],'cpu_affinity':raw.get('cpu_affinity'),'platform':raw.get('platform'),
            'opencv_threads':raw.get('opencv_threads'),'matching':raw.get('matching'),
            'raw_report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'strata':{k:summarize_counts(v) for k,v in raw['strata'].items()},
            'merged_boxes':{'count':len(boxes),'duplicate_invariant':all(b['duplicate_upload_invariant'] for b in boxes),
                'fields':{k:dict(v) for k,v in merged.items()}}}
        runs[name]=entry;ident=summary['identity']
        lines.append(f"| {name} | {raw['screenshots']} | {ident['correct']} / {ident['expected']} | {ident['extra']} | {entry['hidden_field_predictions']} |")
        write(output/'raw'/f'{name}.json',raw)
    for name,entry in runs.items():
        lines+=['',f'## {name}','', '| Field | Correct | Wrong | Unknown after localization | Not localized | Visible truth |', '|---|---:|---:|---:|---:|---:|']
        for key,f in entry['fields'].items():lines.append(f"| {key} | {f['correct']} | {f['wrong']} | {f['unknown_after_localization']} | {f['not_localized']} | {f['visible_truth']} |")
        lines+=['',f"Scan median {entry['latency_ms']['median']:.2f} ms; P95 {entry['latency_ms']['p95']:.2f} ms. CPU affinity: {entry['cpu_affinity']}."]
    report={'schema':'ournotes-boxlens.metrics/1','runs':runs,
        'limits':['Synthetic distribution metrics do not estimate arbitrary deployment accuracy.','Cards and views within a player box are correlated.','Confidence scores are not calibrated probabilities.','Latency excludes file decoding and HTTP; other host workload may be present.']}
    write(output/'metrics.json',report);(output/'METRICS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({'runs':list(runs),'output':str(output)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evaluation',action='append',required=True,metavar='NAME=PATH')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=[]
    for value in a.evaluation:
        name,sep,path=value.partition('=')
        if not sep or not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):p.error('Use a simple NAME=PATH')
        rows.append((name,Path(path)))
    if len({n for n,_ in rows})!=len(rows):p.error('Duplicate run names')
    publish(rows,a.output)
