"""Execute the recorded baseline and screen-acquisition training recipe in a new directory."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]


def recipe(initializers=False,extended=True):
    steps=[]
    def add(name,stage,module,*args):steps.append({'name':name,'source_stage':stage,'module':module,'args':list(map(str,args))})
    if not initializers:
        add('encoder-base','v4','train','encoder','--data','data/v6','--output','models/v4','--steps',1200,'--batch',128)
        add('level-base','v6','train','level','--data','data/v6','--output','models/v6','--steps',12000,'--batch',512,'--samples',202000,'--valid-samples',30300)
    for split,count,seed in [('train',106000,81101),('validation',21200,92107)]:
        add('native-fields-'+split,'v7','fielddata','--data','data/v7','--output','corpora/v7/fields-'+split,'--count',count,'--seed',seed,'--split',split)
    if not initializers:
        add('native-fields-train-model','v7','train_fields','--train','corpora/v7/fields-train','--validation','corpora/v7/fields-validation','--initial','models/v6/level.pt','--output','models/v7-fields','--steps',8000,'--batch',512)
    for split,count,seed in [('validation',512,60017),('stress',256,70019),('train',2048,50021)]:
        add('level-scenes-'+split,'v6','synthetic','generate','--data','data/v6','--output','synthetic/scale-v6/'+split,'--count',count,'--seed',seed,'--profile',split)
    for split,count,seed in [('validation',140,92711),('stress',70,93713),('train',350,91703)]:
        add('mode-scenes-'+split,'v7','synthetic','generate','--data','data/v7','--output','synthetic/modes-v7/'+split,'--count',count,'--seed',seed,'--profile',split,
            '--modes','level,training,total,performance,technic,visual,hide','--locales','ja,en,zh-Hant,zh-Hans,ko')
    for split in ['train','validation','stress']:
        add('scene-corpus-'+split,'v8','corpus','--data','data/v8','--datasets','synthetic/scale-v6/'+split,'synthetic/modes-v7/'+split,'--output','corpora/v8/scenes-'+split)
    common=['--train','corpora/v8/scenes-train','--validation','corpora/v8/scenes-validation']
    add('scene-encoder','v8','train_scene_encoder','--data','data/v8',*common,'--initial','models/v4/encoder.pt','--output','models/v8-encoder','--steps',2400,'--batch',128)
    add('scene-fields','v8','train_fields',*common,'--initial','models/v7-fields/fields.pt','--replay','corpora/v7/fields-train','--output','models/v8-fields','--steps',2400,'--batch',512)
    if extended:
        add('unchanged-fields-control','v11','train_fields',*common,'--stress','corpora/v8/scenes-stress','--initial','models/v8-fields/fields.pt','--replay','corpora/v7/fields-train','--output','models/v11-fields-continued','--steps',2400,'--batch',512,'--eval-every',300)
        add('unchanged-encoder-control','v11','train_scene_encoder','--data','data/v8',*common,'--stress','corpora/v8/scenes-stress','--initial','models/v8-encoder/encoder.pt','--output','models/v11-encoder-continued','--steps',2400,'--batch',128,'--eval-every',300)
        for split,count,seed in [('train',106000,120019),('validation',21200,120031)]:
            add('screen-fields-'+split,'v12','fielddata','--data','data/v8','--output','corpora/v12/fields-'+split,'--count',count,'--seed',seed,'--split',split,'--acquisition','screen')
        add('screen-fields-model','v12','train_fields','--train','corpora/v12/fields-train','--validation','corpora/v8/scenes-validation','--stress','corpora/v8/scenes-stress','--initial','models/v8-fields/fields.pt','--replay','corpora/v8/scenes-train','--output','models/v12-fields','--steps',8000,'--batch',512,'--eval-every',500)
    return steps


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--initializers',type=Path,help='Published initializers directory: skips three pretraining model stages')
    parser.add_argument('--baseline-only',action='store_true');parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--smoke',action='store_true',help='Tiny end-to-end integration run; outputs are NOT usable trained models')
    parser.add_argument('--allow-resource-change',action='store_true',help='Explicitly run a new-resource experiment, not an exact historical-input reproduction')
    a=parser.parse_args();plan=recipe(bool(a.initializers),not a.baseline_only)
    if a.smoke:
        for step in plan:
            replacements={'--steps':'2','--batch':'16','--samples':'202','--valid-samples':'101','--eval-every':'1'}
            if '--count' in step['args']:replacements['--count']='106' if step['module']=='fielddata' else '8'
            for flag,value in replacements.items():
                if flag in step['args']:step['args'][step['args'].index(flag)+1]=value
    manifest=json.loads((ROOT/'reproduction/source-hashes.json').read_text(encoding='utf8'))
    for name,digest in manifest.items():
        if hashlib.sha256((ROOT/'reproduction'/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Changed frozen training source: {name}')
    expected=json.loads((ROOT/'reproduction/metadata/resource-hashes.json').read_text(encoding='utf8'))['data']
    mismatches=[name for name,digest in expected.items() if not (a.data/name).exists() or hashlib.sha256((a.data/name).read_bytes()).hexdigest()!=digest]
    if mismatches and not a.allow_resource_change:raise ValueError(f'Resource snapshot mismatch ({len(mismatches)} files); use the recorded snapshot or explicitly allow a new-resource experiment')
    initializer_map=[('encoder-v4.pt','models/v4/encoder.pt'),('level-v6.pt','models/v6/level.pt'),('fields-v7.pt','models/v7-fields/fields.pt')]
    if a.initializers:
        hashes=json.loads((ROOT/'reproduction/metadata/checkpoint-hashes.json').read_text(encoding='utf8'))
        for source,dest in initializer_map:
            if hashlib.sha256((a.initializers/source).read_bytes()).hexdigest()!=hashes[dest]:raise ValueError(f'Initializer hash mismatch: {source}')
    for step in plan:
        if not (ROOT/'reproduction/stages'/step['source_stage']/'bdon_vision'/(step['module']+'.py')).exists():raise FileNotFoundError(step)
    if a.dry_run:
        print(json.dumps({'commands':plan,'master_hash_mismatches':mismatches},indent=2));return
    run=a.output.resolve()
    if run.exists() and any(run.iterdir()):raise ValueError('Output must be empty; previous experiments are never overwritten')
    run.mkdir(parents=True,exist_ok=True);(run/'logs').mkdir()
    for stage in ['v6','v7','v8']:
        dest=run/'data'/stage;dest.mkdir(parents=True)
        for name in ['master','native','canonical']:shutil.copytree(a.data/name,dest/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for name in ['catalog.json','index.npz','provenance.json']:shutil.copy2(a.data/name,dest/name)
        shutil.copy2(ROOT/'reproduction/renderers'/(stage+'.py'),dest/'native/render_game_components.py')
    if a.initializers:
        for source,dest in initializer_map:
            target=run/dest;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a.initializers/source,target)
    evidence={'schema':'ournotes-boxlens.training-run/1','mode':'smoke_not_for_deployment' if a.smoke else 'full',
        'plan':plan,'resource_changes':mismatches,
        'source_manifest_sha256':hashlib.sha256((ROOT/'reproduction/source-hashes.json').read_bytes()).hexdigest(),
        'input_provenance_sha256':hashlib.sha256((a.data/'provenance.json').read_bytes()).hexdigest(),
        'steps':[],'complete':False}
    for step in plan:
        env=os.environ.copy();env.update(PYTHONPATH=str(ROOT/'reproduction/stages'/step['source_stage']),PYTHONIOENCODING='utf-8',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
        start=time.time();command=[sys.executable,'-u','-m','bdon_vision.'+step['module'],*step['args']]
        print('Running '+step['name'],flush=True)
        with (run/'logs'/(step['name']+'.log')).open('wb') as log:
            result=subprocess.run(command,cwd=run,env=env,stdout=log,stderr=subprocess.STDOUT)
        evidence['steps'].append({'name':step['name'],'exit_code':result.returncode,'elapsed_s':time.time()-start})
        (run/'run.json').write_text(json.dumps(evidence,indent=2),encoding='utf8')
        if result.returncode:raise RuntimeError(f"Stage failed: {step['name']}; completed artifacts retained")
    evidence['complete']=True;(run/'run.json').write_text(json.dumps(evidence,indent=2),encoding='utf8')
    print('Training recipe complete. Select checkpoints using development metrics; final test remains separate.',flush=True)


if __name__=='__main__':main()
