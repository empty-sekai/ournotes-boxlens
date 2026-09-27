"""Train a compact CPU-exportable field reader from native multilingual crops."""
import argparse
import json
import time
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .assets import read,write
from .train import LevelNet,seed_all,export


def load(path):
    meta=read(path/'manifest.json')
    if not meta['complete']:raise ValueError(f'Incomplete corpus: {path}')
    count=meta['field_count']
    images=np.load(path/'fields.npy',mmap_mode='r')[:count]
    labels=np.load(path/'field-labels.npy',mmap_mode='r')[:count]
    # Copy avoids read-only array aliasing; all samples remain resident on GPU.
    x=torch.from_numpy(np.array(images)).permute(0,3,1,2).cuda().float().div_(255)
    y=torch.from_numpy(np.array(labels)).long().cuda()
    return x,y,meta


def validation_metrics(model,x,y,batch):
    was_training=model.training;model.eval();loss=0.;parts=[]
    with torch.inference_mode():
        for start in range(0,len(x),batch):
            logits=model(x[start:start+batch]);loss+=float(F.cross_entropy(logits,y[start:start+batch],reduction='sum'))
            parts.append(logits.softmax(1).cpu().numpy())
    p=np.concatenate(parts);truth=y.cpu().numpy();pred=p.argmax(1);ordered=np.sort(p,axis=1)
    accepted=(p.max(1)>=.995)&(ordered[:,-1]-ordered[:,-2]>=.5)&(pred>0)
    correct=int(((pred==truth)&accepted).sum());wrong=int(((pred!=truth)&(truth>0)&accepted).sum())
    false_visible=int(((truth==0)&accepted).sum())
    model.train(was_training)
    return {'samples':len(truth),'cross_entropy':loss/len(truth),'raw_correct':int((pred==truth).sum()),
        'visible_truth':int((truth>0).sum()),'numeric_correct':correct,'numeric_wrong':wrong,
        'false_visible':false_visible,'accepted_numeric':int(accepted.sum()),
        'numeric_precision':correct/(correct+wrong+false_visible) if correct+wrong+false_visible else None,
        'numeric_coverage':correct/int((truth>0).sum()) if (truth>0).any() else None}


def train(train_dir,valid_dir,output,initial,steps=8000,batch=512,replay=None,eval_every=0,stress=None):
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    output.mkdir(parents=True,exist_ok=True);seed_all(90011);torch.set_num_threads(8)
    xs,ys,tm=load(train_dir);vx,vy,vm=load(valid_dir)
    if stress:sx,sy,sm=load(stress)
    if replay:rx,ry,rm=load(replay)
    model=LevelNet();model.head=nn.Linear(96*3*8,106)
    if initial:
        state=torch.load(initial,map_location='cpu',weights_only=True)
        model.load_state_dict({k:v for k,v in state.items() if not k.startswith('head.')},strict=False)
        with torch.no_grad():
            n=min(model.head.out_features,state['head.weight'].shape[0])
            model.head.weight[:n].copy_(state['head.weight'][:n]);model.head.bias[:n].copy_(state['head.bias'][:n])
    model=model.cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    started=time.perf_counter();model.train()
    curve=[]
    def checkpoint(step,train_loss=None):
        row={'step':step,'train_loss':train_loss,'elapsed_s':time.perf_counter()-started,
            'validation':validation_metrics(model,vx,vy,batch)}
        if stress:row['development_stress']=validation_metrics(model,sx,sy,batch)
        curve.append(row);write(output/'learning-curve.json',{'checkpoints':curve,
            'initial_sha256':hashlib.sha256(initial.read_bytes()).hexdigest() if initial else None,
            'scope':'Development corpus only; additional fine-tuning with fresh AdamW state, not an uninterrupted optimizer resume'})
        torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},output/f'fields-step-{step:06d}.pt')
        print(json.dumps({'validation_checkpoint':row}),flush=True)
    if eval_every:checkpoint(0)
    for step in range(steps):
        ix=torch.randint(len(xs),(batch,),device='cuda')
        x=xs[ix];target=ys[ix]
        if replay:
            count=batch//3;ri=torch.randint(len(rx),(count,),device='cuda')
            x=torch.cat([x[:-count],rx[ri]]);target=torch.cat([target[:-count],ry[ri]])
        x=(x*torch.empty(batch,1,1,1,device='cuda').uniform_(.93,1.07)).clamp(0,1)
        loss=F.cross_entropy(model(x),target)
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if step==0 or (step+1)%200==0:
            status={'step':step+1,'steps':steps,'loss':float(loss.detach()),
                'elapsed_s':round(time.perf_counter()-started,1),'gpu_allocated_gib':round(torch.cuda.memory_allocated()/2**30,2)}
            write(output/'fields-progress.json',status);print(json.dumps(status),flush=True)
        if eval_every and ((step+1)%eval_every==0 or step+1==steps):checkpoint(step+1,float(loss.detach()))
    model.eval();chunks=[]
    with torch.no_grad():
        for start in range(0,len(vx),batch):chunks.append(model(vx[start:start+batch]).softmax(1).cpu().numpy())
    probabilities=np.concatenate(chunks);truth=vy.cpu().numpy();pred=probabilities.argmax(1)
    confidence=probabilities.max(1);order=np.sort(probabilities,axis=1);margin=order[:,-1]-order[:,-2]
    locales=np.load(valid_dir/'locales.npy')[:len(truth)] if (valid_dir/'locales.npy').exists() else np.zeros(len(truth),int)
    reports={}
    for threshold in [.90,.95,.98,.995]:
        accepted=(confidence>=threshold)&(margin>=.5)
        reports[str(threshold)]={'correct':int(((pred==truth)&accepted).sum()),'accepted':int(accepted.sum()),
            'wrong':int(((pred!=truth)&accepted).sum()),
            'false_visible':int(((truth==0)&(pred!=0)&accepted).sum()),'total':len(truth)}
    per_locale={}
    for i,name in enumerate(vm.get('locale_names',['unknown'])):
        mask=locales==i;per_locale[name]={'correct':int((pred[mask]==truth[mask]).sum()),'total':int(mask.sum())}
    np.savez_compressed(output/'fields-validation.npz',truth=truth,pred=pred,confidence=confidence,margin=margin,locales=locales)
    torch.save(model.cpu().state_dict(),output/'fields.pt');export(model,output/'fields.onnx',(1,3,56,144))
    report={'training_samples':len(xs),'validation_samples':len(vx),'steps':steps,'batch':batch,
        'raw_correct':int((pred==truth).sum()),'thresholds':reports,'per_locale':per_locale,
        'train_corpus':tm,'validation_corpus':vm,'replay_corpus':rm if replay else None,'elapsed_s':time.perf_counter()-started}
    write(output/'fields-evaluation.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('train_corpus','validation_corpus')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train',type=Path,required=True);p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--initial',type=Path)
    p.add_argument('--steps',type=int,default=8000);p.add_argument('--batch',type=int,default=512)
    p.add_argument('--replay',type=Path)
    p.add_argument('--eval-every',type=int,default=0);p.add_argument('--stress',type=Path,help='Development stress corpus, never the final test set')
    a=p.parse_args();train(a.train,a.validation,a.output,a.initial,a.steps,a.batch,a.replay,a.eval_every,a.stress)
