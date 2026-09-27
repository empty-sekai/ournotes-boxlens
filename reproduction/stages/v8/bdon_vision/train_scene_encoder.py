"""Metric fine-tuning on complete-scene crops; held-out card IDs remain unseen."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .assets import read,write
from .train import CardEncoder,seed_all,tensor_images,export,distort


def load_corpus(path,cards):
    doc=read(path/'manifest.json')
    if not doc['complete']:raise ValueError('Incomplete corpus')
    if doc['catalog_ids']!=[[c['kind'],c['id']] for c in cards]:raise ValueError('Catalog order differs; rebuild crop bank')
    count=doc['art_count']
    x=torch.from_numpy(np.array(np.load(path/'art.npy',mmap_mode='r')[:count])).permute(0,3,1,2).cuda().float().div_(255)
    labels=np.array(np.load(path/'art-labels.npy',mmap_mode='r')[:count])
    return x,torch.from_numpy(labels).long().cuda(),doc


def train(data,train_dir,valid_dir,output,initial,steps=2400,batch=128):
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    output.mkdir(parents=True,exist_ok=True);seed_all(71933);torch.set_num_threads(8)
    cards=read(data/'catalog.json')['cards'];xs,ys,tm=load_corpus(train_dir,cards)
    if set(tm['profiles'])!={'train'}:raise ValueError('Encoder training must use only train-profile scenes')
    held=np.array([c['id']%7==0 for c in cards]);training=np.flatnonzero(~held)
    if held[ys.cpu().numpy()].any():raise ValueError('Held-out card leaked into training crop bank')
    vx,vy,vm=load_corpus(valid_dir,cards)
    canonical=tensor_images([data/c.get('match_file',c['file']) for c in cards],(128,160)).cuda()
    train_ids=torch.from_numpy(training).long().cuda()
    inverse=torch.full((len(cards),),-1,dtype=torch.long,device='cuda');inverse[train_ids]=torch.arange(len(training),device='cuda')
    model=CardEncoder();model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True));model=model.cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.00012,weight_decay=.0001)
    started=time.perf_counter()
    for step in range(steps):
        # The reference branch only sees training IDs. Evaluation galleries may
        # add held-out/new cards later without updating any model parameter.
        if step%20==0:
            model.eval()
            with torch.no_grad():references=model(canonical[train_ids])
            model.train()
        ix=torch.randint(len(xs),(batch,),device='cuda');targets=inverse[ys[ix]]
        query=xs[ix]
        if step%3==0:query=distort(query,strong=False)
        embeddings=model(query);logits=embeddings@references.T/.08
        loss=F.cross_entropy(logits,targets)+.15*(1-(embeddings*references[targets]).sum(1)).mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if step==0 or (step+1)%100==0:
            status={'step':step+1,'steps':steps,'loss':float(loss.detach()),'elapsed_s':round(time.perf_counter()-started,1),
                    'gpu_allocated_gib':round(torch.cuda.memory_allocated()/2**30,2)}
            write(output/'encoder-progress.json',status);print(json.dumps(status),flush=True)
    model.eval();predictions=[];similarities=[];margins=[]
    with torch.no_grad():
        gallery=model(canonical)
        for start in range(0,len(vx),batch):
            scores=model(vx[start:start+batch])@gallery.T;top=scores.topk(2,dim=1)
            predictions.append(top.indices[:,0].cpu().numpy());similarities.append(top.values[:,0].cpu().numpy())
            margins.append((top.values[:,0]-top.values[:,1]).cpu().numpy())
    pred=np.concatenate(predictions);truth=vy.cpu().numpy();results={}
    for name,mask in [('seen',~held[truth]),('new_card_holdout',held[truth])]:
        results[name]={'correct':int((pred[mask]==truth[mask]).sum()),'total':int(mask.sum())}
    np.savez_compressed(output/'encoder-scene-validation.npz',truth=truth,pred=pred,
        similarity=np.concatenate(similarities),margin=np.concatenate(margins))
    torch.save(model.cpu().state_dict(),output/'encoder.pt');export(model,output/'encoder.onnx',(1,3,160,128))
    report={'method':'train-only native scene crops; disjoint acquisition validation; holdout IDs excluded from both training branches',
        'results':results,'training_steps':steps,'batch':batch,'training_scenes':tm['scenes'],'training_crops':len(xs),
        'validation_scenes':vm['scenes'],'validation_crops':len(vx),'train_fingerprint':tm['source_fingerprint'],
        'validation_fingerprint':vm['source_fingerprint'],'elapsed_s':time.perf_counter()-started}
    write(output/'encoder-evaluation.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--train',type=Path,required=True)
    p.add_argument('--validation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--steps',type=int,default=2400);p.add_argument('--batch',type=int,default=128)
    a=p.parse_args();train(a.data,a.train,a.validation,a.output,a.initial,a.steps,a.batch)
