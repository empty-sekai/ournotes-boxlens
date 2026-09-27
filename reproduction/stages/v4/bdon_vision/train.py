"""GPU-trained open-set artwork embedding and compact level OCR.

Card IDs are used only for metric-learning supervision. Production inference
exports a 128-D encoder, not a classifier over the current catalog. Holdout
card identities are never sampled by training, but are added to the gallery.
"""
import argparse
import io
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image,ImageFilter
import torch
from torch import nn
from torch.nn import functional as F

from .assets import read,write


class CardEncoder(nn.Module):
    def __init__(self,pretrained=False):
        super().__init__()
        from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights
        self.features=mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None).features
        self.projection=nn.Linear(576*4,128)
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))

    def forward(self,x):
        x=self.features((x-self.mean)/self.std)
        # Fixed 160x128 input produces 5x4; equivalent to adaptive 2x2 pooling
        # but exportable by ONNX's standard AveragePool operator.
        x=F.avg_pool2d(x,(3,2),(2,2)).flatten(1)
        return F.normalize(self.projection(x),dim=1)


class LevelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(3,24,3,padding=1),nn.BatchNorm2d(24),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(24,48,3,padding=1),nn.BatchNorm2d(48),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(48,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,96,3,padding=1),nn.ReLU(),nn.AvgPool2d(3,2))
        self.head=nn.Linear(96*3*8,101)

    def forward(self,x):
        return self.head(self.layers(x).flatten(1))


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def tensor_images(paths,size):
    array=np.stack([np.asarray(Image.open(p).convert('RGB').resize(size,Image.Resampling.BILINEAR)) for p in paths])
    return torch.from_numpy(array).permute(0,3,1,2).float().div_(255)


def distort(x,strong=True):
    n,_,h,w=x.shape
    # Batched, GPU-resident augmentation; retain artwork identity and geometry.
    theta=torch.zeros(n,2,3,device=x.device)
    theta[:,0,0]=torch.empty(n,device=x.device).uniform_(.85,1.12)
    theta[:,1,1]=torch.empty(n,device=x.device).uniform_(.85,1.12)
    theta[:,:,2]=torch.empty(n,2,device=x.device).uniform_(-.09,.09)
    grid=F.affine_grid(theta,x.shape,align_corners=False)
    x=F.grid_sample(x,grid,padding_mode='border',align_corners=False)
    brightness=torch.empty(n,1,1,1,device=x.device).uniform_(.80,1.15)
    contrast=torch.empty(n,1,1,1,device=x.device).uniform_(.8,1.2)
    x=((x-.5)*contrast+.5)*brightness
    x=x+torch.randn_like(x)*(.014 if strong else .005)
    if strong:
        x=F.avg_pool2d(x,3,1,1)
        # Card UI overlays occupy corners; do not let the encoder depend on them.
        x[:,:,:int(h*.12),:int(w*.16)]=torch.rand(n,3,1,1,device=x.device)
        x[:,:,int(h*.86):,:int(w*.42)]=torch.rand(n,3,1,1,device=x.device)
        x[:,:,int(h*.79):,int(w*.78):]=torch.rand(n,3,1,1,device=x.device)
    return x.clamp_(0,1)


def export(model,path,shape):
    model=model.eval().cpu()
    torch.onnx.export(model,torch.zeros(*shape),str(path),input_names=['image'],output_names=['output'],
                      dynamic_axes={'image':{0:'batch'},'output':{0:'batch'}},opset_version=17,dynamo=False)


def train_encoder(data,output,steps=1200,batch=256):
    if not torch.cuda.is_available():raise RuntimeError('GPU required for training')
    seed_all(7319)
    torch.set_num_threads(8)
    cards=read(data/'catalog.json')['cards']
    images=tensor_images([data/c.get('match_file',c['file']) for c in cards],(128,160)).cuda()
    held=[i for i,c in enumerate(cards) if c['id']%7==0]
    train=[i for i in range(len(cards)) if i not in held]
    train_ids=torch.tensor(train,device='cuda')
    model=CardEncoder(pretrained=True).cuda()
    prototypes=nn.Parameter(torch.randn(len(cards),128,device='cuda')*.02)
    optim=torch.optim.AdamW([*model.parameters(),prototypes],lr=.0004,weight_decay=.0001)
    started=time.perf_counter();model.train()
    for step in range(steps):
        indices=train_ids[torch.randint(len(train),(batch,),device='cuda')]
        embeddings=model(distort(images[indices]))
        logits=embeddings@F.normalize(prototypes,dim=1).T/.08
        loss=F.cross_entropy(logits,indices)
        # Canonical-reference consistency makes the same encoder usable for new IDs.
        if step%2==0:
            with torch.no_grad():target=model(images[indices])
            loss=loss+.25*(1-(embeddings*target).sum(1)).mean()
        optim.zero_grad(set_to_none=True);loss.backward();optim.step()
        if (step+1)%100==0 or step==0:
            status={'step':step+1,'steps':steps,'loss':float(loss.detach()),'elapsed_s':round(time.perf_counter()-started,1),
                    'gpu_allocated_gib':round(torch.cuda.memory_allocated()/2**30,2)}
            write(output/'encoder-progress.json',status);print(json.dumps(status),flush=True)
    model.eval()
    with torch.no_grad():
        gallery=model(images)
        results={}
        for name,indices in [('seen',train),('new_card_holdout',held)]:
            ids=torch.tensor(indices,device='cuda').repeat_interleave(12)
            predictions=[]
            for start in range(0,len(ids),batch):
                emb=model(distort(images[ids[start:start+batch]]))
                predictions.append((emb@gallery.T).argmax(1))
            pred=torch.cat(predictions)
            results[name]={'correct':int((pred==ids).sum()),'total':len(ids),
                           'ids':[(cards[i]['kind'],cards[i]['id']) for i in indices]}
        np.save(output/'gallery.npy',gallery.cpu().numpy())
    torch.save(model.cpu().state_dict(),output/'encoder.pt')
    export(model,output/'encoder.onnx',(1,3,160,128))
    write(output/'encoder-evaluation.json',{'method':'GPU augmentation retrieval; holdout identities excluded from training',
          'training_steps':steps,'batch':batch,'elapsed_s':time.perf_counter()-started,'results':results})
    print(json.dumps(results),flush=True)


def make_level_data(data,output,count=18000,seed=873):
    from .font import NativeFont
    rng=random.Random(seed);font=NativeFont(data)
    catalog=read(data/'catalog.json')['cards']
    backgrounds=[Image.open(data/c.get('match_file',c['file'])).convert('RGB') for c in catalog]
    label=font.text('Lv.',32)
    digits={i:font.text(str(i),32) for i in range(1,101)}
    masks={}
    for i in range(101):
        canvas=Image.new('RGBA',(144,56))
        if i:
            canvas.alpha_composite(label,(0,round(36-label.height/2)))
            canvas.alpha_composite(digits[i],(46,round(36-digits[i].height/2)))
        masks[i]=canvas
    xs=[];ys=[]
    for j in range(count):
        value=j%101
        bg=rng.choice(backgrounds)
        bw,bh=bg.size;x=rng.randrange(max(1,bw-100));y=rng.randrange(max(1,bh-50))
        patch=bg.crop((x,y,min(bw,x+144),min(bh,y+56))).resize((144,56)).convert('RGBA')
        tile=masks[value]
        factor=rng.uniform(.82,1.15)
        tile=tile.resize((round(tile.width*factor),round(tile.height*factor)),Image.Resampling.BILINEAR)
        patch.alpha_composite(tile,(rng.randint(-5,6),rng.randint(-6,5)))
        if rng.random()<.6:
            small=rng.uniform(.52,.9)
            patch=patch.resize((round(144*small),round(56*small)),Image.Resampling.BILINEAR).resize((144,56),Image.Resampling.BILINEAR)
        if rng.random()<.4:patch=patch.filter(ImageFilter.GaussianBlur(rng.uniform(.15,.6)))
        buffer=io.BytesIO();patch.convert('RGB').save(buffer,format='JPEG',quality=rng.randrange(48,99))
        xs.append(np.asarray(Image.open(io.BytesIO(buffer.getvalue()))));ys.append(value)
    np.savez(output,images=np.stack(xs),labels=np.array(ys,np.int64),seed=seed)


def train_level(data,output,steps=1200,batch=256):
    seed_all(8317);torch.set_num_threads(8)
    trainfile=output/'level-train.npz';validfile=output/'level-valid.npz'
    if not trainfile.exists():make_level_data(data,trainfile,18000,873)
    if not validfile.exists():make_level_data(data,validfile,3030,9873)
    train=np.load(trainfile);valid=np.load(validfile)
    # Entire synthetic corpus in GPU memory; no online CPU data-loader bottleneck.
    xs=torch.from_numpy(train['images']).permute(0,3,1,2).cuda().float().div_(255)
    ys=torch.from_numpy(train['labels']).cuda()
    vx=torch.from_numpy(valid['images']).permute(0,3,1,2).cuda().float().div_(255)
    vy=torch.from_numpy(valid['labels']).cuda()
    model=LevelNet().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    started=time.perf_counter();model.train()
    for step in range(steps):
        idx=torch.randint(len(xs),(batch,),device='cuda')
        x=(xs[idx]*torch.empty(batch,1,1,1,device='cuda').uniform_(.85,1.1)).clamp(0,1)
        loss=F.cross_entropy(model(x),ys[idx])
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if (step+1)%100==0 or step==0:
            status={'step':step+1,'steps':steps,'loss':float(loss.detach()),'elapsed_s':round(time.perf_counter()-started,1),
                    'gpu_allocated_gib':round(torch.cuda.memory_allocated()/2**30,2)}
            write(output/'level-progress.json',status);print(json.dumps(status),flush=True)
    model.eval()
    with torch.no_grad():
        correct=0
        for start in range(0,len(vx),batch):
            pred=model(vx[start:start+batch]).argmax(1)
            correct+=int((pred==vy[start:start+batch]).sum())
    torch.save(model.cpu().state_dict(),output/'level.pt')
    export(model,output/'level.onnx',(1,3,56,144))
    write(output/'level-evaluation.json',{'correct':correct,'total':len(vx),'steps':steps,
         'elapsed_s':time.perf_counter()-started,'method':'independent seed synthetic validation; renderer must be calibrated separately'})
    print(json.dumps({'level_correct':correct,'total':len(vx)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['encoder','level']);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=1200);p.add_argument('--batch',type=int,default=256)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    (train_encoder if a.kind=='encoder' else train_level)(a.data,a.output,a.steps,a.batch)
