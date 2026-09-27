"""Check that public PyTorch checkpoints and ONNX inference weights agree."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from bdon_vision.assets import write
from bdon_vision.train import CardEncoder,LevelNet


def verify(encoder_pt,encoder_onnx,fields_pt,fields_onnx,output):
    torch.set_num_threads(2);rng=np.random.default_rng(64291)
    encoder=CardEncoder();fields=LevelNet();fields.head=nn.Linear(96*3*8,106)
    report={'schema':'ournotes-boxlens.weight-parity/1','seed':64291,'cases':{},'passed':True}
    for name,model,checkpoint,onnx,shape in [
        ('encoder',encoder,encoder_pt,encoder_onnx,(4,3,160,128)),
        ('fields',fields,fields_pt,fields_onnx,(4,3,56,144))]:
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval()
        x=rng.random(shape,dtype=np.float32);x[0]=0.;x[1]=1.
        with torch.inference_mode():expected=model(torch.from_numpy(x)).numpy()
        opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
        session=ort.InferenceSession(str(onnx),sess_options=opts,providers=['CPUExecutionProvider'])
        actual=session.run(None,{'image':x})[0]
        close=np.allclose(expected,actual,atol=2e-4,rtol=2e-4)
        report['cases'][name]={'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'onnx_sha256':hashlib.sha256(onnx.read_bytes()).hexdigest(),'shape':shape,
            'max_absolute_error':float(np.abs(expected-actual).max()),'allclose':bool(close)}
        if name=='fields':report['cases'][name]['class_agreement']=bool(np.array_equal(expected.argmax(1),actual.argmax(1)))
        report['passed']&=bool(close)
    write(output,report);print(report)
    if not report['passed']:raise RuntimeError('Checkpoint/ONNX parity failed')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['encoder-pt','encoder-onnx','fields-pt','fields-onnx','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();verify(a.encoder_pt,a.encoder_onnx,a.fields_pt,a.fields_onnx,a.output)
