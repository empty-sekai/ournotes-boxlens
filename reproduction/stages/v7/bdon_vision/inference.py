"""CPU ONNX artwork embeddings with a versioned, incrementally rebuilt gallery."""
import hashlib
from pathlib import Path
import cv2
import numpy as np


class Gallery:
    def __init__(self,data,cards,threads=2):
        import onnxruntime as ort
        data=Path(data);model=data/'models/encoder.onnx'
        options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
        self.cards=cards
        fingerprint=hashlib.sha256(model.read_bytes())
        images=[]
        for card in cards:
            path=data/card.get('match_file',card['file'])
            fingerprint.update(f"{card['kind']}:{card['id']}".encode())
            fingerprint.update(path.read_bytes())
            images.append(cv2.imread(str(path)))
        digest=fingerprint.hexdigest();cache=data/'models/gallery.npz'
        saved=np.load(cache,allow_pickle=False) if cache.exists() else None
        if saved is not None and str(saved['fingerprint'])==digest:
            self.vectors=saved['vectors'].copy()
        else:
            self.vectors=np.concatenate([self.encode(images[i:i+32]) for i in range(0,len(images),32)])
            np.savez_compressed(cache,vectors=self.vectors,fingerprint=digest)
        if saved is not None:saved.close()

    def encode(self,images):
        xs=np.stack([cv2.resize(im,(128,160))[:,:,::-1].transpose(2,0,1) for im in images]).astype(np.float32)/255
        return self.session.run(None,{'image':xs})[0]

    def retrieve(self,images,kind):
        ids=[i for i,c in enumerate(self.cards) if c['kind']==kind]
        sims=self.encode(images)@self.vectors[ids].T
        result=[]
        for row in sims:
            order=np.argsort(row)
            result.append((ids[int(order[-1])],float(row[order[-1]]),float(row[order[-1]]-row[order[-2]])))
        return result
