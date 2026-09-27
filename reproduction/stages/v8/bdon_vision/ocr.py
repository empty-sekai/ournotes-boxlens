import cv2
import numpy as np
from pathlib import Path

from .assets import native_module
from .engine import field


class NumberReader:
    def __init__(self,data,threads=2):
        self.session=None
        path=Path(data)/'models/fields.onnx'
        self.multifield=path.exists()
        if not self.multifield:path=Path(data)/'models/level.onnx'
        if path.exists():
            import onnxruntime as ort
            options=ort.SessionOptions()
            options.intra_op_num_threads=threads;options.inter_op_num_threads=1
            self.session=ort.InferenceSession(str(path),sess_options=options,
                                             providers=['CPUExecutionProvider'])
            self.templates=[]
            return
        native=native_module(data)
        font=native.GameText()
        self.templates=[]
        for value in range(1,101):
            tile=np.array(font.text('Lv.'+str(value),36))
            self.templates.append((value,cv2.cvtColor(tile,cv2.COLOR_RGBA2BGRA)))

    def read_many(self,image,items):
        return [r['level'] for r in self.read_fields(image,items)]

    def read_fields(self,image,items):
        def empty(reason):
            return {'level':field(reason=reason),'awake_count':field(reason=reason),'display_mode':'unknown'}
        if self.session is None:
            return [{**empty('mode_model_missing'),'level':self.read(image,item)} for item in items]
        patches=[];indices=[];results=[empty('cropped') for _ in items]
        for i,item in enumerate(items):
            x,y,w,h=item['bbox'];scale=w/(212 if item['kind']=='member' else 314)
            left=x-12*scale;top=y+h-50*scale
            right=left+144*scale;bottom=top+56*scale
            if left<0 or top<0 or right>image.shape[1] or bottom>image.shape[0]-2:
                continue
            matrix=np.array([[scale,0,left],[0,scale,top]],np.float32)
            patch=cv2.warpAffine(image,matrix,(144,56),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
            patches.append(patch[:,:,::-1].transpose(2,0,1).astype(np.float32)/255)
            indices.append(i)
        if patches:
            logits=self.session.run(None,{'image':np.stack(patches)})[0]
            probs=np.exp(logits-logits.max(axis=1,keepdims=True));probs/=probs.sum(axis=1,keepdims=True)
            for i,p in zip(indices,probs):
                order=np.argsort(p);value=int(order[-1]);confidence=float(p[value])
                if value==0:
                    results[i]=empty('other_or_hidden_parameter')
                    results[i]['display_mode']='other'
                elif confidence<(.98 if self.multifield else .90) or confidence-float(p[order[-2]])<.5:
                    results[i]=empty('ambiguous_parameter')
                    results[i]['level']['confidence']=round(confidence,4)
                elif value>100:
                    if items[i]['kind']=='member':
                        results[i]=empty('not_visible_in_training_view')
                        results[i]['awake_count']=field(value-100,confidence)
                        results[i]['display_mode']='training'
                    else:results[i]=empty('unsupported_parameter')
                else:
                    results[i]=empty('not_visible_in_level_view')
                    results[i]['level']=field(value,confidence)
                    results[i]['display_mode']='level'
        return results

    def read(self,image,item):
        x,y,w,h=item['bbox']
        if y+h>image.shape[0]-2:
            return field(reason='cropped')
        patch=image[max(0,int(y+h*.78)):min(image.shape[0],int(y+h*1.07)),
                    max(0,int(x-w*.06)):min(image.shape[1],int(x+w*.52))]
        if patch.size==0:
            return field(reason='cropped')
        scores=[]
        target=(w/216 if item['kind']=='member' else w/318)
        for value,full in self.templates:
            best=-1
            for factor in [.90,1.,1.10]:
                scale=target*factor
                templ=cv2.resize(full,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
                th,tw=templ.shape[:2]
                if th>patch.shape[0] or tw>patch.shape[1] or min(th,tw)<3:
                    continue
                mask=(templ[:,:,3]>180).astype(np.uint8)*255
                match=cv2.matchTemplate(patch,templ[:,:,:3],cv2.TM_CCORR_NORMED,mask=mask)
                match=np.nan_to_num(match,nan=-1,posinf=-1,neginf=-1)
                best=max(best,float(match.max()))
            scores.append((best,value))
        scores.sort(reverse=True)
        best,value=scores[0]
        if best<.90 or best-scores[1][0]<.003:
            return field(confidence=best,reason='unreadable_level')
        return field(value,best)
