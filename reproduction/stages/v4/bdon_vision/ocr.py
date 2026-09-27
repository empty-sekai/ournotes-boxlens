import cv2
import numpy as np

from .assets import native_module
from .engine import field


class NumberReader:
    def __init__(self,data):
        native=native_module(data)
        font=native.GameText()
        self.templates=[]
        for value in range(1,101):
            tile=np.array(font.text('Lv.'+str(value),36))
            self.templates.append((value,cv2.cvtColor(tile,cv2.COLOR_RGBA2BGRA)))

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
