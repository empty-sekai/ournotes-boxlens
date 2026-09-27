import hashlib
import math
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .assets import read


def field(value=None, confidence=0., reason=None):
    out = {'value': value, 'confidence': round(float(confidence), 4)}
    if reason:
        out['reason'] = reason
    return out


def overlap(a, b):
    x = max(0., min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0]))
    y = max(0., min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    return x*y / max(1., min(a[2]*a[3], b[2]*b[3]))


class Engine:
    def __init__(self, data, threads=2):
        self.data = Path(data)
        cv2.setNumThreads(threads)
        cv2.setRNGSeed(12345)
        self.cards = read(self.data / 'catalog.json')['cards']
        index = np.load(self.data / 'index.npz')
        self.points = index['points']
        self.owners = index['owners']
        self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
        self.matcher.add([index['descriptors'].astype(np.float32)])
        self.matcher.train()
        self.sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=.018, edgeThreshold=12)
        self.art = [cv2.imread(str(self.data / c.get('match_file',c['file']))) for c in self.cards]
        self.ranks = {}
        for kind, prefix in [('member', 'CardRank'), ('snap', 'snaplimit_')]:
            self.ranks[kind] = [cv2.imread(str(self.data / f'native/game-ui/sprites/{prefix}{i}.png'), cv2.IMREAD_UNCHANGED) for i in range(6)]
        from .ocr import NumberReader
        self.numbers = NumberReader(self.data,threads)
        self.gallery=None
        if (self.data/'models/encoder.onnx').exists():
            from .inference import Gallery
            self.gallery=Gallery(self.data,self.cards,threads)

    def fill_grid(self,image,items):
        """Recover weak-feature cards only at grid positions supported by matches."""
        def clusters(values,tolerance):
            groups=[]
            for value in sorted(values):
                if groups and abs(value-np.median(groups[-1]))<tolerance:groups[-1].append(value)
                else:groups.append([value])
            return [float(np.median(group)) for group in groups]
        def axis(values,size):
            points=clusters(values,size*.18)
            gaps=np.diff(points)
            if len(gaps)>1:
                pitch=float(np.min(gaps))
                if 1.02*size<pitch<1.6*size:
                    points=sorted(set(points+[points[i]+pitch*j for i,gap in enumerate(gaps)
                        for j in range(1,round(gap/pitch)) if abs(gap/pitch-round(gap/pitch))<.08]))
            return points
        for kind in ('member','snap'):
            seed=[c for c in items if c['kind']==kind]
            if len(seed)<3:continue
            sizes=np.array([c['bbox'][2:] for c in seed]);w,h=np.median(sizes,axis=0)
            if np.max(np.std(sizes,axis=0)/[w,h])>.08:continue
            xs=axis([c['bbox'][0] for c in seed],w);ys=axis([c['bbox'][1] for c in seed],h)
            proposals=[];patches=[]
            for y in ys:
                for x in xs:
                    bbox=[x,y,float(w),float(h)]
                    if any(overlap(bbox,c['bbox'])>.4 for c in items):continue
                    if min(x,y)<0 or x+w>image.shape[1] or y+h>image.shape[0]:continue
                    patch=cv2.getRectSubPix(image,(round(w),round(h)),(x+w/2,y+h/2))
                    proposals.append(bbox);patches.append(patch)
            if not proposals:continue
            candidates=[i for i,c in enumerate(self.cards) if c['kind']==kind]
            mask=np.ones((48,48),bool);mask[37:]=False;mask[:9,:10]=False
            art=np.stack([cv2.resize(self.art[i],(48,48))[mask].ravel() for i in candidates]).astype(float)
            art-=art.mean(axis=1,keepdims=True);art/=np.maximum(np.linalg.norm(art,axis=1,keepdims=True),1e-8)
            learned=self.gallery.retrieve(patches,kind) if self.gallery else [None]*len(patches)
            for bbox,patch,neural in zip(proposals,patches,learned):
                query=cv2.resize(patch,(48,48))[mask].ravel().astype(float)
                query-=query.mean();query/=max(np.linalg.norm(query),1e-8)
                scores=art@query;order=np.argsort(scores);i=candidates[int(order[-1])]
                score=float(scores[order[-1]]);margin=float(scores[order[-1]]-scores[order[-2]])
                if neural:
                    ni,ns,ngap=neural
                    if ni!=i or ns<.80 or ngap<.06:continue
                if score<.78 or margin<.08:continue
                card=self.cards[i]
                items.append({'kind':kind,'id':card['id'],'name':card['name'],'rarity':card['rarity'],
                    'card_type':card['card_type'],'bbox':[round(v,2) for v in bbox],
                    'identity_confidence':round(score,4),'identity_method':'grid_gallery',
                    'embedding_similarity':round(neural[1],4) if neural else None,
                    'inliers':0,'visible_fraction':1.,'_index':i})
        return sorted(items,key=lambda c:(round(c['bbox'][1]/20),c['bbox'][0]))

    def identify(self, im):
        h, w = im.shape[:2]
        factor = min(1., 1500 / max(h, w))
        small = cv2.resize(im, None, fx=factor, fy=factor) if factor < 1 else im
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        kp, desc = self.sift.detectAndCompute(gray, None)
        if desc is None:
            return []
        desc = np.sqrt(desc / (desc.sum(axis=1, keepdims=True) + 1e-8))
        grouped = defaultdict(list)
        for matches in self.matcher.knnMatch(desc, k=2):
            if len(matches) == 2 and matches[0].distance < .76 * matches[1].distance:
                m = matches[0]
                grouped[int(self.owners[m.trainIdx])].append((m.trainIdx, m.queryIdx, m.distance))
        found = []
        for i, matches in grouped.items():
            if len(matches) < 5:
                continue
            src = np.float32([self.points[t] for t, q, d in matches])
            dst = np.float32([kp[q].pt for t, q, d in matches]) / factor
            transform, inliers = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC,
                                                           ransacReprojThreshold=2.5/factor, maxIters=1500)
            if transform is None or int(inliers.sum()) < 5:
                continue
            scale = math.hypot(transform[0, 0], transform[1, 0])
            scale_y = math.hypot(transform[0, 1], transform[1, 1])
            angle = math.degrees(math.atan2(transform[1, 0], transform[0, 0]))
            c = self.cards[i]
            if abs(angle) > 4 or abs(transform[0,1]) > .08*scale_y or not .55 < scale/scale_y < 1.8 or not 55 < c['width']*scale < w*.9:
                continue
            x, y = transform[:, 2]
            bw, bh = c['width']*scale, c['height']*scale_y
            if x+bw < 0 or y+bh < 0 or x > w or y > h:
                continue
            # Check the actual artwork, not only local-feature correspondence.
            recovered = cv2.warpAffine(im, cv2.invertAffineTransform(transform),
                                       (c['width'], c['height']), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT)
            a = cv2.resize(self.art[i], (48, 64 if c['kind']=='member' else 27)).astype(float)
            b = cv2.resize(recovered, (a.shape[1], a.shape[0])).astype(float)
            valid = (b.max(axis=2) > 0)
            valid[int(len(valid)*.80):] = False
            valid[:max(1,int(len(valid)*.1)), :8] = False
            if valid.sum() < 150:
                continue
            aa, bb = a[valid].ravel(), b[valid].ravel()
            similarity = float(np.corrcoef(aa, bb)[0, 1])
            if not np.isfinite(similarity) or similarity < .68:
                continue
            visible = max(0,min(w,x+bw)-max(0,x))*max(0,min(h,y+bh)-max(0,y))/(bw*bh)
            found.append({'kind': c['kind'], 'id': c['id'], 'name': c['name'],
                'rarity': c['rarity'], 'card_type': c['card_type'],
                'bbox': [round(float(v),2) for v in (x,y,bw,bh)],
                'identity_confidence': round(similarity,4), 'inliers': int(inliers.sum()),
                'visible_fraction': round(float(visible),4), '_index': i})
        accepted=[]
        for item in sorted(found, key=lambda x:(x['identity_confidence'],x['inliers']), reverse=True):
            if not any(overlap(item['bbox'], other['bbox']) > .5 for other in accepted):
                accepted.append(item)
        return sorted(accepted, key=lambda x:(round(x['bbox'][1]/20),x['bbox'][0]))

    def read_rank(self, image, item):
        x,y,w,h = item['bbox']
        if y+h+6*w/(212 if item['kind']=='member' else 314) > image.shape[0]-2:
            return field(reason='cropped')
        member=item['kind']=='member'
        scale=w/(212 if member else 314)
        cx=x+(188.5 if member else 300.1)*scale
        cy=y+(261.2 if member else 147.8)*scale
        sprites=np.stack(self.ranks[item['kind']]).astype(np.float32)
        alpha=sprites[:,:,:,3:4]/255.
        mask=alpha.max(axis=0)[:,:,0]>.45
        different=(np.ptp(sprites,axis=0).max(axis=2)>35)&mask
        # Rank 0 has translucent dark lobes. An opaque-only mask incorrectly
        # ignores the colored lobes and systematically prefers rank 0.
        weights=(mask.astype(np.float32)+different*3)[:,:,None]
        art=self.art[item['_index']];ah,aw=art.shape[:2]
        scores=[(-1.,i) for i in range(6)]
        for ratio in [.97,1.,1.03]:
            sw=64*scale*ratio;sh=sw*95/102
            for dx,dy in [(0,0),(-.8,0),(.8,0),(0,-.8),(0,.8)]:
                gx=np.linspace(cx+dx-sw/2,cx+dx+sw/2,102,dtype=np.float32)
                gy=np.linspace(cy+dy-sh/2,cy+dy+sh/2,95,dtype=np.float32)
                mx,my=np.meshgrid(gx,gy)
                observed=cv2.remap(image,mx,my,cv2.INTER_LINEAR).astype(np.float32)
                bg=cv2.remap(art,(mx-x)*aw/w,(my-y)*ah/h,cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
                expected=sprites[:,:,:,:3]*alpha+bg[None]*(1-alpha)
                error=(np.abs(expected-observed[None])*weights).sum(axis=(1,2,3))/(weights.sum()*3*255)
                for rank,e in enumerate(error):
                    scores[rank]=(max(scores[rank][0],1-float(e)),rank)
        scores.sort(reverse=True)
        if len(scores)<2:
            return field(reason='unreadable')
        best,rank=scores[0]
        gap=best-scores[1][0]
        if best < .78 or gap < .007:
            return field(confidence=best,reason='ambiguous_icon')
        return field(rank,best)

    def scan(self, image, source='image'):
        started=time.perf_counter()
        items=self.fill_grid(image,self.identify(image))
        levels=self.numbers.read_many(image,items)
        for item,level in zip(items,levels):
            item['card_rank']=self.read_rank(image,item)
            item['level']=level
            item['awake_count']=field(reason='not_visible_in_level_view')
            item.pop('_index',None)
            item['review']=any(item[k]['value'] is None for k in ('level','card_rank'))
        return {'source':source,'width':image.shape[1],'height':image.shape[0],
                'cards':items,'elapsed_ms':round((time.perf_counter()-started)*1000,2),
                'coverage':'observed_only'}


def merge(scans, player='local'):
    grouped=defaultdict(list)
    for scan in scans:
        for card in scan['cards']:
            grouped[(card['kind'],card['id'])].append((scan['source'],card))
    cards=[]
    for (kind,id_),observations in sorted(grouped.items()):
        item={'kind':kind,'id':id_,'name':observations[0][1]['name'],
              'sources':list(dict.fromkeys(source for source,c in observations)),'conflicts':{}}
        for key in ('level','card_rank','awake_count'):
            values=defaultdict(list)
            for source,c in observations:
                if c[key]['value'] is not None:
                    values[c[key]['value']].append({'source':source,'confidence':c[key]['confidence']})
            if len(values)==1:
                value=next(iter(values))
                item[key]=field(value,max(x['confidence'] for x in values[value]))
            elif len(values)>1:
                item[key]=field(reason='conflicting_observations')
                item['conflicts'][key]=dict(values)
            else:
                item[key]=field(reason='not_observed')
        cards.append(item)
    return {'schema':'bdon-box/1','player':player,'coverage':'observed_only','cards':cards,
            'unique_count':len(cards),'observations':sum(len(s['cards']) for s in scans),
            'duplicate_observations':sum(len(s['cards']) for s in scans)-len(cards)}
