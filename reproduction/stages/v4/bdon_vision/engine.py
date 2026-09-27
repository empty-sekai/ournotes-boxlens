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
        self.numbers = NumberReader(self.data)

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
        if y+h > image.shape[0]-2:
            return field(reason='cropped')
        x0,y0=max(0,int(x+w*.68)),max(0,int(y+h*.60))
        x1,y1=min(image.shape[1],int(x+w*1.13)),min(image.shape[0],int(y+h*1.10))
        patch=image[y0:y1,x0:x1]
        scores=[]
        # Sprite units: member 48 on 216 artwork width; Snap 48 on 318.
        for rank, sprite in enumerate(self.ranks[item['kind']]):
            if sprite is None:
                continue
            best=-1
            for ratio in [.96,1.,1.04]:
                size=max(12,round(w*(64/212 if item['kind']=='member' else 64/314)*ratio))
                th=round(size*sprite.shape[0]/sprite.shape[1])
                templ=cv2.resize(sprite,(size,th),interpolation=cv2.INTER_AREA)
                if th>patch.shape[0] or size>patch.shape[1]:
                    continue
                mask=(templ[:,:,3]>220).astype(np.uint8)*255
                score=1-cv2.matchTemplate(patch,templ[:,:,:3],cv2.TM_SQDIFF_NORMED,mask=mask)
                score=np.nan_to_num(score,nan=-1,posinf=-1,neginf=-1)
                best=max(best,float(score.max()))
            scores.append((best,rank))
        scores.sort(reverse=True)
        if len(scores)<2:
            return field(reason='unreadable')
        best,rank=scores[0]
        gap=best-scores[1][0]
        if best < .90 or gap < .003:
            return field(confidence=best,reason='ambiguous_icon')
        return field(rank,best)

    def scan(self, image, source='image'):
        started=time.perf_counter()
        items=self.identify(image)
        for item in items:
            item['card_rank']=self.read_rank(image,item)
            item['level']=self.numbers.read(image,item)
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
