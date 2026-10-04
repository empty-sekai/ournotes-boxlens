"""Screenshot recognition: card-frame locator, per-kind identity and field readers.

One scan runs three stages:

1. ``locator.onnx`` finds member and Snap card tiles (and their kind) anywhere
   in the screenshot, including cards that are not in the local gallery.
2. The per-kind artwork encoder embeds each tile's artwork window and compares
   it with the gallery of the same kind. A tile is identified only when the
   best cosine similarity and its gap to the runner-up both reach the model's
   thresholds; otherwise it is reported as unidentified.
3. Per-kind field and card-rank classifiers read the parameter field (level or
   member training count) and the card-rank icon of every located tile.

Every model file, its SHA-256 and its acceptance thresholds come from
``models/recognition.json``. The gallery is computed from the user's local
catalog artwork and cached in ``models/<kind>-gallery.npz``; adding a card only
adds a gallery vector.
"""
import hashlib
import importlib.metadata
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from . import kind_fields
from .assets import read
from .kind_geometry import KINDS, query_crop, to_chw
from .locator import Locator, visible_fraction

MODELS_SCHEMA = 'ournotes-boxlens.recognition-models/2'
PROVENANCE_SCHEMA = 'ournotes-boxlens.recognition-provenance/2'
GALLERY_SCHEMA = 'ournotes-boxlens.gallery-cache/2'


def field(value=None, confidence=0., reason=None):
    out = {'value': value, 'confidence': round(float(confidence), 4)}
    if reason:
        out['reason'] = reason
    return out


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_models(data):
    """Read ``models/recognition.json`` and check every referenced file's hash."""
    folder = Path(data) / 'models'
    path = folder / 'recognition.json'
    if not path.is_file():
        raise FileNotFoundError(f'Missing model manifest: {path}. Unpack the released weights into {folder}.')
    manifest = read(path)
    if manifest.get('schema') != MODELS_SCHEMA:
        raise ValueError(f'Unsupported model manifest schema in {path}: {manifest.get("schema")!r}')
    entries = [('locator', manifest['locator'])]
    entries += [(f'{kind}-{role}', manifest[role][kind]) for role in ('encoders', 'fields', 'ranks') for kind in KINDS]
    for name, entry in entries:
        file = folder / entry['file']
        if not file.is_file():
            raise FileNotFoundError(f'Missing model file for {name}: {file}')
        if sha256(file) != entry['sha256']:
            raise ValueError(f'Model file hash mismatch for {name}: {file}')
    return manifest


def session(path, threads):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])


def run(model, batch, chunk=64):
    if not len(batch):
        return np.zeros((0, model.get_outputs()[0].shape[-1]), np.float32)
    return np.concatenate([model.run(None, {'image': np.ascontiguousarray(batch[i:i+chunk], np.float32)})[0]
                           for i in range(0, len(batch), chunk)])


class Gallery:
    """Reference embeddings of one kind, rebuilt when the model or artwork changes."""

    def __init__(self, data, cards, kind, model, model_sha256, use_cache=True):
        from .kind_encoder import reference_array, reference_source
        data = Path(data)
        self.kind = kind
        self.indices = [i for i, c in enumerate(cards) if c['kind'] == kind]
        sources = [reference_source(data, cards[i]) for i in self.indices]
        fingerprint = hashlib.sha256(f'{GALLERY_SCHEMA}:{kind}:{model_sha256}'.encode())
        for i, source in zip(self.indices, sources):
            fingerprint.update(f":{cards[i]['id']}:".encode())
            fingerprint.update(source.read_bytes())
        digest = fingerprint.hexdigest()
        cache = data / f'models/{kind}-gallery.npz'
        if use_cache and cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                if str(saved['fingerprint']) == digest:
                    self.vectors = saved['vectors'].copy()
                    self.fingerprint = digest
                    return
        references = np.stack([reference_array(kind, source) for source in sources]) if sources else np.zeros((0, 3, 1, 1), np.float32)
        self.vectors = run(model, references) if sources else np.zeros((0, 128), np.float32)
        self.fingerprint = digest
        if use_cache:
            temporary = cache.with_name(f'{cache.stem}.{os.getpid()}.tmp.npz')
            np.savez_compressed(temporary, vectors=self.vectors, fingerprint=digest)
            os.replace(temporary, cache)

    def retrieve(self, embeddings):
        """Per query: (catalog index or None, similarity, gap to the runner-up)."""
        if not len(self.indices):
            return [(None, 0., 0.) for _ in embeddings]
        scores = embeddings @ self.vectors.T
        out = []
        for row in scores:
            order = np.argsort(row)
            best = float(row[order[-1]])
            second = float(row[order[-2]]) if len(order) > 1 else -1.
            out.append((self.indices[int(order[-1])], best, best - second))
        return out


class Engine:
    def __init__(self, data, threads=2, excluded_identities=None):
        self.data = Path(data)
        self.models = load_models(self.data)
        folder = self.data / 'models'
        cv2.setNumThreads(threads)
        self.cards = read(self.data / 'catalog.json')['cards']
        if excluded_identities:
            self.cards = [c for c in self.cards if (c['kind'], c['id']) not in excluded_identities]
        locator = self.models['locator']
        self.locator = Locator(folder / locator['file'], long_edge=locator['long_edge'], threads=threads,
                               threshold=locator['threshold'])
        self.encoders = {k: session(folder / self.models['encoders'][k]['file'], threads) for k in KINDS}
        self.fields = {k: session(folder / self.models['fields'][k]['file'], threads) for k in KINDS}
        self.ranks = {k: session(folder / self.models['ranks'][k]['file'], threads) for k in KINDS}
        self.galleries = {k: Gallery(self.data, self.cards, k, self.encoders[k], self.models['encoders'][k]['sha256'],
                                     use_cache=not excluded_identities) for k in KINDS}
        core = hashlib.sha256()
        for name in ['engine.py', 'locator.py', 'kind_geometry.py', 'kind_encoder.py', 'kind_fields.py']:
            core.update(name.encode())
            core.update((Path(__file__).parent / name).read_bytes())
        self.provenance = {
            'schema': PROVENANCE_SCHEMA,
            'files': {name: sha256(self.data / name) for name in ['catalog.json', 'models/recognition.json']},
            'galleries': {k: self.galleries[k].fingerprint for k in KINDS},
            'inference_core_sha256': core.hexdigest(),
            'onnxruntime_version': importlib.metadata.version('onnxruntime')}

    def identify(self, image, detections):
        """Gallery decision for each detection: (catalog index or None, similarity, gap, candidate index)."""
        out = [None] * len(detections)
        for kind in KINDS:
            chosen = [i for i, d in enumerate(detections) if d['kind'] == kind]
            if not chosen:
                continue
            crops = np.stack([to_chw(query_crop(image, kind, detections[i]['bbox'])) for i in chosen])
            accept = self.models['encoders'][kind]['accept']
            for i, (index, similarity, gap) in zip(chosen, self.galleries[kind].retrieve(run(self.encoders[kind], crops))):
                identified = index is not None and similarity >= accept['similarity'] and gap >= accept['margin']
                out[i] = (index if identified else None, similarity, gap, index)
        return out

    def read_fields(self, image, detections):
        """Parameter-field and card-rank readings for each detection."""
        height, width = image.shape[:2]
        out = [{'level': field(reason='cropped'), 'awake_count': field(reason='cropped'), 'display_mode': 'unknown',
                'card_rank': field(reason='cropped_rank_icon')} for _ in detections]
        for kind in KINDS:
            chosen = [i for i, d in enumerate(detections) if d['kind'] == kind]
            readable = [i for i in chosen if kind_fields.field_inside(kind, detections[i]['bbox'], width, height)]
            if readable:
                crops = kind_fields.to_input([kind_fields.crop_field(image, kind, detections[i]['bbox']) for i in readable])
                accept = self.models['fields'][kind]['accept']
                for i, p in zip(readable, kind_fields.softmax(run(self.fields[kind], crops))):
                    decision = kind_fields.decide_field(kind, p, accept['confidence'], accept['margin'])
                    confidence, reason = decision['confidence'], decision.get('reason')
                    if decision['level'] is not None:
                        level, awake = field(decision['level'], confidence), field(reason='not_visible_in_level_view')
                    elif decision['awake_count'] is not None:
                        level, awake = field(reason='not_visible_in_training_view'), field(decision['awake_count'], confidence)
                    else:
                        level = awake = field(confidence=confidence, reason=reason)
                    out[i].update(level=level, awake_count=awake, display_mode=decision['display_mode'])
            readable = [i for i in chosen if kind_fields.rank_icon_inside(kind, detections[i]['bbox'], width, height)]
            if readable:
                crops = kind_fields.to_input([kind_fields.crop_rank(image, kind, detections[i]['bbox']) for i in readable])
                accept = self.models['ranks'][kind]['accept']
                for i, p in zip(readable, kind_fields.softmax(run(self.ranks[kind], crops))):
                    decision = kind_fields.decide_rank(p, accept['confidence'], accept['margin'])
                    out[i]['card_rank'] = field(decision['value'], decision['confidence'], decision.get('reason'))
        return out

    def scan(self, image, source='image'):
        started = time.perf_counter()
        height, width = image.shape[:2]
        detections = self.locator(np.ascontiguousarray(image[:, :, ::-1]))
        identities = self.identify(image, detections)
        readings = self.read_fields(image, detections)
        cards, unidentified = [], []
        for detection, (index, similarity, gap, candidate), values in zip(detections, identities, readings):
            item = {'kind': detection['kind'], 'bbox': [round(float(v), 2) for v in detection['bbox']],
                    'locator_score': round(detection['score'], 4),
                    'visible_fraction': round(visible_fraction(detection['bbox'], (width, height)), 4),
                    'identity_similarity': round(similarity, 4), 'identity_margin': round(gap, 4), **values}
            if index is None:
                item['candidate'] = None if candidate is None else {'id': self.cards[candidate]['id'], 'name': self.cards[candidate]['name']}
                item['review'] = True
                unidentified.append(item)
                continue
            catalog = self.cards[index]
            maximum = catalog.get('max_level')
            rank = item['card_rank']['value']
            if rank is not None:
                maximum = catalog.get('rank_level_limits', {}).get(str(rank), maximum)
            if maximum and item['level']['value'] is not None and item['level']['value'] > maximum:
                item['level'] = field(reason='outside_masterdata_level_limit')
            cards.append({'kind': item['kind'], 'id': catalog['id'], 'name': catalog['name'], 'rarity': catalog['rarity'],
                          'card_type': catalog['card_type'], **{k: v for k, v in item.items() if k != 'kind'},
                          'review': any(item[k]['value'] is None for k in ('level', 'card_rank'))})
        order = lambda c: (round(c['bbox'][1] / 20), c['bbox'][0])
        return {'source': source, 'width': width, 'height': height,
                'recognition_provenance': self.provenance,
                'source_id': hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest(),
                'cards': sorted(cards, key=order), 'unidentified': sorted(unidentified, key=order),
                'elapsed_ms': round((time.perf_counter() - started) * 1000, 2),
                'coverage': 'observed_only'}


def merge(scans, player='local'):
    grouped=defaultdict(list)
    for scan in scans:
        for card in scan['cards']:
            grouped[(card['kind'],card['id'])].append((scan['source'],scan.get('source_id'),card))
    cards=[]
    for (kind,id_),observations in sorted(grouped.items()):
        item={'kind':kind,'id':id_,'name':observations[0][2]['name'],
              'sources':list(dict.fromkeys(source for source,_,c in observations)),
              'source_ids':list(dict.fromkeys(sid for _,sid,_ in observations if sid)), 'conflicts':{}}
        for key in ('level','card_rank','awake_count'):
            values=defaultdict(list)
            for source,sid,c in observations:
                if c[key]['value'] is not None:
                    evidence={'source':source,'source_id':sid,'confidence':c[key]['confidence']}
                    if evidence not in values[c[key]['value']]:values[c[key]['value']].append(evidence)
            if len(values)==1:
                value=next(iter(values))
                item[key]=field(value,max(x['confidence'] for x in values[value]))
            elif len(values)>1:
                item[key]=field(reason='conflicting_observations')
                item['conflicts'][key]=dict(values)
            else:
                item[key]=field(reason='not_observed')
        cards.append(item)
    versions={json.dumps(s['recognition_provenance'],sort_keys=True):s['recognition_provenance'] for s in scans if s.get('recognition_provenance')}
    return {'schema':'bdon-box/1','player':player,'coverage':'observed_only','cards':cards,
            'recognition_provenance':list(versions.values()),
            'unique_count':len(cards),'observations':sum(len(s['cards']) for s in scans),
            'duplicate_observations':sum(len(s['cards']) for s in scans)-len(cards)}
