"""Regressions for model pinning, cropped fields and icons, and box-level truth auditing."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bdon_vision.audit_merge import audit_box
from bdon_vision.engine import MODELS_SCHEMA, Engine, field, load_models
from test_merge import observation


class Constant:
    """Stand-in session returning fixed logits for every crop."""

    def __init__(self, logits):
        self.logits = np.asarray(logits, np.float32)

    def get_outputs(self):
        return [type('Output', (), {'shape': [None, len(self.logits)]})()]

    def run(self, _, feeds):
        return [np.repeat(self.logits[None], len(feeds['image']), axis=0)]


def certain(classes, index):
    logits = np.full(classes, -20.)
    logits[index] = 20.
    return logits


def write_models(folder, mutate=None):
    folder.mkdir(parents=True)
    entry = lambda name: {'file': name, 'sha256': hashlib.sha256((folder/name).read_bytes()).hexdigest()}
    names = ['locator.onnx'] + [f'{k}-{r}.onnx' for r in ('encoder', 'fields', 'rank') for k in ('member', 'snap')]
    for name in names:
        (folder/name).write_bytes(name.encode())
    manifest = {'schema': MODELS_SCHEMA, 'locator': {**entry('locator.onnx'), 'long_edge': 640, 'threshold': .4},
                'encoders': {k: {**entry(f'{k}-encoder.onnx'), 'accept': {'similarity': .8, 'margin': .1}} for k in ('member', 'snap')},
                'fields': {k: {**entry(f'{k}-fields.onnx'), 'accept': {'confidence': .995, 'margin': .5}} for k in ('member', 'snap')},
                'ranks': {k: {**entry(f'{k}-rank.onnx'), 'accept': {'confidence': .995, 'margin': .5}} for k in ('member', 'snap')}}
    if mutate:
        mutate(folder)
    (folder/'recognition.json').write_text(json.dumps(manifest), encoding='utf-8')


def reader(member_field=12, snap_field=40, rank=3):
    engine = Engine.__new__(Engine)
    accept = {'confidence': .995, 'margin': .5}
    engine.models = {'fields': {'member': {'accept': accept}, 'snap': {'accept': accept}},
                     'ranks': {'member': {'accept': accept}, 'snap': {'accept': accept}}}
    engine.fields = {'member': Constant(certain(106, member_field)), 'snap': Constant(certain(101, snap_field))}
    engine.ranks = {'member': Constant(certain(6, rank)), 'snap': Constant(certain(6, rank))}
    return engine


class IntegrityTests(unittest.TestCase):
    def test_model_manifest_is_required_and_hashes_are_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FileNotFoundError, 'model manifest'):
                load_models(folder)
            write_models(Path(folder)/'models')
            self.assertEqual(load_models(folder)['schema'], MODELS_SCHEMA)
        with tempfile.TemporaryDirectory() as folder:
            write_models(Path(folder)/'models', lambda models: (models/'snap-rank.onnx').write_bytes(b'changed'))
            with self.assertRaisesRegex(ValueError, 'hash mismatch for snap-ranks'):
                load_models(folder)

    def test_fields_and_icons_outside_the_image_stay_unknown(self):
        engine = reader()
        image = np.zeros((405, 877, 3), np.uint8)
        inside = {'kind': 'member', 'bbox': [100., 50., 112., 147.]}
        right_edge = {'kind': 'member', 'bbox': [763.25, 54.33, 115.71, 152.94]}
        bottom_edge = {'kind': 'snap', 'bbox': [300., 340., 163., 92.]}
        a, b, c = engine.read_fields(image, [inside, right_edge, bottom_edge])
        self.assertEqual((a['level']['value'], a['card_rank']['value'], a['display_mode']), (12, 3, 'level'))
        self.assertEqual(a['awake_count']['reason'], 'not_visible_in_level_view')
        self.assertEqual(b['card_rank']['reason'], 'cropped_rank_icon')
        self.assertIsNone(c['level']['value'])
        self.assertEqual(c['level']['reason'], 'cropped')

    def test_training_count_and_hidden_parameter(self):
        image = np.zeros((400, 400, 3), np.uint8)
        tile = {'kind': 'member', 'bbox': [100., 50., 112., 147.]}
        training = reader(member_field=103).read_fields(image, [tile])[0]
        self.assertEqual((training['awake_count']['value'], training['display_mode']), (3, 'training'))
        self.assertEqual(training['level']['reason'], 'not_visible_in_training_view')
        hidden = reader(member_field=0, rank=0).read_fields(image, [tile])[0]
        self.assertEqual((hidden['level']['value'], hidden['display_mode']), (None, 'other'))
        self.assertEqual(hidden['card_rank']['reason'], 'no_visible_rank_icon')

    def test_box_audit_rejects_invented_fields_and_preserves_truth_conflicts(self):
        rows=[{'cards':[{'kind':'member','id':1,'level':20,'card_rank':None,'awake_count':None}]},
              {'cards':[{'kind':'member','id':1,'level':21,'card_rank':None,'awake_count':None}]}]
        scans=[observation('a','a',20,2),observation('b','b',21)]
        result=audit_box(rows,scans)
        self.assertEqual(result['fields']['level']['conflict_preserved'],1)
        self.assertEqual(result['fields']['card_rank']['fabricated_field'],1)
        self.assertTrue(result['duplicate_upload_invariant'])
        # A confident merged value must not conceal disagreement in the truth.
        scans[1]['cards'][0]['level']=field(20,.99)
        self.assertEqual(audit_box(rows,scans)['fields']['level']['conflict_lost'],1)


if __name__=='__main__':unittest.main()
