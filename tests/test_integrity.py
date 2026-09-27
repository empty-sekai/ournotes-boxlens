"""Regressions for cropped icons, hidden fields, and box-level truth auditing."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bdon_vision.audit_merge import audit_box
from bdon_vision.engine import Engine,field
from bdon_vision.ocr import NumberReader
from test_merge import observation


class IntegrityTests(unittest.TestCase):
    def test_icon_outside_right_edge_is_unknown_before_template_matching(self):
        engine=Engine.__new__(Engine)
        image=np.zeros((405,877,3),np.uint8)
        item={'kind':'member','bbox':[763.25,54.33,115.71,152.94]}
        result=engine.read_rank(image,item)
        self.assertIsNone(result['value'])
        self.assertEqual(result['reason'],'cropped_rank_icon')

    def test_level_only_model_cannot_silently_read_parameter_modes(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);(path/'models').mkdir();(path/'models/level.onnx').touch()
            with self.assertRaisesRegex(FileNotFoundError,'parameter-mode'):
                NumberReader(path)

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
