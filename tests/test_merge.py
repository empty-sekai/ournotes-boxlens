"""Data-integrity checks for overlapping screenshots and progression conflicts."""
import copy
import unittest

from bdon_vision.engine import field,merge


def observation(source,source_id,level=None,rank=None,awake=None):
    return {'source':source,'source_id':source_id,'cards':[{'kind':'member','id':1,'name':'test',
        'level':field(level,.99),'card_rank':field(rank,.97),'awake_count':field(awake,.96)}]}


class MergeTests(unittest.TestCase):
    def test_views_complement_and_duplicates_do_not_change_fields(self):
        scans=[observation('level.jpg','a',20,2),observation('training.jpg','b',rank=2,awake=3)]
        before=copy.deepcopy(scans);result=merge(scans)
        self.assertEqual(result['unique_count'],1)
        self.assertEqual([result['cards'][0][key]['value'] for key in ('level','card_rank','awake_count')],[20,2,3])
        self.assertEqual(result['cards'],merge(scans+scans)['cards'])
        self.assertEqual(scans,before)

    def test_same_filename_different_images_retain_sources_and_conflict(self):
        scans=[observation('image.jpg','a',20,2),observation('image.jpg','b',21,2)]
        card=merge(scans)['cards'][0]
        self.assertIsNone(card['level']['value'])
        self.assertEqual(card['source_ids'],['a','b'])
        self.assertEqual(set(card['conflicts']['level']),{20,21})
        self.assertEqual(card,merge(scans+scans)['cards'][0])

    def test_member_and_snap_ids_are_distinct(self):
        a=observation('member.jpg','a',10,1);b=observation('snap.jpg','b',10,1)
        b['cards'][0]['kind']='snap'
        self.assertEqual(merge([a,b])['unique_count'],2)

    def test_unknown_does_not_overwrite_a_visible_value(self):
        result=merge([observation('known.jpg','a',20,2),observation('hidden.jpg','b')])
        self.assertEqual(result['cards'][0]['level']['value'],20)
        self.assertIsNone(result['cards'][0]['awake_count']['value'])


if __name__=='__main__':unittest.main()
