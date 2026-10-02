"""Regressions for the independently reproduced HTTP duplicate-field bypass."""
from pathlib import Path
import pytest
from bdon_vision.deck.common import read, strict_json, load_deck
from bdon_vision.deck.recommend import write_request

@pytest.mark.parametrize('raw,key',[
    ('{"kind":"power","kind":"power"}','kind'),
    ('{"execution":{"kind":"power","kind":"power"}}','kind'),
    ('{"execution":{"play":{"deltaTimes":[0.016],"deltaTimes":[0.017]}}}','deltaTimes'),
    ('{"cases":[{"request":{"kind":"power","kind":"live"}}]}','kind'),
    ('{"kind":1,"k\\u0069nd":2}','kind'),
])
def test_duplicate_fields_rejected_recursively(raw,key,tmp_path):
    with pytest.raises(ValueError,match='Duplicate JSON field: '+key):strict_json(raw)
    path=tmp_path/'matrix.json';path.write_text(raw,encoding='utf-8')
    with pytest.raises(ValueError,match='Duplicate JSON field'):read(path)

@pytest.mark.parametrize('token',['NaN','Infinity','-Infinity','1e999'])
def test_invalid_non_finite_numbers_rejected(token):
    with pytest.raises(ValueError):strict_json('{"deltaTimes":['+token+']}')

def test_uploaded_request_original_bytes_and_large_integer(tmp_path):
    raw=' {"ticks":638999999999999999,"execution":{"play":{"deltaTimes":[0.016666668,1.6666668e-2,0.017000000]}}}\n'
    parsed=strict_json(raw)
    assert parsed['ticks']==638999999999999999
    path=tmp_path/'actual-binary-request.json'
    write_request({'request':parsed,'requestJson':raw},path)
    assert path.read_bytes()==raw.encode('utf-8')

def test_raw_request_binding_or_duplicate_cannot_be_written(tmp_path):
    path=tmp_path/'not-written.json'
    with pytest.raises(ValueError,match='binding mismatch'):
        write_request({'request':{'kind':'power'},'requestJson':'{"kind":"live"}'},path)
    with pytest.raises(ValueError,match='Duplicate JSON field'):
        write_request({'request':{'kind':'power'},'requestJson':'{"kind":"power","kind":"power"}'},path)
    assert not path.exists()

def test_actual_deck_import_rejects_duplicate_fields(tmp_path):
    path=tmp_path/'deck.json'
    path.write_text('{"format":"nnnotes.deck-data/1","master":{"x":1,"x":2}}',encoding='utf-8')
    with pytest.raises(ValueError,match='Duplicate JSON field: x'):load_deck(path)

@pytest.mark.parametrize('raw',[b'{"name":"\xff"}',b'{"name":"\xc3("}'])
def test_malformed_utf8_rejected(raw):
    with pytest.raises(ValueError):strict_json(raw)

@pytest.mark.parametrize('raw',['{"kind":','{"kind":"power"} trailing'])
def test_malformed_json_rejected(raw):
    with pytest.raises(ValueError):strict_json(raw)
