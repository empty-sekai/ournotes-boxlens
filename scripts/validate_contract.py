"""Validate an exported artifact against the public v1 contract (test extra)."""
import argparse
import json
from pathlib import Path

from jsonschema import Draft202012Validator


def validate(kind,path):
    schema=json.loads((Path(__file__).resolve().parents[1]/'schemas'/(kind+'.schema.json')).read_text(encoding='utf8'))
    Draft202012Validator.check_schema(schema)
    payload=json.loads(Path(path).read_text(encoding='utf8'))
    Draft202012Validator(schema).validate(payload)
    print(f'{kind}: valid — {path}')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['box','observations','scan-request','scan-response','error'])
    p.add_argument('file',type=Path);a=p.parse_args();validate(a.kind,a.file)
