"""Integration check: remove trained-out identities, add their art/catalog, no retraining."""
import argparse
import hashlib
import shutil
from pathlib import Path

import cv2
import numpy as np

from bdon_vision.assets import read,write,build_index
from bdon_vision.engine import Engine
from bdon_vision.inference import Gallery


def verify(data,output,member_image=None,snap_image=None,member_id=7,snap_id=70):
    data=Path(data);output=Path(output);target=output/'gallery-fixture'
    paths={'member':Path(member_image) if member_image else data/'examples/member.jpg',
           'snap':Path(snap_image) if snap_image else data/'examples/snap.jpg'}
    images={name:cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR) if path.is_file() else None for name,path in paths.items()}
    if any(image is None for image in images.values()):
        raise ValueError('Provide local --member-image and --snap-image; game screenshots are not distributed.')
    cards=read(data/'catalog.json')['cards'];held={('member',member_id),('snap',snap_id)}
    if not held.issubset({(c['kind'],c['id']) for c in cards}):
        raise ValueError('Selected test identities must exist in the supplied catalog')
    target.mkdir(parents=True,exist_ok=False)
    old=[c for c in cards if (c['kind'],c['id']) not in held]
    for path in [data/'models/encoder.onnx',data/'models/fields.onnx']:
        dest=target/path.relative_to(data);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    for card in cards:
        path=data/card.get('match_file',card['file']);dest=target/path.relative_to(data)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    shutil.copytree(data/'native/game-ui/sprites',target/'native/game-ui/sprites')
    model_hash=lambda:hashlib.sha256((target/'models/encoder.onnx').read_bytes()).hexdigest()
    before_hash=model_hash()
    write(target/'catalog.json',{'cards':old});build_index(target);Gallery(target,read(target/'catalog.json')['cards'])
    before=Engine(target);old_counts=len(before.cards)
    before_ids={name:sorted((c['kind'],c['id']) for c in before.identify(im)) for name,im in images.items()}
    # The updater rebuilds both RootSIFT and neural lookup. It does not train.
    write(target/'catalog.json',{'cards':cards});build_index(target);Gallery(target,read(target/'catalog.json')['cards'])
    after=Engine(target)
    after_ids={name:sorted((c['kind'],c['id']) for c in after.fill_grid(im,after.identify(im))) for name,im in images.items()}
    restored=all(identity in after_ids[identity[0]] for identity in held)
    report={'removed_ids':sorted(held),'old_catalog_size':old_counts,'new_catalog_size':len(after.cards),
        'encoder_sha256_before':before_hash,'encoder_sha256_after':model_hash(),
        'model_unchanged':before_hash==model_hash(),'new_cards_recognized':restored,
        'before_ids':before_ids,'after_ids':after_ids,
        'scope':'User-provided local test images; incremental update integration, not an independent accuracy estimate'}
    write(output/'gallery-update.json',report);print(report)
    if not report['model_unchanged'] or not restored:raise RuntimeError('Gallery update regression')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--member-image',type=Path);p.add_argument('--snap-image',type=Path)
    p.add_argument('--member-id',type=int,default=7);p.add_argument('--snap-id',type=int,default=70)
    a=p.parse_args();verify(a.data,a.output,a.member_image,a.snap_image,a.member_id,a.snap_id)
