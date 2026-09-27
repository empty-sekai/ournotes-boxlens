"""Assemble explicit resource exports into a portable BoxLens data directory."""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

from bdon_vision.assets import prepare,read,write,native_module


def assemble(master,ui,fonts,art,output,offline=False):
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Output must be empty to avoid mixing resource versions')
    output.mkdir(parents=True,exist_ok=True)
    native=output/'native';native.mkdir()
    (output/'master').mkdir();(native/'assets').mkdir()
    required={'MasterMemberCard.json','MasterSupportCard.json','MasterText.json',
        'MasterMemberCardLevelLimit.json','MasterMemberCardRank.json','MasterSupportCardRank.json'}
    available={p.name for p in master.glob('*.json')}
    if required-available:raise ValueError(f'Missing master tables: {sorted(required-available)}')
    for path in master.glob('Master*.json'):
        if 'Card' in path.name or path.name in ('MasterText.json','MasterCharacter.json'):
            shutil.copy2(path,output/'master'/path.name)
    shutil.copytree(ui,native/'game-ui',ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.py'))
    shutil.copytree(fonts,native/'card-fonts',ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.py'))
    if art:
        for path in art.glob('*.webp'):
            if re.fullmatch(r'(member|snap)-[0-9]+(-square)?\.webp',path.name):shutil.copy2(path,native/'assets'/path.name)
    shutil.copy2(Path(__file__).resolve().parents[1]/'bdon_vision/native_renderer.py',native/'render_game_components.py')
    native_module(output).CardRenderer()
    prepare(output,allow_download=not offline)
    catalog=read(output/'catalog.json')
    inputs={}
    for folder in [output/'master',native/'game-ui',native/'card-fonts',native/'assets']:
        for path in sorted(folder.rglob('*')):
            if path.is_file():inputs[path.relative_to(output).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    write(output/'provenance.json',{'schema':'ournotes-boxlens.resources/1','files':inputs,
        'apk_sha256':read(native/'card-fonts/manifest.json')['client_apk_sha256'],
        'renderer_sha256':hashlib.sha256((native/'render_game_components.py').read_bytes()).hexdigest(),
        'scope':'Explicit user-provided resource exports; approximate static TMP rendering; no account/config files copied'})
    if catalog['unavailable']:raise RuntimeError('Resource pack incomplete; inspect catalog.json unavailable entries')
    print(json.dumps({'cards':len(catalog['cards']),'hashed_resource_files':len(inputs),'complete':True}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('master-dir','ui-dir','font-dir','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--art-dir',type=Path)
    parser.add_argument('--offline',action='store_true',help='Require all artwork locally; never fetch missing files')
    args=parser.parse_args();assemble(args.master_dir,args.ui_dir,args.font_dir,args.art_dir,args.output,args.offline)
