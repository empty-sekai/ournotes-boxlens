"""Extract the shipped card prefab, sprites and layout; no account/network access."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import zipfile

from nnnotes.catalog import Catalog, APK_CATALOG
from nnnotes.export import Exporter
from nnnotes.player import PlayerData
from nnnotes.config import Config
from nnnotes.cli import bundle_key

KEYS = [
    'EmbUI/Prefab/UIMemberCardListWidget',
    'EmbUI/Prefab/UISupportCardListWidget',
    'EmbUI/Prefab/Parts/Formation/UIFormationSlot',
    'EmbUI/Prefab/Parts/Formation/UISimpleFormationItem',
    'EmbUI/Prefab/Parts/Common/UICardDetailDialogCell',
    'EmbUI/Prefab/Parts/Chat/UIPlayerMemberCard_Chat',
    'EmbUI/Prefab/FixParts/Atoms/CardSelectedHighlightFrame/CardSelectedHighlightFrame',
    'EmbUI/Prefab/Parts/Circle/UICircleInformationDialogListItem',
    'EmbUI/Prefab/UICircleInformationDialogWidget',
    'EmbUI/Prefab/UIPlayerProfileTopWidget',
    'EmbUI/Prefab/UIPlayerProfileBackgroundWidget',
]

def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('apk','config','cache','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    OUT=args.output;OUT.mkdir(parents=True,exist_ok=True)
    apk=args.apk
    cfg=Config.load(args.config)
    with zipfile.ZipFile(apk) as z:
        cat = Catalog(z.read(APK_CATALOG), args.cache, bundle_key=lambda: bundle_key(cfg))
    cat.apk = apk
    player = PlayerData(apk)
    ex = Exporter(cat, OUT, player=player, textures='deferred',
                  stub_assets=('TMP_FontAsset', 'TMP_SpriteAsset', 'TMP_StyleSheet'))
    for key in KEYS:
        doc = ex.prefab(key)
        (OUT / (key.rsplit('/', 1)[-1] + '.json')).write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding='utf-8')
        print(key, len(doc['nodes']), flush=True)
        if key.endswith('CardListWidget'):
            env, graph = ex.load(key)
            for pid, go in graph.go.items():
                if go['m_Name'] in ('UIMemberCardListRowView', 'UISupportCardListRowView'):
                    row = {'nodes': ex.hierarchy(env, graph, graph.tf_of_go[pid])}
                    (OUT / (go['m_Name'] + '.json')).write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding='utf-8')
                    print(go['m_Name'], len(row['nodes']), flush=True)
    sprites = {k: {**{a: b for a, b in v.items() if a != 'tex'},
                    'textureRef': f"{v['tex'].assets_file.name}:{v['tex'].path_id}"}
               for k, v in ex.sprite_recs.items()}
    (OUT / 'sprites.json').write_text(json.dumps(sprites, ensure_ascii=False, indent=2), encoding='utf-8')
    texdir = OUT / 'textures'
    texdir.mkdir(exist_ok=True)
    manifest = {}
    for i, (key, obj) in enumerate(ex.tex_objs.items()):
        data = obj.read()
        path = f'textures/{i:03d}-{data.m_Name}.png'
        data.image.save(OUT / path)
        manifest[key] = path
    (OUT / 'textures.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    fontdir = OUT / 'fonts'
    fontdir.mkdir(exist_ok=True)
    for obj in ex._objects.values():
        if obj.type.name == 'Font':
            font = obj.read()
            if font.m_FontData:
                (fontdir / (font.m_Name + '.font')).write_bytes(bytes(font.m_FontData))
                print('font', font.m_Name, len(font.m_FontData), flush=True)
        elif obj.type.name == 'MonoBehaviour' and ex.script_class(obj) == 'TMP_FontAsset':
            tt = obj.read_typetree()
            if tt.get('m_Name', '').startswith('Vibe'):
                (fontdir / (tt['m_Name'] + '.json')).write_text(json.dumps(tt, indent=2), encoding='utf-8')
    (OUT/'extraction.json').write_text(json.dumps({'schema':'ournotes-boxlens.ui-extract/1','apk_sha256':hashlib.sha256(apk.read_bytes()).hexdigest(),'keys':KEYS},indent=2),encoding='utf-8')
    print('textures', len(manifest), 'sprites', len(ex.sprite_recs), flush=True)

if __name__ == '__main__':
    main()
