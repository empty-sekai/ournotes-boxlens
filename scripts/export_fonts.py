"""Export the installed client's actual localized card glyphs, without credentials."""
import argparse
import hashlib
import json
import math
import zipfile
import sys
from pathlib import Path

from nnnotes.export import Exporter
from nnnotes import textstyle,tmpfont,languages
from nnnotes.catalog import Catalog,APK_CATALOG
from nnnotes.player import PlayerData
from nnnotes import cache
from nnnotes.config import Config
from nnnotes.cli import bundle_key
from PIL import Image


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--apk',type=Path,required=True)
    p.add_argument('--master-text',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True,help='Local nnnotes decryption configuration; never exported')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.apk) as apk:raw=apk.read(APK_CATALOG)
    config=Config.load(a.config)
    catalog=Catalog(raw,a.cache,apk=a.apk,bundle_key=lambda:bundle_key(config))
    player=PlayerData(a.apk)
    cache.configure(memory_mb=128,closures=2,closure_mb=256)
    ex=Exporter(catalog,a.output,player=player,textures='deferred',
                stub_assets=('TMP_FontAsset','TMP_SpriteAsset','TMP_StyleSheet'))
    fonts=tmpfont.FontSet(ex);rules=textstyle.language_fonts(player)
    master=json.loads(a.master_text.read_text(encoding='utf-8'))['_allData']
    text={r['_id']:r for r in master}
    charset=set(map(ord,'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.% -'))
    labels={}
    for locale,(mode,col) in languages.LANGUAGES.items():
        labels[locale]={key:text.get(key,{}).get(col,'') for key in ['SpecialTraining','ui_profile_info_total_power']}
        for value in labels[locale].values():charset.update(map(ord,value))
    out={'schema':'bdon-card-fonts/1','client_apk_sha256':hashlib.sha256(a.apk.read_bytes()).hexdigest(),
         'language_rules':rules,'locales':{},'fonts':{},'coverage':{},'labels':labels}
    atlas_cache={};generators={}
    names=sorted({n for v in rules['languages'].values() for n in v['fontNames']})
    for name in names:
        primary=fonts.by_key(f'EmbFont/{name}/{name} SDF')
        mat=ex.material(ex.key_object(f'EmbFont/{name}/{name} - CardParam'))
        record={'face':primary.tt['m_FaceInfo'],'material':mat,'characters':{},'source_font':primary.source_font,
                'normalSpacingOffset':primary.tt['normalSpacingOffset'],'fallbacks':[f.name for f in fonts.fallbacks(primary)]}
        missing=[]
        for cp in sorted(charset):
            match=fonts.lookup(primary,cp)
            if match is None:missing.append(chr(cp));continue
            kind,font=match
            if kind=='synthesized':continue
            padding=math.ceil(mat['floats']['_GradientScale'])+1
            if kind=='baked':
                char=font.chars[cp];glyph=font.glyphs[char['m_GlyphIndex']]
                rect=glyph['m_GlyphRect'];metrics=glyph['m_Metrics']
                key=(font.name,glyph['m_AtlasIndex'])
                if key not in atlas_cache:atlas_cache[key]=font.atlases[glyph['m_AtlasIndex']].read().image.convert('RGBA').getchannel('A')
                atlas=atlas_cache[key]
                cell=atlas.crop((rect['m_X']-padding,atlas.height-rect['m_Y']-rect['m_Height']-padding,
                    rect['m_X']+rect['m_Width']+padding,atlas.height-rect['m_Y']+padding))
                width,height=rect['m_Width'],rect['m_Height']
                glyph_scale=glyph['m_Scale']*char['m_Scale']
            else:
                if font.name not in generators:generators[font.name]=tmpfont.RuntimeGlyphs(font)
                gen=generators[font.name];metrics,(_,_,width,height),pixels=gen.sdf(gen.glyph_index(cp))
                padding=gen.pad;cell=Image.fromarray(pixels,'L');glyph_scale=1.
            directory=a.output/name;directory.mkdir(exist_ok=True)
            path=directory/f'{cp:06x}.png';cell.save(path)
            record['characters'][str(cp)]={'file':path.relative_to(a.output).as_posix(),'metrics':metrics,
                'width':width,'height':height,'padding':padding,'scale':glyph_scale,'source_kind':kind,
                'source_font':font.name,'source_point_size':font.tt['m_FaceInfo']['m_PointSize'],
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        out['fonts'][name]=record;out['coverage'][name]={'exported':len(record['characters']),'missing':missing}
    for locale,(mode,col) in languages.LANGUAGES.items():
        out['locales'][locale]={'mode':mode,'column':col,'primary':rules['languages'][mode]['fontNames'][0],
                              'number':rules['languages'][mode]['fontNames'][1],
                              'line_spacing':textstyle.LANGUAGE_LINE_SPACING[mode]}
    (a.output/'manifest.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'fonts':out['coverage'],'locales':out['locales']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
