"""BDON 1.0.1-25 card renderer: native prefab geometry + CLI-verified view rules.

Network-free. See evidence/ and SOURCES.md. PNGs retain transparent backgrounds.
Python 3.11+, Pillow, numpy. Run this file to regenerate the review artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
import copy
import argparse
import hashlib
import json
import math
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops, ImageOps

ROOT = Path(__file__).resolve().parent
UI = ROOT / 'game-ui'
ASSETS = ROOT / 'assets'
OUT = ROOT / 'components'
RESAMPLE = Image.Resampling.LANCZOS


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@lru_cache(maxsize=128)
def picture(path):
    return Image.open(path).convert('RGBA')


def rgba(color):
    return tuple(round(color.get(c, 1) * 255) for c in 'rgba')


def first_component(node, cls):
    return next((c for c in node['components'] if c.get('class') == cls), None)


def crop_sprites():
    """Unity coordinates are bottom-up; retain trimmed-sprite transparent margins."""
    sprites, textures = read(UI / 'sprites.json'), read(UI / 'textures.json')
    (UI / 'sprites').mkdir(exist_ok=True)
    for key, s in sprites.items():
        tex = picture(UI / textures[s['textureRef']])
        b, off, rect = s['textureRect'], s['textureRectOffset'], s['rect']
        im = Image.new('RGBA', (round(rect['width']), round(rect['height'])))
        crop = tex.crop(tuple(round(v) for v in (b['x'], tex.height-b['y']-b['height'], b['x']+b['width'], tex.height-b['y'])))
        im.alpha_composite(crop, (round(off['x']), round(rect['height']-off['y']-b['height'])))
        s['file'] = 'sprites/' + re.sub('[^A-Za-z0-9_.-]', '_', s['name']) + '.png'
        im.save(UI / s['file'])
    (UI / 'sprite-index.json').write_text(json.dumps(sprites, indent=2), encoding='utf-8')
    return sprites


def nine_slice(im, size, border, unit_scale=1):
    """Unity Image.Type.Sliced. Corners keep their source dimensions."""
    w, h = size
    left, bottom, right, top = (round(border[c]) for c in 'xyzw')
    if not any((left, bottom, right, top)):
        return im.resize(size, RESAMPLE)
    sx = [0, left, im.width-right, im.width]
    sy = [0, top, im.height-bottom, im.height]
    l, r, t, b = [round(v*unit_scale) for v in (left, right, top, bottom)]
    if l+r > w:
        l, r = round(w*l/max(l+r, 1)), round(w*r/max(l+r, 1))
    if t+b > h:
        t, b = round(h*t/max(t+b, 1)), round(h*b/max(t+b, 1))
    dx, dy = [0, l, w-r, w], [0, t, h-b, h]
    out = Image.new('RGBA', size)
    for y in range(3):
        for x in range(3):
            dw, dh = dx[x+1]-dx[x], dy[y+1]-dy[y]
            if dw > 0 and dh > 0:
                # Zero-width source center is legal in Unity's sliced sprites: sample
                # its UV line. Card frames have borders exactly half the texture size.
                ex = max(sx[x]+1, sx[x+1]); ey = max(sy[y]+1, sy[y+1])
                tile = im.crop((sx[x], sy[y], ex, ey)).resize((dw, dh), RESAMPLE)
                out.alpha_composite(tile, (dx[x], dy[y]))
    return out


class GameText:
    """Native VibeMO glyphs for card numbers; native FZ font for CJK labels.

    SDF coverage is rasterized on CPU. This is a static approximation of TMP's
    distance-field shader; glyph outlines and advances are taken from the game.
    """
    def __init__(self):
        self.doc = read(UI / 'fonts/VibeMOPro-Medium SDF.json')
        self.chars = {c['m_Unicode']: c['m_GlyphIndex'] for c in self.doc['m_CharacterTable']}
        self.glyphs = {g['m_Index']: g for g in self.doc['m_GlyphTable']}
        texture = next(v for k, v in read(UI / 'textures.json').items() if k.endswith(':-5829278353647440834'))
        self.atlas = picture(UI / texture).getchannel('A')
        self.native_font = str(UI / 'fonts/FZLTH_GB18030L2_R.font')

    def text(self, value, size, color=(255,255,255,255), shadow=True):
        value = str(value).strip()
        if not value:
            return Image.new('RGBA', (1,1))
        if any(ord(c) not in self.chars for c in value):
            font = ImageFont.truetype(self.native_font, max(1, round(size)))
            box = font.getbbox(value)
            im = Image.new('RGBA', (max(1,box[2]-box[0]+12), max(1,box[3]-box[1]+12)))
            ImageDraw.Draw(im).text((6-box[0],6-box[1]),value,font=font,fill=color,
                                   stroke_width=max(1,round(size*.055)) if shadow else 0,
                                   stroke_fill=(44,28,85,220))
            return im
        scale = size/self.doc['m_FaceInfo']['m_PointSize']
        width = sum(self.glyphs[self.chars[ord(c)]]['m_Metrics']['m_HorizontalAdvance']*scale for c in value)
        pad = max(3, round(size*.15))
        im = Image.new('RGBA', (max(1, math.ceil(width)+pad*2), math.ceil(size*1.35)+pad*2))
        baseline = pad + self.doc['m_FaceInfo']['m_AscentLine']*scale
        x = pad
        for ch in value:
            g = self.glyphs[self.chars[ord(ch)]]; m, b = g['m_Metrics'], g['m_GlyphRect']
            if m['m_Width'] > 0 and m['m_Height'] > 0:
                mask = self.atlas.crop((b['m_X'],self.atlas.height-b['m_Y']-b['m_Height'],b['m_X']+b['m_Width'],self.atlas.height-b['m_Y']))
                mask = mask.resize((max(1,round(b['m_Width']*scale)), max(1,round(b['m_Height']*scale))), RESAMPLE)
                # The atlas is an SDF: convert signed distance to antialiased coverage.
                coverage = np.clip((np.asarray(mask,dtype=float)/255-.5)*max(2,26*scale)+.5,0,1)
                mask = Image.fromarray(np.uint8(coverage*255))
                glyph = Image.new('RGBA',mask.size,color);glyph.putalpha(mask)
                im.alpha_composite(glyph,(round(x+m['m_HorizontalBearingX']*scale),round(baseline-m['m_HorizontalBearingY']*scale)))
            x += m['m_HorizontalAdvance']*scale
        box = im.getbbox()
        if box:
            im = im.crop((max(0,box[0]-pad),max(0,box[1]-pad),min(im.width,box[2]+pad),min(im.height,box[3]+pad)))
        if shadow:
            mask = im.getchannel('A').filter(ImageFilter.MaxFilter(max(3, round(size*.07)//2*2+1)))
            base = Image.new('RGBA', im.size, (40,17,98,230));base.putalpha(mask.point(lambda a: round(a*.85)))
            base.alpha_composite(im);im=base
        return im


@dataclass(frozen=True)
class CardState:
    asset_id: int = 51
    rarity: int = 4
    card_type: int = 4
    level: int | None = 60
    rank: int | None = 5
    awake_count: int = 5
    param: str = 'level'
    power: tuple[int,int,int] = (11248,9873,10625)
    event_bonus: int = 0
    event_disabled: bool = False
    selected: bool = False
    badge: bool = False
    type_bonus: bool = False
    show_band: bool = False
    band_id: int = 1
    empty: bool = False
    leader: bool = False

    def __post_init__(self):
        if self.rarity not in (2,3,4,10): raise ValueError('Unknown rarity')
        if self.card_type not in (1,2,3,4,5): raise ValueError('Unknown card type')
        if self.rank is not None and not 0 <= self.rank <= 5: raise ValueError('Rank must be 0..5')
        if self.param not in ('level','total','performance','technic','visual','hide','training'): raise ValueError('Unknown param')


class CardRenderer:
    def __init__(self):
        self.sprites = crop_sprites()
        self.named = {s['name']:s for s in self.sprites.values()}
        self.text = GameText()
        self.catalogs = {}
        def collect(v):
            if isinstance(v,dict):
                if '_gradientsByKey' in v or '_spriteMap' in v or '_spritesByKey' in v:
                    self.catalogs[(v.get('asset'),v.get('name'))] = v
                for x in v.values(): collect(x)
            elif isinstance(v,list):
                for x in v: collect(x)
        for f in UI.glob('UI*.json'):
            collect(read(f))

    def nodes(self, variant):
        source = {'member':'UIMemberCardListRowView','square':'UISimpleFormationItem','snap':'UISupportCardListRowView',
                  'formation':'UIFormationSlot','formation_snap':'UIFormationSlot'}[variant]
        nodes = read(UI / (source+'.json'))['nodes']
        kind = {'snap':'UISupportCard','formation':'UIFormationMemberCard',
                'formation_snap':'UIFormationSupportCard'}.get(variant,'UIMemberCard')
        start = next(i for i,n in enumerate(nodes) if first_component(n,kind))
        prefix = nodes[start]['path']
        # Use contiguous hierarchy, preserving duplicate sibling names and inactive templates.
        end = next((i for i in range(start+1,len(nodes)) if not nodes[i]['path'].startswith(prefix+'/')),len(nodes))
        return copy.deepcopy(nodes[start:end]), kind

    def gradient(self, image, grad):
        a = np.asarray(image).astype(float)/255
        height = image.height
        t = np.linspace(1,0,height)  # +90 degrees, Unity y-up.
        times=[grad[f'ctime{i}']/65535 for i in range(grad['m_NumColorKeys'])]
        cols=[grad[f'key{i}'] for i in range(grad['m_NumColorKeys'])]
        if grad['m_Mode']==1:
            ix=np.searchsorted(times,t,side='left').clip(0,len(times)-1)
            rgb=np.array([[cols[j][c] for c in 'rgb'] for j in ix])
        else:
            rgb=np.stack([np.interp(t,times,[c[channel] for c in cols]) for channel in 'rgb'],axis=-1)
        a[:,:,:3] *= rgb[:,None,:]
        atimes=[grad[f'atime{i}']/65535 for i in range(grad['m_NumAlphaKeys'])]
        avals=[grad[f'key{i}']['a'] for i in range(grad['m_NumAlphaKeys'])]
        a[:,:,3] *= np.interp(t,atimes,avals)[:,None]
        return Image.fromarray(np.uint8(np.clip(a*255,0,255)))

    def render(self, state=CardState(), variant='square', scale=2, only=None):
        """only: cumulative review stages ('art', 'frame', 'type', 'rank', 'param')."""
        nodes, kind = self.nodes(variant)
        refs = first_component(nodes[0],kind)
        root = nodes[0]['path']
        size=nodes[0]['rect']['m_SizeDelta'];w,h=size['x'],size['y']
        pad=32
        image=Image.new('RGBA',(round((w+pad*2)*scale),round((h+pad*2)*scale)))
        active={};text_values={};sprite_overrides={};gradient_overrides={}
        ref=lambda key: (refs.get(key) or {}).get('gameObject')
        def set_active(key,val):
            if ref(key):active[ref(key)]=val
        set_active('_cardRoot',not state.empty);set_active('_emptyRoot',state.empty)
        set_active('_selected',state.selected);set_active('_batch',state.badge)
        set_active('_eventBonusLabel',state.event_bonus>0 and not state.event_disabled)
        set_active('_lvGroup',state.param=='level' and state.level is not None)
        set_active('_statusGroup',state.param in ('total','performance','technic','visual'))
        set_active('_trainingGroup',state.param=='training')
        set_active('_cardRankIcon',state.param!='hide' and state.rank is not None)
        set_active('_bandLogoAddrImage',state.show_band)
        if variant=='formation_snap':
            set_active('_highlightFrame',state.selected)
            set_active('_highlightFrameInEmpty',False)
            if ref('_levelText'):text_values[ref('_levelText')]='Lv.'+str(state.level) if state.level is not None else ''
        source_images={}
        if variant=='formation':
            if state.rarity!=4 and not state.empty:
                raise ValueError('Formation artwork resources currently cover SSR cards only')
            set_active('_frontContent',True);set_active('_backContent',False)
            set_active('_frontThumbRoot',state.rarity==4);set_active('_backThumbRoot',state.rarity!=4)
            set_active('_highlightFrame',state.selected)
            set_active('_leaderLabel',state.leader and not state.empty)
            if ref('_levelText'):text_values[ref('_levelText')]='Lv.'+str(state.level) if state.level is not None else ''
            for field,part in [('_frontThumbnailAddrImage','character'),('_frontBackgroundAddrImage','background')]:
                source_images[ref(field)]=ASSETS/f'member-{state.asset_id}-{part}-formation.webp'
            if state.show_band:source_images[ref('_bandLogoAddrImage')]=ASSETS/f'band-{state.band_id}-white.webp'
            for node in nodes:
                if '/InvalidBandLabel' in node['path']:active[node['path']]=False
        if ref('_lvValueText'):text_values[ref('_lvValueText')]=str(state.level)
        if ref('_trainingValueText'):text_values[ref('_trainingValueText')]=str(state.awake_count)
        if ref('_statusLabelText'):
            text_values[ref('_statusLabelText')]={'total':'综合力','performance':'Pfm.','technic':'Tec.','visual':'Vis.'}.get(state.param,'')
            text_values[ref('_statusValueText')]=str({'total':sum(state.power),'performance':state.power[0],'technic':state.power[1],'visual':state.power[2]}.get(state.param,''))
        rank_node=ref('_cardRankIcon')
        if rank_node:
            sprite_overrides[rank_node]=self.named[('snaplimit_' if variant in ('snap','formation_snap') else 'CardRank')+str(state.rank or 0)]
        # Change() resolves the original SpriteCatalog mapping, not a guessed color ordering.
        for node in nodes:
            c=first_component(node,'UICardTypeIcon')
            if c:
                cat=c['_spriteCatalog'];cat=self.catalogs.get((cat.get('asset'),cat.get('name')),cat)
                mapping=cat.get('_spriteMap',cat.get('_spritesByKey',{}))
                rows=mapping.get('_list',[])
                match=next((v['Value'] for v in rows if v['Key']==state.card_type),None)
                if match and 'spriteRef'in match:sprite_overrides[node['path']]=self.sprites[match['spriteRef']]
                else:
                    raise ValueError(f'Missing serialized CardType sprite mapping: {state.card_type}')
                # 0x64db43c: strb wzr, [x19,#66]. Link bonus is disabled in this build.
                active[node['path']+'/LinkBonusIcon']=False
                for color,i in [('Red',1),('Blue',2),('Green',3),('Yellow',4),('Purple',5)]:
                    active[node['path']+'/Root/'+color+'Glow']=state.type_bonus and state.card_type==i
            frame=first_component(node,'UIMemberCardFormationFrame') or first_component(node,'UISupportCardFormationFrame')
            if frame and '/EmptyRoot/' not in node['path']:
                cat=frame['_gradientCatalog'];cat=self.catalogs.get((cat.get('asset'),cat.get('name')),cat)
                grad=next((v['Value'] for v in cat.get('_gradientsByKey',{}).get('_list',[]) if v['Key']==state.rarity),None)
                if grad:gradient_overrides[frame['_gradientImage']['gameObject']]=grad
            if first_component(node,'UIEventBonusLabelOnCard'):
                for child in nodes:
                    if child['path'].startswith(node['path']+'/') and first_component(child,'TextMeshProUGUI'):
                        text_values[child['path']]=str(state.event_bonus)+'%'
        stage_order={'art':0,'frame':1,'type':2,'rank':3,'param':4}
        stage=stage_order.get(only,99)
        stack=[]
        for index,node in enumerate(nodes):
            path=node['path'];depth=path.count('/')-root.count('/')
            while len(stack)>depth:stack.pop()
            if index==0:
                box=(0.,0.,w,h);world_scale=(1.,1.);visible=True
            else:
                parent=stack[-1];px,py,pw,ph=parent['box'];psx,psy=parent['scale']
                rect=node.get('rect');visible=parent['visible'] and active.get(path,node['active'])
                if not rect:stack.append(parent);continue
                amin,amax,pos,delta,pivot=[rect[k] for k in ('m_AnchorMin','m_AnchorMax','m_AnchoredPosition','m_SizeDelta','m_Pivot')]
                nw=pw*(amax['x']-amin['x'])+delta['x']*psx;nh=ph*(amax['y']-amin['y'])+delta['y']*psy
                ratio=first_component(node,'AspectRatioFitter')
                ratio_value=ratio['m_AspectRatio'] if ratio else 1
                if ratio and path in source_images:
                    art=picture(source_images[path]);ratio_value=art.width/art.height
                if ratio and ratio['m_AspectMode']==1:nh=nw/ratio_value
                elif ratio and ratio['m_AspectMode']==2:nw=nh*ratio_value
                elif ratio and ratio['m_AspectMode'] in (3,4):
                    fit=min if ratio['m_AspectMode']==3 else max
                    nw=fit(pw,ph*ratio_value);nh=nw/ratio_value
                fitter=first_component(node,'ContentSizeFitter')
                text_comp=first_component(node,'TextMeshProUGUI')
                if fitter and text_comp and fitter.get('m_HorizontalFit')==2:
                    content=text_values.get(path,text_comp['m_text']).strip()
                    preferred=self.text.text(content,text_comp['m_fontSize']*psx,shadow=False)
                    nw=max(1,preferred.width-6)
                nx=px+pw*(amin['x']+(amax['x']-amin['x'])*pivot['x'])+pos['x']*psx-nw*pivot['x']
                ny=py+ph*(1-amin['y']-(amax['y']-amin['y'])*pivot['y'])-pos['y']*psy-nh*(1-pivot['y'])
                if ratio and ratio['m_AspectMode'] in (3,4):
                    nx=px+(pw-nw)/2;ny=py+(ph-nh)/2
                sx,sy=node['localScale']['x'],node['localScale']['y']
                nx+=nw*pivot['x']*(1-sx);ny+=nh*(1-pivot['y'])*(1-sy)
                box=(nx,ny,nw*sx,nh*sy);world_scale=(psx*sx,psy*sy)
            # Loading, glow animation layers, raycast targets are not persistent static card content.
            if 'UIAssetsLoading'in path or '/Button/'in path or 'ArrowGlow'in path:visible=False
            if '/ParamRoot'in path and stage<4:visible=False
            if '/UICardRank'in path or '/UISnapLimit'in path:
                if stage<3:visible=False
            if '/UICardTypeIcon'in path and stage<2:visible=False
            if ('FormationFrame'in path or '/Shadow'in path) and stage<1:visible=False
            stack.append({'box':box,'scale':world_scale,'visible':visible,'path':path,
                          'mask':bool(first_component(node,'Mask') or first_component(node,'RectMask2D'))})
            if not visible:continue
            x,y,bw,bh=box
            if bw<=0 or bh<=0:continue
            draw_size=(max(1,round(bw*scale)),max(1,round(bh*scale)))
            comp=first_component(node,'Image');layer=None
            art_path=ref('_thumbnailAddrImage')
            if comp and path.rsplit('/',1)[0] in source_images:
                layer=picture(source_images[path.rsplit('/',1)[0]]).resize(draw_size,RESAMPLE)
                # Front character may extend beyond its frame, but stays inside
                # the native FrontThumRoot mask; background has its own inner mask.
                masks=[s for s in stack if s.get('mask')]
                if masks:
                    mx,my,mw,mh=masks[-1]['box']
                    left=max(0,round((mx-x)*scale));top=max(0,round((my-y)*scale))
                    right=min(layer.width,round((mx+mw-x)*scale));bottom=min(layer.height,round((my+mh-y)*scale))
                    if right>left and bottom>top:
                        layer=layer.crop((left,top,right,bottom));x+=left/scale;y+=top/scale
                    else:layer=None
            elif comp and art_path and path==art_path+'/Image':
                if variant in ('snap','formation_snap'):asset=f'snap-{state.asset_id}.webp'
                else:asset=f'member-{state.asset_id}'+('-square' if refs.get('_thumbnailType')==1 else '')+'.webp'
                if variant in ('member','snap','formation_snap'):
                    clip=next(s['box'] for s in reversed(stack) if s.get('path','').endswith(('/Thumbnail','/ThumbnailRoot')))
                    x,y,bw,bh=clip;draw_size=(round(bw*scale),round(bh*scale))
                    layer=ImageOps.fit(picture(ASSETS/asset),draw_size,RESAMPLE)
                else:layer=picture(ASSETS/asset).resize(draw_size,RESAMPLE)
            elif comp and ref('_bandLogoAddrImage') and path==ref('_bandLogoAddrImage')+'/Image':
                layer=ImageOps.contain(picture(ASSETS/f'band-{state.band_id}.webp'),draw_size,RESAMPLE)
            elif comp:
                spr=sprite_overrides.get(path)
                if spr is None and comp.get('m_Sprite'):spr=self.sprites.get(comp['m_Sprite'].get('spriteRef'))
                if spr:
                    source=picture(UI/spr['file'])
                    layer=nine_slice(source,draw_size,spr['border'],scale*world_scale[0]*100/spr['pixelsPerUnit']) if comp.get('m_Type')==1 else source.resize(draw_size,RESAMPLE)
                    color=rgba(comp.get('m_Color',{}));layer=ImageChops.multiply(layer,Image.new('RGBA',layer.size,color))
                # Mask graphics have showMaskGraphic=False and only clip their children.
                mask=first_component(node,'Mask')
                if mask and not mask.get('m_ShowMaskGraphic',True):layer=None
            if layer:
                flip=first_component(node,'UIFlipImage')
                if flip and flip['_isFlipX']:layer=ImageOps.mirror(layer)
                if flip and flip['_isFlipY']:layer=ImageOps.flip(layer)
                gradient=first_component(node,'UIGradientImage')
                if gradient:layer=self.gradient(layer,gradient_overrides.get(path,gradient['_gradient']))
                image.alpha_composite(layer,(round((x+pad)*scale),round((y+pad)*scale)))
            text=first_component(node,'TextMeshProUGUI')
            if text:
                value=text_values.get(path,text['m_text']).strip()
                if value:
                    font_size=text['m_fontSize']*world_scale[1]*scale
                    tile=self.text.text(value,font_size,rgba(text.get('m_fontColor',{})))
                    align=text.get('m_HorizontalAlignment',1)
                    tx=x*scale+(draw_size[0]-tile.width)/2 if align==2 else x*scale-3
                    if align==4:tx=x*scale+draw_size[0]-tile.width
                    ty=y*scale+(draw_size[1]-tile.height)/2
                    image.alpha_composite(tile,(round(tx+pad*scale),round(ty+pad*scale)))
        return image

    def formation_slot(self, member, support=None, scale=1):
        """UIFormationSlot: 332×600, Snap 240×135 at (46,383.5).

        These coordinates are read from the prefab each time, including the
        SupportCardHolder center anchor and y=-151 offset.
        """
        im=self.render(member,'formation',scale)
        if support is not None:
            nodes=read(UI/'UIFormationSlot.json')['nodes']
            root=nodes[0]['rect']['m_SizeDelta']
            holder=next(n for n in nodes if n['path']=='UIFormationSlot/Contents/SupportCardHolder')['rect']
            sz,pos=holder['m_SizeDelta'],holder['m_AnchoredPosition']
            x=(root['x']-sz['x'])/2+pos['x'];y=(root['y']-sz['y'])/2-pos['y']
            im.alpha_composite(self.render(support,'formation_snap',scale),(round(x*scale),round(y*scale)))
        return im


def board_bg(w,h):
    # Quiet review surface; card internals are all game-derived.
    im=Image.new('RGBA',(w,h),'#F2F1F8')
    return im


def main():
    OUT.mkdir(exist_ok=True)
    renderer=CardRenderer()
    # Native compact cards, with one state shown per card. No explanatory copy on images.
    states=[CardState(asset_id=1,rarity=2,card_type=5,rank=1,level=20),
            CardState(asset_id=26,rarity=3,card_type=2,rank=3,level=40),
            CardState(),CardState(selected=True),CardState(event_bonus=20)]
    board=board_bg(1400,940)
    for i,state in enumerate(states):
        im=renderer.render(state,'square',scale=1.35)
        board.alpha_composite(im,(i*274+12,44))
        im.save(OUT/f'member-state-{i}.png')
    for i,param in enumerate(['level','total','performance','technic','visual']):
        state=replace(CardState(),param=param,asset_id=51+i,card_type=[4,5,3,1,2][i])
        im=renderer.render(state,'square',scale=1.35)
        board.alpha_composite(im,(i*274+12,330))
        im.save(OUT/f'member-{param}.png')
    for i in range(3):
        state=CardState(asset_id=51+i,card_type=[1,2,3][i],rank=[1,3,5][i],level=50)
        im=renderer.render(state,'snap',scale=1.10)
        board.alpha_composite(im,(i*450+35,640))
        im.save(OUT/f'snap-state-{i}.png')
    board.convert('RGB').save(ROOT/'00-game-components.png',optimize=True)
    board.convert('RGB').save(ROOT/'design-board.png',optimize=True)
    # Cumulative card layers, all based on the same card state and native geometry.
    layers=board_bg(1400,330)
    for i,stage in enumerate(['art','frame','type','rank','param']):
        im=renderer.render(CardState(),'square',scale=1.35,only=stage)
        im.save(OUT/f'layer-{stage}.png');layers.alpha_composite(im,(i*274+12,10))
    layers.convert('RGB').save(ROOT/'01-game-card-layers.png',optimize=True)
    # Five-slot bot composition uses the same native card and snap components.
    deck=board_bg(1540,690)
    logo=ImageOps.contain(picture(ASSETS/'band-1.webp'),(220,95),RESAMPLE);deck.alpha_composite(logo,(52,27))
    for i in range(5):
        state=CardState(asset_id=51+i,card_type=[4,5,3,1,2][i],rank=[5,4,3,5,2][i],level=60)
        im=renderer.render(state,'square',scale=1.40);deck.alpha_composite(im,(i*296+20,135))
        snap=renderer.render(CardState(asset_id=[51,52,51,52,51][i],card_type=[1,2,1,2,1][i],rank=[5,4,3,5,2][i],level=50),'snap',scale=.80)
        deck.alpha_composite(snap,(i*296+23,435))
    deck.convert('RGB').save(ROOT/'02-game-deck.png',optimize=True)
    for param in ['hide','training']:
        renderer.render(replace(CardState(),param=param),'square').save(OUT/f'member-{param}.png')
    renderer.render(replace(CardState(),empty=True),'square').save(OUT/'member-empty.png')
    renderer.render(replace(CardState(),badge=True),'square').save(OUT/'member-badge.png')
    renderer.render(CardState(), 'member').save(OUT/'member-list-native.png')
    ranks=board_bg(1590,650)
    for i in range(6):
        member=renderer.render(replace(CardState(),rank=i),'square',scale=1.25)
        ranks.alpha_composite(member,(i*257+3,10))
        snap=renderer.render(CardState(asset_id=51,card_type=1,rank=i,level=50),'snap',scale=.66)
        ranks.alpha_composite(snap,(i*257+16,345))
    ranks.convert('RGB').save(ROOT/'03-game-card-ranks.png',optimize=True)
    # Keep provenance separate from product imagery.
    manifest={'version':'0.2.0','game':'1.0.1-25','inputs':'game-ui/ + assets/','render_mode':'static CPU',
              'data_kind':'synthetic card growth states; public MasterData identities',
              'files':[]}
    for f in sorted(OUT.glob('*.png')):
        manifest['files'].append({'file':f.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'size':list(Image.open(f).size)})
    (ROOT/'render-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Generated',len(manifest['files']),'components and 4 boards.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state',type=Path,help='CardState JSON; omitted: regenerate component boards')
    parser.add_argument('--variant',choices=['square','member','snap','formation','formation_snap'],default='square')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--scale',type=float,default=2)
    args=parser.parse_args()
    if args.state:
        if not args.out:parser.error('--out is required with --state')
        if not 0<args.scale<=8:parser.error('--scale must be > 0 and <= 8')
        args.out.parent.mkdir(parents=True,exist_ok=True)
        CardRenderer().render(CardState(**read(args.state)),args.variant,args.scale).save(args.out)
    else:main()
