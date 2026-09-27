"""Render game SDF glyphs including atlas padding and the serialized underlay.

The shader's distance thresholds, gradient scale and colors come from CardParam.
Raster sampling is performed by Pillow, so GPU/font AA parity is not asserted.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def srgb(x):
    return 12.92*x if x<=.0031308 else 1.055*x**(1/2.4)-.055


class NativeFont:
    def __init__(self,data):
        root=Path(data)/'native/game-ui'
        self.native_font = str(root/'fonts/FZLTH_GB18030L2_R.font')
        self.doc=json.loads((root/'fonts/VibeMOPro-Medium SDF.json').read_text(encoding='utf-8'))
        self.chars={c['m_Unicode']:c['m_GlyphIndex'] for c in self.doc['m_CharacterTable']}
        self.glyphs={g['m_Index']:g for g in self.doc['m_GlyphTable']}
        textures=json.loads((root/'textures.json').read_text(encoding='utf-8'))
        texture=next(v for k,v in textures.items() if k.endswith(':-5829278353647440834'))
        self.atlas=Image.open(root/texture).convert('RGBA').getchannel('A')
        nodes=json.loads((root/'UIMemberCardListRowView.json').read_text(encoding='utf-8'))['nodes']
        material=next(c['m_sharedMaterial'] for n in nodes if n['name']=='LvValueText'
                      for c in n['components'] if c.get('class')=='TextMeshProUGUI')
        self.params=material['floats'];self.colors=material['colors']

    def text(self,value,size,color=(255,255,255,255),shadow=True):
        value=str(value).strip()
        if not value:return Image.new('RGBA',(1,1))
        if any(ord(c) not in self.chars for c in value):
            font=ImageFont.truetype(self.native_font,max(1,round(size)))
            box=font.getbbox(value)
            out=Image.new('RGBA',(box[2]-box[0]+12,box[3]-box[1]+12))
            ImageDraw.Draw(out).text((6-box[0],6-box[1]),value,font=font,fill=color,
                stroke_width=max(1,round(size*.05)) if shadow else 0,
                stroke_fill=(40,17,98,230))
            return out
        scale=size/self.doc['m_FaceInfo']['m_PointSize']
        pad=max(4,round(size*.2))
        advance=sum(self.glyphs[self.chars[ord(c)]]['m_Metrics']['m_HorizontalAdvance'] for c in value)*scale
        canvas=Image.new('RGBA',(max(1,round(advance)+pad*2),round(size*1.25)+pad*2))
        baseline=pad+self.doc['m_FaceInfo']['m_AscentLine']*scale
        cursor=float(pad)
        atlas_pad=13
        gradient=self.params['_GradientScale']
        sharp=2*gradient*scale
        under=self.colors['_UnderlayColor']
        under_rgb=np.array([srgb(under[c]) for c in ['r','g','b']],np.float32)
        ratio=self.params['_ScaleRatioC']
        under_bias=.5-self.params['_UnderlayDilate']*ratio*.5
        under_sharp=sharp/(1+self.params['_UnderlaySoftness']*ratio*sharp)
        for ch in value:
            glyph=self.glyphs[self.chars[ord(ch)]];metric=glyph['m_Metrics'];rect=glyph['m_GlyphRect']
            if metric['m_Width']>0:
                mask=self.atlas.crop((rect['m_X']-atlas_pad,self.atlas.height-rect['m_Y']-rect['m_Height']-atlas_pad,
                                      rect['m_X']+rect['m_Width']+atlas_pad,self.atlas.height-rect['m_Y']+atlas_pad))
                mask=mask.resize((max(1,round(mask.width*scale)),max(1,round(mask.height*scale))),Image.Resampling.BILINEAR)
                sdf=np.asarray(mask,dtype=np.float32)/255
                face=np.clip((sdf-.5)*sharp+.5,0,1)*color[3]/255
                ua=np.clip((sdf-under_bias)*under_sharp+.5,0,1)*under['a'] if shadow else np.zeros_like(face)
                alpha=face+ua*(1-face)
                rgb=(face[:,:,None]*(np.array(color[:3])/255)+ua[:,:,None]*(1-face[:,:,None])*under_rgb)/np.maximum(alpha[:,:,None],1e-8)
                tile=Image.fromarray(np.uint8(np.clip(np.dstack([rgb,alpha]),0,1)*255),'RGBA')
                x=round(cursor+(metric['m_HorizontalBearingX']-atlas_pad)*scale)
                y=round(baseline-(metric['m_HorizontalBearingY']+atlas_pad)*scale)
                canvas.alpha_composite(tile,(x,y))
            cursor+=metric['m_HorizontalAdvance']*scale
        return canvas
