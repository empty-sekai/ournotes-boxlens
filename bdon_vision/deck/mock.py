"""Mock UI reuses the original BoxLens native prefab renderer/composition."""
from pathlib import Path
from collections import Counter
from .common import read, write, rows, digest

def generate(data, deck_path, out, member_count=5, snap_count=1,newest_snaps=False):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter
    from bdon_vision.assets import native_module
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    deck=read(deck_path); catalog=read(Path(data)/'catalog.json')['cards']
    natives={k:read(Path(data)/('master/'+t+'.json'))['_allData'] for k,t in [('member','MasterMemberCard'),('snap','MasterSupportCard')]}
    current={k:{r['_id']:r for r in rows(deck,t)} for k,t in [('member','MasterMemberCard'),('snap','MasterSupportCard')]}
    old={k:{r['_id']:r for r in value} for k,value in natives.items()}
    chosen=[]; chars=set(); parity=[]
    for card in sorted([c for c in catalog if c['kind']=='member'],key=lambda c:c['id'],reverse=True):
        row=current['member'].get(card['id']); before=old['member'].get(card['id'])
        if not row or not before or row['_characterID'] in chars: continue
        differences=[k for k,v in row.items() if before.get(k)!=v]
        if differences: continue
        chosen.append(card);chars.add(row['_characterID']);parity.append({'kind':'member','id':card['id'],'allScoringColumnsEqual':True})
        if len(chosen)==member_count: break
    snap_source=sorted(catalog,key=lambda c:c['id'],reverse=True) if newest_snaps else catalog
    snapcards=[c for c in snap_source if c['kind']=='snap' and c['id'] in current['snap'] and all(old['snap'][c['id']].get(k)==v for k,v in current['snap'][c['id']].items())][:snap_count]
    chosen.extend(snapcards)
    for c in snapcards: parity.append({'kind':'snap','id':c['id'],'allScoringColumnsEqual':True})
    if len(chosen)!=member_count+snap_count: raise ValueError('No semantically matching requested mock pool')
    members=[];snaps=[]
    for i,c in enumerate(chosen):
        if c['kind']=='member':
            awake=i%5+1;rank=i%5+1
            limits=[r['_limitLevel'] for r in rows(deck,'MasterMemberCardLevelLimit') if r['_rarity']==c['rarity'] and r['_awakeCount']==awake]
            members.append({'id':c['id'],'level':min(max(limits),20+i*6),'awake':awake,'rank':rank,'liveSkillLevel':2,'gekisouSkillLevel':2})
        else: snaps.append({'id':c['id'],'level':24+3*(i-member_count),'rank':2+(i-member_count)%2})
    charids=[r['_id'] for r in rows(deck,'MasterCharacter')]
    itemids=[r['_id'] for r in rows(deck,'MasterBandItem')]
    player={'characterRanks':{str(i):10+i%8 for i in charids},'bandItems':{str(i):1 for i in itemids},
            'vipRank':2,'events':[],'memory':{'musicRanks':{},'unlockedMembers':[members[0]['id']],'unlockedSupports':[snaps[0]['id']]},
            'ownedMemberCardIds':[m['id'] for m in members],'ownedSupportCardIds':[s['id'] for s in snaps]}
    completion={'schema':'ournotes.inventory-completion/1','source':'mock-truth','mock':True,
                'evidence':'Deterministic generated mock player; fixture-manifest.json and completion.json, no real-player assertion',
                'deckDataSha256':digest(deck_path),'ownershipComplete':True,'playerStateComplete':True,
                'player':player,'members':members,'snaps':snaps,'addMissingIdentities':[],'excludeObservedIdentities':[]}
    write(out/'completion.json',completion)
    write(out/'truth-roster.json',{'player':player,'members':members,'snaps':snaps})
    native=native_module(data); renderer=native.CardRenderer('_japanese')
    (out/'cards').mkdir(exist_ok=True)
    for card in chosen:
        st=next(r for r in (members if card['kind']=='member' else snaps) if r['id']==card['id'])
        renderer.render(native.CardState(asset_id=card['asset_id'],rarity=card['rarity'],card_type=card['card_type'],level=st['level'],rank=st['rank'],awake_count=st.get('awake',1),param='level'),card['kind'],scale=1.3).save(out/'cards'/f"{card['kind']}-{card['id']}.png")
    states={('member',s['id']):s for s in members}|{('snap',s['id']):s for s in snaps}
    lookup={(c['kind'],c['id']):c for c in chosen}; ids=[m['id'] for m in members]
    split=3 if member_count==5 else 4
    plans=[('01-level-a.png','member',ids[:split],'level',False,False),
           ('02-level-overlap.png','member',ids[split-1:],'level',False,False),
           ('03-training.png','member',ids,'training',False,False),
           ('04-hidden.png','member',ids,'hide',False,False),
           ('05-snap.png','snap',[c['id'] for c in snapcards],'level',False,False),
           ('06-conflict.png','member',[ids[0]],'level',True,False),
           ('07-degraded-fields.png','member',ids[:2],'level',False,True)]
    records=[]
    for name,kind,cardids,mode,conflict,degraded in plans:
        width,height=1280,720
        noise=np.random.default_rng(20261001).integers(130,220,(45,80,3),dtype=np.uint8)
        scene=Image.fromarray(noise).resize((width,height),Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(12)).convert('RGBA')
        ImageDraw.Draw(scene).text((28,28),'MOCK PLAYER | native card renderer | '+mode,fill=(40,40,60))
        scale=(.92 if member_count==5 else .80) if kind=='member' else (1.65 if snap_count==1 else 1.55)
        # Three+ snaps use a complete three-column grid; never crop a declared visible card.
        if kind=='snap' and snap_count>=3:scale=1.0
        columns=3 if kind=='snap' and snap_count>=3 else 5
        lw,lh=(224,294) if kind=='member' else (326,184)
        truth=[]
        for i,id_ in enumerate(cardids):
            card=lookup[(kind,id_)];s=states[(kind,id_)]
            lv=s['level']+(1 if conflict else 0)
            state=native.CardState(asset_id=card['asset_id'],rarity=card['rarity'],card_type=card['card_type'],
                level=lv,rank=s['rank'],awake_count=s.get('awake',1),param=mode)
            tile=renderer.render(state,kind,scale=1.)
            tile=tile.resize((round((lw+64)*scale),round((lh+64)*scale)),Image.Resampling.BILINEAR)
            x=65+round((i%columns)*lw*scale*1.13); y=(165 if member_count==5 else 125)+round((i//columns)*lh*scale*1.10)
            if not (0<=x and 0<=y and x+round(lw*scale)<=width and y+round(lh*scale)<=height):
                raise ValueError(f'Mock visible card outside screenshot: {kind}#{id_}; fix layout, do not fabricate OCR ownership')
            scene.alpha_composite(tile,(x-round(32*scale),y-round(32*scale)))
            if degraded:
                region=(x,y+round((lh-70)*scale),x+round(lw*scale),y+round(lh*scale))
                patch=scene.crop(region).filter(ImageFilter.GaussianBlur(14));scene.paste(patch,region[:2])
            truth.append({'kind':kind,'id':id_,'bbox':[x,y,round(lw*scale),round(lh*scale)],
                'level':lv if mode=='level' and not degraded else None,
                'awake_count':s.get('awake') if mode=='training' else None,
                'card_rank':s['rank'] if mode!='hide' and not degraded else None,
                'display_mode':mode,'latentState':s,'conflictingHistoricalObservation':conflict})
        path=out/name;scene.convert('RGB').save(path)
        records.append({'file':name,'sha256':digest(path),'cards':truth,'profile':'degraded-fields' if degraded else 'clear',
            'source':'mock','locale':'ja','nativeComposition':'native-padding-preserved-v2'})
    # Duplicate bytes under another file name exercises cross-page/source evidence.
    (out/'08-duplicate.png').write_bytes((out/'01-level-a.png').read_bytes())
    records.append({**records[0],'file':'08-duplicate.png','duplicateOf':'01-level-a.png'})
    from .visual_data import binding
    manifest={'recognitionDataset':binding(data,deck_path),'schema':'ournotes.mock-fixture/1','mock':True,'source':'mock','seed':20261001,
        'region':deck['provenance']['region'],'masterVersion':deck['provenance']['master']['version'],
        'deckDataSha256':digest(deck_path),'recognitionCatalogSha256':digest(Path(data)/'catalog.json'),
        'galleryCoverage':dict(Counter(c['kind'] for c in catalog)),'deckCoverage':{'member':len(current['member']),'snap':len(current['snap'])},
        'selectedIdentityParity':parity,'rendererSha256':digest(Path(data)/'native/render_game_components.py'),
        'generatorSource':'ournotes-boxlens/bdon_vision/assets.py::native_module and synthetic.py::generate native composition',
        'generatorSourceSha256':digest(Path(__file__).parents[1]/'synthetic.py'),
        'scope':'Mock card-list screenshots, generated backgrounds and player progress; original prefab renderer, approximate CPU SDF; not real game/player screenshots',
        'covers':['missing fields','degraded low-confidence fields','cross-page duplicate','conflict','level','awake','rank','skill levels via explicit truth','snap','character ranks','items','VIP','memory','explicit owned sets'],
        'screenshots':records,'truthFile':'completion.json','complete':True}
    write(out/'fixture-manifest.json',manifest)
    return manifest
