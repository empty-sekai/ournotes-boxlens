"""Verify the loaded visual bundle against actual DeckData bytes and identities."""
from pathlib import Path
from collections import Counter
from .common import read,digest,rows,load_deck

def binding(data,deck_path):
    data=Path(data).resolve()
    deck,deck_sha=load_deck(deck_path)
    required=['catalog.json','models/recognition.json']
    absent=[name for name in required if not (data/name).is_file()]
    if absent:raise ValueError('Visual dataset missing required files: '+', '.join(absent))
    cat=read(data/'catalog.json')
    if cat.get('schema')!='bdon-box-catalog/1':raise ValueError('Unsupported visual catalog schema')
    identities=[(r['kind'],r['id']) for r in cat['cards']]
    if len(set(identities))!=len(identities):raise ValueError('Duplicate visual catalog identities')
    known={(kind,r['_id']) for kind,table in [('member','MasterMemberCard'),('snap','MasterSupportCard')] for r in rows(deck,table)}
    present=set(identities);missing=sorted(known-present);extra=sorted(present-known)
    manifest_path=data/'visual-data-manifest.json'
    manifest=read(manifest_path) if manifest_path.exists() else None
    if manifest:
        if manifest.get('region') not in (None,deck['provenance']['region']):raise ValueError('Visual bundle region does not match DeckData')
        if manifest.get('masterVersion') not in (None,deck['provenance']['master']['version']):raise ValueError('Visual bundle masterVersion does not match DeckData')
        if 'deckDataSha256' in manifest and manifest['deckDataSha256']!=deck_sha:raise ValueError('Visual bundle DeckData hash mismatch: declared deckDataSha256 differs from actual loaded bytes')
        catalog=deck['provenance'].get('catalog',{})
        for key,deck_key in [('resourceVersion','resourceVersion'),('resourceHash','resourceHash'),('officialCatalogSha256','sha256')]:
            if key in manifest and manifest[key]!=catalog.get(deck_key):raise ValueError('Visual bundle '+key+' does not match DeckData')
    files={name:digest(data/name) for name in ['catalog.json','models/recognition.json','models/member-gallery.npz','models/snap-gallery.npz','master/MasterMemberCard.json','master/MasterSupportCard.json'] if (data/name).exists()}
    if manifest:
        claimed=manifest.get('files',{})
        if not isinstance(claimed,dict):raise ValueError('Visual manifest files must be a hash map')
        for name,value in claimed.items():
            path=(data/name).resolve()
            if data not in path.parents or not path.is_file():raise ValueError('Visual manifest file absent or outside dataset: '+name)
            expected=value.get('sha256') if isinstance(value,dict) else value
            if not isinstance(expected,str) or digest(path)!=expected:raise ValueError('Visual dataset manifest hash mismatch: '+name)
        for key,name in [('catalogSha256','catalog.json'),('recognitionModelsSha256','models/recognition.json')]:
            if key in manifest and manifest[key]!=files.get(name):raise ValueError('Visual dataset manifest hash mismatch: '+name)
    return {'schema':'ournotes.loaded-visual-data/1','dataPath':str(data),'boundDeckDataSha256':deck_sha,'boundDeckRegion':deck['provenance']['region'],'boundDeckMasterVersion':deck['provenance']['master']['version'],'identityCounts':dict(Counter(k for k,_ in present)),'identities':[{'kind':k,'id':i} for k,i in sorted(present)],'missingBoundIdentities':[{'kind':k,'id':i} for k,i in missing],'extraIdentities':[{'kind':k,'id':i} for k,i in extra],'coversBoundMasterIdentities':not missing and not extra,'files':files,'datasetManifest':manifest,'datasetManifestSha256':digest(manifest_path) if manifest else None,'scope':'Visual recognition capability only; not player ownership, runtime equivalence, or OCR field accuracy'}
