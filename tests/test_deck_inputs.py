"""Hand-authored synthetic master only; no game assets or extracted masterdata."""
from copy import deepcopy
from pathlib import Path
import pytest
from bdon_vision.deck.common import read, write, digest
from bdon_vision.deck.adapter import inventory, adapt, IncompleteInventory
from bdon_vision.deck.visual_data import binding


def table(columns, records):
    return {'columns':columns,'rows':records}


@pytest.fixture
def fixture(tmp_path):
    master={
        'MasterCharacter':table(['_id'],[[i] for i in range(1,8)]),
        'MasterCharacterRank':table(['_rank'],[[1],[10]]),
        'MasterVipRankBonus':table(['_vipRank'],[[2]]),
        'MasterBandItem':table(['_id'],[[100]]),
        'MasterBandItemLevel':table(['_bandItemId','_level'],[[100,1]]),
        'MasterEvent':table(['_id'],[]),
        'MasterMemoryMusic':table(['_id','_groupId'],[[1000,1]]),
        'MasterMemoryMusicBonus':table(['_groupId','_scoreRank'],[[1,1]]),
        'MasterMemberCard':table(['_id','_characterID','_rarity','_liveSkillID','_gekisouSkillID','_memberCardLevelGroup','_memberCardRankGroup','_memberCardAwakeGroup'],[[i,i,1,101,201,1,1,1] for i in range(1,8)]),
        'MasterSupportCard':table(['_id','_supportCardLevelGroup','_supportCardRankGroup'],[[1,1,1],[2,1,1],[3,1,1]]),
        'MasterLiveSkillEffect':table(['_liveSkillID','_level'],[[101,1],[101,2]]),
        'MasterGekisouSkillEffect':table(['_gekisouSkillID','_level'],[[201,1],[201,2]]),
        'MasterMemberCardLevel':table(['_group','_level'],[[1,1],[1,20],[1,21]]),
        'MasterMemberCardRank':table(['_group','_rank'],[[1,1],[1,2]]),
        'MasterMemberCardAwake':table(['_group','_awakeCount'],[[1,1],[1,2]]),
        'MasterMemberCardLevelLimit':table(['_rarity','_awakeCount','_limitLevel'],[[1,1,20],[1,2,21]]),
        'MasterSupportCardLevel':table(['_group','_level'],[[1,1],[1,20]]),
        'MasterSupportCardRank':table(['_group','_rank','_limitLevel'],[[1,1,20],[1,2,20]])
    }
    path=tmp_path/'deck.json'
    write(path,{'format':'nnnotes.deck-data/1','provenance':{'region':'test-region','master':{'version':'synthetic-1'}},'master':master})
    field=lambda value:{'value':value,'confidence':.99}
    cards=[{'kind':'member','id':i,'level':field(20),'card_rank':field(1),'awake_count':field(1),'conflicts':{},'sources':[]} for i in range(1,6)]
    cards.append({'kind':'snap','id':1,'level':field(20),'card_rank':field(1),'awake_count':field(None),'conflicts':{},'sources':[]})
    box={'schema':'bdon-box/1','cards':cards,'unique_count':6,'observations':6,'duplicate_observations':0}
    inv=inventory(box,path,mock=True,region='test-region')
    c={'schema':'ournotes.inventory-completion/1','source':'mock-truth','evidence':'Hand-authored synthetic fixture, no actual player','mock':True,'deckDataSha256':digest(path),'ownershipComplete':True,'playerStateComplete':True,'player':{'characterRanks':{str(i):10 for i in range(1,8)},'bandItems':{'100':1},'vipRank':2,'events':[],'memory':{'musicRanks':{},'unlockedMembers':[],'unlockedSupports':[]},'ownedMemberCardIds':list(range(1,6)),'ownedSupportCardIds':[1]},'members':[{'id':i,'level':20,'rank':1,'awake':1,'liveSkillLevel':2,'gekisouSkillLevel':2} for i in range(1,6)],'snaps':[{'id':1,'level':20,'rank':1}],'addMissingIdentities':[],'excludeObservedIdentities':[]}
    data=tmp_path/'visual'
    for name in ['index.npz','models/encoder.onnx','models/fields.onnx']:
        p=data/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'synthetic-not-a-real-model')
    write(data/'catalog.json',{'schema':'bdon-box-catalog/1','cards':[{'kind':'member','id':i} for i in range(1,8)]+[{'kind':'snap','id':i} for i in range(1,4)]})
    write(data/'visual-data-manifest.json',{'region':'test-region','masterVersion':'synthetic-1','deckDataSha256':digest(path),'files':{'index.npz':digest(data/'index.npz')}})
    return path,data,inv,c


def test_complete_explicit_state_preserves_raw_box(fixture):
    path,_,inv,c=fixture;before=deepcopy(inv)
    result=adapt(inv,c,path)
    assert result['complete'] and result['roster']['player']==c['player']
    assert inv==before and result['originalBox']==inv['box']
    assert not result['addedIdentities']


def test_missing_state_never_defaults_to_rank_one_or_all_owned(fixture):
    path,_,inv,_=fixture
    with pytest.raises(IncompleteInventory) as error:adapt(inv,None,path)
    names={i['path'] for i in error.value.issues}
    assert {'ownershipComplete','playerStateComplete','player.characterRanks.1','member:1.liveSkillLevel'}<=names


@pytest.mark.parametrize('name,value',[('region','forged-region'),('masterVersion','forged-master'),('deckDataSha256','0'*64)])
def test_inventory_envelope_binding(fixture,name,value):
    path,_,inv,c=fixture;inv[name]=value
    with pytest.raises(ValueError,match='mismatch'):adapt(inv,c,path)


@pytest.mark.parametrize('field',['liveSkillLevel','gekisouSkillLevel','level','rank','awake'])
def test_growth_and_skill_bound_by_actual_master_rows(fixture,field):
    path,_,inv,c=fixture;c['members'][0][field]=999
    with pytest.raises(IncompleteInventory) as error:adapt(inv,c,path)
    assert any(i['path']==f'member:1.{field}' for i in error.value.issues)


def test_low_confidence_and_conflict_require_explicit_review(fixture):
    path,_,inv,c=fixture
    del c['members'][0]['level'];inv['box']['cards'][0]['level']['confidence']=.01
    inv['box']['cards'][0]['conflicts']={'level':{'20':['a'],'21':['b']}}
    with pytest.raises(IncompleteInventory) as error:adapt(inv,c,path)
    assert any(i['path']=='member:1.level' for i in error.value.issues)
    c['members'][0]['level']=20
    result=adapt(inv,c,path)
    assert result['corrections'][0]['previousConflicts']=={'20':['a'],'21':['b']}
    assert result['corrections'][0]['source']=='mock-truth'


def test_mock_truth_cannot_complete_real_player(fixture):
    path,_,inv,c=fixture;inv['mock']=False
    with pytest.raises(ValueError,match='real-player'):adapt(inv,c,path)


@pytest.mark.parametrize('field',['ownedMemberCardIds','ownedSupportCardIds'])
def test_null_owned_set_cannot_mean_all_owned(fixture,field):
    path,_,inv,c=fixture;c['player'][field]=None
    with pytest.raises(IncompleteInventory):adapt(inv,c,path)


def test_full_catalog_does_not_expand_inventory_or_owned_sets(fixture):
    path,data,inv,c=fixture;v=binding(data,path)
    assert v['identityCounts']=={'member':7,'snap':3} and v['coversBoundMasterIdentities']
    observed=inventory(inv['box'],path,mock=True,region='test-region',recognition_dataset=v)
    result=adapt(observed,c,path)
    assert observed['coverage']=='observed_only'
    assert len(result['roster']['members'])==5 and len(result['roster']['snaps'])==1
    assert result['roster']['player']['ownedMemberCardIds']==list(range(1,6))
    assert result['roster']['player']['ownedSupportCardIds']==[1]


def test_identity_outside_bound_master_is_not_remapped(fixture):
    path,_,inv,c=fixture;c['addMissingIdentities']=[{'kind':'snap','id':70,'evidence':'explicit synthetic boundary'}]
    with pytest.raises(IncompleteInventory) as error:adapt(inv,c,path)
    assert any(i['path']=='snap:70' for i in error.value.issues)


@pytest.mark.parametrize('name,value',[('deckDataSha256','0'*64),('region','wrong'),('masterVersion','wrong'),('indexSha256','0'*64)])
def test_visual_manifest_binding(fixture,name,value):
    path,data,_,_=fixture;m=read(data/'visual-data-manifest.json');m[name]=value;write(data/'visual-data-manifest.json',m)
    with pytest.raises(ValueError):binding(data,path)


def test_changed_actual_deck_bytes_with_same_reported_provenance_rejected_before_worker(fixture,monkeypatch):
    path,data,_,_=fixture;changed=read(path);changed['master']['MasterMemberCard']['rows'][0][0]=12345;write(path,changed)
    from bdon_vision.deck.server import serve
    def unexpected(*args,**kwargs):raise AssertionError('OCR worker must not start before byte binding')
    monkeypatch.setattr('bdon_vision.deck.server.subprocess.Popen',unexpected)
    with pytest.raises(ValueError,match='DeckData hash mismatch'):
        serve(path.parent/'ui',data=data,deck_data=path)


def test_changed_visual_file_rejected(fixture):
    path,data,_,_=fixture;(data/'index.npz').write_bytes(b'mutated')
    with pytest.raises(ValueError,match='hash mismatch'):binding(data,path)


def test_legacy_catalog_has_explicit_identity_gap(fixture):
    path,data,_,_=fixture;(data/'visual-data-manifest.json').unlink()
    cat=read(data/'catalog.json');cat['cards']=[r for r in cat['cards'] if r!= {'kind':'member','id':7}]+[{'kind':'snap','id':70}];write(data/'catalog.json',cat)
    v=binding(data,path)
    assert not v['coversBoundMasterIdentities']
    assert v['missingBoundIdentities']==[{'kind':'member','id':7}]
    assert v['extraIdentities']==[{'kind':'snap','id':70}]


def test_manifest_path_escape_rejected(fixture):
    path,data,_,_=fixture;m=read(data/'visual-data-manifest.json');m['files']['../deck.json']=digest(path);write(data/'visual-data-manifest.json',m)
    with pytest.raises(ValueError,match='outside dataset'):binding(data,path)
