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
    for name in ['models/recognition.json']:
        p=data/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'synthetic-not-a-real-model')
    write(data/'catalog.json',{'schema':'bdon-box-catalog/1','cards':[{'kind':'member','id':i} for i in range(1,8)]+[{'kind':'snap','id':i} for i in range(1,4)]})
    write(data/'visual-data-manifest.json',{'region':'test-region','masterVersion':'synthetic-1','deckDataSha256':digest(path),'files':{'models/recognition.json':digest(data/'models/recognition.json')}})
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


@pytest.mark.parametrize('name,value',[('deckDataSha256','0'*64),('region','wrong'),('masterVersion','wrong'),('recognitionModelsSha256','0'*64)])
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
    path,data,_,_=fixture;(data/'models/recognition.json').write_bytes(b'mutated')
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

def test_cli_failed_review_revokes_old_roster_and_can_recover(fixture,monkeypatch):
    from types import SimpleNamespace
    from bdon_vision.deck.cli import run
    from bdon_vision.deck.recommend import recommend
    from bdon_vision.deck.review_state import load_review
    import sys
    path,data,inv,c=fixture
    out=path.parent/'workspace';box=path.parent/'box.json';completion=path.parent/'completion.json'
    write(box,inv['box']);write(completion,c)
    args=SimpleNamespace(command='deck-adapt',data=data,deck_data=path,mock=True,region='test-region',box=box,completion=completion,output=out)
    assert run(args)==0
    assert load_review(out)['inventorySha256']==digest(out/'inventory.json')
    fresh=deepcopy(inv['box']);fresh['cards'][4]['id']=6;write(box,fresh)
    args.completion=None
    assert run(args)==2
    assert read(out/'inventory.json')['box']['cards'][4]['id']==6
    assert read(out/'roster.json')['members'][4]['id']==5  # retained evidence, not usable state
    def forbidden(*args,**kwargs):raise AssertionError('solver must not run with failed current review')
    monkeypatch.setattr('bdon_vision.deck.recommend.subprocess.run',forbidden)
    matrix={'entries':[{'id':'audit','request':{'execution':{'kind':'power'},'metric':{'kind':'power'}}}]}
    with pytest.raises(ValueError,match='requires successful review'):
        recommend(sys.executable,path,out/'roster.json',matrix,path.parent/'rejected')
    c['members'][4]['id']=6;c['player']['ownedMemberCardIds'][4]=6
    write(completion,c);args.completion=completion
    assert run(args)==0
    assert not (out/'needs-review.json').exists()
    assert load_review(out)['roster']['members'][4]['id']==6
    def solver(command,**kwargs):
        assert command[1] == 'recommend'
        assert read(command[command.index('--roster')+1])['members'][4]['id']==6
        return SimpleNamespace(stdout='{}',stderr='',returncode=0)
    monkeypatch.setattr('bdon_vision.deck.recommend.subprocess.run',solver)
    assert recommend(sys.executable,path,out/'roster.json',matrix,path.parent/'accepted')['cases'][0]['exitCode']==0


def test_historical_review_requires_matching_current_inventory(fixture):
    from bdon_vision.deck.review_state import load_review
    path,_,inv,c=fixture;out=path.parent/'historical'
    adapted=adapt(inv,c,path)
    write(out/'inventory.json',inv);write(out/'adaptation.json',adapted);write(out/'roster.json',adapted['roster'])
    assert load_review(out)==adapted  # pre-fix R3 without inventorySha256 is still supported
    inv['box']['cards'][4]['id']=6;write(out/'inventory.json',inv)
    with pytest.raises(ValueError,match='inventory binding mismatch'):load_review(out)


def test_partial_review_write_never_commits_complete_state(fixture,monkeypatch):
    from bdon_vision.deck import review_state
    path,_,inv,c=fixture;out=path.parent/'partial'
    write(out/'inventory.json',inv);review_state.begin_review(out)
    original=review_state.write
    def fail_roster(target,value):
        if target.name=='roster.json':raise OSError('simulated disk failure')
        original(target,value)
    monkeypatch.setattr(review_state,'write',fail_roster)
    with pytest.raises(OSError,match='disk failure'):review_state.finish_review(out,c,adapt(inv,c,path))
    with pytest.raises(ValueError,match='requires successful review'):review_state.load_review(out)


@pytest.mark.parametrize('label',['addMissingIdentities','excludeObservedIdentities'])
@pytest.mark.parametrize('evidence',[None,'','   ',{},123])
def test_identity_review_requires_its_own_nonempty_evidence(fixture,label,evidence):
    path,_,inv,c=fixture
    entry={'kind':'member','id':6 if label=='addMissingIdentities' else 5}
    if evidence is not None:entry['evidence']=evidence
    c[label]=[entry]
    if label=='addMissingIdentities':
        c['members'].append({**c['members'][0],'id':6});c['player']['ownedMemberCardIds'].append(6)
    else:
        c['addMissingIdentities']=[{'kind':'member','id':6,'evidence':'explicit replacement'}]
        c['members'][4]['id']=6;c['player']['ownedMemberCardIds'][4]=6
    assert c['evidence']
    with pytest.raises(ValueError,match='per-identity evidence'):adapt(inv,c,path)
    c[label][0]['evidence']='Explicit manual identity review'
    result=adapt(inv,c,path)
    assert result['complete'] and result['addedIdentities']==c['addMissingIdentities']
    assert result['excludedIdentities']==c['excludeObservedIdentities']


@pytest.mark.parametrize("state",["failed","legacy","updated"])
def test_http_run_preserves_only_current_review_and_results(fixture,monkeypatch,state):
    import io,threading
    from types import SimpleNamespace
    from bdon_vision.deck import server
    from bdon_vision.deck.review_state import begin_review,finish_review
    path,data,inv,c=fixture;out=path.parent/'http'
    adapted=adapt(inv,c,path)
    for name,value in [('inventory',inv),('adaptation',adapted),('roster',adapted['roster']),('scene-matrix',{'entries':[]}),('baseline-live.stdout',{'old':True})]:write(out/(name+'.json'),value)
    write(out/'recommendations/recommendations.json',{'rosterSha256':digest(out/'roster.json'),'old':True})
    if state=='failed':
        begin_review(out)
        inv['box']['cards'][4]['id']=6;write(out/'inventory.json',inv)
    elif state=='updated':
        begin_review(out);c['members'][0]['level']=1
        finish_review(out,c,adapt(inv,c,path))
    lock=threading.Lock()
    monkeypatch.setattr(server,'threading',SimpleNamespace(Lock=lambda:lock))
    monkeypatch.setattr(server.subprocess,'Popen',lambda *a,**k:SimpleNamespace(poll=lambda:None,terminate=lambda:None,wait=lambda **k:0))
    monkeypatch.setattr(server,'urlopen',lambda *a,**k:io.BytesIO(b'{"status":"ok"}'))
    responses=[]
    def http_server(address,handler_class):
        def serve_forever():
            handler=object.__new__(handler_class);handler.path='/api/run'
            handler.send=lambda body,*a,**k:responses.append(body)
            started=threading.Event();finished=threading.Event()
            def get():
                started.set()
                try:handler.do_GET()
                finally:finished.set()
            lock.acquire()  # simulate an in-flight POST workspace mutation
            thread=threading.Thread(target=get)
            try:
                thread.start();assert started.wait(2)
                assert not finished.wait(.1), 'GET must wait for the shared mutation lock'
            finally:
                lock.release();thread.join(2)
            assert not thread.is_alive() and finished.is_set()
        return SimpleNamespace(serve_forever=serve_forever)
    monkeypatch.setattr(server,'ThreadingHTTPServer',http_server)
    server.serve(out,port=0,boxport=0,data=data,deck_data=path)
    response=responses[0]
    if state=='failed':
        assert response['inventory']['box']['cards'][4]['id']==6
        assert response['reviewRequired'] is True
        assert not {'roster','adaptation','baseline','recommendations'} & response.keys()
    elif state=='legacy':
        assert response['roster']==adapted['roster']
        assert response['baseline']['old'] and response['recommendations']['old']
        assert not response.get('reviewRequired')
    else:
        assert response['roster']['members'][0]['level']==1
        assert 'baseline' not in response and 'recommendations' not in response
        assert not response.get('reviewRequired')
