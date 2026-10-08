"""Invoke solver-owned recommendation binary and retain exact raw JSON."""
import subprocess, time, re
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from .common import read, write, rows, digest, strict_json
from .review_state import load_review

def write_request(entry, path):
    """Keep uploaded legal numeric tokens intact; never collapse duplicate keys."""
    if 'requestJson' in entry:
        raw = entry['requestJson']
        if not isinstance(raw, str) or strict_json(raw) != entry['request']:
            raise ValueError('Original request JSON binding mismatch')
        Path(path).write_bytes(raw.encode('utf-8'))
    else:
        write(path, entry['request'])

def scene_matrix(deck_path):
    d=read(deck_path)
    shortest=min(d['charts'],key=lambda c:len(c['notes']['id']))
    sid=shortest['scoreId']
    base=next(r for r in rows(d,'MasterLiveMusic') if sid in [r.get('_easyID'),r.get('_normalID'),r.get('_hardID'),r.get('_expertID')])
    specs=[]
    def add(id_, execution, scenario=None, metric=None, expected='supported'):
        req={'format':'ournotes-deck.search-request/1','execution':execution,'metric':metric or {'kind':'score'},
            'constraints':{'noSnaps':False},'k':3,'strategy':{'kind':'exhaustive'},
            'limits':{'timeLimitMs':30000,'maxCandidates':20000,'cacheEntries':2048}}
        if scenario: req['scenario']=scenario
        specs.append({'id':id_,'request':req,'expected':expected})
    add('songless-power',{'kind':'power'},metric={'kind':'power'})
    for mode in ['free','mission','battle']:
        sc={'kind':mode,'musicId':base['_id']}
        add(mode+'-power',{'kind':'power'},sc,{'kind':'power'})
        if mode=='free':add(mode+'-skip',{'kind':'skip','scoreId':sid},sc)
        add(mode+'-live',{'kind':'live','scoreId':sid,'gekisou':mode!='free','play':{'kind':'theoreticalBest'}},sc,expected='supported' if mode!='battle' else 'requires-external-network-confirmations')
        if mode=='free':add('free-live-gekisou',{'kind':'live','scoreId':sid,'gekisou':True,'play':{'kind':'theoreticalBest'}},sc)
        if mode!='free':
            add('guard-'+mode+'-skip',{'kind':'skip','scoreId':sid},sc,expected='reject-native-choice')
            add('guard-'+mode+'-gekisou-off',{'kind':'live','scoreId':sid,'gekisou':False,'play':{'kind':'theoreticalBest'}},sc,expected='reject-native-choice')
    for mode,table in [('arena','MasterArenaMusic'),('challenge','MasterChallengeMusic')]:
        candidates=rows(d,table)
        if not candidates:
            specs.append({'id':mode+'-unavailable','expected':'unavailable','reason':'No current regional '+table+' rows; no fabricated production scenario'})
            continue
        r=candidates[0]
        music=r['_liveMusicId']
        b=next(x for x in rows(d,'MasterLiveMusic') if x['_id']==music)
        scores=[c for c in d['charts'] if c['scoreId'] in [b.get('_easyID'),b.get('_normalID'),b.get('_hardID'),b.get('_expertID')]]
        s=min(scores,key=lambda c:len(c['notes']['id']))['scoreId']
        sc={'kind':mode,'musicId':r['_id']}
        add(mode+'-power',{'kind':'power'},sc,{'kind':'power'})
        if mode=='challenge':add(mode+'-skip',{'kind':'skip','scoreId':s},sc)
        add(mode+'-live',{'kind':'live','scoreId':s,'gekisou':mode=='arena','play':{'kind':'theoreticalBest'}},sc,expected='requires-external-network-confirmations' if mode=='arena' else 'supported')
        if mode=='challenge':add('challenge-live-gekisou',{'kind':'live','scoreId':s,'gekisou':True,'play':{'kind':'theoreticalBest'}},sc)
    add('free-probability',{'kind':'live','scoreId':sid,'gekisou':False,'play':{'kind':'theoreticalBest'}},{'kind':'free','musicId':base['_id']},{'kind':'scoreAtLeast','threshold':2400000})
    events=rows(d,'MasterEvent')
    if events:
        event=events[0]
        start=datetime.strptime(event['_startAt'],'%Y/%m/%d %H:%M:%S')
        end=datetime.strptime(event['_endAt'],'%Y/%m/%d %H:%M:%S')
        def ticks(value):
            delta=value-datetime(1,1,1)
            return delta.days*864000000000+delta.seconds*10000000+delta.microseconds*10
        for original in list(specs):
            req=original.get('request',{})
            execution=req.get('execution',{}).get('kind')
            if execution not in ('skip','live') or original['id'].startswith('guard-') or req.get('metric',{}).get('kind')!='score':continue
            mode=req['scenario']['kind']
            if mode=='challenge':
                special=next(r for r in rows(d,'MasterChallengeMusic') if r['_id']==req['scenario']['musicId'])
                if special['_eventId']!=event['_id']:continue
            reward_rows=rows(d,'MasterChallengeLiveEventReward' if mode=='challenge' else 'MasterLiveEventReward')
            if not reward_rows:continue
            reward=reward_rows[0]
            played=execution=='live'
            clock={'execution':'played','savedStartJstTicks':ticks(start+timedelta(hours=1)),'serverNowJstTicks':ticks(start+timedelta(hours=1,minutes=5))} if played else {'execution':'skip','serverNowJstTicks':ticks(start+timedelta(hours=1))}
            ctx={'powerSnapshot':{'eventIds':[event['_id']],'capturedJstTicks':ticks(start+timedelta(minutes=50))},'resultClock':clock,
                 'eventPayoff':{'consumedCount':0,'localEvents':[{'eventId':event['_id'],'points':0,'challengePoints':500,'added':[]}],
                     'eventWindows':[{'eventId':event['_id'],'startJstTicks':ticks(start),'endJstTicks':ticks(end)}],
                     'selectedRewards':[{'eventId':event['_id'],'rewardId':reward['_id']}]}}
            for metric in [{'kind':'clientEventPoints','eventId':event['_id']},{'kind':'conditionalClientEventItems','eventId':event['_id'],'resourceType':reward['_resourceType'],'resourceId':reward['_resourceId']}]:
                cp=deepcopy(original);cp['id']+='-'+metric['kind'];cp['request']['metric']=metric;cp['request']['context']=deepcopy(ctx)
                cp['contextSource']='Explicit mock event clock/counters/selected reward IDs; event windows and resource identity from bound master; no server selection inferred'
                specs.append(cp)
    for original in list(specs):
        req=original.get('request',{})
        if req.get('execution',{}).get('kind')=='skip' and req.get('metric',{}).get('kind')=='score':
            cp=deepcopy(original);cp['id']+='-probability';cp['request']['metric']={'kind':'scoreAtLeast','threshold':2400000};cp['request']['limits']['maxCandidates']=500000;specs.append(cp)
    # Explicit hypothetical multiplayer inputs, separate from OCR and never inferred.
    # The frame is packet ARRIVAL; the solver owns Finish8 scheduling and percent validation.
    for original in list(specs):
        req=original.get('request',{})
        if req.get('scenario',{}).get('kind')!='battle' or req.get('execution',{}).get('kind')!='live' or req['execution'].get('gekisou') is not True or original.get('expected')!='requires-external-network-confirmations':continue
        cp=deepcopy(original);cp['id']='battle-conditional-mock-'+req['metric']['kind'];cp['expected']='supported-conditional-network'
        missions=[base['_gekisouMission'+str(i)] for i in range(1,4)]
        pattern=0 if 0 in missions else 1 if len(set(missions))==1 else 2 if len(set(missions))==3 else 3
        rank_rows=rows(d,'MasterLiveGekisouRankingScoreBonus')
        ranks=[2,3,4]  # Declared mock group ordinals; not predicted actual peer placement.
        cp['request']['networkConfirmations']=[{'frame':5000,'range':i,'rank':rank,'percent':next(r['_scoreBonusPercent'] for r in rank_rows if r['_missionPattern']==pattern and r['_count']==i+1 and r['_rank']==rank)} for i,rank in enumerate(ranks)]
        cp['contextSource']='Explicit hypothetical mock aggregate packet arrival frame5000 and group ordinals2/3/4; master percentages verified. Same-frame arrival does not mean same-frame application; immutable conditional ranks are not predicted peer scores.'
        if 'context' in cp['request']:
            cp['request']['context']['eventPayoff']['multiplayerResultPanel']={'localPlayerIndex':1,'localDisconnected':False,'otherPlayers':[{'finalScore':1000000,'disconnected':False},{'finalScore':900000,'disconnected':True}]}
            cp['contextSource']+=' Mock result panel has explicit peer scores and one disconnected peer; no server-selected rewards inferred.'
        specs.append(cp)
    for original in list(specs):
        req=original.get('request',{})
        if req.get('execution',{}).get('kind')!='live' or req.get('metric',{}).get('kind')!='score' or original.get('expected') not in ('supported','supported-conditional-network') or original['id']=='free-live':continue
        cp=deepcopy(original);cp['id']+='-probability';cp['request']['metric']={'kind':'scoreAtLeast','threshold':2400000};specs.append(cp)
    for entry in specs:
        if 'request' in entry and entry['request']['execution']['kind']=='live':
            entry['request']['strategy']={'kind':'candidate','powerSeeds':2,'proposals':600,'proposalSeed':20261001}
            entry['request']['limits']={'timeLimitMs':10000,'maxCandidates':1000,'cacheEntries':1024}
    return {'schema':'ournotes.integration-scene-matrix/1','owner':'BoxLens optional Deck recommendation workflow',
        'deckDataSha256':digest(deck_path),'entries':specs,'liveIsSeparate':True,'capabilitySource':'ournotes-deck.search-request/1','randomScope':'Solver-reported probability law and theoretical-best play','unsupportedScope':'Battle/Arena require explicitly provided aggregate peer confirmations; available scenes depend on the bound regional master'}

def recommend(binary, deck_path, roster_path, matrix, out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    sidecar=Path(roster_path).parent/'adaptation.json'
    if not sidecar.exists():raise ValueError('Strict pipeline requires reviewed adaptation.json alongside Roster; use solver directly for independent raw-Roster tests')
    # Preserve exact numeric tokens, and bind actual bytes used by the executable.
    # External source files may change while a long search is running.
    import hashlib,json
    deck_bytes=Path(deck_path).read_bytes();roster_bytes=Path(roster_path).read_bytes()
    deck_sha=hashlib.sha256(deck_bytes).hexdigest();roster_sha=hashlib.sha256(roster_bytes).hexdigest()
    review=load_review(Path(roster_path).parent,deck_sha=deck_sha,roster=strict_json(roster_bytes))
    result=[];start=time.perf_counter()
    if matrix.get('deckDataSha256') not in (None,deck_sha):raise ValueError('Scene matrix DeckData hash mismatch')
    entries=matrix.get('entries',matrix.get('cases',[]))
    ids=[e['id'] for e in entries]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate matrix entry IDs would overwrite request evidence')
    frozen_deck=out/'deck-data.input.json';frozen_roster=out/'roster.input.json'
    if frozen_deck.exists() or frozen_roster.exists():raise ValueError('Recommendation output already has frozen inputs; use a new directory')
    frozen_deck.write_bytes(deck_bytes);frozen_roster.write_bytes(roster_bytes)
    write(out/'input-binding.json',{'schema':'ournotes.frozen-recommendation-input/1','deckDataSha256':deck_sha,'rosterSha256':roster_sha,'sourceDeckData':str(Path(deck_path).resolve()),'sourceRoster':str(Path(roster_path).resolve()),'actualBinaryDeckInput':str(frozen_deck.resolve()),'actualBinaryRosterInput':str(frozen_roster.resolve()),'numericTokens':'unchanged exact bytes'})
    for entry in entries:
        id_=entry['id']
        if not isinstance(id_,str) or not re.fullmatch('[A-Za-z0-9_.-]{1,100}',id_):raise ValueError('Invalid matrix entry ID')
        if 'request' not in entry:
            result.append({'id':id_,'state':'Unavailable','reason':entry.get('reason'),'complete':False});continue
        request=entry['request'];reqfile=out/(id_+'-request.json');write_request(entry,reqfile)
        command=[str(binary),'recommend','--data',str(frozen_deck),'--roster',str(frozen_roster),'--request',str(reqfile)]
        t=time.perf_counter()
        try:
            p=subprocess.run(command,capture_output=True,encoding='utf-8',errors='replace',timeout=min(600,max(180,(request.get('limits',{}).get('timeLimitMs') or 165000)/1000+15)))
            (out/(id_+'-stdout.json')).write_text(p.stdout,encoding='utf-8')
            (out/(id_+'-stderr.log')).write_text(p.stderr,encoding='utf-8')
            try: raw=__import__('json').loads(p.stdout)
            except ValueError:
                try:raw=__import__('json').loads(p.stderr)
                except ValueError:raw={'error':p.stderr.strip() or p.stdout.strip()}
            result.append({'id':id_,'execution':request['execution'],'metric':request['metric'],
                'request':request,'requestJson':reqfile.read_bytes().decode('utf-8'),'requestFile':reqfile.name,'command':command,'exitCode':p.returncode,
                'elapsedMs':round((time.perf_counter()-t)*1000,2),'raw':raw,'rawJson':p.stdout or p.stderr})
        except subprocess.TimeoutExpired:
            result.append({'id':id_,'state':'ProcessTimedOut','complete':False,'command':command,
                'elapsedMs':round((time.perf_counter()-t)*1000,2)})
        write(out/'recommendations.json',{'schema':'ournotes.pipeline-recommendations/1','mock':review['mock'],'source':review['source'],'region':review['region'],'masterVersion':review['masterVersion'],
            'deckDataSha256':deck_sha,'rosterSha256':roster_sha,'solverBinarySha256':digest(binary),
            'elapsedMs':round((time.perf_counter()-start)*1000,2),'cases':result})
        print('recommend:',id_,result[-1].get('exitCode',result[-1].get('state')),flush=True)
    return read(out/'recommendations.json')
