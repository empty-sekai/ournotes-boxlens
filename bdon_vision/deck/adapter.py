"""Strict evidence gate in front of the EXISTING Rust Roster, no scoring model."""
from copy import deepcopy
from .common import read, digest, rows, load_deck

FIELDS = {'member': {'level': 'level', 'awake': 'awake_count', 'rank': 'card_rank'},
          'snap': {'level': 'level', 'rank': 'card_rank'}}
PLAYER = ['characterRanks', 'bandItems', 'vipRank', 'events', 'memory',
          'ownedMemberCardIds', 'ownedSupportCardIds']

class IncompleteInventory(ValueError):
    def __init__(self, issues):
        self.issues = issues
        super().__init__('Inventory requires explicit review/completion: ' + '; '.join(i['path'] for i in issues))

def inventory(box, deck_path, *, mock=False, region=None, fixture_manifest=None,recognition_dataset=None):
    if box.get('schema') != 'bdon-box/1':
        raise ValueError('Unsupported BoxLens contract')
    deck,deck_sha = load_deck(deck_path)
    expected_region = deck['provenance']['region']
    if type(mock) is not bool: raise ValueError('Explicit mock flag must be boolean')
    if region != expected_region:
        raise ValueError('Explicit input region must match bound DeckData region')
    return {'schema': 'ournotes.player-inventory/1', 'source': 'boxlens', 'mock': mock,
            'region': region, 'deckDataSha256': deck_sha,
            'masterVersion': deck['provenance']['master']['version'],
            'boxlensVersion': '0.1.0', 'boxlensCommit':'2ea25d8e2fdcbd3d1514b101a13b34d3e70d2ff1',
            'modelScope':'Offline native model 1.0.1-25; JP client 1.0.4 equivalence not inferred', 'box': deepcopy(box),
            'recognitionProvenance': box.get('recognition_provenance', []),
            'recognitionDataset':deepcopy(recognition_dataset),'fixtureManifest': fixture_manifest, 'coverage': 'observed_only'}

def adapt(inv, completion, deck_path, minimum_confidence=.8):
    if type(inv.get('mock')) is not bool: raise ValueError('Inventory mock declaration must be boolean')
    if inv.get('schema') != 'ournotes.player-inventory/1':
        raise ValueError('Unsupported inventory contract')
    deck,deck_sha = load_deck(deck_path)
    if inv.get('deckDataSha256') != deck_sha:
        raise ValueError('DeckData version/hash mismatch')
    if inv.get('region') != deck['provenance']['region']: raise ValueError('Inventory region binding mismatch')
    if inv.get('masterVersion') != deck['provenance']['master']['version']: raise ValueError('Inventory masterVersion binding mismatch')
    box = inv['box']
    if box.get('schema') != 'bdon-box/1':
        raise ValueError('Unsupported BoxLens contract')
    if not isinstance(minimum_confidence, (int,float)) or not 0 <= minimum_confidence <= 1:
        raise ValueError('Invalid confidence threshold')
    c = completion or {}
    issues = []
    def issue(path, reason): issues.append({'path': path, 'reason': reason})
    def integer(value, path, low=1, high=None):
        if type(value) is not int or value < low or (high is not None and value > high):
            issue(path, 'invalid integer/range')
            return False
        return True
    if c:
        if c.get('schema') != 'ournotes.inventory-completion/1':
            raise ValueError('Unsupported completion contract')
        if c.get('deckDataSha256') != inv['deckDataSha256']:
            raise ValueError('Completion DeckData hash mismatch')
        if c.get('source') not in ('manual', 'mock-truth') or not c.get('evidence'):
            raise ValueError('Completion requires explicit manual/mock evidence')
        if c.get('source') == 'mock-truth' and (not inv['mock'] or c.get('mock') is not True):
            raise ValueError('Mock truth cannot complete real-player inventory')
    if c.get('ownershipComplete') is not True: issue('ownershipComplete', 'partial screenshots do not establish complete ownership')
    if c.get('playerStateComplete') is not True: issue('playerStateComplete', 'explicit player state declaration required')
    player = deepcopy(c.get('player', {}))
    for f in PLAYER:
        if f not in player or player[f] is None: issue('player.'+f, 'not recognized; explicit value required')
    charids = {r['_id'] for r in rows(deck, 'MasterCharacter')}
    ranks = player.get('characterRanks', {})
    if not isinstance(ranks, dict):
        issue('player.characterRanks', 'expected complete map'); ranks = {}
    for key in ranks:
        if not key.isdigit() or str(int(key))!=key or int(key) not in charids: issue('player.characterRanks.'+key,'unknown/noncanonical character ID')
    for char in charids:
        if str(char) not in ranks: issue(f'player.characterRanks.{char}', 'would default to Rank1')
        else:
            integer(ranks[str(char)], f'player.characterRanks.{char}')
            if ranks[str(char)] not in {r['_rank'] for r in rows(deck,'MasterCharacterRank')}: issue(f'player.characterRanks.{char}','unknown master character rank')
    if 'vipRank' in player:
        integer(player['vipRank'], 'player.vipRank')
        if player['vipRank'] not in {1}|{r['_vipRank'] for r in rows(deck,'MasterVipRankBonus')}: issue('player.vipRank','unknown VIP rank')
    if 'bandItems' in player:
        items = {r['_id'] for r in rows(deck, 'MasterBandItem')}
        if not isinstance(player['bandItems'],dict): issue('player.bandItems', 'expected explicit owned item map')
        else:
            levels = {(r['_bandItemId'], r['_level']) for r in rows(deck,'MasterBandItemLevel')}
            for k, v in player['bandItems'].items():
                if type(v) is not int or not str(k).isdigit() or str(int(k))!=str(k) or int(k) not in items or (int(k),v) not in levels:
                    issue('player.bandItems.'+str(k), 'unknown item/level')
    if 'events' in player:
        events={r['_id'] for r in rows(deck,'MasterEvent')}
        if not isinstance(player['events'],list) or any(type(i) is not int or i not in events for i in player['events']):
            issue('player.events','expected explicit known event ID list')
    memory=player.get('memory')
    if isinstance(memory,dict):
        for f in ['musicRanks','unlockedMembers','unlockedSupports']:
            if f not in memory: issue('player.memory.'+f,'explicit collection required')
        music_ranks=memory.get('musicRanks')
        music_rows={r['_id']:r for r in rows(deck,'MasterMemoryMusic')}
        if not isinstance(music_ranks,dict):issue('player.memory.musicRanks','explicit music-rank map required')
        else:
            for key,value in music_ranks.items():
                music=music_rows.get(int(key)) if str(key).isdigit() and str(int(key))==str(key) else None
                if music is None or type(value) is not int or value not in {r['_scoreRank'] for r in rows(deck,'MasterMemoryMusicBonus') if r['_groupId']==music['_groupId']}:
                    issue('player.memory.musicRanks.'+str(key),'unknown memory music/rank in bound master')
    elif 'memory' in player: issue('player.memory','explicit memory object required, even if empty')
    observed={}
    for row in box['cards']:
        key=(row['kind'],row['id'])
        if key in observed: raise ValueError('Duplicate native box identity')
        observed[key]=row
    if box['unique_count'] != len(observed) or box['duplicate_observations'] != box['observations'] - len(observed):
        raise ValueError('Invalid BoxLens observation counts')
    requested={}
    for kind, key in [('member','members'),('snap','snaps')]:
        for entry in c.get(key,[]):
            ident=(kind,entry['id'])
            if ident in requested: raise ValueError('Duplicate completion identity')
            requested[ident]=entry
    add=c.get('addMissingIdentities',[])  # explicit evidence-backed consent; never auto-add truth
    exclude=c.get('excludeObservedIdentities',[])
    for label, entries in [('addMissingIdentities',add),('excludeObservedIdentities',exclude)]:
        if not isinstance(entries,list):raise ValueError(label+' must be an array')
        seen=set()
        for entry in entries:
            if not isinstance(entry,dict) or entry.get('kind') not in FIELDS or type(entry.get('id')) is not int or entry['id']<1:
                raise ValueError('Invalid '+label+' identity')
            ident=(entry['kind'],entry['id'])
            if ident in seen:raise ValueError('Duplicate '+label+' identity')
            seen.add(ident)
            if not isinstance(entry.get('evidence'),str) or not entry['evidence'].strip():
                raise ValueError(label+' requires nonempty per-identity evidence')
    addkeys={(r['kind'],r['id']) for r in add}
    exkeys={(r['kind'],r['id']) for r in exclude}
    identities=(observed.keys() | addkeys) - exkeys
    for ident in requested.keys()-identities-exkeys: issue(str(ident),'unobserved completion identity requires explicit addMissingIdentities')
    corrections=[]; output={'player':player,'members':[],'snaps':[]}
    for kind, id_ in sorted(identities):
        path=f'{kind}:{id_}'
        table='MasterMemberCard' if kind=='member' else 'MasterSupportCard'
        master = next((r for r in rows(deck,table) if r['_id']==id_),None)
        if master is None: issue(path,'identity not in bound regional master');continue
        row=observed.get((kind,id_)); supplied=requested.get((kind,id_),{})
        owned={'id':id_}
        for target, native in FIELDS[kind].items():
            field=(row or {}).get(native,{'value':None,'confidence':0})
            conflicts=(row or {}).get('conflicts',{}).get(native)
            valid=field.get('value') is not None and field.get('confidence',0)>=minimum_confidence and not conflicts
            if target in supplied:
                val=supplied[target]
                if not valid or val!=field['value']:
                    corrections.append({'kind':kind,'id':id_,'field':target,'previous':deepcopy(field),
                        'previousConflicts':deepcopy(conflicts),'value':val,'source':c['source'],'evidence':c['evidence']})
                owned[target]=val
            elif valid: owned[target]=field['value']
            else: issue(path+'.'+target,'conflicting/unknown/low-confidence; explicit completion required')
        if kind=='member':
            for target in ['liveSkillLevel','gekisouSkillLevel']:
                if target not in supplied: issue(path+'.'+target,'not visible on list screenshot')
                else:
                    owned[target]=supplied[target]
                    effect_table='MasterLiveSkillEffect' if target=='liveSkillLevel' else 'MasterGekisouSkillEffect'
                    card_skill='_liveSkillID' if target=='liveSkillLevel' else '_gekisouSkillID'
                    effect_skill='_liveSkillID' if target=='liveSkillLevel' else '_gekisouSkillID'
                    valid_levels={r['_level'] for r in rows(deck,effect_table) if r[effect_skill]==master[card_skill]}
                    if type(owned[target]) is not int or owned[target] not in valid_levels:
                        issue(path+'.'+target,'no matching effect level in bound master for card skill '+str(master[card_skill]))
        for f,v in owned.items():
            if f!='id': integer(v,path+'.'+f)
        group_fields={'level':('MasterMemberCardLevel' if kind=='member' else 'MasterSupportCardLevel','_memberCardLevelGroup' if kind=='member' else '_supportCardLevelGroup','_level'),
                      'rank':('MasterMemberCardRank' if kind=='member' else 'MasterSupportCardRank','_memberCardRankGroup' if kind=='member' else '_supportCardRankGroup','_rank')}
        if kind=='member':group_fields['awake']=('MasterMemberCardAwake','_memberCardAwakeGroup','_awakeCount')
        for field,(table,group,value_column) in group_fields.items():
            if field in owned and (type(owned[field]) is not int or owned[field] not in {r[value_column] for r in rows(deck,table) if r['_group']==master[group]}):
                issue(path+'.'+field,'no matching card group/value row in bound master')
        if 'level' in owned and type(owned['level']) is int:
            if kind=='member' and 'awake' in owned and type(owned['awake']) is int:
                limits=[r['_limitLevel'] for r in rows(deck,'MasterMemberCardLevelLimit') if r['_rarity']==master['_rarity'] and r['_awakeCount']==owned['awake']]
                if not limits or owned['level']>max(limits): issue(path+'.level','exceeds current awake cap')
            if kind=='snap' and 'rank' in owned and type(owned['rank']) is int:
                limits=[r['_limitLevel'] for r in rows(deck,'MasterSupportCardRank') if r['_group']==master['_supportCardRankGroup'] and r['_rank']==owned['rank']]
                if not limits or owned['level']>max(limits): issue(path+'.level','exceeds current snap rank cap')
        output['members' if kind=='member' else 'snaps'].append(owned)
    for kind,f,collection in [('member','ownedMemberCardIds','members'),('snap','ownedSupportCardIds','snaps')]:
        owned=player.get(f)
        valid_ids={r['_id'] for r in rows(deck,'MasterMemberCard' if kind=='member' else 'MasterSupportCard')}
        if not isinstance(owned,list) or any(type(i) is not int or i not in valid_ids for i in owned) or len(set(owned))!=len(owned):
            issue('player.'+f,'explicit unique known owned set required; None means all-owned in Rust')
        elif {r['id'] for r in output[collection]} != set(owned):
            issue('player.'+f,'every declared owned card needs growth data or explicit observed exclusion review; pool must match declared owned set')
    for f,ownedkey in [('unlockedMembers','ownedMemberCardIds'),('unlockedSupports','ownedSupportCardIds')]:
        if isinstance(memory,dict) and f in memory:
            values=memory[f]
            if not isinstance(values,list) or any(type(v) is not int for v in values) or len(set(values))!=len(values) or not set(values).issubset(set(player.get(ownedkey) or [])):
                issue('player.memory.'+f,'expected unique explicit owned unlock IDs')
    memberrows={r['_id']:r for r in rows(deck,'MasterMemberCard')}
    if len({memberrows[r['id']]['_characterID'] for r in output['members']}) < 5: issue('members','at least five different characters needed for a legal deck')
    if issues: raise IncompleteInventory(issues)
    return {'schema':'ournotes.roster-adaptation/1','mock':inv['mock'],'source':c['source'],
            'region':inv['region'],'deckDataSha256':inv['deckDataSha256'],'masterVersion':inv['masterVersion'],
            'recognitionDataset':deepcopy(inv.get('recognitionDataset')),'minimumConfidence':minimum_confidence,'complete':True,'roster':output,'corrections':corrections,
            'addedIdentities':add,'excludedIdentities':exclude,'originalBox':deepcopy(box),
            'assumptions':['Complete ownership and player state explicitly declared by completion evidence',
                'Mock truth completions are not OCR measurements']}
