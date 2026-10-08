"""Exercise the reviewed-input adapter with a released recommendation CLI."""
import os
from pathlib import Path

import pytest

from bdon_vision.deck.common import digest, write
from bdon_vision.deck.recommend import recommend, scene_matrix


DECK = Path(__file__).parent / 'fixtures' / 'deck-synthetic.json'


def test_scene_requests_use_the_solver_contract():
    matrix = scene_matrix(DECK)
    requests = [entry['request'] for entry in matrix['entries'] if 'request' in entry]
    assert {request['execution']['kind'] for request in requests} == {'power', 'skip', 'live'}
    assert all(request['format'] == 'ournotes-deck.search-request/1' for request in requests)
    assert all('seedLaw' not in request for request in requests)


@pytest.fixture
def solver():
    binary = os.environ.get('OURNOTES_DECK_BIN')
    if not binary:
        pytest.skip('Set OURNOTES_DECK_BIN to run the released CLI integration tests')
    path = Path(binary).resolve()
    assert path.is_file(), f'Recommendation CLI not found: {path}'
    return path


@pytest.fixture
def reviewed_roster(tmp_path):
    roster = {'members': [{'id':i, 'level':1} for i in range(1,6)], 'snaps':[], 'player':{}}
    inventory = {'mock':True, 'source':'synthetic', 'region':'synthetic',
                 'masterVersion':'synthetic-1', 'deckDataSha256':digest(DECK), 'box':{'cards':[]}}
    write(tmp_path / 'inventory.json', inventory)
    write(tmp_path / 'adaptation.json', {**inventory, 'complete':True,
          'originalBox':inventory['box'], 'roster':roster})
    path = tmp_path / 'roster.json'
    write(path, roster)
    return path


@pytest.mark.parametrize('scene', [
    'songless-power', 'free-skip', 'free-live', 'free-live-gekisou', 'mission-live',
])
def test_released_cli_recommends_reviewed_roster(solver, reviewed_roster, tmp_path, scene):
    matrix = scene_matrix(DECK)
    entry = next(entry for entry in matrix['entries'] if entry['id'] == scene)
    result = recommend(solver, DECK, reviewed_roster, {**matrix, 'entries':[entry]}, tmp_path / scene)
    case = result['cases'][0]
    assert case['exitCode'] == 0, case['rawJson']
    raw = case['raw']
    assert raw['format'] == 'ournotes-deck.recommendation-result/3'
    assert raw['metric'] == entry['request']['metric']
    assert raw['results']
    assert all(sorted(deck['members']) == [1,2,3,4,5] for deck in raw['results'])
    if scene == 'songless-power':
        assert raw['completion'] == 'Complete'
        assert raw['optimality'] == 'proven'
        assert raw['results'][0]['power'] == 1500
    else:
        assert int(raw['results'][0]['expectedPayoff']['numerator']) > 0
    assert result['deckDataSha256'] == digest(DECK)
    assert result['rosterSha256'] == digest(reviewed_roster)


def test_released_cli_preserves_request_errors(solver, reviewed_roster, tmp_path):
    entry = scene_matrix(DECK)['entries'][0]
    entry['request']['format'] = 'unsupported-request/1'
    case = recommend(solver, DECK, reviewed_roster, {'entries':[entry]}, tmp_path / 'error')['cases'][0]
    assert case['exitCode'] != 0
    assert case['raw']['error']['code'] == 'Input'
    assert 'unsupported recommendation format' in case['raw']['error']['message']
