"""Current workspace review state shared by CLI, HTTP and recommendation."""
from pathlib import Path
from .common import read, write, digest


def begin_review(directory):
    # Keep old files as evidence, but revoke their authority before any mutation.
    write(Path(directory) / 'needs-review.json', {'complete': False, 'state': 'pending'})


def finish_review(directory, completion, adaptation):
    directory = Path(directory)
    adaptation = {**adaptation, 'inventorySha256': digest(directory / 'inventory.json')}
    write(directory / 'reviewed-completion.json', completion)
    write(directory / 'adaptation.json', adaptation)
    write(directory / 'roster.json', adaptation['roster'])
    # This is the commit point: partial writes remain unavailable to consumers.
    (directory / 'needs-review.json').unlink()
    return adaptation


def load_review(directory, *, deck_sha=None, roster=None):
    directory = Path(directory)
    if (directory / 'needs-review.json').exists():
        raise ValueError('Current inventory requires successful review before recommendation')
    inventory_path = directory / 'inventory.json'
    inv = read(inventory_path)
    review = read(directory / 'adaptation.json')
    if roster is None:
        roster = read(directory / 'roster.json')
    if review.get('complete') is not True or review.get('roster') != roster:
        raise ValueError('Reviewed roster binding mismatch')
    if (review.get('originalBox') != inv.get('box') or
            any(review.get(key) != inv.get(key) for key in
                ('mock', 'region', 'masterVersion', 'deckDataSha256'))):
        raise ValueError('Reviewed inventory binding mismatch')
    # Preserve valid historical R3 workspaces; new reviews also bind exact bytes.
    if ('inventorySha256' in review and
            review['inventorySha256'] != digest(inventory_path)):
        raise ValueError('Reviewed inventory hash mismatch')
    if deck_sha is not None and review.get('deckDataSha256') != deck_sha:
        raise ValueError('Reviewed DeckData binding mismatch')
    return review
