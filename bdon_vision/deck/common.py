from pathlib import Path
import json, hashlib, math

def strict_json(raw):
    """Reject duplicate fields recursively before any typed solver boundary."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result
    def finite(token):
        value = float(token)
        if not math.isfinite(value):
            raise ValueError(f"Non-finite JSON number: {token}")
        return value
    def invalid_constant(token):
        raise ValueError(f"Invalid JSON constant: {token}")
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8-sig')
    return json.loads(raw, object_pairs_hook=unique, parse_float=finite,
                      parse_constant=invalid_constant)

def read(path):
    return strict_json(Path(path).read_text(encoding='utf-8-sig'))

def load_deck(path):
    """Parse and fingerprint the SAME bytes, not self-reported provenance.

    The digest binds every master row and chart as well as the region metadata.
    """
    raw = Path(path).read_bytes()
    doc = strict_json(raw)
    if doc.get("format") != "nnnotes.deck-data/1":
        raise ValueError("Unsupported DeckData contract")
    return doc, hashlib.sha256(raw).hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rows(deck, name):
    t = deck['master'][name]
    return [dict(zip(t['columns'], row)) for row in t['rows']]

def prepare_deck(source, target):
    # Preserve EVERY numeric token from nnnotes' native exporter. Extra music-data
    # fields are ignored by DeckData; never roundtrip master floats through Python.
    raw = Path(source).read_text(encoding='utf-8-sig')
    doc = strict_json(raw)
    assert doc['format'] == 'nnnotes.music-data/1'
    assert raw.count('"nnnotes.music-data/1"') == 1
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(raw.replace('"nnnotes.music-data/1"', '"nnnotes.deck-data/1"', 1), encoding='utf-8')
    return {'source': str(source), 'sourceSha256': digest(source),
            'targetSha256': digest(target), 'masterNumericTokens': 'unchanged',
            'charts': len(doc['charts']), 'masterVersion': doc['provenance']['master']['version']}
