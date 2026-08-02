import hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_contract_has_five_live_claims_and_ten_points():
    claims=json.loads((ROOT/'contract/live_claims.json').read_text())
    assert len(claims)==5
    m=json.loads((ROOT/'contract/contract_manifest.json').read_text())
    assert m['openreview_id']=='UC6YiTOeKb' and m['maximum_points']==10
def test_source_manifest_is_complete():
    rows=(ROOT/'evidence/source/SHA256SUMS').read_text().strip().splitlines()
    assert len(rows)==3
    for row in rows:
        h,n=row.split(maxsplit=1); p=ROOT/'evidence/source'/n.strip()
        assert p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==h
