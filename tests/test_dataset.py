"""데이터셋 무결성 테스트.  python -m pytest -q"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kgdata.io import INVERSE, load_split  # noqa: E402


def _check(ver, n_total):
    tr, va, te = load_split(ROOT / "data" / ver)
    allt = tr + va + te
    assert len(allt) == n_total
    assert len(set(allt)) == len(allt)                     # 중복 없음
    assert not (set(tr) & set(te)) and not (set(tr) & set(va))
    return tr, va, te


def test_v1_0():
    _check("v1.0", 7354)


def test_v1_1():
    _check("v1.1", 8327)


def test_noleak_has_no_inverse_leak():
    tr, va, te = _check("v1.1_noleak", 8327)
    trs = set(tr)
    assert all((t, INVERSE[r], h) not in trs for h, r, t in te if r in INVERSE)
