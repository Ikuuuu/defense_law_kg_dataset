"""트리플 JSON → train/valid/test (관계별 층화 80/10/10, seed 42) + ID 매핑.

    python scripts/build_split.py --triples data/v1.1/triples_all.json --out data/v1.1
    python scripts/build_split.py --triples data/v1.1/triples_all.json --out data/v1.1_noleak --no-inverse-leak

--no-inverse-leak : test/valid 트리플의 역관계 트리플이 train 에 남지 않도록 같은 split 으로 묶는다
                    (FB15k → FB15k-237 에서 지적된 역관계 누수 방지).
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kgdata.io import INVERSE, build_id_mappings, save_split  # noqa: E402
from kgdata.splitter import stratified_split  # noqa: E402


def split_without_inverse_leak(triples, ratios=(0.8, 0.1, 0.1), seed=42):
    """(h,r,t) 와 (t,r⁻¹,h) 를 한 묶음으로 보고 묶음 단위로 관계별 층화 분할."""
    tset = set(triples)
    groups, seen = [], set()
    for tr in triples:
        if tr in seen:
            continue
        h, r, t = tr
        g = [tr]
        inv = (t, INVERSE[r], h) if r in INVERSE else None
        if inv and inv in tset and inv not in seen:
            g.append(inv)
        seen.update(g)
        groups.append(g)
    by_rel = {}
    for g in groups:
        key = min(x[1] for x in g)          # 역관계 쌍은 같은 층으로
        by_rel.setdefault(key, []).append(g)
    rng = random.Random(seed)
    tr_, va_, te_ = [], [], []
    for key in sorted(by_rel):
        gs = by_rel[key]
        rng.shuffle(gs)
        n = len(gs)
        n_tr, n_va = int(n * ratios[0]), int(n * ratios[1])
        for i, g in enumerate(gs):
            (tr_ if i < n_tr else va_ if i < n_tr + n_va else te_).extend(g)
    for x in (tr_, va_, te_):
        rng.shuffle(x)
    return tr_, va_, te_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", default="data/v1.1/triples_all.json")
    ap.add_argument("--out", default="data/v1.1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-inverse-leak", action="store_true")
    args = ap.parse_args()

    triples = [tuple(x) for x in json.loads((ROOT / args.triples).read_text(encoding="utf-8"))]
    triples = sorted(set(triples))          # 입력 순서와 무관하게 재현되도록 정렬
    if args.no_inverse_leak:
        split = split_without_inverse_leak(triples, seed=args.seed)
    else:
        split = stratified_split(triples, 0.8, 0.1, 0.1, seed=args.seed, stratify_by="relation")
    e2i, r2i = build_id_mappings(triples)
    save_split(ROOT / args.out, split, e2i, r2i)
    print(f"→ {args.out}: train={len(split[0])}, valid={len(split[1])}, test={len(split[2])}, "
          f"|E|={len(e2i)}, |R|={len(r2i)}")


if __name__ == "__main__":
    main()
