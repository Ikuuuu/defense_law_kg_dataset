"""데이터셋 버전별 기준 성능 (TransE / TransO / Ours) — 역관계 누수 영향 확인.

    python scripts/benchmark_versions.py                                  # v1.0, v1.1, v1.1_noleak × 3모델 × 3seed
    python scripts/benchmark_versions.py --versions v1.1 --models TransE --seeds 42 --epochs 5   # 스모크

결과 : results/benchmark/raw/<ver>_<model>_s<seed>.json, results/benchmark_versions.md
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kgekit.data import load_kg  # noqa: E402
from kgekit.evaluate import evaluate  # noqa: E402
from kgekit.train import config_for, train_chunk  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="+", default=["v1.0", "v1.1", "v1.1_noleak"])
    ap.add_argument("--models", nargs="+", default=["TransE", "TransO", "Ours"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--out", default="results/benchmark")
    args = ap.parse_args()

    raw = ROOT / args.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for v in args.versions:
        kg = load_kg(ROOT / "data" / v)
        for m in args.models:
            for s in args.seeds:
                out = raw / f"{v}_{m}_s{s}.json"
                if out.exists():
                    continue
                t0 = time.time()
                model, P, _, _ = train_chunk(kg, kg.train, config_for(m, seed=s, epochs=args.epochs))
                res = evaluate(model, P, kg, kg.test)
                out.write_text(json.dumps({k: x for k, x in res.items() if not k.startswith("_")},
                                          ensure_ascii=False), encoding="utf-8")
                print(f"{v:12s} {m:6s} s{s} MRR={res['overall']['mrr']:.4f} ({time.time() - t0:.0f}s)",
                      flush=True)

    L = ["# 데이터셋 버전별 기준 성능 (test, filtered, seed 평균 ± 표준편차)", "",
         "| 버전 | 모델 | MRR | Hits@1 | Hits@10 | 방향성 MRR | 계층 MRR | seeds |",
         "|---|---|---|---|---|---|---|---|"]
    for v in args.versions:
        for m in args.models:
            rs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(raw.glob(f"{v}_{m}_s*.json"))]
            if not rs:
                continue
            f = lambda g: [r["overall"][g] for r in rs]  # noqa: E731
            sd = statistics.stdev(f("mrr")) if len(rs) > 1 else 0.0
            L.append(f"| {v} | {m} | {statistics.mean(f('mrr')):.4f} ± {sd:.4f} | "
                     f"{statistics.mean(f('hits@1')):.4f} | {statistics.mean(f('hits@10')):.4f} | "
                     f"{statistics.mean(r['groups']['directional']['mrr'] for r in rs):.4f} | "
                     f"{statistics.mean(r['groups']['hierarchy']['mrr'] for r in rs):.4f} | {len(rs)} |")
    summary = (ROOT / "results" / "benchmark_versions.md" if args.out == "results/benchmark"
               else ROOT / args.out / "benchmark_versions.md")
    summary.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
