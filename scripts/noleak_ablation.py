"""누수 없는 분할(v1.1_noleak)에서 Ours 손실 구성요소 ablation.

역관계 누수가 사라졌을 때 어떤 보조 손실이 도움/방해가 되는지 확인한다.

    python scripts/noleak_ablation.py                       # 4변형 × 3seed
    python scripts/noleak_ablation.py --seeds 42 --epochs 5 # 스모크

결과 : results/benchmark/raw/<ver>_<variant>_s<seed>.json, results/noleak_ablation.md
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

VARIANTS = {
    "Ours-L_attr": {"gamma_attr": 0.0},
    "Ours-L_type": {"alpha_type": 0.0},
    "Ours-L_dir": {"delta_dir": 0.0},
    "Ours-L_hier_attn": {"beta_hier": 0.0},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1.1_noleak")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--out", default="results/benchmark")
    args = ap.parse_args()
    kg = load_kg(ROOT / "data" / args.version)
    raw = ROOT / args.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name, ov in VARIANTS.items():
        for s in args.seeds:
            out = raw / f"{args.version}_{name}_s{s}.json"
            if out.exists():
                continue
            t0 = time.time()
            m, P, _, _ = train_chunk(kg, kg.train, config_for("Ours", seed=s, epochs=args.epochs, **ov))
            res = evaluate(m, P, kg, kg.test)
            out.write_text(json.dumps({k: v for k, v in res.items() if not k.startswith("_")}, ensure_ascii=False),
                           encoding="utf-8")
            print(f"{name:18s} s{s} MRR={res['overall']['mrr']:.4f} ({time.time() - t0:.0f}s)", flush=True)

    L = [f"# {args.version} — Ours 손실 ablation (test MRR, seed 평균)", "",
         "| 구성 | MRR | Hits@1 | Hits@10 | 방향성 MRR | 계층 MRR | seeds |", "|---|---|---|---|---|---|---|"]
    for name in ["TransE", "TransO", "Ours", *VARIANTS]:
        rs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(raw.glob(f"{args.version}_{name}_s*.json"))]
        if not rs:
            continue
        mean = lambda f: statistics.mean(f(r) for r in rs)  # noqa: E731
        L.append(f"| {name} | {mean(lambda r: r['overall']['mrr']):.4f} | {mean(lambda r: r['overall']['hits@1']):.4f} | "
                 f"{mean(lambda r: r['overall']['hits@10']):.4f} | {mean(lambda r: r['groups']['directional']['mrr']):.4f} | "
                 f"{mean(lambda r: r['groups']['hierarchy']['mrr']):.4f} | {len(rs)} |")
    (ROOT / "results" / "noleak_ablation.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
