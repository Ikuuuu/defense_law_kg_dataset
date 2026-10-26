"""데이터셋 통계·품질 분석 + 그림.

    python scripts/analyze.py                         # data/v1.0, data/v1.1, data/v1.1_noleak 비교
    python scripts/analyze.py --versions v1.0

산출물 : results/dataset_stats.md, results/stats_<ver>.json, results/figures/*.png
분석 항목
  - 규모 (트리플·엔티티·관계, split 크기), 관계 분포 (split 별)
  - 엔티티 차수 분포 (평균·중앙값·최대, 허브 엔티티)
  - 콜드스타트 : test 트리플 중 head/tail 이 train 에 한 번도 안 나온 비율
  - 역관계 누수 : test 트리플 (h,r,t) 의 역관계 (t,r⁻¹,h) 가 train 에 있는 비율
"""
import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kgdata.io import INVERSE, REL_EN, load_split  # noqa: E402

REL_ORDER = list(REL_EN)


def analyze(train, valid, test):
    allt = train + valid + test
    ents = {h for h, _, _ in allt} | {t for _, _, t in allt}
    deg = Counter()
    for h, _, t in allt:
        deg[h] += 1
        deg[t] += 1
    tr_ents = {h for h, _, _ in train} | {t for _, _, t in train}
    tr_set = set(train)
    cold = sum(1 for h, _, t in test if h not in tr_ents or t not in tr_ents)
    inv_rel = [x for x in test if x[1] in INVERSE]
    leak = sum(1 for h, r, t in inv_rel if (t, INVERSE[r], h) in tr_set)
    return {
        "n_triples": len(allt), "n_entities": len(ents),
        "n_relations": len({r for _, r, _ in allt}),
        "split": [len(train), len(valid), len(test)],
        "rel_all": Counter(r for _, r, _ in allt),
        "rel_test": Counter(r for _, r, _ in test),
        "degree_mean": statistics.mean(deg.values()), "degree_median": statistics.median(deg.values()),
        "degree_max": max(deg.values()), "hubs": deg.most_common(8),
        "density": len(allt) / (len(ents) ** 2),
        "test_cold_start": cold / max(len(test), 1),
        "test_inverse_leak": leak / max(len(inv_rel), 1),
        "test_inverse_rel_share": len(inv_rel) / max(len(test), 1),
        "type_dist": Counter(t for _, r, t in allt if r == "type"),
        "deg_values": sorted(deg.values(), reverse=True),
    }


def figures(stats, fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    names = list(stats)
    # 1) 관계 분포
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        vals = [stats[n]["rel_all"].get(r, 0) for r in REL_ORDER]
        ax.bar([x + i * w for x in range(len(REL_ORDER))], vals, w, label=n)
    ax.set_xticks([x + w * (len(names) - 1) / 2 for x in range(len(REL_ORDER))])
    ax.set_xticklabels([REL_EN[r] for r in REL_ORDER], rotation=35, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("# triples (log)")
    ax.set_title("Relation distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "relation_distribution.png", dpi=150)
    plt.close(fig)
    # 2) 차수 분포 (rank-degree, log-log)
    fig, ax = plt.subplots(figsize=(5, 4))
    for n in names:
        d = stats[n]["deg_values"]
        ax.loglog(range(1, len(d) + 1), d, label=n)
    ax.set_xlabel("entity rank")
    ax.set_ylabel("degree")
    ax.set_title("Degree distribution (rank plot)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "degree_distribution.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="+", default=["v1.0", "v1.1", "v1.1_noleak"])
    args = ap.parse_args()

    stats = {}
    for v in args.versions:
        d = ROOT / "data" / v
        if not (d / "train.tsv").exists():
            print(f"skip {v} (train.tsv 없음 — scripts/build_split.py 먼저 실행)")
            continue
        stats[v] = analyze(*load_split(d))
        js = {k: (dict(val) if isinstance(val, Counter) else val) for k, val in stats[v].items()
              if k != "deg_values"}
        (ROOT / "results" / f"stats_{v}.json").write_text(json.dumps(js, ensure_ascii=False, indent=1),
                                                          encoding="utf-8")
    names = list(stats)
    row = lambda label, f: f"| {label} | " + " | ".join(f(stats[n]) for n in names) + " |"  # noqa: E731
    L = ["# 데이터셋 통계", "", "| 항목 | " + " | ".join(names) + " |", "|---|" + "---|" * len(names),
         row("트리플", lambda s: f"{s['n_triples']:,}"),
         row("엔티티", lambda s: f"{s['n_entities']:,}"),
         row("관계", lambda s: str(s["n_relations"])),
         row("train / valid / test", lambda s: " / ".join(f"{x:,}" for x in s["split"])),
         row("평균 / 중앙 / 최대 차수", lambda s: f"{s['degree_mean']:.2f} / {s['degree_median']:.0f} / {s['degree_max']}"),
         row("밀도 (|T| / |E|²)", lambda s: f"{s['density']:.2e}"),
         row("test 콜드스타트 비율", lambda s: f"{s['test_cold_start'] * 100:.1f}%"),
         row("test 중 역관계 보유 관계 비중", lambda s: f"{s['test_inverse_rel_share'] * 100:.1f}%"),
         row("└ 그중 역관계가 train 에 있는 비율 (누수)", lambda s: f"{s['test_inverse_leak'] * 100:.1f}%"),
         "", "## 관계별 트리플 수 (전체 / test)", "",
         "| 관계 | 영문 | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for r in REL_ORDER:
        L.append(f"| {r} | {REL_EN[r]} | " + " | ".join(
            f"{stats[n]['rel_all'].get(r, 0)} / {stats[n]['rel_test'].get(r, 0)}" for n in names) + " |")
    L += ["", "## 엔티티 타입 분포", "", "| 타입 | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    types = sorted({t for n in names for t in stats[n]["type_dist"]},
                   key=lambda t: -stats[names[0]]["type_dist"].get(t, 0))
    for t in types:
        L.append(f"| {t} | " + " | ".join(str(stats[n]["type_dist"].get(t, 0)) for n in names) + " |")
    L += ["", f"## 허브 엔티티 (차수 상위, {names[0]})", ""]
    L += [f"- {e} ({d})" for e, d in stats[names[0]]["hubs"]]
    (ROOT / "results" / "dataset_stats.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    figures(stats, ROOT / "results" / "figures")
    print("\n".join(L[:13]))


if __name__ == "__main__":
    main()
