"""Long-tail Hit@20 of HGNN v2 against GRU (no graph), per seed (plan C3), and the recheck of
the 'unseen in train' share quoted on the progress slide (C3, item 2).

Per query: a hit is 1 when the target is in the top 20 of the ranking. Pairs are the same
queries under both methods. McNemar exact is run per seed; Holm-Bonferroni runs over the
3 seeds of one city. The bootstrap CI covers the mean difference in Hit@20 (paired, over
queries). Query groups:
- tail: target is a long-tail POI (80% least-visited POIs by train count)
- tail_head_prefix: long-tail target and the prefix contains at least one head POI

Usage:
    python -m graph_retriever.tail_significance --raw <TripBuilderRawData> \
        --pool results/pool --scores results/scores --out results/c3_a4_significance
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from graph_retriever.data import CITIES, load_city
from graph_retriever.dataset import prepare
from graph_retriever.significance import (
    bootstrap_ci,
    holm,
    load_baseline,
    load_pool,
    mcnemar_exact,
    per_query,
)

K = 20
SEEDS = (0, 1, 2)
BASELINE_STEM = "gru"  # GRU (no graph), the same score files as A2 and A4


def group_masks(prep, samples, targets: np.ndarray) -> dict[str, np.ndarray]:
    tgt_tail = prep.tail[targets]
    has_head = np.array([any(not prep.tail[v] for v in s.prefix) for s in samples], dtype=bool)
    return {"tail": tgt_tail, "tail_head_prefix": tgt_tail & has_head}


def paired_rows(city: str, raw: Path, pool: Path, scores: Path) -> tuple[list[dict], dict]:
    prep = prepare(load_city(raw, city))
    rows: list[dict] = []
    pooled: dict[str, dict[str, list]] = {g: {"v2": [], "gru": []} for g in ("tail", "tail_head_prefix")}
    for seed in SEEDS:
        v2, targets = load_pool(pool, city, seed)
        gru = load_baseline(scores, BASELINE_STEM, city, seed)
        if gru is None:
            raise RuntimeError(f"missing {BASELINE_STEM}_{city}_seed{seed}.npy")
        if gru.shape[0] != v2.shape[0]:
            raise RuntimeError(f"{city} seed {seed}: query count differs ({gru.shape[0]} vs {v2.shape[0]})")
        expected = np.array([s.target for s in prep.test], dtype=np.int64)
        if not np.array_equal(targets, expected):
            raise RuntimeError(f"{city} seed {seed}: pool targets differ from prepare()")
        v2_hit, _ = per_query(v2, targets, K)
        gru_hit, _ = per_query(gru, targets, K)
        masks = group_masks(prep, prep.test, targets)
        for g, m in masks.items():
            a, b = v2_hit[m], gru_hit[m]
            diff = a - b
            ci_lo, ci_hi = bootstrap_ci(diff.astype(np.float64))
            rows.append({"city": city, "seed": seed, "group": g, "n": int(m.sum()),
                         "v2": float(a.mean()), "gru": float(b.mean()), "delta": float(diff.mean()),
                         "ci": (ci_lo, ci_hi), "p": mcnemar_exact(a, b),
                         "n_v2_only": int((a.astype(bool) & ~b.astype(bool)).sum()),
                         "n_gru_only": int((~a.astype(bool) & b.astype(bool)).sum())})
            pooled[g]["v2"].append(a)
            pooled[g]["gru"].append(b)
    # Holm over the seeds of each (city, group)
    for g in ("tail", "tail_head_prefix"):
        idx = [i for i, r in enumerate(rows) if r["group"] == g]
        adj = holm([rows[i]["p"] for i in idx])
        for i, pa in zip(idx, adj):
            rows[i]["p_holm"] = float(pa)
    pooled_out = {g: {"n": int(sum(x.size for x in pooled[g]["v2"])),
                      "v2": float(np.concatenate(pooled[g]["v2"]).mean()),
                      "gru": float(np.concatenate(pooled[g]["gru"]).mean())}
                  for g in pooled}
    return rows, pooled_out


def unseen_rows(city: str, raw: Path) -> list[dict]:
    prep = prepare(load_city(raw, city))
    out = []
    for split, samples in (("val", prep.val), ("test", prep.test)):
        tgt = np.array([s.target for s in samples], dtype=np.int64)
        unseen = prep.pop_count[tgt] == 0
        tail = prep.tail[tgt]
        out.append({"city": city, "split": split, "queries": int(tgt.size),
                    "unseen_target": float(unseen.mean()), "n_unseen": int(unseen.sum()),
                    "tail_target": float(tail.mean()), "n_tail": int(tail.sum())})
    return out


def _fmt_p(p: float) -> str:
    return f"{p:.2e}" if p < 0.001 else f"{p:.4f}"


def render(sig: list[dict], pooled: dict[str, dict], unseen: list[dict]) -> str:
    lines = ["# C3: kiểm định long-tail Hit@20, v2 so với GRU (không đồ thị), và kiểm lại slide 4",
             "",
             (f"Hit@{K}, test. McNemar exact theo từng seed; Holm–Bonferroni trong từng (thành phố, nhóm), m = 3 seed. "
              "Δ = Hit(v2) − Hit(GRU), CI 95% bootstrap theo query."), ""]
    for city in CITIES:
        for g in ("tail", "tail_head_prefix"):
            rs = [r for r in sig if r["city"] == city and r["group"] == g]
            if not rs:
                continue
            lines += [f"## {city}, nhóm `{g}`", "",
                      "| seed | n | Hit v2 | Hit GRU | Δ | CI 95% | v2 đúng, GRU sai | GRU đúng, v2 sai | p McNemar | p Holm |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
            for r in rs:
                lines.append(
                    f"| {r['seed']} | {r['n']} | {r['v2']:.4f} | {r['gru']:.4f} | {r['delta']:+.4f} | "
                    f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] | {r['n_v2_only']} | {r['n_gru_only']} | "
                    f"{_fmt_p(r['p'])} | {_fmt_p(r['p_holm'])} |")
            lines.append("")
    lines += ["## Gộp 3 seed (để đối chiếu với A4)", "",
              "| Nhóm | Thành phố | n (gộp) | Hit v2 | Hit GRU |", "|---|---|---|---|---|"]
    for city, groups in pooled.items():
        for g in ("tail", "tail_head_prefix"):
            p = groups[g]
            lines.append(f"| {g} | {city} | {p['n']} | {p['v2']:.4f} | {p['gru']:.4f} |")
    lines += ["", "## Slide 4: target không có trong train và target long-tail", "",
              "| thành phố | split | số query | target chưa thấy trong train | target long-tail |",
              "|---|---|---|---|---|"]
    for r in unseen:
        lines.append(f"| {r['city']} | {r['split']} | {r['queries']} | "
                     f"{r['unseen_target']*100:.2f}% ({r['n_unseen']}) | "
                     f"{r['tail_target']*100:.2f}% ({r['n_tail']}) |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--pool", type=Path, default=Path("results/pool"))
    ap.add_argument("--scores", type=Path, default=Path("results/scores"))
    ap.add_argument("--cities", nargs="+", default=list(CITIES), choices=CITIES)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sig: list[dict] = []
    pooled: dict[str, dict] = {}
    unseen: list[dict] = []
    for city in args.cities:
        rows, pl = paired_rows(city, args.raw, args.pool, args.scores)
        sig.extend(rows)
        pooled[city] = pl
        unseen.extend(unseen_rows(city, args.raw))
        print(f"[{city}] done", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "table.md").write_text(render(sig, pooled, unseen), encoding="utf-8")
    print(f"wrote {args.out / 'table.md'}", flush=True)


if __name__ == "__main__":
    main()
