"""D0 checks and table: exposure metrics of plain and penalty on the v2 pool (plan D0).

Checks: plain R@20 / nDCG@20 equal the C4 plain rows (results/c4_xquad), and plain
exp_gini > penalty (lam = 1) exp_gini in every city (seed mean and per seed).

Usage:
    python -m graph_retriever.d0_report --d0 results/d0_metrics --c4 results/c4_xquad \
        --out results/d0_metrics/table.md
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from graph_retriever.data import CITIES
from graph_retriever.tradeoff import load_rows

COLS = ("recall20", "ndcg20", "lt_hit20", "coverage", "exp_gini", "exp_entropy", "exp_top1pct",
        "exp_head_share", "arp")


def _ms(v) -> str:
    v = np.asarray(v, dtype=np.float64)
    return f"{v.mean():.4f} ± {v.std():.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d0", type=Path, required=True)
    ap.add_argument("--c4", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = ["# D0: metric phân tán exposure, plain và penalty (pool top-100 v2)", "", "Chỉ báo số.", "",
           "## Kiểm tra", ""]
    tables = []
    for city in CITIES:
        rows = load_rows(args.d0, city)
        seeds = sorted({r["seed"] for r in rows})
        for s in seeds:
            with (args.c4 / f"{city}_seed{s}.csv").open(encoding="utf-8") as f:
                c4 = {(r["split"]): r for r in csv.DictReader(f) if r["method"] == "plain"}
            for split in ("val", "test"):
                p = next(r for r in rows if r["seed"] == s and r["split"] == split and r["method"] == "plain")
                dr = abs(p["recall20"] - float(c4[split]["recall"]))
                dn = abs(p["ndcg20"] - float(c4[split]["ndcg"]))
                out.append(f"- {city} seed {s} {split}: plain R@20 {p['recall20']:.4f} (C4 {float(c4[split]['recall']):.4f}), "
                           f"nDCG@20 {p['ndcg20']:.4f} (C4 {float(c4[split]['ndcg']):.4f}), |Δ| {max(dr, dn):.1e}")
        gp = [next(r for r in rows if r["seed"] == s and r["split"] == "test" and r["method"] == "plain")["exp_gini"]
              for s in seeds]
        g1 = [next(r for r in rows if r["seed"] == s and r["split"] == "test" and r["method"] == "penalty"
                   and r["params"] == "lam=1")["exp_gini"] for s in seeds]
        ok = all(a > b for a, b in zip(gp, g1))
        out.append(f"- {city} test exp_gini: plain {_ms(gp)} vs penalty λ=1 {_ms(g1)}; plain > penalty ở mọi seed: {ok}")
        for split in ("test", "val"):
            tables += [f"### {city}, {split} (mean ± std, {len(seeds)} seed)", "",
                       "| method | params | " + " | ".join(COLS) + " |", "|---|---|" + "---|" * len(COLS)]
            keys = []
            for r in rows:
                if r["split"] == split and (r["method"], r["params"]) not in keys:
                    keys.append((r["method"], r["params"]))
            for m, p in keys:
                rs = [r for r in rows if r["split"] == split and r["method"] == m and r["params"] == p]
                tables.append(f"| {m} | {p} | " + " | ".join(_ms([r[c] for r in rs]) for c in COLS) + " |")
            tables.append("")
    out += ["", "## Bảng", ""] + tables
    args.out.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
