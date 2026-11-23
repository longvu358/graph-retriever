"""Select pool-level settings on VALIDATION and report test (plans C4, C5).

Reads the per-(city, seed) CSVs written by graph_retriever.pool_rerank from one or more
directories, then renders, per city:
  R1: maximise val coverage among feasible settings (tune_v2 rule)
  R2: maximise val long-tail Hit@20 among feasible settings
Feasible = val Recall@20 >= plain val Recall@20 - ACCURACY_BUDGET (plain = penalty lam 0).
The test columns are reported for the chosen setting only. Also checks that every
gamma = 0 row of the prefix methods equals the plain row on test, seed by seed.

Usage:
    python -m graph_retriever.pool_select --dirs results/c4_xquad results/c5_prefix_subst \
        --out results/c5_prefix_subst/table.md
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from graph_retriever.data import CITIES
from graph_retriever.tune import CRITERIA, render_rows

NUMERIC = ("recall", "ndcg", "tail_hit", "recall_hp", "tail_hit_hp", "ILD", "coverage",
           "long_tail_ratio", "ARP", "area_entropy", "lam")


def load(dirs: list[Path], city: str) -> list[dict]:
    rows: list[dict] = []
    for d in dirs:
        for path in sorted(d.glob(f"{city}_seed*.csv")):
            with path.open(encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    for k in NUMERIC:
                        if r.get(k) not in (None, ""):
                            r[k] = float(r[k])
                    rows.append(r)
    return rows


def gamma_zero_check(rows: list[dict]) -> list[str]:
    """Each prefix method at gamma = 0 must reproduce the plain row (test split, per seed)."""
    plain = {(r["seed"], r["split"]): r for r in rows if r["method"] == "plain"}
    lines = []
    for r in rows:
        if r["method"].startswith("prefix_") and r["lam"] == 0.0 and r["split"] == "test":
            p = plain.get((r["seed"], "test"))
            if p is None:
                continue
            diff = max(abs(r[k] - p[k]) for k in ("recall", "ndcg", "tail_hit", "coverage",
                                                    "long_tail_ratio", "ARP"))
            lines.append(f"seed {r['seed']} {r['method']}: max |Δ| vs plain = {diff:.2e}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    parts = ["# Chọn tham số trên val, báo test (pool top-100 của v2)", "",
             ("R1: tối đa val coverage. R2: tối đa val long-tail Hit@20. Ràng buộc: val R@20 ≥ plain − 0.01. "
              "Chỉ báo số."), ""]
    for city in CITIES:
        rows = load(args.dirs, city)
        if not rows:
            continue
        for criterion in CRITERIA:
            parts.append(render_rows(rows, f"{city}, {criterion}", criterion=criterion))
        checks = gamma_zero_check(rows)
        if checks:
            parts += [f"Kiểm tra γ = 0 ({city}):", ""] + [f"- {c}" for c in checks] + [""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
