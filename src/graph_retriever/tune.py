"""Select dispersal settings on VALIDATION only, then report the chosen settings on test.

Rule (fixed before looking at test):
  - reference = penalty with lambda = 0 (pure relevance), val Recall@20, mean over seeds
  - feasible  = val Recall@20 >= reference - ACCURACY_BUDGET
  - for each method (penalty, or ours with a given beta/delta), choose the lambda with the
    highest val coverage among feasible lambdas
  - across methods, pick the highest val coverage among the chosen settings

Usage:
    uv run python -m graph_retriever.tune results/tune results/tune_summary.md
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ACCURACY_BUDGET = 0.01  # max absolute Recall@20 loss on val, relative to lambda = 0
ILD_BUDGET = 0.02  # used only by the ILD-constrained rule: max ILD loss on val vs lambda = 0
CITIES = ("pisa", "florence", "rome")
REPORT = ("recall", "ndcg", "ILD", "coverage", "long_tail_ratio", "ARP")
# Pool-level rows (plans C4, C5) also carry these; read only when present.
POOL_METRICS = ("tail_hit", "recall_hp", "tail_hit_hp")
CRITERIA = ("coverage", "tail_hit")  # R1: coverage (as tune_v2); R2: long-tail Hit@20


def _load(tune_dir: Path, city: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(tune_dir.glob(f"{city}_seed*.csv")):
        with path.open() as f:
            rows.extend(csv.DictReader(f))
    return rows


def _mean_std(vals: list[float]) -> str:
    a = np.array(vals)
    return f"{a.mean():.3f} ± {a.std():.3f}"


def select(rows: list[dict], ild_constrained: bool = False,
           criterion: str = "coverage") -> list[tuple[str, float, dict]]:
    """Return [(method, lambda, {split: {metric: [per-seed values]}})] for each chosen setting.

    `criterion` is the val metric maximised among feasible lambdas (and across methods)."""
    if criterion not in CRITERIA:
        raise ValueError(criterion)
    by_key: dict[tuple[str, str, float], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["split"], r["method"], float(r["lam"]))
        for m in ("recall", "coverage", "ndcg", "ILD", "long_tail_ratio", "ARP") + POOL_METRICS:
            if r.get(m) not in ("", None):
                by_key[key][m].append(float(r[m]))

    def mean(split: str, method: str, lam: float, metric: str) -> float:
        return float(np.mean(by_key[(split, method, lam)][metric]))

    ref_ild = mean("val", "penalty", 0.0, "ILD")

    methods = sorted({m for (_, m, _) in by_key})
    lambdas_of = {m: sorted({l for (_, mm, l) in by_key if mm == m}) for m in methods}
    ref = mean("val", "penalty", 0.0, "recall")

    chosen: list[tuple[str, float, float]] = []  # (method, lam, val coverage)
    for method in methods:
        feasible = [
            l for l in lambdas_of[method]
            if mean("val", method, l, "recall") >= ref - ACCURACY_BUDGET
            and (not ild_constrained or mean("val", method, l, "ILD") >= ref_ild - ILD_BUDGET)
        ]
        if not feasible:
            continue
        best_lam = max(feasible, key=lambda l: mean("val", method, l, criterion))
        chosen.append((method, best_lam, mean("val", method, best_lam, criterion)))

    out = []
    for method, lam, _ in sorted(chosen, key=lambda x: -x[2]):
        report = {split: {m: by_key[(split, method, lam)][m] for m in REPORT + POOL_METRICS}
                  for split in ("val", "test")}
        out.append((method, lam, report))
    return out


def summarize(tune_dir: Path, city: str, ild_constrained: bool = False) -> str:
    rows = _load(tune_dir, city)
    if not rows:
        return f"### {city}: no tune rows\n"
    chosen = select(rows, ild_constrained=ild_constrained)
    rule = "val coverage, + val ILD constraint" if ild_constrained else "val coverage"
    n_seeds = len({r["seed"] for r in rows})
    lines = [
        f"### {city} (K=20, val-selected by {rule}, budget={ACCURACY_BUDGET}, seeds={n_seeds})",
        "",
        "| method | λ | val R@20 | val coverage | test R@20 | test nDCG | test ILD | test coverage | test long-tail | test ARP |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for method, lam, rep in chosen:
        lines.append(
            f"| {method} | {lam} | {_mean_std(rep['val']['recall'])} | {_mean_std(rep['val']['coverage'])} | "
            f"{_mean_std(rep['test']['recall'])} | {_mean_std(rep['test']['ndcg'])} | "
            f"{_mean_std(rep['test']['ILD'])} | {_mean_std(rep['test']['coverage'])} | "
            f"{_mean_std(rep['test']['long_tail_ratio'])} | {_mean_std(rep['test']['ARP'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(tune_dir: Path, out_path: Path) -> None:
    parts = []
    for ild in (False, True):
        parts += [summarize(tune_dir, c, ild_constrained=ild) for c in CITIES]
    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))


def render_rows(rows: list[dict], label: str, criterion: str = "coverage") -> str:
    """Markdown table of the selected setting per method, from pool-level rows (plans C4, C5)."""
    chosen = select(rows, criterion=criterion)
    lines = [
        f"### {label} (K=20, pool top-100, val-selected by val {criterion}, budget={ACCURACY_BUDGET})",
        "",
        "| method | λ | val R@20 | val " + criterion + " | test R@20 | test nDCG | test long-tail Hit@20 | "
        "test coverage | test long-tail ratio | test ILD | test ARP | test R@20 (head in prefix) | "
        "test long-tail Hit@20 (head in prefix) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for method, lam, rep in chosen:
        t = rep["test"]
        lines.append(
            f"| {method} | {lam} | {_mean_std(rep['val']['recall'])} | "
            f"{_mean_std(rep['val'][criterion])} | {_mean_std(t['recall'])} | {_mean_std(t['ndcg'])} | "
            f"{_mean_std(t['tail_hit'])} | {_mean_std(t['coverage'])} | {_mean_std(t['long_tail_ratio'])} | "
            f"{_mean_std(t['ILD'])} | {_mean_std(t['ARP'])} | {_mean_std(t['recall_hp'])} | "
            f"{_mean_std(t['tail_hit_hp'])} |"
        )
    lines.append("")
    return "\n".join(lines)
