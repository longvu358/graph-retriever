"""Novelty and aspect-diversity report for every method (plan A3).

Lists are the top-K (K = 20) per query. Popularity and Markov are computed here from the
train split. Retriever methods are read from the saved score matrices (`{stem}_{city}_seed{s}.npy`)
and from the v2 pool (`{city}_seed{s}_test.npz`). Penalty rows need the full v2 score
matrix, which is not saved, so they are not in this report.

Usage:
    python -m graph_retriever.novelty_report --raw <TripBuilderRawData> \
        --pool results/pool --scores results/scores --out results/a3_novelty
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from graph_retriever.data import CITIES, load_city
from graph_retriever.dataset import prepare
from graph_retriever.experiment import markov_scores
from graph_retriever.hypergraph import build_hypergraph
from graph_retriever.metrics import (
    alpha_ndcg,
    category_sets,
    dispersal_metrics,
    novelty_pop,
    unseen_fraction,
)
from graph_retriever.significance import topk

K = 20
SCORE_METHODS = {"SR-GNN": "srgnn", "GCE-GNN": "gcegnn", "DHCN": "dhcn",
                 "GRU (no graph)": "gru", "GRU4Rec": "gru4rec"}
ALPHA = 0.5


def _alpha_mean(rank: np.ndarray, targets: np.ndarray, relevant_of) -> float:
    vals = []
    for row, t in zip(rank, targets):
        rel = relevant_of(int(t))
        if rel is None:
            continue
        vals.append(alpha_ndcg(row.tolist(), rel, K, ALPHA))
    return float(np.mean(vals)) if vals else float("nan")


def method_row(rank: np.ndarray, targets: np.ndarray, prep, area, cats, pop_count) -> dict:
    lists = [row.tolist() for row in rank[:, :K]]
    dm = dispersal_metrics(lists, prep, area, cats)

    def by_category(t: int):
        return {t: cats[t]} if cats[t] else None

    def by_area(t: int):
        return {t: frozenset({f"area{area[t]}"})} if area[t] >= 0 else None

    return {
        "novelty_pop": novelty_pop(lists, pop_count),
        "unseen_frac": unseen_fraction(lists, pop_count),
        "ILD": dm["ILD"],
        "area_entropy": dm["area_entropy"],
        "coverage": dm["coverage"],
        "alpha_ndcg_category": _alpha_mean(rank[:, :K], targets, by_category),
        "alpha_ndcg_area": _alpha_mean(rank[:, :K], targets, by_area),
    }


COLUMNS = ("novelty_pop", "unseen_frac", "ILD", "area_entropy", "coverage",
           "alpha_ndcg_category", "alpha_ndcg_area")


def render(table: dict[str, dict[str, list[dict]]]) -> str:
    lines = ["# A3: metric novelty", "",
             f"Top-{K} mỗi query, mean ± std theo seed. α-nDCG: α = {ALPHA}, relevance = target.",
             ("Lưu ý: với một target mỗi query, α-nDCG không đo diversity của list "
              "(xem docstring của `alpha_ndcg`). Diversity chính là ILD, area_entropy, coverage."),
             "`novelty_pop` cao = item hiếm hơn. `unseen_frac` = tỉ lệ item không có trong train.", ""]
    for city, methods in table.items():
        lines += [f"## {city}", "", "| Method | " + " | ".join(COLUMNS) + " | seeds |",
                  "|---|" + "---|" * (len(COLUMNS) + 1)]
        for method, rows in methods.items():
            cells = []
            for c in COLUMNS:
                vals = np.array([r[c] for r in rows], dtype=np.float64)
                cells.append(f"{np.nanmean(vals):.4f} ± {np.nanstd(vals):.4f}")
            lines.append(f"| {method} | " + " | ".join(cells) + f" | {len(rows)} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--pool", type=Path, default=Path("results/pool"))
    ap.add_argument("--scores", type=Path, default=Path("results/scores"))
    ap.add_argument("--out", type=Path, default=Path("results/a3_novelty"))
    ap.add_argument("--cities", nargs="+", default=list(CITIES), choices=CITIES)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()

    table: dict[str, dict[str, list[dict]]] = {}
    for city in args.cities:
        data = load_city(args.raw, city)
        prep = prepare(data)
        cats = category_sets(data, prep.node_ids)
        area = build_hypergraph(data, prep, channels=("area",)).area
        targets = np.array([s.target for s in prep.test], dtype=np.int64)
        methods: dict[str, list[dict]] = {"Popularity": [], "Markov": [], "HGNN v2": []}
        for name in SCORE_METHODS:
            methods[name] = []
        pop_rank = np.argsort(-prep.pop_count, kind="stable")
        pop_all = np.tile(pop_rank, (targets.size, 1))
        markov = topk(markov_scores(prep, prep.test), 100)

        for seed in args.seeds:
            # Popularity and Markov do not depend on the seed.
            z = np.load(args.pool / f"{city}_seed{seed}_test.npz")
            if not np.array_equal(z["targets"].astype(np.int64), targets):
                raise SystemExit(f"[{city}] seed={seed}: pool targets do not match prep.test")
            methods["HGNN v2"].append(method_row(z["items"].astype(np.int64), targets, prep, area, cats, prep.pop_count))
            for name, stem in SCORE_METHODS.items():
                path = args.scores / f"{stem}_{city}_seed{seed}.npy"
                if path.exists():
                    rank = topk(np.load(path), 100)
                    methods[name].append(method_row(rank, targets, prep, area, cats, prep.pop_count))
        methods["Popularity"].append(method_row(pop_all, targets, prep, area, cats, prep.pop_count))
        methods["Markov"].append(method_row(markov, targets, prep, area, cats, prep.pop_count))
        table[city] = {m: rows for m, rows in methods.items() if rows}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.md").write_text(render(table), encoding="utf-8")
    print(f"wrote {args.out / 'summary.md'}")


if __name__ == "__main__":
    main()
