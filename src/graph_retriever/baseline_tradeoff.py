"""Plan D5: trade-off curves of every retriever under the same popularity penalty. CPU only.

For each retriever (v2 = ours, gru = no-graph control, gru4rec, srgnn, gcegnn, dhcn):
  rel  = log-softmax of its test score matrix (results/scores/{stem}_{city}_seed{s}.npy);
         v2 reads its stored pool (results/pool), which is already log-softmax top-100.
  pool = top-100 by rel, as for v2.
  list = top-20 of rel - lam * pop inside the pool, lam in D2_LAMBDAS (lam = 0 = the retriever).
Without --with-val only test scores are used (plan D5, no val selection). With --with-val
(plan D6, scores re-exported with val) val rows are written too, so plot_metrics and
pair_ci select settings on the val Pareto front. Test lists of every setting go to lists/.
Metrics = pool_rerank.tradeoff_metrics with the D4 novelty columns.

Output: {out}/{city}_seed{s}.csv with method = "penalty", variant = retriever (curve key
penalty_<retriever>), split = test. Plot with graph_retriever.plot_metrics.

Usage:
    python -m graph_retriever.baseline_tradeoff --raw <TripBuilderRawData> --pool results/pool \
        --scores results/scores --out results/d5_baseline_tradeoff --cities pisa
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from graph_retriever.data import CITIES, load_city
from graph_retriever.dataset import prepare
from graph_retriever.metrics import (
    category_sets,
    cosine_distance_matrix,
    jaccard_similarity_matrix,
)
from graph_retriever.pool_rerank import (
    D2_LAMBDAS,
    NOVELTY_FIELDS,
    TRADEOFF_FIELDS,
    K,
    _fmt,
    novelty_context,
    tradeoff_metrics,
)

POOL = 100
BASELINES = ("gru", "gru4rec", "srgnn", "gcegnn", "dhcn")
FIELDS = TRADEOFF_FIELDS + NOVELTY_FIELDS


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


def pool_from_scores(scores: np.ndarray, width: int = POOL) -> tuple[np.ndarray, np.ndarray]:
    rel = log_softmax(scores)
    items = np.argsort(-rel, axis=1, kind="stable")[:, :width]
    return items, np.take_along_axis(rel, items, axis=1)


def penalty_lists(items: np.ndarray, rel: np.ndarray, pop: np.ndarray, lam: float) -> np.ndarray:
    order = np.argsort(-(rel - lam * pop[items]), axis=1, kind="stable")[:, :K]
    return np.take_along_axis(items, order, axis=1)


def _pool_npz(path: Path, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    if not np.array_equal(z["targets"].astype(np.int64), targets):
        raise RuntimeError(f"{path}: targets differ from prepare()")
    return z["items"].astype(np.int64)[:, :POOL], z["rel"].astype(np.float64)[:, :POOL]


def _pool_scores(path: Path, n_rows: int, n_items: int) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    s = np.load(path)
    if s.shape != (n_rows, n_items):
        raise RuntimeError(f"{path.name}: shape {s.shape}, expected {(n_rows, n_items)}")
    return pool_from_scores(s)


def load_pools(args, city: str, seed: int, split: str, targets: np.ndarray, n_items: int) -> dict:
    """name -> (items, rel) for one split. `--pools name=dir` are v2-style pool folders;
    baselines come from `--scores` ({stem}_{city}_seed{s}.npy for test, ..._val.npy for val)."""
    pools = {}
    for spec in args.pools:
        name, d = spec.split("=", 1)
        path = Path(d) / f"{city}_seed{seed}_{split}.npz"
        if path.exists():
            pools[name] = _pool_npz(path, targets)
        else:
            print(f"[{city}] seed={seed} {split}: missing {path}, skipped", flush=True)
    for stem in args.baselines:
        suffix = "_val" if split == "val" else ""
        got = _pool_scores(args.scores / f"{stem}_{city}_seed{seed}{suffix}.npy", len(targets), n_items)
        if got is None:
            print(f"[{city}] seed={seed} {split}: missing {stem} scores, skipped", flush=True)
            continue
        pools[stem] = got
    return pools


def run_city(args, city: str) -> None:
    cityobj = load_city(args.raw, city)
    prep = prepare(cityobj)
    cat_sim = jaccard_similarity_matrix(category_sets(cityobj, prep.node_ids))
    cos_d = cosine_distance_matrix(prep.train_sessions, prep.n_items)
    splits = ("val", "test") if args.with_val else ("test",)
    nov = {sp: novelty_context(prep, [s.prefix for s in (prep.val if sp == "val" else prep.test)],
                               cos_d, cat_sim) for sp in splits}
    (args.out / "lists").mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        rows, saved = [], {}
        for split in splits:
            samples = prep.val if split == "val" else prep.test
            targets = np.array([s.target for s in samples], dtype=np.int64)
            pools = load_pools(args, city, seed, split, targets, prep.n_items)
            for name, (items, rel) in pools.items():
                for lam in D2_LAMBDAS:
                    lists = penalty_lists(items, rel, prep.pop, lam)
                    params = f"lam={_fmt(lam)}"
                    row = {"city": city, "method": "penalty", "variant": name, "params": params,
                           "seed": seed, "split": split, "n_swaps": 0.0}
                    row.update(tradeoff_metrics(lists, targets, prep, nov[split]))
                    rows.append(row)
                    if split == "test":
                        saved[f"penalty_{name}|{params}"] = lists.astype(np.int16)
        path = args.out / f"{city}_seed{seed}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in FIELDS})
        keys = list(saved)
        np.savez_compressed(args.out / "lists" / f"{city}_seed{seed}.npz",
                            keys=np.array(json.dumps(keys)),
                            targets=np.array([s.target for s in prep.test]),
                            n_items=np.array(prep.n_items),
                            **{f"c{j}": saved[k] for j, k in enumerate(keys)})
        print(f"[{city}] seed={seed}: {len(rows)} rows, wrote {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--pools", nargs="*", default=["v2=results/pool"],
                    help="name=dir of v2-style pool .npz folders (val and test)")
    ap.add_argument("--scores", type=Path, default=Path("results/scores"))
    ap.add_argument("--baselines", nargs="*", default=list(BASELINES))
    ap.add_argument("--with-val", action="store_true",
                    help="plan D6: also score val (needs {stem}_..._val.npy), enables val selection")
    ap.add_argument("--out", type=Path, default=Path("results/d5_baseline_tradeoff"))
    ap.add_argument("--cities", nargs="+", default=list(CITIES), choices=CITIES)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for city in args.cities:
        run_city(args, city)


if __name__ == "__main__":
    main()
