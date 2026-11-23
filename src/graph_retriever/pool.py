"""Candidate pool for the downstream reranker (the pre-rerank stage).

For each (city, seed) this trains the chosen retriever once, then exports for the val and
test splits the top-POOL_MAX candidates per sample, with scores, and a diversity/recall
report used to choose the pool size N on VALIDATION only.

Retriever: GRU + theme/area hypergraph (v2 design, leak-free).

Usage:
    uv run python -m graph_retriever.pool --raw <TripBuilderRawData> --out results/pool \
        --cities pisa florence rome --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from graph_retriever.data import CITIES, load_city
from graph_retriever.dataset import prepare
from graph_retriever.dispersal import rank_penalty
from graph_retriever.hypergraph import build_hypergraph
from graph_retriever.metrics import category_sets
from graph_retriever.train import TrainConfig, score_matrix, train_model

POOL_MAX = 100
POOL_SIZES = (10, 20, 50, 100)
DEEP_POOL_SIZES = (10, 20, 50, 100, 200, 300)  # report sizes allowed by --pool-max
PENALTY_LAMBDAS = (0.25, 1.0)  # dispersal-aware pools, reported alongside the plain pool
RETRIEVER_CHANNELS = ("theme", "area")


def _pool_stats(items: np.ndarray, targets: np.ndarray, area: np.ndarray,
                cats: list[frozenset[str]], n: int) -> dict[str, float]:
    top = items[:, :n]
    hit = (top == targets[:, None]).any(axis=1)
    distinct_area = []
    distinct_cat = []
    for row in top:
        areas = {int(area[i]) if area[i] >= 0 else -1 - int(i) for i in row}
        distinct_area.append(len(areas))
        distinct_cat.append(len(set().union(*(cats[i] for i in row))))
    return {
        "recall": float(hit.mean()),
        "distinct_area_mean": float(np.mean(distinct_area)),
        "distinct_category_mean": float(np.mean(distinct_cat)),
    }


def export_pool(city_name: str, raw_root: Path, seed: int, out: Path, tcfg: TrainConfig,
                pool_max: int = POOL_MAX, theme_split_area: bool = False) -> dict:
    city = load_city(raw_root, city_name)
    prep = prepare(city)
    cats = category_sets(city, prep.node_ids)
    hg = build_hypergraph(city, prep, RETRIEVER_CHANNELS, theme_split_area=theme_split_area)
    model, info = train_model(prep, hg, tcfg, seed, learn_gate=True)

    report: dict = {"city": city_name, "seed": seed, "best_epoch": info["best_epoch"],
                    "val_recall_model": info["best_val_recall"], "pool_max": pool_max,
                    "splits": {}}
    sizes = [n for n in DEEP_POOL_SIZES if n <= pool_max]
    out.mkdir(parents=True, exist_ok=True)
    for split, samples in (("val", prep.val), ("test", prep.test)):
        rel = score_matrix(model, samples)
        targets = np.array([s.target for s in samples])
        items = np.argsort(-rel, axis=1, kind="stable")[:, :pool_max]
        rel_top = np.take_along_axis(rel, items, axis=1)
        np.savez_compressed(out / f"{city_name}_seed{seed}_{split}.npz",
                            items=items.astype(np.int32), rel=rel_top.astype(np.float32),
                            targets=targets.astype(np.int32),
                            prefix_last=np.array([s.prefix[-1] for s in samples], dtype=np.int32),
                            ctx=np.array([s.ctx for s in samples], dtype=np.int8))
        split_report: dict = {"plain": {str(n): _pool_stats(items, targets, hg.area, cats, n)
                                        for n in sizes}}
        for lam in PENALTY_LAMBDAS:
            pen = np.stack([rank_penalty(rel[i], prep.pop, lam, pool_max) for i in range(len(samples))])
            split_report[f"penalty_{lam}"] = {str(n): _pool_stats(pen, targets, hg.area, cats, n)
                                              for n in sizes}
        report["splits"][split] = split_report
    return report


CANDIDATE_LIST_SIZE = 50  # chosen on validation: see plan/pool_interface.md


def write_candidates(city_name: str, raw_root: Path, npz_path: Path, out_path: Path,
                     n: int = CANDIDATE_LIST_SIZE) -> int:
    """One JSON line per query for the reranker. Contains NO target (no label leakage)."""
    city = load_city(raw_root, city_name)
    prep = prepare(city)
    cats_of = {i: sorted(city.clusters[cid].categories) for i, cid in enumerate(prep.node_ids)}
    split = "val" if "_val" in npz_path.stem else "test"
    data = np.load(npz_path)
    items, rel = data["items"], data["rel"]
    samples = prep.val if split == "val" else prep.test

    def node(i: int) -> dict:
        c = city.clusters[prep.node_ids[i]]
        return {"item_id": prep.node_ids[i], "name": c.name, "categories": cats_of[i],
                "lat": c.lat, "lon": c.lon}

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for q, smp in enumerate(samples):
            cand = []
            for rank, (i, r) in enumerate(zip(items[q, :n], rel[q, :n]), start=1):
                entry = node(int(i))
                entry.update({"rank": rank, "retriever_score": round(float(r), 4),
                              "pop": round(float(prep.pop[int(i)]), 4)})
                cand.append(entry)
            line = {
                "query_id": f"{city_name}-{npz_path.stem}-{q}",
                "city": city_name,
                "split": split,
                "history": [node(v) for v in smp.prefix],
                "time_bucket": int(smp.ctx),
                "candidates": cand,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            written += 1
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/pool"))
    ap.add_argument("--cities", nargs="+", default=list(CITIES), choices=CITIES)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--candidates", action="store_true",
                    help="write reranker input JSONL from existing pool .npz files (no training)")
    ap.add_argument("--n", type=int, default=CANDIDATE_LIST_SIZE)
    ap.add_argument("--pool-max", type=int, default=POOL_MAX,
                    help="store the top-P candidates per query (default 100; C2 uses 300)")
    ap.add_argument("--logq-tau", type=float, default=0.0,
                    help="plan D6: logit adjustment strength during training (0 = v2)")
    ap.add_argument("--future-gamma", type=float, default=0.0,
                    help="plan D8: weight decay of later visits as extra targets (0 = v2)")
    ap.add_argument("--future-horizon", type=int, default=3)
    ap.add_argument("--disp-beta", type=float, default=0.0,
                    help="plan D8: batch dispersion regulariser weight (0 = v2)")
    ap.add_argument("--theme-split-area", action="store_true",
                    help="plan D9: split broad themes by area instead of dropping them")
    args = ap.parse_args()

    if args.candidates:
        cand_dir = args.out.parent / "candidates"
        cand_dir.mkdir(parents=True, exist_ok=True)
        for c in args.cities:
            for seed in args.seeds:
                for split in ("val", "test"):
                    npz = args.out / f"{c}_seed{seed}_{split}.npz"
                    out = cand_dir / f"{c}_seed{seed}_{split}_n{args.n}.jsonl"
                    count = write_candidates(c, args.raw, npz, out, n=args.n)
                    print(f"[{c}] seed={seed} {split}: {count} queries -> {out}", flush=True)
        return

    tcfg = TrainConfig(encoder="gru", residual=True, max_epochs=args.max_epochs,
                       logq_tau=args.logq_tau, future_gamma=args.future_gamma,
                       future_horizon=args.future_horizon, disp_beta=args.disp_beta)
    for c in args.cities:
        for seed in args.seeds:
            rep = export_pool(c, args.raw, seed, args.out, tcfg, pool_max=args.pool_max,
                              theme_split_area=args.theme_split_area)
            (args.out / f"{c}_seed{seed}_report.json").write_text(
                json.dumps(rep, indent=2), encoding="utf-8")
            print(f"[{c}] seed={seed} exported pool (val R@20 model={rep['val_recall_model']:.4f})",
                  flush=True)


if __name__ == "__main__":
    main()
