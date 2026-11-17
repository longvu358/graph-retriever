"""Run plan v0 experiments for one or more cities and write CSV results.

Usage:
    uv run python -m graph_retriever.experiment --raw <TripBuilderRawData> --out results \
        --cities florence rome --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import numpy as np

from graph_retriever.data import CITIES, City, load_city
from graph_retriever.dataset import Prepared, prepare
from graph_retriever.dispersal import (
    DispersalConfig,
    rank_dispersal,
    rank_mmr,
    rank_popularity,
    share_matrix,
)
from graph_retriever.hypergraph import CHANNELS, Hypergraph, build_hypergraph
from graph_retriever.metrics import (
    category_sets,
    dispersal_metrics,
    jaccard_similarity_matrix,
    recall_ndcg,
)
from graph_retriever.train import TrainConfig, score_matrix, train_model

LAMBDAS = (0.0, 0.1, 0.25, 0.5, 1.0)
THETAS = (0.0, 0.1, 0.25, 0.5, 1.0)
PLAIN_KS = (10, 20, 50)
DISP_K = 20
ABLATION_CONFIGS = (
    # name, channels, learn_gate
    ("session", ("session",), True),
    ("session+theme", ("theme", "session"), True),
    ("session+theme+area", ("theme", "area", "session"), False),
    ("full", CHANNELS, True),
    ("full_uniform_gate", CHANNELS, False),
)


def markov_scores(prep: Prepared, samples) -> np.ndarray:
    n = prep.n_items
    trans = np.zeros((n, n))
    for s in prep.train_sessions:
        for a, b in pairwise(s):
            trans[a, b] += 1
    out = np.zeros((len(samples), n))
    eps = 0.01
    for i, smp in enumerate(samples):
        row = trans[smp.prefix[-1]] + eps
        out[i] = np.log(row / row.sum())
    return out


def plain_rows(
    city: str, seed: int | str, method: str, lam: float | None, rel: np.ndarray,
    prep: Prepared, targets: np.ndarray, area: np.ndarray, cats, extra: dict,
) -> list[dict]:
    rows = []
    max_k = max(PLAIN_KS)
    ranked = np.argsort(-rel, axis=1, kind="stable")[:, :max_k]
    for k in PLAIN_KS:
        rec, ndcg = recall_ndcg(ranked, targets, k)
        row = dict(city=city, seed=seed, method=method, lam=lam, K=k, recall=rec, ndcg=ndcg, **extra)
        if k == DISP_K:
            row.update(dispersal_metrics(ranked[:, :k].tolist(), prep, area, cats))
        rows.append(row)
    return rows


def dispersal_row(
    city: str, seed: int, method: str, lam: float, lists: np.ndarray, targets: np.ndarray,
    prep: Prepared, area: np.ndarray, cats, extra: dict,
) -> dict:
    rec, ndcg = recall_ndcg(lists, targets, DISP_K)
    row = dict(city=city, seed=seed, method=method, lam=lam, K=DISP_K, recall=rec, ndcg=ndcg, **extra)
    row.update(dispersal_metrics(lists.tolist(), prep, area, cats))
    return row


def run_city(city_name: str, raw_root: Path, seeds: list[int], tcfg: TrainConfig) -> list[dict]:
    city: City = load_city(raw_root, city_name)
    prep = prepare(city)
    cats = category_sets(city, prep.node_ids)
    sim = jaccard_similarity_matrix(cats)
    targets = np.array([s.target for s in prep.test])
    rows: list[dict] = []

    print(f"[{city_name}] nodes={prep.n_items} train={len(prep.train)} "
          f"val={len(prep.val)} test={len(prep.test)}", flush=True)

    pop_lists = np.tile(np.array(rank_popularity(prep.pop_count, max(PLAIN_KS))), (len(targets), 1))
    rows += [
        dict(city=city_name, seed="-", method="popularity", lam=None, K=k,
             **dict(zip(("recall", "ndcg"), recall_ndcg(pop_lists, targets, k))),
             **(dispersal_metrics(pop_lists[:, :k].tolist(), prep, np.full(prep.n_items, -1), cats)
                if k == DISP_K else {}))
        for k in PLAIN_KS
    ]
    rows += plain_rows(city_name, "-", "markov", None, markov_scores(prep, prep.test),
                       prep, targets, build_hypergraph(city, prep).area, cats, {})

    for seed in seeds:
        full_hg: Hypergraph = build_hypergraph(city, prep, CHANNELS)
        area = full_hg.area
        share = share_matrix(full_hg)
        full_rel = None
        for name, chans, learn in ABLATION_CONFIGS:
            hg = full_hg if chans == CHANNELS else build_hypergraph(city, prep, chans)
            model, info = train_model(prep, hg, tcfg, seed, learn_gate=learn)
            rel = score_matrix(model, prep.test)
            extra = {
                "best_epoch": info["best_epoch"],
                "val_recall": info["best_val_recall"],
                "gate": ",".join(f"{g:.3f}" for g in model.gate().tolist()),
            }
            rows += plain_rows(city_name, seed, name, None, rel, prep, targets, area, cats, extra)
            print(f"[{city_name}] seed={seed} {name}: best_epoch={info['best_epoch']} "
                  f"val_R@20={info['best_val_recall']:.4f} gate={extra['gate']}", flush=True)
            if name == "full":
                full_rel = rel

        assert full_rel is not None
        for lam in LAMBDAS:
            pen = np.argsort(-(full_rel - lam * prep.pop[None, :]), axis=1, kind="stable")[:, :DISP_K]
            rows.append(dispersal_row(city_name, seed, "penalty", lam, pen, targets, prep, area, cats, {}))

        for theta in THETAS:
            lists = np.array([rank_mmr(full_rel[i], DISP_K, sim, theta) for i in range(len(targets))])
            rows.append(dispersal_row(city_name, seed, "mmr", theta, lists, targets, prep, area, cats, {}))

        variants = {
            "ours": DispersalConfig(substitution=True, coverage=True, delta=1.0),
            "ours_d0.5": DispersalConfig(substitution=True, coverage=True, delta=0.5),
            "ours_d2.0": DispersalConfig(substitution=True, coverage=True, delta=2.0),
            "ours_no_sub": DispersalConfig(substitution=False, coverage=True),
            "ours_no_cov": DispersalConfig(substitution=True, coverage=False, delta=1.0),
        }
        for method, base in variants.items():
            for lam in LAMBDAS:
                cfg = replace(base, lam=lam)
                picks = [rank_dispersal(full_rel[i], DISP_K, full_hg, prep.pop, share, cfg)
                         for i in range(len(targets))]
                lists = np.array([[p.item for p in r.picks] for r in picks])
                subs = sum(p.substituted_for is not None for r in picks for p in r.picks)
                row_extra = {"substituted_slots": subs}
                rows.append(dispersal_row(city_name, seed, method, lam, lists, targets,
                                          prep, area, cats, row_extra))
                print(f"[{city_name}] seed={seed} {method} lam={lam}: "
                      f"done (substituted slots={subs})", flush=True)
    return rows


TUNE_LAMBDAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0)
TUNE_THETAS = (0.0, 0.1, 0.25, 0.5, 1.0)  # MMR trade-off, stored in the 'lam' column
TUNE_BETAS = (0.0, 0.01, 0.1, 1.0)
TUNE_DELTAS = (0.1, 0.25, 0.5, 1.0)


def run_tune(city_name: str, raw_root: Path, seed: int, tcfg: TrainConfig,
             v2: bool = False) -> list[dict]:
    """Full model once per (city, seed); sweep penalty and substitution grids on val AND test.

    Selection happens in graph_retriever.tune, on val only. Test rows are kept for the
    final report of the chosen configuration.
    """
    city = load_city(raw_root, city_name)
    prep = prepare(city)
    cats = category_sets(city, prep.node_ids)
    if v2:  # retriever v2: GRU + theme/area, no session channel (leak-free)
        from dataclasses import replace as dc_replace

        chans = ("theme", "area")
        tcfg = dc_replace(tcfg, encoder="gru", residual=True)
    else:
        chans = CHANNELS
    full_hg = build_hypergraph(city, prep, chans)
    share = share_matrix(full_hg)
    area = full_hg.area
    model, info = train_model(prep, full_hg, tcfg, seed, learn_gate=True)
    splits = {
        "val": (score_matrix(model, prep.val), np.array([s.target for s in prep.val])),
        "test": (score_matrix(model, prep.test), np.array([s.target for s in prep.test])),
    }
    extra = {"best_epoch": info["best_epoch"], "val_recall": info["best_val_recall"]}
    rows: list[dict] = []
    for split, (rel, targets) in splits.items():
        for lam in TUNE_LAMBDAS:
            pen = np.argsort(-(rel - lam * prep.pop[None, :]), axis=1, kind="stable")[:, :DISP_K]
            row = dispersal_row(city_name, seed, "penalty", lam, pen, targets, prep, area, cats, extra)
            row["split"] = split
            rows.append(row)
        sim = jaccard_similarity_matrix(cats)
        for theta in TUNE_THETAS:
            lists = np.array([rank_mmr(rel[i], DISP_K, sim, theta) for i in range(len(targets))])
            row = dispersal_row(city_name, seed, "mmr", theta, lists, targets, prep, area, cats, extra)
            row["split"] = split
            rows.append(row)
        for beta in TUNE_BETAS:
            for delta in TUNE_DELTAS:
                for lam in TUNE_LAMBDAS:
                    cfg = DispersalConfig(lam=lam, beta=beta, delta=delta,
                                          substitution=True, coverage=True)
                    lists = np.array([
                        [p.item for p in rank_dispersal(rel[i], DISP_K, full_hg, prep.pop, share, cfg).picks]
                        for i in range(len(targets))
                    ])
                    row = dispersal_row(city_name, seed, f"ours_b{beta}_d{delta}", lam, lists,
                                        targets, prep, area, cats, extra)
                    row["split"] = split
                    rows.append(row)
        print(f"[{city_name}] seed={seed} split={split} done", flush=True)
    return rows


V2_CONFIGS = (
    # name, channels, encoder, residual, session_edge, time_ctx
    ("v2_theme_area_gru", ("theme", "area"), "gru", True, False, False),
    ("v2_theme_area_attn", ("theme", "area"), "attn", True, False, False),
    ("v2_theme_gru", ("theme",), "gru", True, False, False),
    ("v2_area_gru", ("area",), "gru", True, False, False),
    ("v2_none_gru", (), "gru", True, False, False),  # control: GRU on plain embeddings
)

V3_CONFIGS = (
    # step 1: one hyperedge over the prefix (leak-free); step 2: time-of-day/weekend context
    ("v3_ref_gru", ("theme", "area"), "gru", True, False, False),
    ("v3_sedge", ("theme", "area"), "gru", True, True, False),
    ("v3_time", ("theme", "area"), "gru", True, False, True),
    ("v3_sedge_time", ("theme", "area"), "gru", True, True, True),
)


V4_CONFIGS = (
    # step 3: DHCN-style two-view InfoNCE (hypergraph view vs clique view), theme+area only
    ("v4_ssl0", ("theme", "area"), "gru", True, False, False, 0.0),
    ("v4_ssl01", ("theme", "area"), "gru", True, False, False, 0.01),
    ("v4_ssl1", ("theme", "area"), "gru", True, False, False, 0.1),
)


def run_v2(
    city_name: str, raw_root: Path, seeds: list[int], tcfg: TrainConfig,
    only: list[str] | None = None, configs_all: tuple = V2_CONFIGS,
) -> list[dict]:
    """Leak-free designs: no session co-occurrence channel; theme/area only (no target info)."""
    from dataclasses import replace as dc_replace

    city = load_city(raw_root, city_name)
    prep = prepare(city)
    cats = category_sets(city, prep.node_ids)
    rows: list[dict] = []
    configs = [c for c in configs_all if only is None or c[0] in only]
    for seed in seeds:
        for name, chans, encoder, residual, sedge, tctx, *extra_fields in configs:
            ssl_beta = extra_fields[0] if extra_fields else 0.0
            hg = build_hypergraph(city, prep, chans)
            cfg = dc_replace(tcfg, encoder=encoder, residual=residual,
                             session_edge=sedge, time_ctx=tctx, ssl_beta=ssl_beta)
            model, info = train_model(prep, hg, cfg, seed, learn_gate=True)
            rel = score_matrix(model, prep.test)
            targets = np.array([s.target for s in prep.test])
            extra = {
                "best_epoch": info["best_epoch"],
                "val_recall": info["best_val_recall"],
                "gate": ",".join(f"{g:.3f}" for g in model.gate().tolist()),
            }
            rows += plain_rows(city_name, seed, name, None, rel, prep, targets, hg.area, cats, extra)
            print(f"[{city_name}] seed={seed} {name}: best_epoch={info['best_epoch']} "
                  f"val_R@20={info['best_val_recall']:.4f}", flush=True)
    return rows


def run_baselines(
    city_name: str, raw_root: Path, seeds: list[int], models: list[str], scores_dir: Path,
) -> list[dict]:
    """Evaluate third-party retrievers from score files written by graph_retriever.baselines."""
    from scipy.special import logsumexp

    city = load_city(raw_root, city_name)
    prep = prepare(city)
    cats = category_sets(city, prep.node_ids)
    sim = jaccard_similarity_matrix(cats)
    targets = np.array([s.target for s in prep.test])
    area = build_hypergraph(city, prep, channels=("area",)).area
    rows: list[dict] = []
    for seed in seeds:
        for model in models:
            path = scores_dir / f"{model}_{city_name}_seed{seed}.npy"
            if not path.exists():
                print(f"[{city_name}] missing {path}, skipped", flush=True)
                continue
            info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            raw = np.load(path).astype(np.float64)
            rel = raw - logsumexp(raw, axis=1, keepdims=True)
            extra = {"best_epoch": info["best_epoch"], "val_recall": info["best_val_recall"]}
            plain = plain_rows(city_name, seed, model, None, rel, prep, targets, area, cats, extra)
            rows += plain
            for lam in LAMBDAS:
                pen = np.argsort(-(rel - lam * prep.pop[None, :]), axis=1, kind="stable")[:, :DISP_K]
                rows.append(dispersal_row(city_name, seed, f"{model}_penalty", lam, pen,
                                          targets, prep, area, cats, {}))
            for theta in THETAS:
                lists = np.array([rank_mmr(rel[i], DISP_K, sim, theta) for i in range(len(targets))])
                rows.append(dispersal_row(city_name, seed, f"{model}_mmr", theta, lists,
                                          targets, prep, area, cats, {}))
            print(f"[{city_name}] seed={seed} {model}: R@20={plain[1]['recall']:.4f}", flush=True)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--cities", nargs="+", default=["florence", "rome"], choices=CITIES)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--baselines", nargs="*", default=None,
                    help="evaluate saved baseline scores instead of training our models")
    ap.add_argument("--scores", type=Path, default=Path("results/scores"))
    ap.add_argument("--v2-only", nargs="*", default=None,
                    help="run only these v2 config names (e.g. v2_none_gru)")
    ap.add_argument("--v4", action="store_true",
                    help="two-view InfoNCE (DHCN-style) with beta in {0, 0.01, 0.1}")
    ap.add_argument("--v3", action="store_true",
                    help="run step-1/step-2 variants (prefix session edge, time context)")
    ap.add_argument("--v2", action="store_true",
                    help="run leak-free v2 designs (theme/area channels, GRU encoder)")
    ap.add_argument("--tune-v2", action="store_true",
                    help="tune dispersal on the v2 retriever (theme+area+GRU) instead of v1")
    ap.add_argument("--tune", action="store_true",
                    help="sweep dispersal grids on val and test for one seed per process")
    args = ap.parse_args()

    if args.v4:
        tcfg = TrainConfig(max_epochs=args.max_epochs)
        for c in args.cities:
            rows = run_v2(c, args.raw, args.seeds, tcfg, only=args.v2_only, configs_all=V4_CONFIGS)
            name = f"{c}_only" if args.v2_only else c
            write_csv(rows, args.out / "v4" / f"{name}.csv")
            print(f"[{c}] wrote {len(rows)} v4 rows", flush=True)
        return

    if args.v3:
        tcfg = TrainConfig(max_epochs=args.max_epochs)
        for c in args.cities:
            rows = run_v2(c, args.raw, args.seeds, tcfg, only=args.v2_only, configs_all=V3_CONFIGS)
            name = f"{c}_only" if args.v2_only else c
            write_csv(rows, args.out / "v3" / f"{name}.csv")
            print(f"[{c}] wrote {len(rows)} v3 rows", flush=True)
        return

    if args.v2:
        tcfg = TrainConfig(max_epochs=args.max_epochs)
        for c in args.cities:
            rows = run_v2(c, args.raw, args.seeds, tcfg, only=args.v2_only)
            name = f"{c}_only" if args.v2_only else c
            write_csv(rows, args.out / "v2" / f"{name}.csv")
            print(f"[{c}] wrote {len(rows)} v2 rows", flush=True)
        return

    if args.tune or args.tune_v2:
        tcfg = TrainConfig(max_epochs=args.max_epochs)
        for c in args.cities:
            for seed in args.seeds:
                rows = run_tune(c, args.raw, seed, tcfg, v2=args.tune_v2)
                sub = "tune_v2" if args.tune_v2 else "tune"
                write_csv(rows, args.out / sub / f"{c}_seed{seed}.csv")
                print(f"[{c}] seed={seed} wrote {len(rows)} tune rows", flush=True)
        return

    if args.baselines is not None:
        for c in args.cities:
            rows = run_baselines(c, args.raw, args.seeds, args.baselines, args.scores)
            write_csv(rows, args.out / f"{c}_baselines.csv")
            print(f"[{c}] wrote {len(rows)} baseline rows", flush=True)
        return

    tcfg = TrainConfig(max_epochs=args.max_epochs)
    for c in args.cities:
        rows = run_city(c, args.raw, args.seeds, tcfg)
        write_csv(rows, args.out / f"{c}.csv")
        print(f"[{c}] wrote {len(rows)} rows to {args.out / f'{c}.csv'}", flush=True)


if __name__ == "__main__":
    main()
