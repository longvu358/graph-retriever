"""Pool-level rankers on the v2 top-100 pool (plans C4, C5). CPU only.

Every method reads the stored top-100 candidates of `results/pool` (items, rel) and ranks
inside them, so all methods see the same candidate set. Per (city, seed, split) the output
rows carry the test-independent metrics needed by graph_retriever.tune:
  recall, ndcg, tail_hit (Hit@20 over long-tail targets), recall_hp / tail_hit_hp (restricted
  to queries whose prefix contains a head POI), coverage, ILD, long_tail_ratio, ARP, area_entropy.

Methods (value of the `lam` column in brackets):
  plain                       top-20 by rel
  penalty                     rel - lam * pop                      [lam in TUNE_LAMBDAS]
  mmr                         MMR over the pool, Jaccard on cats   [theta in TUNE_THETAS]
  xquad_{area,cat}_{unif,prefix}  xQuAD over the pool              [lam in XQUAD_LAMBDAS]
  ours_tune_v2                rank_dispersal with the tune_v2 R1 setting for this city
  prefix_{binary,gap}         rank_prefix_subst                    [gamma in PREFIX_GAMMAS]

Usage:
    python -m graph_retriever.pool_rerank --raw <TripBuilderRawData> --pool results/pool \
        --out results/c4_xquad --methods plain penalty mmr xquad --cities pisa
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from graph_retriever.data import CITIES, load_city
from graph_retriever.dataset import Prepared, prepare
from graph_retriever.dispersal import (
    NEG_REL,
    DispersalConfig,
    _hyperedges,
    aspect_dist_prefix,
    aspect_dist_uniform,
    channel_masks,
    pop_threshold,
    rank_dispersal,
    rank_mmr,
    rank_prefix_subst,
    rank_xquad,
    share_matrix,
    swap_sequence,
)
from graph_retriever.experiment import TUNE_LAMBDAS, TUNE_THETAS
from graph_retriever.hypergraph import build_hypergraph, flat_category
from graph_retriever.metrics import (
    category_sets,
    cosine_distance_matrix,
    dispersal_metrics,
    epc,
    epd,
    exposure_metrics,
    history_distance,
    ild_batch,
    jaccard_similarity_matrix,
    self_information,
)
from graph_retriever.tradeoff import select_on_val

K = 20
SEEDS = (0, 1, 2)
SPLITS = ("val", "test")
XQUAD_LAMBDAS = (0.0, 0.1, 0.25, 0.5, 1.0)  # lam in [0, 1]; the tune grid's 2, 3, 5 do not apply
PREFIX_GAMMAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
# tune_v2 R1 choice per city (first row of each city in results/tune_v2_summary.md)
OURS_TUNE_V2 = {
    "pisa": (0.1, 0.1, 1.0),  # (beta, delta, lam)
    "florence": (0.0, 0.25, 1.0),
    "rome": (0.1, 0.1, 3.0),
}
ALL_METHODS = ("plain", "penalty", "mmr", "xquad", "ours", "prefix")


def _metrics(lists: np.ndarray, targets: np.ndarray, prep: Prepared, area, cats,
             has_head: np.ndarray) -> dict:
    hit = (lists == targets[:, None]).any(axis=1)
    tail_t = prep.tail[targets]
    hp = has_head
    out = {
        "recall": float(hit.mean()),
        "ndcg": float(np.mean(np.where(hit, 1.0 / np.log2(np.argmax(lists == targets[:, None], axis=1) + 2), 0.0))),
        "tail_hit": float(hit[tail_t].mean()) if tail_t.any() else float("nan"),
        "recall_hp": float(hit[hp].mean()) if hp.any() else float("nan"),
        "tail_hit_hp": float(hit[tail_t & hp].mean()) if (tail_t & hp).any() else float("nan"),
        "n_q": len(targets),
        "n_tail_q": int(tail_t.sum()),
        "n_hp_q": int(hp.sum()),
    }
    dm = dispersal_metrics(lists.tolist(), prep, area, cats)
    out.update({k: dm[k] for k in ("ILD", "coverage", "long_tail_ratio", "ARP", "area_entropy")})
    return out


def _aspects_area(area: np.ndarray) -> list[frozenset]:
    return [frozenset({f"area{a}"}) if a >= 0 else frozenset() for a in area]


def _aspects_cat(cats: list[frozenset]) -> list[frozenset]:
    return [frozenset(c) for c in cats]


def run_city_split(city: str, prep: Prepared, pool: Path, seed: int, split: str,
                   methods: tuple[str, ...], ctx: dict) -> list[dict]:
    samples = prep.val if split == "val" else prep.test
    data = np.load(pool / f"{city}_seed{seed}_{split}.npz")
    items = data["items"].astype(np.int64)
    rel_top = data["rel"].astype(np.float64)
    targets = data["targets"].astype(np.int64)
    expected = np.array([s.target for s in samples], dtype=np.int64)
    if not np.array_equal(targets, expected):
        raise RuntimeError(f"{city} seed {seed} {split}: pool targets differ from prepare()")
    Q, _width = items.shape
    N = prep.n_items
    prefixes = [s.prefix for s in samples]
    has_head = np.array([any(not prep.tail[v] for v in p) for p in prefixes], dtype=bool)
    area, share, sim, cats = ctx["area"], ctx["share"], ctx["sim"], ctx["cats"]
    is_head = ~prep.tail
    out: list[dict] = []

    def emit(method: str, lam: float, lists: np.ndarray) -> None:
        row = {"city": city, "seed": seed, "split": split, "method": method, "lam": lam}
        row.update(_metrics(lists, targets, prep, area, cats, has_head))
        out.append(row)

    # full-length rel vectors per query, NEG outside the pool
    def full_rel(i: int) -> np.ndarray:
        r = np.full(N, NEG_REL)
        r[items[i]] = rel_top[i]
        return r

    if "plain" in methods:
        emit("plain", 0.0, items[:, :K])

    if "penalty" in methods:
        pop_it = prep.pop[items]
        for lam in TUNE_LAMBDAS:
            order = np.argsort(-(rel_top - lam * pop_it), axis=1, kind="stable")[:, :K]
            emit("penalty", lam, np.take_along_axis(items, order, axis=1))

    if "mmr" in methods:
        for theta in TUNE_THETAS:
            lists = np.empty((Q, K), dtype=np.int64)
            for i in range(Q):
                idx = items[i]
                loc = rank_mmr(rel_top[i], K, sim[np.ix_(idx, idx)], theta)
                lists[i] = idx[loc]
            emit("mmr", theta, lists)

    if "xquad" in methods:
        asp = {"area": _aspects_area(area), "cat": _aspects_cat(cats)}
        for akind, aspects in asp.items():
            for dist in ("unif", "prefix"):
                for lam in XQUAD_LAMBDAS:
                    lists = np.empty((Q, K), dtype=np.int64)
                    for i in range(Q):
                        r = full_rel(i)
                        pool_bool = np.zeros(N, dtype=bool)
                        pool_bool[items[i]] = True
                        p = (aspect_dist_uniform(aspects, pool_bool) if dist == "unif"
                             else aspect_dist_prefix(aspects, pool_bool, prefixes[i]))
                        lists[i] = rank_xquad(r, K, aspects, p, lam)
                    emit(f"xquad_{akind}_{dist}", lam, lists)

    if "ours" in methods:
        beta, delta, lam = OURS_TUNE_V2[city]
        cfg = DispersalConfig(lam=lam, beta=beta, delta=delta, substitution=True, coverage=True)
        hg = ctx["hg"]
        lists = np.array([[p.item for p in rank_dispersal(full_rel(i), K, hg, prep.pop, share, cfg).picks]
                          for i in range(Q)])
        emit("ours_tune_v2", lam, lists)

    if "prefix" in methods:
        for mode in ("binary", "gap"):
            for gamma in PREFIX_GAMMAS:
                lists = np.array([rank_prefix_subst(full_rel(i), K, prefixes[i], prep.pop, is_head,
                                                    share, gamma, mode) for i in range(Q)])
                emit(f"prefix_{mode}", gamma, lists)
    return out


# --- Trade-off mode (plans D0, D2): exposure metrics, clean swap, val Pareto selection ---

D2_LAMBDAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0)
D2_XQUAD = ("cat_prefix", "area_prefix")
SWAP_TAUS = (0.05, 0.10, 0.20)  # top fraction of POIs by pop
SWAP_DELTAS = (0.25, 0.5, 1.0, 2.0, 4.0, np.inf)
SWAP_BUDGETS = (0, 1, 2, 5, 10, 20)  # 0 = plain, kept for the D2 check
SWAPSESS_DELTAS = (1.0, 2.0, np.inf)
SWAPSESS_WS = (0.25, 0.5, 1.0, 2.0)
SWAP_VARIANTS = ("theme", "area", "both", "union", "category", "random")
SWAPSESS_VARIANTS = ("theme", "union")
TRADEOFF_METHODS = ("plain", "penalty", "mmr", "xquad", "ours", "swap", "swapsess")
TRADEOFF_FIELDS = ("city", "method", "variant", "params", "seed", "split", "ndcg20", "recall20",
                   "lt_hit20", "coverage", "exp_gini", "exp_entropy", "exp_top1pct",
                   "exp_head_share", "arp", "n_swaps")
# plan D4: novelty columns, written only when tradeoff_metrics gets `nov`
NOVELTY_FIELDS = ("epc20", "epc20_all", "epd20", "epd20_all", "epd20_cat_all", "ild20",
                  "self_info20")


def novelty_context(prep: Prepared, prefixes: list[list[int]], cos_d: np.ndarray,
                    cat_sim: np.ndarray) -> dict:
    """Per (city, split) inputs of novelty_metrics. cos_d: item x train-session cosine distance."""
    return {"hist_cos": history_distance(cos_d, prefixes),
            "hist_cat": history_distance(1.0 - cat_sim, prefixes),
            "cat_sim": cat_sim}


def novelty_metrics(lists: np.ndarray, targets: np.ndarray, prep: Prepared, nov: dict) -> dict:
    """EPC / EPD (Lacic et al. 2020): `*_all` = relevance-unaware (p(rel) = 1)."""
    return {
        "epc20": epc(lists, prep.pop_count, targets),
        "epc20_all": epc(lists, prep.pop_count),
        "epd20": epd(lists, nov["hist_cos"], targets),
        "epd20_all": epd(lists, nov["hist_cos"]),
        "epd20_cat_all": epd(lists, nov["hist_cat"]),
        "ild20": ild_batch(lists, nov["cat_sim"]),
        "self_info20": self_information(lists, prep.pop_count),
    }


def tradeoff_metrics(lists: np.ndarray, targets: np.ndarray, prep: Prepared,
                     nov: dict | None = None) -> dict:
    hit = lists == targets[:, None]
    found = hit.any(axis=1)
    tail_t = prep.tail[targets]
    out = {
        "ndcg20": float(np.where(found, 1.0 / np.log2(np.argmax(hit, axis=1) + 2), 0.0).mean()),
        "recall20": float(found.mean()),
        "lt_hit20": float(found[tail_t].mean()) if tail_t.any() else float("nan"),
        "coverage": float(np.unique(lists).size / prep.n_items),
        "arp": float(prep.pop[lists].mean()),
    }
    out.update(exposure_metrics(lists, prep.n_items, ~prep.tail))
    if nov is not None:
        out.update(novelty_metrics(lists, targets, prep, nov))
    return out


def _fmt(v: float) -> str:
    return "inf" if np.isinf(v) else f"{v:g}"


def run_tradeoff_split(city: str, prep: Prepared, pool: Path, seed: int, split: str,
                       methods: tuple[str, ...], ctx: dict, emit) -> None:
    """Call emit(method, variant, params, lists, n_swaps) for every setting of `methods`."""
    samples = prep.val if split == "val" else prep.test
    data = np.load(pool / f"{city}_seed{seed}_{split}.npz")
    items = data["items"].astype(np.int64)
    rel_top = data["rel"].astype(np.float64)
    targets = data["targets"].astype(np.int64)
    expected = np.array([s.target for s in samples], dtype=np.int64)
    if not np.array_equal(targets, expected):
        raise RuntimeError(f"{city} seed {seed} {split}: pool targets differ from prepare()")
    Q, _width = items.shape
    N = prep.n_items
    prefixes = [s.prefix for s in samples]
    pop = prep.pop
    plain = items[:, :K]

    def full_rel(i: int) -> np.ndarray:
        r = np.full(N, NEG_REL)
        r[items[i]] = rel_top[i]
        return r

    if "plain" in methods:
        emit("plain", "", "", plain, 0.0)

    if "penalty" in methods:
        pop_it = pop[items]
        for lam in D2_LAMBDAS:
            order = np.argsort(-(rel_top - lam * pop_it), axis=1, kind="stable")[:, :K]
            emit("penalty", "", f"lam={_fmt(lam)}", np.take_along_axis(items, order, axis=1), 0.0)

    if "mmr" in methods:
        sim = ctx["sim"]
        for theta in TUNE_THETAS:
            lists = np.empty((Q, K), dtype=np.int64)
            for i in range(Q):
                idx = items[i]
                lists[i] = idx[rank_mmr(rel_top[i], K, sim[np.ix_(idx, idx)], theta)]
            emit("mmr", "", f"theta={_fmt(theta)}", lists, 0.0)

    if "xquad" in methods:
        asp = {"area": _aspects_area(ctx["area"]), "cat": _aspects_cat(ctx["cats"])}
        for variant in D2_XQUAD:
            aspects = asp[variant.split("_")[0]]
            for lam in XQUAD_LAMBDAS:
                lists = np.empty((Q, K), dtype=np.int64)
                for i in range(Q):
                    pool_bool = np.zeros(N, dtype=bool)
                    pool_bool[items[i]] = True
                    p = aspect_dist_prefix(aspects, pool_bool, prefixes[i])
                    lists[i] = rank_xquad(full_rel(i), K, aspects, p, lam)
                emit("xquad", variant, f"lam={_fmt(lam)}", lists, 0.0)

    if "ours" in methods:
        beta, delta, lam = OURS_TUNE_V2[city]
        cfg = DispersalConfig(lam=lam, beta=beta, delta=delta, substitution=True, coverage=True)
        lists = np.array([[p.item for p in rank_dispersal(full_rel(i), K, ctx["hg"], pop, ctx["share"], cfg).picks]
                          for i in range(Q)])
        emit("ours_tune_v2", "", f"beta={_fmt(beta)};delta={_fmt(delta)};lam={_fmt(lam)}", lists, 0.0)

    # Swap: one run per (variant, delta[, w]) with the loosest tau and full budget. A lower
    # tau or budget is a prefix of that run (swap_sequence), so every (tau, budget) is a
    # truncation. Verified in tests/test_swap.py.
    thr = {f: pop_threshold(pop, f) for f in SWAP_TAUS}
    loose = min(thr.values())
    masks = ctx["masks"]

    def swap_runs(method: str, variant: str, deltas, ws) -> None:
        select = "random" if variant == "random" else ("rel+session" if method == "swapsess" else "rel")
        for d_i, delta in enumerate(deltas):
            for w in ws:
                rng = np.random.default_rng([seed, SPLITS.index(split), d_i, 7])
                seqs = []
                for i in range(Q):
                    idx = items[i]
                    if variant == "random":
                        sub = np.broadcast_to(prep.tail[idx][None, :], (idx.size, idx.size))
                    else:
                        sub = masks[variant][np.ix_(idx, idx)]
                    bonus = w * ctx["sess_overlap"][split][i] if method == "swapsess" else None
                    seqs.append(swap_sequence(rel_top[i], pop[idx], sub, K, loose, delta, K,
                                              select, bonus, rng))
                for f in SWAP_TAUS:
                    for b in SWAP_BUDGETS:
                        lists = plain.copy()
                        n_sw = np.zeros(Q)
                        for i, seq in enumerate(seqs):
                            use = [s for s in seq if s[2] >= thr[f]][:b]
                            for pos, v, _ in use:
                                lists[i, pos] = items[i, v]
                            n_sw[i] = len(use)
                        params = f"tau={_fmt(f)};delta={_fmt(delta)};budget={b}"
                        if method == "swapsess":
                            params = f"tau={_fmt(f)};delta={_fmt(delta)};w={_fmt(w)};budget={b}"
                        emit(method, variant, params, lists, float(n_sw.mean()))

    if "swap" in methods:
        for variant in SWAP_VARIANTS:
            swap_runs("swap", variant, SWAP_DELTAS, (0.0,))
    if "swapsess" in methods:
        if split not in ctx["sess_overlap"]:
            ie = ctx["item_edges"]
            ov = []
            for i in range(Q):
                pe = set().union(*(ie[v] for v in prefixes[i]))
                ov.append(np.array([len(ie[j] & pe) for j in items[i]], dtype=np.float64))
            ctx["sess_overlap"][split] = ov
        for variant in SWAPSESS_VARIANTS:
            swap_runs("swapsess", variant, SWAPSESS_DELTAS, SWAPSESS_WS)


def collect_tradeoff(city: str, prep: Prepared, pool: Path, seed: int, split: str,
                     methods: tuple[str, ...], ctx: dict,
                     keep: set | None = None) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Rows for one (city, seed, split); lists of the settings in `keep` are returned too."""
    targets = np.array([s.target for s in (prep.val if split == "val" else prep.test)])
    rows: list[dict] = []
    saved: dict[str, np.ndarray] = {}

    def emit(method, variant, params, lists, n_sw):
        row = {"city": city, "method": method, "variant": variant, "params": params,
               "seed": seed, "split": split, "n_swaps": n_sw}
        row.update(tradeoff_metrics(lists, targets, prep, ctx.get("nov", {}).get(split)))
        rows.append(row)
        key = method if not variant else f"{method}_{variant}"
        if keep is not None and (key, params) in keep:
            saved[f"{key}|{params}"] = lists.astype(np.int16)

    run_tradeoff_split(city, prep, pool, seed, split, methods, ctx, emit)
    return rows, saved


def run_tradeoff(args) -> None:
    """Per (city, seed): val rows, val Pareto selection per curve, test rows, and the test
    lists of the selected settings (for the bootstrap in graph_retriever.tradeoff)."""
    (args.out / "lists").mkdir(parents=True, exist_ok=True)
    methods = tuple(args.methods)
    for city in args.cities:
        cityobj = load_city(args.raw, city)
        prep = prepare(cityobj)
        hg = build_hypergraph(cityobj, prep, ("theme", "area"))
        cats = category_sets(cityobj, prep.node_ids)
        flat, _names = flat_category(cityobj, prep.node_ids)
        ctx = {"hg": hg, "area": hg.area, "share": share_matrix(hg),
               "sim": jaccard_similarity_matrix(cats), "cats": cats,
               "masks": channel_masks(hg, flat),
               "item_edges": [frozenset(_hyperedges(hg, i)) for i in range(prep.n_items)]}
        if args.novelty:
            cos_d = cosine_distance_matrix(prep.train_sessions, prep.n_items)
            ctx["nov"] = {sp: novelty_context(prep, [s.prefix for s in (prep.val if sp == "val" else prep.test)],
                                              cos_d, ctx["sim"]) for sp in SPLITS}
        fields = TRADEOFF_FIELDS + (NOVELTY_FIELDS if args.novelty else ())
        for seed in args.seeds:
            ctx["sess_overlap"] = {}
            rows, _ = collect_tradeoff(city, prep, args.pool, seed, "val", methods, ctx)
            print(f"[{city}] seed={seed} val: {len(rows)} rows", flush=True)
            sel = select_on_val(rows)
            keep = {(c, p) for (c, _s), ps in sel.items() for p in ps}
            test_rows, saved = collect_tradeoff(city, prep, args.pool, seed, "test", methods, ctx, keep)
            rows += test_rows
            print(f"[{city}] seed={seed} test: {len(test_rows)} rows, {len(saved)} lists saved", flush=True)
            path = args.out / f"{city}_seed{seed}.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k, "") for k in fields})
            keys = list(saved)
            np.savez_compressed(args.out / "lists" / f"{city}_seed{seed}.npz",
                                keys=np.array(json.dumps(keys)),
                                targets=np.array([s.target for s in prep.test]),
                                n_items=np.array(prep.n_items),
                                **{f"c{j}": saved[k] for j, k in enumerate(keys)})
            print(f"wrote {path}", flush=True)


FIELDS = ("city", "seed", "split", "method", "lam", "recall", "ndcg", "tail_hit", "recall_hp",
          "tail_hit_hp", "ILD", "coverage", "long_tail_ratio", "ARP", "area_entropy",
          "n_q", "n_tail_q", "n_hp_q")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--pool", type=Path, default=Path("results/pool"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cities", nargs="+", default=list(CITIES), choices=CITIES)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--methods", nargs="+", default=None,
                    choices=sorted(set(ALL_METHODS) | set(TRADEOFF_METHODS)))
    ap.add_argument("--tradeoff", action="store_true",
                    help="plans D0/D2: exposure metrics, swap methods, val Pareto lists")
    ap.add_argument("--novelty", action="store_true",
                    help="plan D4 (with --tradeoff): add EPC/EPD/ILD/self-information columns")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.tradeoff:
        args.methods = args.methods or list(TRADEOFF_METHODS)
        if not set(args.methods) <= set(TRADEOFF_METHODS):
            ap.error(f"--tradeoff methods: {TRADEOFF_METHODS}")
        run_tradeoff(args)
        return
    args.methods = args.methods or list(ALL_METHODS)
    if not set(args.methods) <= set(ALL_METHODS):
        ap.error(f"methods: {ALL_METHODS}")
    for city in args.cities:
        prep = prepare(load_city(args.raw, city))
        hg = build_hypergraph(load_city(args.raw, city), prep, ("theme", "area"))
        cats = category_sets(load_city(args.raw, city), prep.node_ids)
        ctx = {"hg": hg, "area": hg.area, "share": share_matrix(hg),
               "sim": jaccard_similarity_matrix(cats), "cats": cats}
        for seed in args.seeds:
            rows: list[dict] = []
            for split in SPLITS:
                rows += run_city_split(city, prep, args.pool, seed, split, tuple(args.methods), ctx)
                print(f"[{city}] seed={seed} {split}: {len(rows)} rows so far", flush=True)
            path = args.out / f"{city}_seed{seed}.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k, "") for k in FIELDS})
            print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
