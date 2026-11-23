"""Check 1 for the in-list swap: reachability and cost on the plain top-20 (plan D1). CPU.

Lists are the plain top-20 of the v2 pool (`results/pool`). Head = not long-tail.
Channels X (dispersal.channel_masks): theme, area, both, union (= share_matrix), category
(hypergraph.flat_category: broadest Wikipedia category, no raw type field in TripBuilder).

Per (city, seed, split):
  reach_X      over long-tail-target queries: some head u in the top-20 shares X with target.
  reach_rand_X same, target replaced by a random long-tail POI of the pool (100 draws/query).
  in_pool      long-tail target inside the pool top-100.
  sub_top1_X   some head u in the top-20 has the target as its highest-rel candidate;
  sub_top3_X   ... target among its 3 highest-rel candidates. Candidates of u: in the pool,
               outside the top-20, pop < pop(u), share X with u.
  cost_X_tau   over head-target queries with the target in the top-20: pop(target) >= tau
               threshold (top 5/10/20% POIs) and the target has at least one candidate.

Usage:
    python -m graph_retriever.reachability --raw <TripBuilderRawData> --out results/d1_reachability
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from graph_retriever.data import CITIES, load_city
from graph_retriever.dataset import prepare
from graph_retriever.dispersal import SWAP_CHANNELS, channel_masks, pop_threshold
from graph_retriever.hypergraph import build_hypergraph, flat_category

K = 20
SEEDS = (0, 1, 2)
SPLITS = ("val", "test")
TAUS = (0.05, 0.10, 0.20)
N_RAND = 100


def split_stats(prep, masks: dict, items: np.ndarray, targets: np.ndarray, rng) -> dict:
    tail, pop = prep.tail, prep.pop
    top = items[:, :K]
    lt_q = np.flatnonzero(tail[targets])
    out: dict = {"n_lt": int(lt_q.size)}
    out["in_pool"] = float(np.mean([targets[q] in set(items[q]) for q in lt_q]))
    thr = {f: pop_threshold(pop, f) for f in TAUS}
    hd_q = [q for q in np.flatnonzero(~tail[targets]) if targets[q] in set(top[q])]
    out["n_head_in_top"] = len(hd_q)
    for f in TAUS:
        out[f"eligible_tau{f:g}"] = float(np.mean([pop[targets[q]] >= thr[f] for q in hd_q]))
    draws = {}
    for q in lt_q:
        lt_pool = items[q][tail[items[q]]]
        draws[q] = rng.choice(lt_pool, N_RAND) if lt_pool.size else np.array([], dtype=np.int64)
    for x in SWAP_CHANNELS:
        m = masks[x]
        reach, rand, top1, top3 = [], [], [], []
        for q in lt_q:
            t = targets[q]
            heads = top[q][~tail[top[q]]]
            reach.append(bool(m[heads, t].any()))
            d = draws[q]
            rand.append(float(m[np.ix_(heads, d)].any(axis=0).mean()) if d.size and heads.size else 0.0)
            rest = items[q][K:]  # pool outside the top-20, rel order
            h1 = h3 = False
            for u in heads:
                cand = rest[m[u, rest] & (pop[rest] < pop[u])]
                h1 |= cand.size > 0 and cand[0] == t
                h3 |= bool((cand[:3] == t).any())
            top1.append(h1)
            top3.append(h3)
        out[f"reach_{x}"] = float(np.mean(reach))
        out[f"reach_rand_{x}"] = float(np.mean(rand))
        out[f"sub_top1_{x}"] = float(np.mean(top1))
        out[f"sub_top3_{x}"] = float(np.mean(top3))
        has_cand = []
        for q in hd_q:
            t, rest = targets[q], items[q][K:]
            has_cand.append(bool((m[t, rest] & (pop[rest] < pop[t])).any()))
        has_cand = np.array(has_cand, dtype=bool)
        for f in TAUS:
            elig = np.array([pop[targets[q]] >= thr[f] for q in hd_q], dtype=bool)
            out[f"cost_{x}_tau{f:g}"] = float((has_cand & elig).mean()) if hd_q else float("nan")
    return out


def _ms(v) -> str:
    v = np.asarray(v, dtype=np.float64)
    return f"{v.mean():.3f} ± {v.std():.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--pool", type=Path, default=Path("results/pool"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cities", nargs="+", default=list(CITIES), choices=CITIES)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    cat_info: dict = {}
    for city in args.cities:
        cityobj = load_city(args.raw, city)
        prep = prepare(cityobj)
        hg = build_hypergraph(cityobj, prep, ("theme", "area"))
        flat, names = flat_category(cityobj, prep.node_ids)
        sizes = np.bincount(flat[flat >= 0])
        cat_info[city] = (len(names), float(sizes.mean()), int(sizes.max()), int((flat < 0).sum()), prep.n_items)
        masks = channel_masks(hg, flat)
        for seed in args.seeds:
            for si, split in enumerate(SPLITS):
                z = np.load(args.pool / f"{city}_seed{seed}_{split}.npz")
                items, targets = z["items"].astype(np.int64), z["targets"].astype(np.int64)
                rng = np.random.default_rng([seed, si, 11])
                r = {"city": city, "seed": seed, "split": split}
                r.update(split_stats(prep, masks, items, targets, rng))
                rows.append(r)
                print(f"[{city}] seed={seed} {split}: n_lt={r['n_lt']}", flush=True)

    fields = list(rows[0])
    with (args.out / "reachability.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    md = ["# D1: reachability và cost của swap trong top-20 plain (pool top-100 của v2)", "",
          "Chỉ báo số. CPU. Mean ± std theo 3 seed.", "",
          "## Category phẳng (đối chứng)", "",
          ("Raw TripBuilder không có trường loại POI thô (pois-clusters.txt chỉ có `categories` = Wikipedia "
           "categories; pois.txt chỉ có id, tên, toạ độ). Cách chọn: category của POI = token Wikipedia "
           "(đã chuẩn hoá như Theme, bỏ tag quận) có nhiều POI nhất, kể cả token bị THEME_MAX_FRAC loại. "
           "Hoà thì theo tên. POI không có category: không khớp với ai."), "",
          "| city | N | #category | mean size | max size | POI không có category |", "|---|---|---|---|---|---|"]
    for c, (n_cat, mean_s, max_s, none, n) in cat_info.items():
        md.append(f"| {c} | {n} | {n_cat} | {mean_s:.2f} | {max_s} | {none} |")
    md.append("")
    lt_test = {c: sum(r["n_lt"] for r in rows if r["city"] == c and r["split"] == "test") for c in cat_info}
    md += ["## Số query", "",
           "Query long-tail trên test, gộp seed: " + ", ".join(f"{c} {v}" for c, v in lt_test.items())
           + " (A4: pisa 321, florence 3555, rome 4998).", ""]
    for split in SPLITS:
        md += [f"## {split}", ""]
        for city in cat_info:
            rs = [r for r in rows if r["city"] == city and r["split"] == split]
            md += [f"### {city} ({split})", "",
                   (f"Query long-tail/seed: {rs[0]['n_lt']}. Tỉ lệ target long-tail trong pool top-100: "
                    f"{_ms([r['in_pool'] for r in rs])}. Query có target head nằm trong top-20: "
                    f"{_ms([r['n_head_in_top'] for r in rs])} /seed."), "",
                   ("| kênh X | reach (target) | reach (random LT, 100/query) | sub top-1 | sub top-3 | "
                    "cost τ=5% | cost τ=10% | cost τ=20% |"),
                   "|---|---|---|---|---|---|---|---|"]
            for x in SWAP_CHANNELS:
                md.append(f"| {x} | " + " | ".join(_ms([r[k] for r in rs]) for k in (
                    f"reach_{x}", f"reach_rand_{x}", f"sub_top1_{x}", f"sub_top3_{x}",
                    f"cost_{x}_tau0.05", f"cost_{x}_tau0.1", f"cost_{x}_tau0.2")) + " |")
            md += ["", "Tỉ lệ target head (trong top-20) có pop ≥ ngưỡng τ: "
                   + ", ".join(f"τ={f:g}: {_ms([r[f'eligible_tau{f:g}'] for r in rs])}" for f in TAUS), ""]
    (args.out / "table.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {args.out / 'table.md'}", flush=True)


if __name__ == "__main__":
    main()
