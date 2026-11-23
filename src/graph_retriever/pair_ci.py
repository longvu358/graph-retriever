"""Plan D6: pairwise ΔnDCG@20 / ΔRecall@20 between two trade-off curves at equal exposure Gini,
with a query-bootstrap 95% CI.

Input: a folder written by pool_rerank --tradeoff (D2/D4) or baseline_tradeoff --with-val (D6):
{city}_seed{s}.csv with val and test rows, and lists/{city}_seed{s}.npz with the test top-K
lists of (at least) every val-selected setting.

Per pair (A, B) and city:
  1. per seed, settings on the val Pareto front of each curve (tradeoff.select_on_val);
  2. common Gini range of the PAIR = intersection over seeds and both curves of the test spans;
  3. N_LEVELS equal levels in that range; Δ(level) = seed mean of interp_A − interp_B;
  4. CI: B bootstrap resamples of the test queries per seed (fixed rng); Gini, nDCG and Recall
     are recomputed for every selected setting and the curves re-interpolated.
A pair whose common range is shorter than MIN_RANGE gets no Δ (reported as such). When A is a
single setting (one point), Δ is taken at A's own Gini, if B's span covers it.

Usage:
    python -m graph_retriever.pair_ci --dir results/d6_tradeoff --ref penalty_v2 \
        --against penalty_gru4rec penalty_dhcn --out results/d6_tradeoff/pair_ci.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from graph_retriever.data import CITIES
from graph_retriever.metrics import gini
from graph_retriever.tradeoff import (
    _load_lists,
    curve_key,
    interp,
    load_rows,
    select_on_val,
    test_points,
)

N_LEVELS = 5
MIN_RANGE = 0.01


def pair_range(pts: dict, a: str, b: str, seeds: list[int]) -> tuple[float, float]:
    spans = [[r["exp_gini"] for r in pts[(c, s)]] for c in (a, b) for s in seeds]
    return max(min(x) for x in spans), min(max(x) for x in spans)


def _curve(entries, cnt, w, n, q, grid):
    g = np.array([gini(np.bincount(lst.ravel(), weights=w, minlength=n)) for lst, _, _ in entries])
    nd = np.array([float(cnt @ v) / q for _, v, _ in entries])
    rc = np.array([float(cnt @ h) / q for _, _, h in entries])
    return interp(g, nd, grid), interp(g, rc, grid)


def bootstrap_pair(d: Path, city: str, seeds: list[int], sel: dict, a: str, b: str,
                   grid: np.ndarray, B: int, rng_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(B, len(grid)) seed-mean ΔnDCG and ΔRecall (A − B)."""
    rng = np.random.default_rng(rng_seed)
    data = {}
    for s in seeds:
        lists, targets, n = _load_lists(d, city, s)
        per = {}
        for c in (a, b):
            entries = []
            for p in sel[(c, s)]:
                lst = lists[f"{c}|{p}"].astype(np.int64)
                hit = lst == targets[:, None]
                found = hit.any(axis=1)
                nd = np.where(found, 1.0 / np.log2(np.argmax(hit, axis=1) + 2), 0.0)
                entries.append((lst, nd, found.astype(np.float64)))
            per[c] = entries
        disc = 1.0 / np.log2(np.arange(per[a][0][0].shape[1]) + 2)
        data[s] = (per, len(targets), n, disc)
    dn = np.zeros((B, len(grid)))
    dr = np.zeros((B, len(grid)))
    for i in range(B):
        for s in seeds:
            per, q, n, disc = data[s]
            cnt = np.bincount(rng.integers(q, size=q), minlength=q).astype(np.float64)
            w = (cnt[:, None] * disc[None, :]).ravel()
            na, ra = _curve(per[a], cnt, w, n, q, grid)
            nb, rb = _curve(per[b], cnt, w, n, q, grid)
            dn[i] += (na - nb) / len(seeds)
            dr[i] += (ra - rb) / len(seeds)
    return dn, dr


def point_delta(pts: dict, a: str, b: str, seeds: list[int], metric: str, grid: np.ndarray) -> np.ndarray:
    def at(c, s):
        rs = pts[(c, s)]
        return interp(np.array([r["exp_gini"] for r in rs]), np.array([r[metric] for r in rs]), grid)
    return np.mean([at(a, s) - at(b, s) for s in seeds], axis=0)


def _cell(v: float, boot: np.ndarray) -> str:
    lo, hi = np.percentile(boot, [2.5, 97.5])
    mark = " *" if lo > 0 or hi < 0 else ""
    return f"{v:+.4f} [{lo:+.4f}, {hi:+.4f}]{mark}"


def report(d: Path, city: str, pairs: list[tuple[str, str]], B: int, rng_seed: int) -> list[str]:
    rows = load_rows(d, city)
    if not rows:
        return []
    for r in rows:
        r["recall20"] = float(r["recall20"])
    seeds = sorted({r["seed"] for r in rows})
    sel = select_on_val(rows)
    for r in rows:  # curves with no val rows (e.g. an LLM run on test only): keep every setting
        key = (curve_key(r), r["seed"])
        if r["split"] == "test" and not any(x["split"] == "val" and curve_key(x) == key[0]
                                            and x["seed"] == key[1] for x in rows):
            sel.setdefault(key, [])
            if r["params"] not in sel[key]:
                sel[key].append(r["params"])
    pts = test_points(rows, sel)
    out = [f"## {city}", "", f"Seeds {seeds}. Δ = A − B, CI 95% query bootstrap (B = {B}); * = CI excludes 0.", "",
           "| A | B | Gini level | ΔnDCG@20 | ΔRecall@20 |", "|---|---|---|---|---|"]
    for a, b in pairs:
        if any((c, s) not in sel for c in (a, b) for s in seeds):
            out.append(f"| {a} | {b} | curve missing | | |")
            continue
        lo, hi = pair_range(pts, a, b, seeds)
        if all(len(sel[(a, s)]) == 1 for s in seeds):
            # A is a single setting (e.g. one LLM prompt): compare at A's own Gini, if B covers it
            g_a = float(np.mean([pts[(a, s)][0]["exp_gini"] for s in seeds]))
            b_lo, b_hi = pair_range(pts, b, b, seeds)
            if not b_lo <= g_a <= b_hi:
                out.append(f"| {a} | {b} | point at Gini {g_a:.3f} outside {b}'s span "
                           f"[{b_lo:.3f}, {b_hi:.3f}] | | |")
                continue
            grid = np.array([g_a])
        elif hi - lo < MIN_RANGE:
            out.append(f"| {a} | {b} | no common range ({lo:.3f} > {hi:.3f} − {MIN_RANGE}) | | |")
            continue
        else:
            grid = np.linspace(lo, hi, N_LEVELS)
        dn, dr = bootstrap_pair(d, city, seeds, sel, a, b, grid, B, rng_seed)
        pn = point_delta(pts, a, b, seeds, "ndcg20", grid)
        pr = point_delta(pts, a, b, seeds, "recall20", grid)
        for j, g in enumerate(grid):
            out.append(f"| {a} | {b} | {g:.3f} | {_cell(pn[j], dn[:, j])} | {_cell(pr[j], dr[:, j])} |")
    out.append("")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--ref", required=True, help="curve A of every pair")
    ap.add_argument("--against", nargs="+", required=True, help="curves B")
    ap.add_argument("--cities", nargs="+", default=list(CITIES), choices=CITIES)
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--rng", type=int, default=20261122)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    pairs = [(args.ref, b) for b in args.against]
    lines = [f"# Pairwise Δ at equal exposure Gini ({args.dir})", ""]
    for c in args.cities:
        lines += report(args.dir, c, pairs, args.B, args.rng)
        print(f"[{c}] done", flush=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
