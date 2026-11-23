"""nDCG@20 vs exposure-Gini trade-off curves (plan D0 step 3, used by D2).

Input: CSVs (one per city and seed) with columns
  city, method, variant, params, seed, split, ndcg20, recall20, lt_hit20, coverage,
  exp_gini, exp_entropy, exp_top1pct, exp_head_share, arp, n_swaps
and, for the bootstrap, `lists/{city}_seed{seed}.npz` with the test top-K lists of the
val-selected settings (written by pool_rerank --tradeoff).

Curve = (method, variant). Per seed:
  1. val: keep the settings on the Pareto front (lower exp_gini, higher ndcg20).
  2. test: the points of those settings, sorted by test Gini, linearly interpolated.
Common Gini range = intersection over curves and seeds of the test Gini spans. Curves with a
single setting (plain, ours_tune_v2) are points and do not enter the common range.
Grid = 10 equal levels over the common range. AUC = integral of nDCG over the range / length.
ΔnDCG CI: B query bootstraps per seed (fixed rng), Gini and nDCG recomputed for every
selected setting, curves re-interpolated on the same grid, Δ averaged over seeds.

Usage:
    python -m graph_retriever.tradeoff --dir results/d2_swap --out results/d2_swap/table.md
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from graph_retriever.data import CITIES
from graph_retriever.metrics import gini

NUMERIC = ("ndcg20", "recall20", "lt_hit20", "coverage", "exp_gini", "exp_entropy",
           "exp_top1pct", "exp_head_share", "arp", "n_swaps")
POINT_METHODS = ("plain", "ours_tune_v2")
N_LEVELS = 10
MIN_RANGE = 0.05
DENSE = 1001


def curve_key(row: dict) -> str:
    return row["method"] if not row["variant"] else f"{row['method']}_{row['variant']}"


def pareto_front(gini_v: np.ndarray, ndcg_v: np.ndarray) -> list[int]:
    """Indices of non-dominated points (min Gini, max nDCG). Exact duplicates keep the first."""
    order = np.lexsort((np.arange(len(gini_v)), -np.asarray(ndcg_v), np.asarray(gini_v)))
    keep, best = [], -np.inf
    for i in order:
        if ndcg_v[i] > best:
            keep.append(int(i))
            best = ndcg_v[i]
    return sorted(keep)


def interp(x: np.ndarray, y: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Piecewise-linear y(x) through the points sorted by x; equal x keep the max y.
    Outside the span the end values are held (np.interp)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.lexsort((-y, x))
    x, y = x[order], y[order]
    first = np.r_[True, np.diff(x) > 0]
    return np.interp(at, x[first], y[first])


def auc(x: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    dense = np.linspace(lo, hi, DENSE)
    return float(np.trapezoid(interp(x, y, dense), dense) / (hi - lo))


def load_rows(d: Path, city: str) -> list[dict]:
    rows = []
    for path in sorted(d.glob(f"{city}_seed*.csv")):
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                for k in NUMERIC:
                    r[k] = float(r[k])
                r["seed"] = int(r["seed"])
                r["variant"] = r.get("variant") or ""
                rows.append(r)
    return rows


def select_on_val(rows: list[dict]) -> dict[tuple[str, int], list[str]]:
    """(curve, seed) -> params on the val Pareto front."""
    out: dict[tuple[str, int], list[str]] = {}
    groups: dict[tuple[str, int], list[dict]] = {}
    for r in rows:
        if r["split"] == "val":
            groups.setdefault((curve_key(r), r["seed"]), []).append(r)
    for key, rs in groups.items():
        idx = pareto_front(np.array([r["exp_gini"] for r in rs]), np.array([r["ndcg20"] for r in rs]))
        out[key] = [rs[i]["params"] for i in idx]
    return out


def test_points(rows: list[dict], sel: dict) -> dict[tuple[str, int], list[dict]]:
    test = {(curve_key(r), r["seed"], r["params"]): r for r in rows if r["split"] == "test"}
    return {key: [test[(key[0], key[1], p)] for p in params] for key, params in sel.items()}


def common_range(pts: dict, curves: list[str], seeds: list[int]) -> tuple[float, float, dict]:
    spans = {}
    for c in curves:
        lo = max(min(r["exp_gini"] for r in pts[(c, s)]) for s in seeds)
        hi = min(max(r["exp_gini"] for r in pts[(c, s)]) for s in seeds)
        spans[c] = (lo, hi)
    return max(v[0] for v in spans.values()), min(v[1] for v in spans.values()), spans


def zero_checks(rows: list[dict]) -> list[str]:
    """Settings that must equal plain: every budget = 0 swap and penalty lam = 0 (all metrics,
    both splits, every seed). Also penalty lam = 1 test coverage (C4 reference)."""
    plain = {(r["seed"], r["split"]): r for r in rows if r["method"] == "plain"}
    worst: dict[str, float] = {}
    for r in rows:
        zero = (r["method"] in ("swap", "swapsess") and r["params"].endswith("budget=0")) or \
               (r["method"] == "penalty" and r["params"] == "lam=0")
        if not zero or (r["seed"], r["split"]) not in plain:
            continue
        p = plain[(r["seed"], r["split"])]
        d = max(abs(r[k] - p[k]) for k in NUMERIC)
        worst[curve_key(r)] = max(worst.get(curve_key(r), 0.0), d)
    out = [f"- {c}: max |Δ| vs plain over all metrics, splits, seeds = {d:.1e}" for c, d in sorted(worst.items())]
    cov = [r["coverage"] for r in rows if r["method"] == "penalty" and r["params"] == "lam=1" and r["split"] == "test"]
    if cov:
        out.append(f"- penalty λ = 1, test coverage: {np.mean(cov):.3f} ± {np.std(cov):.3f}")
    return out


def curve_at(pts: list[dict], metric: str, at: np.ndarray) -> np.ndarray:
    return interp(np.array([r["exp_gini"] for r in pts]), np.array([r[metric] for r in pts]), at)


# --- bootstrap over test queries ---


def _load_lists(d: Path, city: str, seed: int) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    z = np.load(d / "lists" / f"{city}_seed{seed}.npz")
    keys = json.loads(str(z["keys"]))
    return {k: z[f"c{j}"] for j, k in enumerate(keys)}, z["targets"], int(z["n_items"])


def bootstrap_delta(d: Path, city: str, seeds: list[int], sel: dict, curves: list[str],
                    refs: list[str], grid: np.ndarray, B: int, rng_seed: int) -> dict:
    """Return {(curve, ref): (B, len(grid)) array of seed-mean ΔnDCG}."""
    rng = np.random.default_rng(rng_seed)
    per_seed = {}
    for s in seeds:
        lists, targets, n = _load_lists(d, city, s)
        disc = 1.0 / np.log2(np.arange(next(iter(lists.values())).shape[1]) + 2)
        cfg = {}
        for c in curves:
            entries = []
            for p in sel[(c, s)]:
                lst = lists[f"{c}|{p}"]
                hit = lst == targets[:, None]
                nd = np.where(hit.any(axis=1), 1.0 / np.log2(np.argmax(hit, axis=1) + 2), 0.0)
                entries.append((lst, nd))
            cfg[c] = entries
        per_seed[s] = (cfg, len(targets), n, disc)
    vals = {c: np.zeros((B, len(grid))) for c in curves}
    for b in range(B):
        for s in seeds:
            cfg, q, n, disc = per_seed[s]
            cnt = np.bincount(rng.integers(q, size=q), minlength=q).astype(np.float64)
            w = (cnt[:, None] * disc[None, :]).ravel()
            for c in curves:
                g = [gini(np.bincount(lst.ravel(), weights=w, minlength=n)) for lst, _ in cfg[c]]
                y = [float(cnt @ nd) / q for _, nd in cfg[c]]
                vals[c][b] += interp(np.array(g), np.array(y), grid) / len(seeds)
    return {(c, r): vals[c] - vals[r] for c in curves for r in refs if r in vals and c != r}


# --- SVG (hand-written, no plotting package) ---

PALETTE = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
           "#7f7f7f", "#bcbd22", "#17becf", "#393b79", "#ad494a", "#637939", "#843c39")


def svg_plot(city: str, curves: dict[str, tuple[np.ndarray, np.ndarray]],
             points: dict[str, tuple[float, float]], lo: float, hi: float) -> str:
    W, H, ml, mr, mt, mb = 760, 460, 60, 210, 30, 50
    xs = np.concatenate([v[0] for v in curves.values()] + [np.array([p[0] for p in points.values()])])
    ys = np.concatenate([v[1] for v in curves.values()] + [np.array([p[1] for p in points.values()])])
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    pad_x, pad_y = (x1 - x0) * 0.03 or 0.01, (y1 - y0) * 0.05 or 0.01
    x0, x1, y0, y1 = x0 - pad_x, x1 + pad_x, y0 - pad_y, y1 + pad_y

    def X(v): return ml + (v - x0) / (x1 - x0) * (W - ml - mr)
    def Y(v): return H - mb - (v - y0) / (y1 - y0) * (H - mt - mb)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="11">',
           f'<rect width="{W}" height="{H}" fill="white"/>',
           f'<rect x="{X(lo):.1f}" y="{mt}" width="{X(hi) - X(lo):.1f}" height="{H - mt - mb}" fill="#eee"/>',
           f'<text x="{W / 2 - mr / 2}" y="18" text-anchor="middle" font-size="13">{city}: test nDCG@20 vs exposure Gini (mean of 3 seeds; grey = common range)</text>',
           f'<line x1="{ml}" y1="{H - mb}" x2="{W - mr}" y2="{H - mb}" stroke="black"/>',
           f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{H - mb}" stroke="black"/>',
           f'<text x="{(ml + W - mr) / 2}" y="{H - 12}" text-anchor="middle">exp_gini</text>',
           f'<text x="14" y="{(mt + H - mb) / 2}" transform="rotate(-90 14 {(mt + H - mb) / 2})" text-anchor="middle">nDCG@20</text>']
    for t in np.linspace(x0, x1, 6):
        out.append(f'<text x="{X(t):.1f}" y="{H - mb + 15}" text-anchor="middle">{t:.3f}</text>')
    for t in np.linspace(y0, y1, 6):
        out.append(f'<text x="{ml - 5}" y="{Y(t) + 4:.1f}" text-anchor="end">{t:.3f}</text>')
    for j, (name, (gx, gy)) in enumerate(curves.items()):
        col = PALETTE[j % len(PALETTE)]
        dash = ' stroke-dasharray="5,3"' if name.startswith("swapsess") or name.endswith("random") else ""
        path = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(gx, gy))
        out.append(f'<polyline points="{path}" fill="none" stroke="{col}" stroke-width="1.6"{dash}/>')
        ly = mt + 14 * j
        out.append(f'<line x1="{W - mr + 10}" y1="{ly}" x2="{W - mr + 30}" y2="{ly}" stroke="{col}" stroke-width="2"{dash}/>')
        out.append(f'<text x="{W - mr + 35}" y="{ly + 4}">{name}</text>')
    for j, (name, (px, py)) in enumerate(points.items()):
        ly = mt + 14 * (len(curves) + j)
        out.append(f'<circle cx="{X(px):.1f}" cy="{Y(py):.1f}" r="4" fill="black"/>' if j == 0 else
                   f'<rect x="{X(px) - 4:.1f}" y="{Y(py) - 4:.1f}" width="8" height="8" fill="black"/>')
        out.append(f'<text x="{W - mr + 35}" y="{ly + 4}">{name} ({"●" if j == 0 else "■"})</text>')
    out.append("</svg>")
    return "\n".join(out)


def _ms(a) -> str:
    a = np.asarray(a, dtype=np.float64)
    return f"{a.mean():.4f} ± {a.std():.4f}"


def report_city(d: Path, city: str, B: int, rng_seed: int, svg_dir: Path) -> list[str]:
    rows = load_rows(d, city)
    if not rows:
        return []
    seeds = sorted({r["seed"] for r in rows})
    sel = select_on_val(rows)
    pts = test_points(rows, sel)
    names = sorted({k[0] for k in sel})
    curves = [c for c in names if c not in POINT_METHODS]
    lo, hi, spans = common_range(pts, curves, seeds)
    out = [f"## {city}", "", f"Seeds: {seeds}. Curves: {len(curves)}. Settings on the val Pareto front, per seed: "
           + ", ".join(f"{c} {np.mean([len(sel[(c, s)]) for s in seeds]):.1f}" for c in curves), ""]
    out += ["Kiểm tra (budget = 0, λ = 0 trùng plain):", ""] + zero_checks(rows) + [""]
    out += ["Test Gini span per curve (intersection over seeds):", "",
            "| curve | Gini lo | Gini hi |", "|---|---|---|"]
    out += [f"| {c} | {spans[c][0]:.4f} | {spans[c][1]:.4f} |" for c in curves]
    out += ["", f"Common Gini range: [{lo:.4f}, {hi:.4f}], length {hi - lo:.4f}.", ""]
    if hi - lo < MIN_RANGE:
        out += [f"**Common range shorter than {MIN_RANGE}: stop at step 5 (no table, no bootstrap).**", ""]
        return out

    grid = np.linspace(lo, hi, N_LEVELS)
    mid = np.array([(lo + hi) / 2])
    levels = np.r_[grid, mid]
    out += ["### nDCG@20 (test) at the Gini grid, mean ± std over seeds, and AUC", "",
            "| curve | " + " | ".join(f"G={g:.4f}" for g in grid) + " | AUC |",
            "|---|" + "---|" * (len(grid) + 1)]
    aucs = {}
    for c in curves:
        per = np.array([curve_at(pts[(c, s)], "ndcg20", grid) for s in seeds])
        aucs[c] = [auc([r["exp_gini"] for r in pts[(c, s)]], [r["ndcg20"] for r in pts[(c, s)]], lo, hi)
                   for s in seeds]
        out.append(f"| {c} | " + " | ".join(_ms(per[:, j]) for j in range(len(grid))) + f" | {_ms(aucs[c])} |")
    out.append("")
    out += ["Points (single setting, test, mean ± std):", ""]
    for c in [n for n in names if n in POINT_METHODS]:
        out.append(f"- {c}: Gini {_ms([pts[(c, s)][0]['exp_gini'] for s in seeds])}, "
                   f"nDCG@20 {_ms([pts[(c, s)][0]['ndcg20'] for s in seeds])}")
    out.append("")

    refs = [r for r in ("penalty", "swap_category") if r in curves]
    targets = [c for c in curves if c.startswith("swap")]
    boot = bootstrap_delta(d, city, seeds, sel, curves, refs, levels, B, rng_seed)
    point = {c: np.mean([curve_at(pts[(c, s)], "ndcg20", levels) for s in seeds], axis=0) for c in curves}
    out += [f"### ΔnDCG@20 (test) vs reference, point = seed mean, CI 95% = query bootstrap (B = {B}, rng {rng_seed})", ""]
    gate = []
    for ref in refs:
        out += [f"Reference: {ref}", "",
                "| curve | " + " | ".join(f"G={g:.4f}" for g in grid) + f" | G_mid={mid[0]:.4f} | #levels CI>0 |",
                "|---|" + "---|" * (len(grid) + 2)]
        for c in targets:
            if c == ref:
                continue
            dpt = point[c] - point[ref]
            lo_ci = np.percentile(boot[(c, ref)], 2.5, axis=0)
            hi_ci = np.percentile(boot[(c, ref)], 97.5, axis=0)
            cells = [f"{dpt[j]:+.4f} [{lo_ci[j]:+.4f}, {hi_ci[j]:+.4f}]" for j in range(len(levels))]
            npos = int((lo_ci[:len(grid)] > 0).sum())
            out.append(f"| {c} | " + " | ".join(cells) + f" | {npos} |")
            gate.append((c, ref, npos))
        out.append("")
    out += ["### Gate line (numbers only)", ""]
    a_mean = {c: float(np.mean(v)) for c, v in aucs.items()}
    above = [c for c in targets if c not in refs and all(a_mean[c] > a_mean[r] for r in refs)]
    out.append(f"- AUC mean: penalty {a_mean.get('penalty', float('nan')):.4f}, "
               f"swap_category {a_mean.get('swap_category', float('nan')):.4f}.")
    out.append(f"- swap_*/swapsess_* with AUC above both: {', '.join(f'{c} ({a_mean[c]:.4f})' for c in above) or 'none'}.")
    for c in above:
        n = {r: npos for cc, r, npos in gate if cc == c}
        out.append(f"  - {c}: levels with CI of Δ > 0: vs penalty {n.get('penalty')}/{len(grid)}, "
                   f"vs swap_category {n.get('swap_category')}/{len(grid)}.")
    out.append("")

    three = np.array([lo, (lo + hi) / 2, hi])
    out += ["### LT Hit@20, exp_head_share, swaps per list at Gini low / mid / high (test, seed mean ± std)", "",
            f"Levels: {three[0]:.4f} / {three[1]:.4f} / {three[2]:.4f}.", "",
            "| curve | LT Hit@20 | exp_head_share | swaps/list |", "|---|---|---|---|"]
    for c in curves:
        cells = []
        for m in ("lt_hit20", "exp_head_share", "n_swaps"):
            per = np.array([curve_at(pts[(c, s)], m, three) for s in seeds])
            cells.append(" / ".join(f"{per[:, j].mean():.4f} ± {per[:, j].std():.4f}" for j in range(3)))
        out.append(f"| {c} | " + " | ".join(cells) + " |")
    out.append("")

    svg_curves = {}
    for c in curves:
        clo = max(min(r["exp_gini"] for r in pts[(c, s)]) for s in seeds)
        chi = min(max(r["exp_gini"] for r in pts[(c, s)]) for s in seeds)
        gx = np.linspace(clo, chi, 60) if chi > clo else np.array([clo])
        svg_curves[c] = (gx, np.mean([curve_at(pts[(c, s)], "ndcg20", gx) for s in seeds], axis=0))
    svg_points = {c: (float(np.mean([pts[(c, s)][0]["exp_gini"] for s in seeds])),
                      float(np.mean([pts[(c, s)][0]["ndcg20"] for s in seeds])))
                  for c in names if c in POINT_METHODS}
    svg_dir.mkdir(parents=True, exist_ok=True)
    path = svg_dir / f"tradeoff_{city}.svg"
    path.write_text(svg_plot(city, svg_curves, svg_points, lo, hi), encoding="utf-8")
    out += [f"![trade-off {city}]({path.name})", ""]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cities", nargs="+", default=list(CITIES), choices=CITIES)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--boot-seed", type=int, default=0)
    args = ap.parse_args()
    parts = ["# Trade-off nDCG@20 vs exposure Gini (val Pareto selection, test report)", "",
             "Chỉ báo số.", ""]
    for city in args.cities:
        parts += report_city(args.dir, city, args.boot, args.boot_seed, args.out.parent)
        print(f"[{city}] done", flush=True)
    args.out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
