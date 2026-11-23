"""Figures for the trade-off / novelty metrics of plan D4 (CSV from pool_rerank --tradeoff --novelty).

Selection (same rule as graph_retriever.tradeoff): for each curve (method[_variant]) and seed,
the settings on the val Pareto front (lower exp_gini, higher ndcg20). A setting kept for at
least one seed is plotted at the mean of its 3-seed TEST values; error bars = std over seeds.
Lines join the kept settings of one curve in order of mean test Gini. plain and ours_tune_v2
are single points. Curves with no val rows (plan D5 baselines, test scores only) keep the
Pareto front of their 3-seed mean test points instead (optimistic, the same for every curve).

Figures written to --out (PNG and SVG):
  tradeoff_<y>.png      x = exp_gini, y = metric, one panel per city
  pair_<x>__<y>.png     any extra (x, y) pair given by --pairs, default nDCG vs novelty
  summary_<city>.csv    the plotted points (curve, params, mean, std of every metric)

Usage (matplotlib is not a project dependency, so pull it in for the run):
    uv run --with matplotlib python -m graph_retriever.plot_metrics \
        --dir results_remote/d4_novelty --out results_remote/d4_novelty/fig
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from graph_retriever.data import CITIES
from graph_retriever.tradeoff import curve_key, pareto_front

LABELS = {
    "ndcg20": "nDCG@20",
    "recall20": "Recall@20",
    "lt_hit20": "Long-tail Hit@20",
    "coverage": "Coverage",
    "exp_gini": "Exposure Gini (lower = more dispersed)",
    "exp_entropy": "Exposure entropy (normalised)",
    "exp_head_share": "Head exposure share",
    "arp": "ARP",
    "epc20": "EPC@20 (rel-aware)",
    "epc20_all": "EPC@20 (whole list)",
    "epd20": "EPD@20 (rel-aware, cosine)",
    "epd20_all": "EPD@20 (whole list, cosine)",
    "epd20_cat_all": "EPD@20 (whole list, category)",
    "ild20": "ILD@20 (category)",
    "self_info20": "Self-information@20",
}
DEFAULT_Y = ("ndcg20", "recall20", "epc20", "epc20_all", "epd20", "epd20_all",
             "epd20_cat_all", "ild20", "coverage", "lt_hit20")
DEFAULT_PAIRS = ("epd20_all:ndcg20", "epc20_all:ndcg20", "exp_gini:epd20_all")

# Fixed curve -> colour slot (categorical order of the dataviz reference palette), never
# reassigned by rank. At most 8 lines; reference points are neutral.
CURVE_COLORS = {
    "penalty": "#2a78d6",
    "mmr": "#eb6834",
    "xquad_area_prefix": "#1baf7a",
    "xquad_cat_prefix": "#eda100",
    "swap_category": "#e87ba4",
    "swap_union": "#008300",
    "swapsess_union": "#6250d6",
    "swap_random": "#e34948",
}
# plan D5: one curve per retriever under the same penalty (curve key penalty_<retriever>)
RETRIEVER_COLORS = {
    "penalty_v2": "#2a78d6",
    "penalty_gru": "#eb6834",
    "penalty_gru4rec": "#1baf7a",
    "penalty_srgnn": "#eda100",
    "penalty_gcegnn": "#e87ba4",
    "penalty_dhcn": "#6250d6",
    # plan D6: v2 trained with logit adjustment tau (srgnn's slot is reused only when srgnn is
    # filtered out with --curves, so no two curves of one figure share a colour)
    "penalty_v2_lq0.5": "#008300",
    "penalty_v2_lq1": "#e34948",
}
D6_SWAP = {"penalty_v2_lq0.25": "#eda100"}
# plan D8: v2 trained with multi-target (fg) or the dispersion regulariser (db); these reuse
# baseline slots, so a figure may not mix them with the baselines that own those colours
D8_COLORS = {"penalty_v2_fg0.25": "#1baf7a", "penalty_v2_fg0.5": "#e87ba4",
             "penalty_v2_db0.1": "#6250d6", "penalty_v2_db1": "#e34948",
             "penalty_v2_db0.01": "#eb6834",
             # plan D9 (same slots; the clash check refuses a figure that mixes owners)
             "penalty_v2_split": "#e87ba4", "penalty_v2_split_lq0.25": "#1baf7a",
             "penalty_v2_lq0.25_db0.1": "#6250d6", "penalty_v2_lq0.25_db0.3": "#e34948",
             "penalty_v2_lq0.25_db1": "#eb6834"}
POINTS = {"plain": ("o", "#1a1a19"), "ours_tune_v2": ("D", "#6b6a64")}
INK, MUTED, GRID = "#1a1a19", "#6b6a64", "#e4e3dd"


def load(d: Path, city: str) -> list[dict]:
    rows = []
    for path in sorted(d.glob(f"{city}_seed*.csv")):
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out = {"curve": curve_key({"method": r["method"], "variant": r.get("variant") or ""}),
                       "params": r["params"], "seed": int(r["seed"]), "split": r["split"]}
                for k, v in r.items():
                    if k not in ("city", "method", "variant", "params", "seed", "split"):
                        out[k] = float(v) if v not in ("", None) else float("nan")
                rows.append(out)
    return rows


def kept_points(rows: list[dict], metrics: list[str]) -> dict[str, list[dict]]:
    """curve -> list of {params, <m>: mean, <m>_std: std} over seeds, sorted by mean Gini."""
    val: dict[tuple[str, int], list[dict]] = {}
    test: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        if r["split"] == "val":
            val.setdefault((r["curve"], r["seed"]), []).append(r)
        else:
            test.setdefault((r["curve"], r["params"]), []).append(r)
    keep: dict[str, set[str]] = {}
    for (curve, _seed), rs in val.items():
        idx = pareto_front(np.array([r["exp_gini"] for r in rs]), np.array([r["ndcg20"] for r in rs]))
        keep.setdefault(curve, set()).update(rs[i]["params"] for i in idx)
    for curve, params in test:  # curves with no val rows (plan D5 baselines): keep every setting
        if not any(c == curve for c, _ in val):
            keep.setdefault(curve, set()).add(params)
    out: dict[str, list[dict]] = {}
    for curve, params in keep.items():
        pts = []
        for p in params:
            rs = test.get((curve, p), [])
            if not rs:
                continue
            pt = {"params": p, "n_seeds": len(rs)}
            for m in metrics:
                v = np.array([r.get(m, np.nan) for r in rs], dtype=float)
                pt[m], pt[m + "_std"] = float(np.nanmean(v)), float(np.nanstd(v))
            pts.append(pt)
        if not any(c == curve for c, _ in val) and pts:
            # no val rows (D5): keep the Pareto front of the 3-seed TEST means. Optimistic, but
            # equally so for every retriever; needed because Gini is not monotone in lam.
            front = pareto_front(np.array([q["exp_gini"] for q in pts]),
                                 np.array([q["ndcg20"] for q in pts]))
            pts = [pts[i] for i in front]
        out[curve] = sorted(pts, key=lambda q: q["exp_gini"])
    return out


def _style(ax, x: str, y: str, title: str) -> None:
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(LABELS.get(x, x), color=MUTED, fontsize=9)
    ax.set_ylabel(LABELS.get(y, y), color=MUTED, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def colors_for(pts: dict) -> dict[str, str]:
    """D5/D6 retriever palette when those curves are present, else the D2/D4 method palette."""
    if not any(c in RETRIEVER_COLORS or c in D6_SWAP or c in D8_COLORS for c in pts):
        return CURVE_COLORS
    palette = {**RETRIEVER_COLORS, **D6_SWAP, **D8_COLORS}
    used: dict[str, str] = {}
    for c in pts:
        col = palette.get(c)
        if col is None:
            continue
        if col in used and used[col] != c:
            raise SystemExit(f"{used[col]} and {c} share a colour: drop one with --curves")
        used[col] = c
    return {c: col for c, col in palette.items() if c in pts}


def draw(ax, pts: dict[str, list[dict]], x: str, y: str) -> None:
    """Lines join the settings of a curve in order of mean test Gini (also when x is not Gini)."""
    for curve, color in colors_for(pts).items():
        ps = pts.get(curve)
        if not ps:
            continue
        xs, ys = np.array([p[x] for p in ps]), np.array([p[y] for p in ps])
        es = np.array([p[y + "_std"] for p in ps])
        ax.errorbar(xs, ys, yerr=es, color=color, linewidth=2, marker="o", markersize=4,
                    elinewidth=0.8, capsize=0, label=curve,
                    markeredgecolor="white", markeredgewidth=0.8)
        for p in ps:  # the unmodified retriever (lam = 0) of a D5 curve
            if curve.startswith("penalty_") and p["params"] == "lam=0":
                ax.plot([p[x]], [p[y]], marker="o", markersize=9, color=color,
                        markeredgecolor="white", markeredgewidth=1.5, zorder=6)
    for curve, (marker, color) in POINTS.items():
        for p in pts.get(curve, []):
            ax.errorbar([p[x]], [p[y]], yerr=[p[y + "_std"]], color=color, marker=marker,
                        markersize=8, linestyle="none", label=curve, zorder=5,
                        markeredgecolor="white", markeredgewidth=1.2)


def figure(city_pts: dict[str, dict], x: str, y: str, path: Path, plt) -> None:
    from matplotlib.lines import Line2D

    cities = list(city_pts)
    fig, axes = plt.subplots(1, len(cities), figsize=(max(9.0, 4.6 * len(cities)), 4.4),
                             squeeze=False, layout="constrained")
    for ax, city in zip(axes[0], cities):
        draw(ax, city_pts[city], x, y)
        _style(ax, x, y, city.capitalize())
    present = {c for pts in city_pts.values() for c in pts}
    palette = colors_for({c: None for c in present})
    handles = [Line2D([], [], color=col, linewidth=2, marker="o", markersize=4,
                      label=c.removeprefix("penalty_") if palette is not CURVE_COLORS else c)
               for c, col in palette.items() if c in present]
    handles += [Line2D([], [], color=col, marker=m, markersize=7, linestyle="none", label=c)
                for c, (m, col) in POINTS.items() if c in present]
    fig.legend(handles=handles, loc="outside lower center", ncol=6, frameon=False, fontsize=8)
    fig.suptitle(f"{LABELS.get(y, y)} vs {LABELS.get(x, x).split(' (')[0]}  ·  test, mean ± std, 3 seeds",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.savefig(path.with_suffix(".png"), dpi=160, facecolor="white")
    fig.savefig(path.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def write_summary(pts: dict[str, list[dict]], metrics: list[str], path: Path) -> None:
    cols = ["curve", "params", "n_seeds"] + [c for m in metrics for c in (m, m + "_std")]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for curve, ps in sorted(pts.items()):
            for p in ps:
                w.writerow({"curve": curve, **{k: p.get(k, "") for k in cols[1:]}})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True, help="folder with {city}_seed{s}.csv")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cities", nargs="+", default=list(CITIES), choices=CITIES)
    ap.add_argument("--y", nargs="+", default=list(DEFAULT_Y), help="metrics plotted against exp_gini")
    ap.add_argument("--pairs", nargs="*", default=list(DEFAULT_PAIRS), help="extra x:y pairs")
    ap.add_argument("--curves", nargs="*", default=None, help="plot only these curve keys")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.out.mkdir(parents=True, exist_ok=True)
    rows = {c: load(args.dir, c) for c in args.cities}
    rows = {c: r for c, r in rows.items() if r}
    if not rows:
        raise SystemExit(f"no CSV in {args.dir}")
    present = set.intersection(*(set(r[0]) for r in rows.values()))
    pairs = [tuple(p.split(":")) for p in args.pairs]
    wanted = {"exp_gini", "ndcg20", *args.y, *(m for p in pairs for m in p)}
    missing = sorted(wanted - present)
    if missing:
        print(f"skip metrics not in the CSVs: {missing}")
    metrics = sorted((wanted | set(LABELS)) & present)
    if args.curves:
        rows = {c: [r for r in rs if r["curve"] in args.curves] for c, rs in rows.items()}
    city_pts = {c: kept_points(r, metrics) for c, r in rows.items()}
    for c, pts in city_pts.items():
        write_summary(pts, metrics, args.out / f"summary_{c}.csv")
    for y in args.y:
        if y in present:
            figure(city_pts, "exp_gini", y, args.out / f"tradeoff_{y}", plt)
    for x, y in pairs:
        if x in present and y in present:
            figure(city_pts, x, y, args.out / f"pair_{x}__{y}", plt)
    print(f"wrote figures to {args.out}")


if __name__ == "__main__":
    main()
