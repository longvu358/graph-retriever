"""Diagnosis of the hypergraph's role for substituting head POIs with long-tail POIs (plan A4).

Reports numbers only. The interpretation is left to the reader.

Per city, the three seeds are pooled (query-level rows are concatenated).
- Head/tail split of test targets. Long tail = `prep.tail` (bottom 80% of POIs by train count).
- Long-tail target: hit in top-20 / top-50 for HGNN v2 (saved top-100 pool) and GRU (no graph,
  full score matrix). A v2 target outside its top-100 gets rank 101.
- Hyperedge share: target shares an area or a theme with at least one head POI in the prefix,
  against 100 random long-tail POIs drawn per query (same prefix, excluding the target).
  Ranks of both models are capped at POOL_MAX + 1 in the rank table, so the means compare.
- Rank of the long-tail target, on queries where the prefix contains a head POI.

Usage:
    python -m graph_retriever.diagnose --raw <TripBuilderRawData> \
        --pool results/pool --scores results/scores --out results/a4_diagnosis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from graph_retriever.data import CITIES, load_city
from graph_retriever.dataset import prepare
from graph_retriever.dispersal import share_matrix
from graph_retriever.hypergraph import build_hypergraph

POOL_MAX = 100
RANK_BINS = ((1, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 101))
RANK_LABELS = ("1-5", "6-10", "11-20", "21-50", "51-100", ">100")
RANDOM_DRAWS = 100
SEED = 0


def rank_in_scores(scores: np.ndarray, target: int) -> int:
    return int((scores > scores[target]).sum()) + 1


def rank_in_pool(items: np.ndarray, target: int) -> int:
    hit = np.flatnonzero(items == target)
    return int(hit[0]) + 1 if hit.size else POOL_MAX + 1


def bin_counts(ranks: np.ndarray) -> np.ndarray:
    """Share of queries in each rank bin. Ranks above POOL_MAX are capped at POOL_MAX + 1."""
    out = []
    for lo, hi in RANK_BINS:
        mask = (ranks >= lo) & (ranks <= hi) if hi <= POOL_MAX else ranks >= lo
        out.append(mask.mean() if ranks.size else 0.0)
    return np.array(out)


def grouped_bars_svg(path: Path, title: str, series: dict[str, np.ndarray]) -> None:
    """Dependency-free SVG bar chart. One group per rank bin, one bar per series."""
    w, h, left, bottom, top = 640, 360, 60, 50, 40
    plot_w, plot_h = w - left - 20, h - top - bottom
    colors = ("#3b6fb6", "#d9822b")
    ymax = max(0.05, max(float(v.max()) for v in series.values()) * 1.1)
    group_w = plot_w / len(RANK_LABELS)
    bar_w = group_w * 0.35
    parts = [(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
              f'font-family="sans-serif" font-size="12">'),
             f'<rect width="{w}" height="{h}" fill="white"/>',
             f'<text x="{w / 2}" y="22" text-anchor="middle" font-size="14">{title}</text>',
             f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="black"/>',
             f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="black"/>']
    for i, label in enumerate(RANK_LABELS):
        gx = left + i * group_w
        parts.append(f'<text x="{gx + group_w / 2}" y="{top + plot_h + 18}" text-anchor="middle">{label}</text>')
        for j, (name, vals) in enumerate(series.items()):
            bh = plot_h * float(vals[i]) / ymax
            x = gx + group_w / 2 + (j - 0.5) * bar_w
            parts.append(f'<rect x="{x}" y="{top + plot_h - bh}" width="{bar_w}" height="{bh}" '
                         f'fill="{colors[j % len(colors)]}"/>')
    for j, name in enumerate(series):
        lx = left + 10 + j * 160
        parts.append(f'<rect x="{lx}" y="{h - 18}" width="12" height="12" fill="{colors[j % len(colors)]}"/>')
        parts.append(f'<text x="{lx + 18}" y="{h - 7}">{name}</text>')
    parts.append(f'<text x="{left - 8}" y="{top + 4}" text-anchor="end">{ymax:.2f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def analyse_city(raw: Path, pool: Path, scores: Path, city: str, seeds: list[int]) -> dict:
    data = load_city(raw, city)
    prep = prepare(data)
    hg = build_hypergraph(data, prep, channels=("area",))
    share = share_matrix(hg)
    head = ~prep.tail
    tail_nodes = np.flatnonzero(prep.tail)
    rng = np.random.default_rng(SEED)

    acc = {k: [] for k in (
        "is_tail", "head_in_prefix", "share_target", "share_random",
        "v2_rank", "gru_rank", "gru_top20", "gru_top50", "v2_top20", "v2_top50", "seed",
    )}
    n_queries = {}
    for seed in seeds:
        z = np.load(pool / f"{city}_seed{seed}_test.npz")
        items = z["items"].astype(np.int64)
        targets = z["targets"].astype(np.int64)
        gru_path = scores / f"gru_{city}_seed{seed}.npy"
        if not gru_path.exists():
            raise SystemExit(f"missing {gru_path}: generate GRU (no graph) scores first (A2 step 1)")
        gru = np.load(gru_path)
        if gru.shape[0] != targets.size or len(prep.test) != targets.size:
            raise SystemExit(f"[{city}] seed={seed}: query count mismatch")
        n_queries[seed] = targets.size
        for q, smp in enumerate(prep.test):
            t = int(targets[q])
            is_tail = bool(prep.tail[t])
            heads = [p for p in smp.prefix if head[p]]
            has_head = len(heads) > 0
            acc["is_tail"].append(is_tail)
            acc["head_in_prefix"].append(has_head)
            acc["seed"].append(seed)
            if not is_tail:
                continue
            share_t = bool(share[t, heads].any()) if has_head else False
            cand = tail_nodes[tail_nodes != t]
            draws = rng.choice(cand, size=min(RANDOM_DRAWS, cand.size), replace=False)
            share_r = float(share[np.ix_(draws, heads)].any(axis=1).mean()) if has_head else 0.0
            r_v2 = rank_in_pool(items[q], t)
            r_gru = rank_in_scores(gru[q], t)
            acc["share_target"].append(share_t)
            acc["share_random"].append(share_r)
            acc["v2_rank"].append(r_v2)
            acc["gru_rank"].append(r_gru)
            acc["v2_top20"].append(r_v2 <= 20)
            acc["v2_top50"].append(r_v2 <= 50)
            acc["gru_top20"].append(r_gru <= 20)
            acc["gru_top50"].append(r_gru <= 50)
    return {k: np.array(v) for k, v in acc.items()} | {"n_queries": n_queries}


def render(results: dict[str, dict], seeds: list[int]) -> str:
    lines = ["# A4: chẩn đoán vai trò hypergraph (substitution)", "",
             f"Seed gộp: {seeds}. Chỉ báo số, không kết luận. Long tail = 80% POI ít phổ biến nhất (theo số POI).",
             "Rank của v2 ngoài top-100 của pool được tính là 101 (pool không lưu rank sâu hơn).", ""]
    for city, r in results.items():
        is_tail = r["is_tail"]
        n_total = int(is_tail.size)
        tail_q = int(is_tail.sum())
        head_q = n_total - tail_q
        tail_in_prefix = int((is_tail & r["head_in_prefix"]).sum())
        tail_mask = np.flatnonzero(is_tail)
        share_t = r["share_target"].astype(float)
        share_r = r["share_random"]
        lines += [f"## {city}", "",
                  f"- Query theo seed: {r['n_queries']}. Tổng query (gộp seed): {n_total}.",
                  (f"- Head target: {head_q}. Long-tail target: {tail_q}. "
                   f"Long-tail target có head trong prefix: {tail_in_prefix}."), ""]
        if tail_q:
            lines += [
                "| Chỉ số (long-tail target) | v2 | GRU (no graph) |", "|---|---|---|",
                f"| Hit@20 | {r['v2_top20'].mean():.4f} | {r['gru_top20'].mean():.4f} |",
                f"| Hit@50 | {r['v2_top50'].mean():.4f} | {r['gru_top50'].mean():.4f} |", "",
                "| Chia sẻ hyperedge với head trong prefix | target | random long-tail (trung bình 100 lần/query) |",
                "|---|---|---|",
                f"| Tỉ lệ query | {share_t.mean():.4f} | {share_r.mean():.4f} |", ""]
        sub = tail_mask[r["head_in_prefix"][tail_mask]]
        if sub.size:
            v2r = r["v2_rank"][r["head_in_prefix"][is_tail]]
            gr = np.minimum(r["gru_rank"][r["head_in_prefix"][is_tail]], POOL_MAX + 1)  # same cap as v2
            lines += [f"| Rank long-tail target, prefix có head (n = {sub.size}) | v2 | GRU |",
                      "|---|---|---|",
                      f"| Median | {np.median(v2r):.1f} | {np.median(gr):.1f} |",
                      f"| Mean | {v2r.mean():.2f} | {gr.mean():.2f} |",
                      f"| Tỉ lệ rank > 100 (v2 bị cắt) | {(v2r > POOL_MAX).mean():.4f} | {(gr > POOL_MAX).mean():.4f} |",
                      ""]
        else:
            lines += ["Không có query long-tail target nào có head trong prefix.", ""]
        lines += [f"![rank histogram {city}](hist_{city}.svg)", ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--pool", type=Path, default=Path("results/pool"))
    ap.add_argument("--scores", type=Path, default=Path("results/scores"))
    ap.add_argument("--out", type=Path, default=Path("results/a4_diagnosis"))
    ap.add_argument("--cities", nargs="+", default=list(CITIES), choices=CITIES)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    for city in args.cities:
        r = analyse_city(args.raw, args.pool, args.scores, city, args.seeds)
        results[city] = r
        tail = r["is_tail"]
        mask = r["head_in_prefix"][tail]
        v2r = r["v2_rank"][mask]
        gr = r["gru_rank"][mask]
        grouped_bars_svg(args.out / f"hist_{city}.svg", f"{city}: rank của long-tail target (có head trong prefix)",
                         {"HGNN v2": bin_counts(v2r), "GRU (no graph)": bin_counts(gr)})
    (args.out / "summary.md").write_text(render(results, args.seeds), encoding="utf-8")
    print(f"wrote {args.out / 'summary.md'}")


if __name__ == "__main__":
    main()
