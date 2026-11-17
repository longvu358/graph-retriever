"""Per-query significance of HGNN v2 against each baseline (plan A2).

Hit@20 is binary, so the paired comparison uses McNemar's exact test. nDCG@20 uses the
paired t-test and the Wilcoxon signed-rank test. Holm-Bonferroni runs across the pairs of
one (city, seed). A 95% bootstrap CI covers the Recall@20 difference. Each seed is tested
on its own; seeds are not averaged before testing, so n stays the number of queries.

Usage:
    python -m graph_retriever.significance --raw <TripBuilderRawData> \
        --pool results/pool --scores results/scores --out results/a2_significance
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import stats

from graph_retriever.data import CITIES, load_city
from graph_retriever.dataset import prepare

K = 20
TOP = 100
BASELINES = {"GRU (no graph)": "gru", "GCE-GNN": "gcegnn", "DHCN": "dhcn", "SR-GNN": "srgnn",
             "GRU4Rec": "gru4rec"}
# Recall@20 of v2 and GRU (no graph), mean over 3 seeds, from plan/report_comparison.md.
REFERENCE_RECALL = {
    "florence": {"HGNN v2": 0.497, "GRU (no graph)": 0.480},
    "pisa": {"HGNN v2": 0.830, "GRU (no graph)": 0.806},
    "rome": {"HGNN v2": 0.720, "GRU (no graph)": 0.714},
}


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def per_query(rankings: np.ndarray, targets: np.ndarray, k: int = K) -> tuple[np.ndarray, np.ndarray]:
    """Hit@k and nDCG@k for each query (one target per query)."""
    hits = rankings[:, :k] == targets[:, None]
    found = hits.any(axis=1)
    pos = hits.argmax(axis=1)
    ndcg = np.where(found, 1.0 / np.log2(pos + 2), 0.0)
    return found.astype(np.float64), ndcg


def mcnemar_exact(a_hit: np.ndarray, b_hit: np.ndarray) -> float:
    a = a_hit.astype(bool)
    b = b_hit.astype(bool)
    n01 = int((a & ~b).sum())
    n10 = int((~a & b).sum())
    n = n01 + n10
    if n == 0:
        return 1.0
    return float(stats.binomtest(n01, n, 0.5).pvalue)


def paired_ndcg_tests(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if np.allclose(a, b):
        return 1.0, 1.0
    t_p = float(stats.ttest_rel(a, b).pvalue)
    w_p = float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
    return t_p, w_p


def bootstrap_ci(diff: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    lo, hi = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def holm(pvals: list[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, same order as the input."""
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def compare_seed(city: str, seed: int, v2: np.ndarray, targets: np.ndarray,
                 others: dict[str, np.ndarray]) -> list[dict]:
    """One row per baseline for this (city, seed). v2 and others are (Q, >=K) rankings."""
    v2_hit, v2_ndcg = per_query(v2, targets)
    rows = []
    for name, rank in others.items():
        hit, ndcg = per_query(rank, targets)
        t_p, w_p = paired_ndcg_tests(v2_ndcg, ndcg)
        diff = v2_hit - hit
        lo, hi = bootstrap_ci(diff, seed=seed)
        rows.append({
            "city": city, "seed": seed, "pair": f"HGNN v2 vs {name}",
            "recall_v2": float(v2_hit.mean()), "recall_base": float(hit.mean()),
            "delta_recall": float(diff.mean()), "ci_lo": lo, "ci_hi": hi,
            "p_mcnemar": mcnemar_exact(v2_hit, hit),
            "ndcg_v2": float(v2_ndcg.mean()), "ndcg_base": float(ndcg.mean()),
            "p_ttest": t_p, "p_wilcoxon": w_p,
        })
    for key in ("p_mcnemar", "p_ttest", "p_wilcoxon"):
        adj = holm([r[key] for r in rows]) if rows else []
        for r, a in zip(rows, adj):
            r[key + "_holm"] = float(a)
    return rows


def reference_check(rows: list[dict]) -> list[str]:
    """Recomputed mean Recall@20 against plan/report_comparison.md (tolerance 0.002)."""
    out = ["", "## Kiểm tra khớp với report_comparison.md", "",
           "| Thành phố | Method | R@20 tính lại | R@20 trong report | Lệch | OK |", "|---|---|---|---|---|---|"]
    for city, ref in REFERENCE_RECALL.items():
        sub = [r for r in rows if r["city"] == city]
        if not sub:
            continue
        got = {
            "HGNN v2": float(np.mean([r["recall_v2"] for r in sub])),
            "GRU (no graph)": float(np.mean([r["recall_base"] for r in sub if r["pair"].endswith("GRU (no graph)")] or [np.nan])),
        }
        for method, want in ref.items():
            have = got[method]
            ok = abs(have - want) < 0.002
            out.append(f"| {city} | {method} | {have:.4f} | {want:.3f} | {have - want:+.4f} | "
                       f"{'yes' if ok else 'NO'} |")
    return out


def load_pool(pool_dir: Path, city: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(pool_dir / f"{city}_seed{seed}_test.npz")
    return z["items"].astype(np.int64), z["targets"].astype(np.int64)


def load_baseline(scores_dir: Path, stem: str, city: str, seed: int) -> np.ndarray | None:
    path = scores_dir / f"{stem}_{city}_seed{seed}.npy"
    if not path.exists():
        return None
    return topk(np.load(path), TOP)


def render(rows: list[dict], missing: list[str]) -> str:
    lines = ["# A2: kiểm định thống kê, HGNN v2 vs baseline", "",
             "Test theo từng seed (query là đơn vị mẫu). Holm–Bonferroni trong từng (thành phố, seed).",
             "Hit@20: McNemar exact. nDCG@20: paired t-test và Wilcoxon signed-rank.",
             "ΔR@20 = Recall@20(v2) − Recall@20(baseline), CI 95% bootstrap theo query.", ""]
    if missing:
        lines += ["**Thiếu score, không so sánh được:** " + ", ".join(missing), ""]
    lines += reference_check(rows) + [""]
    for city in CITIES:
        sub = [r for r in rows if r["city"] == city]
        if not sub:
            continue
        lines += [f"## {city}", "",
                  ("| Cặp | seed | R@20 v2 | R@20 base | ΔR@20 [CI 95%] | p McNemar (Holm) | "
                   "nDCG v2 | nDCG base | p t (Holm) | p Wilcoxon (Holm) |"),
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in sub:
            lines.append(
                f"| {r['pair']} | {r['seed']} | {r['recall_v2']:.4f} | {r['recall_base']:.4f} | "
                f"{r['delta_recall']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | "
                f"{r['p_mcnemar']:.3g} ({r['p_mcnemar_holm']:.3g}) | "
                f"{r['ndcg_v2']:.4f} | {r['ndcg_base']:.4f} | "
                f"{r['p_ttest']:.3g} ({r['p_ttest_holm']:.3g}) | "
                f"{r['p_wilcoxon']:.3g} ({r['p_wilcoxon_holm']:.3g}) |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--pool", type=Path, default=Path("results/pool"))
    ap.add_argument("--scores", type=Path, default=Path("results/scores"))
    ap.add_argument("--out", type=Path, default=Path("results/a2_significance"))
    ap.add_argument("--cities", nargs="+", default=list(CITIES), choices=CITIES)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()

    rows: list[dict] = []
    missing: set[str] = set()
    for city in args.cities:
        prep = prepare(load_city(args.raw, city))
        for seed in args.seeds:
            v2, pool_targets = load_pool(args.pool, city, seed)
            targets = np.array([s.target for s in prep.test], dtype=np.int64)
            if not np.array_equal(pool_targets, targets):
                raise SystemExit(f"[{city}] seed={seed}: pool targets do not match prep.test order")
            others: dict[str, np.ndarray] = {}
            for name, stem in BASELINES.items():
                rank = load_baseline(args.scores, stem, city, seed)
                if rank is None:
                    missing.add(f"{name} ({city} seed{seed})")
                    continue
                others[name] = rank
            rows += compare_seed(city, seed, v2, targets, others)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.md").write_text(render(rows, sorted(missing)), encoding="utf-8")
    print(f"wrote {args.out / 'summary.md'} ({len(rows)} rows, {len(missing)} missing)")


if __name__ == "__main__":
    main()
