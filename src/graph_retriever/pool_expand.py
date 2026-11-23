"""Pool expansion along substitution relations (plan C6). CPU only.

Source: the deep v2 pool (`results/pool_deep`, top-300 by rel, plan C2).
For a query whose prefix contains a head POI, the list of size N is:
  - the top-(N - M) of the v2 pool, then
  - M long-tail POIs taken from ranks > N - M that are not in the prefix and share a hyperedge
    (area or theme) with some head POI of the prefix, in rel order;
  - if there are fewer than M such POIs, the next items in rel order from ranks > N - M.
Queries without a head in the prefix keep the plain top-N. M = 0 is the plain pool.

Selection of M is on VALIDATION only. The criterion is the val pool recall (all queries) at
each N, the largest value, ties broken by the smaller M. Test is reported for every M.

Usage:
    python -m graph_retriever.pool_expand --raw <TripBuilderRawData> --deep results/pool_deep \
        --pool results/pool --out results/c6_pool_expansion
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from graph_retriever.data import CITIES, load_city
from graph_retriever.dataset import Prepared, prepare
from graph_retriever.dispersal import share_matrix
from graph_retriever.hypergraph import build_hypergraph

NS = (50, 100)
MS = (0, 5, 10, 20)
SEEDS = (0, 1, 2)
SPLITS = ("val", "test")
FIELDS = ("city", "seed", "split", "N", "M", "recall_all", "recall_tail", "recall_hp",
          "tail_in_list", "changed_slots", "n_q", "n_tail_q", "n_hp_q")


def expand_list(deep: np.ndarray, rel: np.ndarray, prefix: list[int], prep: Prepared,
                share: np.ndarray, n: int, m: int) -> np.ndarray:
    """One query. `deep` and `rel` are the 300 candidates sorted by rel (descending)."""
    head_pos = [v for v in prefix if not prep.tail[v]]
    if m == 0 or not head_pos:
        return deep[:n].copy()
    base = list(deep[: n - m])
    rest = deep[n - m:]
    in_prefix = set(prefix)
    heads = sorted(set(head_pos))
    # preferred: long-tail, not in prefix, shares a hyperedge with a head of the prefix
    pref = [c for c in rest if prep.tail[c] and c not in in_prefix and share[c, heads].any()]
    chosen = pref[:m]
    if len(chosen) < m:  # fill with the next items by rel
        taken = set(chosen)
        for c in rest:
            if c not in taken:
                chosen.append(c)
                taken.add(c)
            if len(chosen) == m:
                break
    return np.array(base + chosen[:m], dtype=np.int64)


def evaluate(city: str, prep: Prepared, deep_dir: Path, seed: int, split: str,
             share: np.ndarray) -> list[dict]:
    samples = prep.val if split == "val" else prep.test
    data = np.load(deep_dir / f"{city}_seed{seed}_{split}.npz")
    deep_all = data["items"].astype(np.int64)
    rel_all = data["rel"].astype(np.float64)
    targets = data["targets"].astype(np.int64)
    expected = np.array([s.target for s in samples], dtype=np.int64)
    if not np.array_equal(targets, expected):
        raise RuntimeError(f"{city} seed {seed} {split}: pool targets differ from prepare()")
    tail_t = prep.tail[targets]
    has_head = np.array([any(not prep.tail[v] for v in s.prefix) for s in samples], dtype=bool)
    rows = []
    for n in NS:
        for m in MS:
            lists = np.empty((len(targets), n), dtype=np.int64)
            changed = np.zeros(len(targets))
            for i in range(len(targets)):
                lists[i] = expand_list(deep_all[i], rel_all[i], samples[i].prefix, prep,
                                       share, n, m)
                base = deep_all[i][:n]
                changed[i] = np.sum(~np.isin(lists[i], base))
            hit = (lists == targets[:, None]).any(axis=1)
            rows.append({
                "city": city, "seed": seed, "split": split, "N": n, "M": m,
                "recall_all": float(hit.mean()),
                "recall_tail": float(hit[tail_t].mean()) if tail_t.any() else float("nan"),
                "recall_hp": float(hit[has_head].mean()) if has_head.any() else float("nan"),
                "tail_in_list": float(prep.tail[lists].sum(axis=1).mean()),
                "changed_slots": float(changed.mean()),
                "n_q": len(targets), "n_tail_q": int(tail_t.sum()), "n_hp_q": int(has_head.sum()),
            })
    return rows


def choose_m(rows: list[dict], n: int) -> int:
    """M with the largest val recall_all at size n, averaged over seeds; ties -> smaller M."""
    by_m: dict[int, list[float]] = {}
    for r in rows:
        if r["N"] == n and r["split"] == "val":
            by_m.setdefault(r["M"], []).append(r["recall_all"])
    means = {m: float(np.mean(v)) for m, v in by_m.items()}
    return int(max(means, key=lambda m: (round(means[m], 12), -m)))


def check_m0(rows: list[dict], ceiling_json: Path) -> list[str]:
    """M = 0 rows must equal the plain pool recall of C1 (results/c1_pool_ceiling)."""
    c1 = json.loads(ceiling_json.read_text(encoding="utf-8"))
    out = []
    for r in rows:
        if r["M"] != 0 or str(r["N"]) not in c1[r["city"]][str(r["seed"])][r["split"]]["by_n"]:
            continue
        ref = c1[r["city"]][str(r["seed"])][r["split"]]["by_n"][str(r["N"])]
        d_all = abs(r["recall_all"] - ref["all"]["recall"])
        d_tail = abs(r["recall_tail"] - ref["tail"]["recall"])
        out.append(f"{r['city']} seed {r['seed']} {r['split']} N={r['N']}: "
                   f"|Δ all| = {d_all:.2e}, |Δ tail| = {d_tail:.2e}")
    return out


def _ms(rows: list[dict], key: str) -> str:
    v = np.array([r[key] for r in rows], dtype=np.float64)
    return f"{np.nanmean(v):.4f} ± {np.nanstd(v):.4f}"


def render(rows: list[dict], chosen: dict, checks: list[str]) -> str:
    lines = ["# C6: mở rộng pool theo quan hệ thay thế (pool sâu top-300 của v2)", "",
             ("Chỉ báo số, không kết luận. M chọn trên val (max val recall_all, hòa thì M nhỏ hơn). "
              "Test cho mọi M."), ""]
    for city in CITIES:
        for n in NS:
            if (city, n) not in chosen:
                continue
            lines += [f"## {city}, N = {n} (M chọn trên val: {chosen[(city, n)]})", "",
                      ("| M | split | recall (tất cả) | recall (long-tail) | recall (có head trong prefix) | "
                       "ứng viên long-tail trong list | slot thay đổi so với top-N |"),
                      "|---|---|---|---|---|---|---|"]
            for m in MS:
                for split in SPLITS:
                    cur = [r for r in rows if r["city"] == city and r["N"] == n and r["M"] == m
                           and r["split"] == split]
                    if not cur:
                        continue

                    cells = [_ms(cur, key) for key in ("recall_all", "recall_tail", "recall_hp",
                                                       "tail_in_list", "changed_slots")]
                    lines.append(f"| {m} | {split} | " + " | ".join(cells) + " |")
            lines.append("")
    if checks:
        lines += ["## Kiểm tra: M = 0 so với C1", ""] + [f"- {c}" for c in checks] + [""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--deep", type=Path, default=Path("results/pool_deep"))
    ap.add_argument("--c1", type=Path, default=Path("results/c1_pool_ceiling/pool_ceiling.json"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cities", nargs="+", default=list(CITIES), choices=CITIES)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = ap.parse_args()

    rows: list[dict] = []
    for city in args.cities:
        prep = prepare(load_city(args.raw, city))
        hg = build_hypergraph(load_city(args.raw, city), prep, ("theme", "area"))
        share = share_matrix(hg)
        for seed in args.seeds:
            for split in SPLITS:
                rows += evaluate(city, prep, args.deep, seed, split, share)
                print(f"[{city}] seed={seed} {split} done", flush=True)

    chosen = {}
    for city in args.cities:
        for n in NS:
            chosen[(city, n)] = choose_m([r for r in rows if r["city"] == city and r["seed"] in args.seeds],
                                         n)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "rows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in FIELDS})
    checks = check_m0(rows, args.c1) if args.c1.exists() else ["C1 json not found"]
    (args.out / "table.md").write_text(render(rows, chosen, checks), encoding="utf-8")
    print(f"wrote {args.out / 'table.md'}", flush=True)


if __name__ == "__main__":
    main()
