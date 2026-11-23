"""Recall ceiling of the candidate pool, split by target popularity (plan C1), and the
seed-by-seed comparison of two pool directories used to check a deeper pool (plan C2).

Groups of queries (per split, per seed):
- all: every query
- head: target is a head POI (not in the long tail of train popularity)
- tail: target is a long-tail POI
- tail_head_prefix: long-tail target and the prefix contains at least one head POI

Recall@N = fraction of queries whose target is in the first N items of the stored pool.
Only sizes N <= the stored pool width are reported.

Usage:
    python -m graph_retriever.pool_ceiling --raw <TripBuilderRawData> --pool results/pool \
        --out results/c1_pool_ceiling
    python -m graph_retriever.pool_ceiling --raw <TripBuilderRawData> --compare results/pool_deep \
        results/pool --out results/c2_pool_deep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from graph_retriever.data import CITIES, load_city
from graph_retriever.dataset import prepare

NS = (10, 20, 50, 100, 200, 300)
GROUPS = ("all", "head", "tail", "tail_head_prefix")
SPLITS = ("val", "test")
SEEDS = (0, 1, 2)


def group_masks(prep, samples, targets: np.ndarray) -> dict[str, np.ndarray]:
    tgt_tail = prep.tail[targets]
    has_head = np.array([any(not prep.tail[v] for v in s.prefix) for s in samples], dtype=bool)
    return {
        "all": np.ones(len(targets), dtype=bool),
        "head": ~tgt_tail,
        "tail": tgt_tail,
        "tail_head_prefix": tgt_tail & has_head,
    }


def load_split(raw: Path, city: str, pool: Path, seed: int, split: str):
    prep = prepare(load_city(raw, city))
    samples = prep.val if split == "val" else prep.test
    data = np.load(pool / f"{city}_seed{seed}_{split}.npz")
    targets = data["targets"].astype(np.int64)
    expected = np.array([s.target for s in samples], dtype=np.int64)
    if not np.array_equal(targets, expected):
        raise RuntimeError(f"{city} seed {seed} {split}: pool targets differ from prepare()")
    return data["items"], targets, group_masks(prep, samples, targets)


def recall_by_group(items: np.ndarray, targets: np.ndarray, masks: dict[str, np.ndarray]) -> dict:
    out: dict[str, dict] = {}
    for n in NS:
        if n > items.shape[1]:
            continue
        hit = (items[:, :n] == targets[:, None]).any(axis=1)
        out[str(n)] = {g: {"recall": float(hit[m].mean()) if m.any() else None,
                           "count": int(m.sum())} for g, m in masks.items()}
    return out


def ceiling(raw: Path, pool: Path, cities: list[str], seeds: list[int]) -> dict:
    res: dict = {}
    for city in cities:
        res[city] = {}
        for seed in seeds:
            res[city][str(seed)] = {}
            for split in SPLITS:
                items, targets, masks = load_split(raw, city, pool, seed, split)
                res[city][str(seed)][split] = {
                    "pool_width": int(items.shape[1]),
                    "by_n": recall_by_group(items, targets, masks),
                }
                print(f"[{city}] seed={seed} {split}: {len(targets)} queries", flush=True)
    return res


def compare(raw: Path, deep: Path, plain: Path, cities: list[str], seeds: list[int]) -> dict:
    """Per seed and split: Recall@N of the deep pool and the plain pool, and their
    val R@20 / R@100 from the export reports (model-level check)."""
    res: dict = {}
    for city in cities:
        res[city] = {}
        for seed in seeds:
            row: dict = {}
            for split in SPLITS:
                d_items, d_t, _ = load_split(raw, city, deep, seed, split)
                p_items, p_t, _ = load_split(raw, city, plain, seed, split)
                if not np.array_equal(d_t, p_t):
                    raise RuntimeError(f"{city} seed {seed} {split}: targets differ")
                row[split] = {}
                for n in (20, 100):
                    d_hit = (d_items[:, :n] == d_t[:, None]).any(axis=1).mean()
                    p_hit = (p_items[:, :n] == p_t[:, None]).any(axis=1).mean()
                    row[split][str(n)] = {"deep": float(d_hit), "plain": float(p_hit),
                                          "diff": float(d_hit - p_hit)}
            res[city][str(seed)] = row
    return res


def _ms(vals: list[float | None]) -> str:
    v = [x for x in vals if x is not None]
    if not v:
        return "-"
    return f"{np.mean(v):.4f} ± {np.std(v):.4f}"


def render_ceiling(res: dict) -> str:
    lines = ["# C1: trần recall của pool theo nhóm target", "",
             ("Mean ± std (ddof = 0) qua 3 seed. Số query là số trong một seed; "
              "`gộp` = tổng qua 3 seed. Pool v2 top-100 (`results/pool`)."), ""]
    for city, seeds in res.items():
        for split in SPLITS:
            widths = {seeds[s][split]["pool_width"] for s in seeds}
            any_seed = next(iter(seeds.values()))[split]["by_n"]
            lines += [f"## {city}, {split}", "",
                      "| N | " + " | ".join(GROUPS) + " |",
                      "|---|" + "---|" * len(GROUPS)]
            for n in any_seed:
                cells = []
                for g in GROUPS:
                    vals = [seeds[s][split]["by_n"][n][g]["recall"] for s in seeds]
                    cells.append(_ms(vals))
                lines.append(f"| {n} | " + " | ".join(cells) + " |")
            counts = {g: [seeds[s][split]["by_n"][next(iter(any_seed))][g]["count"] for s in seeds]
                      for g in GROUPS}
            lines += ["", "Số query mỗi seed: " + ", ".join(
                f"{g} = {counts[g][0]}" for g in GROUPS)
                + "; gộp 3 seed: " + ", ".join(f"{g} = {sum(counts[g])}" for g in GROUPS),
                f"Độ rộng pool: {sorted(widths)}", ""]
    return "\n".join(lines)


def render_compare(res: dict, deep_dir: Path, plain_dir: Path) -> str:
    lines = ["# C2: so sánh pool sâu với pool cũ (top-100 của v2)", "",
             (f"Pool sâu: `{deep_dir}`. Pool cũ: `{plain_dir}`. Recall@N tính trực tiếp từ `items`, "
              "so với cùng target. Chênh lệch = sâu − cũ."), ""]
    for city, seeds in res.items():
        lines += [f"## {city}", "", "| seed | split | R@20 sâu | R@20 cũ | Δ | R@100 sâu | R@100 cũ | Δ |",
                  "|---|---|---|---|---|---|---|---|"]
        for seed, row in seeds.items():
            for split in SPLITS:
                a, b = row[split]["20"], row[split]["100"]
                lines.append(
                    f"| {seed} | {split} | {a['deep']:.4f} | {a['plain']:.4f} | {a['diff']:+.4f} | "
                    f"{b['deep']:.4f} | {b['plain']:.4f} | {b['diff']:+.4f} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--pool", type=Path, default=Path("results/pool"))
    ap.add_argument("--compare", type=Path, default=None,
                    help="deep pool directory to compare against --pool (plan C2)")
    ap.add_argument("--cities", nargs="+", default=list(CITIES), choices=CITIES)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.compare is None:
        res = ceiling(args.raw, args.pool, args.cities, args.seeds)
        (args.out / "pool_ceiling.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        (args.out / "table.md").write_text(render_ceiling(res), encoding="utf-8")
    else:
        res = compare(args.raw, args.compare, args.pool, args.cities, args.seeds)
        (args.out / "pool_compare.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        (args.out / "table.md").write_text(render_compare(res, args.compare, args.pool),
                                           encoding="utf-8")
    print(f"wrote {args.out / 'table.md'}", flush=True)


if __name__ == "__main__":
    main()
