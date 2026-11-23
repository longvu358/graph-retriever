"""Score LLM reranker output in the trade-off framework (plan D7). CPU only.

Input: one or more runs, each a JSON Lines file (val and test lines may be mixed) with
    {"query_id": "pisa-pisa_seed0_test-12", "ranking": ["70", "15", ...]}
where query_id is the id of the candidates file (pool.write_candidates) and ranking is the
reranker's order of the candidate item_ids (best first). Invalid ids and duplicates are
dropped; missing candidates are appended in pool order; every repair is counted and reported.

For each (city, seed, split) the scored query set = the queries present in EVERY run. On that
same set this script also scores the references, on the same top-N candidates:
  penalty   rel − λ·pop inside the top-N candidates, λ in D2_LAMBDAS (λ = 0 = pool order)
so that exposure Gini, which depends on the query set, is comparable.

Output (same format as baseline_tradeoff, readable by plot_metrics and pair_ci):
  {out}/{city}_seed{s}.csv      rows: method = "llm" (variant = run name) or "penalty"
  {out}/lists/{city}_seed{s}.npz test lists of every setting
  {out}/repairs.csv             per run and split: queries, invalid ids, missing ids

A run name may carry a setting after a colon (name:setting, e.g. llm:div=2): runs with the same
name form one curve whose points are the settings; val rows let pair_ci select them on val.

Usage:
    python -m graph_retriever.llm_score --raw <TripBuilderRawData> --pool results/pool \
        --runs llm:div=0=out/llm_div0.jsonl llm:div=1=out/llm_div1.jsonl --out results/d7_llm
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from graph_retriever.data import load_city
from graph_retriever.dataset import prepare
from graph_retriever.metrics import (
    category_sets,
    cosine_distance_matrix,
    jaccard_similarity_matrix,
)
from graph_retriever.pool_rerank import (
    D2_LAMBDAS,
    NOVELTY_FIELDS,
    TRADEOFF_FIELDS,
    K,
    _fmt,
    novelty_context,
    tradeoff_metrics,
)

QID = re.compile(r"^(?P<city>[a-z]+)-(?P=city)_seed(?P<seed>\d+)_(?P<split>val|test)-(?P<q>\d+)$")
FIELDS = TRADEOFF_FIELDS + NOVELTY_FIELDS


def parse_run(spec: str) -> tuple[str, str, Path]:
    """'name[:setting]=path' -> (name, setting, path). The setting may contain '=' (div=2):
    the path is what follows the LAST '='."""
    head, path = spec.rsplit("=", 1)
    name, _, setting = head.partition(":")
    return name, setting, Path(path)


def load_rankings(path: Path) -> dict[tuple[str, int, str], dict[int, list[str]]]:
    """(city, seed, split) -> {query index -> ranking}."""
    out: dict = defaultdict(dict)
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            m = QID.match(row["query_id"])
            if not m:
                raise ValueError(f"{path}:{n}: bad query_id {row['query_id']!r}")
            out[(m["city"], int(m["seed"]), m["split"])][int(m["q"])] = [str(x) for x in row["ranking"]]
    return out


def repair(ranking: list[str], cand: list[int], id_of: dict[str, int], stats: dict) -> list[int]:
    """Valid, de-duplicated candidates in the reranker's order, then the missing ones in pool order."""
    allowed = set(cand)
    seen: list[int] = []
    for x in ranking:
        i = id_of.get(x)
        if i is None or i not in allowed or i in seen:
            stats["invalid"] += 1
            continue
        seen.append(i)
    missing = [i for i in cand if i not in seen]
    stats["missing"] += len(missing)
    return seen + missing


def score_city_seed(args, city: str, seed: int, runs, prep, nov_ctx, id_of) -> tuple[list[dict], dict, list[dict]]:
    rows, saved, rep = [], {}, []
    for split in ("val", "test"):
        per_run = {(n, s): r.get((city, seed, split)) for (n, s), r in runs.items()}
        per_run = {k: v for k, v in per_run.items() if v}
        if not per_run:
            continue
        qs = sorted(set.intersection(*(set(v) for v in per_run.values())))
        for k, v in per_run.items():
            if len(v) != len(qs):
                print(f"[{city}] seed={seed} {split}: run {k} has {len(v)} queries, scored on the "
                      f"common {len(qs)}", flush=True)
        samples = prep.val if split == "val" else prep.test
        z = np.load(args.pool / f"{city}_seed{seed}_{split}.npz")
        items = z["items"].astype(np.int64)[qs, :args.pool_n]
        rel = z["rel"].astype(np.float64)[qs, :args.pool_n]
        targets = np.array([samples[q].target for q in qs], dtype=np.int64)
        if not np.array_equal(z["targets"].astype(np.int64)[qs], targets):
            raise RuntimeError(f"{city} seed {seed} {split}: pool targets differ from prepare()")
        nov = nov_ctx(split, qs)

        def emit(method, variant, params, lists, split=split, targets=targets, nov=nov):
            row = {"city": city, "method": method, "variant": variant, "params": params,
                   "seed": seed, "split": split, "n_swaps": 0.0}
            row.update(tradeoff_metrics(lists, targets, prep, nov))
            rows.append(row)
            if split == "test":
                key = method if not variant else f"{method}_{variant}"
                saved[f"{key}|{params}"] = lists.astype(np.int16)

        for lam in D2_LAMBDAS:
            order = np.argsort(-(rel - lam * prep.pop[items]), axis=1, kind="stable")[:, :K]
            emit("penalty", "", f"lam={_fmt(lam)}", np.take_along_axis(items, order, axis=1))
        for (name, setting), rk in per_run.items():
            stats = {"invalid": 0, "missing": 0}
            lists = np.array([repair(rk[q], list(items[j]), id_of, stats)[:K] for j, q in enumerate(qs)])
            emit("llm", name, setting or "default", lists)
            rep.append({"city": city, "seed": seed, "split": split, "run": f"{name}:{setting}",
                        "queries": len(qs), **stats})
    return rows, saved, rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--pool", type=Path, default=Path("results/pool"),
                    help="pool the candidates file was written from")
    ap.add_argument("--pool-n", type=int, default=50, help="candidates given to the reranker")
    ap.add_argument("--runs", nargs="+", required=True, help="name[:setting]=path.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    runs = {}
    for spec in args.runs:
        name, setting, path = parse_run(spec)
        runs[(name, setting)] = load_rankings(path)
    keys = sorted({k for r in runs.values() for k in r})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "lists").mkdir(exist_ok=True)
    repairs = []
    for city in sorted({c for c, _, _ in keys}):
        cityobj = load_city(args.raw, city)
        prep = prepare(cityobj)
        id_of = {nid: i for i, nid in enumerate(prep.node_ids)}
        cat_sim = jaccard_similarity_matrix(category_sets(cityobj, prep.node_ids))
        cos_d = cosine_distance_matrix(prep.train_sessions, prep.n_items)

        def nov_ctx(split, qs, prep=prep, cos_d=cos_d, cat_sim=cat_sim):
            samples = prep.val if split == "val" else prep.test
            return novelty_context(prep, [samples[q].prefix for q in qs], cos_d, cat_sim)

        for seed in sorted({s for c, s, _ in keys if c == city}):
            rows, saved, rep = score_city_seed(args, city, seed, runs, prep, nov_ctx, id_of)
            repairs += rep
            with (args.out / f"{city}_seed{seed}.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k, "") for k in FIELDS})
            qs_test = sorted(set.intersection(*(set(r.get((city, seed, "test"), {})) for r in runs.values())))
            keys_s = list(saved)
            np.savez_compressed(args.out / "lists" / f"{city}_seed{seed}.npz",
                                keys=np.array(json.dumps(keys_s)),
                                targets=np.array([prep.test[q].target for q in qs_test]),
                                n_items=np.array(prep.n_items),
                                **{f"c{j}": saved[k] for j, k in enumerate(keys_s)})
            print(f"[{city}] seed={seed}: {len(rows)} rows", flush=True)
    with (args.out / "repairs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["city", "seed", "split", "run", "queries", "invalid", "missing"])
        w.writeheader()
        w.writerows(repairs)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
