"""Exploratory statistics for plan v0 step 1 (survival checks, category tokens)."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

from graph_retriever.data import CITIES, City, load_city
from graph_retriever.dataset import Prepared, prepare


def dedupe_consecutive(visits: list[str]) -> list[str]:
    out: list[str] = []
    for v in visits:
        if not out or out[-1] != v:
            out.append(v)
    return out


def describe(city: City) -> dict:
    clusters = city.clusters
    all_visits = [v for t in city.trajectories for v in t.visits]
    unknown = {v for v in all_visits if v not in clusters}

    sessions = [dedupe_consecutive(t.visits) for t in city.trajectories]
    sessions = [s for s in sessions if len(s) >= 2]
    lengths = Counter(len(s) for s in sessions)

    n_consec_dup = sum(
        1 for t in city.trajectories for a, b in zip(t.visits, t.visits[1:]) if a == b
    )
    revisit_sessions = sum(1 for s in sessions if len(set(s)) < len(s))

    tok_count: Counter[str] = Counter()
    for c in clusters.values():
        for tok in set(c.categories):
            tok_count[tok] += 1
    n_cat_clusters = sum(1 for c in clusters.values() if c.categories)

    return {
        "city": city.name,
        "clusters": len(clusters),
        "clusters_visited": len({v for v in all_visits if v in clusters}),
        "trajectories": len(city.trajectories),
        "sessions_len>=2_after_dedupe": len(sessions),
        "session_len_mean": round(sum(lengths.elements()) / max(1, len(sessions)), 2),
        "unknown_cluster_ids_in_traj": len(unknown),
        "consecutive_duplicate_pairs": n_consec_dup,
        "sessions_with_revisit": revisit_sessions,
        "clusters_with_categories": n_cat_clusters,
        "distinct_category_tokens": len(tok_count),
        "top_tokens": tok_count.most_common(12),
    }


def split_stats(prep: Prepared) -> dict:
    train_seen = set(np.flatnonzero(prep.pop_count > 0).tolist())
    out = {
        "nodes": prep.n_items,
        "train_nodes_seen": len(train_seen),
        "train_sessions": len(prep.train_sessions),
        "samples_train/val/test": (len(prep.train), len(prep.val), len(prep.test)),
    }
    for name, split in (("val", prep.val), ("test", prep.test)):
        targets = [s.target for s in split]
        out[f"{name}_target_unseen_in_train_%"] = round(
            100 * sum(t not in train_seen for t in targets) / max(1, len(targets)), 2
        )
        out[f"{name}_target_in_prefix_%"] = round(
            100 * sum(s.target in s.prefix for s in split) / max(1, len(split)), 2
        )
    out["long_tail_nodes"] = int(prep.tail.sum())
    return out


def main(raw_root: Path) -> None:
    for name in CITIES:
        city = load_city(raw_root, name)
        stats = describe(city)
        for k, v in stats.items():
            print(f"{k}: {v}")
        for k, v in split_stats(prepare(city)).items():
            print(f"{k}: {v}")
        print()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
