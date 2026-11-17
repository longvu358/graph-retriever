"""Inference-time rankers over relevance scores.

All rankers take a log-softmax relevance vector `rel` (N,) for one session and
return a list of item ids (length K). Dispersal-aware ranking is inference-only:
no lambda enters training.

Steps of `rank_dispersal`:
  1. score s = rel - lambda * pop; pool = top max(K, pool_factor*K) by s.
  2. coverage selection (greedy over the pool): gain(c) = s(c) + beta * sum_h 1/(1+n_h),
     h ranges over area/theme edges of c, n_h = number already selected in h.
     Without coverage, the list is the top-K of the pool.
  3. substitution (swap, not pool insertion): each selected item u with pop(u) > tau is
     replaced by the best-scoring item v that shares an area or theme with u, has
     pop(v) < pop(u), and rel(v) >= rel(u) - delta.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from graph_retriever.hypergraph import Hypergraph


@dataclass(frozen=True)
class DispersalConfig:
    lam: float = 0.25
    tau: float = 0.5  # pop threshold above which an item is a substitution target
    delta: float = 1.0  # max relevance loss (nats) accepted per substitution
    beta: float = 1.0  # coverage weight
    pool_factor: int = 2  # pool size = max(K, pool_factor * K)
    substitution: bool = True
    coverage: bool = True


@dataclass
class Pick:
    item: int
    score: float
    rel: float
    pop: float
    area: int
    themes: tuple[str, ...]
    substituted_for: int | None = None


@dataclass
class Ranking:
    picks: list[Pick] = field(default_factory=list)

    @property
    def items(self) -> list[int]:
        return [p.item for p in self.picks]


def top_k(rel: np.ndarray, k: int) -> list[int]:
    return np.argsort(-rel, kind="stable")[:k].tolist()


def rank_popularity(pop_count: np.ndarray, k: int) -> list[int]:
    return np.argsort(-pop_count, kind="stable")[:k].tolist()


def share_matrix(hg: Hypergraph) -> np.ndarray:
    """(N, N) bool: u and v share an area or at least one theme."""
    n = hg.n_items
    same_area = (hg.area[:, None] == hg.area[None, :]) & (hg.area[:, None] >= 0)
    share = same_area.copy()
    by_theme: dict[str, list[int]] = {}
    for node, names in hg.theme_edges.items():
        for name in names:
            by_theme.setdefault(name, []).append(node)
    for members in by_theme.values():
        idx = np.array(members)
        share[np.ix_(idx, idx)] = True
    np.fill_diagonal(share, False)
    assert share.shape == (n, n)
    return share


def _hyperedges(hg: Hypergraph, item: int) -> list[str]:
    keys = [f"theme:{t}" for t in hg.theme_edges[item]]
    if hg.area[item] >= 0:
        keys.append(f"area:{hg.area[item]}")
    return keys


def rank_dispersal(
    rel: np.ndarray,
    k: int,
    hg: Hypergraph,
    pop: np.ndarray,
    share: np.ndarray,
    cfg: DispersalConfig,
) -> Ranking:
    s = rel - cfg.lam * pop
    pool_size = max(k, cfg.pool_factor * k)
    pool = np.argsort(-s, kind="stable")[:pool_size].tolist()

    chosen: list[int] = []
    if cfg.coverage:
        counts: dict[str, int] = {}
        while len(chosen) < k:
            best, best_gain = None, -np.inf
            for c in pool:
                if c in chosen:
                    continue
                gain = float(s[c]) + cfg.beta * sum(
                    1.0 / (1 + counts.get(h, 0)) for h in _hyperedges(hg, c)
                )
                if gain > best_gain:
                    best, best_gain = c, gain
            chosen.append(best)
            for h in _hyperedges(hg, best):
                counts[h] = counts.get(h, 0) + 1
    else:
        chosen = pool[:k]

    sub_of: dict[int, int] = {}
    if cfg.substitution:
        # Replace each head item in the final list with a tail item of the same
        # area/theme, only when its relevance is within `delta` nats of the head's.
        for pos, u in enumerate(list(chosen)):
            if pop[u] <= cfg.tau:
                continue
            ok = share[u] & (pop < pop[u]) & (rel >= rel[u] - cfg.delta)
            ok[chosen] = False
            if not ok.any():
                continue
            cand = np.flatnonzero(ok)
            v = int(cand[np.argmax(s[cand])])
            chosen[pos] = v
            sub_of[v] = u

    picks = [
        Pick(
            item=c,
            score=float(s[c]),
            rel=float(rel[c]),
            pop=float(pop[c]),
            area=int(hg.area[c]),
            themes=tuple(hg.theme_edges[c]),
            substituted_for=sub_of.get(c),
        )
        for c in chosen
    ]
    return Ranking(picks=picks)


def rank_penalty(rel: np.ndarray, pop: np.ndarray, lam: float, k: int) -> list[int]:
    """Popularity-penalty baseline: rel - lambda * pop, top-K."""
    return top_k(rel - lam * pop, k)


def rank_mmr(
    rel: np.ndarray,
    k: int,
    sim: np.ndarray,
    theta: float,
) -> list[int]:
    """MMR baseline: argmax (1 - theta) * rel - theta * max_{j in S} sim(c, j).

    `sim` is a precomputed (N, N) similarity (here 1 - Jaccard distance on categories).
    `rel` is the log-softmax score, so the MMR trade-off is applied to the same scale
    as the rest of the pipeline.
    """
    chosen: list[int] = []
    max_sim = np.zeros(len(rel))
    for _ in range(k):
        obj = (1 - theta) * rel - theta * max_sim
        if chosen:
            obj[chosen] = -np.inf
        c = int(np.argmax(obj))
        chosen.append(c)
        max_sim = np.maximum(max_sim, sim[c])
    return chosen
