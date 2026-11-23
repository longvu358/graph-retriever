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


# --- Pool-level rankers (plans C4, C5). `rel` is a full-length (N,) vector in which every
# item outside the candidate pool is NEG_REL, so the rankers only see the pool. ---

NEG_REL = -1e9
POOL_FLOOR = -1e8  # rel above this value is inside the pool


def pool_mask(rel: np.ndarray) -> np.ndarray:
    return rel > POOL_FLOOR


def aspect_dist_uniform(aspects: list[frozenset], pool: np.ndarray) -> dict:
    """P(a|q) uniform over the aspects that occur in the pool."""
    present = sorted({a for i in np.flatnonzero(pool) for a in aspects[i]})
    return {a: 1.0 / len(present) for a in present} if present else {}


def aspect_dist_prefix(aspects: list[frozenset], pool: np.ndarray, prefix: list[int]) -> dict:
    """P(a|q) from the prefix: Laplace-smoothed counts of prefix items per aspect, over the
    aspects that occur in the pool. Prefix aspects missing from the pool get no mass."""
    present = sorted({a for i in np.flatnonzero(pool) for a in aspects[i]})
    if not present:
        return {}
    counts = {a: 1.0 for a in present}  # Laplace
    for v in prefix:
        for a in aspects[v]:
            if a in counts:
                counts[a] += 1.0
    total = sum(counts.values())
    return {a: c / total for a, c in counts.items()}


def rank_xquad(
    rel: np.ndarray,
    k: int,
    aspects: list[frozenset],
    p_aspect: dict,
    lam: float,
) -> list[int]:
    """xQuAD (Santos et al. 2010) over the pool, uniform P(c|a) = 1 / |pool ∩ a|.

    score(c | S) = (1 - lam) * P(c|q) + lam * sum_a P(a|q) P(c|a) prod_{j in S} (1 - P(j|a))
    P(c|q) is the softmax of `rel` over the pool. Items without any aspect only get the
    relevance term.
    """
    pool = pool_mask(rel)
    idx = np.flatnonzero(pool)
    z = rel[idx] - rel[idx].max()
    p_rel = np.exp(z) / np.exp(z).sum()
    n_a: dict = {}
    for i in idx:
        for a in aspects[i]:
            n_a[a] = n_a.get(a, 0) + 1
    m_a: dict = {a: 0 for a in n_a}  # selected items per aspect
    chosen: list[int] = []
    taken = np.zeros(rel.shape[0], dtype=bool)
    for _ in range(min(k, idx.size)):
        best, best_score = -1, -np.inf
        for pos, c in enumerate(idx):
            if taken[c]:
                continue
            div = 0.0
            for a in aspects[c]:
                if a not in p_aspect:
                    continue
                inv = 1.0 / n_a[a]
                div += p_aspect[a] * inv * (1.0 - inv) ** m_a[a]
            score = (1.0 - lam) * p_rel[pos] + lam * div
            if score > best_score:
                best, best_score = int(c), score
        chosen.append(best)
        taken[best] = True
        for a in aspects[best]:
            if a in m_a:
                m_a[a] += 1
    return chosen


def rank_prefix_subst(
    rel: np.ndarray,
    k: int,
    prefix: list[int],
    pop: np.ndarray,
    is_head: np.ndarray,
    share: np.ndarray,
    gamma: float,
    mode: str,
) -> list[int]:
    """Prefix-anchored substitution (plan C5).

    H = head POIs in the prefix. A candidate c is boosted when c is a tail POI not in the
    prefix and shares a hyperedge (area or theme) with some h in H.
      mode "binary": score = rel + gamma * 1[boost]
      mode "gap":    score = rel + gamma * max_{h in H, share(h,c)} (pop[h] - pop[c]) on boosted c
    With H empty the ranking is the plain top-k by rel.
    """
    if mode not in ("binary", "gap"):
        raise ValueError(mode)
    heads = sorted({v for v in prefix if is_head[v]})
    if not heads:
        s = rel.copy()
    else:
        in_prefix = np.zeros(rel.shape[0], dtype=bool)
        in_prefix[list(prefix)] = True
        share_h = share[:, heads]  # (N, |H|)
        boosted = (~is_head) & (~in_prefix) & share_h.any(axis=1)
        if mode == "binary":
            bonus = boosted.astype(np.float64)
        else:
            gap = np.where(share_h, pop[heads][None, :] - pop[:, None], -np.inf).max(axis=1)
            bonus = np.where(boosted, np.maximum(gap, 0.0), 0.0)
        s = rel + gamma * bonus
    return np.argsort(-s, kind="stable")[:k].tolist()


# --- Clean in-list swap (plan D2): no lambda, no coverage. ---

SWAP_CHANNELS = ("theme", "area", "both", "union", "category")


def channel_masks(hg: Hypergraph, category: np.ndarray) -> dict[str, np.ndarray]:
    """(N, N) bool substitution masks per channel, zero diagonal.

    theme: share >= 1 theme; area: same area; both: same area and share a theme;
    union: share_matrix (area or theme); category: same flat category (hypergraph.flat_category).
    """
    n = hg.n_items
    theme = np.zeros((n, n), dtype=bool)
    by_theme: dict[str, list[int]] = {}
    for node, names in hg.theme_edges.items():
        for name in names:
            by_theme.setdefault(name, []).append(node)
    for members in by_theme.values():
        idx = np.array(members)
        theme[np.ix_(idx, idx)] = True
    area = (hg.area[:, None] == hg.area[None, :]) & (hg.area[:, None] >= 0)
    cat = (category[:, None] == category[None, :]) & (category[:, None] >= 0)
    out = {"theme": theme, "area": area, "both": theme & area, "union": share_matrix(hg),
           "category": cat}
    for m in out.values():
        np.fill_diagonal(m, False)
    return out


def pop_threshold(pop: np.ndarray, frac: float) -> float:
    """pop value of the ceil(frac * N)-th most popular POI: pop >= threshold is the top frac."""
    order = np.sort(pop)[::-1]
    return float(order[max(1, int(np.ceil(frac * pop.shape[0]))) - 1])


def swap_sequence(
    rel: np.ndarray,
    pop: np.ndarray,
    sub: np.ndarray,
    k: int,
    tau: float,
    delta: float,
    budget: int,
    select: str = "rel",
    bonus: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> list[tuple[int, int, float]]:
    """Swaps applied to the top-k of a pool, in local pool coordinates.

    `rel`, `pop`, `bonus` are (P,) over the pool, sorted by rel descending (position 0..k-1
    is the starting list); `sub` is the (P, P) pool block of the substitution mask.
    Heads of the starting list with pop >= tau are visited by pop descending (ties: list
    position). Candidates of u: not in the current list, pop < pop(u), sub[u, v],
    rel(v) >= rel(u) - delta. Returns (position, v, pop(u)) per swap, at most `budget`.
    A visited head without candidates costs no budget and draws no random number, so a
    run with a lower tau or budget is a prefix of this sequence.
    """
    if select not in ("rel", "rel+session", "random"):
        raise ValueError(select)
    k = min(k, rel.shape[0])
    in_list = np.zeros(rel.shape[0], dtype=bool)
    in_list[:k] = True
    score = rel if bonus is None or select != "rel+session" else rel + bonus
    heads = [p for p in range(k) if pop[p] >= tau]
    heads.sort(key=lambda p: (-pop[p], p))
    out: list[tuple[int, int, float]] = []
    for u in heads:
        if len(out) >= budget:
            break
        ok = sub[u] & ~in_list & (pop < pop[u]) & (rel >= rel[u] - delta)
        if not ok.any():
            continue
        cand = np.flatnonzero(ok)
        if select == "random":
            v = int(cand[rng.integers(cand.size)])
        else:
            v = int(cand[np.argmax(score[cand])])
        in_list[u] = False
        in_list[v] = True
        out.append((u, v, float(pop[u])))
    return out


def rank_swap(
    rel: np.ndarray,
    k: int,
    pop: np.ndarray,
    pool: np.ndarray,
    sub_mask: np.ndarray,
    tau: float,
    delta: float,
    budget: int,
    select: str = "rel",
    w: float = 0.0,
    prefix_edges: set | None = None,
    rng: np.random.Generator | None = None,
    item_edges: list | None = None,
) -> tuple[list[int], int]:
    """Clean swap reranker (plan D2). Start from the top-k of `pool` by `rel`; no lambda and
    no coverage. Each head u (pop(u) >= tau, a pop value, see pop_threshold) of the list is
    replaced in place by one candidate v (see swap_sequence), at most `budget` swaps.

    select "rel": max rel(v). "rel+session": max rel(v) + w * |edges(v) ∩ prefix_edges|,
    edges from `item_edges` (per-item set of theme/area keys). "random": uniform over the
    candidates with `rng` (the D2 control, used with a long-tail-only `sub_mask`).
    `rel`, `pop` are (N,), `sub_mask` (N, N), `pool` item ids. Returns (list, number of swaps).
    """
    pool = np.asarray(pool)
    order = pool[np.argsort(-rel[pool], kind="stable")]
    bonus = None
    if select == "rel+session":
        pe = prefix_edges or set()
        bonus = w * np.array([len(item_edges[i] & pe) for i in order], dtype=np.float64)
    seq = swap_sequence(rel[order], pop[order], sub_mask[np.ix_(order, order)], k, tau, delta,
                        budget, select, bonus, rng)
    lst = order[:k].copy()
    for pos, v, _ in seq:
        lst[pos] = order[v]
    return lst.tolist(), len(seq)
