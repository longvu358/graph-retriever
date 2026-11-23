"""Accuracy and dispersal metrics.

Decisions fixed before running (plan v0 step 7):
- ILD distance = Jaccard distance over raw category tokens of each cluster.
- Long tail = the bottom LONG_TAIL_FRAC of nodes by train popularity count (dataset.py).
- HR@K equals Recall@K here: each sample has exactly one target.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from graph_retriever.data import City
from graph_retriever.dataset import Prepared

ILD_METRIC = "jaccard_categories"


def category_sets(city: City, node_ids: list[str]) -> list[frozenset[str]]:
    return [frozenset(city.clusters[c].categories) for c in node_ids]


def jaccard_distance(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def jaccard_similarity_matrix(cats: list[frozenset[str]]) -> np.ndarray:
    n = len(cats)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            s = 1.0 - jaccard_distance(cats[i], cats[j])
            sim[i, j] = sim[j, i] = s
    return sim


def recall_ndcg(rankings: np.ndarray, targets: np.ndarray, k: int) -> tuple[float, float]:
    top = rankings[:, :k]
    hits = top == targets[:, None]
    recall = float(hits.any(axis=1).mean())
    pos = np.argmax(hits, axis=1)
    found = hits.any(axis=1)
    dcg = np.where(found, 1.0 / np.log2(pos + 2), 0.0)
    return recall, float(dcg.mean())


def ild(items: list[int], cats: list[frozenset[str]]) -> float:
    k = len(items)
    if k < 2:
        return 0.0
    total, pairs = 0.0, 0
    for i in range(k):
        for j in range(i + 1, k):
            total += jaccard_distance(cats[items[i]], cats[items[j]])
            pairs += 1
    return total / pairs


def entropy(labels: list[int]) -> float:
    counts = Counter(labels)
    n = sum(counts.values())
    return -sum((c / n) * math.log(c / n) for c in counts.values())


def gini(counts: np.ndarray) -> float:
    x = np.sort(counts.astype(np.float64))
    n = x.size
    total = x.sum()
    if total == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * total) - (n + 1) / n)


def dispersal_metrics(
    lists: list[list[int]],
    prep: Prepared,
    area: np.ndarray,
    cats: list[frozenset[str]],
) -> dict[str, float]:
    n = prep.n_items
    exposure = np.zeros(n)
    for lst in lists:
        for it in lst:
            exposure[it] += 1
    area_labels = [
        [int(area[it]) if area[it] >= 0 else -(it + 2) for it in lst] for lst in lists
    ]
    return {
        "ILD": float(np.mean([ild(lst, cats) for lst in lists])),
        "coverage": float((exposure > 0).sum() / n),
        "gini": gini(exposure),
        "area_entropy": float(np.mean([entropy(a) for a in area_labels])),
        "long_tail_ratio": float(np.mean([prep.tail[lst].mean() for lst in lists])),
        "ARP": float(np.mean([prep.pop[lst].mean() for lst in lists])),
    }


# --- Novelty and aspect diversity (plan A3) ---


def novelty_pop(lists: list[list[int]], pop_count: np.ndarray) -> float:
    """Mean over lists of mean -log2 p(i), p(i) = (count_train(i) + 1) / (N_train + |I|).

    Laplace smoothing keeps p > 0 for items never seen in train. Counts are train-only.
    """
    n = pop_count.shape[0]
    p = (pop_count + 1.0) / (pop_count.sum() + n)
    nov = -np.log2(p)
    return float(np.mean([nov[lst].mean() for lst in lists]))


def unseen_fraction(lists: list[list[int]], pop_count: np.ndarray) -> float:
    """Share of list entries whose item has zero train count (reported next to novelty_pop)."""
    flat = np.concatenate([np.asarray(lst, dtype=int) for lst in lists])
    return float((pop_count[flat] == 0).mean())


def _alpha_gain(aspects: frozenset[str], seen: Counter, alpha: float) -> float:
    return sum((1.0 - alpha) ** seen[a] for a in aspects)


def _alpha_dcg(order: list[int], relevant: dict[int, frozenset[str]], alpha: float) -> float:
    seen: Counter = Counter()
    dcg = 0.0
    for r, item in enumerate(order, start=1):
        aspects = relevant.get(item, frozenset())
        dcg += _alpha_gain(aspects, seen, alpha) / math.log2(r + 1)
        seen.update(aspects)
    return dcg


def _alpha_ideal(relevant: dict[int, frozenset[str]], alpha: float, k: int) -> list[int]:
    """Greedy ideal ordering over the relevant items (Clarke et al. 2008)."""
    remaining = dict(relevant)
    seen: Counter = Counter()
    order: list[int] = []
    while remaining and len(order) < k:
        best = max(remaining, key=lambda it: (_alpha_gain(remaining[it], seen, alpha), -it))
        order.append(best)
        seen.update(remaining.pop(best))
    return order


def alpha_ndcg(
    ranking: list[int],
    relevant: dict[int, frozenset[str]],
    k: int,
    alpha: float = 0.5,
) -> float:
    """alpha-nDCG@k for one query.

    relevant maps item -> aspects for which that item is relevant to the query.
    Gain of the item at rank r = sum over its relevant aspects a of (1 - alpha)^(number of
    earlier items relevant to a). Returns 0.0 when no item is relevant.

    Plan A3 uses relevance = target only, aspects = category (or area). With one relevant
    item the redundancy term is always 0, so the value reduces to a target-position score:
    it does not measure diversity of the list. Read it with that limit in mind.
    """
    ideal = _alpha_dcg(_alpha_ideal(relevant, alpha, k), relevant, alpha)
    if ideal <= 0.0:
        return 0.0
    return _alpha_dcg(ranking[:k], relevant, alpha) / ideal


# --- Exposure dispersal over a whole split (plan D0) ---


def exposure_vector(lists: np.ndarray, n_items: int) -> np.ndarray:
    """(N,) exposure: sum over queries of 1/log2(r + 1) for the item at rank r = 1..K.

    Items that never appear get 0. N is every POI of the city, not only the pool.
    """
    lists = np.asarray(lists)
    disc = 1.0 / np.log2(np.arange(lists.shape[1]) + 2)
    return np.bincount(lists.ravel(), weights=np.tile(disc, lists.shape[0]), minlength=n_items)


def exposure_stats(exposure: np.ndarray, is_head: np.ndarray) -> dict[str, float]:
    """Gini, normalised entropy, top-1% share and head share of an exposure vector."""
    n = exposure.shape[0]
    total = exposure.sum()
    p = exposure / total
    nz = p[p > 0]
    top = max(1, int(0.01 * n))
    return {
        "exp_gini": gini(exposure),
        "exp_entropy": float(-(nz * np.log(nz)).sum() / math.log(n)),
        "exp_top1pct": float(np.sort(p)[::-1][:top].sum()),
        "exp_head_share": float(p[is_head].sum()),
    }


def exposure_metrics(lists: np.ndarray, n_items: int, is_head: np.ndarray) -> dict[str, float]:
    return exposure_stats(exposure_vector(lists, n_items), is_head)


# --- Rank- and relevance-aware novelty (Vargas & Castells 2011, as used by Lacic et al. 2020) ---
#
# Both metrics follow Lacic et al. (UMUAI 2020, Sect. 4.4):
#   EPC@k = 1/|S| sum_s 1/|R_k|         sum_i disc(i) p(rel|R_i,s) (1 - p(seen|R_i))
#   EPD@k = 1/|S| sum_s 1/(|R_k||H_s|)  sum_i sum_j disc(i) p(rel|R_i,s) d(R_i, H_j)
# with disc(i) = 1/log2(i + 1), p(seen|i) = log2(pop_i + 1) / log2(pop_max + 1).
# p(rel) is 1 only for the target here (one target per query), so the relevance-aware form
# scores the target alone. `rel_aware=False` sets p(rel) = 1 for every list item: that form
# measures the whole list. Report both.


def _disc(k: int) -> np.ndarray:
    return 1.0 / np.log2(np.arange(1, k + 1) + 1)


def p_seen(pop_count: np.ndarray) -> np.ndarray:
    """(N,) log2(pop + 1) / log2(pop_max + 1), train counts. All zero if pop_max = 0."""
    top = math.log2(float(pop_count.max()) + 1.0)
    if top == 0.0:
        return np.zeros(pop_count.shape[0])
    return np.log2(pop_count.astype(np.float64) + 1.0) / top


def _rel_weight(lists: np.ndarray, targets: np.ndarray | None) -> np.ndarray:
    if targets is None:
        return np.ones(lists.shape, dtype=np.float64)
    return (lists == np.asarray(targets)[:, None]).astype(np.float64)


def epc(lists: np.ndarray, pop_count: np.ndarray, targets: np.ndarray | None = None) -> float:
    """EPC@k over (Q, k) lists. targets=None gives the relevance-unaware form (p(rel) = 1)."""
    lists = np.asarray(lists)
    k = lists.shape[1]
    nov = 1.0 - p_seen(pop_count)
    per_q = (_disc(k)[None, :] * _rel_weight(lists, targets) * nov[lists]).sum(axis=1) / k
    return float(per_q.mean())


def cosine_distance_matrix(sessions: list[list[int]], n_items: int) -> np.ndarray:
    """(N, N) 1 - cosine over binary item x session vectors (item in session or not).

    An item with no session (zero vector) gets distance 1 to every other item and 0 to itself.
    """
    rows, cols = [], []
    for s_id, s in enumerate(sessions):
        for v in set(s):
            rows.append(v)
            cols.append(s_id)
    m = np.zeros((n_items, len(sessions)), dtype=np.float64)
    m[rows, cols] = 1.0
    co = m @ m.T
    norm = np.sqrt(np.diag(co))
    denom = np.outer(norm, norm)
    sim = np.divide(co, denom, out=np.zeros_like(co), where=denom > 0)
    d = 1.0 - sim
    np.fill_diagonal(d, 0.0)
    return d


def history_distance(dist: np.ndarray, prefixes: list[list[int]]) -> np.ndarray:
    """(Q, N) float32: row q = mean over the prefix items h of dist[:, h] (1/|H_s| sum_j d)."""
    out = np.empty((len(prefixes), dist.shape[0]), dtype=np.float32)
    for q, p in enumerate(prefixes):
        out[q] = dist[:, p].mean(axis=1)
    return out


def epd(lists: np.ndarray, hist: np.ndarray, targets: np.ndarray | None = None) -> float:
    """EPD@k over (Q, k) lists; hist from history_distance. targets=None: p(rel) = 1."""
    lists = np.asarray(lists)
    k = lists.shape[1]
    d = np.take_along_axis(hist, lists, axis=1).astype(np.float64)
    per_q = (_disc(k)[None, :] * _rel_weight(lists, targets) * d).sum(axis=1) / k
    return float(per_q.mean())


def ild_batch(lists: np.ndarray, sim: np.ndarray) -> float:
    """Mean over lists of the mean pairwise (1 - sim) inside the list. Same as `ild` with
    Jaccard when sim = jaccard_similarity_matrix."""
    lists = np.asarray(lists)
    k = lists.shape[1]
    if k < 2:
        return 0.0
    d = 1.0 - sim[lists[:, :, None], lists[:, None, :]]
    return float((d.sum(axis=(1, 2)) / (k * (k - 1))).mean())


def self_information(lists: np.ndarray, pop_count: np.ndarray) -> float:
    """novelty_pop for a (Q, k) array."""
    n = pop_count.shape[0]
    nov = -np.log2((pop_count + 1.0) / (pop_count.sum() + n))
    return float(nov[np.asarray(lists)].mean())
