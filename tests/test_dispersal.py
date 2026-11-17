import numpy as np

from graph_retriever.dispersal import (
    DispersalConfig,
    rank_dispersal,
    rank_mmr,
    share_matrix,
)
from graph_retriever.hypergraph import Hypergraph
from graph_retriever.metrics import gini, jaccard_distance, recall_ndcg


def _toy_hg() -> Hypergraph:
    # 6 nodes; areas: {0,1,2} and {3,4}; node 5 in no area. Theme "a" on {0,3,5}.
    area = np.array([0, 0, 0, 1, 1, -1])
    theme_edges = {0: ["a"], 1: [], 2: [], 3: ["a"], 4: [], 5: ["a"]}
    return Hypergraph(
        channels=("area",),
        H={},
        edges={},
        operators={},
        area=area,
        theme_edges=theme_edges,
    )


def test_share_matrix_area_and_theme():
    share = share_matrix(_toy_hg())
    assert share[0, 1] and share[1, 0]  # same area
    assert share[0, 3] and share[0, 5]  # shared theme "a"
    assert not share[1, 4]  # different area, no theme
    assert not share.diagonal().any()


def test_dispersal_returns_k_unique_items_and_substitutes_lower_pop():
    hg = _toy_hg()
    share = share_matrix(hg)
    # node 0 is the head and best; node 2 (same area, low pop) is close enough to swap in
    rel = np.array([-0.1, -0.5, -0.8, -2.5, -2.5, -2.5])
    pop = np.array([0.9, 0.1, 0.2, 0.1, 0.1, 0.1])
    cfg = DispersalConfig(lam=0.0, tau=0.5, beta=0.0, pool_factor=1, delta=1.0,
                          substitution=True, coverage=False)
    r = rank_dispersal(rel, 2, hg, pop, share, cfg)
    items = r.items
    assert len(items) == len(set(items)) == 2
    assert 0 not in items  # head item 0 was swapped out
    sub = [p for p in r.picks if p.substituted_for is not None]
    assert [(p.item, p.substituted_for) for p in sub] == [(2, 0)]
    for p in sub:
        assert p.pop < pop[p.substituted_for]
        assert share[p.substituted_for, p.item]
        assert rel[p.item] >= rel[p.substituted_for] - cfg.delta


def test_substitution_respects_relevance_budget():
    hg = _toy_hg()
    share = share_matrix(hg)
    rel = np.array([-0.1, -9.0, -9.0, -9.0, -9.0, -9.0])  # every tail item is far below item 0
    pop = np.array([0.9, 0.1, 0.2, 0.1, 0.1, 0.1])
    cfg = DispersalConfig(lam=0.0, tau=0.5, beta=0.0, pool_factor=1, delta=1.0,
                          substitution=True, coverage=False)
    r = rank_dispersal(rel, 1, hg, pop, share, cfg)
    assert r.items == [0]
    assert all(p.substituted_for is None for p in r.picks)


def test_coverage_prefers_new_area_when_scores_tie():
    hg = _toy_hg()
    share = share_matrix(hg)
    rel = np.zeros(6)
    pop = np.zeros(6)
    cfg = DispersalConfig(lam=0.0, beta=1.0, pool_factor=3,
                          substitution=False, coverage=True)
    r = rank_dispersal(rel, 2, hg, pop, share, cfg)
    areas = {hg.area[p.item] for p in r.picks}
    assert len(areas) == 2


def test_mmr_theta_zero_is_plain_topk():
    rel = np.array([-1.0, -0.5, -2.0, -0.7])
    sim = np.eye(4)
    assert rank_mmr(rel, 2, sim, theta=0.0) == [1, 3]


def test_recall_ndcg_single_target():
    rankings = np.array([[3, 1, 2], [0, 1, 2]])
    targets = np.array([1, 5])
    rec, ndcg = recall_ndcg(rankings, targets, 3)
    assert rec == 0.5
    assert np.isclose(ndcg, (1 / np.log2(3)) / 2)


def test_jaccard_distance_and_gini():
    a, b = frozenset({"x", "y"}), frozenset({"y", "z"})
    assert np.isclose(jaccard_distance(a, b), 1 - 1 / 3)
    assert gini(np.array([1.0, 1.0, 1.0])) == 0.0
    assert gini(np.array([0.0, 0.0, 3.0])) > 0.5


def test_clique_operator_rows_sum_to_one():
    from graph_retriever.hypergraph import clique_operator

    H = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.float32)
    P = clique_operator([H])
    assert np.allclose(P.sum(axis=1), 1.0)
    assert (np.diag(P) > 0).all()  # self-loop kept
