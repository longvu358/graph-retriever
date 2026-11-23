import numpy as np

from graph_retriever.dispersal import rank_prefix_subst


def _toy():
    # items 0 and 1 are head; 2, 3, 4 are tail. Sharing: 0-2 and 1-4.
    is_head = np.array([True, True, False, False, False])
    pop = np.array([0.9, 0.8, 0.1, 0.2, 0.3])
    share = np.zeros((5, 5), dtype=bool)
    for u, v in [(0, 2), (1, 4)]:
        share[u, v] = share[v, u] = True
    rel = np.array([-0.1, -0.3, -0.5, -0.6, -0.7])
    return is_head, pop, share, rel


def test_gamma_zero_equals_plain_top_k():
    is_head, pop, share, rel = _toy()
    plain = np.argsort(-rel, kind="stable")[:3].tolist()
    for mode in ("binary", "gap"):
        assert rank_prefix_subst(rel, 3, [0], pop, is_head, share, 0.0, mode) == plain


def test_binary_boost_moves_sharing_tail_item_up():
    is_head, pop, share, rel = _toy()
    # prefix [0]: H = {0}; item 2 is tail, not in prefix, shares with 0 -> boosted by 1.0
    assert rank_prefix_subst(rel, 2, [0], pop, is_head, share, 1.0, "binary") == [2, 0]


def test_gap_boost_uses_popularity_gap():
    is_head, pop, share, rel = _toy()
    # boost for item 2 = pop[0] - pop[2] = 0.8 -> score -0.5 + 0.8 = 0.3 > -0.1 (item 0)
    assert rank_prefix_subst(rel, 3, [0], pop, is_head, share, 1.0, "gap") == [2, 0, 1]


def test_empty_head_prefix_is_plain():
    is_head, pop, share, rel = _toy()
    assert rank_prefix_subst(rel, 3, [2], pop, is_head, share, 4.0, "binary") == [0, 1, 2]


def test_prefix_items_are_not_boosted():
    is_head, pop, share, rel = _toy()
    # item 2 in the prefix: it must not get the bonus
    out = rank_prefix_subst(rel, 5, [0, 2], pop, is_head, share, 4.0, "binary")
    assert out.index(2) > out.index(0)
