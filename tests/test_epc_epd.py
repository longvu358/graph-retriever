import math

import numpy as np
import pytest

from graph_retriever.metrics import (
    cosine_distance_matrix,
    epc,
    epd,
    history_distance,
    ild,
    ild_batch,
    jaccard_similarity_matrix,
    p_seen,
)


def test_p_seen_bounds():
    ps = p_seen(np.array([0.0, 1.0, 7.0]))
    assert ps[0] == 0.0
    assert ps[2] == pytest.approx(1.0)
    assert ps[1] == pytest.approx(1.0 / 3.0)  # log2(2) / log2(8)


def test_epc_hand_computed():
    pop = np.array([0.0, 1.0, 7.0])
    lists = np.array([[0, 2]])
    d1 = 1.0
    # unaware: (d1 * (1 - 0) + d2 * (1 - 1)) / 2, d2 = 1/log2(3)
    assert epc(lists, pop) == pytest.approx(d1 / 2)
    # aware, target = item 2 (most popular): only its term, which is 0
    assert epc(lists, pop, np.array([2])) == pytest.approx(0.0)
    # aware, target = item 0 at rank 1
    assert epc(lists, pop, np.array([0])) == pytest.approx(d1 / 2)
    # target outside the list
    assert epc(np.array([[1, 2]]), pop, np.array([0])) == 0.0


def test_cosine_distance_matrix():
    sessions = [[0, 1], [0, 1], [0, 2]]
    d = cosine_distance_matrix(sessions, 4)
    # item0 in sessions {0,1,2}, item1 in {0,1}: cos = 2 / sqrt(3*2)
    assert d[0, 1] == pytest.approx(1 - 2 / math.sqrt(6))
    assert d[1, 2] == pytest.approx(1.0)  # disjoint sessions
    assert d[3, 0] == 1.0 and d[3, 3] == 0.0  # item never seen
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0.0)


def test_epd_hand_computed():
    dist = np.array([[0.0, 0.2, 1.0],
                     [0.2, 0.0, 0.6],
                     [1.0, 0.6, 0.0]])
    hist = history_distance(dist, [[0, 1]])
    lists = np.array([[2, 0]])
    d1, d2 = 1.0, 1.0 / math.log2(3)
    m2 = (1.0 + 0.6) / 2  # item 2 to history {0, 1}
    m0 = (0.0 + 0.2) / 2
    assert epd(lists, hist) == pytest.approx((d1 * m2 + d2 * m0) / 2, rel=1e-6)
    assert epd(lists, hist, np.array([0])) == pytest.approx(d2 * m0 / 2, rel=1e-6)
    assert epd(lists, hist, np.array([1])) == 0.0


def test_ild_batch_matches_ild():
    cats = [frozenset("ab"), frozenset("b"), frozenset("c"), frozenset()]
    sim = jaccard_similarity_matrix(cats)
    lists = np.array([[0, 1, 2], [3, 2, 0]])
    expected = np.mean([ild(list(row), cats) for row in lists])
    assert ild_batch(lists, sim) == pytest.approx(expected)
