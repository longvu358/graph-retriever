import numpy as np

from graph_retriever.metrics import exposure_metrics, exposure_vector, gini
from graph_retriever.tradeoff import auc, interp, pareto_front


def test_gini_uniform_and_one_hot():
    assert gini(np.ones(10)) == 0.0
    for n in (2, 5, 110):
        x = np.zeros(n)
        x[3 % n] = 7.0
        assert np.isclose(gini(x), (n - 1) / n)


def test_exposure_vector_and_stats():
    lists = np.array([[0, 1], [1, 2]])
    e = exposure_vector(lists, 4)
    d2 = 1 / np.log2(3)
    assert np.allclose(e, [1.0, d2 + 1.0, d2, 0.0])
    m = exposure_metrics(lists, 4, np.array([False, True, False, False]))
    assert np.isclose(m["exp_head_share"], (d2 + 1) / e.sum())
    assert np.isclose(m["exp_top1pct"], (d2 + 1) / e.sum())  # max(1, 0.04) = 1 POI


def test_pareto_front():
    g = np.array([0.5, 0.4, 0.4, 0.3, 0.6])
    y = np.array([0.50, 0.48, 0.45, 0.40, 0.49])
    # 0.6/0.49 dominated by 0.5/0.50; 0.4/0.45 dominated by 0.4/0.48
    assert pareto_front(g, y) == [0, 1, 3]


def test_interp_and_auc_by_hand():
    g = np.array([0.5, 0.3, 0.4])
    y = np.array([0.5, 0.3, 0.4])  # y = x on the frontier
    assert np.allclose(interp(g, y, np.array([0.35, 0.45])), [0.35, 0.45])
    # mean of y = x over [0.3, 0.5] = 0.4
    assert np.isclose(auc(g, y, 0.3, 0.5), 0.4)
    # kink: (0.3, 0.2), (0.4, 0.4), (0.5, 0.4) -> integral 0.03 + 0.04 over 0.2 = 0.35
    assert np.isclose(auc(np.array([0.3, 0.4, 0.5]), np.array([0.2, 0.4, 0.4]), 0.3, 0.5), 0.35)
