"""Small hand-checked cases for the A2 significance helpers."""

import numpy as np
import pytest

from graph_retriever.significance import holm, mcnemar_exact, per_query, topk


def test_per_query_hit_and_ndcg():
    rankings = np.array([[5, 7, 9], [1, 2, 3]])
    targets = np.array([7, 4])
    hit, ndcg = per_query(rankings, targets, k=3)
    assert hit.tolist() == [1.0, 0.0]
    assert ndcg[0] == pytest.approx(1.0 / np.log2(3))
    assert ndcg[1] == 0.0


def test_topk_orders_by_score_descending():
    scores = np.array([[0.1, 0.9, 0.5], [3.0, 1.0, 2.0]])
    assert topk(scores, 2).tolist() == [[1, 2], [0, 2]]


def test_mcnemar_exact_hand_example():
    # discordant pairs: a only = 3, b only = 1 -> two-sided exact p = 10/16
    a = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    b = np.array([0, 0, 0, 1, 0, 0, 0, 0])
    assert mcnemar_exact(a, b) == pytest.approx(0.625)


def test_mcnemar_no_discordance_is_one():
    a = np.array([1, 0, 1])
    assert mcnemar_exact(a, a.copy()) == 1.0


def test_holm_hand_example():
    # sorted p: 0.01, 0.03, 0.04 -> raw adj 0.03, 0.06, 0.04 -> monotone max -> 0.03, 0.06, 0.06
    adj = holm([0.01, 0.04, 0.03])
    assert adj.tolist() == pytest.approx([0.03, 0.06, 0.06])


def test_holm_caps_at_one():
    assert holm([0.9, 0.8])[0] == pytest.approx(1.0)
