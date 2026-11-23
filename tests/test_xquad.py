import numpy as np

from graph_retriever.dispersal import (
    aspect_dist_prefix,
    aspect_dist_uniform,
    rank_xquad,
)


def test_xquad_hand_example():
    # Four pool items, equal relevance -> P(c|q) = 1/4. Two aspects A, B, each with 2 items.
    # P(A) = P(B) = 1/2, P(c|a) = 1/2 for members.
    rel = np.zeros(4)
    aspects = [frozenset({"A"}), frozenset({"A"}), frozenset({"B"}), frozenset({"B"})]
    p_aspect = {"A": 0.5, "B": 0.5}
    # Step 1: every item scores 0.5 * 1/4 + 0.5 * (1/2 * 1/2) = 0.25; ties keep index 0.
    # Step 2 (S = {0}): item 1 (aspect A) gets 0.125 + 0.5 * 0.5 * 0.5 * 0.5 = 0.1875,
    # item 2 (aspect B, no overlap with S) gets 0.125 + 0.5 * 0.5 * 0.5 = 0.25 -> chosen.
    assert rank_xquad(rel, 2, aspects, p_aspect, lam=0.5) == [0, 2]


def test_xquad_lambda_zero_is_relevance_order():
    rel = np.log(np.array([0.1, 0.4, 0.3, 0.2]))
    aspects = [frozenset({"A"})] * 4
    p_aspect = {"A": 1.0}
    assert rank_xquad(rel, 3, aspects, p_aspect, lam=0.0) == [1, 2, 3]


def test_xquad_ignores_outside_pool():
    rel = np.array([-1e9, 0.0, -1e9, -0.5])
    aspects = [frozenset({"A"}), frozenset({"A"}), frozenset({"B"}), frozenset({"B"})]
    out = rank_xquad(rel, 5, aspects, {"A": 0.5, "B": 0.5}, lam=0.5)
    assert sorted(out) == [1, 3]


def test_aspect_distributions():
    aspects = [frozenset({"A"}), frozenset({"A"}), frozenset({"B"}), frozenset()]
    pool = np.array([True, True, True, False])
    assert aspect_dist_uniform(aspects, pool) == {"A": 0.5, "B": 0.5}
    # prefix = [0]: counts A = 1 + 1 = 2, B = 1 + 0 = 1, total 3
    p = aspect_dist_prefix(aspects, pool, [0])
    assert np.isclose(p["A"], 2 / 3) and np.isclose(p["B"], 1 / 3)
