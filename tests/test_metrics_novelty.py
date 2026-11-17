"""Hand-computed examples for novelty and alpha-nDCG (plan A3)."""

import math

import numpy as np
import pytest

from graph_retriever.metrics import alpha_ndcg, novelty_pop, unseen_fraction


def test_novelty_pop_hand_example():
    # N_train = 4, |I| = 4 -> p = (c + 1) / 8 = [4/8, 2/8, 1/8, 1/8] -> -log2 p = [1, 2, 3, 3]
    pop_count = np.array([3.0, 1.0, 0.0, 0.0])
    assert novelty_pop([[0, 1]], pop_count) == pytest.approx(1.5)
    assert novelty_pop([[2, 3]], pop_count) == pytest.approx(3.0)
    # mean over lists of per-list mean: (1.5 + 3.0) / 2
    assert novelty_pop([[0, 1], [2, 3]], pop_count) == pytest.approx(2.25)


def test_novelty_pop_popular_lists_are_less_novel():
    pop_count = np.array([10.0, 5.0, 1.0, 0.0])
    head = novelty_pop([[0, 0, 1]], pop_count)
    tail = novelty_pop([[2, 3, 3]], pop_count)
    assert head < tail


def test_unseen_fraction_hand_example():
    pop_count = np.array([3.0, 1.0, 0.0, 0.0])
    # items 2 and 3 are unseen: 2 of 4 entries
    assert unseen_fraction([[0, 2], [3, 1]], pop_count) == pytest.approx(0.5)


def test_alpha_ndcg_single_target_at_rank_two():
    # target 7 has 2 aspects, sits at rank 2. gain 2 / log2(3); ideal = gain 2 / log2(2).
    relevant = {7: frozenset({"a", "b"})}
    got = alpha_ndcg([5, 7, 9], relevant, k=3)
    assert got == pytest.approx(1.0 / math.log2(3))


def test_alpha_ndcg_target_missing_is_zero():
    relevant = {7: frozenset({"a"})}
    assert alpha_ndcg([5, 9, 8], relevant, k=3) == 0.0


def test_alpha_ndcg_no_relevant_is_zero():
    assert alpha_ndcg([1, 2, 3], {}, k=3) == 0.0


def test_alpha_ndcg_redundancy_is_discounted():
    # t1 covers {a}, t2 covers {a, b}. Ranking [t1, t2]:
    # DCG = 1/log2(2) + (1 + 0.5) / log2(3)  (t2's aspect a was seen once -> factor 0.5)
    # Ideal (greedy) = [t2, t1]: DCG = 2/log2(2) + 0.5/log2(3)
    relevant = {1: frozenset({"a"}), 2: frozenset({"a", "b"})}
    x = 1.0 / math.log2(3)
    expected = (1.0 + 1.5 * x) / (2.0 + 0.5 * x)
    assert alpha_ndcg([1, 2], relevant, k=2) == pytest.approx(expected, abs=1e-9)
