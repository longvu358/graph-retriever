"""Rank helpers and bin shares used by the A4 diagnosis."""

import numpy as np

from graph_retriever.diagnose import POOL_MAX, bin_counts, rank_in_pool, rank_in_scores


def test_rank_in_scores_counts_strictly_higher():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    assert rank_in_scores(scores, 1) == 1
    assert rank_in_scores(scores, 2) == 3
    assert rank_in_scores(scores, 0) == 4


def test_rank_in_pool_position_or_cap():
    items = np.array([9, 4, 7])
    assert rank_in_pool(items, 4) == 2
    assert rank_in_pool(items, 1) == POOL_MAX + 1


def test_bin_counts_shares_sum_to_one():
    ranks = np.array([1, 3, 7, 15, 30, 80, 101, 101])
    shares = bin_counts(ranks)
    assert shares.sum() == 1.0
    assert shares[-1] == 2 / 8  # two queries capped at 101
    assert shares[0] == 2 / 8  # ranks 1 and 3 fall in 1-5
