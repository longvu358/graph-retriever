import numpy as np

from graph_retriever.dispersal import pop_threshold, rank_swap, swap_sequence


def _toy():
    # 6 items, pool = all; rel strictly decreasing with id
    rel = np.array([0.0, -0.5, -1.0, -1.2, -1.5, -3.0])
    pop = np.array([1.0, 0.2, 0.9, 0.1, 0.3, 0.05])
    sub = np.zeros((6, 6), dtype=bool)
    sub[0, [3, 5]] = sub[[3, 5], 0] = True
    sub[2, [4, 5]] = sub[[4, 5], 2] = True
    return rel, pop, sub


def test_budget_zero_and_delta_zero_are_plain():
    rel, pop, sub = _toy()
    pool = np.arange(6)
    assert rank_swap(rel, 3, pop, pool, sub, tau=0.5, delta=10, budget=0) == ([0, 1, 2], 0)
    assert rank_swap(rel, 3, pop, pool, sub, tau=0.5, delta=0.0, budget=5) == ([0, 1, 2], 0)


def test_hand_example():
    rel, pop, sub = _toy()
    pool = np.arange(6)
    # heads with pop >= 0.5 in [0, 1, 2]: 0 (1.0), then 2 (0.9).
    # u=0: candidates 3 (rel -1.2), 5 (-3.0) -> 3. u=2: candidates 4 (-1.5), 5 -> 4.
    assert rank_swap(rel, 3, pop, pool, sub, tau=0.5, delta=10, budget=5) == ([3, 1, 4], 2)
    # delta = 1: u=0 needs rel >= -1 -> none; u=2 needs rel >= -2 -> 4
    assert rank_swap(rel, 3, pop, pool, sub, tau=0.5, delta=1.0, budget=5) == ([0, 1, 4], 1)
    # budget 1 stops after the most popular head
    assert rank_swap(rel, 3, pop, pool, sub, tau=0.5, delta=10, budget=1) == ([3, 1, 2], 1)
    # session bonus flips the choice for u=0 towards item 5
    edges = [frozenset()] * 5 + [frozenset({"theme:x"})]
    lst, n = rank_swap(rel, 3, pop, pool, sub, tau=0.95, delta=10, budget=5,
                       select="rel+session", w=2.0, prefix_edges={"theme:x"}, item_edges=edges)
    assert (lst, n) == ([5, 1, 2], 1)


def test_pool_restriction_and_no_duplicates():
    rng = np.random.default_rng(0)
    n = 40
    for _ in range(50):
        rel = rng.normal(size=n)
        pop = rng.random(n)
        sub = rng.random((n, n)) < 0.3
        pool = rng.choice(n, 25, replace=False)
        for select in ("rel", "random"):
            lst, _n_sw = rank_swap(rel, 10, pop, pool, sub, tau=0.3, delta=np.inf, budget=10,
                                  select=select, rng=np.random.default_rng(1))
            assert len(set(lst)) == 10
            assert set(lst) <= set(pool.tolist())


def test_lower_tau_and_budget_are_prefixes():
    rng = np.random.default_rng(3)
    n = 60
    for _ in range(30):
        rel = np.sort(rng.normal(size=n))[::-1]
        pop = rng.random(n)
        sub = rng.random((n, n)) < 0.3
        full = swap_sequence(rel, pop, sub, 20, pop_threshold(pop, 0.2), 1.0, 20)
        for frac in (0.05, 0.1, 0.2):
            thr = pop_threshold(pop, frac)
            for b in (0, 1, 2, 5, 10, 20):
                part = swap_sequence(rel, pop, sub, 20, thr, 1.0, b)
                trunc = [s for s in full if s[2] >= thr][:b]
                assert part == trunc


def test_pop_threshold():
    pop = np.arange(100) / 99.0
    assert (pop >= pop_threshold(pop, 0.05)).sum() == 5
    assert (pop >= pop_threshold(pop, 0.2)).sum() == 20
