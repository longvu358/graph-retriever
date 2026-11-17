"""Leakage and protocol invariants from plan v0 (checked on a small synthetic city)."""

import numpy as np

from graph_retriever.data import City, Cluster, Trajectory
from graph_retriever.dataset import build_sessions, prepare
from graph_retriever.hypergraph import build_hypergraph, normalize_theme


def _city() -> City:
    clusters = {
        str(i): Cluster(
            cid=str(i),
            lat=43.77 + 0.001 * (i % 5),
            lon=11.25 + 0.001 * (i // 5),
            categories=("chiesedifirenze",) if i % 2 == 0 else ("palazzidifirenze",),
        )
        for i in range(20)
    }
    # 10 trajectories, start times increasing; visits only use clusters 0..14
    trajs = []
    for t in range(10):
        visits = [str((t + k) % 15) for k in range(4)]
        trajs.append(Trajectory(user=str(t % 3), visits=visits, start_ts=1000 + 100 * t))
    # cluster 19 is never visited: must remain a node
    return City(name="synthetic", clusters=clusters, trajectories=trajs)


def test_every_cluster_is_a_node():
    city = _city()
    prep = prepare(city)
    assert sorted(prep.node_ids, key=int) == sorted(city.clusters, key=int)
    assert prep.n_items == len(city.clusters)


def test_train_precedes_val_and_test_in_time():
    city = _city()
    sessions = build_sessions(city.trajectories)
    n_train = int(len(sessions) * 0.7)
    train_max = max(ts for ts, _, _ in sessions[:n_train])
    later_min = min(ts for ts, _, _ in sessions[n_train:])
    assert train_max <= later_min


def test_pop_uses_train_only():
    city = _city()
    prep = prepare(city)
    recount = np.zeros(prep.n_items)
    for s in prep.train_sessions:
        for v in s:
            recount[v] += 1
    assert np.array_equal(recount, prep.pop_count)
    # pop is normalized to [0, 1]
    assert prep.pop.min() == 0.0 and prep.pop.max() == 1.0


def test_session_hyperedges_come_from_train_only():
    city = _city()
    prep = prepare(city)
    hg = build_hypergraph(city, prep)
    train_sets = {tuple(sorted(set(s))) for s in prep.train_sessions if len(set(s)) >= 2}
    edge_sets = set()
    for e in hg.edges["session"]:
        edge_sets.add(e.members)
    assert edge_sets <= train_sets
    val_only = {
        tuple(sorted(set(s.prefix + [s.target])))
        for s in prep.val + prep.test
    }
    assert not (edge_sets & (val_only - train_sets))


def test_long_tail_is_bottom_fraction_by_train_count():
    city = _city()
    prep = prepare(city)
    assert prep.tail.sum() == round(0.8 * prep.n_items)
    if prep.tail.any() and (~prep.tail).any():
        assert prep.pop_count[prep.tail].max() <= prep.pop_count[~prep.tail].min()


def test_normalize_theme_strips_city_and_drops_district_tags():
    assert normalize_theme("palazzidifirenze") == "palazzi"
    assert normalize_theme("scultureafirenze") == "sculture"
    assert normalize_theme("romar.iitrevi") is None
    assert normalize_theme("chiesediroma") == "chiese"
