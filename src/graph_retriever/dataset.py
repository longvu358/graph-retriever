"""Sessions, time split, prefix augmentation and train-only popularity.

Protocol (identical for every method):
- session = trajectory with consecutive duplicate clusters collapsed; length >= 2.
- split by the first timestamp of each trajectory: 70 / 10 / 20 (train/val/test).
- every prefix of a session becomes one sample (prefix -> next cluster), all splits.
- node set = every cluster in the clusters file, never filtered by train coverage.
- pop(v) is computed from train sessions only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from graph_retriever.data import City, Trajectory

TRAIN_FRAC = 0.7
VAL_FRAC = 0.1
LONG_TAIL_FRAC = 0.8  # bottom 80% of nodes by train popularity count as long tail


@dataclass
class Sample:
    prefix: list[int]
    target: int
    ctx: int = 0  # time bucket of the LAST prefix visit (never the target's time)


@dataclass
class Prepared:
    city: str
    node_ids: list[str]
    train_sessions: list[list[int]]
    train: list[Sample]
    val: list[Sample]
    test: list[Sample]
    pop_count: np.ndarray  # (N,) train visit counts
    pop: np.ndarray  # (N,) log1p count, min-max to [0, 1], train only
    tail: np.ndarray  # (N,) bool, long-tail nodes

    @property
    def n_items(self) -> int:
        return len(self.node_ids)


def collapse_consecutive(visits: list[str]) -> list[str]:
    out: list[str] = []
    for v in visits:
        if not out or out[-1] != v:
            out.append(v)
    return out


def time_bucket(ts: int) -> int:
    """12 buckets: 4-hour block of the (approximate local, UTC+1) day x weekday/weekend."""
    local = ts + 3600
    hour_block = (local // 3600) % 24 // 4
    weekday = ((local // 86400) + 3) % 7  # 1970-01-01 was a Thursday; Monday = 0
    return hour_block + (6 if weekday >= 5 else 0)


def build_sessions(trajectories: list[Trajectory]) -> list[tuple[int, list[str], list[int]]]:
    """Return (start_ts, visits, times) sorted by start time; consecutive duplicates collapsed."""
    sessions = []
    for t in trajectories:
        visits: list[str] = []
        times: list[int] = []
        for v, ts in zip(t.visits, t.times or [0] * len(t.visits)):
            if visits and visits[-1] == v:
                continue
            visits.append(v)
            times.append(ts)
        if len(visits) >= 2:
            sessions.append((t.start_ts, visits, times))
    sessions.sort(key=lambda x: x[0])
    return sessions


def augment(session: list[int], times: list[int]) -> list[Sample]:
    return [
        Sample(prefix=session[:i], target=session[i], ctx=time_bucket(times[i - 1]))
        for i in range(1, len(session))
    ]


def prepare(city: City) -> Prepared:
    node_ids = sorted(city.clusters, key=int)
    index = {cid: i for i, cid in enumerate(node_ids)}
    n = len(node_ids)

    sessions = build_sessions(city.trajectories)
    n_total = len(sessions)
    n_train = int(n_total * TRAIN_FRAC)
    n_val = int(n_total * (TRAIN_FRAC + VAL_FRAC))

    def encode(s: list[str]) -> list[int]:
        return [index[c] for c in s]

    encoded = [(encode(v), t) for _, v, t in sessions]
    train_sessions = [s for s, _ in encoded[:n_train]]
    val_sessions = encoded[n_train:n_val]
    test_sessions = encoded[n_val:]

    pop_count = np.zeros(n, dtype=np.float64)
    for s in train_sessions:
        for v in s:
            pop_count[v] += 1

    logc = np.log1p(pop_count)
    span = logc.max() - logc.min()
    pop = (logc - logc.min()) / span if span > 0 else np.zeros(n)

    n_tail = round(LONG_TAIL_FRAC * n)
    order = np.lexsort((np.arange(n), pop_count))  # ascending count, stable on id
    tail = np.zeros(n, dtype=bool)
    tail[order[:n_tail]] = True

    return Prepared(
        city=city.name,
        node_ids=node_ids,
        train_sessions=train_sessions,
        train=[x for s, t in encoded[:n_train] for x in augment(s, t)],
        val=[x for s, t in val_sessions for x in augment(s, t)],
        test=[x for s, t in test_sessions for x in augment(s, t)],
        pop_count=pop_count,
        pop=pop,
        tail=tail,
    )
