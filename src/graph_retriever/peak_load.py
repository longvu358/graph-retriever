"""Can peak load be computed from TripBuilder raw? (plan D0 step 2). Report only.

trajectories.txt stores, per visit, `clus;n_photos;first_ts;last_ts` (unix seconds, from
Flickr photos). Sessions and the time split are rebuilt exactly as in dataset.prepare
(consecutive duplicates collapsed, length >= 2, sorted by start, 70/10/20); the session end
is the last_ts of its last visit (also over collapsed duplicates). Local time = UTC+1, the
same approximation as dataset.time_bucket (no DST).

Reported on test, per city: start-hour histogram (24 bins), and concurrent sessions per
(local date, hour): a session is active in every hour bin between its start and end.

Usage:
    python -m graph_retriever.peak_load --raw <TripBuilderRawData> --out results/d0_metrics/peak_load.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from graph_retriever.data import CITIES, load_city
from graph_retriever.dataset import TRAIN_FRAC, VAL_FRAC, prepare

UTC_OFFSET = 3600
LATE_TS = 1356998400 + UTC_OFFSET  # 2013-01-01, after the TripBuilder collection period


def sessions_with_end(path: Path) -> list[tuple[int, int, int]]:
    """(start_ts, end_ts, n_visits) per session, same order and filter as build_sessions."""
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            visits, start, end = [], None, None
            for point in parts[1:]:
                cid, _, first, last = point.split(";")
                if start is None:
                    start = int(first)
                end = max(end or 0, int(last))
                if visits and visits[-1] == cid:
                    continue
                visits.append(cid)
            if len(visits) >= 2:
                out.append((start, end, len(visits)))
    out.sort(key=lambda x: x[0])
    return out


def city_report(raw: Path, city: str) -> list[str]:
    sess = sessions_with_end(raw / city / "trajectories.txt")
    n = len(sess)
    test = sess[int(n * (TRAIN_FRAC + VAL_FRAC)):]
    prep = prepare(load_city(raw, city))
    n_samples = sum(k - 1 for _, _, k in test)
    start = np.array([s for s, _, _ in test]) + UTC_OFFSET
    end = np.array([e for _, e, _ in test]) + UTC_OFFSET
    dur_h = (end - start) / 3600.0
    hours = (start // 3600) % 24
    hist = np.bincount(hours, minlength=24)
    counts: dict[int, int] = {}
    for s, e in zip(start // 3600, end // 3600):
        for h in range(int(s), int(e) + 1):
            counts[h] = counts.get(h, 0) + 1
    c = np.array(list(counts.values()))
    days = len({h // 24 for h in counts})
    span_days = (start.max() - start.min()) / 86400.0
    lines = [
        f"## {city}",
        "",
        f"- Test sessions: {len(test)} (samples rebuilt {n_samples}, prepare() test samples {len(prep.test)}).",
        f"- Test start dates (UTC+1): {np.datetime64(int(start.min()), 's')} → {np.datetime64(int(start.max()), 's')} ({span_days:.0f} days); distinct active dates {days}; sessions starting after 2012-12-31: {int((start >= LATE_TS).sum())}.",
        f"- Session duration (h): median {np.median(dur_h):.2f}, mean {dur_h.mean():.2f}, p90 {np.percentile(dur_h, 90):.2f}.",
        f"- Concurrent test sessions per active (date, hour) bin: bins {c.size}, mean {c.mean():.3f}, median {np.median(c):.0f}, p99 {np.percentile(c, 99):.0f}, max {c.max()}.",
        "",
        "Start hour (local, 24 bins):",
        "",
        "| hour | " + " | ".join(str(h) for h in range(24)) + " |",
        "|---|" + "---|" * 24,
        "| sessions | " + " | ".join(str(int(x)) for x in hist) + " |",
        "",
    ]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    parts = ["# D0 mục 2: timestamp và peak load (test)", "",
             ("Nguồn: trajectories.txt, mỗi visit có first_ts và last_ts (unix giây, từ ảnh Flickr). "
              "photos.txt có datetaken (độ phân giải giây). Chỉ báo số."), ""]
    for city in CITIES:
        parts += city_report(args.raw, city)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
