"""Load TripBuilder raw files (Rome, Florence, Pisa) into plain Python objects.

Unit of recommendation is a POI *cluster* (column `clus`), because trajectories
reference clusters, not raw POIs. Every cluster listed in `pois-clusters.txt` is a
node, including clusters no trajectory visits.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

CITIES = ("pisa", "florence", "rome")


@dataclass(frozen=True)
class Cluster:
    cid: str
    lat: float
    lon: float
    categories: tuple[str, ...]
    name: str = ""  # first POI name in the cluster (for human/LLM-readable input)


@dataclass
class Trajectory:
    user: str
    visits: list[str]
    start_ts: int
    times: list[int] = field(default_factory=list)  # first photo timestamp of each visit


@dataclass
class City:
    name: str
    clusters: dict[str, Cluster]
    trajectories: list[Trajectory] = field(default_factory=list)


def load_clusters(path: Path) -> dict[str, Cluster]:
    clusters: dict[str, Cluster] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row["clus"].strip()
            cats = tuple(c.strip() for c in row["categories"].split(";") if c.strip())
            pois = [p.strip() for p in row["pois"].split(";") if p.strip()]
            clusters[cid] = Cluster(
                cid=cid,
                lat=float(row["latitude"]),
                lon=float(row["longitude"]),
                categories=cats,
                name=pois[0].replace("_", " ") if pois else "",
            )
    return clusters


def load_trajectories(path: Path) -> list[Trajectory]:
    """Each line: user \\t point \\t point ...; point = clus;n_photos;first_ts;last_ts."""
    trajs: list[Trajectory] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            user = parts[0]
            visits: list[str] = []
            times: list[int] = []
            for point in parts[1:]:
                cid, _, first, _ = point.split(";")
                visits.append(cid)
                times.append(int(first))
            trajs.append(
                Trajectory(user=user, visits=visits, start_ts=times[0] if times else 0, times=times)
            )
    return trajs


def load_city(raw_root: Path, city: str) -> City:
    """`raw_root` is the TripBuilderRawData directory."""
    base = raw_root / city
    return City(
        name=city,
        clusters=load_clusters(base / "pois-clusters.txt"),
        trajectories=load_trajectories(base / "trajectories.txt"),
    )
