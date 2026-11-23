"""Build the three hyperedge channels (theme, area, session) and per-channel operators.

Channel operator (HGNN, identity edge weights):  P_c = Dv^-1 H De^-1 H^T
Only train data is used: session edges come from train sessions only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from graph_retriever.data import City
from graph_retriever.dataset import Prepared

EARTH_RADIUS_KM = 6371.0
AREA_DIAMETER_KM = 0.5  # complete linkage: every area has diameter <= 500 m
THEME_MAX_FRAC = 0.05  # themes covering more than 5% of nodes are too broad
CHANNELS = ("theme", "area", "session")

_CITY_SUFFIX = re.compile(r"(di|a)?(roma|firenze|pisa)$")


def normalize_theme(token: str) -> str | None:
    """Strip the city suffix (palazzidifirenze -> palazzi). Drop district tags.

    District tags such as `romar.iitrevi` are geographic. They duplicate the Area
    channel, so they are excluded from Theme to keep the +Theme / +Area ablation clean.
    """
    if "." in token:
        return None
    base = _CITY_SUFFIX.sub("", token).strip()
    return base or None


@dataclass(frozen=True)
class EdgeInfo:
    channel: str
    name: str
    members: tuple[int, ...]


@dataclass
class Hypergraph:
    channels: tuple[str, ...]
    H: dict[str, np.ndarray]  # channel -> (N, E_c) incidence, float32
    edges: dict[str, list[EdgeInfo]]
    operators: dict[str, np.ndarray]  # channel -> (N, N) propagation operator
    area: np.ndarray  # (N,) area id, -1 when the node is in no area edge
    theme_edges: dict[int, list[str]]  # node -> theme names it belongs to

    @property
    def n_items(self) -> int:
        return self.area.shape[0]


def _theme_edges(
    city: City, node_ids: list[str], max_frac: float = THEME_MAX_FRAC, min_size: int = 2,
    split_by: np.ndarray | None = None,
) -> dict[str, list[int]]:
    """Theme hyperedges. A theme covering more than max_frac of the nodes is dropped, or, when
    split_by (an (N,) area label per node) is given, cut into one edge per area (plan D9):
    'chiese' becomes 'chiese@a3', 'chiese@a7', ... each kept if it has >= min_size members."""
    members: dict[str, set[int]] = {}
    for i, cid in enumerate(node_ids):
        for raw in city.clusters[cid].categories:
            name = normalize_theme(raw)
            if name is not None:
                members.setdefault(name, set()).add(i)
    n = len(node_ids)
    kept: dict[str, list[int]] = {}
    for name, mem in members.items():
        if len(mem) < min_size:
            continue  # size-1 edges connect nothing
        if len(mem) / n > max_frac:
            if split_by is not None:  # too broad: keep it only inside each area
                parts: dict[int, list[int]] = {}
                for i in mem:
                    parts.setdefault(int(split_by[i]), []).append(i)
                for lab, sub in parts.items():
                    if len(sub) >= min_size:
                        kept[f"{name}@a{lab}"] = sorted(sub)
            continue  # too broad to be informative as one edge
        kept[name] = sorted(mem)
    return kept


def _haversine_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    phi = np.radians(lat)
    lam = np.radians(lon)
    dphi = phi[:, None] - phi[None, :]
    dlam = lam[:, None] - lam[None, :]
    a = np.sin(dphi / 2) ** 2 + np.cos(phi)[:, None] * np.cos(phi)[None, :] * np.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _area_labels(
    city: City, node_ids: list[str], diameter_km: float = AREA_DIAMETER_KM,
) -> np.ndarray:
    lat = np.array([city.clusters[c].lat for c in node_ids])
    lon = np.array([city.clusters[c].lon for c in node_ids])
    dist = _haversine_matrix(lat, lon)
    model = AgglomerativeClustering(
        metric="precomputed",
        linkage="complete",
        distance_threshold=diameter_km,
        n_clusters=None,
    )
    return model.fit_predict(dist)


def clique_operator(H_list: list[np.ndarray]) -> np.ndarray:
    """Second view (DHCN-style line/clique graph): item adjacency from co-membership.

    A = H Hᵀ over all given incidence matrices, zero diagonal, then self-loop and
    row-normalise: P = D⁻¹(A + I). Built only from theme/area membership, so it carries
    no target information.
    """
    n = H_list[0].shape[0]
    A = np.zeros((n, n), dtype=np.float32)
    for H in H_list:
        A += H @ H.T
    np.fill_diagonal(A, 0.0)
    A += np.eye(n, dtype=np.float32)
    return (A / A.sum(axis=1, keepdims=True)).astype(np.float32)


def _incidence(n: int, edges: list[EdgeInfo]) -> np.ndarray:
    H = np.zeros((n, len(edges)), dtype=np.float32)
    for j, e in enumerate(edges):
        H[list(e.members), j] = 1.0
    return H


def _operator(H: np.ndarray) -> np.ndarray:
    """P = Dv^-1 H De^-1 H^T with W = I. Isolated nodes/edges contribute zero."""
    dv = H.sum(axis=1)
    de = H.sum(axis=0)
    dv_inv = np.divide(1.0, dv, out=np.zeros_like(dv), where=dv > 0)
    de_inv = np.divide(1.0, de, out=np.zeros_like(de), where=de > 0)
    return (dv_inv[:, None] * (H @ (de_inv[:, None] * H.T))).astype(np.float32)


def build_hypergraph(
    city: City,
    prep: Prepared,
    channels: tuple[str, ...] = CHANNELS,
    theme_max_frac: float = THEME_MAX_FRAC,
    area_diameter_km: float = AREA_DIAMETER_KM,
    min_edge_size: int = 2,
    theme_split_area: bool = False,
) -> Hypergraph:
    """min_edge_size applies to theme and area edges (prune ablation); defaults are v2.
    theme_split_area (plan D9): broad themes are split by area instead of dropped."""
    node_ids = prep.node_ids
    n = len(node_ids)
    edges: dict[str, list[EdgeInfo]] = {}
    H: dict[str, np.ndarray] = {}
    operators: dict[str, np.ndarray] = {}

    labels = _area_labels(city, node_ids, area_diameter_km)
    theme = _theme_edges(city, node_ids, theme_max_frac, min_edge_size,
                         split_by=labels if theme_split_area else None)
    theme_edges: dict[int, list[str]] = {i: [] for i in range(n)}
    for name, mem in theme.items():
        for i in mem:
            theme_edges[i].append(name)

    area = np.full(n, -1, dtype=np.int64)
    area_edges: list[EdgeInfo] = []
    for lab in np.unique(labels):
        mem = np.flatnonzero(labels == lab)
        if len(mem) < min_edge_size:
            continue
        area[mem] = len(area_edges)
        area_edges.append(EdgeInfo("area", f"area_{len(area_edges)}", tuple(mem.tolist())))

    session_edges: list[EdgeInfo] = []
    seen_sets: set[tuple[int, ...]] = set()
    for s in prep.train_sessions:
        mem = tuple(sorted(set(s)))
        if len(mem) < 2 or mem in seen_sets:
            continue
        seen_sets.add(mem)
        session_edges.append(EdgeInfo("session", f"session_{len(session_edges)}", mem))

    theme_list = [
        EdgeInfo("theme", name, tuple(mem)) for name, mem in sorted(theme.items())
    ]

    all_edges = {"theme": theme_list, "area": area_edges, "session": session_edges}
    for ch in channels:
        edges[ch] = all_edges[ch]
        H[ch] = _incidence(n, edges[ch])
        operators[ch] = _operator(H[ch])

    return Hypergraph(
        channels=tuple(channels),
        H=H,
        edges=edges,
        operators=operators,
        area=area,
        theme_edges=theme_edges,
    )


def graph_stats(hg: Hypergraph) -> dict:
    out: dict = {}
    for ch in hg.channels:
        sizes = hg.H[ch].sum(axis=0)
        out[ch] = {
            "edges": int(hg.H[ch].shape[1]),
            "mean_size": round(float(sizes.mean()), 2) if sizes.size else 0.0,
            "max_size": int(sizes.max()) if sizes.size else 0,
            "nodes_covered": int((hg.H[ch].sum(axis=1) > 0).sum()),
        }
    return out


def flat_category(city: City, node_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """Flat POI type (plan D1 control). TripBuilder raw has no coarse type field, so each
    node gets its broadest Wikipedia category: the normalised category token shared by the
    most nodes, including tokens that THEME_MAX_FRAC drops from Theme. District tags are
    excluded as in Theme (they duplicate Area). Ties break by name. Nodes without any
    category get -1 (they match nothing).
    Returns (N,) category id and the id -> name list.
    """
    tokens = [
        sorted({t for t in (normalize_theme(r) for r in city.clusters[c].categories) if t})
        for c in node_ids
    ]
    size: dict[str, int] = {}
    for ts in tokens:
        for t in ts:
            size[t] = size.get(t, 0) + 1
    chosen = [min(ts, key=lambda t: (-size[t], t)) if ts else None for ts in tokens]
    names = sorted({t for t in chosen if t is not None})
    ids = {t: i for i, t in enumerate(names)}
    cat = np.array([ids[t] if t is not None else -1 for t in chosen], dtype=np.int64)
    return cat, names
