"""Build the report table: our retriever vs baselines, K=20, test split, mean ± std over seeds.

Reads the CSVs written by experiment.py (main, --baselines, --v2) from a results directory.

Usage:
    uv run python -m graph_retriever.compare results results/report_comparison.md
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

CITIES = ("pisa", "florence", "rome")
# (label, file stem candidates, method name in CSV)
ROWS = (
    ("Popularity", ("{c}.csv",), "popularity"),
    ("Markov", ("{c}.csv",), "markov"),
    ("SR-GNN", ("{c}_baselines.csv",), "srgnn"),
    ("GCE-GNN", ("{c}_baselines.csv",), "gcegnn"),
    ("DHCN", ("{c}_baselines.csv",), "dhcn"),
    ("GRU, không đồ thị (đối chứng)", ("v2/{c}.csv", "v2/{c}_only.csv"), "v2_none_gru"),
    ("HGNN v1 (có kênh session)", ("{c}.csv",), "full"),
    ("**HGNN v2: GRU + theme + area (của ta)**", ("v2/{c}.csv", "v2/{c}_only.csv"), "v2_theme_area_gru"),
)
METRICS = ("recall", "ndcg", "ILD", "coverage", "long_tail_ratio")
HEADER = ("Method", "Recall@20", "nDCG@20", "ILD", "Coverage", "Long-tail ratio", "n")


def _load(results: Path, city: str, stems: tuple[str, ...], method: str) -> list[dict]:
    rows: list[dict] = []
    for stem in stems:
        path = results / stem.format(c=city)
        if not path.exists():
            continue
        with path.open() as f:
            rows.extend(r for r in csv.DictReader(f) if r["method"] == method and r["K"] == "20")
    return rows


def _cell(rows: list[dict], metric: str) -> str:
    vals = [float(r[metric]) for r in rows if r.get(metric) not in ("", None)]
    if not vals:
        return "-"
    a = np.array(vals)
    if len(a) == 1:
        return f"{a[0]:.3f}"
    return f"{a.mean():.3f} ± {a.std():.3f}"


def build(results: Path) -> str:
    parts = []
    for city in CITIES:
        lines = [f"### {city.capitalize()} (K=20, test)", "",
                 "| " + " | ".join(HEADER) + " |", "|" + "---|" * len(HEADER)]
        for label, stems, method in ROWS:
            rows = _load(results, city, stems, method)
            if not rows:
                lines.append(f"| {label} | " + " | ".join(["-"] * (len(HEADER) - 2)) + " | 0 |")
                continue
            cells = [_cell(rows, m) for m in METRICS]
            lines.append(f"| {label} | " + " | ".join(cells) + f" | {len(rows)} |")
        parts.append("\n".join(lines) + "\n")
    return "\n".join(parts)


def main(results: Path, out: Path) -> None:
    text = build(results)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
