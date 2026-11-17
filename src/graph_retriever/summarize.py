"""Aggregate per-seed CSVs into mean ± std markdown tables.

Usage: uv run python -m graph_retriever.summarize results
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ("recall", "ndcg", "ILD", "coverage", "gini", "area_entropy", "long_tail_ratio", "ARP")


def _f(v: str) -> float | None:
    return float(v) if v not in ("", None) else None


def summarize(paths: list[Path], K: str, title: str) -> str:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in paths:
        if not path.exists():
            continue
        with path.open() as f:
            for r in csv.DictReader(f):
                if r["K"] != K:
                    continue
                groups[(r["method"], r["lam"] or "-")].append(r)

    lines = [f"### {title}, K={K}", "",
             "| method | λ/θ | " + " | ".join(METRICS) + " | n |",
             "|---|---|" + "---|" * len(METRICS) + "---|"]
    methods: list[str] = []
    for m, _ in groups:
        if m not in methods:
            methods.append(m)
    for method in methods:
        for (m, lam), rs in sorted(
            [(k, v) for k, v in groups.items() if k[0] == method], key=lambda x: x[0][1]
        ):
            cells = []
            for metric in METRICS:
                vals = [_f(r[metric]) for r in rs if _f(r.get(metric)) is not None]
                if not vals:
                    cells.append("-")
                    continue
                a = np.array(vals)
                cells.append(f"{a.mean():.3f} ± {a.std():.3f}")
            lines.append(f"| {m} | {lam} | " + " | ".join(cells) + f" | {len(rs)} |")
    lines.append("")
    return "\n".join(lines)


def main(out_dir: Path) -> None:
    parts = []
    for city in ("pisa", "florence", "rome"):
        parts.append(summarize([out_dir / f"{city}.csv", out_dir / f"{city}_baselines.csv"], "20", city))
    (out_dir / "summary_K20.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {out_dir / 'summary_K20.md'}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "results"))
