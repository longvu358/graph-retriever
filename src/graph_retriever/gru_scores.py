"""GRU (no graph) test scores for A2 and A4: the `v2_none_gru` control, exported like pool.py.

Writes `{out}/gru_{city}_seed{s}.npy` (float32, (n_test, N), prep.test order, same as the
baseline score files) and a `.json` with the run info. Existing files are never overwritten.

Usage (on the server, inside tmux, after choosing a free GPU):
    CUDA_VISIBLE_DEVICES=<id> python -m graph_retriever.gru_scores --raw <TripBuilderRawData> \
        --out results/scores --cities pisa florence rome --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from graph_retriever.data import CITIES, load_city
from graph_retriever.dataset import prepare
from graph_retriever.hypergraph import build_hypergraph
from graph_retriever.train import TrainConfig, score_matrix, train_model

CHANNELS: tuple[str, ...] = ()  # v2_none_gru: plain GRU embeddings, no hyperedge channel


def export(city: str, raw_root: Path, seed: int, out: Path, tcfg: TrainConfig) -> str:
    stem = f"gru_{city}_seed{seed}"
    if (out / f"{stem}.npy").exists():
        return f"[{city}] seed={seed}: {stem}.npy exists, skipped"
    data = load_city(raw_root, city)
    prep = prepare(data)
    hg = build_hypergraph(data, prep, CHANNELS)
    model, info = train_model(prep, hg, tcfg, seed, learn_gate=True)
    rel = score_matrix(model, prep.test)
    np.save(out / f"{stem}.npy", rel.astype(np.float32))
    meta = {"city": city, "seed": seed, "config": "v2_none_gru", "channels": list(CHANNELS),
            "best_epoch": info["best_epoch"], "val_recall": info["best_val_recall"],
            "n_test": int(rel.shape[0])}
    (out / f"{stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return f"[{city}] seed={seed}: best_epoch={info['best_epoch']} val_R@20={info['best_val_recall']:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/scores"))
    ap.add_argument("--cities", nargs="+", default=list(CITIES), choices=CITIES)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--max-epochs", type=int, default=50)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tcfg = TrainConfig(encoder="gru", residual=True, max_epochs=args.max_epochs)
    for city in args.cities:
        for seed in args.seeds:
            print(export(city, args.raw, seed, args.out, tcfg), flush=True)


if __name__ == "__main__":
    main()
