"""Run third-party session-based retrievers on our split and dump test scores.

Each model runs in its own process (the repos share module names such as `model`
and `utils`). Training and early stopping use OUR protocol: train split for fitting,
val split for choosing the epoch. Test scores are written as float32 (n_test, N) so
that accuracy and dispersal metrics are computed by the same code for every method.

Usage:
    python -m graph_retriever.baselines --model srgnn --city florence --seed 0 \
        --raw <TripBuilderRawData> --out results/scores --max-epochs 50 --patience 5
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from graph_retriever.data import CITIES, load_city
from graph_retriever.dataset import Prepared, Sample, prepare

THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party"


def _to_repo_data(samples: list[Sample]) -> tuple[list[list[int]], list[int]]:
    """Our node i -> repo item id i+1 (repos reserve 0 for padding)."""
    return [[v + 1 for v in s.prefix] for s in samples], [s.target + 1 for s in samples]


def _ragged_repo_data(samples: list[Sample]) -> tuple[np.ndarray, list[int]]:
    """DHCN's Data calls np.asarray on ragged sessions, which needs an object array in numpy 2."""
    inputs, targets = _to_repo_data(samples)
    arr = np.empty(len(inputs), dtype=object)
    for i, seq in enumerate(inputs):
        arr[i] = seq
    return arr, targets


def _recall_at(scores: np.ndarray, targets: np.ndarray, k: int = 20) -> float:
    topk = np.argpartition(-scores, k, axis=1)[:, :k]
    return float((topk == targets[:, None]).any(axis=1).mean())


def run_srgnn(prep: Prepared, seed: int, max_epochs: int, patience: int) -> tuple[np.ndarray, dict]:
    repo = THIRD_PARTY / "SR-GNN" / "pytorch_code"
    sys.path.insert(0, str(repo))
    import torch
    from model import SessionGraph, forward, trans_to_cpu
    from utils import Data

    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = SimpleNamespace(
        hiddenSize=100, step=1, lr=1e-3, lr_dc=0.1, lr_dc_step=3, l2=1e-5,
        batchSize=100, nonhybrid=False,
    )
    n_node = prep.n_items + 1
    model = SessionGraph(opt, n_node)
    if torch.cuda.is_available():
        model = model.cuda()

    train = Data(_to_repo_data(prep.train), shuffle=True)
    val = Data(_to_repo_data(prep.val), shuffle=False)
    test = Data(_to_repo_data(prep.test), shuffle=False)
    val_targets = np.array([s.target for s in prep.val])

    def predict(data: Data) -> np.ndarray:
        model.eval()
        out = []
        with torch.no_grad():
            for i in data.generate_batch(model.batch_size):
                _, scores = forward(model, i, data)
                out.append(trans_to_cpu(scores).numpy())
        return np.concatenate(out, axis=0)

    best_state, best_val, best_epoch, bad = copy.deepcopy(model.state_dict()), -1.0, -1, 0
    history = []
    for epoch in range(max_epochs):
        model.train()
        model.scheduler.step()
        for i in train.generate_batch(model.batch_size):
            model.optimizer.zero_grad()
            targets, scores = forward(model, i, train)
            loss = model.loss_function(scores, torch.as_tensor(targets, dtype=torch.long, device=scores.device) - 1)
            loss.backward()
            model.optimizer.step()
        val_recall = _recall_at(predict(val), val_targets, 20)
        history.append({"epoch": epoch, "val_recall": val_recall})
        if val_recall > best_val:
            best_val, best_epoch, bad = val_recall, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    scores = predict(test)
    return scores, {"best_epoch": best_epoch, "best_val_recall": best_val, "history": history,
                    "_val_scores": predict(val)}


def _global_graph(sessions: list[list[int]], n_node: int, sample_num: int) -> tuple[list, list]:
    """Mirror GCE-GNN build_graph.py: co-occurrence within window 1..3 on train sessions."""
    counts: list[dict[int, int]] = [{} for _ in range(n_node)]
    for seq in sessions:
        for k in range(1, 4):
            for j in range(len(seq) - k):
                for a, b in ((seq[j], seq[j + k]), (seq[j + k], seq[j])):
                    counts[a][b] = counts[a].get(b, 0) + 1
    adj, num = [[] for _ in range(n_node)], [[] for _ in range(n_node)]
    for t in range(n_node):
        ranked = sorted(counts[t].items(), key=lambda x: x[1], reverse=True)[:sample_num]
        adj[t] = [v for v, _ in ranked]
        num[t] = [c for _, c in ranked]
    return adj, num


def run_gcegnn(prep: Prepared, seed: int, max_epochs: int, patience: int) -> tuple[np.ndarray, dict]:
    repo = THIRD_PARTY / "GCE-GNN"
    sys.path.insert(0, str(repo))
    import torch
    from model import CombineGraph, forward
    from torch.utils.data import DataLoader
    from utils import Data, handle_adj

    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = SimpleNamespace(
        hiddenSize=100, activate="relu", n_iter=1, n_sample_all=12, n_sample=12,
        batch_size=100, lr=1e-3, lr_dc=0.1, lr_dc_step=3, l2=1e-5,
        dropout_gcn=0.0, dropout_local=0.0, dropout_global=0.5, alpha=0.2,
    )
    n_node = prep.n_items + 1
    sessions = [[v + 1 for v in s] for s in prep.train_sessions]
    adj, num = _global_graph(sessions, n_node, opt.n_sample_all)
    adj, num = handle_adj(adj, n_node, opt.n_sample_all, num)
    model = CombineGraph(opt, n_node, adj, num)
    if torch.cuda.is_available():
        model = model.cuda()

    train_ds = Data(_to_repo_data(prep.train))
    val_ds = Data(_to_repo_data(prep.val))
    test_ds = Data(_to_repo_data(prep.test))
    val_targets = np.array([s.target for s in prep.val])

    def predict(ds: Data) -> np.ndarray:
        model.eval()
        out = []
        with torch.no_grad():
            for batch in DataLoader(ds, batch_size=opt.batch_size, shuffle=False):
                _, scores = forward(model, batch)
                out.append(scores.cpu().numpy())
        return np.concatenate(out, axis=0)

    best_state, best_val, best_epoch, bad = copy.deepcopy(model.state_dict()), -1.0, -1, 0
    history = []
    for epoch in range(max_epochs):
        model.train()
        for batch in DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True):
            model.optimizer.zero_grad()
            targets, scores = forward(model, batch)
            targets = targets.to(scores.device).long()
            loss = model.loss_function(scores, targets - 1)
            loss.backward()
            model.optimizer.step()
        model.scheduler.step()
        val_recall = _recall_at(predict(val_ds), val_targets, 20)
        history.append({"epoch": epoch, "val_recall": val_recall})
        if val_recall > best_val:
            best_val, best_epoch, bad = val_recall, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    scores = predict(test_ds)
    return scores, {"best_epoch": best_epoch, "best_val_recall": best_val, "history": history,
                    "_val_scores": predict(val_ds)}


def run_dhcn(prep: Prepared, seed: int, max_epochs: int, patience: int) -> tuple[np.ndarray, dict]:
    """DHCN hard-codes CUDA tensors in LineConv, so this runner needs a GPU."""
    repo = THIRD_PARTY / "DHCN"
    sys.path.insert(0, str(repo))
    import torch
    from model import DHCN, forward, trans_to_cpu
    from util import Data

    torch.manual_seed(seed)
    np.random.seed(seed)
    n_node = prep.n_items  # DHCN ids are 1..N, embedding has N rows (id-1)
    train = Data(_ragged_repo_data(prep.train), shuffle=True, n_node=n_node)
    val = Data(_ragged_repo_data(prep.val), shuffle=False, n_node=n_node)
    test = Data(_ragged_repo_data(prep.test), shuffle=False, n_node=n_node)
    model = DHCN(adjacency=train.adjacency, n_node=n_node, lr=1e-3, layers=3, l2=1e-5, beta=0.01,
                 dataset="custom", emb_size=100, batch_size=100)
    if torch.cuda.is_available():
        model = model.cuda()
    val_targets = np.array([s.target for s in prep.val])

    def predict(data: Data) -> np.ndarray:
        model.eval()
        out = []
        with torch.no_grad():
            for i in data.generate_batch(model.batch_size):
                _, scores, _ = forward(model, i, data)
                out.append(trans_to_cpu(scores).numpy())
        # generate_batch pads the last batch; eval is unshuffled, so the first rows are ours
        return np.concatenate(out, axis=0)[: data.length]

    best_state, best_val, best_epoch, bad = copy.deepcopy(model.state_dict()), -1.0, -1, 0
    history = []
    for epoch in range(max_epochs):
        model.train()
        for i in train.generate_batch(model.batch_size):
            model.zero_grad()
            targets, scores, con_loss = forward(model, i, train)
            targets = torch.as_tensor(targets, dtype=torch.long, device=scores.device)
            loss = model.loss_function(scores + 1e-8, targets) + con_loss
            loss.backward()
            model.optimizer.step()
        val_recall = _recall_at(predict(val), val_targets, 20)
        history.append({"epoch": epoch, "val_recall": val_recall})
        if val_recall > best_val:
            best_val, best_epoch, bad = val_recall, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    scores = predict(test)
    return scores, {"best_epoch": best_epoch, "best_val_recall": best_val, "history": history,
                    "_val_scores": predict(val)}


GRU4REC_DIR = THIRD_PARTY / "GRU4Rec_PyTorch_Official"  # official PyTorch port by Hidasi
NEG_SCORE = -1e9  # score for items the GRU4Rec model cannot rank (never seen in train)


def _session_frame(sessions: list[list[int]]):
    """Train sessions as a (SessionId, ItemId, Time) frame. Time is a global counter, so the
    iterator's session_order='time' keeps the original chronology of the sessions."""
    import pandas as pd

    sid = np.concatenate([np.full(len(s), i) for i, s in enumerate(sessions)])
    item = np.concatenate([np.asarray(s, dtype=np.int64) for s in sessions])
    return pd.DataFrame({"SessionId": sid, "ItemId": item, "Time": np.arange(item.size)})


def _gru4rec_scores(gru, samples: list[Sample], n_nodes: int, device, batch: int = 512) -> np.ndarray:
    """Full (n, N) scores for prefix -> next. A prefix runs through the GRU alone, so padding on
    the right does not change the output at the prefix's last position. Prefix items the model
    never saw in train are dropped, as the repo's batch_eval does."""
    import torch

    gru.model.eval()
    idmap = gru.data_iterator.itemidmap  # node id -> model index
    to_model = {int(node): int(idx) for node, idx in zip(idmap.index.values, idmap.values)}
    node_of = np.empty(gru.data_iterator.n_items, dtype=np.int64)
    node_of[idmap.values] = idmap.index.values

    out = np.full((len(samples), n_nodes), NEG_SCORE, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(samples), batch):
            chunk = samples[start:start + batch]
            seqs = [[to_model[v] for v in s.prefix if v in to_model] for s in chunk]
            lens = np.array([len(q) for q in seqs])
            B, L = len(chunk), max(1, int(lens.max()))
            X = torch.zeros((B, L), dtype=torch.long, device=device)
            for i, q in enumerate(seqs):
                X[i, :len(q)] = torch.tensor(q, dtype=torch.long, device=device)
            H = [torch.zeros((B, dim), device=device) for dim in gru.layers]
            final = torch.zeros((B, gru.data_iterator.n_items), device=device)
            for t in range(L):
                O = gru.model.forward(X[:, t], H, None, training=False)
                sel = torch.as_tensor(lens - 1 == t, device=device)
                if bool(sel.any()):
                    final[sel] = O[sel]
            block = np.full((B, n_nodes), NEG_SCORE, dtype=np.float32)
            block[:, node_of] = final.cpu().numpy()
            block[lens == 0] = NEG_SCORE
            out[start:start + B] = block
    return out


def run_gru4rec(prep: Prepared, seed: int, max_epochs: int, patience: int) -> tuple[np.ndarray, dict]:
    """Official PyTorch GRU4Rec with the repo's default hyperparameters (layers=[100],
    cross-entropy, n_sample=2048). Epochs run one by one so that early stopping works on val
    R@20, as in the other baselines. The loop mirrors GRU4Rec.fit() in gru4rec_pytorch.py."""
    repo = str(GRU4REC_DIR)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import torch
    from gru4rec_pytorch import (
        GRU4Rec,
        GRU4RecModel,
        IndexedAdagradM,
        SessionDataIterator,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gru = GRU4Rec(device=device)  # repo defaults; n_epochs is driven by the loop below

    gru.data_iterator = SessionDataIterator(
        _session_frame(prep.train_sessions), gru.batch_size, n_sample=gru.n_sample,
        sample_alpha=gru.sample_alpha, sample_cache_max_size=10_000_000,
        session_order="time", device=device,
    )
    gru.model = GRU4RecModel(gru.data_iterator.n_items, gru.layers, gru.dropout_p_embed,
                             gru.dropout_p_hidden, gru.embedding, gru.constrained_embedding).to(device)
    gru.model._reset_weights_to_compatibility_mode()
    opt = IndexedAdagradM(gru.model.parameters(), gru.learning_rate, gru.momentum)
    val_targets = np.array([s.target for s in prep.val])

    best_state, best_val, best_epoch, bad = copy.deepcopy(gru.model.state_dict()), -1.0, -1, 0
    history = []
    for epoch in range(max_epochs):
        gru.model.train()
        H = [torch.zeros((gru.batch_size, dim), device=device) for dim in gru.layers]

        def reset_hook(n_valid, finished_mask, valid_mask, H=H):
            return gru._adjust_hidden(n_valid, finished_mask, valid_mask, H)

        for in_idx, out_idx in gru.data_iterator(enable_neg_samples=gru.n_sample > 0, reset_hook=reset_hook):
            for h in H:
                h.detach_()
            gru.model.zero_grad()
            R = gru.model.forward(in_idx, H, out_idx, training=True)
            loss = gru.loss_function(R, out_idx, gru.batch_size) / gru.batch_size
            if not torch.isfinite(loss):
                raise RuntimeError(f"GRU4Rec loss is not finite at epoch {epoch}")
            loss.backward()
            opt.step()

        val_recall = _recall_at(_gru4rec_scores(gru, prep.val, prep.n_items, device), val_targets, 20)
        history.append({"epoch": epoch, "val_recall": val_recall})
        if val_recall > best_val:
            best_val, best_epoch, bad = val_recall, epoch, 0
            best_state = copy.deepcopy(gru.model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break

    gru.model.load_state_dict(best_state)
    scores = _gru4rec_scores(gru, prep.test, prep.n_items, device)
    return scores, {"best_epoch": best_epoch, "best_val_recall": best_val, "history": history,
                    "_val_scores": _gru4rec_scores(gru, prep.val, prep.n_items, device),
                    "hyperparams": {"layers": gru.layers, "loss": gru.loss, "batch_size": gru.batch_size,
                                    "learning_rate": gru.learning_rate, "n_sample": gru.n_sample,
                                    "sample_alpha": gru.sample_alpha, "elu_param": gru.elu_param}}


RUNNERS = {"srgnn": run_srgnn, "gcegnn": run_gcegnn, "dhcn": run_dhcn, "gru4rec": run_gru4rec}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(RUNNERS), required=True)
    ap.add_argument("--city", choices=CITIES, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--raw", type=str, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/scores"))
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--save-val", action="store_true", help="also write {stem}_val.npy")
    args = ap.parse_args()

    prep = prepare(load_city(Path(args.raw), args.city))
    scores, info = RUNNERS[args.model](prep, args.seed, args.max_epochs, args.patience)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}_{args.city}_seed{args.seed}"
    val_scores = info.pop("_val_scores")
    np.save(args.out / f"{stem}.npy", scores.astype(np.float32))
    if args.save_val:  # plan D6: val scores of the selected epoch, for val-side selection
        if val_scores.shape[0] != len(prep.val):
            raise RuntimeError(f"{stem}: val scores {val_scores.shape}, expected {len(prep.val)} rows")
        np.save(args.out / f"{stem}_val.npy", val_scores.astype(np.float32))
    (args.out / f"{stem}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"{stem}: best_epoch={info['best_epoch']} val_R@20={info['best_val_recall']:.4f}", flush=True)


if __name__ == "__main__":
    main()
