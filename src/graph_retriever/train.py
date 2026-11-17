"""Training loop: cross-entropy over all items, early stopping on val Recall@20."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from graph_retriever.dataset import Prepared, Sample
from graph_retriever.hypergraph import Hypergraph, clique_operator
from graph_retriever.model import HGRetriever

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TIME_BUCKETS = 12  # see dataset.time_bucket


@dataclass
class TrainConfig:
    dim: int = 64
    n_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 50
    patience: int = 5
    eval_k: int = 20
    encoder: str = "attn"
    residual: bool = False
    session_edge: bool = False  # add a one-edge hyperedge over the prefix (leak-free)
    time_ctx: bool = False  # add a learned embedding of the last prefix visit's time bucket
    ssl_beta: float = 0.0  # weight of the two-view InfoNCE loss (0 = off)


def _pad(samples: list[Sample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    L = max(len(s.prefix) for s in samples)
    prefix = torch.zeros(len(samples), L, dtype=torch.long)
    mask = torch.zeros(len(samples), L, dtype=torch.bool)
    for i, s in enumerate(samples):
        prefix[i, : len(s.prefix)] = torch.tensor(s.prefix, dtype=torch.long)
        mask[i, : len(s.prefix)] = True
    ctx = torch.tensor([s.ctx for s in samples], dtype=torch.long)
    target = torch.tensor([s.target for s in samples], dtype=torch.long)
    return prefix.to(DEVICE), mask.to(DEVICE), ctx.to(DEVICE), target.to(DEVICE)


def score_matrix(model: HGRetriever, samples: list[Sample], batch_size: int = 1024) -> np.ndarray:
    """Log-softmax relevance (n, N) for each sample, computed without dropout."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            prefix, mask, ctx, _ = _pad(samples[i : i + batch_size])
            out.append(F.log_softmax(model(prefix, mask, ctx), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def recall_at_k(scores: np.ndarray, targets: np.ndarray, k: int) -> float:
    topk = np.argpartition(-scores, k, axis=1)[:, :k]
    hit = (topk == targets[:, None]).any(axis=1)
    return float(hit.mean())


def train_model(
    prep: Prepared,
    hg: Hypergraph,
    cfg: TrainConfig,
    seed: int,
    learn_gate: bool = True,
) -> tuple[HGRetriever, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    operators = {ch: torch.from_numpy(hg.operators[ch]) for ch in hg.channels}
    clique = None
    if cfg.ssl_beta > 0:
        if not hg.H:
            raise ValueError("ssl needs at least one hypergraph channel")
        clique = torch.from_numpy(clique_operator(list(hg.H.values())))
    model = HGRetriever(
        n_items=prep.n_items,
        operators=operators,
        dim=cfg.dim,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        learn_gate=learn_gate,
        encoder=cfg.encoder,
        residual=cfg.residual,
        session_edge=cfg.session_edge,
        n_time=N_TIME_BUCKETS if cfg.time_ctx else 0,
        clique_op=clique if cfg.ssl_beta > 0 else None,
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    val_targets = np.array([s.target for s in prep.val])

    best_state = copy.deepcopy(model.state_dict())
    best_val = -1.0
    best_epoch = -1
    bad = 0
    history = []
    rng = np.random.default_rng(seed)
    train = prep.train

    for epoch in range(cfg.max_epochs):
        model.train()
        order = rng.permutation(len(train))
        total = 0.0
        for i in range(0, len(order), cfg.batch_size):
            batch = [train[j] for j in order[i : i + cfg.batch_size]]
            prefix, mask, ctx, target = _pad(batch)
            loss = F.cross_entropy(model(prefix, mask, ctx), target)
            if cfg.ssl_beta > 0:
                loss = loss + cfg.ssl_beta * model.ssl_loss(prefix, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        val_recall = recall_at_k(score_matrix(model, prep.val), val_targets, cfg.eval_k)
        history.append({"epoch": epoch, "loss": total / len(train), "val_recall": val_recall})
        if val_recall > best_val:
            best_val = val_recall
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_epoch, "best_val_recall": best_val, "history": history}
