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
    # LRU encoder init (Orvieto et al., 2023): |lambda| on [r_min, r_max], phase on [0, max_phase]
    lru_r_min: float = 0.0
    lru_r_max: float = 1.0
    lru_max_phase: float = 6.283185307179586
    session_edge: bool = False  # add a one-edge hyperedge over the prefix (leak-free)
    time_ctx: bool = False  # add a learned embedding of the last prefix visit's time bucket
    ssl_beta: float = 0.0  # weight of the two-view InfoNCE loss (0 = off)
    # plan D6: logit adjustment. Train on logits + logq_tau * log q(i), q = smoothed train
    # popularity; score with the raw logits. Mimics GRU4Rec's uncorrected sampled softmax,
    # whose popular negatives push popular items down (0 = off, plain cross-entropy).
    logq_tau: float = 0.0
    # plan D8 (2a): multi-target training. A sample also predicts the next `future_horizon`
    # visits after its target, weighted gamma^k (k = 1 for the visit after the target). 0 = off.
    future_gamma: float = 0.0
    future_horizon: int = 3
    # plan D8 (2b): batch dispersion regulariser beta * N * sum_i pbar_i^2, pbar = mean softmax
    # over the batch (= 1 when uniform, larger when concentrated). 0 = off.
    disp_beta: float = 0.0


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


def future_targets(prep: Prepared, horizon: int) -> list[list[int]]:
    """For prep.train[j], the visits after its target in the same session (at most `horizon`).

    prep.train lists, per train session in order, the samples of dataset.augment (prefix
    lengths 1..len-1), so the futures are rebuilt from prep.train_sessions and checked.
    """
    out: list[list[int]] = []
    for s in prep.train_sessions:
        for i in range(1, len(s)):
            out.append(list(s[i + 1 : i + 1 + horizon]))
    if len(out) != len(prep.train):
        raise RuntimeError(f"future_targets: {len(out)} samples, prep.train has {len(prep.train)}")
    j = 0
    for s in prep.train_sessions:
        for i in range(1, len(s)):
            smp = prep.train[j]
            if smp.prefix != list(s[:i]) or smp.target != s[i]:
                raise RuntimeError(f"future_targets: sample {j} does not match its session")
            j += 1
    return out


def multi_target_loss(logits: torch.Tensor, target: torch.Tensor, futures: list[list[int]],
                      gamma: float) -> torch.Tensor:
    """sum_k w_k * CE(logits, t_k) / sum_k w_k per sample, t_0 = target (w = 1), t_k = k-th
    later visit (w = gamma^k); mean over the batch."""
    logp = F.log_softmax(logits, dim=1)
    h = max((len(f) for f in futures), default=0)
    idx = torch.zeros((len(futures), h + 1), dtype=torch.long, device=logits.device)
    w = torch.zeros((len(futures), h + 1), dtype=logp.dtype, device=logits.device)
    idx[:, 0] = target
    w[:, 0] = 1.0
    for b, f in enumerate(futures):
        for k, v in enumerate(f, start=1):
            idx[b, k] = v
            w[b, k] = gamma ** k
    nll = -logp.gather(1, idx)
    return ((nll * w).sum(dim=1) / w.sum(dim=1)).mean()


def dispersion_penalty(logits: torch.Tensor) -> torch.Tensor:
    """N * sum_i pbar_i^2 with pbar = batch mean of softmax(logits): 1 if uniform, N if all
    probability mass of every sample sits on one item."""
    pbar = F.softmax(logits, dim=1).mean(dim=0)
    return logits.shape[1] * (pbar * pbar).sum()


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
        lru_init={"r_min": cfg.lru_r_min, "r_max": cfg.lru_r_max, "max_phase": cfg.lru_max_phase},
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
    futures = future_targets(prep, cfg.future_horizon) if cfg.future_gamma > 0 else None
    log_q = None
    if cfg.logq_tau != 0.0:
        q = (prep.pop_count + 1.0) / (prep.pop_count.sum() + prep.n_items)
        log_q = torch.as_tensor(cfg.logq_tau * np.log(q), dtype=torch.float32, device=DEVICE)

    for epoch in range(cfg.max_epochs):
        model.train()
        order = rng.permutation(len(train))
        total = 0.0
        for i in range(0, len(order), cfg.batch_size):
            idx = order[i : i + cfg.batch_size]
            batch = [train[j] for j in idx]
            prefix, mask, ctx, target = _pad(batch)
            raw = model(prefix, mask, ctx)
            logits = raw + log_q if log_q is not None else raw
            if futures is not None:
                loss = multi_target_loss(logits, target, [futures[j] for j in idx], cfg.future_gamma)
            else:
                loss = F.cross_entropy(logits, target)
            if cfg.disp_beta > 0:
                loss = loss + cfg.disp_beta * dispersion_penalty(raw)
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
