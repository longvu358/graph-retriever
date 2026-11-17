"""HGNN retriever: per-channel hypergraph convolution, learned channel gate,
session encoder (attention pooling or order-aware GRU), softmax over all items."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class HGRetriever(nn.Module):
    def __init__(
        self,
        n_items: int,
        operators: dict[str, torch.Tensor],
        dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
        learn_gate: bool = True,
        encoder: str = "attn",
        residual: bool = False,
        session_edge: bool = False,
        n_time: int = 0,
        clique_op: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if encoder not in ("attn", "gru"):
            raise ValueError(f"unknown encoder: {encoder}")
        self.channels = tuple(operators)
        self.dropout = dropout
        self.encoder = encoder
        self.residual = residual
        self.session_edge = session_edge
        self.emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.emb.weight, std=0.1)
        for ch, P in operators.items():
            self.register_buffer(f"P_{ch}", P)
        self.layers = nn.ModuleDict(
            {
                ch: nn.ModuleList(
                    [nn.Linear(dim, dim, bias=False) for _ in range(n_layers)]
                )
                for ch in self.channels
            }
        )
        n_ch = len(self.channels)
        if learn_gate:
            self.gate_logits = nn.Parameter(torch.zeros(n_ch))
        else:
            self.register_buffer("gate_logits", torch.zeros(n_ch))
        if encoder == "attn":
            self.att = nn.Linear(dim, 1)
        else:
            self.gru = nn.GRU(dim, dim, batch_first=True)
        self.time_emb = None
        if n_time > 0:
            self.time_emb = nn.Embedding(n_time, dim)
            nn.init.normal_(self.time_emb.weight, std=0.1)
        # second view for self-supervision (DHCN-style): clique/line-graph propagation
        self.clique_op = None
        if clique_op is not None:
            self.register_buffer("P_clique", clique_op)
            self.layers_b = nn.ModuleList(
                [nn.Linear(dim, dim, bias=False) for _ in range(n_layers)]
            )
            self.clique_op = True

    def gate(self) -> torch.Tensor:
        return torch.softmax(self.gate_logits, dim=0)

    def item_reprs(self) -> torch.Tensor:
        """z_v = sum_c alpha_c * z_v^(c), with z^(c) = sigma(P_c X Theta_c) stacked over layers."""
        x0 = self.emb.weight
        if not self.channels:  # no-graph control: plain item embeddings
            return x0
        per_channel = []
        for ch in self.channels:
            P = getattr(self, f"P_{ch}")
            x = x0
            for lin in self.layers[ch]:
                out = F.relu(P @ lin(x))
                x = out + x if self.residual else out
                x = F.dropout(x, self.dropout, self.training)
            per_channel.append(x)
        Z = torch.stack(per_channel)  # (C, N, d)
        return (self.gate()[:, None, None] * Z).sum(dim=0)

    def view_b_reprs(self) -> torch.Tensor:
        """Second view: item states propagated on the clique graph (same embeddings)."""
        x = self.emb.weight
        for lin in self.layers_b:
            out = F.relu(self.P_clique @ lin(x))
            x = out + x if self.residual else out
            x = F.dropout(x, self.dropout, self.training)
        return x

    def ssl_loss(self, prefix: torch.Tensor, mask: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
        """InfoNCE between session representations of the two views (in-batch negatives).

        Uses prefix only, so it cannot see the target.
        """
        if self.clique_op is None:
            raise RuntimeError("ssl_loss needs a second view (clique_op)")
        s_a = self.session_repr(self.item_reprs()[prefix], mask)
        s_b = self.session_repr(self.view_b_reprs()[prefix], mask)
        s_a = F.normalize(s_a, dim=1)
        s_b = F.normalize(s_b, dim=1)
        logits = s_a @ s_b.T / tau
        labels = torch.arange(prefix.shape[0], device=prefix.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

    def session_repr(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """h: (B, L, d) prefix item states, right-padded; mask: (B, L) bool."""
        if self.encoder == "attn":
            logits = self.att(h).squeeze(-1).masked_fill(~mask, float("-inf"))
            a = torch.softmax(logits, dim=1)
            return (a.unsqueeze(-1) * h).sum(dim=1)
        out, _ = self.gru(h)  # right padding does not affect states before it
        last = mask.sum(dim=1) - 1
        return out[torch.arange(h.shape[0], device=h.device), last]

    def forward(
        self,
        prefix: torch.Tensor,
        mask: torch.Tensor,
        ctx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """prefix: (B, L) item ids, mask: (B, L) bool, ctx: (B,) time bucket of last prefix visit.

        Only prefix information enters the session representation (no target leakage).
        Returns scores (B, N).
        """
        z = self.item_reprs()
        h = z[prefix]  # (B, L, d)
        s = self.session_repr(h, mask)  # (B, d)
        if self.session_edge:
            # one hyperedge containing exactly the prefix: node -> edge -> node is a masked mean
            m = mask.unsqueeze(-1).float()
            s = s + (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        if self.time_emb is not None and ctx is not None:
            s = s + self.time_emb(ctx)
        return s @ z.T
