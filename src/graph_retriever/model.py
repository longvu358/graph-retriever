"""HGNN retriever: per-channel hypergraph convolution, learned channel gate,
session encoder (attention pooling or an order-aware recurrent encoder: GRU, LSTM or LRU),
softmax over all items."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

ENCODERS = ("attn", "gru", "lstm", "lru")


class LRU(nn.Module):
    """Linear Recurrent Unit (Orvieto et al., 2023), diagonal complex recurrence:

        h_t = lambda * h_{t-1} + gamma * (B x_t),  y_t = Re(C h_t) + D * x_t,
        lambda = exp(-exp(nu_log) + i exp(theta_log)),  gamma = sqrt(1 - |lambda|^2).

    Complex numbers are kept as (real, imag) pairs. The scan runs step by step, which is
    cheap for short sessions and causal, so right padding never changes earlier outputs.
    """

    def __init__(self, dim: int, state: int | None = None, r_min: float = 0.0,
                 r_max: float = 1.0, max_phase: float = 2 * math.pi) -> None:
        super().__init__()
        n = state or dim
        u1, u2 = torch.rand(n), torch.rand(n)
        # |lambda| uniform on the ring [r_min, r_max], phase uniform on [0, max_phase]
        nu_log = torch.log(-0.5 * torch.log(u1 * (r_max**2 - r_min**2) + r_min**2))
        theta_log = torch.log(max_phase * u2)
        self.nu_log = nn.Parameter(nu_log)
        self.theta_log = nn.Parameter(theta_log)
        mod = torch.exp(-torch.exp(nu_log))
        self.gamma_log = nn.Parameter(torch.log(torch.sqrt(1 - mod**2)))
        self.B_re = nn.Parameter(torch.randn(n, dim) / math.sqrt(2 * dim))
        self.B_im = nn.Parameter(torch.randn(n, dim) / math.sqrt(2 * dim))
        self.C_re = nn.Parameter(torch.randn(dim, n) / math.sqrt(n))
        self.C_im = nn.Parameter(torch.randn(dim, n) / math.sqrt(n))
        self.D = nn.Parameter(torch.randn(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d) -> (B, L, d)."""
        mod = torch.exp(-torch.exp(self.nu_log))
        phase = torch.exp(self.theta_log)
        lam_re, lam_im = mod * torch.cos(phase), mod * torch.sin(phase)
        gamma = torch.exp(self.gamma_log)
        bu_re = gamma * (x @ self.B_re.T)  # (B, L, n)
        bu_im = gamma * (x @ self.B_im.T)
        h_re = x.new_zeros(x.shape[0], self.nu_log.shape[0])
        h_im = torch.zeros_like(h_re)
        states_re, states_im = [], []
        for t in range(x.shape[1]):
            h_re, h_im = (lam_re * h_re - lam_im * h_im + bu_re[:, t],
                          lam_re * h_im + lam_im * h_re + bu_im[:, t])
            states_re.append(h_re)
            states_im.append(h_im)
        H_re = torch.stack(states_re, dim=1)
        H_im = torch.stack(states_im, dim=1)
        return H_re @ self.C_re.T - H_im @ self.C_im.T + self.D * x


class LRUBlock(nn.Module):
    """LRU block as in the paper: x + GLU(GELU(LRU(LayerNorm(x))))."""

    def __init__(self, dim: int, **lru_init: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.lru = LRU(dim, **lru_init)
        self.glu = nn.Linear(dim, 2 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.gelu(self.lru(self.norm(x)))
        return x + F.glu(self.glu(y), dim=-1)


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
        lru_init: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        if encoder not in ENCODERS:
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
        elif encoder == "gru":
            self.gru = nn.GRU(dim, dim, batch_first=True)
        elif encoder == "lstm":
            self.lstm = nn.LSTM(dim, dim, batch_first=True)
        else:
            self.lru = LRUBlock(dim, **(lru_init or {}))
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
        # recurrent encoders are causal: right padding does not affect states before it
        if self.encoder == "gru":
            out, _ = self.gru(h)
        elif self.encoder == "lstm":
            out, _ = self.lstm(h)
        else:
            out = self.lru(h)
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
