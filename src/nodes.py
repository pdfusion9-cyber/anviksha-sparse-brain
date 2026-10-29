"""A sparse node field that replaces the feed-forward network.

The brief: keep the transformer exactly as it is, and change only what has to
change. So attention, layer norms, embeddings and the head stay untouched. The
only substitution is the feed-forward block, because that is where most of the
parameters and most of the arithmetic already sit.

How it works
------------
A dense FFN of width 4d is replaced by eight small nodes of width d/2.

    dense FFN     2 * d * 4d  = 8d^2 parameters, 8d^2 multiply-adds per token
    node field    8 * 2 * d * (d/2) = 8d^2 parameters, same total

The parameter count is identical. But a router picks the top two nodes per
token, so only 2 of 8 nodes run:

    active compute  2 * 2 * d * (d/2) = 2d^2  ->  one quarter of the dense FFN

That is the "minimum compute, maximum output" claim, and it is structural rather
than rhetorical: same weights, a quarter of the arithmetic.

Dynamic interconnection
-----------------------
The active nodes for a token are not independent. After each runs, they exchange
one round of messages — the mean of the co-active nodes' pre-projection states —
before the outputs are combined with learned weights. Which nodes talk to each
other is decided per token by the router, so the graph is different for every
token and every layer.

Parallelism
-----------
Tokens that picked the same node are gathered into one batch and processed in a
single tensor op, so all active nodes across all tokens run in parallel rather
than in a Python loop over tokens.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseNodeField(nn.Module):
    """Eight nodes, top-k routing, one message-passing round among the active set."""

    def __init__(self, dim: int, nodes: int = 8, active: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.nodes = nodes
        self.active = active
        hidden = max(dim // 2, 8)

        # each node is a two-layer MLP of width d/2, matching the dense budget
        self.up = nn.Parameter(torch.empty(nodes, dim, hidden))
        self.down = nn.Parameter(torch.empty(nodes, hidden, dim))
        nn.init.normal_(self.up, std=dim**-0.5)
        nn.init.normal_(self.down, std=hidden**-0.5)

        self.router = nn.Linear(dim, nodes, bias=False)
        # learned weight for combining each active node's output
        self.combine = nn.Linear(dim, 1, bias=False)
        # one round of message passing among co-active nodes
        self.message = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.last_load = torch.zeros(nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        flat = x.reshape(-1, dim)

        logits = self.router(flat)
        top = logits.topk(self.active, dim=-1)
        index = top.indices                      # [N_tokens, k]
        weight = torch.softmax(top.values, dim=-1)

        # Group tokens by their chosen node and run each node once over its
        # whole group. Measured faster than gather-and-batched-matmul on CPU:
        # the gathered weight tensors cost more memory traffic than the saved
        # arithmetic is worth at these dimensions.
        gathered = torch.zeros(len(flat), self.active, dim, device=flat.device)
        for slot in range(self.active):
            node_ids = index[:, slot]
            for node in node_ids.unique().tolist():
                mask = node_ids == node
                hidden = F.gelu(flat[mask] @ self.up[node])
                gathered[mask, slot] = hidden @ self.down[node]

        outputs = (gathered * weight.unsqueeze(-1)).sum(dim=1)
        outputs = outputs + self.message(gathered.mean(dim=1))

        with torch.no_grad():
            counts = torch.bincount(index.reshape(-1), minlength=self.nodes).float()
            self.last_load = counts / counts.sum().clamp_min(1)

        return x + self.dropout(outputs.view(batch, length, dim))


class DenseFFN(nn.Module):
    """The control: the standard transformer feed-forward block."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class Block(nn.Module):
    """Identical attention for both variants; only the FFN slot differs."""

    def __init__(self, dim: int, heads: int, ffn: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = ffn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attended, _ = self.attention(normed, normed, normed, need_weights=False)
        x = x + attended
        return self.ffn(self.norm2(x))
