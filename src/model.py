"""Same transformer, two feed-forward choices: dense (control) or sparse nodes."""
from __future__ import annotations

import torch
import torch.nn as nn

from nodes import Block, DenseFFN, SparseNodeField


class CharLM(nn.Module):
    def __init__(self, vocab: int, dim: int, layers: int, heads: int, seq_len: int,
                 ffn: str = "dense", nodes: int = 8, active: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.position = nn.Embedding(seq_len, dim)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

        blocks = []
        for _ in range(layers):
            module = (
                DenseFFN(dim, dropout)
                if ffn == "dense"
                else SparseNodeField(dim, nodes=nodes, active=active, dropout=dropout)
            )
            blocks.append(Block(dim, heads, module, dropout))
        self.layers = nn.ModuleList(blocks)
        self.ffn = ffn
        self.nodes = nodes
        self.active = active

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.embed(tokens) + self.position(positions)[None]
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))

    def active_load(self) -> list[float]:
        """Per-layer node utilisation, for checking that routing stays balanced."""
        loads = []
        for layer in self.layers:
            if isinstance(layer.ffn, SparseNodeField):
                loads.append(layer.ffn.last_load.tolist())
        return loads

    def compute_ratio(self) -> float:
        """Fraction of the dense FFN arithmetic this model performs."""
        if self.ffn == "dense":
            return 1.0
        return self.active / self.nodes
