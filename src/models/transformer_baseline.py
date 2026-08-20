"""Full-precision tiny decoder-only Transformer baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class TinyTransformerConfig:
    vocab_size: int
    n_layer: int = 4
    n_embd: int = 256
    n_head: int = 4
    block_size: int = 256
    dropout: float = 0.1
    bias: bool = True


def make_linear(
    in_dim: int,
    out_dim: int,
    bias: bool = True,
    use_bitlinear: bool = False,
) -> nn.Module:
    """Factory used by both baseline and BitNet models for architecture parity."""
    if use_bitlinear:
        from src.layers.bitlinear import BitLinear

        return BitLinear(in_dim, out_dim, bias=bias)
    return nn.Linear(in_dim, out_dim, bias=bias)


class TinyTransformer(nn.Module):
    """Placeholder tiny LM. Implement blocks with ``make_linear(..., use_bitlinear=False)``."""

    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.config = config
        # TODO: embeddings, transformer blocks (attn + MLP via make_linear), LM head
        raise NotImplementedError("Implement full-precision tiny Transformer")

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError
