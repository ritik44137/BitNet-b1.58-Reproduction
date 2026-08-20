"""BitNet-style tiny Transformer: attention/MLP projections use BitLinear."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from src.models.transformer_baseline import TinyTransformerConfig, make_linear


@dataclass
class TinyBitNetConfig(TinyTransformerConfig):
    use_bitlinear: bool = True
    bitlinear_on_attention: bool = True
    bitlinear_on_mlp: bool = True
    quantize_lm_head: bool = False


class TinyBitNetTransformer(nn.Module):
    """Same architecture as the baseline; selected linears swapped via factory flags.

    Keep embeddings, LayerNorm, and (by default) the LM head full precision.
    """

    def __init__(self, config: TinyBitNetConfig) -> None:
        super().__init__()
        self.config = config
        # TODO: share block construction with baseline; pass use_bitlinear into make_linear
        _ = make_linear  # referenced so the factory stays in the integration path
        raise NotImplementedError(
            "Implement BitNet tiny Transformer with BitLinear on attention/MLP"
        )

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError
