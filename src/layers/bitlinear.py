"""Drop-in BitLinear: full-precision master weights, ternary effective forward."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers.quantization import ste_ternary_weight, ternary_quantize


class BitLinear(nn.Module):
    """Linear layer with absmean ternary weight quantization + STE training.

    Matches ``nn.Linear`` closely enough to be used as a drop-in replacement for
    attention and MLP projections. Master ``weight`` stays full precision;
    the forward pass uses a quantized ternary proxy.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: w_eff = ste_ternary_weight(self.weight, self.eps); return F.linear(...)
        raise NotImplementedError("Wire ste_ternary_weight into F.linear")

    @torch.no_grad()
    def quantized_weight(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(q, scale, q * scale)`` for inspection / logging."""
        # TODO: return ternary_quantize result and scaled weights
        raise NotImplementedError("Expose q, scale, and scaled ternary weights")
