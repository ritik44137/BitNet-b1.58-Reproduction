"""BitLinear: nn.Linear-compatible layer with ternary effective weights."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers.quantization import ste_ternary_weight, ternary_quantize


class BitLinear(nn.Module):
    """Stores fp32 master weights; forward matmul uses absmean-ternary weights."""

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
        return F.linear(x, ste_ternary_weight(self.weight, self.eps), self.bias)

    @torch.no_grad()
    def quantized_weight(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, scale = ternary_quantize(self.weight, self.eps)
        return q, scale, q * scale

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, eps={self.eps}"
        )
