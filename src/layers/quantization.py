"""Absmean ternary quantization with a straight-through estimator."""

from __future__ import annotations

import torch


def absmean_scale(weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """mean(|W|), clamped away from zero."""
    return weight.abs().mean().clamp_min(eps)


def ternary_quantize(
    weight: torch.Tensor, eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ternary codes Q in {-1,0,+1} and the absmean scale s."""
    scale = absmean_scale(weight, eps)
    q = (weight / scale).round().clamp(-1, 1)
    return q, scale


def ste_ternary_weight(weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Forward uses s*Q; backward treats the op like identity w.r.t. ``weight``."""
    q, scale = ternary_quantize(weight, eps)
    quantized = q * scale
    return weight + (quantized - weight).detach()
