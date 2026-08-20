"""Absmean ternary quantization and straight-through estimator helpers."""

from __future__ import annotations

import torch


def absmean_scale(weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return mean(|W|) clamped to at least ``eps``."""
    raise NotImplementedError("Implement absmean scale: weight.abs().mean().clamp_min(eps)")


def ternary_quantize(
    weight: torch.Tensor, eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``weight`` to ternary codes in {-1, 0, +1} with absmean scale.

    Returns
    -------
    q : Tensor
        Ternary codes in {-1, 0, +1}.
    scale : Tensor
        Absmean scale used for normalization.
    """
    raise NotImplementedError(
        "Implement: scale = absmean_scale(w); q = (w / scale).round().clamp(-1, 1)"
    )


def ste_ternary_weight(weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return STE-wrapped effective weights: forward uses W_q, backward updates W.

    Canonical form::

        w_eff = w + (quantize(w) - w).detach()
    """
    raise NotImplementedError(
        "Implement STE: q, s = ternary_quantize(w); return w + (q * s - w).detach()"
    )
