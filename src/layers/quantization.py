"""Absmean ternary quantization and straight-through estimator helpers."""

from __future__ import annotations

import torch


def absmean_scale(weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return mean(|W|) clamped to at least ``eps``.

    Used to normalize weights before rounding so ternarization stays relative
    to the layer's typical magnitude instead of collapsing or saturating.
    """
    return weight.abs().mean().clamp_min(eps)


def ternary_quantize(
    weight: torch.Tensor, eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``weight`` to ternary codes in {-1, 0, +1} with absmean scale.

    Computes::

        s = mean(|W|).clamp_min(eps)
        Q = clip(round(W / s), -1, 1)

    Returns
    -------
    q : Tensor
        Ternary codes in {-1, 0, +1}.
    scale : Tensor
        Absmean scale used for normalization.
    """
    scale = absmean_scale(weight, eps)
    q = (weight / scale).round().clamp(-1, 1)
    return q, scale


def ste_ternary_weight(weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return STE-wrapped effective weights: forward uses W_q, backward updates W.

    Canonical form::

        w_eff = w + (quantize(w) - w).detach()

    Forward value equals the scaled ternary weights ``s * Q``. Backward
    gradients flow as if the output had been the full-precision ``weight``.
    """
    q, scale = ternary_quantize(weight, eps)
    quantized = q * scale
    return weight + (quantized - weight).detach()
