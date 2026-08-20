"""Sanity tests for quantization and BitLinear (to be filled in during Day 2)."""

from __future__ import annotations

import pytest
import torch


@pytest.mark.skip(reason="Implement absmean + ternary quantization first")
def test_ternary_values_only():
    from src.layers.quantization import ternary_quantize

    w = torch.randn(16, 8)
    q, _scale = ternary_quantize(w)
    unique = set(q.unique().tolist())
    assert unique.issubset({-1.0, 0.0, 1.0})


@pytest.mark.skip(reason="Implement BitLinear.forward first")
def test_bitlinear_shape_matches_linear():
    from src.layers.bitlinear import BitLinear

    x = torch.randn(4, 8)
    layer = BitLinear(8, 16)
    y = layer(x)
    assert y.shape == (4, 16)


@pytest.mark.skip(reason="Implement STE path first")
def test_bitlinear_gradients_finite():
    from src.layers.bitlinear import BitLinear

    layer = BitLinear(8, 16)
    x = torch.randn(4, 8, requires_grad=True)
    y = layer(x).sum()
    y.backward()
    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()
