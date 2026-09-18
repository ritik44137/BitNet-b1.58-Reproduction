"""Sanity tests for quantization and BitLinear."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.layers.bitlinear import BitLinear
from src.layers.quantization import absmean_scale, ste_ternary_weight, ternary_quantize


def test_absmean_scale_matches_mean_abs():
    w = torch.tensor([[-2.0, 0.0], [4.0, -2.0]])
    assert absmean_scale(w).item() == pytest.approx(2.0)


def test_absmean_scale_clamps_eps():
    w = torch.zeros(4, 4)
    assert absmean_scale(w, eps=1e-5).item() == pytest.approx(1e-5)


def test_ternary_values_only():
    w = torch.randn(16, 8)
    q, scale = ternary_quantize(w)
    assert set(q.unique().tolist()).issubset({-1.0, 0.0, 1.0})
    assert scale.ndim == 0
    assert scale.item() > 0


def test_ternary_quantize_known_values():
    # scale = 1.5 → round(W/s) clips to [-1, -1, 0, 1, 1, 1]
    w = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    q, scale = ternary_quantize(w)
    assert scale.item() == pytest.approx(1.5)
    assert torch.equal(q, torch.tensor([-1.0, -1.0, 0.0, 1.0, 1.0, 1.0]))


def test_ste_forward_equals_quantized():
    w = torch.randn(8, 4, requires_grad=True)
    q, scale = ternary_quantize(w)
    assert torch.allclose(ste_ternary_weight(w), q * scale)


def test_ste_backward_reaches_master_weights():
    w = torch.randn(8, 4, requires_grad=True)
    ste_ternary_weight(w).sum().backward()
    assert w.grad is not None
    assert torch.isfinite(w.grad).all()
    assert torch.allclose(w.grad, torch.ones_like(w))


def test_bitlinear_shape_matches_linear():
    x = torch.randn(4, 8)
    y = BitLinear(8, 16)(x)
    assert y.shape == (4, 16)
    assert y.shape == nn.Linear(8, 16)(x).shape


def test_bitlinear_quantized_weight_ternary():
    layer = BitLinear(8, 16)
    q, scale, w_q = layer.quantized_weight()
    assert set(q.unique().tolist()).issubset({-1.0, 0.0, 1.0})
    assert torch.allclose(w_q, q * scale)


def test_bitlinear_gradients_finite():
    layer = BitLinear(8, 16)
    layer(torch.randn(4, 8)).sum().backward()
    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()
    assert layer.bias is not None and torch.isfinite(layer.bias.grad).all()


def test_bitlinear_no_bias():
    layer = BitLinear(8, 16, bias=False)
    assert layer(torch.randn(2, 8)).shape == (2, 16)
    assert layer.bias is None
