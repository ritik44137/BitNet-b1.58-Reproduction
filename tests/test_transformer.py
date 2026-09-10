"""Integration tests for baseline vs BitNet tiny Transformer."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.layers.bitlinear import BitLinear
from src.models import TinyBitNetConfig, TinyBitNetTransformer, TinyTransformer, TinyTransformerConfig


def _tiny_cfg(**overrides) -> TinyTransformerConfig:
    defaults = dict(
        vocab_size=32,
        n_layer=2,
        n_embd=32,
        n_head=4,
        block_size=16,
        dropout=0.0,
        bias=True,
    )
    defaults.update(overrides)
    return TinyTransformerConfig(**defaults)


def test_baseline_forward_shapes_and_loss():
    model = TinyTransformer(_tiny_cfg())
    idx = torch.randint(0, 32, (2, 8))
    targets = torch.randint(0, 32, (2, 8))
    logits, loss = model(idx, targets)
    assert logits.shape == (2, 8, 32)
    assert loss is not None and torch.isfinite(loss)


def test_bitnet_uses_bitlinear_on_attention_and_mlp():
    model = TinyBitNetTransformer(
        TinyBitNetConfig(
            vocab_size=32,
            n_layer=2,
            n_embd=32,
            n_head=4,
            block_size=16,
            dropout=0.0,
        )
    )
    # 2 layers * (1 fused qkv + 1 out_proj + 2 MLP) = 8 BitLinear modules
    assert model.count_bitlinear_modules() == 8
    assert isinstance(model.lm_head, nn.Linear)
    assert not isinstance(model.lm_head, BitLinear)

    block = model.transformer.h[0]
    assert isinstance(block.attn.c_attn, BitLinear)
    assert isinstance(block.attn.c_proj, BitLinear)
    assert isinstance(block.mlp.c_fc, BitLinear)
    assert isinstance(block.mlp.c_proj, BitLinear)


def test_baseline_has_no_bitlinear():
    model = TinyTransformer(_tiny_cfg())
    assert model.count_bitlinear_modules() == 0


def test_quantize_lm_head_flag():
    cfg = _tiny_cfg(use_bitlinear=True, quantize_lm_head=True)
    model = TinyTransformer(cfg)
    assert isinstance(model.lm_head, BitLinear)


def test_attention_only_ablation():
    cfg = _tiny_cfg(
        use_bitlinear=True,
        bitlinear_on_attention=True,
        bitlinear_on_mlp=False,
    )
    model = TinyTransformer(cfg)
    block = model.transformer.h[0]
    assert isinstance(block.attn.c_attn, BitLinear)
    assert isinstance(block.mlp.c_fc, nn.Linear)
    assert not isinstance(block.mlp.c_fc, BitLinear)
    # 2 layers * 2 attn projections
    assert model.count_bitlinear_modules() == 4


def test_bitnet_backward_finite():
    model = TinyBitNetTransformer(
        TinyBitNetConfig(vocab_size=32, n_layer=2, n_embd=32, n_head=4, block_size=16, dropout=0.0)
    )
    idx = torch.randint(0, 32, (2, 8))
    targets = torch.randint(0, 32, (2, 8))
    _, loss = model(idx, targets)
    assert loss is not None
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, name
            assert torch.isfinite(p.grad).all(), name


def test_param_count_parity():
    baseline = TinyTransformer(_tiny_cfg())
    bitnet = TinyBitNetTransformer(
        TinyBitNetConfig(vocab_size=32, n_layer=2, n_embd=32, n_head=4, block_size=16, dropout=0.0)
    )
    assert sum(p.numel() for p in baseline.parameters()) == sum(
        p.numel() for p in bitnet.parameters()
    )


def test_generate_grows_sequence():
    model = TinyTransformer(_tiny_cfg())
    model.eval()
    idx = torch.zeros((1, 4), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=3)
    assert out.shape == (1, 7)
