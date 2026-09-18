"""Tiny-batch overfit checks for baseline and BitNet."""

from __future__ import annotations

import torch

from src.data import CharTokenizer, get_batch
from src.models import (
    TinyBitNetConfig,
    TinyBitNetTransformer,
    TinyTransformer,
    TinyTransformerConfig,
)
from src.utils.seed import set_seed


def _overfit_once(use_bitlinear: bool, steps: int = 80) -> tuple[float, float]:
    set_seed(0)
    text = "To be, or not to be, that is the question:\nWhether 'tis nobler "
    tokenizer = CharTokenizer.from_text(text)
    data = torch.tensor(tokenizer.encode(text * 20), dtype=torch.long)

    block_size = 32
    common = dict(
        vocab_size=tokenizer.vocab_size,
        n_layer=2,
        n_embd=64,
        n_head=4,
        block_size=block_size,
        dropout=0.0,
        bias=True,
    )
    model = (
        TinyBitNetTransformer(TinyBitNetConfig(**common))
        if use_bitlinear
        else TinyTransformer(TinyTransformerConfig(**common))
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    model.train()

    x, y = get_batch(data, batch_size=4, block_size=block_size, device="cpu")
    _, loss0 = model(x, y)
    start = float(loss0.item())
    for _ in range(steps):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return start, float(loss.item())


def test_baseline_overfits_tiny_batch():
    start, end = _overfit_once(use_bitlinear=False)
    assert end < start * 0.5
    assert end < 1.0


def test_bitnet_overfits_tiny_batch():
    start, end = _overfit_once(use_bitlinear=True)
    assert end < start * 0.5
    assert end < 1.5
