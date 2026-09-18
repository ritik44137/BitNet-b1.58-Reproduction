"""Build a tiny Transformer from a YAML config dict."""

from __future__ import annotations

from typing import Any

from src.models.transformer_baseline import TinyTransformer, TinyTransformerConfig
from src.models.transformer_bitnet import TinyBitNetConfig, TinyBitNetTransformer


def build_model_from_config(cfg: dict[str, Any], vocab_size: int):
    m = cfg["model"]
    common = dict(
        vocab_size=vocab_size,
        n_layer=int(m["n_layer"]),
        n_embd=int(m["n_embd"]),
        n_head=int(m["n_head"]),
        block_size=int(m.get("block_size", cfg.get("data", {}).get("seq_len", 256))),
        dropout=float(m.get("dropout", 0.0)),
        bias=bool(m.get("bias", True)),
        use_bitlinear=bool(m.get("use_bitlinear", False)),
        bitlinear_on_attention=bool(m.get("bitlinear_on_attention", False)),
        bitlinear_on_mlp=bool(m.get("bitlinear_on_mlp", False)),
        quantize_lm_head=bool(m.get("quantize_lm_head", False)),
    )
    if common["use_bitlinear"]:
        return TinyBitNetTransformer(TinyBitNetConfig(**common))
    return TinyTransformer(TinyTransformerConfig(**common))
