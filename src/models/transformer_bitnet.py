"""BitNet-style tiny Transformer with ternary attention/MLP projections."""

from __future__ import annotations

from dataclasses import dataclass

from src.models.transformer_baseline import TinyTransformer, TinyTransformerConfig


@dataclass
class TinyBitNetConfig(TinyTransformerConfig):
    use_bitlinear: bool = True
    bitlinear_on_attention: bool = True
    bitlinear_on_mlp: bool = True
    quantize_lm_head: bool = False


class TinyBitNetTransformer(TinyTransformer):
    """Same architecture as the baseline; BitLinear flags default to on."""

    def __init__(self, config: TinyBitNetConfig | TinyTransformerConfig) -> None:
        if not isinstance(config, TinyBitNetConfig):
            config = TinyBitNetConfig(
                vocab_size=config.vocab_size,
                n_layer=config.n_layer,
                n_embd=config.n_embd,
                n_head=config.n_head,
                block_size=config.block_size,
                dropout=config.dropout,
                bias=config.bias,
                use_bitlinear=getattr(config, "use_bitlinear", True),
                bitlinear_on_attention=getattr(config, "bitlinear_on_attention", True),
                bitlinear_on_mlp=getattr(config, "bitlinear_on_mlp", True),
                quantize_lm_head=getattr(config, "quantize_lm_head", False),
            )
        super().__init__(config)
