"""Theoretical vs runtime memory accounting helpers."""

from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn as nn

from src.layers.bitlinear import BitLinear


def theoretical_weight_bits(n_params: int, *, bits_per_param: float) -> float:
    return float(n_params) * float(bits_per_param)


def ternary_bits_per_param() -> float:
    return math.log2(3)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def summarize_param_memory(module: nn.Module) -> Mapping[str, float]:
    """Split BitLinear weights from dense params for theoretical storage.

    Training still keeps full-precision master weights. ``reported_weight_bits``
    assumes BitLinear weights could be packed at log2(3) bits while all other
    parameters stay fp32.
    """
    ternary_weight_params = sum(
        m.weight.numel() for m in module.modules() if isinstance(m, BitLinear)
    )
    total = count_parameters(module)
    dense_params = total - ternary_weight_params

    return {
        "n_params": float(total),
        "ternary_eligible_params": float(ternary_weight_params),
        "dense_params": float(dense_params),
        "fp32_bits": theoretical_weight_bits(total, bits_per_param=32),
        "fp16_bits": theoretical_weight_bits(total, bits_per_param=16),
        "ideal_ternary_bits": theoretical_weight_bits(
            ternary_weight_params, bits_per_param=ternary_bits_per_param()
        ),
        "reported_weight_bits": theoretical_weight_bits(
            ternary_weight_params, bits_per_param=ternary_bits_per_param()
        )
        + theoretical_weight_bits(dense_params, bits_per_param=32),
    }


def peak_cuda_memory_bytes() -> int | None:
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())
