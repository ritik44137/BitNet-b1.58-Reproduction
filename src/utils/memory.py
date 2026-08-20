"""Theoretical vs runtime memory accounting helpers."""

from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn as nn


def theoretical_weight_bits(
    n_params: int,
    *,
    bits_per_param: float,
) -> float:
    """Return total bits for ``n_params`` at ``bits_per_param`` (e.g. 32, 16, ~1.58)."""
    return float(n_params) * float(bits_per_param)


def ternary_bits_per_param() -> float:
    """Ideal packed ternary storage: log2(3) ≈ 1.58 bits per weight."""
    return math.log2(3)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def summarize_param_memory(module: nn.Module) -> Mapping[str, float]:
    """Stub summary separating theoretical ternary vs dense storage.

    TODO: split ternary-eligible vs always-full-precision parameter groups.
    """
    n = count_parameters(module)
    return {
        "n_params": float(n),
        "fp32_bits": theoretical_weight_bits(n, bits_per_param=32),
        "fp16_bits": theoretical_weight_bits(n, bits_per_param=16),
        "ideal_ternary_bits": theoretical_weight_bits(n, bits_per_param=ternary_bits_per_param()),
    }


def peak_cuda_memory_bytes() -> int | None:
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())
