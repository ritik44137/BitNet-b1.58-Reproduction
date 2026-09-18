"""Character-level language modeling dataset utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass
class CharTokenizer:
    """Simple char-level tokenizer with stable stoi/itos maps."""

    chars: list[str]

    def __post_init__(self) -> None:
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.itos[int(i)] for i in ids)

    def to_meta(self) -> dict[str, Any]:
        return {"chars": self.chars, "vocab_size": self.vocab_size}

    @classmethod
    def from_meta(cls, meta: dict[str, Any]) -> "CharTokenizer":
        return cls(chars=list(meta["chars"]))

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(chars=sorted(set(text)))


class CharLMDataset(Dataset):
    """Fixed-length next-token windows over a 1-D token tensor."""

    def __init__(self, tokens: torch.Tensor, block_size: int) -> None:
        if tokens.ndim != 1:
            raise ValueError("tokens must be a 1-D tensor")
        if len(tokens) <= block_size:
            raise ValueError(
                f"Need more than block_size tokens; got {len(tokens)} <= {block_size}"
            )
        self.tokens = tokens.long()
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.tokens[idx : idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def load_processed_split(data_dir: str | Path, split: str) -> torch.Tensor:
    path = Path(data_dir) / f"{split}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python data/prepare_dataset.py"
        )
    return torch.load(path, map_location="cpu", weights_only=True)


def load_tokenizer(data_dir: str | Path) -> CharTokenizer:
    meta_path = Path(data_dir) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {meta_path}. Run: python data/prepare_dataset.py"
        )
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return CharTokenizer.from_meta(meta)


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of contiguous token windows."""
    max_start = data.size(0) - block_size - 1
    if max_start < 0:
        raise ValueError("Token tensor shorter than block_size + 1")
    ix = torch.randint(0, max_start + 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)
