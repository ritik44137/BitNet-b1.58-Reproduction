"""Download and prepare Tiny Shakespeare for character-level LM."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import torch
import yaml

# so `python data/prepare_dataset.py` works from the repo root
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.char_lm import CharTokenizer

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare dataset for BitNet repro.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_tiny.yaml",
        help="Path to YAML config with a data section.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory for tokenized / split artifacts (overrides config data_dir).",
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        default="data/raw/tinyshakespeare.txt",
        help="Where to cache the downloaded raw text.",
    )
    return parser.parse_args()


def download_text(url: str, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest.read_text(encoding="utf-8")
    print(f"Downloading {url} -> {dest}")
    with urllib.request.urlopen(url) as resp:
        text = resp.read().decode("utf-8")
    dest.write_text(text, encoding="utf-8")
    return text


def prepare_tinyshakespeare(
    text: str,
    out_dir: Path,
    train_split: float = 0.9,
) -> dict:
    tokenizer = CharTokenizer.from_text(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(train_split * len(ids))
    train_ids = ids[:n]
    val_ids = ids[n:]

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(train_ids, out_dir / "train.pt")
    torch.save(val_ids, out_dir / "val.pt")

    meta = {
        **tokenizer.to_meta(),
        "dataset": "tinyshakespeare",
        "train_tokens": int(train_ids.numel()),
        "val_tokens": int(val_ids.numel()),
        "train_split": train_split,
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    dataset = data_cfg.get("dataset", "tinyshakespeare")
    if dataset != "tinyshakespeare":
        raise ValueError(f"Unsupported dataset: {dataset}")

    out_dir = Path(args.out_dir or data_cfg.get("data_dir", "data/processed"))
    train_split = float(data_cfg.get("train_split", 0.9))

    text = download_text(TINY_SHAKESPEARE_URL, Path(args.raw_path))
    meta = prepare_tinyshakespeare(text, out_dir, train_split=train_split)
    print(
        f"Wrote {out_dir}/train.pt ({meta['train_tokens']} tokens), "
        f"val.pt ({meta['val_tokens']} tokens), vocab_size={meta['vocab_size']}"
    )


if __name__ == "__main__":
    main()
