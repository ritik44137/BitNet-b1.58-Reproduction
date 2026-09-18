"""Evaluate a checkpoint: loss / perplexity on a split."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from src.data import get_batch, load_processed_split, load_tokenizer
from src.models.build import build_model_from_config
from src.utils.config import load_config, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None, help="Optional config override.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = load_config(args.config) if args.config else ckpt["config"]

    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    device = resolve_device(str(train_cfg.get("device", "cpu")))
    data_dir = Path(data_cfg["data_dir"])
    tokenizer = load_tokenizer(data_dir)
    data = load_processed_split(data_dir, args.split)
    block_size = int(cfg["model"].get("block_size", data_cfg.get("seq_len", 256)))
    batch_size = int(args.batch_size or train_cfg["batch_size"])

    model = build_model_from_config(cfg, vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    losses = []
    for _ in range(args.eval_iters):
        x, y = get_batch(data, batch_size, block_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())

    mean_loss = sum(losses) / len(losses)
    ppl = math.exp(mean_loss)
    print(
        f"checkpoint={ckpt_path} split={args.split} "
        f"loss={mean_loss:.4f} perplexity={ppl:.2f} iters={args.eval_iters}"
    )


if __name__ == "__main__":
    main()
