"""Evaluate a checkpoint: loss / optional perplexity on a split."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None, help="Optional config override.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    # TODO: load model + data, compute mean loss / perplexity, write results/tables
    raise NotImplementedError(f"Evaluation not implemented yet (checkpoint={ckpt}).")


if __name__ == "__main__":
    main()
