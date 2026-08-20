"""Download and prepare a modest language-modeling dataset.

Default target: Tiny Shakespeare (or another small corpus configured in YAML).
"""

from __future__ import annotations

import argparse
from pathlib import Path


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
        default="data/processed",
        help="Directory for tokenized / split artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # TODO: load config, download corpus, tokenize, write train/val splits
    raise NotImplementedError(
        f"Dataset preparation not implemented yet (config={args.config}, out={out_dir})."
    )


if __name__ == "__main__":
    main()
