"""Train baseline or BitNet tiny Transformer from a YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tiny Transformer (baseline or BitNet).")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    # TODO: load config, set seed, build model/data, train loop, log metrics, save ckpt
    raise NotImplementedError(f"Training loop not implemented yet (config={config_path}).")


if __name__ == "__main__":
    main()
