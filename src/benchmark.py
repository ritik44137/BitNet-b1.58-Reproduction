"""Runtime benchmarks: train step time, tokens/sec, inference latency, peak memory."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark training and inference throughput.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=50)
    parser.add_argument("--gen-tokens", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    # TODO: warmup + timed train steps; timed generation; write results/tables
    raise NotImplementedError(f"Benchmark harness not implemented yet (config={config_path}).")


if __name__ == "__main__":
    main()
