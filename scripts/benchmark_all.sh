#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Benchmark baseline"
python -m src.benchmark --config configs/baseline_tiny.yaml "$@"

echo "==> Benchmark BitNet"
python -m src.benchmark --config configs/bitnet_tiny.yaml "$@"
