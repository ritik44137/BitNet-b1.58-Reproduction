#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

bench() {
  local config="$1"
  local ckpt="$2"
  local args=(--config "$config")
  if [[ -f "$ckpt" ]]; then
    args+=(--checkpoint "$ckpt")
  fi
  python -m src.benchmark "${args[@]}"
}

echo "==> Benchmark baseline"
bench configs/baseline_tiny.yaml runs/baseline_tiny/ckpt_best.pt

echo "==> Benchmark BitNet"
bench configs/bitnet_tiny.yaml runs/bitnet_tiny/ckpt_best.pt
