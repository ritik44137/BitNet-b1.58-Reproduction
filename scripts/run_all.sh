#!/usr/bin/env bash
# Prepare data, train both models, evaluate, sample, benchmark, and report.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

python data/prepare_dataset.py --config configs/baseline_tiny.yaml

bash scripts/run_baseline.sh
bash scripts/run_bitnet.sh

python -m src.evaluate --checkpoint runs/baseline_tiny/ckpt_best.pt --split val
python -m src.evaluate --checkpoint runs/bitnet_tiny/ckpt_best.pt --split val

python -m src.sample --checkpoint runs/baseline_tiny/ckpt_best.pt \
  --out results/samples/baseline_tiny_generations.txt
python -m src.sample --checkpoint runs/bitnet_tiny/ckpt_best.pt \
  --out results/samples/bitnet_tiny_generations.txt

bash scripts/benchmark_all.sh
python -m src.report
