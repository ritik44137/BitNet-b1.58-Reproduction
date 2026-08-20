# BitNet b1.58 Reproduction from Scratch

This repo implements the **core algorithmic idea** from [*The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*](https://arxiv.org/abs/2402.17764): replace standard linear layers with a custom **BitLinear** layer whose effective weights are constrained to **{-1, 0, +1}** via **absmean ternary quantization**, trained with a **straight-through estimator (STE)**.

## What this repo is (and is not)

**Aims to show**

- absmean ternary quantization
- a drop-in `BitLinear` module
- a controlled baseline vs BitNet-style comparison
- honest reporting of loss, theoretical memory, runtime, and samples
- a clear faithful-vs-simplified writeup

**Does not claim**

- packed 1.58-bit storage in the PyTorch path
- paper-level low-bit throughput/latency from standard `torch.matmul`
- that a tiny run proves large-scale LLM scaling behavior

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Suggested workflow

```bash
# Prepare data
python data/prepare_dataset.py --config configs/baseline_tiny.yaml

# Train
bash scripts/run_baseline.sh
bash scripts/run_bitnet.sh

# Evaluate / benchmark / sample
python -m src.evaluate --checkpoint path/to/checkpoint.pt --split val
python -m src.benchmark --config configs/baseline_tiny.yaml
python -m src.sample --checkpoint path/to/checkpoint.pt --prompt "Once upon a time"

# Or run all benchmarks
bash scripts/benchmark_all.sh
```

## Repository layout

```
.
├── README.md
├── requirements.txt
├── configs/
│   ├── baseline_tiny.yaml
│   └── bitnet_tiny.yaml
├── data/
│   └── prepare_dataset.py
├── src/
│   ├── layers/
│   │   ├── quantization.py
│   │   └── bitlinear.py
│   ├── models/
│   │   ├── transformer_baseline.py
│   │   └── transformer_bitnet.py
│   ├── train.py
│   ├── evaluate.py
│   ├── benchmark.py
│   ├── sample.py
│   └── utils/
├── scripts/
├── results/
└── tests/
```

## Core idea

```
s = mean(abs(W))
Q = clip(round(W / s), -1, 1)
W_q = s * Q
```

STE training pattern:

```python
w_eff = w + (quantize(w) - w).detach()
```

## References

- Ma et al., [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764)
- [Oxen.ai ArXiv Dives walkthrough](https://www.oxen.ai/blog/arxiv-dives-bitnet-1-58)

## Status

Skeleton only — modules are stubs. Implement in this order:

1. `src/layers/quantization.py`
2. `src/layers/bitlinear.py`
3. sanity tests
4. Transformer linear-factory integration
5. training / eval / benchmark pipelines
