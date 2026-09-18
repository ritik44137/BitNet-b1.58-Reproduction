# BitNet b1.58 Reproduction from Scratch

Small-scale reproduction of the core training idea from
[*The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*](https://arxiv.org/abs/2402.17764).

Train a tiny decoder-only Transformer on Tiny Shakespeare with ordinary
`nn.Linear` layers, then train the same architecture again with attention and
MLP projections swapped for `BitLinear`. On the forward pass, each weight
matrix is absmean-scaled and rounded to `{-1, 0, +1}`; a straight-through
estimator keeps gradients flowing to the full-precision master weights.

This repo targets the **algorithm**, not packed 1.58-bit kernels. Do not expect
the paper's wall-clock speedups from plain PyTorch.

Latest numbers: [RESULTS.md](RESULTS.md).

## Setup

```bash
pip install -r requirements.txt
export PYTHONPATH=.
python data/prepare_dataset.py
```

## Train

```bash
bash scripts/run_baseline.sh
bash scripts/run_bitnet.sh

# full pipeline: train → eval → sample → benchmark → plots
bash scripts/run_all.sh
```

Smoke configs: `configs/overfit_*.yaml`.

## Eval / sample / benchmark

```bash
python -m src.evaluate --checkpoint runs/baseline_tiny/ckpt_best.pt --split val
python -m src.evaluate --checkpoint runs/bitnet_tiny/ckpt_best.pt --split val

python -m src.sample --checkpoint runs/baseline_tiny/ckpt_best.pt
python -m src.sample --checkpoint runs/bitnet_tiny/ckpt_best.pt

bash scripts/benchmark_all.sh
python -m src.report
```

## BitLinear in one line

```python
w_eff = w + (quantize(w) - w).detach()  # forward: s*Q, backward: updates w
```

where `s = mean(|W|)` and `Q = clip(round(W / s), -1, 1)`.

Embeddings, LayerNorms, and the LM head stay full precision by default.

## Layout

```
configs/     baseline + BitNet (+ overfit) YAML
data/        Tiny Shakespeare download / tokenize
src/layers/  absmean ternary quant + BitLinear
src/models/  shared tiny Transformer (make_linear)
src/*.py     train / evaluate / sample / benchmark / report
scripts/     thin wrappers
tests/
results/     tables, figures, samples
```

## Scope

| Area | Match? |
| --- | --- |
| Ternary weights via absmean | yes |
| STE training | yes |
| Same-size dense baseline | yes |
| Large-scale LLM training | no |
| Packed ternary storage | no |
| Custom low-bit matmul | no |

## Tests

```bash
PYTHONPATH=. python -m pytest -q
```

## References

- Ma et al., [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764)
- [Oxen.ai BitLinear walkthrough](https://www.oxen.ai/blog/arxiv-dives-bitnet-1-58)

```bibtex
@article{ma2024era,
  title={The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits},
  author={Ma, Shuming and Wang, Hongyu and Ma, Lingxiao and Wang, Lei and Wang, Wenhui and Huang, Shaohan and Dong, Li and Wang, Ruiping and Xue, Jilong and Wei, Furu},
  journal={arXiv preprint arXiv:2402.17764},
  year={2024}
}
```
