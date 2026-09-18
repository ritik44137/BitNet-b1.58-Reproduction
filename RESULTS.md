# Results

Controlled comparison with `configs/baseline_tiny.yaml` and
`configs/bitnet_tiny.yaml`:

- model: 4 layers, 256-d, 4 heads, block size 256 (~3.26M params)
- data: Tiny Shakespeare, char-level, 90/10 split
- train: 2000 steps, batch 32, AdamW, seed 42, fp32 on CPU

Regenerate artifacts with:

```bash
bash scripts/benchmark_all.sh
python -m src.report
```

## Comparison

| Model | Params | Train Loss | Val Loss | Perplexity | Theoretical Weight Memory | Peak Runtime Memory | Train Tokens/sec | Inference tok/sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline FP | 3,258,368 | 1.5805 | 1.7519 | 5.77 | 13.03 MB | n/a (CPU) | 5822 | 168.2 |
| BitNet b1.58 Tiny | 3,258,368 | 1.8195 | 1.9334 | 6.91 | 1.07 MB | n/a (CPU) | 5698 | 86.0 |

### Fidelity

| Area | Faithful to paper? | Notes |
| --- | --- | --- |
| Ternary weights | Yes | Effective weights in {-1, 0, +1} |
| Absmean scaling | Yes | Per-tensor absmean during forward quantize |
| Straight-through training | Yes | Gradients update full-precision master weights |
| Large-scale training | No | Tiny model and Tiny Shakespeare only |
| Packed ternary storage | No | Standard PyTorch float tensors |
| Custom low-bit kernels | No | Dense matmul path only |

Loss curves: [`results/figures/loss_curves.png`](results/figures/loss_curves.png)

Samples: [`results/samples/`](results/samples/)

## Takeaways

Both models train stably from the same recipe. The dense baseline ends ahead on
validation loss (~1.75 vs ~1.93), which is reasonable at this scale and token
budget. Theoretical weight storage drops sharply for BitNet if ternary weights
were packed (~1.07 MB vs ~13 MB fp32), but training still keeps fp32 master
weights, so that compression does not show up as lower runtime memory here.

On this CPU PyTorch path, BitNet is a bit slower to train and about 2× slower
at inference — expected without packed ternary kernels.

## Hardware

- device: `cpu`
- platform: `Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.35`
- torch: `2.13.0+cpu`
- cpu: `x86_64`

Raw benchmark JSON also lives under `results/tables/`.
