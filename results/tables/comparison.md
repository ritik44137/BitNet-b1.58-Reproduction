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
