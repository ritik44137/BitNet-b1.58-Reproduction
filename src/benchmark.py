"""Training and inference throughput benchmarks."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from src.data import get_batch, load_processed_split, load_tokenizer
from src.models.build import build_model_from_config
from src.utils.config import load_config, resolve_device
from src.utils.memory import peak_cuda_memory_bytes, summarize_param_memory
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark training and inference throughput.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=50)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--gen-tokens", type=int, default=64)
    parser.add_argument("--infer-trials", type=int, default=5)
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()


def hardware_info(device: str) -> dict:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
    }
    if device == "cuda" and torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda_capability"] = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
    else:
        info["cpu"] = platform.processor() or platform.machine()
    return info


@torch.no_grad()
def benchmark_inference(
    model: torch.nn.Module,
    vocab_size: int,
    prompt_tokens: int,
    gen_tokens: int,
    trials: int,
    device: str,
) -> dict:
    model.eval()
    latencies = []
    for _ in range(trials):
        idx = torch.randint(0, vocab_size, (1, prompt_tokens), device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model.generate(idx, max_new_tokens=gen_tokens, temperature=1.0, top_k=None)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)
    mean_s = sum(latencies) / len(latencies)
    return {
        "infer_latency_s": mean_s,
        "infer_tokens_per_sec": gen_tokens / mean_s,
        "infer_trials": trials,
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen_tokens,
    }


def benchmark_train_step(
    model: torch.nn.Module,
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str,
    warmup_steps: int,
    measure_steps: int,
) -> dict:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for _ in range(warmup_steps):
        x, y = get_batch(data, batch_size, block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(measure_steps):
        x, y = get_batch(data, batch_size, block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tokens = measure_steps * batch_size * block_size
    peak = peak_cuda_memory_bytes()
    return {
        "train_step_ms": (elapsed / measure_steps) * 1000.0,
        "train_tokens_per_sec": tokens / elapsed,
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "peak_runtime_memory_bytes": peak,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["train"].get("seed", 42)))

    device = resolve_device(str(cfg["train"].get("device", "cpu")))
    data_dir = Path(cfg["data"]["data_dir"])
    tokenizer = load_tokenizer(data_dir)
    train_data = load_processed_split(data_dir, "train")
    block_size = int(cfg["model"].get("block_size", cfg["data"].get("seq_len", 256)))
    batch_size = int(cfg["train"]["batch_size"])

    model = build_model_from_config(cfg, vocab_size=tokenizer.vocab_size).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])

    mem = summarize_param_memory(model)
    train_stats = benchmark_train_step(
        model,
        train_data,
        batch_size=batch_size,
        block_size=block_size,
        device=device,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
    )
    infer_stats = benchmark_inference(
        model,
        vocab_size=tokenizer.vocab_size,
        prompt_tokens=args.prompt_tokens,
        gen_tokens=args.gen_tokens,
        trials=args.infer_trials,
        device=device,
    )

    result = {
        "model": cfg["model"].get("name", "model"),
        "config": args.config,
        "hardware": hardware_info(device),
        "params": mem,
        **train_stats,
        **infer_stats,
    }

    out_path = Path(args.out or f"results/tables/{result['model']}_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
