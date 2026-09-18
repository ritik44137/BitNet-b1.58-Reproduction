"""Train baseline or BitNet tiny Transformer from a YAML config."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import torch

from src.data import get_batch, load_processed_split, load_tokenizer
from src.models.build import build_model_from_config
from src.utils.config import load_config, resolve_device
from src.utils.logging import append_jsonl, setup_logger
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tiny Transformer (baseline or BitNet).")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint path.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override train.max_steps (useful for smoke/overfit runs).",
    )
    return parser.parse_args()


def get_lr(step: int, warmup_steps: int, max_steps: int, learning_rate: float) -> float:
    if step < warmup_steps:
        return learning_rate * (step + 1) / max(warmup_steps, 1)
    if step >= max_steps:
        return learning_rate * 0.1
    # Cosine decay to 10% of peak LR
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return learning_rate * (0.1 + 0.9 * coeff)


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    splits: dict[str, torch.Tensor],
    batch_size: int,
    block_size: int,
    device: str,
    eval_iters: int = 20,
) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for split_name, data in splits.items():
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split_name] = losses.mean().item()
    model.train()
    return out


def bitlinear_quant_stats(model: torch.nn.Module) -> dict[str, float]:
    from src.layers.bitlinear import BitLinear

    scales: list[float] = []
    neg = zero = pos = total = 0
    for module in model.modules():
        if not isinstance(module, BitLinear):
            continue
        q, scale, _ = module.quantized_weight()
        scales.append(float(scale.item()))
        flat = q.detach().view(-1)
        total += flat.numel()
        neg += int((flat == -1).sum().item())
        zero += int((flat == 0).sum().item())
        pos += int((flat == 1).sum().item())
    if total == 0:
        return {}
    return {
        "bitlinear_scale_mean": sum(scales) / len(scales),
        "bitlinear_neg_frac": neg / total,
        "bitlinear_zero_frac": zero / total,
        "bitlinear_pos_frac": pos / total,
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: dict[str, Any],
    best_val: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": cfg,
            "best_val": best_val,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    log_cfg = cfg.get("logging", {})

    if args.max_steps is not None:
        train_cfg["max_steps"] = args.max_steps

    set_seed(int(train_cfg.get("seed", 42)))
    device = resolve_device(str(train_cfg.get("device", "cpu")))
    out_dir = Path(train_cfg["out_dir"])
    log_dir = Path(log_cfg.get("log_dir", out_dir / "logs"))
    logger = setup_logger("bitnet", log_dir)
    metrics_path = log_dir / "metrics.jsonl"
    if args.resume is None and metrics_path.exists():
        metrics_path.unlink()

    data_dir = Path(data_cfg["data_dir"])
    tokenizer = load_tokenizer(data_dir)
    train_data = load_processed_split(data_dir, "train")
    val_data = load_processed_split(data_dir, "val")
    block_size = int(cfg["model"].get("block_size", data_cfg.get("seq_len", 256)))
    batch_size = int(train_cfg["batch_size"])
    max_steps = int(train_cfg["max_steps"])
    eval_interval = int(train_cfg.get("eval_interval", 100))
    learning_rate = float(train_cfg["learning_rate"])
    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    model = build_model_from_config(cfg, vocab_size=tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(float(train_cfg.get("beta1", 0.9)), float(train_cfg.get("beta2", 0.95))),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    start_step = 0
    best_val = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        best_val = float(ckpt.get("best_val", best_val))
        logger.info("Resumed from %s at step %d", args.resume, start_step)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model=%s params=%.2fM device=%s vocab=%d block=%d",
        cfg["model"].get("name", "model"),
        n_params / 1e6,
        device,
        tokenizer.vocab_size,
        block_size,
    )

    model.train()
    t0 = time.time()
    for step in range(start_step, max_steps):
        lr = get_lr(step, warmup_steps, max_steps, learning_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = get_batch(train_data, batch_size, block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        step_time_ms = (time.time() - t0) * 1000.0
        t0 = time.time()
        tokens_per_sec = (batch_size * block_size) / max(step_time_ms / 1000.0, 1e-8)

        record: dict[str, Any] = {
            "step": step,
            "train_loss": float(loss.item()),
            "learning_rate": lr,
            "grad_norm": float(grad_norm),
            "step_time_ms": step_time_ms,
            "tokens_per_sec": tokens_per_sec,
        }

        if bool(log_cfg.get("log_quantization_stats", False)):
            record.update(bitlinear_quant_stats(model))

        if step % 10 == 0 or step == max_steps - 1:
            logger.info(
                "step %d | loss %.4f | lr %.2e | %.0f tok/s",
                step,
                loss.item(),
                lr,
                tokens_per_sec,
            )

        do_eval = (step > 0 and step % eval_interval == 0) or step == max_steps - 1
        if do_eval:
            losses = estimate_loss(
                model,
                {"train": train_data, "val": val_data},
                batch_size=batch_size,
                block_size=block_size,
                device=device,
            )
            record["val_loss"] = losses["val"]
            record["train_loss_eval"] = losses["train"]
            logger.info(
                "eval step %d | train %.4f | val %.4f",
                step,
                losses["train"],
                losses["val"],
            )
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(out_dir / "ckpt_best.pt", model, optimizer, step, cfg, best_val)

        append_jsonl(metrics_path, record)

        if (step + 1) % max(eval_interval, 1) == 0 or step == max_steps - 1:
            save_checkpoint(out_dir / "ckpt_last.pt", model, optimizer, step + 1, cfg, best_val)

    logger.info("Done. best_val=%.4f checkpoints in %s", best_val, out_dir)


if __name__ == "__main__":
    main()
