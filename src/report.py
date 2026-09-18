"""Aggregate metrics, write comparison tables, and plot loss curves."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build result tables and loss plots.")
    parser.add_argument(
        "--baseline-metrics",
        type=str,
        default="runs/baseline_tiny/logs/metrics.jsonl",
    )
    parser.add_argument(
        "--bitnet-metrics",
        type=str,
        default="runs/bitnet_tiny/logs/metrics.jsonl",
    )
    parser.add_argument(
        "--baseline-benchmark",
        type=str,
        default="results/tables/baseline_tiny_benchmark.json",
    )
    parser.add_argument(
        "--bitnet-benchmark",
        type=str,
        default="results/tables/bitnet_tiny_benchmark.json",
    )
    parser.add_argument("--out-dir", type=str, default="results")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def last_with_key(rows: list[dict[str, Any]], key: str) -> float | None:
    for row in reversed(rows):
        if key in row:
            return float(row[key])
    return None


def preferred_train_loss(rows: list[dict[str, Any]]) -> float | None:
    return last_with_key(rows, "train_loss_eval") or last_with_key(rows, "train_loss")


def format_bytes(n_bits: float) -> str:
    mb = n_bits / 8.0 / 1e6
    return f"{mb:.2f} MB"


def write_comparison_table(
    baseline_metrics: list[dict[str, Any]],
    bitnet_metrics: list[dict[str, Any]],
    baseline_bench: dict[str, Any],
    bitnet_bench: dict[str, Any],
    out_path: Path,
) -> str:
    rows = []
    for name, metrics, bench in [
        ("Baseline FP", baseline_metrics, baseline_bench),
        ("BitNet b1.58 Tiny", bitnet_metrics, bitnet_bench),
    ]:
        train_loss = preferred_train_loss(metrics)
        val_loss = last_with_key(metrics, "val_loss")
        ppl = math.exp(val_loss) if val_loss is not None else None
        params = bench["params"]
        theoretical = (
            format_bytes(params["reported_weight_bits"])
            if "reported_weight_bits" in params
            else format_bytes(params["fp32_bits"])
        )
        peak = bench.get("peak_runtime_memory_bytes")
        peak_str = f"{peak / 1e6:.1f} MB" if peak else "n/a (CPU)"
        rows.append(
            {
                "Model": name,
                "Params": f"{int(params['n_params']):,}",
                "Train Loss": f"{train_loss:.4f}" if train_loss is not None else "n/a",
                "Val Loss": f"{val_loss:.4f}" if val_loss is not None else "n/a",
                "Perplexity": f"{ppl:.2f}" if ppl is not None else "n/a",
                "Theoretical Weight Memory": theoretical,
                "Peak Runtime Memory": peak_str,
                "Train Tokens/sec": f"{bench['train_tokens_per_sec']:.0f}",
                "Inference tok/sec": f"{bench['infer_tokens_per_sec']:.1f}",
            }
        )

    headers = list(rows[0].keys())
    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        md_lines.append("| " + " | ".join(row[h] for h in headers) + " |")

    fidelity = """
| Area | Faithful to paper? | Notes |
| --- | --- | --- |
| Ternary weights | Yes | Effective weights in {-1, 0, +1} |
| Absmean scaling | Yes | Per-tensor absmean during forward quantize |
| Straight-through training | Yes | Gradients update full-precision master weights |
| Large-scale training | No | Tiny model and Tiny Shakespeare only |
| Packed ternary storage | No | Standard PyTorch float tensors |
| Custom low-bit kernels | No | Dense matmul path only |
""".strip()

    body = "\n".join(md_lines) + "\n\n### Fidelity\n\n" + fidelity + "\n"
    out_path.write_text(body, encoding="utf-8")

    csv_path = out_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(row[h] for h in headers) + "\n")
    return body


def plot_loss_curves(
    baseline_metrics: list[dict[str, Any]],
    bitnet_metrics: list[dict[str, Any]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, rows, color in [
        ("baseline train", baseline_metrics, "#1f77b4"),
        ("bitnet train", bitnet_metrics, "#d62728"),
    ]:
        steps = [r["step"] for r in rows if "train_loss" in r]
        losses = [r["train_loss"] for r in rows if "train_loss" in r]
        ax.plot(steps, losses, label=label, color=color, alpha=0.85, linewidth=1.2)

    for label, rows, color in [
        ("baseline val", baseline_metrics, "#1f77b4"),
        ("bitnet val", bitnet_metrics, "#d62728"),
    ]:
        steps = [r["step"] for r in rows if "val_loss" in r]
        losses = [r["val_loss"] for r in rows if "val_loss" in r]
        if steps:
            ax.plot(
                steps,
                losses,
                label=label,
                color=color,
                linestyle="--",
                marker="o",
                markersize=3,
            )

    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Baseline vs BitNet tiny training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    tables = out_dir / "tables"
    figures = out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    baseline_metrics = load_jsonl(Path(args.baseline_metrics))
    bitnet_metrics = load_jsonl(Path(args.bitnet_metrics))
    baseline_bench = json.loads(Path(args.baseline_benchmark).read_text(encoding="utf-8"))
    bitnet_bench = json.loads(Path(args.bitnet_benchmark).read_text(encoding="utf-8"))

    table_path = tables / "comparison.md"
    body = write_comparison_table(
        baseline_metrics, bitnet_metrics, baseline_bench, bitnet_bench, table_path
    )
    plot_path = figures / "loss_curves.png"
    plot_loss_curves(baseline_metrics, bitnet_metrics, plot_path)

    results_md = Path("RESULTS.md")
    if results_md.exists():
        text = results_md.read_text(encoding="utf-8")
        marker = "## Comparison"
        if marker in text:
            hardware = baseline_bench.get("hardware", {})
            hw_line = (
                f"- device: `{hardware.get('device', 'unknown')}`\n"
                f"- platform: `{hardware.get('platform', 'unknown')}`\n"
                f"- torch: `{hardware.get('torch', 'unknown')}`\n"
            )
            if "gpu_name" in hardware:
                hw_line += f"- gpu: `{hardware['gpu_name']}`\n"
            elif "cpu" in hardware:
                hw_line += f"- cpu: `{hardware['cpu']}`\n"

            new_section = (
                "## Comparison\n\n"
                f"{body}\n"
                f"Loss curves: `results/figures/loss_curves.png`\n\n"
                "## Hardware\n\n"
                f"{hw_line}\n"
                "## Notes\n\n"
                "- Runtime memory on CPU is not reported via CUDA peak stats.\n"
                "- BitNet being slower here is expected: quantization runs in eager "
                "PyTorch without packed ternary kernels.\n"
                "- Qualitative samples are illustrative only.\n"
            )
            # Replace from Comparison through end (report owns the trailing sections).
            head = text.split(marker, 1)[0].rstrip() + "\n\n"
            results_md.write_text(head + new_section, encoding="utf-8")
            print(f"updated {results_md}")

    print(body)
    print(f"wrote {table_path}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
