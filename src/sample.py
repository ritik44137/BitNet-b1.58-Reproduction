"""Generate qualitative samples from a checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data import load_tokenizer
from src.models.build import build_model_from_config
from src.utils.config import load_config, resolve_device
from src.utils.seed import set_seed

DEFAULT_PROMPTS = [
    "First Citizen:\n",
    "ROMEO:\n",
    "To be, or not to be,",
    "O, for a muse of fire",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample text from a checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = load_config(args.config) if args.config else ckpt["config"]
    device = resolve_device(str(cfg["train"].get("device", "cpu")))
    data_dir = Path(cfg["data"]["data_dir"])
    tokenizer = load_tokenizer(data_dir)

    model = build_model_from_config(cfg, vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS
    lines: list[str] = []
    model_name = cfg["model"].get("name", "model")
    lines.append(f"# samples from {args.checkpoint} ({model_name})")
    lines.append("")

    for prompt in prompts:
        idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        out = model.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        text = tokenizer.decode(out[0])
        block = f"## prompt\n{prompt}\n\n## completion\n{text}\n"
        lines.append(block)
        print(block)

    out_path = Path(
        args.out
        or f"results/samples/{model_name}_generations.txt"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
