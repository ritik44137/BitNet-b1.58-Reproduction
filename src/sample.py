"""Generate qualitative samples from a checkpoint with a fixed prompt set."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_PROMPTS = [
    "Once upon a time",
    "User: Hello\nAssistant:",
    "## Shopping list\n- ",
    "In a distant galaxy,",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample text from a checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt; else use defaults.")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--out", type=str, default="results/samples/generations.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS
    # TODO: load model, generate for each prompt, write side-by-side samples
    raise NotImplementedError(
        f"Sampling not implemented yet (checkpoint={ckpt}, n_prompts={len(prompts)})."
    )


if __name__ == "__main__":
    main()
