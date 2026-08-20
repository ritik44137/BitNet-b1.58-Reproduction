"""Training / eval metric logging helpers."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Mapping


def setup_logger(name: str = "bitnet", log_dir: str | Path | None = None) -> logging.Logger:
    """Create a stdout logger; optionally also write to ``log_dir/train.log``."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)

    if log_dir is not None:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path / "train.log")
        file_handler.setFormatter(handler.formatter)
        logger.addHandler(file_handler)

    return logger


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(record)) + "\n")


def append_csv(path: str | Path, record: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(dict(record))
