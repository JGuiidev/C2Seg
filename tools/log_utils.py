# -*- log_utils.py -*-
"""Logging and progress display utilities for C2Seg."""

from __future__ import annotations

import logging
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List



ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

class MaxLevelFilter(logging.Filter):
    """Allow log records with level <= max_level."""

    def __init__(self, max_level: int) -> None:
        super().__init__()
        self.max_level = int(max_level)

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level


class MinLevelFilter(logging.Filter):
    """Allow log records with level >= min_level."""

    def __init__(self, min_level: int) -> None:
        super().__init__()
        self.min_level = int(min_level)

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= self.min_level


def create_logger(
    log_path: str | Path,
    *,
    logger_name: str = "C2Seg",
    level: int = logging.INFO,
    console: bool = True,
    append: bool = True,
    file_time: bool = True,
) -> logging.Logger:
    """Create a logger for C2Seg."""
    log_path = Path(log_path).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.propagate = False
    logger.setLevel(level)
    logger.handlers.clear()

    file_mode = "a" if append else "w"
    file_fmt = "%(asctime)s | %(levelname)s | %(message)s" if file_time else "%(levelname)s | %(message)s"
    file_handler = logging.FileHandler(log_path, mode=file_mode, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(file_fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    if console:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(level)
        stdout_handler.addFilter(MaxLevelFilter(logging.INFO))
        stdout_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stdout_handler)

        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.addFilter(MinLevelFilter(logging.WARNING))
        stderr_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
        logger.addHandler(stderr_handler)

    return logger


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def visible_len(text: str) -> int:
    return len(strip_ansi(text))


def pad_visible(text: str, width: int) -> str:
    return text + (" " * max(0, width - visible_len(text)))


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def build_ascii_table(headers: List[str], rows: List[List[str]]) -> str:
    """Build a simple ASCII table."""
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], visible_len(str(cell)))

    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    header_line = "| " + " | ".join(pad_visible(header, widths[idx]) for idx, header in enumerate(headers)) + " |"
    lines = [border, header_line, border]
    for row in rows:
        lines.append("| " + " | ".join(pad_visible(str(cell), widths[idx]) for idx, cell in enumerate(row)) + " |")
    lines.append(border)
    return "\n".join(lines)


def format_epoch_lines(
    *,
    epoch: int,
    total_epochs: int,
    train_metrics: Dict[str, float],
    valid_metrics: Dict[str, float],
    best_dice: float,
    best_epoch: int,
    improved: bool,
    patience: int,
    patience_limit: int,
) -> List[str]:
    """Return exactly three concise lines for one segmentation epoch."""
    epoch_tag = f"Epoch {epoch:03d}/{total_epochs:03d}"

    def metric_line(phase: str, values: Dict[str, float]) -> str:
        loss = safe_float(values.get("Loss"))
        dice = safe_float(values.get("Dice"))
        miou = safe_float(values.get("mIoU"))
        lr = safe_float(values.get("LR"))
        elapsed = format_seconds(values.get("EpochTimeSec", 0.0))
        lr_text = f"{lr:.2e}" if math.isfinite(lr) else "-"
        return (
            f"{epoch_tag} | {phase:<5} | Loss={loss:.5f} | Dice={dice:.4f} | "
            f"mIoU={miou:.4f} | LR={lr_text} | Time={elapsed}"
        )

    status = "YES" if improved else "NO"
    third = (
        f"Best Val Dice={best_dice:.5f} | Improved={status} | "
        f"BestEpoch={best_epoch:03d} | Patience={patience}/{patience_limit}"
    )
    return [metric_line("Train", train_metrics), metric_line("Val", valid_metrics), third]


__all__ = [
    "create_logger",
    "format_epoch_lines",
    "build_ascii_table",
    "format_seconds",
    "safe_float",
]
