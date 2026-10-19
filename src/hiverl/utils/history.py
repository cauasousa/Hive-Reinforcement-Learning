"""Incremental CSV logging of training history."""
from __future__ import annotations

import csv
import os
import time


def _existing_header(path: str) -> list[str] | None:
    with open(path, newline="") as f:
        return next(csv.reader(f), None)


def _rotate_aside(path: str) -> None:
    """Rename a history file whose header no longer matches."""
    root, ext = os.path.splitext(path)
    rotated = f"{root}.{time.strftime('%Y%m%d_%H%M%S')}{ext}"
    if os.path.exists(rotated):
        rotated = f"{root}.{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}{ext}"
    os.rename(path, rotated)


def append_history_row(path: str, row: dict) -> None:
    """Append one row (writing the header if needed)."""
    if os.path.exists(path) and _existing_header(path) != list(row.keys()):
        _rotate_aside(path)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_history_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))
