"""Small I/O helpers for deterministic analysis artifacts."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def load_csv_or_empty(path: str | Path | None) -> pd.DataFrame:
    """Return an empty frame when an optional CSV is absent or empty."""
    if path is None:
        return pd.DataFrame()
    resolved = Path(path)
    if not resolved.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(resolved)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_csv_atomic(frame: pd.DataFrame, path: str | Path, **kwargs) -> Path:
    """Write a CSV via temp file then replace, avoiding half-written artifacts."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temp_path = resolved.with_name(f".{resolved.name}.tmp")
    frame.to_csv(temp_path, **kwargs)
    os.replace(temp_path, resolved)
    return resolved


def write_text_atomic(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Write text via temp file then replace, avoiding half-written artifacts."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temp_path = resolved.with_name(f".{resolved.name}.tmp")
    temp_path.write_text(text, encoding=encoding)
    os.replace(temp_path, resolved)
    return resolved
