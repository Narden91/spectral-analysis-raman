"""Small I/O helpers for deterministic analysis artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

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


def write_parquet_atomic(frame: pd.DataFrame, path: str | Path, **kwargs: Any) -> Path:
    """Write a Parquet table via a same-directory temporary file."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temp_path = resolved.with_name(f".{resolved.stem}.tmp{resolved.suffix}")
    frame.to_parquet(temp_path, **kwargs)
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


def write_json_atomic(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    """Write JSON deterministically through the shared atomic text writer."""
    text = json.dumps(payload, indent=indent, sort_keys=True) + "\n"
    return write_text_atomic(path, text)
