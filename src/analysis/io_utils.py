"""Small I/O helpers for deterministic analysis artifacts."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _atomic_temp_path(resolved: Path, *, keep_suffix: bool = False) -> Path:
    """Build a per-process, per-thread temporary path in the destination directory."""
    tag = f"{os.getpid()}_{threading.get_ident()}"
    if keep_suffix:
        return resolved.with_name(f".{resolved.stem}.{tag}.tmp{resolved.suffix}")
    return resolved.with_name(f".{resolved.name}.{tag}.tmp")


def _replace_atomic(temp_path: Path, target_path: Path, *, retries: int = 6) -> None:
    """Atomically replace ``target_path`` with brief backoff for transient Windows replace locks."""
    for attempt in range(retries):
        try:
            os.replace(temp_path, target_path)
            return
        except PermissionError:
            if attempt >= retries - 1:
                raise
            time.sleep(0.005 * (2**attempt))


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
    temp_path = _atomic_temp_path(resolved)
    try:
        frame.to_csv(temp_path, **kwargs)
        _replace_atomic(temp_path, resolved)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
    return resolved


def write_parquet_atomic(frame: pd.DataFrame, path: str | Path, **kwargs: Any) -> Path:
    """Write a Parquet table via a same-directory temporary file."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _atomic_temp_path(resolved, keep_suffix=True)
    try:
        frame.to_parquet(temp_path, **kwargs)
        _replace_atomic(temp_path, resolved)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
    return resolved


def write_text_atomic(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Write text via temp file then replace, avoiding half-written artifacts."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _atomic_temp_path(resolved)
    try:
        temp_path.write_text(text, encoding=encoding)
        _replace_atomic(temp_path, resolved)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
    return resolved


def write_npz_atomic(path: str | Path, arrays: Mapping[str, Any]) -> Path:
    """Write a NumPy ``.npz`` archive via a same-directory temporary file."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _atomic_temp_path(resolved)
    try:
        with temp_path.open("wb") as handle:
            np.savez(handle, **arrays)
        _replace_atomic(temp_path, resolved)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
    return resolved


def write_json_atomic(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    """Write JSON deterministically through the shared atomic text writer."""
    text = json.dumps(payload, indent=indent, sort_keys=True) + "\n"
    return write_text_atomic(path, text)
