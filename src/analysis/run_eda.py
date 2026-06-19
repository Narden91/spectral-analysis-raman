"""Public API entry point for the Raman EDA pipeline."""

from __future__ import annotations

from src.analysis._single_range import _run_single_range_eda as run_eda

__all__ = ["run_eda"]
