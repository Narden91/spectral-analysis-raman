"""Minimal reporting helpers that avoid optional dependencies."""

from __future__ import annotations

import pandas as pd


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a markdown table without requiring ``tabulate``."""
    if df.empty:
        return ""
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join([header, separator, *rows])
