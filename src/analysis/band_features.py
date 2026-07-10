"""Shared band-integral and named-index helpers for spectral feature panels."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def block_columns(columns: Iterable[str], block: str) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    prefix = f"{block}__"
    for column in columns:
        if not str(column).startswith(prefix):
            continue
        try:
            wavenumber = float(str(column).split("__", 1)[1])
        except ValueError:
            continue
        out.append((str(column), wavenumber))
    return sorted(out, key=lambda item: item[1])


def _select_window_columns(
    columns: Iterable[str],
    block: str,
    lower: float,
    upper: float,
) -> tuple[list[str], np.ndarray]:
    selected = [
        (column, wavenumber)
        for column, wavenumber in block_columns(columns, block)
        if lower <= wavenumber <= upper
    ]
    if not selected:
        return [], np.array([], dtype=float)
    selected_columns = [column for column, _ in selected]
    wavenumbers = np.array([wavenumber for _, wavenumber in selected], dtype=float)
    return selected_columns, wavenumbers


def band_integral(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    block: str,
    lower: float,
    upper: float,
    min_channels: int = 3,
    min_coverage: float = 0.8,
) -> pd.Series:
    selected_columns, wavenumbers = _select_window_columns(columns, block, lower, upper)
    requested_span = max(float(upper) - float(lower), 0.0)
    represented_span = float(wavenumbers[-1] - wavenumbers[0]) if wavenumbers.size > 1 else 0.0
    coverage = represented_span / requested_span if requested_span > 0 else 1.0
    if len(selected_columns) < min_channels or coverage < min_coverage:
        return pd.Series(np.nan, index=frame.index, dtype=float)

    values = frame[selected_columns].to_numpy(dtype=float)
    if values.shape[1] == 1:
        integrated = values[:, 0]
    else:
        integrated = np.trapz(values, wavenumbers, axis=1)
    return pd.Series(integrated, index=frame.index, dtype=float)


def band_mean(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    block: str,
    lower: float,
    upper: float,
) -> pd.Series:
    selected_columns, _ = _select_window_columns(columns, block, lower, upper)
    if not selected_columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return frame[selected_columns].mean(axis=1)


def count_window_channels(columns: Iterable[str], *, block: str, lower: float, upper: float) -> int:
    selected_columns, _ = _select_window_columns(columns, block, lower, upper)
    return len(selected_columns)


def band_median_snr(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    block: str,
    lower: float,
    upper: float,
) -> float:
    selected_columns, _ = _select_window_columns(columns, block, lower, upper)
    if not selected_columns:
        return float("nan")

    values = frame[selected_columns].to_numpy(dtype=float)
    per_subject = []
    for row in values:
        finite = row[np.isfinite(row)]
        if finite.size == 0:
            continue
        peak = float(np.max(np.abs(finite)))
        noise = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        if noise <= 0:
            continue
        per_subject.append(peak / noise)
    if not per_subject:
        return float("nan")
    return float(np.median(per_subject))


def ratio_from_integrals(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    finite_scale = pd.to_numeric(denominator, errors="coerce").abs()
    reference = float(finite_scale[finite_scale > 0].median()) if (finite_scale > 0).any() else 0.0
    tolerance = max(1e-12, reference * 1e-8)
    safe_denominator = denominator.mask(finite_scale <= tolerance)
    return numerator / safe_denominator


def build_named_index(integrals: dict[str, pd.Series], components: list[str]) -> pd.Series:
    if len(components) < 2:
        raise ValueError("Named index requires at least numerator and denominator integrals")

    numerator_name = str(components[0])
    if numerator_name not in integrals:
        return pd.Series(np.nan, index=next(iter(integrals.values())).index, dtype=float)

    numerator = integrals[numerator_name]
    if len(components) == 2:
        denominator_name = str(components[1])
        if denominator_name not in integrals:
            return pd.Series(np.nan, index=numerator.index, dtype=float)
        return ratio_from_integrals(numerator, integrals[denominator_name])

    denominator_parts = [integrals[str(name)] for name in components[1:] if str(name) in integrals]
    if len(denominator_parts) != len(components) - 1:
        return pd.Series(np.nan, index=numerator.index, dtype=float)
    denominator = sum(denominator_parts)
    return ratio_from_integrals(numerator, denominator)


def infer_block_from_upper(upper: float) -> str:
    return "1300" if upper < 2000 else "2850"
