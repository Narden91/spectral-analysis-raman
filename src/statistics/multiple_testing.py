"""Multiple-testing corrections shared by the analysis phases."""

import numpy as np
import pandas as pd


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Return Holm family-wise adjusted p-values in input order."""
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=float)
    order = np.argsort(p_values)
    adjusted_sorted = np.maximum.accumulate(p_values[order] * (n - np.arange(n)))
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def holm_adjust_series(p_values: pd.Series) -> pd.Series:
    """Correct numeric values while preserving the index and missing entries."""
    numeric = pd.to_numeric(p_values, errors="coerce")
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = numeric.dropna()
    if not valid.empty:
        result.loc[valid.index] = holm_adjust(valid.to_numpy(dtype=float))
    return result
