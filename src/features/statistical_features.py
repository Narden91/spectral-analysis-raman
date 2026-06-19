from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


def _safe_entropy(x: np.ndarray) -> float:
    mag = np.abs(x)
    denom = float(np.sum(mag))
    if denom <= 0:
        return 0.0
    p = mag / denom
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def _safe_zcr(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    s = np.sign(x)
    return float(np.sum(s[:-1] * s[1:] < 0) / (x.size - 1))


def _safe_shape_moments(x: np.ndarray) -> tuple[float, float]:
    """Return stable kurtosis/skewness for degenerate spectral contexts."""
    finite = x[np.isfinite(x)]
    if finite.size < 3 or float(np.std(finite)) <= 1.0e-12:
        return 0.0, 0.0
    return (
        float(kurtosis(finite, fisher=True, bias=False)),
        float(skew(finite, bias=False)),
    )


def _context_features(x: np.ndarray) -> dict[str, float]:
    kurt_value, skew_value = _safe_shape_moments(x)
    idx = np.arange(x.size, dtype=float)
    abs_x = np.abs(x)
    total_abs = float(np.sum(abs_x))

    q = np.array_split(np.arange(x.size), 4)
    q_energy = [float(np.sum(abs_x[q_i] ** 2)) if q_i.size else 0.0 for q_i in q]

    centroid = float(np.sum(idx * abs_x) / total_abs) if total_abs > 0 else 0.0
    spread = (
        float(np.sqrt(np.sum(((idx - centroid) ** 2) * abs_x) / total_abs))
        if total_abs > 0
        else 0.0
    )

    dx = np.diff(x)
    q1 = q_energy[0]
    q4 = q_energy[3]
    q_ratio = float(q1 / q4) if q4 > 1e-12 else 0.0

    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "kurtosis": kurt_value,
        "skewness": skew_value,
        "iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
        "centroid": centroid,
        "spread": spread,
        "entropy": _safe_entropy(x),
        "q1_energy": q_energy[0],
        "q2_energy": q_energy[1],
        "q3_energy": q_energy[2],
        "q4_energy": q_energy[3],
        "q1_q4_ratio": q_ratio,
        "derivative_std": float(np.std(dx)) if dx.size > 0 else 0.0,
        "zcr": _safe_zcr(x),
    }


def extract_statistical_features(
    vectors: np.ndarray,
    split_index: int,
) -> pd.DataFrame:
    """Extract Track C statistical features for fingerprint, CH-stretch, and full contexts.

    Parameters
    ----------
    vectors
        Subject-level spectra as ``(n_subjects, n_channels)``.
    split_index
        Region boundary index used to split fingerprint vs CH-stretch channels.
        This must be provided explicitly by callers to avoid accidental defaults.
    """
    if vectors.ndim != 2:
        raise ValueError("vectors must be 2D")
    if split_index <= 0 or split_index >= vectors.shape[1]:
        raise ValueError("split_index must be between 1 and n_features-1")

    rows: list[dict[str, float]] = []
    for i in range(vectors.shape[0]):
        full = vectors[i]
        fp = full[:split_index]
        ch = full[split_index:]

        row: dict[str, float] = {}
        for prefix, x in (("fp", fp), ("ch", ch), ("full", full)):
            f = _context_features(x)
            keys = [
                "mean",
                "std",
                "kurtosis",
                "skewness",
                "iqr",
                "centroid",
                "spread",
                "entropy",
                "q1_energy",
                "q2_energy",
                "q3_energy",
                "q4_energy",
                "q1_q4_ratio",
                "derivative_std",
                "zcr",
            ]
            for key in keys:
                row[f"{prefix}_{key}"] = f[key]
        rows.append(row)

    return pd.DataFrame(rows)
