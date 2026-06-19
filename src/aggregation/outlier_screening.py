from __future__ import annotations

import numpy as np
import pandas as pd


def fit_outlier_threshold(
    train_area_df: pd.DataFrame,
    threshold_factor: float = 2.0,
) -> dict:
    """Fit a train-only correlation-distance threshold using median + k*MAD."""
    required = {"subject_id", "slide_number", "area_mean"}
    missing = required - set(train_area_df.columns)
    if missing:
        raise ValueError(f"train_area_df missing required columns: {sorted(missing)}")

    distances = _compute_area_distances(train_area_df)
    if distances.size == 0:
        return {"median": 0.0, "mad": 0.0, "threshold": np.inf, "threshold_factor": float(threshold_factor)}

    med = float(np.median(distances))
    mad = float(np.median(np.abs(distances - med)))
    threshold = med + float(threshold_factor) * mad
    return {
        "median": med,
        "mad": mad,
        "threshold": float(threshold),
        "threshold_factor": float(threshold_factor),
    }


def apply_outlier_threshold(area_df: pd.DataFrame, model: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a pre-fitted outlier threshold to any area table."""
    threshold = float(model.get("threshold", np.inf))
    distances = _compute_area_distances(area_df)
    if distances.size != len(area_df):
        raise RuntimeError("Distance computation mismatch")

    keep = distances <= threshold
    out = area_df.copy().reset_index(drop=True)
    out["outlier_distance"] = distances
    out["is_outlier"] = ~keep
    return out.loc[keep].reset_index(drop=True), out


def screen_area_outliers(
    train_area_df: pd.DataFrame,
    test_area_df: pd.DataFrame,
    threshold_factor: float = 2.0,
) -> dict:
    """Fit threshold on train only and apply to train and test."""
    model = fit_outlier_threshold(train_area_df, threshold_factor=threshold_factor)
    train_kept, train_scored = apply_outlier_threshold(train_area_df, model)
    test_kept, test_scored = apply_outlier_threshold(test_area_df, model)
    return {
        "model": model,
        "train_kept": train_kept,
        "test_kept": test_kept,
        "train_scored": train_scored,
        "test_scored": test_scored,
    }


def _compute_area_distances(area_df: pd.DataFrame) -> np.ndarray:
    if area_df.empty:
        return np.array([], dtype=float)

    out = np.zeros(len(area_df), dtype=float)
    work = area_df.reset_index(drop=True)

    for (subject_id, slide_number), grp in work.groupby(["subject_id", "slide_number"], sort=False):
        idx = grp.index.to_numpy()
        vectors = np.vstack([np.asarray(v, dtype=float) for v in grp["area_mean"].to_numpy()])
        centroid = np.mean(vectors, axis=0)
        out[idx] = _correlation_distances_batch(vectors, centroid)

    return out


def _correlation_distances_batch(vectors: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Vectorized Pearson correlation distance between each row and the centroid."""
    a = vectors - vectors.mean(axis=1, keepdims=True)  # (n, p) centred rows
    b = centroid - centroid.mean()                      # (p,) centred centroid
    dot = a @ b                                         # (n,)
    norm_a = np.linalg.norm(a, axis=1)                  # (n,)
    norm_b = float(np.linalg.norm(b))
    denom = norm_a * norm_b                             # (n,)
    # where degenerate (flat spectrum or flat centroid), treat as perfectly correlated
    corr = np.where(denom > 1e-12, dot / denom, 1.0)
    return 1.0 - np.clip(corr, -1.0, 1.0)
