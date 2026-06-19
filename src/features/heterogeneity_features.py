from __future__ import annotations

import numpy as np
import pandas as pd


def extract_heterogeneity_features(
    area_df: pd.DataFrame,
    peak_indices: np.ndarray | None = None,
    n_targeted_peaks: int = 5,
    expected_pixels_per_area: int = 25,
) -> pd.DataFrame:
    """Compute subject-level heterogeneity features from area summaries."""
    required = {"subject_id", "area_mean", "area_std", "n_valid_pixels"}
    missing = required - set(area_df.columns)
    if missing:
        raise ValueError(f"area_df missing required columns: {sorted(missing)}")

    peaks = np.array([], dtype=int) if peak_indices is None else np.asarray(peak_indices, dtype=int)
    peaks = peaks[: max(0, n_targeted_peaks)]

    rows: list[dict[str, float | str]] = []
    for subject_id, df_sub in area_df.groupby("subject_id"):
        means = np.vstack(list(df_sub["area_mean"].to_numpy()))
        stds = np.vstack(list(df_sub["area_std"].to_numpy()))

        inter_area_var = np.var(means, axis=0)
        global_mean = np.mean(means, axis=0)
        global_std = np.std(means, axis=0)

        cv_per_channel = np.divide(
            global_std,
            np.abs(global_mean),
            out=np.zeros_like(global_std),
            where=np.abs(global_mean) > 1e-12,
        )

        n_areas = int(len(df_sub))
        n_valid_total = int(df_sub["n_valid_pixels"].sum())
        pixel_survival_ratio = float(
            n_valid_total / max(n_areas * expected_pixels_per_area, 1)
        )

        max_var_idx = int(np.argmax(inter_area_var)) if inter_area_var.size > 0 else -1
        max_var_val = float(np.max(inter_area_var)) if inter_area_var.size > 0 else 0.0

        row: dict[str, float | str] = {
            "subject_id": str(subject_id),
            "inter_area_variance_mean": float(np.mean(inter_area_var)),
            "cv_mean": float(np.mean(cv_per_channel)),
            "pooled_within_area_std_mean": float(np.mean(stds)),
            "pixel_survival_ratio": pixel_survival_ratio,
            "wavenumber_max_inter_area_variance_index": float(max_var_idx),
            "max_inter_area_variance_value": max_var_val,
        }

        for i, p_idx in enumerate(peaks):
            if p_idx < 0 or p_idx >= means.shape[1]:
                row[f"targeted_peak_std_{i}"] = 0.0
            else:
                row[f"targeted_peak_std_{i}"] = float(np.std(means[:, p_idx]))

        rows.append(row)

    return pd.DataFrame(rows)
