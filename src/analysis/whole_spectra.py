"""Whole-spectra analyses that require both Raman regions at once."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.statistics import REGION_BOUNDS, cohens_d, fdr_correction


def _region_bounds(config: dict) -> dict[str, tuple[float, float]]:
    bounds_cfg = config.get("data", {}).get("spectral_range_bounds", {})
    fingerprint = tuple(
        bounds_cfg.get("fingerprint", REGION_BOUNDS["fingerprint"])
    )
    ch_stretch = tuple(
        bounds_cfg.get("ch_stretch", REGION_BOUNDS["ch_stretch"])
    )
    return {
        "fingerprint": (float(fingerprint[0]), float(fingerprint[1])),
        "ch_stretch": (float(ch_stretch[0]), float(ch_stretch[1])),
    }


def _validate_spectra_shape(
    spectra: np.ndarray, wavenumbers: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    spectra = np.asarray(spectra, dtype=float)
    wavenumbers = np.asarray(wavenumbers, dtype=float)
    if spectra.ndim != 2:
        raise ValueError("spectra must be a 2D array")
    if wavenumbers.ndim != 1:
        raise ValueError("wavenumbers must be a 1D array")
    if spectra.shape[1] != wavenumbers.size:
        raise ValueError("spectra and wavenumbers must have matching feature counts")
    return spectra, wavenumbers


def _validate_split_index(split_index: int, n_features: int) -> int:
    split_index = int(split_index)
    if split_index <= 0 or split_index >= n_features:
        raise ValueError("split_index must fall strictly inside the feature axis")
    return split_index


def _ratio_column_name(band_a: str, band_b: str) -> str:
    return f"{band_a}__over__{band_b}"


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    min_value = float(min(numerator.min(), denominator.min()))
    offset = max(0.0, 1.0 - min_value)
    adjusted_numerator = numerator + offset
    adjusted_denominator = denominator + offset
    adjusted_denominator = adjusted_denominator.where(adjusted_denominator > 0, np.nan)
    return adjusted_numerator / adjusted_denominator


def _correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    if method == "spearman":
        return float(stats.spearmanr(x, y, nan_policy="omit").statistic)
    if method == "pearson":
        return float(stats.pearsonr(x, y)[0])
    raise ValueError(f"Unsupported correlation method '{method}'")


def _bootstrap_ci(
    x: np.ndarray, y: np.ndarray, method: str, n_bootstrap: int
) -> tuple[float, float]:
    if x.size < 3 or y.size < 3 or n_bootstrap <= 0:
        return np.nan, np.nan

    rng = np.random.default_rng(42)
    samples = []
    for _ in range(int(n_bootstrap)):
        indices = rng.integers(0, x.size, size=x.size)
        sample_corr = _correlation(x[indices], y[indices], method)
        if np.isfinite(sample_corr):
            samples.append(sample_corr)

    if not samples:
        return np.nan, np.nan

    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _mad(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    median = float(np.nanmedian(array))
    return float(np.nanmedian(np.abs(array - median)))


def load_band_definitions(config: dict) -> dict[str, tuple[float, float]]:
    """Load whole-spectra band definitions validated against the two region bounds."""
    band_cfg = config.get("whole_spectra", {}).get("bands", {})
    bounds = _region_bounds(config)
    validated: dict[str, tuple[float, float]] = {}

    for band_name, band_limits in band_cfg.items():
        if not isinstance(band_limits, (list, tuple)) or len(band_limits) != 2:
            raise ValueError(f"Band '{band_name}' must define exactly two limits")
        lower, upper = float(band_limits[0]), float(band_limits[1])
        if lower >= upper:
            raise ValueError(f"Band '{band_name}' must satisfy lower < upper")

        in_fingerprint = bounds["fingerprint"][0] <= lower <= upper <= bounds["fingerprint"][1]
        in_ch_stretch = bounds["ch_stretch"][0] <= lower <= upper <= bounds["ch_stretch"][1]
        if not (in_fingerprint or in_ch_stretch):
            raise ValueError(
                f"Band '{band_name}' ({lower:.1f}, {upper:.1f}) lies outside the configured "
                "fingerprint and CH-stretch ranges"
            )
        validated[band_name] = (lower, upper)

    return validated


def compute_band_intensities(
    spectra: np.ndarray,
    wavenumbers: np.ndarray,
    bands: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """Compute per-spectrum mean intensity inside each named band."""
    spectra, wavenumbers = _validate_spectra_shape(spectra, wavenumbers)
    rows: dict[str, np.ndarray] = {}

    for band_name, (lower, upper) in bands.items():
        mask = (wavenumbers >= lower) & (wavenumbers <= upper)
        if not mask.any():
            raise ValueError(
                f"Band '{band_name}' does not overlap the supplied wavenumber axis"
            )
        rows[band_name] = np.nanmean(spectra[:, mask], axis=1)

    return pd.DataFrame(rows)


def compute_cross_region_ratios(
    band_df: pd.DataFrame,
    ratio_pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    """Compute stable per-spectrum band ratios for the requested band pairs."""
    ratio_data: dict[str, pd.Series] = {}

    for band_a, band_b in ratio_pairs:
        if band_a not in band_df.columns or band_b not in band_df.columns:
            raise ValueError(f"Missing ratio band(s): {band_a}, {band_b}")
        ratio_name = _ratio_column_name(band_a, band_b)
        ratio_data[ratio_name] = _safe_ratio(band_df[band_a], band_df[band_b])

    return pd.DataFrame(ratio_data, index=band_df.index)


def test_ratios_between_groups(
    ratio_df: pd.DataFrame,
    group_labels: np.ndarray,
    contrasts: list[tuple[str, str]],
) -> pd.DataFrame:
    """Test log-ratios between requested group contrasts with BH-FDR per contrast."""
    labels = np.asarray(group_labels)
    if len(labels) != len(ratio_df):
        raise ValueError("group_labels must align with ratio_df rows")

    rows: list[dict[str, object]] = []
    for group_a, group_b in contrasts:
        contrast_rows = []
        mask_a = labels == group_a
        mask_b = labels == group_b
        for ratio_name in ratio_df.columns:
            values_a = np.log(
                np.clip(
                    ratio_df.loc[mask_a, ratio_name].to_numpy(dtype=float),
                    1e-12,
                    None,
                )
            )
            values_b = np.log(
                np.clip(
                    ratio_df.loc[mask_b, ratio_name].to_numpy(dtype=float),
                    1e-12,
                    None,
                )
            )
            values_a = values_a[np.isfinite(values_a)]
            values_b = values_b[np.isfinite(values_b)]

            if values_a.size < 3 or values_b.size < 3:
                contrast_rows.append(
                    {
                        "contrast": f"{group_a}_vs_{group_b}",
                        "group_a": group_a,
                        "group_b": group_b,
                        "ratio": ratio_name,
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "effect_size": np.nan,
                        "mean_ratio_a": np.nan,
                        "mean_ratio_b": np.nan,
                        "mean_log_ratio_a": np.nan,
                        "mean_log_ratio_b": np.nan,
                        "test_used": "none",
                    }
                )
                continue

            statistic, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)
            contrast_rows.append(
                {
                    "contrast": f"{group_a}_vs_{group_b}",
                    "group_a": group_a,
                    "group_b": group_b,
                    "ratio": ratio_name,
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "effect_size": cohens_d(values_a, values_b),
                    "mean_ratio_a": float(np.mean(ratio_df.loc[mask_a, ratio_name])),
                    "mean_ratio_b": float(np.mean(ratio_df.loc[mask_b, ratio_name])),
                    "mean_log_ratio_a": float(np.mean(values_a)),
                    "mean_log_ratio_b": float(np.mean(values_b)),
                    "test_used": "welch_t_log_ratio",
                }
            )

        contrast_df = pd.DataFrame(contrast_rows)
        corrected, _ = fdr_correction(contrast_df["p_value"].to_numpy(dtype=float))
        contrast_df["p_corrected"] = corrected
        contrast_df["significant"] = contrast_df["p_corrected"] < 0.05
        rows.extend(contrast_df.to_dict(orient="records"))

    return pd.DataFrame(rows)



test_ratios_between_groups.__test__ = False


def cross_region_band_correlation(
    fingerprint_band_df: pd.DataFrame,
    ch_band_df: pd.DataFrame,
    group_labels: np.ndarray,
    method: str = "spearman",
    n_bootstrap: int = 200,
) -> dict[str, pd.DataFrame]:
    """Estimate per-group cross-region band correlations with bootstrap confidence intervals."""
    labels = np.asarray(group_labels)
    if len(labels) != len(fingerprint_band_df) or len(labels) != len(ch_band_df):
        raise ValueError("group_labels must align with both band dataframes")

    outputs: dict[str, pd.DataFrame] = {}
    for group_name in np.unique(labels):
        mask = labels == group_name
        fp_group = fingerprint_band_df.loc[mask]
        ch_group = ch_band_df.loc[mask]
        rows = []
        for fp_band in fp_group.columns:
            for ch_band in ch_group.columns:
                x = fp_group[fp_band].to_numpy(dtype=float)
                y = ch_group[ch_band].to_numpy(dtype=float)
                pair_mask = np.isfinite(x) & np.isfinite(y)
                x = x[pair_mask]
                y = y[pair_mask]
                correlation = _correlation(x, y, method) if x.size >= 3 else np.nan
                ci_low, ci_high = _bootstrap_ci(x, y, method, n_bootstrap)
                rows.append(
                    {
                        "fingerprint_band": fp_band,
                        "ch_band": ch_band,
                        "correlation": correlation,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "n_samples": int(x.size),
                        "method": method,
                    }
                )
        outputs[str(group_name)] = pd.DataFrame(rows)

    return outputs


def regional_snv_residual_variance(
    subject_spectra: dict[str, np.ndarray],
    split_index: int,
) -> pd.DataFrame:
    """Compute per-subject residual variance per region and flag MAD outliers."""
    if not subject_spectra:
        return pd.DataFrame(
            columns=[
                "subject_id",
                "fingerprint_variance",
                "ch_stretch_variance",
                "fingerprint_threshold",
                "ch_stretch_threshold",
                "fingerprint_outlier",
                "ch_stretch_outlier",
                "is_outlier",
            ]
        )

    first_spectrum = np.asarray(next(iter(subject_spectra.values())), dtype=float)
    split_index = _validate_split_index(split_index, first_spectrum.size)

    rows = []
    for subject_id, spectrum in subject_spectra.items():
        spectrum = np.asarray(spectrum, dtype=float)
        if spectrum.ndim != 1 or spectrum.size != first_spectrum.size:
            raise ValueError("All subject spectra must be 1D arrays with a shared length")

        fingerprint = spectrum[:split_index]
        ch_stretch = spectrum[split_index:]
        rows.append(
            {
                "subject_id": str(subject_id),
                "fingerprint_variance": float(np.nanvar(fingerprint)),
                "ch_stretch_variance": float(np.nanvar(ch_stretch)),
            }
        )

    result = pd.DataFrame(rows)
    fp_threshold = float(
        np.nanmedian(result["fingerprint_variance"])
        + 2.0 * _mad(result["fingerprint_variance"])
    )
    ch_threshold = float(
        np.nanmedian(result["ch_stretch_variance"])
        + 2.0 * _mad(result["ch_stretch_variance"])
    )
    result["fingerprint_threshold"] = fp_threshold
    result["ch_stretch_threshold"] = ch_threshold
    result["fingerprint_outlier"] = result["fingerprint_variance"] > fp_threshold
    result["ch_stretch_outlier"] = result["ch_stretch_variance"] > ch_threshold
    result["is_outlier"] = result["fingerprint_outlier"] | result["ch_stretch_outlier"]
    return result
