"""
Shared spectral preprocessing pipeline for Raman data.

The conservative default path follows the repo's biomarker-discovery guidance:
keep spectra at pixel/file level until QC is complete, remove only clear
instrument artefacts, and make every step configurable from YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pybaselines.smooth import snip
from pybaselines.whittaker import asls, arpls
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, savgol_filter
from scipy.spatial import ConvexHull

from src import config

logger = logging.getLogger(__name__)


@dataclass
class PreprocessResult:
    """Container for one preprocessing run on a homogeneous spectral block."""

    spectra: np.ndarray
    wavenumbers: np.ndarray
    retained_mask: np.ndarray
    qc_flags: pd.DataFrame
    qc_metrics: pd.DataFrame
    stage_arrays: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)


def _pre_cfg() -> dict:
    return config.get("preprocessing", {})


def _range_cfg(region_key: str) -> dict:
    return _pre_cfg().get("ranges", {}).get(region_key, {})


def _artifacts_cfg() -> dict:
    return _pre_cfg().get("artifacts", {})


def _resolve_region_key(wavenumbers: np.ndarray, region: Optional[str]) -> str:
    if region is not None:
        key = str(region).strip().lower().replace("-", "_")
        aliases = {
            "1300": "fingerprint",
            "fingerprint": "fingerprint",
            "fingerprint_1300": "fingerprint",
            "2850": "ch_stretch",
            "ch": "ch_stretch",
            "ch_stretch": "ch_stretch",
            "ch_stretch_2850": "ch_stretch",
        }
        if key not in aliases:
            raise ValueError("region must be one of: fingerprint, 1300, ch_stretch, 2850")
        return aliases[key]

    if wavenumbers.size == 0:
        return "fingerprint"
    return (
        "ch_stretch"
        if float(np.nanmax(wavenumbers)) > 2000 and float(np.nanmin(wavenumbers)) > 1800
        else "fingerprint"
    )


def _normalization_method(method: Optional[str], mode: Optional[str]) -> str:
    nm_cfg = _pre_cfg().get("normalization", {})
    mode_map = {
        "eda": "vector",
        "cross": "none",
        "ml": "robust",
        **nm_cfg.get("modes", {}),
    }
    if mode is not None:
        if mode not in mode_map:
            raise ValueError(f"Unknown normalization mode: {mode}")
        return str(mode_map[mode]).strip().lower()
    return str(method if method is not None else nm_cfg.get("method", "vector")).strip().lower()


def _validate_preprocessing_config(region_key: str) -> None:
    pre_cfg = _pre_cfg()
    nm_cfg = pre_cfg.get("normalization", {})
    norm_method = _normalization_method(nm_cfg.get("method"), None)
    allowed = {"none", "area", "snv", "vector", "minmax", "robust"}
    if norm_method not in allowed:
        raise ValueError(f"Unknown normalization method in config: {norm_method}")
    if norm_method == "snv" and bool(nm_cfg.get("vector_norm_after_snv", False)):
        raise ValueError("SNV and vector normalization cannot be stacked in one preprocessing pass")

    sm_cfg = {**pre_cfg.get("smoothing", {}), **_range_cfg(region_key).get("smoothing", {})}
    if bool(sm_cfg.get("enabled", False)):
        window_length = int(sm_cfg.get("window_length", 5))
        polyorder = int(sm_cfg.get("polyorder", 2))
        if window_length < 3:
            raise ValueError("Smoothing window_length must be >= 3")
        if window_length % 2 == 0:
            raise ValueError("Smoothing window_length must be odd (Savitzky-Golay requirement)")
        if polyorder >= window_length:
            raise ValueError("Smoothing polyorder must be smaller than window_length")

    bl_cfg = {**pre_cfg.get("baseline", {}), **_range_cfg(region_key).get("baseline", {})}
    bl_method = str(bl_cfg.get("method", "arpls")).strip().lower()
    allowed_bl = {"arpls", "als", "snip"}
    if bl_method not in allowed_bl:
        raise ValueError(f"Unknown baseline method in config: {bl_method!r}. Must be one of {allowed_bl}")
    if bl_method in {"arpls", "als"}:
        lam = float(bl_cfg.get("lam", 1e6))
        ratio = float(bl_cfg.get("ratio", 1e-3))
        max_iter = int(bl_cfg.get("max_iter", 100))
        if lam <= 0:
            raise ValueError(f"Baseline lam must be > 0, got {lam}")
        if ratio <= 0:
            raise ValueError(f"Baseline ratio must be > 0, got {ratio}")
        if max_iter < 1:
            raise ValueError(f"Baseline max_iter must be >= 1, got {max_iter}")


def _compute_qc_metrics(spectra: np.ndarray) -> pd.DataFrame:
    if spectra.size == 0:
        return pd.DataFrame(
            columns=[
                "row_index",
                "max_intensity",
                "min_intensity",
                "dynamic_range",
                "total_area",
                "mean_intensity",
                "variance",
                "baseline_slope",
                "fluorescence_level",
                "snr",
                "n_cosmic_channels",
                "cosmic_spike_count",
                "cosmic_spike_fraction",
                "n_suppressed_cosmic_channels",
                "fraction_corrected",
                "n_spike_repaired_channels",
                "spike_repair_fraction",
                "baseline_fraction",
                "saturation_fraction",
                "is_outlier",
                "outlier_score",
            ]
        )

    finite_mask = np.isfinite(spectra)
    finite_counts = finite_mask.sum(axis=1)
    has_finite = finite_counts > 0

    finite_or_neg_inf = np.where(finite_mask, spectra, -np.inf)
    max_intensity = finite_or_neg_inf.max(axis=1)
    max_intensity[~has_finite] = np.nan

    finite_or_pos_inf = np.where(finite_mask, spectra, np.inf)
    min_intensity = finite_or_pos_inf.min(axis=1)
    min_intensity[~has_finite] = np.nan

    finite_sum = np.where(finite_mask, spectra, 0.0).sum(axis=1)
    mean_intensity = np.divide(
        finite_sum,
        finite_counts,
        out=np.full(spectra.shape[0], np.nan, dtype=float),
        where=has_finite,
    )
    centered = np.where(finite_mask, spectra - mean_intensity[:, np.newaxis], 0.0)
    variance = np.divide(
        (centered**2).sum(axis=1),
        finite_counts,
        out=np.full(spectra.shape[0], np.nan, dtype=float),
        where=has_finite,
    )

    fluorescence_level = np.full(spectra.shape[0], np.nan, dtype=float)
    for idx, row in enumerate(spectra):
        valid = row[np.isfinite(row)]
        if valid.size:
            fluorescence_level[idx] = float(np.median(valid))

    finite_abs = np.where(finite_mask, np.abs(spectra), 0.0)
    total_area = np.trapezoid(finite_abs, axis=1, dx=1.0)
    total_area[~has_finite] = np.nan

    row_index = np.arange(spectra.shape[0], dtype=int)
    dynamic_range = max_intensity - min_intensity
    baseline_slope = np.divide(
        spectra[:, -1] - spectra[:, 0],
        max(spectra.shape[1] - 1, 1),
    )

    col_finite = finite_mask.sum(axis=0)
    mean_spec = np.divide(
        np.where(finite_mask, spectra, 0.0).sum(axis=0),
        col_finite,
        out=np.zeros(spectra.shape[1], dtype=float),
        where=col_finite > 0,
    )
    peak_idx, _ = find_peaks(mean_spec)
    anchor_idx = int(peak_idx[np.argmax(mean_spec[peak_idx])]) if peak_idx.size else int(np.nanargmax(mean_spec))
    lo = max(0, anchor_idx - 5)
    hi = min(spectra.shape[1], anchor_idx + 6)
    signal_window = spectra[:, lo:hi]
    signal_finite = np.isfinite(signal_window)
    signal = np.where(signal_finite, signal_window, -np.inf).max(axis=1)
    signal[~signal_finite.any(axis=1)] = np.nan

    noise_window = spectra[:, : min(spectra.shape[1], 20)]
    noise_finite = np.isfinite(noise_window)
    noise_counts = noise_finite.sum(axis=1)
    noise_sum = np.where(noise_finite, noise_window, 0.0).sum(axis=1)
    noise_mean = np.divide(
        noise_sum,
        noise_counts,
        out=np.zeros(spectra.shape[0], dtype=float),
        where=noise_counts > 0,
    )
    noise_centered = np.where(noise_finite, noise_window - noise_mean[:, np.newaxis], 0.0)
    noise_var = np.divide(
        (noise_centered**2).sum(axis=1),
        noise_counts,
        out=np.full(spectra.shape[0], np.nan, dtype=float),
        where=noise_counts > 0,
    )
    noise_std = np.sqrt(noise_var)
    noise_std[(noise_std == 0) & np.isfinite(noise_std)] = 1e-10
    snr = signal / noise_std

    return pd.DataFrame(
        {
            "row_index": row_index,
            "max_intensity": max_intensity,
            "min_intensity": min_intensity,
            "dynamic_range": dynamic_range,
            "total_area": total_area,
            "mean_intensity": mean_intensity,
            "variance": variance,
            "baseline_slope": baseline_slope,
            "fluorescence_level": fluorescence_level,
            "snr": snr,
            "n_cosmic_channels": np.zeros(spectra.shape[0], dtype=int),
            "cosmic_spike_count": np.zeros(spectra.shape[0], dtype=int),
            "cosmic_spike_fraction": np.zeros(spectra.shape[0], dtype=float),
            "n_suppressed_cosmic_channels": np.zeros(spectra.shape[0], dtype=int),
            "fraction_corrected": np.zeros(spectra.shape[0], dtype=float),
            "n_spike_repaired_channels": np.zeros(spectra.shape[0], dtype=int),
            "spike_repair_fraction": np.zeros(spectra.shape[0], dtype=float),
            "baseline_fraction": np.zeros(spectra.shape[0], dtype=float),
            "saturation_fraction": np.zeros(spectra.shape[0], dtype=float),
            "is_outlier": np.zeros(spectra.shape[0], dtype=bool),
            "outlier_score": np.zeros(spectra.shape[0], dtype=float),
        }
    )


def _apply_initial_qc(
    spectra: np.ndarray,
    metrics_df: pd.DataFrame,
    min_spectral_std: Optional[float] = None,
    saturation_value: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    dp_cfg = _pre_cfg().get("dead_pixel", {})
    min_std = float(min_spectral_std if min_spectral_std is not None else dp_cfg.get("min_spectral_std", 1.0))
    sat_val = float(saturation_value if saturation_value is not None else dp_cfg.get("saturation_value", 65000))

    warning_columns = {
        "qc_warning": np.zeros(spectra.shape[0], dtype=bool),
        "qc_warning_reason": np.full(spectra.shape[0], "", dtype=object),
        "qc_warning_metrics": np.full(spectra.shape[0], "", dtype=object),
    }

    if spectra.size == 0:
        empty_mask = np.array([], dtype=bool)
        empty_flags = pd.DataFrame(
            {
                "row_index": pd.Series(np.array([], dtype=int)),
                "qc_flag": pd.Series(np.array([], dtype=bool)),
                "qc_reason": pd.Series(np.array([], dtype=object), dtype=object),
                "qc_warning": pd.Series(np.array([], dtype=bool)),
                "qc_warning_reason": pd.Series(np.array([], dtype=object), dtype=object),
                "qc_warning_metrics": pd.Series(np.array([], dtype=object), dtype=object),
            }
        )
        return spectra, empty_mask, empty_flags

    finite_mask = np.isfinite(spectra)
    has_invalid = ~finite_mask.all(axis=1)
    finite_counts = finite_mask.sum(axis=1)
    finite_sum = np.where(finite_mask, spectra, 0.0).sum(axis=1)
    finite_mean = np.divide(finite_sum, finite_counts, out=np.zeros_like(finite_sum), where=finite_counts > 0)
    centered = np.where(finite_mask, spectra - finite_mean[:, np.newaxis], 0.0)
    finite_var = np.divide((centered**2).sum(axis=1), finite_counts, out=np.zeros_like(finite_sum), where=finite_counts > 0)
    row_std = np.sqrt(finite_var)
    low_std = row_std < min_std
    finite_or_neg_inf = np.where(np.isfinite(spectra), spectra, -np.inf)
    saturated = finite_or_neg_inf.max(axis=1) > sat_val
    invalid = has_invalid | low_std | saturated
    valid_mask = ~invalid

    reasons = np.full(spectra.shape[0], "ok", dtype=object)
    reasons[has_invalid] = "non_finite"
    reasons[low_std & ~has_invalid] = "flat_spectrum"
    reasons[saturated & ~has_invalid] = "saturated"

    metrics_df.loc[:, "dynamic_range"] = metrics_df["dynamic_range"].astype(float)
    flags_df = pd.DataFrame(
        {
            "row_index": np.arange(spectra.shape[0], dtype=int),
            "qc_flag": ~valid_mask,
            "qc_reason": pd.Series(reasons, dtype=object),
            "qc_warning": pd.Series(warning_columns["qc_warning"], dtype=bool),
            "qc_warning_reason": pd.Series(warning_columns["qc_warning_reason"], dtype=object),
            "qc_warning_metrics": pd.Series(warning_columns["qc_warning_metrics"], dtype=object),
        }
    )
    if int(invalid.sum()) > 0:
        logger.info("Initial QC removed %d/%d spectra", int(invalid.sum()), spectra.shape[0])
    return spectra[valid_mask], valid_mask, flags_df


def compute_qc_metrics(spectra: np.ndarray) -> pd.DataFrame:
    """Public, stable entry point for QC metric computation.

    Delegates to the internal helper so downstream modules do not depend on
    the private function name.
    """
    return _compute_qc_metrics(np.asarray(spectra, dtype=float))


def screen_dead_pixels(
    spectra: np.ndarray,
    min_spectral_std: Optional[float] = None,
    saturation_value: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    metrics_df = compute_qc_metrics(np.asarray(spectra, dtype=float))
    filtered, valid_mask, _ = _apply_initial_qc(
        np.asarray(spectra, dtype=float),
        metrics_df,
        min_spectral_std=min_spectral_std,
        saturation_value=saturation_value,
    )
    return filtered, valid_mask


def _local_cosmic_detection(
    spectra: np.ndarray,
    z_threshold: float,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect cosmic-ray spikes and compute local-median replacement values.

    Returns
    -------
    spike_mask : np.ndarray
        Boolean mask of detected spikes.
    replacement : np.ndarray
        Local median spectrum used for replacement.
    """
    kernel = max(1, int(window)) * 2 + 1
    local_median = median_filter(spectra, size=(1, kernel), mode="nearest")
    residual = spectra - local_median
    mad = np.median(np.abs(residual), axis=1, keepdims=True)
    mad[mad == 0] = 1e-10
    robust_sigma = 1.4826 * mad
    spike_mask = np.abs(residual) > z_threshold * robust_sigma
    return spike_mask, local_median


def _spatial_cosmic_detection(
    spectra: np.ndarray,
    pixel_coords: np.ndarray,
    z_threshold: float,
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect cosmic-ray spikes using spatial neighbor medians.

    Falls back to local median detection when coordinates are missing or invalid.
    """
    if spectra.size == 0 or pixel_coords is None or len(pixel_coords) != len(spectra):
        return _local_cosmic_detection(spectra, z_threshold=z_threshold, window=2)

    coords = np.asarray(pixel_coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 3:
        return _local_cosmic_detection(spectra, z_threshold=z_threshold, window=2)

    diffs = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(distances, np.inf)
    neighbor_count = max(1, min(int(n_neighbors), len(coords) - 1))
    nearest = np.argpartition(distances, kth=neighbor_count - 1, axis=1)[:, :neighbor_count]

    replacement = np.empty_like(spectra, dtype=float)
    spike_mask = np.zeros_like(spectra, dtype=bool)
    for idx in range(len(spectra)):
        neighbor_median = np.median(spectra[nearest[idx]], axis=0)
        replacement[idx] = neighbor_median
        residual = spectra[idx] - neighbor_median
        mad = np.median(np.abs(residual))
        if mad == 0:
            mad = 1e-10
        local_spikes = np.abs(residual) > (1.4826 * mad * z_threshold)
        spike_mask[idx] = local_spikes
    return spike_mask, replacement


def _detect_cosmic_spikes(
    spectra: np.ndarray,
    pixel_coords: np.ndarray | None,
    cr_cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect cosmic-ray spikes and return a mask plus replacement values."""
    arr = np.asarray(spectra, dtype=float)
    if arr.size == 0:
        return np.zeros_like(arr, dtype=bool), np.zeros_like(arr, dtype=float)

    thr = float(cr_cfg.get("z_threshold", 8.0))
    method = str(cr_cfg.get("method", "spatial")).strip().lower()
    if method == "spatial" and pixel_coords is not None:
        neighbor_count = int(cr_cfg.get("n_neighbors", 8))
        return _spatial_cosmic_detection(
            arr,
            pixel_coords=np.asarray(pixel_coords, dtype=float),
            z_threshold=thr,
            n_neighbors=neighbor_count,
        )
    fallback_window = int(cr_cfg.get("window", 2))
    return _local_cosmic_detection(arr, z_threshold=thr, window=fallback_window)


def _suppress_cosmic_spikes(
    spectra: np.ndarray,
    spike_mask: np.ndarray,
    replacement: np.ndarray,
) -> np.ndarray:
    """Replace detected spike channels with supplied replacement values."""
    arr = np.asarray(spectra, dtype=float).copy()
    if arr.size == 0:
        return arr
    arr[spike_mask] = replacement[spike_mask]
    return arr


def suppress_cosmic_rays(
    spectra: np.ndarray,
    enabled: Optional[bool] = None,
    z_threshold: Optional[float] = None,
    window: Optional[int] = None,
    pixel_coords: np.ndarray | None = None,
) -> np.ndarray:
    """Public convenience wrapper: detect and optionally suppress cosmic rays.

    This function preserves the historical API.  It does **not** record metrics;
    callers that need detection statistics should use ``_detect_cosmic_spikes``
    directly.
    """
    cr_cfg = dict(_pre_cfg().get("cosmic_ray", {}))
    use_filter = bool(enabled if enabled is not None else cr_cfg.get("enabled", False))
    if not use_filter or spectra.size == 0:
        return np.asarray(spectra, dtype=float)

    if z_threshold is not None:
        cr_cfg["z_threshold"] = float(z_threshold)
    if window is not None:
        cr_cfg["window"] = int(window)
    if pixel_coords is not None:
        cr_cfg["method"] = "spatial"

    spike_mask, replacement = _detect_cosmic_spikes(spectra, pixel_coords, cr_cfg)
    filtered = _suppress_cosmic_spikes(spectra, spike_mask, replacement)
    if np.any(spike_mask):
        logger.debug("Cosmic-ray suppression replaced %d channels", int(spike_mask.sum()))
    return filtered


def correct_baseline(
    spectra: np.ndarray,
    method: Optional[str] = None,
    lam: Optional[float] = None,
    p: Optional[float] = None,
    arpls_lam: Optional[float] = None,
    arpls_ratio: Optional[float] = None,
    arpls_max_iter: Optional[int] = None,
    return_baseline: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Baseline correction with explicit arPLS parameter support.

    When ``arpls_*`` arguments are omitted they are read from the merged
    range-specific baseline config that the caller supplies via ``bl_cfg``
    or from the global preprocessing config.
    """
    bl_cfg = _pre_cfg().get("baseline", {})
    chosen_method = str(method if method is not None else bl_cfg.get("method", "arpls")).strip().lower()
    lam_value = float(lam if lam is not None else bl_cfg.get("lam", 1000000.0))
    p_value = float(p if p is not None else bl_cfg.get("p", 0.01))
    snip_cfg = bl_cfg.get("snip", {})
    corrected = np.empty_like(spectra, dtype=float)
    baselines = np.empty_like(spectra, dtype=float)

    arpls_cfg = bl_cfg.get("arpls", {})
    resolved_arpls_lam = float(arpls_lam if arpls_lam is not None else arpls_cfg.get("lam", lam_value))
    resolved_arpls_ratio = float(arpls_ratio if arpls_ratio is not None else arpls_cfg.get("ratio", 0.001))
    resolved_arpls_max_iter = int(arpls_max_iter if arpls_max_iter is not None else arpls_cfg.get("max_iter", 100))

    for index, row in enumerate(np.asarray(spectra, dtype=float)):
        if chosen_method == "als":
            baseline, _ = asls(row, lam=lam_value, p=p_value)
        elif chosen_method == "snip":
            max_half_window = int(snip_cfg.get("max_half_window", 50))
            decreasing = bool(snip_cfg.get("decreasing", True))
            smooth_half_window = int(snip_cfg.get("smooth_half_window", 3))
            local_max_half_window = min(max_half_window, max((row.shape[0] - 1) // 2, 1))
            baseline, _ = snip(
                row,
                max_half_window=local_max_half_window,
                decreasing=decreasing,
                smooth_half_window=smooth_half_window,
            )
        elif chosen_method == "arpls":
            baseline, _ = arpls(
                row,
                lam=resolved_arpls_lam,
                diff_order=2,
                max_iter=resolved_arpls_max_iter,
                tol=resolved_arpls_ratio,
            )
        else:
            raise ValueError(f"Unknown baseline method: {chosen_method}")
        baselines[index] = baseline
        corrected[index] = row - baseline

    logger.debug("Baseline correction (%s) applied to %d spectra", chosen_method, spectra.shape[0])
    if return_baseline:
        return corrected, baselines
    return corrected


def repair_isolated_spikes(
    spectra: np.ndarray,
    z_threshold: float = 8.0,
    window: int = 5,
    max_corrected_fraction: float = 0.02,
) -> np.ndarray:
    """Repair isolated cosmic-ray-like spikes via Whitaker-Hayes modified Z-score.

    Computes the modified Z-score on the discrete first difference of each
    spectrum (Whitaker & Hayes, 2018): flagged channels are replaced by the
    local mean over `window`. Per-spectrum repair is skipped if the flagged
    fraction exceeds `max_corrected_fraction`, to avoid smearing real signal
    on legitimately noisy spectra. Deterministic and subject-independent.
    """
    repaired, _ = repair_isolated_spikes_with_mask(
        spectra,
        z_threshold=z_threshold,
        window=window,
        max_corrected_fraction=max_corrected_fraction,
    )
    return repaired


def repair_isolated_spikes_with_mask(
    spectra: np.ndarray,
    z_threshold: float = 8.0,
    window: int = 5,
    max_corrected_fraction: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Repair isolated spikes and return the repaired spectra plus a mask.

    Returns
    -------
    repaired : np.ndarray
        Spectra after isolated-spike repair.
    repair_mask : np.ndarray
        Boolean mask of channels that were actually replaced.
    """
    if spectra.size == 0:
        return np.asarray(spectra, dtype=float), np.zeros_like(spectra, dtype=bool)

    arr = np.asarray(spectra, dtype=float).copy()
    n_channels = arr.shape[1]
    half_win = max(1, int(window) // 2)

    diff = np.diff(arr, axis=1)
    mad = np.median(np.abs(diff - np.median(diff, axis=1, keepdims=True)), axis=1, keepdims=True)
    mad[mad == 0] = 1e-10
    mod_z = 0.6745 * np.abs(diff - np.median(diff, axis=1, keepdims=True)) / mad
    spike_diff = mod_z > z_threshold
    candidate_mask = np.zeros_like(arr, dtype=bool)
    candidate_mask[:, 1:] |= spike_diff
    candidate_mask[:, :-1] |= spike_diff

    repair_mask = np.zeros_like(arr, dtype=bool)
    for row_idx in range(arr.shape[0]):
        flagged = np.where(candidate_mask[row_idx])[0]
        if flagged.size == 0:
            continue
        if flagged.size / n_channels > max_corrected_fraction:
            continue
        for channel in flagged:
            lo = max(0, channel - half_win)
            hi = min(n_channels, channel + half_win + 1)
            neighborhood = np.delete(arr[row_idx, lo:hi], channel - lo)
            if neighborhood.size > 0:
                arr[row_idx, channel] = float(np.mean(neighborhood))
                repair_mask[row_idx, channel] = True

    return arr, repair_mask


def trim_edges(
    spectra: np.ndarray,
    wavenumbers: np.ndarray,
    trim: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    if trim <= 0:
        return spectra, wavenumbers
    n_cols = spectra.shape[1]
    if n_cols <= 2 * trim:
        logger.warning("Requested trim (%d) too large for spectra width (%d); skipping trim", trim, n_cols)
        return spectra, wavenumbers
    return spectra[:, trim:-trim], wavenumbers[trim:-trim]


def trim_edges_cm(
    spectra: np.ndarray,
    wavenumbers: np.ndarray,
    edge_trim_cm: float,
) -> tuple[np.ndarray, np.ndarray]:
    if edge_trim_cm <= 0 or wavenumbers.size == 0:
        return spectra, wavenumbers
    wn_min = float(np.nanmin(wavenumbers))
    wn_max = float(np.nanmax(wavenumbers))
    keep = (wavenumbers >= wn_min + edge_trim_cm) & (wavenumbers <= wn_max - edge_trim_cm)
    if int(keep.sum()) < 2:
        logger.warning("Requested edge_trim_cm %.2f leaves <2 channels; skipping trim", edge_trim_cm)
        return spectra, wavenumbers
    return spectra[:, keep], wavenumbers[keep]


def smooth_spectra(
    spectra: np.ndarray,
    method: Optional[str] = None,
    window_length: Optional[int] = None,
    polyorder: Optional[int] = None,
) -> np.ndarray:
    sm_cfg = _pre_cfg().get("smoothing", {})
    chosen_method = str(method if method is not None else sm_cfg.get("method", "savgol")).strip().lower()
    if chosen_method != "savgol":
        raise ValueError(f"Unknown smoothing method: {chosen_method}")

    win = int(window_length if window_length is not None else sm_cfg.get("window_length", 5))
    poly = int(polyorder if polyorder is not None else sm_cfg.get("polyorder", 2))
    if win % 2 == 0:
        win += 1
    n_chan = spectra.shape[1]
    if win >= n_chan:
        win = max(3, n_chan - 1 if (n_chan - 1) % 2 == 1 else n_chan - 2)
    if poly >= win:
        poly = max(1, win - 1)
    return savgol_filter(spectra, window_length=win, polyorder=poly, axis=1)


def interpolate_spectra(
    spectra: np.ndarray,
    wavenumbers: np.ndarray,
    target_spacing: Optional[float] = None,
    method: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if spectra.size == 0 or wavenumbers.size == 0:
        return spectra, wavenumbers
    ip_cfg = _pre_cfg().get("interpolation", {})
    spacing = float(target_spacing if target_spacing is not None else ip_cfg.get("target_spacing", 1.0))
    interp_kind = str(method if method is not None else ip_cfg.get("method", "cubic")).strip().lower()
    wn_min = int(np.ceil(float(np.nanmin(wavenumbers))))
    wn_max = int(np.floor(float(np.nanmax(wavenumbers))))
    if wn_max <= wn_min:
        return spectra, wavenumbers
    wn_uniform = np.arange(wn_min, wn_max + spacing, spacing, dtype=float)
    if wn_uniform.size < 2:
        return spectra, wavenumbers
    if interp_kind == "cubic" and wavenumbers.size < 4:
        interp_kind = "linear"
    interpolator = interp1d(
        wavenumbers,
        spectra,
        kind=interp_kind,
        axis=1,
        bounds_error=False,
        fill_value="extrapolate",
    )
    return interpolator(wn_uniform), wn_uniform


_DEFAULT_ANCHOR_WINDOWS: dict[str, list[tuple[float, float]]] = {
    "fingerprint": [(995.0, 1015.0), (1435.0, 1470.0), (1630.0, 1680.0)],
    "ch_stretch": [(2840.0, 2975.0)],
}


def _anchor_windows_for_region(region: Optional[str]) -> list[tuple[float, float]]:
    qf_cfg = _pre_cfg().get("quality_filter", {}).get("spectral_quality", {})
    configured = qf_cfg.get("anchor_windows", {})
    region_key = str(region).strip().lower().replace("-", "_") if region else None
    windows = configured.get(region_key) if region_key else None
    if not windows:
        windows = _DEFAULT_ANCHOR_WINDOWS.get(region_key, [])
    return [(float(lo), float(hi)) for lo, hi in windows]


def _resolve_anchor_index(
    spectra: np.ndarray,
    wavenumbers: Optional[np.ndarray],
    region: Optional[str],
    peak_window: int,
) -> int:
    """Resolve a deterministic SNR anchor channel from fixed wavenumber windows.

    Falls back to the legacy peak-detection anchor only when no configured
    window overlaps the supplied wavenumber axis (or no axis is given), so the
    anchor never depends on subject/batch-specific spikes.
    """
    wn = np.asarray(wavenumbers, dtype=float) if wavenumbers is not None else None
    if wn is not None and wn.size == spectra.shape[1]:
        windows = _anchor_windows_for_region(region)
        in_window = np.zeros(wn.shape, dtype=bool)
        for lo, hi in windows:
            in_window |= (wn >= lo) & (wn <= hi)
        if np.any(in_window):
            mean_spec = np.nanmean(spectra, axis=0)
            candidate_idx = np.where(in_window)[0]
            return int(candidate_idx[np.nanargmax(mean_spec[candidate_idx])])

    mean_spec = np.nanmean(spectra, axis=0)
    peak_idx, _ = find_peaks(mean_spec)
    if peak_idx.size:
        return int(peak_idx[np.argmax(mean_spec[peak_idx])])
    return int(np.nanargmax(mean_spec))


def spectral_quality_mask(
    spectra: np.ndarray,
    snr_threshold: Optional[float] = None,
    peak_window: Optional[int] = None,
    noise_channels: Optional[int] = None,
    min_area_percentile: Optional[float] = None,
    wavenumbers: Optional[np.ndarray] = None,
    region: Optional[str] = None,
) -> tuple[np.ndarray, dict[str, float]]:
    if spectra.size == 0:
        return np.array([], dtype=bool), {"anchor_index": -1.0, "area_threshold": 0.0}
    qf_cfg = _pre_cfg().get("quality_filter", {}).get("spectral_quality", {})
    snr_thr = float(snr_threshold if snr_threshold is not None else qf_cfg.get("snr_threshold", 5.0))
    pwin = int(peak_window if peak_window is not None else qf_cfg.get("peak_window", 5))
    noise_n = int(noise_channels if noise_channels is not None else qf_cfg.get("noise_channels", 20))
    area_pct = float(min_area_percentile if min_area_percentile is not None else qf_cfg.get("min_area_percentile", 10))

    anchor_idx = _resolve_anchor_index(spectra, wavenumbers, region, pwin)
    lo = max(0, anchor_idx - pwin)
    hi = min(spectra.shape[1], anchor_idx + pwin + 1)
    signal = np.nanmax(spectra[:, lo:hi], axis=1)
    noise_hi = min(spectra.shape[1], max(noise_n, 1))
    noise_std = np.nanstd(spectra[:, :noise_hi], axis=1)
    noise_std[noise_std == 0] = 1e-10
    snr = signal / noise_std
    area = np.trapezoid(np.abs(spectra), axis=1, dx=1.0)
    area_threshold = float(np.nanpercentile(area, area_pct))
    valid = np.isfinite(snr) & np.isfinite(area)
    mask = valid & (snr > snr_thr) & (area > area_threshold)
    stats = {
        "anchor_index": float(anchor_idx),
        "area_threshold": area_threshold,
        "snr_median": float(np.nanmedian(snr)),
        "n_valid": float(mask.sum()),
    }
    return mask, stats


def apply_spectral_quality_filter(
    spectra: np.ndarray,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if spectra.size == 0:
        return spectra, np.array([], dtype=bool), {"anchor_index": -1.0, "area_threshold": 0.0}
    mask, stats = spectral_quality_mask(spectra, **kwargs)
    return spectra[mask], mask, stats


def _resolve_preview_trim_slice(
    wavenumbers: np.ndarray,
    trim: Optional[int] = None,
    trim_cm: Optional[float] = None,
) -> tuple[int, int]:
    """Return a (start, stop) slice that previews the edge trim that will be applied."""
    n = wavenumbers.size
    if n == 0:
        return 0, 0

    if trim_cm is not None:
        if float(trim_cm) <= 0:
            return 0, n
        wn_min = float(np.nanmin(wavenumbers))
        wn_max = float(np.nanmax(wavenumbers))
        keep = (wavenumbers >= wn_min + float(trim_cm)) & (wavenumbers <= wn_max - float(trim_cm))
        if int(keep.sum()) < 2:
            return 0, n
        start = int(np.argmax(keep))
        stop = n - int(np.argmax(keep[::-1]))
        return start, stop

    if trim is not None:
        t = int(trim)
        if t <= 0 or 2 * t >= n:
            return 0, n
        return t, n - t

    edge_cfg = _pre_cfg().get("edge_trim", {})
    edge_trim_cm = edge_cfg.get("edge_trim_cm", None)
    if edge_trim_cm is not None and float(edge_trim_cm) > 0:
        return _resolve_preview_trim_slice(wavenumbers, trim_cm=float(edge_trim_cm))

    trim_val = edge_cfg.get("trim", 0)
    if trim_val is not None and int(trim_val) > 0:
        return _resolve_preview_trim_slice(wavenumbers, trim=int(trim_val))

    return 0, n


def _spectral_quality_mask_with_edge_guard(
    spectra: np.ndarray,
    wavenumbers: np.ndarray,
    region: Optional[str] = None,
    trim: Optional[int] = None,
    trim_cm: Optional[float] = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute the spectral-quality mask using a trim-aware preview.

    When the edge-artifact guard is enabled, SNR and area statistics are
    computed on the subset of channels that will remain after edge trimming,
    so transient edge corruption cannot be the sole reason for rejecting a
    spectrum.
    """
    qf_cfg = _pre_cfg().get("quality_filter", {}).get("spectral_quality", {})
    use_guard = bool(qf_cfg.get("use_trim_aware_preview", True))
    if not use_guard or wavenumbers.size == 0:
        return spectral_quality_mask(spectra, wavenumbers=wavenumbers, region=region)

    start, stop = _resolve_preview_trim_slice(wavenumbers, trim=trim, trim_cm=trim_cm)
    if start == 0 and stop == wavenumbers.size:
        return spectral_quality_mask(spectra, wavenumbers=wavenumbers, region=region)

    preview_spectra = spectra[:, start:stop]
    preview_wavenumbers = wavenumbers[start:stop]
    mask, stats = spectral_quality_mask(preview_spectra, wavenumbers=preview_wavenumbers, region=region)
    stats["edge_guard_preview_start"] = float(start)
    stats["edge_guard_preview_stop"] = float(stop)
    return mask, stats


def normalize_spectra(
    spectra: np.ndarray,
    method: Optional[str] = None,
    mode: Optional[str] = None,
) -> np.ndarray:
    chosen_method = _normalization_method(method, mode)
    if chosen_method == "none":
        return np.asarray(spectra, dtype=float)
    arr = np.asarray(spectra, dtype=float)
    if chosen_method == "area":
        areas = np.trapezoid(np.abs(arr), axis=1, dx=1.0)
        areas[areas == 0] = 1e-10
        return arr / areas[:, np.newaxis]
    if chosen_method == "snv":
        means = arr.mean(axis=1, keepdims=True)
        stds = arr.std(axis=1, keepdims=True)
        stds[stds == 0] = 1e-10
        return (arr - means) / stds
    if chosen_method == "vector":
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return arr / norms
    if chosen_method == "minmax":
        mins = arr.min(axis=1, keepdims=True)
        maxs = arr.max(axis=1, keepdims=True)
        ranges = maxs - mins
        ranges[ranges == 0] = 1e-10
        return (arr - mins) / ranges
    if chosen_method == "robust":
        med = np.median(arr, axis=0, keepdims=True)
        mad = np.median(np.abs(arr - med), axis=0, keepdims=True)
        mad[mad == 0] = 1e-10
        return (arr - med) / mad
    raise ValueError(f"Unknown normalization method: {chosen_method}")


def apply_region_weights(
    spectra: np.ndarray,
    wavenumbers: np.ndarray,
    regions_cfg: dict | None,
) -> np.ndarray:
    """Apply multiplicative weights or hard-zero exclusions to spectral regions.

    Parameters
    ----------
    spectra : np.ndarray
        2D array (n_spectra, n_wavenumbers).
    wavenumbers : np.ndarray
        1D strictly increasing wavenumber axis.
    regions_cfg : dict or None
        Config dict with optional keys:
        - ``enabled``: bool; if false or missing, returns input unchanged.
        - ``downweight_windows``: list of [lo, hi, factor] windows.
        - ``exclude_windows``: list of [lo, hi] windows (sets factor 0).

    Returns
    -------
    np.ndarray
        Weighted copy of ``spectra``.
    """
    if regions_cfg is None or not bool(regions_cfg.get("enabled", False)):
        return np.asarray(spectra, dtype=float)

    arr = np.asarray(spectra, dtype=float).copy()
    wn = np.asarray(wavenumbers, dtype=float)
    if wn.size < 2:
        return arr
    if not np.all(np.diff(wn) > 0):
        raise ValueError("wavenumbers must be strictly increasing for region weighting")

    weights = np.ones(wn.shape, dtype=float)

    for window in regions_cfg.get("downweight_windows", []) or []:
        if len(window) != 3:
            raise ValueError(f"downweight_windows entries must be [lo, hi, factor]; got {window}")
        lo, hi, factor = window
        lo, hi, factor = float(lo), float(hi), float(factor)
        if not 0.0 <= factor <= 1.0:
            raise ValueError(f"downweight factor must be in [0, 1]; got {factor}")
        if lo >= hi:
            raise ValueError(f"window bounds must satisfy lo < hi; got [{lo}, {hi}]")
        mask = (wn >= lo) & (wn <= hi)
        weights[mask] *= factor

    for window in regions_cfg.get("exclude_windows", []) or []:
        if len(window) != 2:
            raise ValueError(f"exclude_windows entries must be [lo, hi]; got {window}")
        lo, hi = float(window[0]), float(window[1])
        if lo >= hi:
            raise ValueError(f"window bounds must satisfy lo < hi; got [{lo}, {hi}]")
        mask = (wn >= lo) & (wn <= hi)
        weights[mask] = 0.0

    n_affected = int((weights != 1.0).sum())
    if n_affected > 0:
        logger.info("Region weighting affected %d/%d channels", n_affected, len(wn))
    return arr * weights


def _flag_outliers(
    metrics_df: pd.DataFrame,
    flags_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag area/total-area outliers as QC warnings, not exclusions."""
    out_cfg = _pre_cfg().get("outlier_detection", {})
    if not bool(out_cfg.get("enabled", True)) or metrics_df.empty:
        return metrics_df, flags_df

    factor = float(out_cfg.get("threshold_factor", 3.0))
    total_area = metrics_df["total_area"].to_numpy(dtype=float)
    med = float(np.median(total_area))
    mad = float(np.median(np.abs(total_area - med)))
    score = np.abs(total_area - med) / max(mad, 1e-10)
    is_outlier = score > factor

    metrics_df = metrics_df.copy()
    metrics_df["outlier_score"] = score
    metrics_df["is_outlier"] = is_outlier

    flags_df = flags_df.copy()
    if "qc_warning" not in flags_df.columns:
        flags_df["qc_warning"] = False
        flags_df["qc_warning_reason"] = ""
        flags_df["qc_warning_metrics"] = ""

    current_rows = metrics_df["row_index"].to_numpy(dtype=int)
    warning_rows = current_rows[is_outlier]
    if warning_rows.size > 0:
        flags_df.loc[warning_rows, "qc_warning"] = True
        existing_reason = flags_df.loc[warning_rows, "qc_warning_reason"].astype(str)
        new_reason = "spectral_outlier"
        combined_reason = existing_reason.where(existing_reason != "", new_reason)
        flags_df.loc[warning_rows, "qc_warning_reason"] = combined_reason
        metric_notes = [
            f"outlier_score={s:.3f};total_area={a:.3f}"
            for s, a in zip(score[is_outlier], total_area[is_outlier])
        ]
        flags_df.loc[warning_rows, "qc_warning_metrics"] = metric_notes

    return metrics_df, flags_df


def run_preprocessing(
    spectra: np.ndarray,
    wavenumbers: np.ndarray,
    *,
    region: Optional[str] = None,
    pixel_coords: np.ndarray | None = None,
    screen_dead: bool = True,
    suppress_cosmic: Optional[bool] = None,
    correct_bl: bool = True,
    smooth: Optional[bool] = None,
    interpolate: bool = False,
    filter_quality: Optional[bool] = None,
    normalize: bool = True,
    trim: Optional[int] = None,
    trim_cm: Optional[float] = None,
    norm_for: Optional[str] = None,
    record_stages: Optional[bool] = None,
) -> PreprocessResult:
    arr = np.asarray(spectra, dtype=float)
    wn = np.asarray(wavenumbers, dtype=float)
    region_key = _resolve_region_key(wn, region)
    _validate_preprocessing_config(region_key)
    pre_cfg = _pre_cfg()
    ranges_cfg = _range_cfg(region_key)
    artifact_cfg = _artifacts_cfg()
    keep_stages = bool(record_stages if record_stages is not None else artifact_cfg.get("save_stage_arrays", False))
    quality_enabled = bool(
        filter_quality
        if filter_quality is not None
        else pre_cfg.get("quality_filter", {}).get("spectral_quality", {}).get("enabled", True)
    )
    smooth_enabled = bool(
        smooth
        if smooth is not None
        else {**pre_cfg.get("smoothing", {}), **ranges_cfg.get("smoothing", {})}.get("enabled", False)
    )

    original_count = arr.shape[0]
    metrics_df = _compute_qc_metrics(arr)
    flags_df = pd.DataFrame(
        {
            "row_index": np.arange(original_count, dtype=int),
            "qc_flag": np.zeros(original_count, dtype=bool),
            "qc_reason": pd.Series(np.full(original_count, "ok", dtype=object), dtype=object),
            "qc_warning": np.zeros(original_count, dtype=bool),
            "qc_warning_reason": pd.Series(np.full(original_count, "", dtype=object), dtype=object),
            "qc_warning_metrics": pd.Series(np.full(original_count, "", dtype=object), dtype=object),
        }
    )
    retained_mask = np.ones(original_count, dtype=bool)
    stage_arrays: dict[str, np.ndarray] = {}
    if keep_stages:
        stage_arrays["raw"] = arr.copy()

    result = arr.copy()
    if screen_dead:
        result, retained_mask, initial_flags = _apply_initial_qc(result, metrics_df)
        flags_df.loc[:, "qc_flag"] = initial_flags["qc_flag"].to_numpy()
        flags_df.loc[:, "qc_reason"] = initial_flags["qc_reason"].to_numpy()
        for col in ("qc_warning", "qc_warning_reason", "qc_warning_metrics"):
            if col in initial_flags.columns:
                flags_df.loc[:, col] = initial_flags[col].to_numpy()
        metrics_df = metrics_df.loc[retained_mask].reset_index(drop=True)
    else:
        metrics_df = metrics_df.reset_index(drop=True)

    cr_cfg = pre_cfg.get("cosmic_ray", {})
    cosmic_suppress_enabled = bool(
        suppress_cosmic if suppress_cosmic is not None else cr_cfg.get("enabled", False)
    )
    if result.size > 0:
        retained_pixel_coords = pixel_coords[retained_mask] if pixel_coords is not None else None
        spike_mask, replacement = _detect_cosmic_spikes(result, retained_pixel_coords, cr_cfg)
        cosmic_counts = spike_mask.sum(axis=1).astype(int)
        n_channels = max(result.shape[1], 1)
        metrics_df.loc[:, "n_cosmic_channels"] = cosmic_counts
        metrics_df.loc[:, "cosmic_spike_count"] = cosmic_counts
        metrics_df.loc[:, "cosmic_spike_fraction"] = cosmic_counts / n_channels
        if cosmic_suppress_enabled:
            result = _suppress_cosmic_spikes(result, spike_mask, replacement)
            suppressed_counts = spike_mask.sum(axis=1).astype(int)
            metrics_df.loc[:, "n_suppressed_cosmic_channels"] = suppressed_counts
            metrics_df.loc[:, "fraction_corrected"] = spike_mask.mean(axis=1).astype(float)
            if keep_stages:
                stage_arrays["cosmic_corrected"] = result.copy()
        else:
            metrics_df.loc[:, "n_suppressed_cosmic_channels"] = 0
            metrics_df.loc[:, "fraction_corrected"] = 0.0

    if quality_enabled and result.size > 0:
        quality_mask, quality_stats = _spectral_quality_mask_with_edge_guard(
            result,
            wavenumbers=wn,
            region=region_key,
            trim=trim,
            trim_cm=trim_cm,
        )
        kept_indices = np.where(retained_mask)[0]
        dropped = kept_indices[~quality_mask]
        flags_df.loc[dropped, "qc_flag"] = True
        flags_df.loc[dropped, "qc_reason"] = "quality_filter"
        retained_mask[kept_indices[~quality_mask]] = False
        result = result[quality_mask]
        metrics_df = metrics_df.loc[quality_mask].reset_index(drop=True)
    else:
        quality_stats = {}

    if trim_cm is not None:
        if float(trim_cm) > 0:
            result, wn = trim_edges_cm(result, wn, edge_trim_cm=float(trim_cm))
    elif trim is not None:
        trim_val = int(trim)
        if trim_val > 0:
            result, wn = trim_edges(result, wn, trim=trim_val)
    else:
        edge_cfg = pre_cfg.get("edge_trim", {})
        edge_trim_cm = edge_cfg.get("edge_trim_cm", None)
        if edge_trim_cm is not None and float(edge_trim_cm) > 0:
            result, wn = trim_edges_cm(result, wn, edge_trim_cm=float(edge_trim_cm))
        else:
            trim_val = int(edge_cfg.get("trim", 0))
            if trim_val > 0:
                result, wn = trim_edges(result, wn, trim=trim_val)

    if correct_bl and result.size > 0:
        bl_cfg = {**pre_cfg.get("baseline", {}), **ranges_cfg.get("baseline", {})}
        result, baselines = correct_baseline(
            result,
            method=bl_cfg.get("method"),
            lam=bl_cfg.get("lam"),
            p=bl_cfg.get("p"),
            arpls_lam=bl_cfg.get("arpls", {}).get("lam") if "arpls" in bl_cfg else None,
            arpls_ratio=bl_cfg.get("arpls", {}).get("ratio") if "arpls" in bl_cfg else None,
            arpls_max_iter=bl_cfg.get("arpls", {}).get("max_iter") if "arpls" in bl_cfg else None,
            return_baseline=True,
        )
        if len(metrics_df) == len(baselines):
            baseline_level = np.nanmean(np.abs(baselines), axis=1)
            corrected_level = np.nanmean(np.abs(result), axis=1)
            total_level = baseline_level + corrected_level
            with np.errstate(divide="ignore", invalid="ignore"):
                baseline_fraction = np.divide(
                    baseline_level,
                    total_level,
                    out=np.zeros_like(baseline_level, dtype=float),
                    where=total_level > 0,
                )
            metrics_df.loc[:, "baseline_fraction"] = baseline_fraction
        if keep_stages:
            stage_arrays["baseline"] = baselines
            stage_arrays["baseline_corrected"] = result.copy()

    spike_cfg = pre_cfg.get("spike_repair", {})
    if bool(spike_cfg.get("enabled", True)) and result.size > 0:
        result, repair_mask = repair_isolated_spikes_with_mask(
            result,
            z_threshold=float(spike_cfg.get("z_threshold", 8.0)),
            window=int(spike_cfg.get("window", 5)),
            max_corrected_fraction=float(spike_cfg.get("max_corrected_fraction", 0.02)),
        )
        repair_counts = repair_mask.sum(axis=1).astype(int)
        metrics_df.loc[:, "n_spike_repaired_channels"] = repair_counts
        metrics_df.loc[:, "spike_repair_fraction"] = repair_mask.mean(axis=1).astype(float)
        if keep_stages:
            stage_arrays["spike_repaired"] = result.copy()

    if smooth_enabled and result.size > 0:
        sm_cfg = {**pre_cfg.get("smoothing", {}), **ranges_cfg.get("smoothing", {})}
        result = smooth_spectra(
            result,
            method=sm_cfg.get("method"),
            window_length=sm_cfg.get("window_length"),
            polyorder=sm_cfg.get("polyorder"),
        )
        if keep_stages:
            stage_arrays["smoothed"] = result.copy()

    if interpolate and result.size > 0:
        result, wn = interpolate_spectra(result, wn)

    metrics_df, flags_df = _flag_outliers(metrics_df, flags_df)

    if normalize and result.size > 0:
        result = normalize_spectra(result, mode=norm_for)
        if keep_stages:
            stage_arrays["normalized"] = result.copy()

    return PreprocessResult(
        spectra=result,
        wavenumbers=wn,
        retained_mask=retained_mask,
        qc_flags=flags_df,
        qc_metrics=metrics_df.reset_index(drop=True),
        stage_arrays=stage_arrays,
        metadata={
            "region": region_key,
            "n_input": int(original_count),
            "n_retained": int(retained_mask.sum()),
            "quality_stats": quality_stats,
            "normalization_method": _normalization_method(None, norm_for),
            "cosmic_suppression_enabled": cosmic_suppress_enabled,
            "isolated_spike_repair_enabled": bool(spike_cfg.get("enabled", True)),
        },
    )


def preprocess_spectra(
    spectra: np.ndarray,
    wavenumbers: np.ndarray,
    screen_dead: bool = True,
    suppress_cosmic: Optional[bool] = None,
    correct_bl: bool = True,
    smooth: Optional[bool] = None,
    interpolate: bool = False,
    filter_quality: Optional[bool] = None,
    normalize: bool = True,
    trim: Optional[int] = None,
    trim_cm: Optional[float] = None,
    norm_for: Optional[str] = None,
    region: Optional[str] = None,
    pixel_coords: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    result = run_preprocessing(
        spectra,
        wavenumbers,
        region=region,
        pixel_coords=pixel_coords,
        screen_dead=screen_dead,
        suppress_cosmic=suppress_cosmic,
        correct_bl=correct_bl,
        smooth=smooth,
        interpolate=interpolate,
        filter_quality=filter_quality,
        normalize=normalize,
        trim=trim,
        trim_cm=trim_cm,
        norm_for=norm_for,
        record_stages=False,
    )
    return result.spectra, result.wavenumbers


def sample_baseline_qc_artifacts(
    result: PreprocessResult,
    output_dir: str | Path,
    stem: str,
    n_samples: int = 5,
    seed: int = 42,
    condition_labels: dict[int, str] | None = None,
) -> dict[str, str]:
    """Write representative raw/baseline/corrected/residual baseline QC artifacts.

    Sampling is deterministic, seeded, and bounded.  When ``condition_labels``
    maps row index to condition, the sampler attempts to pick at least one
    spectrum per observed condition up to ``n_samples``.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    raw = result.stage_arrays.get("raw")
    baselines = result.stage_arrays.get("baseline")
    corrected = result.stage_arrays.get("baseline_corrected")
    wn = result.wavenumbers
    if raw is None or baselines is None or corrected is None or wn.size == 0:
        return written

    n_rows = raw.shape[0]
    if n_rows == 0:
        return written

    retained_indices = np.where(result.retained_mask)[0]
    if retained_indices.size == 0:
        return written

    rng = np.random.default_rng(seed)
    if condition_labels:
        indices_by_condition: dict[str, list[int]] = {}
        for idx in retained_indices:
            label = str(condition_labels.get(int(idx), "unknown"))
            indices_by_condition.setdefault(label, []).append(int(idx))
        selected_global: list[int] = []
        for label in sorted(indices_by_condition):
            pool = indices_by_condition[label]
            k = max(1, min(len(pool), n_samples // max(1, len(indices_by_condition))))
            selected_global.extend(sorted(rng.choice(pool, size=k, replace=False).tolist()))
        selected_global = sorted(set(selected_global))
        if len(selected_global) > n_samples:
            selected_global = sorted(rng.choice(selected_global, size=n_samples, replace=False).tolist())
    else:
        k = min(int(retained_indices.size), n_samples)
        selected_global = sorted(rng.choice(retained_indices, size=k, replace=False).tolist())

    local_index_map = {int(global_idx): int(local_idx) for local_idx, global_idx in enumerate(retained_indices)}
    for sample_global in selected_global:
        local_idx = local_index_map[sample_global]
        base = output_path / f"{stem}_sample{sample_global}"
        np.savez_compressed(
            f"{base}_baseline_qc.npz",
            wavenumbers=wn,
            raw=raw[sample_global],
            baseline=baselines[local_idx],
            corrected=corrected[local_idx],
            residual=corrected[local_idx] - baselines[local_idx],
        )
        written[f"baseline_qc_sample_{sample_global}"] = str(f"{base}_baseline_qc.npz")

    return written


def save_preprocessing_artifacts(
    result: PreprocessResult,
    output_dir: str | Path,
    stem: str,
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    if bool(_artifacts_cfg().get("save_qc_tables", True)):
        qc_metrics_path = output_path / f"{stem}_qc_metrics.csv"
        qc_flags_path = output_path / f"{stem}_qc_flags.csv"
        manifest_path = output_path / f"{stem}_preprocessing_manifest.json"
        result.qc_metrics.to_csv(qc_metrics_path, index=False)
        result.qc_flags.to_csv(qc_flags_path, index=False)
        manifest_path.write_text(json.dumps(result.metadata, indent=2, default=str), encoding="utf-8")
        written["qc_metrics_csv"] = str(qc_metrics_path)
        written["qc_flags_csv"] = str(qc_flags_path)
        written["preprocessing_manifest_json"] = str(manifest_path)

    if bool(_artifacts_cfg().get("save_stage_arrays", False)) and result.stage_arrays:
        arrays_path = output_path / f"{stem}_stage_arrays.npz"
        payload = {"wavenumbers": result.wavenumbers, "retained_mask": result.retained_mask}
        payload.update({name: np.asarray(values, dtype=float) for name, values in result.stage_arrays.items()})
        np.savez_compressed(arrays_path, **payload)
        written["stage_arrays_npz"] = str(arrays_path)

    return written


def coffee_ring_mask(
    pixel_coords: np.ndarray,
    spectra: np.ndarray,
    valid_mask: np.ndarray,
    intensity_ratio_threshold: Optional[float] = None,
    min_surviving_pixels: Optional[int] = None,
) -> np.ndarray:
    cr_cfg = _pre_cfg().get("quality_filter", {}).get("coffee_ring", {})
    ratio_thr = float(
        intensity_ratio_threshold if intensity_ratio_threshold is not None else cr_cfg.get("intensity_ratio_threshold", 2.0)
    )
    min_pix = int(min_surviving_pixels if min_surviving_pixels is not None else cr_cfg.get("min_surviving_pixels", 4))

    valid_idx = np.where(valid_mask)[0]
    if valid_idx.size < 4:
        logger.warning("Coffee-ring filter skipped: only %d valid pixels", valid_idx.size)
        return valid_mask.copy()

    valid_coords = pixel_coords[valid_idx]
    try:
        hull = ConvexHull(valid_coords)
        hull_vertex_local = set(hull.vertices)
    except Exception:
        logger.warning("Coffee-ring filter skipped: ConvexHull construction failed")
        return valid_mask.copy()

    is_border_local = np.array([i in hull_vertex_local for i in range(valid_idx.size)], dtype=bool)
    is_interior_local = ~is_border_local
    interior_idx = valid_idx[is_interior_local]
    border_idx = valid_idx[is_border_local]
    if interior_idx.size == 0:
        logger.warning("Coffee-ring filter skipped: no interior pixels")
        return valid_mask.copy()

    pixel_areas = np.trapezoid(np.abs(spectra), axis=1, dx=1.0)
    interior_median = float(np.median(pixel_areas[interior_idx]))
    if interior_median == 0:
        interior_median = 1e-10
    flagged_global_idx = border_idx[pixel_areas[border_idx] > ratio_thr * interior_median]
    n_surviving = int(valid_mask.sum() - flagged_global_idx.size)
    if n_surviving < min_pix:
        logger.warning("Coffee-ring filter skipped: only %d pixels would survive (min=%d)", n_surviving, min_pix)
        return valid_mask.copy()

    updated_mask = valid_mask.copy()
    updated_mask[flagged_global_idx] = False
    return updated_mask


def resolve_region_key(region: str) -> str:
    """Public, stable entry point mapping a range alias to its canonical region key."""
    return _resolve_region_key(np.array([]), region)


def retention_guard_config() -> dict[str, int]:
    ret_cfg = _pre_cfg().get("retention", {})
    return {
        "min_pixels_per_area": int(ret_cfg.get("min_pixels_per_area", 4)),
        "min_pixels_per_subject": int(ret_cfg.get("min_pixels_per_subject", 10)),
    }


def intersect_region_masks(mask_1300: np.ndarray, mask_2850: np.ndarray) -> np.ndarray:
    if mask_1300.shape != mask_2850.shape:
        raise ValueError(f"Region masks must have the same shape; got {mask_1300.shape} and {mask_2850.shape}")
    return np.asarray(mask_1300, dtype=bool) & np.asarray(mask_2850, dtype=bool)
