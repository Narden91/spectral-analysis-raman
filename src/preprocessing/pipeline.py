"""
Shared spectral preprocessing pipeline for Raman data.

The conservative default path follows the repo's biomarker-discovery guidance:
keep spectra at pixel/file level until QC is complete, remove only clear
instrument artefacts, and make every step configurable from YAML.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from pybaselines._banded_utils import diff_penalty_diagonals
from pybaselines.smooth import snip
from pybaselines.utils import ParameterWarning
from pybaselines.whittaker import arpls, asls
from scipy.interpolate import interp1d
from scipy.linalg import cho_solve_banded, cholesky_banded, solveh_banded
from scipy.linalg.lapack import dpbsv
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, savgol_filter
from scipy.special import expit

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


@dataclass(frozen=True)
class _PipelineOptions:
    quality_enabled: bool
    smooth_enabled: bool
    keep_stages: bool


def flatten_spectra_rows(spectra: np.ndarray) -> np.ndarray:
    """Return a 2D (n_rows, n_channels) view of spectra-like input."""
    arr = np.asarray(spectra, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    if arr.ndim > 2:
        return arr.reshape(-1, arr.shape[-1])
    return arr.reshape(0, 0)


def _pre_cfg(cfg: dict | None = None) -> dict:
    return (config if cfg is None else cfg).get("preprocessing", {})


def _range_cfg(region_key: str, cfg: dict | None = None) -> dict:
    return _pre_cfg(cfg=cfg).get("ranges", {}).get(region_key, {})


def _artifacts_cfg(cfg: dict | None = None) -> dict:
    return _pre_cfg(cfg=cfg).get("artifacts", {})


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


def _normalization_method(
    method: Optional[str], mode: Optional[str], region_key: Optional[str] = None, cfg: dict | None = None,
) -> str:
    nm_cfg = _pre_cfg(cfg=cfg).get("normalization", {})
    if region_key is not None:
        range_nm_cfg = _range_cfg(region_key, cfg=cfg).get("normalization", {})
        if "method" in range_nm_cfg:
            if mode == "cross":
                return "none"
            return str(range_nm_cfg["method"]).strip().lower()

    mode_map = {
        "cross": "none",
        **nm_cfg.get("modes", {}),
    }
    if mode is not None:
        if mode not in mode_map:
            raise ValueError(f"Unknown normalization mode: {mode}")
        return str(mode_map[mode]).strip().lower()
    return str(method if method is not None else nm_cfg.get("method", "vector")).strip().lower()


def _validate_preprocessing_config(region_key: str, cfg: dict | None = None) -> None:
    pre_cfg = _pre_cfg(cfg=cfg)
    nm_cfg = pre_cfg.get("normalization", {})
    norm_method = _normalization_method(nm_cfg.get("method"), None, region_key, cfg=cfg)
    allowed = {"none", "area", "snv", "vector", "minmax", "robust"}
    if norm_method not in allowed:
        raise ValueError(f"Unknown normalization method in config: {norm_method}")
    if norm_method == "snv" and bool(nm_cfg.get("vector_norm_after_snv", False)):
        raise ValueError("SNV and vector normalization cannot be stacked in one preprocessing pass")

    sm_cfg = {**pre_cfg.get("smoothing", {}), **_range_cfg(region_key, cfg=cfg).get("smoothing", {})}
    if bool(sm_cfg.get("enabled", False)):
        window_length = int(sm_cfg.get("window_length", 5))
        polyorder = int(sm_cfg.get("polyorder", 2))
        if window_length < 3:
            raise ValueError("Smoothing window_length must be >= 3")
        if window_length % 2 == 0:
            raise ValueError("Smoothing window_length must be odd (Savitzky-Golay requirement)")
        if polyorder >= window_length:
            raise ValueError("Smoothing polyorder must be smaller than window_length")

    bl_cfg = {**pre_cfg.get("baseline", {}), **_range_cfg(region_key, cfg=cfg).get("baseline", {})}
    bl_method = str(bl_cfg.get("method", "arpls")).strip().lower()
    allowed_bl = {"arpls", "als", "snip"}
    if bl_method not in allowed_bl:
        raise ValueError(f"Unknown baseline method in config: {bl_method!r}. Must be one of {allowed_bl}")
    if bl_method == "arpls":
        arpls_cfg = bl_cfg.get("arpls", {})
        lam = float(arpls_cfg.get("lam", bl_cfg.get("lam", 1e6)))
        ratio = float(arpls_cfg.get("ratio", 1e-3))
        max_iter = int(arpls_cfg.get("max_iter", 100))
        if not np.isfinite(lam) or lam <= 0:
            raise ValueError("arPLS lam must be finite and > 0")
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError("arPLS ratio must be finite and > 0")
        if max_iter < 1:
            raise ValueError("arPLS max_iter must be >= 1")
    elif bl_method == "als":
        lam = float(bl_cfg.get("lam", 1e6))
        p = float(bl_cfg.get("p", 0.01))
        if not np.isfinite(lam) or lam <= 0:
            raise ValueError("ALS lam must be finite and > 0")
        if not np.isfinite(p) or not 0 < p < 1:
            raise ValueError("ALS p must lie strictly between 0 and 1")


def _spectral_quality_cfg(cfg: dict | None = None) -> dict:
    return _pre_cfg(cfg=cfg).get("quality_filter", {}).get("spectral_quality", {})


def _resolve_pipeline_options(
    region_key: str,
    *,
    filter_quality: Optional[bool],
    smooth: Optional[bool],
    record_stages: Optional[bool],
    cfg: dict | None = None,
) -> _PipelineOptions:
    pre_cfg = _pre_cfg(cfg=cfg)
    ranges_cfg = _range_cfg(region_key, cfg=cfg)
    artifact_cfg = _artifacts_cfg(cfg=cfg)
    quality_enabled = bool(
        filter_quality
        if filter_quality is not None
        else _spectral_quality_cfg(cfg=cfg).get("enabled", True)
    )
    smooth_enabled = bool(
        smooth
        if smooth is not None
        else {**pre_cfg.get("smoothing", {}), **ranges_cfg.get("smoothing", {})}.get("enabled", False)
    )
    keep_stages = bool(record_stages if record_stages is not None else artifact_cfg.get("save_stage_arrays", False))
    return _PipelineOptions(
        quality_enabled=quality_enabled,
        smooth_enabled=smooth_enabled,
        keep_stages=keep_stages,
    )


def _noise_window(spectra: np.ndarray, noise_channels: int) -> np.ndarray:
    n_channels = spectra.shape[1]
    window_width = min(n_channels, max(int(noise_channels), 1))
    return spectra[:, n_channels - window_width :]


def _build_qc_flags(original_count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_index": np.arange(original_count, dtype=int),
            "qc_flag": np.zeros(original_count, dtype=bool),
            "qc_reason": pd.Series(np.full(original_count, "ok", dtype=object), dtype=object),
            "qc_warning": np.zeros(original_count, dtype=bool),
            "qc_warning_reason": pd.Series(np.full(original_count, "", dtype=object), dtype=object),
            "qc_warning_metrics": pd.Series(np.full(original_count, "", dtype=object), dtype=object),
        }
    )


def _add_qc_reason(flags_df: pd.DataFrame, row_indices: np.ndarray, reason: str) -> None:
    """Record one hard-failure reason per row without duplicating prior reasons."""
    for row_index in np.asarray(row_indices, dtype=int):
        current = str(flags_df.at[row_index, "qc_reason"] or "")
        parts = [] if current in {"", "ok"} else [part.strip() for part in current.split(";") if part.strip()]
        if reason not in parts:
            parts.append(reason)
        flags_df.at[row_index, "qc_reason"] = "; ".join(parts) if parts else "ok"


def _update_qc_metrics(audit_metrics_df: pd.DataFrame, stage_metrics_df: pd.DataFrame) -> None:
    """Persist metrics for retained and subsequently rejected input rows."""
    if stage_metrics_df.empty:
        return
    row_indices = stage_metrics_df["row_index"].to_numpy(dtype=int)
    n_audit = len(audit_metrics_df)
    standard_index = audit_metrics_df.index.equals(pd.RangeIndex(n_audit))
    full_range = (
        standard_index
        and len(row_indices) == n_audit
        and np.array_equal(row_indices, np.arange(n_audit))
    )
    if full_range:
        for column in stage_metrics_df.columns:
            if column != "row_index":
                audit_metrics_df[column] = stage_metrics_df[column].to_numpy()
        return

    if standard_index and np.all((row_indices >= 0) & (row_indices < n_audit)):
        for column in stage_metrics_df.columns:
            if column == "row_index":
                continue
            vals = stage_metrics_df[column].to_numpy()
            if column in audit_metrics_df.columns:
                existing = audit_metrics_df[column].to_numpy(copy=True)
                if np.can_cast(vals.dtype, existing.dtype, casting="safe"):
                    existing[row_indices] = vals
                    audit_metrics_df[column] = existing
                else:
                    audit_metrics_df.loc[row_indices, column] = vals
            else:
                out_dtype = vals.dtype if np.issubdtype(vals.dtype, np.floating) else float
                arr = np.full(n_audit, np.nan, dtype=out_dtype)
                arr[row_indices] = vals
                audit_metrics_df[column] = arr
        return

    for column in stage_metrics_df.columns:
        if column != "row_index":
            audit_metrics_df.loc[row_indices, column] = stage_metrics_df[column].to_numpy()


def _compute_qc_metrics(spectra: np.ndarray, cfg: dict | None = None) -> pd.DataFrame:
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
                "clipped_fraction",
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
    all_finite = finite_mask.all(axis=1)
    if all_finite.any():
        fluorescence_level[all_finite] = np.median(spectra[all_finite], axis=1)
    for idx in np.flatnonzero(has_finite & ~all_finite):
        row = spectra[idx]
        fluorescence_level[idx] = float(np.median(row[finite_mask[idx]]))

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

    noise_channels = int(_spectral_quality_cfg(cfg=cfg).get("noise_channels", 20))
    noise_window = _noise_window(spectra, noise_channels)
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
    cfg: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    dp_cfg = _pre_cfg(cfg=cfg).get("dead_pixel", {})
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
        logger.debug("Initial QC removed %d/%d spectra", int(invalid.sum()), spectra.shape[0])
    return spectra[valid_mask], valid_mask, flags_df


def _local_cosmic_detection(
    spectra: np.ndarray,
    z_threshold: float,
    window: int,
    max_width: int,
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
    candidate_mask = residual > z_threshold * robust_sigma
    return _limit_spike_width(candidate_mask, max_width), local_median


def _limit_spike_width(mask: np.ndarray, max_width: int) -> np.ndarray:
    """Keep only contiguous positive excursions no wider than ``max_width``."""
    limited = np.zeros_like(mask, dtype=bool)
    maximum = max(1, int(max_width))
    for row_index in np.flatnonzero(mask.any(axis=1)):
        row_mask = mask[row_index]
        padded = np.r_[False, row_mask, False].astype(int)
        edges = np.diff(padded)
        starts = np.flatnonzero(edges == 1)
        stops = np.flatnonzero(edges == -1)
        for start, stop in zip(starts, stops):
            if stop - start <= maximum:
                limited[row_index, start:stop] = True
    return limited


def _spatial_cosmic_detection(
    spectra: np.ndarray,
    pixel_coords: np.ndarray,
    z_threshold: float,
    n_neighbors: int,
    max_width: int,
    window: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect cosmic-ray spikes using spatial neighbor medians.

    Falls back to local median detection when coordinates are missing or invalid.
    """
    if spectra.size == 0 or pixel_coords is None or len(pixel_coords) != len(spectra):
        return _local_cosmic_detection(spectra, z_threshold=z_threshold, window=window, max_width=max_width)

    coords = np.asarray(pixel_coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 3 or not np.isfinite(coords).all():
        return _local_cosmic_detection(spectra, z_threshold=z_threshold, window=window, max_width=max_width)

    diffs = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(distances, np.inf)
    neighbor_count = max(1, min(int(n_neighbors), len(coords) - 1))
    nearest = np.argpartition(distances, kth=neighbor_count - 1, axis=1)[:, :neighbor_count]

    replacement = np.asarray(np.median(spectra[nearest], axis=1), dtype=float)
    residual = spectra - replacement
    mad = np.median(np.abs(residual), axis=1, keepdims=True)
    mad[mad == 0] = 1e-10
    local_spikes = residual > (1.4826 * mad * z_threshold)
    spike_mask = _limit_spike_width(local_spikes, max_width)
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
    max_width = int(cr_cfg.get("max_width", 3))
    fallback_window = int(cr_cfg.get("window", 2))
    method = str(cr_cfg.get("method", "spatial")).strip().lower()
    if method == "spatial" and pixel_coords is not None:
        neighbor_count = int(cr_cfg.get("n_neighbors", 8))
        return _spatial_cosmic_detection(
            arr,
            pixel_coords=np.asarray(pixel_coords, dtype=float),
            z_threshold=thr,
            n_neighbors=neighbor_count,
            max_width=max_width,
            window=fallback_window,
        )
    return _local_cosmic_detection(arr, z_threshold=thr, window=fallback_window, max_width=max_width)


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




_MIN_FLOAT = float(np.finfo(float).eps)
_ARPLS_EARLY_EXIT_MSG = (
    'almost all baseline points are below the data, indicating that "tol" is too low and/or "max_iter" is too high'
)


@lru_cache(maxsize=64)
def _banded_diff2_penalty(n_channels: int, lam: float) -> np.ndarray:
    """Return read-only lower-banded ``lam * D^T D`` penalty for order-2 differences."""
    pen = np.ascontiguousarray(
        float(lam) * diff_penalty_diagonals(int(n_channels), 2, lower_only=True),
        dtype=float,
    )
    pen.setflags(write=False)
    return pen


def _fast_whittaker_batch(
    arr: np.ndarray,
    *,
    method: str,
    lam: float,
    p: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, list[int]]:
    """Exact-parity Whittaker baseline solver (`arpls`/`als`) with cached banded penalty."""
    if arr.shape[0] == 0:
        return np.empty_like(arr, dtype=float), []
    if not np.isfinite(arr).all():
        raise ValueError("array must not contain infs or NaNs")

    n_rows, n_ch = arr.shape
    if (
        n_ch < 3
        or not np.isfinite(lam)
        or lam <= 0.0
        or max_iter < 0
        or (method == "als" and (not np.isfinite(p) or not (0.0 < p < 1.0)))
    ):
        baselines = np.empty_like(arr, dtype=float)
        warned_indices: list[int] = []
        for index, row in enumerate(arr):
            if method == "als":
                baseline, _ = asls(row, lam=lam, p=p, max_iter=max_iter, tol=tol)
            else:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ParameterWarning)
                    baseline, _ = arpls(row, lam=lam, diff_order=2, max_iter=max_iter, tol=tol)
                if caught:
                    warned_indices.append(index)
            baselines[index] = baseline
        return baselines, warned_indices

    pen = _banded_diff2_penalty(n_ch, lam)
    p0 = pen[0]
    ab_buf = np.empty_like(pen)
    rhs_buf = np.empty(n_ch, dtype=float)
    expit_buf = np.empty(n_ch, dtype=float) if method == "arpls" else None
    diff_buf = np.empty(n_ch, dtype=float)
    w_buf = np.empty(n_ch, dtype=float)
    ones_w = np.ones(n_ch, dtype=float)
    ones_norm = math.sqrt(float(n_ch))
    baselines = np.empty((n_rows, n_ch), dtype=float)
    warned_indices = []
    one_minus_p = 1.0 - p

    np.copyto(ab_buf, pen)
    ab_buf[0] = p0 + 1.0
    cb0 = cholesky_banded(ab_buf, lower=True, overwrite_ab=True, check_finite=False)
    b0_all = np.ascontiguousarray(cho_solve_banded((cb0, True), arr.T, check_finite=False).T)

    for k in range(n_rows):
        y = arr[k]
        w_arr = ones_w
        w_norm = ones_norm
        b = b0_all[k]
        for it in range(max_iter + 1):
            if it > 0:
                np.copyto(ab_buf, pen)
                ab_buf[0] = p0 + w_arr
                np.multiply(w_arr, y, out=rhs_buf)
                _, b, info = dpbsv(ab_buf, rhs_buf, lower=True, overwrite_ab=True, overwrite_b=True)
                if info != 0:
                    np.copyto(ab_buf, pen)
                    ab_buf[0] = p0 + w_arr
                    np.multiply(w_arr, y, out=rhs_buf)
                    b = solveh_banded(ab_buf, rhs_buf, overwrite_ab=True, overwrite_b=True, lower=True, check_finite=False)
            if method == "arpls":
                res = y - b
                neg = res[res < 0]
                if neg.size < 2:
                    warned_indices.append(k)
                    break
                s = float(neg.std(ddof=1))
                if s == 0.0:
                    s = _MIN_FLOAT
                assert expit_buf is not None
                np.subtract(res, 2.0 * s - float(neg.mean()), out=expit_buf)
                expit_buf *= -2.0 / s
                nw = expit(expit_buf, out=expit_buf)
                np.subtract(nw, w_arr, out=diff_buf)
                rel = math.sqrt(float(np.dot(diff_buf, diff_buf))) / max(w_norm, _MIN_FLOAT)
                if rel < tol:
                    break
                if it < max_iter:
                    np.copyto(w_buf, nw)
                    w_arr = w_buf
                    w_norm = math.sqrt(float(np.dot(w_arr, w_arr)))
            else:
                nw = np.where(y > b, p, one_minus_p)
                np.subtract(nw, w_arr, out=diff_buf)
                rel = math.sqrt(float(np.dot(diff_buf, diff_buf))) / max(w_norm, _MIN_FLOAT)
                if rel < tol:
                    break
                if it < max_iter:
                    w_arr = nw
                    w_norm = math.sqrt(float(np.dot(w_arr, w_arr)))
        baselines[k] = b

    return baselines, warned_indices


def correct_baseline(
    spectra: np.ndarray,
    method: Optional[str] = None,
    lam: Optional[float] = None,
    p: Optional[float] = None,
    arpls_lam: Optional[float] = None,
    arpls_ratio: Optional[float] = None,
    arpls_max_iter: Optional[int] = None,
    return_baseline: bool = False,
    cfg: dict | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Baseline correction with explicit arPLS parameter support.

    Explicit arguments override settings in ``cfg``. When ``cfg`` is omitted,
    the module-level preprocessing defaults are used.
    """
    bl_cfg = _pre_cfg(cfg=cfg).get("baseline", {})
    chosen_method = str(method if method is not None else bl_cfg.get("method", "arpls")).strip().lower()
    if chosen_method not in {"als", "snip", "arpls"}:
        raise ValueError(f"Unknown baseline method: {chosen_method}")
    lam_value = float(lam if lam is not None else bl_cfg.get("lam", 1000000.0))
    p_value = float(p if p is not None else bl_cfg.get("p", 0.01))
    snip_cfg = bl_cfg.get("snip", {})
    arr = np.asarray(spectra, dtype=float)

    arpls_cfg = bl_cfg.get("arpls", {})
    resolved_arpls_lam = float(arpls_lam if arpls_lam is not None else arpls_cfg.get("lam", lam_value))
    resolved_arpls_ratio = float(arpls_ratio if arpls_ratio is not None else arpls_cfg.get("ratio", 0.001))
    resolved_arpls_max_iter = int(arpls_max_iter if arpls_max_iter is not None else arpls_cfg.get("max_iter", 100))

    if arr.ndim == 2 and chosen_method == "arpls":
        baselines, warned_indices = _fast_whittaker_batch(
            arr,
            method="arpls",
            lam=resolved_arpls_lam,
            p=0.5,
            max_iter=resolved_arpls_max_iter,
            tol=resolved_arpls_ratio,
        )
        for index in warned_indices:
            logger.debug("arPLS baseline warning for spectrum %d: %s", index, _ARPLS_EARLY_EXIT_MSG)
        corrected = arr - baselines
    elif arr.ndim == 2 and chosen_method == "als":
        baselines, _ = _fast_whittaker_batch(
            arr,
            method="als",
            lam=lam_value,
            p=p_value,
            max_iter=50,
            tol=1e-3,
        )
        corrected = arr - baselines
    else:
        corrected = np.empty_like(arr, dtype=float)
        baselines = np.empty_like(arr, dtype=float)
        max_half_window = int(snip_cfg.get("max_half_window", 50))
        decreasing = bool(snip_cfg.get("decreasing", True))
        smooth_half_window = int(snip_cfg.get("smooth_half_window", 3))
        for index, row in enumerate(arr):
            if chosen_method == "als":
                baseline, _ = asls(row, lam=lam_value, p=p_value)
            elif chosen_method == "snip":
                local_max_half_window = min(max_half_window, max((row.shape[0] - 1) // 2, 1))
                baseline, _ = snip(
                    row,
                    max_half_window=local_max_half_window,
                    decreasing=decreasing,
                    smooth_half_window=smooth_half_window,
                )
            else:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ParameterWarning)
                    baseline, _ = arpls(
                        row,
                        lam=resolved_arpls_lam,
                        diff_order=2,
                        max_iter=resolved_arpls_max_iter,
                        tol=resolved_arpls_ratio,
                    )
                for warning in caught:
                    logger.debug("arPLS baseline warning for spectrum %d: %s", index, warning.message)
            baselines[index] = baseline
            corrected[index] = row - baseline

    logger.debug("Baseline correction (%s) applied to %d spectra", chosen_method, arr.shape[0])
    if return_baseline:
        return corrected, baselines
    return corrected




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
    arr = np.asarray(spectra, dtype=float)
    if arr.size == 0:
        return arr
    sm_cfg = _pre_cfg().get("smoothing", {})
    chosen_method = str(method if method is not None else sm_cfg.get("method", "savgol")).strip().lower()
    if chosen_method != "savgol":
        raise ValueError(f"Unknown smoothing method: {chosen_method}")

    win = int(window_length if window_length is not None else sm_cfg.get("window_length", 5))
    poly = int(polyorder if polyorder is not None else sm_cfg.get("polyorder", 2))
    if win % 2 == 0:
        win += 1
    n_chan = arr.shape[1]
    if n_chan < 3:
        return arr.copy()
    if win >= n_chan:
        win = max(3, n_chan - 1 if (n_chan - 1) % 2 == 1 else n_chan - 2)
    if poly >= win:
        poly = max(1, win - 1)
    return savgol_filter(arr, window_length=win, polyorder=poly, axis=1)


def interpolate_spectra(
    spectra: np.ndarray,
    wavenumbers: np.ndarray,
    target_spacing: Optional[float] = None,
    method: Optional[str] = None,
    target_wavenumbers: Optional[np.ndarray] = None,
    cfg: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if spectra.size == 0 or wavenumbers.size == 0:
        return spectra, wavenumbers
    ip_cfg = _pre_cfg(cfg=cfg).get("interpolation", {})
    spacing = float(target_spacing if target_spacing is not None else ip_cfg.get("target_spacing", 1.0))
    interp_kind = str(method if method is not None else ip_cfg.get("method", "cubic")).strip().lower()
    if target_wavenumbers is None:
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("target_spacing must be finite and positive")
        wn_min = int(np.ceil(float(np.nanmin(wavenumbers))))
        wn_max = int(np.floor(float(np.nanmax(wavenumbers))))
        if wn_max <= wn_min:
            return spectra, wavenumbers
        wn_uniform = np.arange(wn_min, wn_max + spacing, spacing, dtype=float)
        wn_uniform = wn_uniform[wn_uniform <= wn_max]
    else:
        wn_uniform = np.asarray(target_wavenumbers, dtype=float)
        if wn_uniform.ndim != 1 or wn_uniform.size < 2 or not np.all(np.diff(wn_uniform) > 0):
            raise ValueError("target_wavenumbers must be a strictly increasing one-dimensional axis")
        if wn_uniform[0] < wavenumbers[0] - 1e-8 or wn_uniform[-1] > wavenumbers[-1] + 1e-8:
            raise ValueError("Frozen target axis extends outside the measured native axis; extrapolation is forbidden")
    if wn_uniform.size < 2:
        return spectra, wavenumbers
    if interp_kind == "cubic" and wavenumbers.size < 4:
        interp_kind = "linear"
    wn_src = np.asarray(wavenumbers, dtype=float)
    arr = np.asarray(spectra, dtype=float)
    if (
        interp_kind == "linear"
        and arr.ndim == 2
        and wn_src.ndim == 1
        and wn_src.size >= 2
        and arr.shape[1] == wn_src.size
        and wn_uniform[0] >= wn_src[0]
        and wn_uniform[-1] <= wn_src[-1]
        and np.all(np.diff(wn_src) > 0)
    ):
        idx = np.clip(np.searchsorted(wn_src, wn_uniform, side="left"), 1, wn_src.size - 1)
        lo = idx - 1
        x_lo = wn_src[lo]
        y_lo = arr[:, lo]
        slope = (arr[:, idx] - y_lo) / (wn_src[idx] - x_lo)[None, :]
        return slope * (wn_uniform - x_lo)[None, :] + y_lo, wn_uniform
    interpolator = interp1d(
        wn_src,
        arr,
        kind=interp_kind,
        axis=1,
        bounds_error=True,
    )
    return interpolator(wn_uniform), wn_uniform


_DEFAULT_ANCHOR_WINDOWS: dict[str, list[tuple[float, float]]] = {
    "fingerprint": [(995.0, 1015.0), (1435.0, 1470.0), (1630.0, 1680.0)],
    "ch_stretch": [(2840.0, 2975.0)],
}


def _anchor_windows_for_region(region: Optional[str], cfg: dict | None = None) -> list[tuple[float, float]]:
    qf_cfg = _pre_cfg(cfg=cfg).get("quality_filter", {}).get("spectral_quality", {})
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
    cfg: dict | None = None,
) -> int:
    """Resolve a deterministic SNR anchor channel from fixed wavenumber windows.

    Falls back to the legacy peak-detection anchor only when no configured
    window overlaps the supplied wavenumber axis (or no axis is given), so the
    anchor never depends on subject/batch-specific spikes.
    """
    wn = np.asarray(wavenumbers, dtype=float) if wavenumbers is not None else None
    if wn is not None and wn.size == spectra.shape[1]:
        windows = _anchor_windows_for_region(region, cfg=cfg)
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


def _spectral_quality_details(
    spectra: np.ndarray,
    *,
    snr_threshold: Optional[float] = None,
    peak_window: Optional[int] = None,
    noise_channels: Optional[int] = None,
    min_integrated_signal: Optional[float] = None,
    wavenumbers: Optional[np.ndarray] = None,
    region: Optional[str] = None,
    cfg: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    if spectra.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty, np.array([], dtype=bool), {"anchor_index": -1.0, "area_threshold": 0.0}
    qf_cfg = _spectral_quality_cfg(cfg=cfg)
    snr_thr = float(snr_threshold if snr_threshold is not None else qf_cfg.get("snr_threshold", 5.0))
    pwin = int(peak_window if peak_window is not None else qf_cfg.get("peak_window", 5))
    noise_n = int(noise_channels if noise_channels is not None else qf_cfg.get("noise_channels", 20))
    configured_floor = min_integrated_signal
    if configured_floor is None:
        configured_floor = qf_cfg.get("min_integrated_signal")

    arr = np.asarray(spectra, dtype=float)
    anchor_idx = _resolve_anchor_index(arr, wavenumbers, region, cfg=cfg)
    lo = max(0, anchor_idx - pwin)
    hi = min(arr.shape[1], anchor_idx + pwin + 1)
    signal = np.nanmax(arr[:, lo:hi], axis=1)
    noise_std = np.nanstd(_noise_window(arr, noise_n), axis=1)
    noise_std[noise_std == 0] = 1e-10
    snr = signal / noise_std
    area = np.trapezoid(np.abs(arr), axis=1, dx=1.0)
    area_threshold = float(configured_floor) if configured_floor is not None else float("nan")
    area_ok = area > area_threshold if configured_floor is not None else np.ones_like(area, dtype=bool)
    valid = np.isfinite(snr) & np.isfinite(area)
    mask = valid & (snr > snr_thr) & area_ok
    stats = {
        "anchor_index": float(anchor_idx),
        "area_threshold": area_threshold,
        "snr_median": float(np.nanmedian(snr)),
        "signal_area_median": float(np.nanmedian(area)),
        "n_valid": float(mask.sum()),
    }
    return snr, area, mask, stats


def spectral_quality_mask(
    spectra: np.ndarray,
    snr_threshold: Optional[float] = None,
    peak_window: Optional[int] = None,
    noise_channels: Optional[int] = None,
    min_area_percentile: Optional[float] = None,
    min_integrated_signal: Optional[float] = None,
    wavenumbers: Optional[np.ndarray] = None,
    region: Optional[str] = None,
    cfg: dict | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    _, _, mask, stats = _spectral_quality_details(
        spectra,
        snr_threshold=snr_threshold,
        peak_window=peak_window,
        noise_channels=noise_channels,
        min_integrated_signal=min_integrated_signal,
        wavenumbers=wavenumbers,
        region=region,
        cfg=cfg,
    )
    return mask, stats


def normalize_spectra(
    spectra: np.ndarray,
    method: Optional[str] = None,
    mode: Optional[str] = None,
    region: Optional[str] = None,
    wavenumbers: Optional[np.ndarray] = None,
    cfg: dict | None = None,
) -> np.ndarray:
    chosen_method = _normalization_method(method, mode, region, cfg=cfg)
    if chosen_method == "none":
        return np.asarray(spectra, dtype=float)
    arr = np.asarray(spectra, dtype=float)
    if chosen_method == "area":
        axis = np.asarray(wavenumbers, dtype=float) if wavenumbers is not None else None
        if axis is not None and (axis.ndim != 1 or axis.size != arr.shape[1] or not np.all(np.diff(axis) > 0)):
            raise ValueError("wavenumbers must be a strictly increasing axis matching spectra for area normalization")
        areas = np.trapezoid(np.abs(arr), x=axis, axis=1) if axis is not None else np.trapezoid(np.abs(arr), axis=1, dx=1.0)
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
    cfg: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag area/total-area outliers as QC warnings, not exclusions."""
    out_cfg = _pre_cfg(cfg=cfg).get("outlier_detection", {})
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


def _apply_quality_filter(
    result: np.ndarray,
    retained_mask: np.ndarray,
    metrics_df: pd.DataFrame,
    flags_df: pd.DataFrame,
    *,
    wavenumbers: np.ndarray,
    region_key: str,
    cfg: dict | None = None,
    audit_metrics_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    snr, area, quality_mask, quality_stats = _spectral_quality_details(
        result,
        wavenumbers=wavenumbers,
        region=region_key,
        cfg=cfg,
    )
    metrics_df.loc[:, "post_baseline_snr"] = snr
    metrics_df.loc[:, "post_baseline_signal_area"] = area
    if audit_metrics_df is not None:
        _update_qc_metrics(audit_metrics_df, metrics_df)
    kept_indices = np.where(retained_mask)[0]
    dropped = kept_indices[~quality_mask]
    flags_df.loc[dropped, "qc_flag"] = True
    _add_qc_reason(flags_df, dropped, "quality_filter")
    retained_mask = retained_mask.copy()
    retained_mask[dropped] = False
    return (
        result[quality_mask],
        retained_mask,
        metrics_df.loc[quality_mask].reset_index(drop=True),
        flags_df,
        quality_stats,
    )


def _apply_edge_trim(
    result: np.ndarray,
    wavenumbers: np.ndarray,
    *,
    trim: Optional[int],
    trim_cm: Optional[float],
    cfg: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if trim_cm is not None:
        if float(trim_cm) > 0:
            return trim_edges_cm(result, wavenumbers, edge_trim_cm=float(trim_cm))
        return result, wavenumbers

    if trim is not None:
        trim_val = int(trim)
        if trim_val > 0:
            return trim_edges(result, wavenumbers, trim=trim_val)
        return result, wavenumbers

    edge_cfg = _pre_cfg(cfg=cfg).get("edge_trim", {})
    edge_trim_cm = edge_cfg.get("edge_trim_cm", None)
    if edge_trim_cm is not None and float(edge_trim_cm) > 0:
        return trim_edges_cm(result, wavenumbers, edge_trim_cm=float(edge_trim_cm))

    trim_val = int(edge_cfg.get("trim", 0))
    if trim_val > 0:
        return trim_edges(result, wavenumbers, trim=trim_val)
    return result, wavenumbers


def _apply_baseline_stage(
    result: np.ndarray,
    metrics_df: pd.DataFrame,
    stage_arrays: dict[str, np.ndarray],
    *,
    keep_stages: bool,
    pre_cfg: dict,
    ranges_cfg: dict,
    cfg: dict | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    bl_cfg = {**pre_cfg.get("baseline", {}), **ranges_cfg.get("baseline", {})}
    if keep_stages:
        stage_arrays["baseline_input"] = result.copy()
    result, baselines = correct_baseline(
        result,
        method=bl_cfg.get("method"),
        lam=bl_cfg.get("lam"),
        p=bl_cfg.get("p"),
        arpls_lam=bl_cfg.get("arpls", {}).get("lam") if "arpls" in bl_cfg else None,
        arpls_ratio=bl_cfg.get("arpls", {}).get("ratio") if "arpls" in bl_cfg else None,
        arpls_max_iter=bl_cfg.get("arpls", {}).get("max_iter") if "arpls" in bl_cfg else None,
        return_baseline=True,
        cfg=cfg,
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

    negative = np.minimum(result, 0.0)
    negative_fraction = np.mean(result < 0, axis=1)
    absolute_area = np.sum(np.abs(result), axis=1)
    negative_area = np.sum(np.abs(negative), axis=1)
    negative_area_fraction = np.divide(
        negative_area,
        absolute_area,
        out=np.zeros_like(negative_area),
        where=absolute_area > 0,
    )
    metrics_df.loc[:, "negative_fraction"] = negative_fraction
    metrics_df.loc[:, "negative_area_fraction"] = negative_area_fraction
    return result, metrics_df


def _build_preprocess_result(
    *,
    result: np.ndarray,
    wavenumbers: np.ndarray,
    retained_mask: np.ndarray,
    flags_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    initial_metrics_df: pd.DataFrame,
    stage_arrays: dict[str, np.ndarray],
    region_key: str,
    original_count: int,
    quality_stats: dict[str, float],
    norm_for: Optional[str],
    cosmic_suppress_enabled: bool,
    spike_repair_enabled: bool,
    cfg: dict | None = None,
) -> PreprocessResult:
    full_metrics = initial_metrics_df.set_index("row_index").copy()
    stage_metrics = metrics_df.set_index("row_index")
    for column in stage_metrics.columns:
        full_metrics.loc[stage_metrics.index, column] = stage_metrics[column]
    full_metrics = full_metrics.reset_index()
    return PreprocessResult(
        spectra=result,
        wavenumbers=wavenumbers,
        retained_mask=retained_mask,
        qc_flags=flags_df,
        qc_metrics=full_metrics.reset_index(drop=True),
        stage_arrays=stage_arrays,
        metadata={
            "region": region_key,
            "n_input": int(original_count),
            "n_retained": int(retained_mask.sum()),
            "quality_stats": quality_stats,
            "normalization_method": _normalization_method(None, norm_for, region_key, cfg=cfg),
            "cosmic_suppression_enabled": cosmic_suppress_enabled,
            "isolated_spike_repair_enabled": spike_repair_enabled,
        },
    )


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
    target_wavenumbers: np.ndarray | None = None,
    filter_quality: Optional[bool] = None,
    normalize: bool = True,
    trim: Optional[int] = None,
    trim_cm: Optional[float] = None,
    norm_for: Optional[str] = None,
    record_stages: Optional[bool] = None,
    return_pre_normalization: bool = False,
    cfg: dict | None = None,
) -> PreprocessResult:
    """Preprocess one spectral block with supplied config or module defaults."""
    arr = flatten_spectra_rows(spectra)
    wn = np.asarray(wavenumbers, dtype=float)
    region_key = _resolve_region_key(wn, region)
    _validate_preprocessing_config(region_key, cfg=cfg)
    pre_cfg = _pre_cfg(cfg=cfg)
    ranges_cfg = _range_cfg(region_key, cfg=cfg)
    options = _resolve_pipeline_options(
        region_key,
        filter_quality=filter_quality,
        smooth=smooth,
        record_stages=record_stages,
        cfg=cfg,
    )

    original_count = arr.shape[0]
    metrics_df = _compute_qc_metrics(arr, cfg=cfg)
    initial_metrics_df = metrics_df.copy()
    flags_df = _build_qc_flags(original_count)
    retained_mask = np.ones(original_count, dtype=bool)
    stage_arrays: dict[str, np.ndarray] = {}
    if options.keep_stages:
        stage_arrays["raw"] = arr.copy()

    result = arr.copy()
    if screen_dead:
        result, retained_mask, initial_flags = _apply_initial_qc(result, metrics_df, cfg=cfg)
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
        _update_qc_metrics(initial_metrics_df, metrics_df)
        max_corrected_fraction = float(cr_cfg.get("max_corrected_fraction", 0.02))
        excessive_mask = metrics_df["cosmic_spike_fraction"].to_numpy(dtype=float) > max_corrected_fraction
        if excessive_mask.any():
            rejected_rows = metrics_df.loc[excessive_mask, "row_index"].to_numpy(dtype=int)
            flags_df.loc[rejected_rows, "qc_flag"] = True
            _add_qc_reason(flags_df, rejected_rows, "excessive_cosmic_correction")
            retained_mask[rejected_rows] = False
            keep_mask = ~excessive_mask
            result = result[keep_mask]
            metrics_df = metrics_df.loc[keep_mask].reset_index(drop=True)
            spike_mask = spike_mask[keep_mask]
            replacement = replacement[keep_mask]
        if cosmic_suppress_enabled:
            result = _suppress_cosmic_spikes(result, spike_mask, replacement)
            suppressed_counts = spike_mask.sum(axis=1).astype(int)
            metrics_df.loc[:, "n_suppressed_cosmic_channels"] = suppressed_counts
            metrics_df.loc[:, "fraction_corrected"] = spike_mask.mean(axis=1).astype(float)
            if options.keep_stages:
                stage_arrays["cosmic_corrected"] = result.copy()
        else:
            metrics_df.loc[:, "n_suppressed_cosmic_channels"] = 0
            metrics_df.loc[:, "fraction_corrected"] = 0.0

    result, wn = _apply_edge_trim(result, wn, trim=trim, trim_cm=trim_cm, cfg=cfg)

    if interpolate and result.size > 0:
        result, wn = interpolate_spectra(result, wn, target_wavenumbers=target_wavenumbers, cfg=cfg)

    if correct_bl and result.size > 0:
        result, metrics_df = _apply_baseline_stage(
            result,
            metrics_df,
            stage_arrays,
            keep_stages=options.keep_stages,
            pre_cfg=pre_cfg,
            ranges_cfg=ranges_cfg,
            cfg=cfg,
        )
        _update_qc_metrics(initial_metrics_df, metrics_df)

    if options.quality_enabled and result.size > 0:
        result, retained_mask, metrics_df, flags_df, quality_stats = _apply_quality_filter(
            result,
            retained_mask,
            metrics_df,
            flags_df,
            wavenumbers=wn,
            region_key=region_key,
            cfg=cfg,
            audit_metrics_df=initial_metrics_df,
        )
    else:
        quality_stats = {}

    if options.smooth_enabled and result.size > 0:
        sm_cfg = {**pre_cfg.get("smoothing", {}), **ranges_cfg.get("smoothing", {})}
        result = smooth_spectra(
            result,
            method=sm_cfg.get("method"),
            window_length=sm_cfg.get("window_length"),
            polyorder=sm_cfg.get("polyorder"),
        )
        if options.keep_stages:
            stage_arrays["smoothed"] = result.copy()

    metrics_df, flags_df = _flag_outliers(metrics_df, flags_df, cfg=cfg)
    _update_qc_metrics(initial_metrics_df, metrics_df)

    if return_pre_normalization:
        stage_arrays["pre_normalization"] = result.copy()

    if normalize and result.size > 0:
        result = normalize_spectra(result, mode=norm_for, region=region_key, wavenumbers=wn, cfg=cfg)
        if options.keep_stages:
            stage_arrays["normalized"] = result.copy()

    return _build_preprocess_result(
        result=result,
        wavenumbers=wn,
        retained_mask=retained_mask,
        flags_df=flags_df,
        metrics_df=metrics_df,
        initial_metrics_df=initial_metrics_df,
        stage_arrays=stage_arrays,
        region_key=region_key,
        original_count=original_count,
        quality_stats=quality_stats,
        norm_for=norm_for,
        cosmic_suppress_enabled=cosmic_suppress_enabled,
        spike_repair_enabled=False,
        cfg=cfg,
    )
