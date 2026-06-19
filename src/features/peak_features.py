from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths

from src.features.eda_band_features import _integrate_band

try:
    from scipy.signal._peak_finding_utils import PeakPropertyWarning
except ImportError:  # pragma: no cover - SciPy compatibility fallback
    PeakPropertyWarning = RuntimeWarning


def _canonicalize_peaks(peaks: np.ndarray, n_channels: int, bin_width: int) -> np.ndarray:
    if peaks.size == 0:
        return np.array([], dtype=int)

    if int(bin_width) <= 1:
        binned = np.asarray(peaks, dtype=int)
    else:
        binned = np.round(np.asarray(peaks, dtype=float) / float(bin_width)).astype(int) * int(bin_width)

    binned = np.clip(binned, 0, max(n_channels - 1, 0))

    # Keep first occurrence to preserve importance ordering while stabilizing labels.
    seen: set[int] = set()
    ordered_unique: list[int] = []
    for p in binned.tolist():
        p_i = int(p)
        if p_i not in seen:
            seen.add(p_i)
            ordered_unique.append(p_i)

    return np.asarray(ordered_unique, dtype=int)


def _nearest_peak_apex(spectrum: np.ndarray, center_idx: int, lo: int, hi: int) -> int | None:
    if spectrum.size < 3:
        return None

    start = max(lo, 1)
    stop = min(hi, spectrum.size - 1)

    candidates: list[int] = []
    for idx in range(start, stop):
        if spectrum[idx] >= spectrum[idx - 1] and spectrum[idx] >= spectrum[idx + 1]:
            candidates.append(idx)

    if candidates:
        return min(candidates, key=lambda idx: abs(idx - center_idx))

    local_slice = spectrum[lo:hi]
    if local_slice.size == 0:
        return None

    idx = int(lo + np.argmax(local_slice))
    idx = max(1, min(idx, spectrum.size - 2))
    return idx


def _fallback_fwhm(lo: int, hi: int) -> float:
    return float(max(hi - lo - 1, 0))


def _safe_peak_fwhm(spectrum: np.ndarray, apex_idx: int, lo: int, hi: int) -> float:
    fallback = _fallback_fwhm(lo, hi)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PeakPropertyWarning)
            widths = peak_widths(spectrum, np.array([int(apex_idx)]), rel_height=0.5)
    except ValueError:
        return fallback

    width = float(widths[0][0]) if widths[0].size > 0 else fallback
    if not np.isfinite(width) or width <= 0.0:
        return fallback
    return width


def detect_training_peaks(
    training_mean_spectrum: np.ndarray,
    prominence_percentile: float = 75.0,
    min_distance: int = 15,
    canonical_bin_width: int = 5,
) -> np.ndarray:
    """Detect data-driven peaks from the training cohort mean spectrum."""
    if training_mean_spectrum.ndim != 1:
        raise ValueError("training_mean_spectrum must be 1D")

    if training_mean_spectrum.size == 0:
        return np.array([], dtype=int)

    prominence_threshold = np.percentile(training_mean_spectrum, prominence_percentile)
    peaks, props = find_peaks(
        training_mean_spectrum,
        prominence=prominence_threshold,
        distance=min_distance,
    )

    if peaks.size == 0:
        raw = np.array([int(np.argmax(training_mean_spectrum))], dtype=int)
        return _canonicalize_peaks(raw, n_channels=training_mean_spectrum.size, bin_width=canonical_bin_width)

    prominences = props.get("prominences")
    if prominences is None or len(prominences) != len(peaks):
        raw = np.sort(peaks.astype(int))
        return _canonicalize_peaks(raw, n_channels=training_mean_spectrum.size, bin_width=canonical_bin_width)

    order = np.argsort(prominences)[::-1]
    raw = peaks[order].astype(int)
    return _canonicalize_peaks(raw, n_channels=training_mean_spectrum.size, bin_width=canonical_bin_width)


def extract_peak_features(
    vectors: np.ndarray,
    peak_indices: np.ndarray,
    peak_labels: list[str] | None = None,
    peak_window: int = 5,
    n_top_for_ratios: int = 6,
) -> pd.DataFrame:
    """Extract intensity, FWHM, derivative, and ratio features at detected peaks."""
    if vectors.ndim != 2:
        raise ValueError("vectors must be 2D")

    peaks = np.asarray(peak_indices, dtype=int)
    if peaks.size == 0:
        return pd.DataFrame(index=np.arange(vectors.shape[0]))

    if peak_labels is None:
        labels = [f"peak_{int(p)}" for p in peaks.tolist()]
    else:
        labels = [str(lab) for lab in peak_labels]
        if len(labels) != len(peaks):
            raise ValueError("peak_labels must match peak_indices length")

    rows: list[dict[str, float]] = []
    for spectrum in vectors:
        row: dict[str, float] = {}
        intensities: list[tuple[str, float]] = []

        for j, peak_idx in enumerate(peaks):
            label = labels[j]
            lo = max(0, int(peak_idx) - peak_window)
            hi = min(spectrum.size, int(peak_idx) + peak_window + 1)

            intensity = float(np.mean(spectrum[lo:hi]))
            # Local central-difference derivative at peak (avoids full-spectrum gradient)
            p = int(peak_idx)
            if spectrum.size > 2:
                lo_d = max(0, p - 1)
                hi_d = min(spectrum.size - 1, p + 1)
                deriv = float((spectrum[hi_d] - spectrum[lo_d]) / max(hi_d - lo_d, 1))
            else:
                deriv = 0.0

            apex_idx = _nearest_peak_apex(spectrum, int(peak_idx), lo=lo, hi=hi)
            if apex_idx is not None:
                fwhm = _safe_peak_fwhm(spectrum, int(apex_idx), lo, hi)
            else:
                # Fallback to local window width if no valid local apex can be identified.
                fwhm = _fallback_fwhm(lo, hi)

            row[f"{label}_intensity"] = intensity
            row[f"{label}_fwhm"] = fwhm
            row[f"{label}_derivative"] = deriv
            intensities.append((label, intensity))

        n_ratio = min(n_top_for_ratios, len(intensities))
        for a in range(n_ratio):
            for b in range(a + 1, n_ratio):
                label_a, val_a = intensities[a]
                label_b, val_b = intensities[b]
                ratio = float(val_a / val_b) if abs(val_b) > 1e-12 else 0.0
                row[f"ratio_{label_a}_{label_b}"] = ratio

        rows.append(row)

    return pd.DataFrame(rows)


def extract_relative_peak_features(
    vectors: np.ndarray,
    peak_indices: np.ndarray,
    peak_labels: list[str] | None = None,
    peak_window: int = 5,
    reference_idx: int | None = None,
    wavenumbers: np.ndarray | None = None,
    eps: float = 1e-9,
) -> pd.DataFrame:
    if vectors.ndim != 2:
        raise ValueError("vectors must be 2D")

    peaks = np.asarray(peak_indices, dtype=int)
    if peaks.size == 0:
        return pd.DataFrame(index=np.arange(vectors.shape[0]))

    if peak_labels is None:
        labels = [f"peak_{int(p)}" for p in peaks.tolist()]
    else:
        labels = [str(lab) for lab in peak_labels]
        if len(labels) != len(peaks):
            raise ValueError("peak_labels must match peak_indices length")

    ref_idx = None if reference_idx is None else int(reference_idx)
    if ref_idx is not None and (ref_idx < 0 or ref_idx >= vectors.shape[1]):
        raise ValueError("reference_idx must be within vector bounds")

    wn = None
    if wavenumbers is not None:
        wn = np.asarray(wavenumbers, dtype=float)
        if wn.ndim != 1 or wn.size != vectors.shape[1]:
            raise ValueError("wavenumbers must be 1D and match vector width")

    eps_val = float(eps)
    rows: list[dict[str, float]] = []
    for spectrum in np.asarray(vectors, dtype=float):
        peak_rows: list[tuple[str, float, float, float]] = []

        for j, peak_idx in enumerate(peaks):
            label = labels[j]
            lo = max(0, int(peak_idx) - peak_window)
            hi = min(spectrum.size, int(peak_idx) + peak_window + 1)

            intensity = max(float(np.mean(spectrum[lo:hi])), 0.0)
            area = float(_integrate_band(np.asarray(spectrum[lo:hi], dtype=float)))

            apex_idx = _nearest_peak_apex(spectrum, int(peak_idx), lo=lo, hi=hi)
            if apex_idx is not None:
                fwhm = _safe_peak_fwhm(spectrum, int(apex_idx), lo, hi)
            else:
                fwhm = _fallback_fwhm(lo, hi)

            center = float(wn[int(peak_idx)]) if wn is not None else float(int(peak_idx))
            if center <= eps_val:
                relwidth_center = np.nan
            else:
                relwidth_center = float(fwhm / center)

            relwidth_area = float(area / (intensity + eps_val))
            peak_rows.append((label, intensity, relwidth_center, relwidth_area))

        total_intensity = float(sum(item[1] for item in peak_rows))

        if ref_idx is None:
            reference_intensity = None
        else:
            ref_lo = max(0, ref_idx - peak_window)
            ref_hi = min(spectrum.size, ref_idx + peak_window + 1)
            reference_intensity = max(float(np.mean(spectrum[ref_lo:ref_hi])), 0.0)

        row: dict[str, float] = {}
        for label, intensity, relwidth_center, relwidth_area in peak_rows:
            row[f"{label}_relint_ref"] = (
                np.nan
                if reference_intensity is None
                else float(intensity / (reference_intensity + eps_val))
            )
            row[f"{label}_relint_total"] = float(intensity / (total_intensity + eps_val))
            row[f"{label}_relwidth_center"] = relwidth_center
            row[f"{label}_relwidth_area"] = relwidth_area

        rows.append(row)

    return pd.DataFrame(rows)
