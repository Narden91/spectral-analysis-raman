"""
Preprocessing Consistency Checks (PCS)
Formal validation of the preprocessing pipeline to ensure biological and scientific rigour.
"""

import logging

import numpy as np
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

def run_pcs_validation(raw_spectra: np.ndarray, proc_spectra: np.ndarray, wavenumbers_raw: np.ndarray, wavenumbers_proc: np.ndarray) -> dict:
    """
    Run Preprocessing Consistency Checks (PCS) to validate the preprocessing pipeline.
    
    Validates:
    - Peak Shifts: Verifies that preprocessing (e.g. smoothing) does not artificially shift physical peak locations.
    - Axis Consistency: Confirms that the preprocessed spectral wavenumber axis matches the raw axis.
    - Normalization Distortion: Evaluates whether vector normalization distorted chemical peak ratios.
    
    Returns
    -------
    dict:
        Metrics and pass/fail status of the validation.
    """
    if raw_spectra.size == 0 or proc_spectra.size == 0:
        return {"passed": False, "error": "Empty spectra"}

    # 1. Axis Consistency
    axis_consistent = bool(np.all(np.isin(np.round(wavenumbers_proc, 2), np.round(wavenumbers_raw, 2))))

    # 2. Peak Shifts & Normalization Distortion
    shift_errors = []
    ratio_differences = []

    mean_raw = np.mean(raw_spectra, axis=0)
    mean_proc = np.mean(proc_spectra, axis=0)

    # Subtract global baseline to ensure peaks are detectable above zero
    mean_raw_sub = mean_raw - np.min(mean_raw)
    mean_proc_sub = mean_proc - np.min(mean_proc)

    # Find peaks in preprocessed data
    peaks_proc, _ = find_peaks(mean_proc_sub, distance=10)
    if len(peaks_proc) >= 5:
        top_peaks_proc = peaks_proc[np.argsort(mean_proc_sub[peaks_proc])][-5:]
    else:
        top_peaks_proc = peaks_proc

    peaks_raw, _ = find_peaks(mean_raw_sub, distance=10)
    matched_proc = []
    matched_raw = []

    for p_idx in top_peaks_proc:
        target_wn = wavenumbers_proc[p_idx]
        if len(peaks_raw) > 0:
            nearest_raw_idx = peaks_raw[np.argmin(np.abs(wavenumbers_raw[peaks_raw] - target_wn))]
            shift = np.abs(wavenumbers_raw[nearest_raw_idx] - target_wn)
            shift_errors.append(shift)
            matched_proc.append(p_idx)
            matched_raw.append(nearest_raw_idx)

    if len(matched_proc) >= 2:
        # Compare ratios of the top 2 peaks
        p1_proc, p2_proc = matched_proc[-1], matched_proc[-2]
        p1_raw, p2_raw = matched_raw[-1], matched_raw[-2]

        raw_ratio = mean_raw_sub[p1_raw] / (mean_raw_sub[p2_raw] + 1e-6)
        proc_ratio = mean_proc_sub[p1_proc] / (mean_proc_sub[p2_proc] + 1e-6)

        ratio_diff = np.abs(raw_ratio - proc_ratio) / max(raw_ratio, 1e-6)
        ratio_differences.append(ratio_diff)

    mean_shift = float(np.mean(shift_errors)) if shift_errors else 0.0
    max_shift = float(np.max(shift_errors)) if shift_errors else 0.0
    mean_ratio_diff = float(np.mean(ratio_differences)) if ratio_differences else 0.0

    # Validation criteria thresholds
    passed = bool((mean_shift < 5.0) and axis_consistent and (mean_ratio_diff < 0.5))

    metrics = {
        "mean_shift_cm": mean_shift,
        "max_shift_cm": max_shift,
        "axis_consistent": axis_consistent,
        "ratio_distortion_pct": mean_ratio_diff * 100,
        "passed": passed
    }

    if passed:
        logger.info("PCS Validation PASSED: Axis=%s, Mean Shift=%.2f cm-1, Ratio Distortion=%.1f%%",
                    axis_consistent, mean_shift, mean_ratio_diff * 100)
    else:
        logger.warning("PCS Validation FAILED: Axis=%s, Mean Shift=%.2f cm-1, Ratio Distortion=%.1f%%",
                       axis_consistent, mean_shift, mean_ratio_diff * 100)

    return metrics
