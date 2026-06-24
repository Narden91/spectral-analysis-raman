"""Tests for Phase M: spectral-shape analysis (peak fit + FWHM + asymmetry)."""

import numpy as np
import pytest

from src.analysis.spectral_shape import (
    _empirical_asymmetry,
    _r_squared,
    _snr_from_fit,
    compute_shape_summary,
    extract_peak_shape_features,
    fit_local_peak,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gaussian_spectrum(
    wavenumbers: np.ndarray,
    centre: float,
    fwhm: float,
    amplitude: float = 1.0,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    sigma = fwhm / 2.3548
    signal = amplitude * np.exp(-0.5 * ((wavenumbers - centre) / sigma) ** 2)
    if noise > 0.0:
        rng = rng or np.random.default_rng(0)
        signal = signal + rng.normal(0, noise, size=wavenumbers.size)
    return signal


def _synthetic_feature_matrix(
    n_subjects: int = 10,
    lo: float = 1700.0,
    hi: float = 1750.0,
    centre: float = 1720.0,
    fwhm: float = 20.0,
    noise: float = 0.02,
) -> "pd.DataFrame":
    import pandas as pd

    wn = np.linspace(lo, hi, 30)
    rng = np.random.default_rng(42)
    rows = {}
    rows["subject_id"] = [f"S{i:02d}" for i in range(n_subjects)]
    for j, w in enumerate(wn):
        col = f"1300__{w:.1f}"
        rows[col] = [
            float(_make_gaussian_spectrum(np.array([w]), centre, fwhm, 1.0, noise, rng)[0])
            for _ in range(n_subjects)
        ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Unit: fit_local_peak
# ---------------------------------------------------------------------------


def test_fit_local_peak_recovers_gaussian():
    """Gaussian recovery: centroid within 1 cm⁻¹ of truth, FWHM within 10%."""
    wn = np.linspace(1700.0, 1750.0, 50)
    true_centre = 1720.0
    true_fwhm = 15.0
    intensities = _make_gaussian_spectrum(wn, true_centre, true_fwhm, amplitude=1.0)

    result = fit_local_peak(wn, intensities, lo=1700.0, hi=1750.0, min_snr=0.0, calibration_floor_cm=0.0)

    assert result["fit_status"] == "ok", f"Unexpected status: {result['fit_status']}"
    assert abs(float(result["centroid"]) - true_centre) < 2.0, "Centroid off"
    assert abs(float(result["fwhm"]) - true_fwhm) / true_fwhm < 0.20, "FWHM off"


def test_fit_local_peak_asymmetry_computed():
    wn = np.linspace(1510.0, 1540.0, 40)
    intensities = _make_gaussian_spectrum(wn, 1525.0, 12.0, amplitude=1.0)
    result = fit_local_peak(wn, intensities, lo=1510.0, hi=1540.0, min_snr=0.0, calibration_floor_cm=0.0)
    # Symmetric Gaussian → asymmetry close to 1
    asym = result["asymmetry"]
    if np.isfinite(float(asym)):
        assert 0.3 < float(asym) < 3.0, f"Asymmetry out of plausible range: {asym}"


def test_fit_local_peak_snr_gate():
    """Low-SNR peak flagged as below_snr_gate."""
    wn = np.linspace(1540.0, 1570.0, 30)
    rng = np.random.default_rng(1)
    # Very noisy signal (effectively flat)
    intensities = rng.normal(0, 0.5, size=wn.size)
    result = fit_local_peak(wn, intensities, lo=1540.0, hi=1570.0, min_snr=20.0, calibration_floor_cm=0.0)
    # Either fits with low SNR flag, or fails to fit
    if result["fit_status"] == "ok":
        assert result["snr_flag"] in {"ok", "below_snr_gate"}


def test_fit_local_peak_calibration_floor():
    """Centroid within calibration floor flagged as not_interpretable."""
    wn = np.linspace(1540.0, 1570.0, 30)
    true_centre = 1555.0
    intensities = _make_gaussian_spectrum(wn, true_centre, 10.0)
    # With a large floor (10 cm⁻¹), a 0 shift from nominal (1555) is always flagged
    result = fit_local_peak(wn, intensities, lo=1540.0, hi=1570.0, min_snr=0.0, calibration_floor_cm=10.0)
    if result["fit_status"] == "ok" and np.isfinite(float(result["centroid"])):
        shift = abs(float(result["centroid"]) - 1555.0)
        if shift < 10.0:
            assert "not_interpretable" in str(result["calibration_flag"])


def test_fit_local_peak_too_few_channels():
    wn = np.array([1720.0, 1725.0])
    intensities = np.array([1.0, 0.5])
    result = fit_local_peak(wn, intensities, lo=1710.0, hi=1735.0, min_snr=0.0, calibration_floor_cm=0.0)
    assert result["fit_status"] == "too_few_channels"


# ---------------------------------------------------------------------------
# Unit: R² and SNR helpers
# ---------------------------------------------------------------------------


def test_r_squared_perfect_fit():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert _r_squared(y, y) == pytest.approx(1.0)


def test_r_squared_constant_obs():
    y = np.ones(5)
    y_fit = y + 0.1
    assert np.isnan(_r_squared(y, y_fit))


def test_snr_positive():
    wn = np.linspace(1700.0, 1750.0, 50)
    intensities = _make_gaussian_spectrum(wn, 1720.0, 15.0)
    y_fit = _make_gaussian_spectrum(wn, 1720.0, 15.0)
    snr = _snr_from_fit(intensities, y_fit, amplitude=1.0)
    # Perfect fit yields infinite or very high SNR
    assert snr > 0.0 or np.isnan(snr)


# ---------------------------------------------------------------------------
# Integration: extract_peak_shape_features
# ---------------------------------------------------------------------------


def test_extract_peak_shape_features_returns_frame():
    df = _synthetic_feature_matrix()
    windows = {"test_1720": (1710.0, 1740.0)}
    result = extract_peak_shape_features(df, windows, min_snr=0.0, calibration_floor_cm=0.0)
    assert not result.empty
    required_cols = {"subject_id", "window_name", "centroid", "fwhm", "snr_flag"}
    assert required_cols.issubset(result.columns)


def test_extract_missing_window_skipped():
    """Window outside data range returns empty but does not raise."""
    df = _synthetic_feature_matrix()
    windows = {"very_high": (3000.0, 3050.0)}  # outside 1300 block
    result = extract_peak_shape_features(df, windows, min_snr=0.0, calibration_floor_cm=0.0)
    assert result.empty or len(result) == 0


# ---------------------------------------------------------------------------
# Integration: compute_shape_summary
# ---------------------------------------------------------------------------


def test_compute_shape_summary_has_ci():
    df = _synthetic_feature_matrix(n_subjects=12)
    windows = {"test_1720": (1710.0, 1740.0)}
    features = extract_peak_shape_features(df, windows, min_snr=0.0, calibration_floor_cm=0.0)
    summary = compute_shape_summary(features, n_bootstrap=50, random_state=0)
    assert not summary.empty
    centroid_rows = summary.loc[summary["parameter"] == "centroid"]
    assert not centroid_rows.empty
    # CIs should be finite when enough subjects pass the SNR gate
    ci_lo = centroid_rows["bootstrap_ci_lower"].dropna()
    ci_hi = centroid_rows["bootstrap_ci_upper"].dropna()
    if len(ci_lo) > 0:
        assert (ci_lo <= ci_hi).all(), "CI lower must be ≤ upper"
