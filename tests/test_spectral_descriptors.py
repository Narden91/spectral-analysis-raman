"""Tests for Phase O: spectral entropy and dispersion descriptors."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.spectral_descriptors import (
    compute_spectral_entropy_features,
    gini_coefficient,
    spectral_entropy,
)


# ---------------------------------------------------------------------------
# Unit: spectral_entropy
# ---------------------------------------------------------------------------


def test_entropy_spread_spectrum():
    """Broadly distributed spectrum → higher entropy than concentrated."""
    # A gradually varying spectrum has more spread than a spike
    n = 20
    spread = np.linspace(0.1, 1.0, n)  # gradual ramp, all non-negative
    h = spectral_entropy(spread)
    assert np.isfinite(h)
    assert h > 0.0


def test_entropy_concentrated_spike():
    """Single-spike spectrum → near-zero entropy."""
    intensities = np.zeros(20)
    intensities[10] = 1.0
    h = spectral_entropy(intensities)
    assert np.isfinite(h)
    assert h == pytest.approx(0.0, abs=1e-9)


def test_entropy_spread_gt_spike():
    """A spread spectrum has higher entropy than a single spike."""
    n = 20
    spread = np.linspace(0.1, 1.0, n)
    spike = np.zeros(n)
    spike[5] = 1.0
    h_spread = spectral_entropy(spread)
    h_spike = spectral_entropy(spike)
    # spike entropy is 0 (or -0); spread should be higher
    assert np.isfinite(h_spread)
    assert h_spread > h_spike


def test_entropy_negative_values_handled():
    """SNV-normalized spectra can be negative; entropy shifts to ≥0 before computing."""
    intensities = np.array([-0.5, 0.0, 0.3, 1.0, 0.8])
    h = spectral_entropy(intensities)
    assert np.isfinite(h)
    assert h >= 0.0


def test_entropy_constant_returns_nan():
    """All-zero spectrum after shifting → NaN (no information)."""
    intensities = np.full(10, 0.5)
    # After shifting: all zeros, sum = 0 → NaN
    h = spectral_entropy(intensities)
    # A flat non-zero spectrum after shift-to-min becomes all zeros → nan
    assert np.isnan(h)


def test_entropy_single_value_returns_nan():
    h = spectral_entropy(np.array([1.0]))
    assert np.isnan(h)


def test_entropy_all_nan_returns_nan():
    h = spectral_entropy(np.array([np.nan, np.nan]))
    assert np.isnan(h)


# ---------------------------------------------------------------------------
# Unit: gini_coefficient
# ---------------------------------------------------------------------------


def test_gini_uniform_near_zero():
    g = gini_coefficient(np.ones(20))
    # All same → all zeros after shift → treated as constant → 0.0
    assert g == pytest.approx(0.0, abs=1e-10)


def test_gini_pure_spike_near_one():
    intensities = np.zeros(20)
    intensities[10] = 1.0
    g = gini_coefficient(intensities)
    assert np.isfinite(g)
    assert g > 0.8  # very concentrated distribution → high inequality


def test_gini_bounded():
    rng = np.random.default_rng(0)
    intensities = rng.exponential(1.0, size=50)
    g = gini_coefficient(intensities)
    assert 0.0 <= g <= 1.0 + 1e-9


def test_gini_uniform_lt_skewed():
    uniform = np.ones(10)
    skewed = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    assert gini_coefficient(uniform) < gini_coefficient(skewed)


def test_gini_single_value_nan():
    assert np.isnan(gini_coefficient(np.array([1.0])))


# ---------------------------------------------------------------------------
# Integration: compute_spectral_entropy_features
# ---------------------------------------------------------------------------


def _synthetic_feature_matrix(n_subjects: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    wn = np.linspace(800.0, 1800.0, 50)
    rows: dict[str, list] = {"subject_id": [f"S{i:02d}" for i in range(n_subjects)]}
    for w in wn:
        col = f"1300__{w:.1f}"
        rows[col] = rng.standard_normal(n_subjects).tolist()
    return pd.DataFrame(rows)


def test_compute_spectral_entropy_features_returns_frame():
    df = _synthetic_feature_matrix()
    result = compute_spectral_entropy_features(df, block="1300")
    assert not result.empty
    assert "spectral_entropy" in result.columns
    assert "gini_dispersion" in result.columns
    assert len(result) == len(df)


def test_spectral_entropy_features_no_block_returns_empty():
    df = pd.DataFrame({"subject_id": ["A", "B"], "2850__2900.0": [1.0, 2.0]})
    result = compute_spectral_entropy_features(df, block="1300")
    assert result.empty


def test_spectral_entropy_values_finite():
    df = _synthetic_feature_matrix()
    result = compute_spectral_entropy_features(df, block="1300")
    finite_h = result["spectral_entropy"].dropna()
    assert len(finite_h) > 0
    assert (finite_h >= 0.0).all()
