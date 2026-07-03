"""Tests for Phase N: NMF decomposition."""

import numpy as np
import pytest

from src.analysis.nmf_decomposition import (
    _annotate_component,
    _fit_nmf_multi_init,
    _reconstruction_error,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_spectra(
    n_subjects: int = 15,
    n_channels: int = 40,
    n_components: int = 3,
    random_state: int = 42,
) -> np.ndarray:
    """Non-negative matrix as a simple rank-k outer product plus small noise."""
    rng = np.random.default_rng(random_state)
    W_true = rng.random((n_subjects, n_components))
    H_true = rng.random((n_components, n_channels))
    X = W_true @ H_true + rng.random((n_subjects, n_channels)) * 0.05
    return np.clip(X, 0.0, None)


# ---------------------------------------------------------------------------
# Unit: reconstruction error
# ---------------------------------------------------------------------------


def test_reconstruction_error_zero_for_perfect():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    # W @ H = I @ X = X, so reconstruction error is zero
    W = np.eye(2)
    H = X.copy()
    err = _reconstruction_error(X, W, H)
    assert err == pytest.approx(0.0, abs=1e-10)


def test_reconstruction_error_positive():
    X = np.ones((3, 4))
    W = np.ones((3, 2)) * 0.5
    H = np.ones((2, 4)) * 0.5
    err = _reconstruction_error(X, W, H)
    assert err > 0.0


# ---------------------------------------------------------------------------
# Unit: _fit_nmf_multi_init
# ---------------------------------------------------------------------------


def test_fit_nmf_non_negative():
    X = _make_synthetic_spectra()
    W, H, best_error, error_std = _fit_nmf_multi_init(
        X, n_components=3, alpha_w=0.1, alpha_h=0.0, n_init=3, random_state=0
    )
    assert np.all(W >= -1e-10), "W should be non-negative"
    assert np.all(H >= -1e-10), "H should be non-negative"
    assert np.isfinite(best_error)
    assert best_error >= 0.0


def test_fit_nmf_stability_metric_finite():
    X = _make_synthetic_spectra()
    W, H, best_error, error_std = _fit_nmf_multi_init(
        X, n_components=2, alpha_w=0.0, alpha_h=0.0, n_init=5, random_state=42
    )
    assert np.isfinite(error_std)
    assert error_std >= 0.0


def test_fit_nmf_sparsity_applied():
    """With L1 sparsity, W should be sparser than without."""
    X = _make_synthetic_spectra(n_subjects=20, n_channels=50)
    W_sparse, _, _, _ = _fit_nmf_multi_init(
        X, n_components=3, alpha_w=1.0, alpha_h=0.0, n_init=3, random_state=0
    )
    W_dense, _, _, _ = _fit_nmf_multi_init(
        X, n_components=3, alpha_w=0.0, alpha_h=0.0, n_init=3, random_state=0
    )
    # Sparsity should produce at least as many near-zero entries
    n_zero_sparse = int(np.sum(W_sparse < 1e-6))
    n_zero_dense = int(np.sum(W_dense < 1e-6))
    assert n_zero_sparse >= n_zero_dense


def test_fit_nmf_reproducible():
    X = _make_synthetic_spectra()
    W1, H1, e1, _ = _fit_nmf_multi_init(
        X, n_components=2, alpha_w=0.0, alpha_h=0.0, n_init=3, random_state=7
    )
    W2, H2, e2, _ = _fit_nmf_multi_init(
        X, n_components=2, alpha_w=0.0, alpha_h=0.0, n_init=3, random_state=7
    )
    np.testing.assert_allclose(e1, e2, rtol=1e-6)


# ---------------------------------------------------------------------------
# Unit: _annotate_component
# ---------------------------------------------------------------------------


def test_annotate_component_unknown_when_no_overlap():
    wavenumbers = np.linspace(800.0, 900.0, 20)
    loadings = np.ones(20)
    label = _annotate_component(loadings, wavenumbers)
    # Minimal expectation: returns a string
    assert isinstance(label, str)


def test_annotate_component_empty_loadings():
    wavenumbers = np.array([1720.0, 1725.0])
    loadings = np.array([0.0, 0.0])
    label = _annotate_component(loadings, wavenumbers)
    assert isinstance(label, str)
