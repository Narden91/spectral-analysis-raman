"""Tests for Phase N: NMF decomposition."""

import numpy as np
import pytest

from src.analysis.nmf_decomposition import (
    _annotate_component,
    _fit_nmf_multi_init,
    _matched_component_stability,
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
    result = _fit_nmf_multi_init(
        X, n_components=3, alpha_w=0.1, alpha_h=0.0, n_init=3, random_state=0
    )
    assert np.all(result.W >= -1e-10), "W should be non-negative"
    assert np.all(result.H >= -1e-10), "H should be non-negative"
    assert np.isfinite(result.best_error)
    assert result.best_error >= 0.0


def test_fit_nmf_stability_metric_finite():
    X = _make_synthetic_spectra()
    result = _fit_nmf_multi_init(
        X, n_components=2, alpha_w=0.0, alpha_h=0.0, n_init=5, random_state=42
    )
    assert np.isfinite(result.reconstruction_error_std)
    assert result.reconstruction_error_std >= 0.0
    assert np.all((result.component_stability >= 0.0) & (result.component_stability <= 1.0))


def test_fit_nmf_sparsity_applied():
    """With L1 sparsity, W should be sparser than without."""
    X = _make_synthetic_spectra(n_subjects=20, n_channels=50)
    W_sparse = _fit_nmf_multi_init(
        X, n_components=3, alpha_w=1.0, alpha_h=0.0, n_init=3, random_state=0
    ).W
    W_dense = _fit_nmf_multi_init(
        X, n_components=3, alpha_w=0.0, alpha_h=0.0, n_init=3, random_state=0
    ).W
    # Sparsity should produce at least as many near-zero entries
    n_zero_sparse = int(np.sum(W_sparse < 1e-6))
    n_zero_dense = int(np.sum(W_dense < 1e-6))
    assert n_zero_sparse >= n_zero_dense


def test_fit_nmf_reproducible():
    X = _make_synthetic_spectra()
    result1 = _fit_nmf_multi_init(
        X, n_components=2, alpha_w=0.0, alpha_h=0.0, n_init=3, random_state=7
    )
    result2 = _fit_nmf_multi_init(
        X, n_components=2, alpha_w=0.0, alpha_h=0.0, n_init=3, random_state=7
    )
    np.testing.assert_allclose(result1.best_error, result2.best_error, rtol=1e-6)
    np.testing.assert_allclose(result1.component_stability, result2.component_stability, rtol=1e-6)


def test_component_stability_matches_permuted_components() -> None:
    reference = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    swapped = reference[[1, 0]]
    np.testing.assert_allclose(_matched_component_stability(reference, [swapped]), [1.0, 1.0])


def test_component_stability_is_unavailable_for_one_successful_start() -> None:
    X = _make_synthetic_spectra()
    result = _fit_nmf_multi_init(
        X, n_components=2, alpha_w=0.0, alpha_h=0.0, n_init=1, random_state=42
    )
    assert np.isnan(result.component_stability).all()


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


def test_fit_nmf_multi_init_serial_equals_parallel():
    X = _make_synthetic_spectra(n_subjects=24, n_channels=60, n_components=3, random_state=9)
    serial = _fit_nmf_multi_init(
        X, n_components=3, alpha_w=0.1, alpha_h=0.0, n_init=6, random_state=42, n_jobs=1
    )
    parallel = _fit_nmf_multi_init(
        X, n_components=3, alpha_w=0.1, alpha_h=0.0, n_init=6, random_state=42, n_jobs=2
    )
    np.testing.assert_allclose(parallel.best_error, serial.best_error, atol=1e-14)
    np.testing.assert_allclose(parallel.reconstruction_error_std, serial.reconstruction_error_std, atol=1e-14)
    np.testing.assert_allclose(parallel.W, serial.W, atol=1e-14)
    np.testing.assert_allclose(parallel.H, serial.H, atol=1e-14)
    np.testing.assert_allclose(parallel.component_stability, serial.component_stability, atol=1e-14)
    assert parallel.n_successful_initializations == serial.n_successful_initializations


def test_fit_nmf_multi_init_nested_pool_guard():
    from joblib import Parallel, delayed

    X = _make_synthetic_spectra(n_subjects=20, n_channels=40, n_components=2, random_state=5)
    serial = _fit_nmf_multi_init(
        X, n_components=2, alpha_w=0.05, alpha_h=0.0, n_init=4, random_state=11, n_jobs=1
    )
    nested = Parallel(n_jobs=2)(
        delayed(_fit_nmf_multi_init)(
            X, n_components=2, alpha_w=0.05, alpha_h=0.0, n_init=4, random_state=11, n_jobs=2
        )
        for _ in range(2)
    )
    np.testing.assert_allclose(nested[0].best_error, serial.best_error, atol=1e-14)
    np.testing.assert_allclose(nested[0].H, serial.H, atol=1e-14)


def test_run_nmf_decomposition_returns_empty_for_zero_or_one_disease_subject(tmp_path):
    import pandas as pd

    from src.analysis.nmf_decomposition import run_nmf_decomposition

    cfg = {
        "analysis": {
            "nmf_decomposition": {"enabled": True},
            "preregistered_panel": {
                "nmf": {"n_components": 3, "sparsity_alpha": 0.1, "n_init": 2, "apply_to": "fingerprint"},
                "endpoints": {"continuous": []},
            },
        }
    }
    cols = {f"1300__{800 + i}": [float(i + 1)] for i in range(8)}
    for n_disease in (0, 1):
        fm_path = tmp_path / f"nmf_{n_disease}.parquet"
        frame = pd.DataFrame(
            {
                "subject_id": ["C1"] + ([f"D{i}" for i in range(n_disease)]),
                "is_control": [True] + ([False] * n_disease),
                **{k: v * (1 + n_disease) for k, v in cols.items()},
            }
        )
        frame.to_parquet(fm_path, index=False)
        out = run_nmf_decomposition(feature_matrix_path=fm_path, output_dir=tmp_path / f"out_{n_disease}", cfg=cfg)
        assert out == {}
