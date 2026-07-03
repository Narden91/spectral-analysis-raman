"""Tests for Phase P: difference spectra and signed-effect maps."""

import numpy as np
import pandas as pd

from src.analysis.discriminant_spectra import (
    _annotate_wavenumbers,
    _group_spectra,
    compute_difference_spectrum,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(
    n_per_group: int = 6,
    block: str = "1300",
    group_col: str = "delta_moca_group_extreme",
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    wn = np.linspace(1200.0, 1700.0, 30)
    n_total = 2 * n_per_group
    rows: dict[str, list] = {
        "subject_id": [f"S{i:02d}" for i in range(n_total)],
        group_col: ["decline"] * n_per_group + ["recovery"] * n_per_group,
    }
    for w in wn:
        col = f"{block}__{w:.1f}"
        base = rng.standard_normal(n_total)
        # Add a signal at the first wavenumber for group A
        signal = np.zeros(n_total)
        signal[:n_per_group] = 0.5  # group A is higher
        rows[col] = (base + signal).tolist()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Unit: _group_spectra
# ---------------------------------------------------------------------------


def test_group_spectra_handles_na_middle_group():
    """Extreme contrasts leave pd.NA for excluded middle subjects."""
    rng = np.random.default_rng(42)
    wn = np.linspace(1200.0, 1700.0, 30)
    rows: dict[str, list] = {
        "subject_id": [f"S{i:02d}" for i in range(6)],
        "delta_moca_group_extreme": ["decline", "decline", pd.NA, pd.NA, "recovery", "recovery"],
    }
    for w in wn:
        rows[f"1300__{w:.1f}"] = rng.standard_normal(6).tolist()
    df = pd.DataFrame(rows)

    wavenumbers, X_left, X_right = _group_spectra(
        df,
        group_column="delta_moca_group_extreme",
        left_label="decline",
        right_label="recovery",
        block="1300",
    )
    assert wavenumbers.size == 30
    assert X_left.shape[0] == 2
    assert X_right.shape[0] == 2


def test_group_spectra_shapes():
    df = _make_frame()
    wavenumbers, X_left, X_right = _group_spectra(
        df,
        group_column="delta_moca_group_extreme",
        left_label="decline",
        right_label="recovery",
        block="1300",
    )
    assert wavenumbers.size == 30
    assert X_left.shape[0] == 6
    assert X_right.shape[0] == 6


def test_group_spectra_missing_block_returns_empty():
    df = _make_frame(block="1300")
    wavenumbers, X_left, X_right = _group_spectra(
        df,
        group_column="delta_moca_group_extreme",
        left_label="decline",
        right_label="recovery",
        block="2850",  # not present
    )
    assert wavenumbers.size == 0


# ---------------------------------------------------------------------------
# Unit: compute_difference_spectrum
# ---------------------------------------------------------------------------


def test_compute_difference_spectrum_shape():
    df = _make_frame()
    wn, X_left, X_right = _group_spectra(
        df,
        group_column="delta_moca_group_extreme",
        left_label="decline",
        right_label="recovery",
        block="1300",
    )
    result = compute_difference_spectrum(X_left, X_right, wn)
    assert len(result) == 30
    required_cols = {"wavenumber", "mean_group_a", "mean_group_b", "difference", "cohens_d", "signed_effect"}
    assert required_cols.issubset(result.columns)


def test_difference_zero_for_identical_groups():
    """When both groups have the same spectra, difference ≈ 0."""
    rng = np.random.default_rng(0)
    n, p = 6, 20
    X = rng.standard_normal((n, p))
    wn = np.linspace(1200.0, 1700.0, p)
    result = compute_difference_spectrum(X.copy(), X.copy(), wn)
    np.testing.assert_allclose(result["difference"].to_numpy(), 0.0, atol=1e-10)


def test_difference_sign_reflects_group_a_higher():
    df = _make_frame()
    wn, X_left, X_right = _group_spectra(
        df,
        group_column="delta_moca_group_extreme",
        left_label="decline",
        right_label="recovery",
        block="1300",
    )
    result = compute_difference_spectrum(X_left, X_right, wn)
    # decline has +0.5 signal → mean_group_a > mean_group_b → difference > 0 on average
    assert float(result["difference"].mean()) > 0


def test_signed_effect_sign_consistent_with_difference():
    df = _make_frame()
    wn, X_left, X_right = _group_spectra(
        df,
        group_column="delta_moca_group_extreme",
        left_label="decline",
        right_label="recovery",
        block="1300",
    )
    result = compute_difference_spectrum(X_left, X_right, wn)
    diff = result["difference"].to_numpy(dtype=float)
    se = result["signed_effect"].to_numpy(dtype=float)
    # Where both are finite and non-zero, sign should agree
    mask = np.isfinite(diff) & np.isfinite(se) & (diff != 0.0) & (se != 0.0)
    if mask.sum() > 0:
        assert (np.sign(diff[mask]) == np.sign(se[mask])).all()


def test_cohens_d_finite_for_separated_groups():
    df = _make_frame(n_per_group=8)
    wn, X_left, X_right = _group_spectra(
        df,
        group_column="delta_moca_group_extreme",
        left_label="decline",
        right_label="recovery",
        block="1300",
    )
    result = compute_difference_spectrum(X_left, X_right, wn)
    finite_d = result["cohens_d"].dropna()
    assert len(finite_d) > 0


# ---------------------------------------------------------------------------
# Unit: _annotate_wavenumbers
# ---------------------------------------------------------------------------


def test_annotate_wavenumbers_returns_list():
    wn = np.linspace(1200.0, 1800.0, 10)
    labels = _annotate_wavenumbers(wn)
    assert len(labels) == len(wn)
    assert all(isinstance(label, str) for label in labels)
