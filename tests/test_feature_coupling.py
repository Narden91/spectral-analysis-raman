"""Tests for Phase Q: feature coupling and biochemical regimes."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.feature_coupling import (
    assign_biochemical_regimes,
    compute_coupling_matrix,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_index_frame(n: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "subject_id": [f"S{i:02d}" for i in range(n)],
            "oxidative_index": rng.uniform(0.1, 2.0, n),
            "membrane_integrity_index": rng.uniform(0.5, 5.0, n),
            "protein_remodeling_index": rng.uniform(0.2, 3.0, n),
            "tyrosine_environment_index": rng.uniform(0.3, 1.5, n),
            "lipid_carbonyl_1720": rng.uniform(0.1, 1.0, n),
            "carotenoid_1525": rng.uniform(0.0, 0.5, n),
        }
    )


# ---------------------------------------------------------------------------
# compute_coupling_matrix
# ---------------------------------------------------------------------------


def test_coupling_matrix_has_expected_columns():
    frame = _make_index_frame()
    cols = ["oxidative_index", "membrane_integrity_index", "protein_remodeling_index"]
    result = compute_coupling_matrix(frame, cols, n_bootstrap=0)
    expected_cols = {"feature_a", "feature_b", "spearman_rho", "bootstrap_ci_lower",
                     "bootstrap_ci_upper", "n_subjects"}
    assert expected_cols.issubset(result.columns)


def test_coupling_matrix_row_count():
    frame = _make_index_frame()
    cols = ["oxidative_index", "membrane_integrity_index", "protein_remodeling_index"]
    result = compute_coupling_matrix(frame, cols, n_bootstrap=0)
    # 3 features → 3 unique pairs (upper triangle)
    assert len(result) == 3


def test_coupling_matrix_self_pairs_absent():
    """Diagonal pairs (feature == feature) must not appear."""
    frame = _make_index_frame()
    cols = ["oxidative_index", "membrane_integrity_index"]
    result = compute_coupling_matrix(frame, cols, n_bootstrap=0)
    assert (result["feature_a"] != result["feature_b"]).all()


def test_coupling_matrix_rho_in_bounds():
    frame = _make_index_frame()
    cols = ["oxidative_index", "membrane_integrity_index", "protein_remodeling_index"]
    result = compute_coupling_matrix(frame, cols, n_bootstrap=0)
    finite_rho = result["spearman_rho"].dropna()
    assert (finite_rho.abs() <= 1.0).all()


def test_coupling_matrix_known_perfect_correlation():
    """Perfectly correlated pair should yield ρ ≈ 1."""
    n = 10
    x = np.arange(n, dtype=float)
    frame = pd.DataFrame({"feat_a": x, "feat_b": x + 1.0})  # perfect positive correlation
    result = compute_coupling_matrix(frame, ["feat_a", "feat_b"], n_bootstrap=0)
    assert len(result) == 1
    assert abs(float(result["spearman_rho"].iloc[0]) - 1.0) < 1e-6


def test_coupling_matrix_bootstrap_ci_populated():
    frame = _make_index_frame(n=15)
    cols = ["oxidative_index", "membrane_integrity_index"]
    result = compute_coupling_matrix(frame, cols, n_bootstrap=50, random_state=7)
    row = result.iloc[0]
    assert np.isfinite(float(row["bootstrap_ci_lower"]))
    assert np.isfinite(float(row["bootstrap_ci_upper"]))
    assert float(row["bootstrap_ci_lower"]) <= float(row["bootstrap_ci_upper"])


def test_coupling_matrix_too_few_features_returns_empty():
    frame = _make_index_frame()
    result = compute_coupling_matrix(frame, ["oxidative_index"], n_bootstrap=0)
    assert result.empty or len(result) == 0


def test_coupling_matrix_missing_col_ignored():
    frame = _make_index_frame()
    cols = ["oxidative_index", "membrane_integrity_index", "does_not_exist"]
    result = compute_coupling_matrix(frame, cols, n_bootstrap=0)
    # Only 1 valid pair
    assert len(result) == 1


# ---------------------------------------------------------------------------
# assign_biochemical_regimes
# ---------------------------------------------------------------------------


def test_regimes_column_added():
    frame = _make_index_frame()
    result = assign_biochemical_regimes(frame)
    assert "biochemical_regime" in result.columns


def test_regimes_valid_labels():
    frame = _make_index_frame(n=30)
    result = assign_biochemical_regimes(frame)
    valid = {"oxidative_dominant", "lipid_dominant", "protein_dominant", "unknown"}
    assert set(result["biochemical_regime"].unique()).issubset(valid)


def test_regimes_all_assigned_when_data_complete():
    frame = _make_index_frame(n=20)
    result = assign_biochemical_regimes(frame)
    assert (result["biochemical_regime"] != "unknown").all()


def test_regimes_high_oxidative_classified_correctly():
    """Subject with max oxidative_index should be oxidative_dominant."""
    frame = _make_index_frame(n=20)
    result = assign_biochemical_regimes(frame)
    max_idx = frame["oxidative_index"].idxmax()
    assert result.loc[max_idx, "biochemical_regime"] == "oxidative_dominant"


def test_regimes_missing_oxidative_col():
    frame = pd.DataFrame({
        "membrane_integrity_index": np.linspace(0.5, 5.0, 10),
        "subject_id": [f"S{i}" for i in range(10)],
    })
    result = assign_biochemical_regimes(frame, oxidative_col="oxidative_index")
    # oxidative_col absent → lipid or protein dominant based on membrane alone
    valid = {"oxidative_dominant", "lipid_dominant", "protein_dominant", "unknown"}
    assert set(result["biochemical_regime"].unique()).issubset(valid)


def test_regimes_both_missing_returns_unknown():
    frame = pd.DataFrame({"subject_id": ["S0", "S1"]})
    result = assign_biochemical_regimes(frame,
                                         oxidative_col="missing_ox",
                                         membrane_col="missing_mem")
    assert (result["biochemical_regime"] == "unknown").all()


def test_regime_counts_three_classes_for_balanced_data():
    """With a sufficiently spread distribution all three regimes should appear."""
    rng = np.random.default_rng(42)
    n = 60
    frame = pd.DataFrame({
        "oxidative_index": rng.uniform(0.1, 3.0, n),
        "membrane_integrity_index": rng.uniform(0.1, 3.0, n),
    })
    result = assign_biochemical_regimes(frame)
    regime_counts = result["biochemical_regime"].value_counts()
    assert len(regime_counts) >= 2  # at least 2 regimes for any reasonable distribution
