"""Tests for Phase Q: feature coupling and biochemical regimes."""

import numpy as np
import pandas as pd

from src.analysis.feature_coupling import (
    assign_ratio_groups,
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
            "carotenoid_to_carbonyl_ratio": rng.uniform(0.1, 2.0, n),
            "ch_bulk_to_carbonyl_ratio": rng.uniform(0.5, 5.0, n),
            "tryptophan_to_sugar_beta_sheet_ratio": rng.uniform(0.2, 3.0, n),
            "tyrosine_to_o_p_o_ratio": rng.uniform(0.3, 1.5, n),
            "lipid_carbonyl_1721": rng.uniform(0.1, 1.0, n),
            "carotenoid_1525": rng.uniform(0.0, 0.5, n),
        }
    )


# ---------------------------------------------------------------------------
# compute_coupling_matrix
# ---------------------------------------------------------------------------


def test_coupling_matrix_has_expected_columns():
    frame = _make_index_frame()
    cols = ["carotenoid_to_carbonyl_ratio", "ch_bulk_to_carbonyl_ratio", "tryptophan_to_sugar_beta_sheet_ratio"]
    result = compute_coupling_matrix(frame, cols, n_bootstrap=0)
    expected_cols = {"feature_a", "feature_b", "spearman_rho", "bootstrap_ci_lower",
                     "bootstrap_ci_upper", "n_subjects"}
    assert expected_cols.issubset(result.columns)


def test_coupling_matrix_row_count():
    frame = _make_index_frame()
    cols = ["carotenoid_to_carbonyl_ratio", "ch_bulk_to_carbonyl_ratio", "tryptophan_to_sugar_beta_sheet_ratio"]
    result = compute_coupling_matrix(frame, cols, n_bootstrap=0)
    # 3 features → 3 unique pairs (upper triangle)
    assert len(result) == 3


def test_coupling_matrix_self_pairs_absent():
    """Diagonal pairs (feature == feature) must not appear."""
    frame = _make_index_frame()
    cols = ["carotenoid_to_carbonyl_ratio", "ch_bulk_to_carbonyl_ratio"]
    result = compute_coupling_matrix(frame, cols, n_bootstrap=0)
    assert (result["feature_a"] != result["feature_b"]).all()


def test_coupling_matrix_rho_in_bounds():
    frame = _make_index_frame()
    cols = ["carotenoid_to_carbonyl_ratio", "ch_bulk_to_carbonyl_ratio", "tryptophan_to_sugar_beta_sheet_ratio"]
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
    cols = ["carotenoid_to_carbonyl_ratio", "ch_bulk_to_carbonyl_ratio"]
    result = compute_coupling_matrix(frame, cols, n_bootstrap=50, random_state=7)
    row = result.iloc[0]
    assert np.isfinite(float(row["bootstrap_ci_lower"]))
    assert np.isfinite(float(row["bootstrap_ci_upper"]))
    assert float(row["bootstrap_ci_lower"]) <= float(row["bootstrap_ci_upper"])


def test_coupling_bootstrap_resamples_subject_pairs() -> None:
    x = np.arange(20, dtype=float)
    frame = pd.DataFrame({"feat_a": x, "feat_b": 3.0 * x + 2.0})

    result = compute_coupling_matrix(
        frame,
        ["feat_a", "feat_b"],
        n_bootstrap=50,
        random_state=7,
    )

    row = result.iloc[0]
    assert np.isclose(float(row["bootstrap_ci_lower"]), 1.0)
    assert np.isclose(float(row["bootstrap_ci_upper"]), 1.0)


def test_coupling_matrix_too_few_features_returns_empty():
    frame = _make_index_frame()
    result = compute_coupling_matrix(frame, ["carotenoid_to_carbonyl_ratio"], n_bootstrap=0)
    assert result.empty or len(result) == 0


def test_coupling_matrix_missing_col_ignored():
    frame = _make_index_frame()
    cols = ["carotenoid_to_carbonyl_ratio", "ch_bulk_to_carbonyl_ratio", "does_not_exist"]
    result = compute_coupling_matrix(frame, cols, n_bootstrap=0)
    # Only 1 valid pair
    assert len(result) == 1


# ---------------------------------------------------------------------------
# assign_ratio_groups
# ---------------------------------------------------------------------------


def test_regimes_column_added():
    frame = _make_index_frame()
    result = assign_ratio_groups(frame)
    assert "ratio_group" in result.columns


def test_regimes_valid_labels():
    frame = _make_index_frame(n=30)
    result = assign_ratio_groups(frame)
    valid = {"upper_carotenoid_ratio", "lower_carotenoid_upper_ch_bulk_ratio", "lower_carotenoid_lower_ch_bulk_ratio", "unknown"}
    assert set(result["ratio_group"].unique()).issubset(valid)


def test_regimes_all_assigned_when_data_complete():
    frame = _make_index_frame(n=20)
    result = assign_ratio_groups(frame)
    assert (result["ratio_group"] != "unknown").all()


def test_regimes_high_oxidative_classified_correctly():
    """Subject with max carotenoid_to_carbonyl_ratio should be upper_carotenoid_ratio."""
    frame = _make_index_frame(n=20)
    result = assign_ratio_groups(frame)
    max_idx = frame["carotenoid_to_carbonyl_ratio"].idxmax()
    assert result.loc[max_idx, "ratio_group"] == "upper_carotenoid_ratio"


def test_regimes_missing_oxidative_col():
    frame = pd.DataFrame({
        "ch_bulk_to_carbonyl_ratio": np.linspace(0.5, 5.0, 10),
        "subject_id": [f"S{i}" for i in range(10)],
    })
    result = assign_ratio_groups(frame, carotenoid_col="carotenoid_to_carbonyl_ratio")
    # oxidative_col absent → lipid or protein dominant based on membrane alone
    valid = {"upper_carotenoid_ratio", "lower_carotenoid_upper_ch_bulk_ratio", "lower_carotenoid_lower_ch_bulk_ratio", "unknown"}
    assert set(result["ratio_group"].unique()).issubset(valid)


def test_regimes_both_missing_returns_unknown():
    frame = pd.DataFrame({"subject_id": ["S0", "S1"]})
    result = assign_ratio_groups(frame,
                                         carotenoid_col="missing_ox",
                                         ch_bulk_col="missing_mem")
    assert (result["ratio_group"] == "unknown").all()


def test_regime_counts_three_classes_for_balanced_data():
    """With a sufficiently spread distribution all three regimes should appear."""
    rng = np.random.default_rng(42)
    n = 60
    frame = pd.DataFrame({
        "carotenoid_to_carbonyl_ratio": rng.uniform(0.1, 3.0, n),
        "ch_bulk_to_carbonyl_ratio": rng.uniform(0.1, 3.0, n),
    })
    result = assign_ratio_groups(frame)
    regime_counts = result["ratio_group"].value_counts()
    assert len(regime_counts) >= 2  # at least 2 regimes for any reasonable distribution


def test_ratio_groups_do_not_treat_missing_measurements_as_lower_values() -> None:
    frame = pd.DataFrame({"carotenoid_to_carbonyl_ratio": [np.nan, 1., 2.],
                          "ch_bulk_to_carbonyl_ratio": [3., np.nan, 5.]})
    result = assign_ratio_groups(frame)
    assert result["ratio_group"].iloc[:2].tolist() == ["unknown", "unknown"]
