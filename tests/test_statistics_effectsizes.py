from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.statistics import (
    cliffs_delta,
    compare_band_intensities_between_groups,
    hedges_g,
    partial_spearman,
    run_pca,
    variance_homogeneity_test,
)
from src.analysis.statistics import test_normality as check_normality


def test_cliffs_delta_known_cases() -> None:
    assert cliffs_delta([3, 4], [1, 2]) == 1.0
    assert cliffs_delta([1, 2], [3, 4]) == -1.0
    assert cliffs_delta([1, 2], [2, 3]) == -0.75


def test_band_comparisons_emit_effect_type_without_sample_size_switching() -> None:
    values = pd.DataFrame({"band": [0.0, 1.0, 2.0, 4.0]})
    labels = np.array(["control", "control", "disease", "disease"])

    parametric = compare_band_intensities_between_groups(values, labels, "control", "disease", method="parametric")
    non_parametric = compare_band_intensities_between_groups(values, labels, "control", "disease", method="non-parametric")

    assert parametric.loc[0, "effect_size_type"] == "hedges_g"
    assert np.isclose(parametric.loc[0, "effect_size"], hedges_g(np.array([0.0, 1.0]), np.array([2.0, 4.0])))
    assert non_parametric.loc[0, "effect_size_type"] == "cliffs_delta"


def test_partial_spearman_rejects_multilevel_nominal_covariates() -> None:
    with pytest.raises(ValueError, match="numeric or binary"):
        partial_spearman(
            np.arange(6, dtype=float),
            np.arange(6, dtype=float),
            pd.DataFrame({"site": ["a", "b", "c", "a", "b", "c"]}),
        )


def test_partial_spearman_recovers_raw_when_covariate_constant() -> None:
    x = np.arange(10, dtype=float)
    y = x * 2.0
    covars = pd.DataFrame({"age": np.ones(10)})

    rho, p_value, n = partial_spearman(x, y, covars)

    assert n == 10
    assert np.isclose(rho, 1.0)
    assert p_value <= 0.001


def test_partial_spearman_reduces_synthetic_age_confound() -> None:
    rng = np.random.default_rng(0)
    age = np.linspace(40, 80, 40)
    x = age + rng.normal(0, 3, 40)
    y = age + rng.normal(0, 3, 40)

    rho, _, n = partial_spearman(x, y, pd.DataFrame({"age": age}))

    assert n == 40
    assert abs(rho) < 0.35


def test_partial_spearman_constant_outcome_returns_not_testable() -> None:
    rho, p_value, n = partial_spearman(
        np.ones(10),
        np.arange(10, dtype=float),
        pd.DataFrame(index=range(10)),
    )

    assert n == 10
    assert np.isnan(rho)
    assert np.isnan(p_value)


def test_variance_homogeneity() -> None:
    equal_a = np.array([1, 2, 3, 4, 5], dtype=float)
    equal_b = np.array([2, 3, 4, 5, 6], dtype=float)
    _, equal_p = variance_homogeneity_test(equal_a, equal_b, center="mean")

    unequal_a = np.array([9, 10, 11, 10, 9], dtype=float)
    unequal_b = np.array([0, 8, 16, 24, 32], dtype=float)
    _, unequal_p = variance_homogeneity_test(unequal_a, unequal_b, center="median")

    assert equal_p > 0.05
    assert unequal_p < 0.05


def test_normality_fraction_excludes_undersampled_channels() -> None:
    rng = np.random.default_rng(0)
    n_samples = 30
    normal_col = rng.normal(size=n_samples)
    undersampled_col = np.full(n_samples, np.nan)
    undersampled_col[:5] = rng.normal(size=5)
    data = np.column_stack([normal_col, undersampled_col])

    result = check_normality(data, alpha=0.05)

    assert result["n_tested"] == 2
    assert result["n_tested_finite"] == 1
    assert np.isnan(result["p_values"][1])
    assert result["is_normal"][1] == False  # noqa: E712 — untested channel, not a claim of non-normality
    assert result["fraction_normal"] == float(result["is_normal"][0])


def test_normality_fraction_is_nan_when_nothing_tested() -> None:
    data = np.full((3, 2), np.nan)
    result = check_normality(data, alpha=0.05)

    assert result["n_tested_finite"] == 0
    assert np.isnan(result["fraction_normal"])


def test_run_pca_raises_on_partially_nan_column() -> None:
    rng = np.random.default_rng(0)
    spectra = rng.normal(size=(10, 5))
    spectra[3, 2] = np.nan

    with pytest.raises(ValueError, match="partial NaN"):
        run_pca(spectra, n_components=2, scale=False)


def test_run_pca_still_drops_all_nan_column() -> None:
    rng = np.random.default_rng(0)
    spectra = rng.normal(size=(10, 5))
    spectra[:, 2] = np.nan

    result = run_pca(spectra, n_components=2, scale=False)

    assert result["scores"].shape == (10, 2)
