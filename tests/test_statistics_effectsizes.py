from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.statistics import cliffs_delta, partial_spearman, variance_homogeneity_test


def test_cliffs_delta_known_cases() -> None:
    assert cliffs_delta([3, 4], [1, 2]) == 1.0
    assert cliffs_delta([1, 2], [3, 4]) == -1.0
    assert cliffs_delta([1, 2], [2, 3]) == -0.75


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


def test_variance_homogeneity() -> None:
    equal_a = np.array([1, 2, 3, 4, 5], dtype=float)
    equal_b = np.array([2, 3, 4, 5, 6], dtype=float)
    _, equal_p = variance_homogeneity_test(equal_a, equal_b, center="mean")

    unequal_a = np.array([9, 10, 11, 10, 9], dtype=float)
    unequal_b = np.array([0, 8, 16, 24, 32], dtype=float)
    _, unequal_p = variance_homogeneity_test(unequal_a, unequal_b, center="median")

    assert equal_p > 0.05
    assert unequal_p < 0.05
