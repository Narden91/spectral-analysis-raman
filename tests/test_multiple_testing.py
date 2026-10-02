import numpy as np
import pandas as pd
import pytest

from src.statistics.multiple_testing import holm_adjust, holm_adjust_series


@pytest.mark.parametrize(
    ("p_values", "expected"),
    [
        ([], []),
        ([0.02], [0.02]),
        ([0.01, 0.04, 0.03], [0.03, 0.06, 0.06]),
        ([0.7, 0.7, 0.7], [1.0, 1.0, 1.0]),
        ([0.0, 1.0], [0.0, 1.0]),
    ],
)
def test_holm_adjust_known_results(p_values, expected):
    actual = holm_adjust(np.array(p_values, dtype=float))
    np.testing.assert_allclose(actual, expected)


def test_holm_adjust_restores_input_order_and_does_not_modify_input():
    original = np.array([0.04, 0.01, 0.03])
    np.testing.assert_allclose(holm_adjust(original), [0.06, 0.03, 0.06])
    np.testing.assert_array_equal(original, [0.04, 0.01, 0.03])


def test_holm_adjust_series_keeps_labels_and_excludes_missing_values():
    values = pd.Series(["0.04", "missing", "0.01", None, "0.03"], index=[9, 2, 7, 4, 8])
    expected = pd.Series([0.06, np.nan, 0.03, np.nan, 0.06], index=values.index)
    pd.testing.assert_series_equal(holm_adjust_series(values), expected)


@pytest.mark.parametrize("values", [[], [None, "missing"]])
def test_holm_adjust_series_without_numeric_values(values):
    series = pd.Series(values, dtype=object)
    expected = pd.Series(np.nan, index=series.index, dtype=float)
    pd.testing.assert_series_equal(holm_adjust_series(series), expected)
