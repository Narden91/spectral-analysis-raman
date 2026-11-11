from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.band_features import (
    band_integral,
    band_mean,
    block_columns,
    build_named_index,
    count_window_channels,
    ratio_from_integrals,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "1300__1240.0": [10.0, 12.0],
            "1300__1245.0": [11.0, 13.0],
            "1300__1250.0": [12.0, 14.0],
            "2850__2840.0": [8.0, 9.0],
            "2850__2850.0": [10.0, 11.0],
        }
    )


def test_block_columns_sorts_wavenumbers() -> None:
    columns = ["1300__1250.0", "1300__1240.0", "2850__2850.0"]
    assert block_columns(columns, "1300") == [
        ("1300__1240.0", 1240.0),
        ("1300__1250.0", 1250.0),
    ]


def test_band_integral_uses_trapezoidal_rule() -> None:
    frame = _frame()
    columns = list(frame.columns)
    integral = band_integral(frame, columns, block="1300", lower=1240.0, upper=1250.0)
    manual = np.trapezoid(frame[["1300__1240.0", "1300__1245.0", "1300__1250.0"]].iloc[0], [1240.0, 1245.0, 1250.0])
    assert np.isclose(float(integral.iloc[0]), float(manual))


def test_band_mean_matches_column_average() -> None:
    frame = _frame()
    columns = list(frame.columns)
    mean_value = band_mean(frame, columns, block="1300", lower=1240.0, upper=1250.0)
    assert np.isclose(float(mean_value.iloc[0]), 11.0)


def test_build_named_index_supports_multi_denominator_sum() -> None:
    integrals = {
        "protein": pd.Series([30.0, 40.0]),
        "struct_a": pd.Series([5.0, 10.0]),
        "struct_b": pd.Series([5.0, 10.0]),
    }
    index = build_named_index(integrals, ["protein", "struct_a", "struct_b"])
    assert np.allclose(index.to_numpy(), [3.0, 2.0])


def test_ratio_from_integrals_avoids_zero_denominator() -> None:
    numerator = pd.Series([10.0, 10.0])
    denominator = pd.Series([2.0, 0.0])
    ratio = ratio_from_integrals(numerator, denominator)
    assert np.isclose(float(ratio.iloc[0]), 5.0)
    assert np.isnan(float(ratio.iloc[1]))


def test_count_window_channels() -> None:
    columns = list(_frame().columns)
    assert count_window_channels(columns, block="1300", lower=1240.0, upper=1250.0) == 3
