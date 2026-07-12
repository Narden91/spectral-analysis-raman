from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.biochemical_ratios import run_biochemical_ratios


def _cfg() -> dict:
    return {
        "analysis": {
            "biochemical_ratios": {
                "enabled": True,
                "targets": ["delta_moca"],
                "min_subjects": 5,
                "fdr_alpha": 0.05,
            }
        },
        "whole_spectra": {
            "bands": {
                "amide_iii_1258": [1240.0, 1250.0],
                "ch2_sym_stretch_2853": [2840.0, 2850.0],
            },
            "ratio_pairs": [["amide_iii_1258", "ch2_sym_stretch_2853"]],
        },
    }


def _feature_matrix() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(12):
        delta = float(index - 5)
        rows.append(
            {
                "subject_id": f"D{index:02d}",
                "condition": "Disease",
                "is_control": False,
                "delta_moca": delta,
                "1300__1240.0": 10.0 + delta,
                "1300__1245.0": 11.0 + delta,
                "1300__1250.0": 12.0 + delta,
                "2850__2840.0": 8.0 - delta * 0.2,
                "2850__2845.0": 9.0 - delta * 0.2,
                "2850__2850.0": 10.0 - delta * 0.2,
            }
        )
    rows.append(
        {
            "subject_id": "C00",
            "condition": "Control",
            "is_control": True,
            "delta_moca": 0.0,
            "1300__1240.0": 100.0,
            "1300__1245.0": 100.0,
            "1300__1250.0": 100.0,
            "2850__2840.0": 100.0,
            "2850__2845.0": 100.0,
            "2850__2850.0": 100.0,
        }
    )
    return pd.DataFrame(rows)


def test_run_biochemical_ratios_writes_expected_outputs(tmp_path: Path) -> None:
    matrix_path = tmp_path / "subject_feature_matrix_whole.parquet"
    _feature_matrix().to_parquet(matrix_path, index=False)

    outputs = run_biochemical_ratios(
        feature_matrix_path=matrix_path,
        output_dir=tmp_path / "10_biochemical_ratios",
        cfg=_cfg(),
    )

    bands = pd.read_csv(outputs["biochemical_band_integrals"])
    ratios = pd.read_csv(outputs["biochemical_ratios"])
    associations = pd.read_csv(outputs["biochemical_ratio_associations"])

    assert set(outputs) == {
        "biochemical_band_integrals",
        "biochemical_ratios",
        "biochemical_ratio_associations",
    }
    assert len(bands) == 12
    assert "amide_iii_1258__over__ch2_sym_stretch_2853" in ratios.columns
    assert {"band_integral", "band_ratio"}.issubset(set(associations["feature_source"]))
    assert set(associations["target"]) == {"delta_moca"}


def test_run_biochemical_ratios_disabled_returns_empty(tmp_path: Path) -> None:
    outputs = run_biochemical_ratios(
        output_dir=tmp_path,
        cfg={"analysis": {"biochemical_ratios": {"enabled": False}}},
    )
    assert outputs == {}
