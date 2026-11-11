from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.analysis.preregistered_panel import (
    build_panel_matrix,
    panel_cfg,
    run_preregistered_panel,
    write_preregistration_snapshot,
)


def _panel_config() -> dict:
    return {
        "analysis": {
            "preregistered_panel": {
                "enabled": True,
                "provenance_type": "post_hoc_fixed",
                "min_snr_for_peak_fit": 0.1,
                "integrals": {
                    "lipid_carbonyl_1721": ["1300", 1710.0, 1730.0],
                    "carotenoid_1525": ["1300", 1510.0, 1535.0],
                    "tryptophan_1555": ["1300", 1550.0, 1560.0],
                    "ch_bulk_2885": ["2850", 2850.0, 2950.0],
                },
                "indices": {
                    "ch_bulk_to_carbonyl_ratio": ["ch_bulk_2885", "lipid_carbonyl_1721"],
                    "carotenoid_to_carbonyl_ratio": ["carotenoid_1525", "lipid_carbonyl_1721"],
                },
                "primary_hypothesis": {
                    "feature": "carotenoid_to_carbonyl_ratio",
                    "endpoint": "delta_moca",
                    "direction": "negative",
                    "test": "partial_spearman",
                    "covariates": ["age", "sex"],
                },
            }
        }
    }


def _feature_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": ["D01", "D02"],
            "condition": ["Disease", "Disease"],
            "is_control": [False, False],
            "1300__1715.0": [20.0, 22.0],
            "1300__1725.0": [21.0, 23.0],
            "1300__1710.0": [19.0, 21.0],
            "1300__1730.0": [22.0, 24.0],
            "1300__1520.0": [10.0, 11.0],
            "1300__1530.0": [11.0, 12.0],
            "1300__1510.0": [9.0, 10.0],
            "1300__1535.0": [12.0, 13.0],
            "1300__1550.0": [15.0, 16.0],
            "1300__1560.0": [16.0, 17.0],
            "1300__1540.0": [14.0, 15.0],
            "1300__1570.0": [17.0, 18.0],
            "2850__2860.0": [30.0, 32.0],
            "2850__2890.0": [31.0, 33.0],
            "2850__2850.0": [29.0, 31.0],
            "2850__2950.0": [32.0, 34.0],
        }
    )


def test_preregistration_snapshot_is_frozen(tmp_path: Path) -> None:
    cfg = _panel_config()
    path = write_preregistration_snapshot(cfg=cfg, output_dir=tmp_path)
    text = path.read_text(encoding="utf-8")

    assert "Primary exploratory anchor" in text
    assert "Fixed post hoc Raman panel" in text
    assert "carotenoid_to_carbonyl_ratio" in text
    assert "delta_moca" in text
    assert "partial Spearman" in text
    assert "analysis:" in text
    assert "preregistered_panel:" in text

    expected_block = yaml.safe_dump({"analysis": {"preregistered_panel": panel_cfg(cfg)}}, sort_keys=False)
    assert expected_block.strip() in text


def test_named_indices_are_primary_columns() -> None:
    frame = _feature_matrix().rename(columns=lambda col: col.replace("1300__", "1300_bc__").replace("2850__", "2850_bc__"))
    columns = [column for column in frame.columns if "__" in column]
    panel_matrix, column_order = build_panel_matrix(frame, columns, cfg=_panel_config())

    assert column_order[:2] == ["ch_bulk_to_carbonyl_ratio", "carotenoid_to_carbonyl_ratio"]
    assert "lipid_carbonyl_1721" in column_order
    assert "carotenoid_to_carbonyl_ratio" in panel_matrix.columns


def test_narrow_carbonyl_window_is_not_expanded_when_coverage_is_missing() -> None:
    frame = pd.DataFrame({"1300_bc__1700.0": [19.0], "1300_bc__1715.0": [20.0],
                          "1300_bc__1740.0": [21.0], "1300_bc__1750.0": [22.0]})
    panel, order = build_panel_matrix(frame, list(frame), cfg=_panel_config())
    assert "lipid_carbonyl_1721" not in order
    assert "carotenoid_to_carbonyl_ratio" not in panel


def test_run_preregistered_panel_writes_outputs(tmp_path: Path) -> None:
    matrix_path = tmp_path / "subject_feature_matrix_whole.parquet"
    _feature_matrix().to_parquet(matrix_path, index=False)
    _feature_matrix().rename(columns=lambda col: col.replace("1300__", "1300_bc__").replace("2850__", "2850_bc__")).to_parquet(
        tmp_path / "subject_feature_matrix_whole_bc.parquet", index=False,
    )

    outputs = run_preregistered_panel(
        feature_matrix_path=matrix_path,
        output_dir=tmp_path / "panel",
        cfg=_panel_config(),
    )

    assert set(outputs) == {"preregistration", "panel_provenance", "panel_feature_matrix"}
    panel = pd.read_csv(outputs["panel_feature_matrix"])
    assert list(panel.columns[:3]) == ["subject_id", "ch_bulk_to_carbonyl_ratio", "carotenoid_to_carbonyl_ratio"]
    provenance = pd.read_csv(outputs["panel_provenance"])
    assert provenance.loc[0, "provenance_type"] == "post_hoc_fixed"


def test_external_registration_requires_complete_provenance(tmp_path: Path) -> None:
    cfg = _panel_config()
    cfg["analysis"]["preregistered_panel"]["provenance_type"] = "externally_preregistered"
    with pytest.raises(ValueError, match="missing provenance"):
        write_preregistration_snapshot(cfg=cfg, output_dir=tmp_path)
