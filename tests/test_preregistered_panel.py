from __future__ import annotations

from pathlib import Path

import pandas as pd
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
                "min_snr_for_peak_fit": 0.1,
                "integrals": {
                    "lipid_carbonyl_1720": ["1300", 1710.0, 1735.0],
                    "lipid_carbonyl_1720_fallback": ["1300", 1700.0, 1750.0],
                    "carotenoid_1525": ["1300", 1510.0, 1535.0],
                    "protein_1555": ["1300", 1540.0, 1570.0],
                    "ch_bulk_2885": ["2850", 2850.0, 2950.0],
                },
                "indices": {
                    "membrane_integrity_index": ["ch_bulk_2885", "lipid_carbonyl_1720"],
                    "oxidative_index": ["carotenoid_1525", "lipid_carbonyl_1720"],
                },
                "primary_hypothesis": {
                    "feature": "oxidative_index",
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
            "1300__1520.0": [10.0, 11.0],
            "1300__1530.0": [11.0, 12.0],
            "1300__1550.0": [15.0, 16.0],
            "1300__1560.0": [16.0, 17.0],
            "2850__2860.0": [30.0, 32.0],
            "2850__2890.0": [31.0, 33.0],
        }
    )


def test_preregistration_snapshot_is_frozen(tmp_path: Path) -> None:
    cfg = _panel_config()
    path = write_preregistration_snapshot(cfg=cfg, output_dir=tmp_path)
    text = path.read_text(encoding="utf-8")

    assert "Mandatory anchor hypothesis" in text
    assert "oxidative_index" in text
    assert "delta_moca" in text
    assert "partial Spearman" in text
    assert "analysis:" in text
    assert "preregistered_panel:" in text

    expected_block = yaml.safe_dump({"analysis": {"preregistered_panel": panel_cfg(cfg)}}, sort_keys=False)
    assert expected_block.strip() in text


def test_named_indices_are_primary_columns() -> None:
    frame = _feature_matrix()
    columns = [column for column in frame.columns if "__" in column]
    panel_matrix, column_order = build_panel_matrix(frame, columns, cfg=_panel_config())

    assert column_order[:2] == ["membrane_integrity_index", "oxidative_index"]
    assert "lipid_carbonyl_1720" in column_order
    assert "oxidative_index" in panel_matrix.columns


def test_1720_fallback_when_narrow_window_too_small() -> None:
    frame = pd.DataFrame(
        {
            "1300__1715.0": [20.0],
            "1300__1740.0": [21.0],
            "1300__1520.0": [10.0],
            "2850__2860.0": [30.0],
        }
    )
    columns = [column for column in frame.columns if column.startswith("1300__") or column.startswith("2850__")]
    notes: list[str] = []
    _, column_order = build_panel_matrix(frame, columns, cfg=_panel_config(), runtime_notes=notes)

    assert any("fallback" in note for note in notes)
    assert "lipid_carbonyl_1720" in column_order


def test_run_preregistered_panel_writes_outputs(tmp_path: Path) -> None:
    matrix_path = tmp_path / "subject_feature_matrix_whole.parquet"
    _feature_matrix().to_parquet(matrix_path, index=False)

    outputs = run_preregistered_panel(
        feature_matrix_path=matrix_path,
        output_dir=tmp_path / "panel",
        cfg=_panel_config(),
    )

    assert set(outputs) == {"preregistration", "panel_feature_matrix"}
    panel = pd.read_csv(outputs["panel_feature_matrix"])
    assert list(panel.columns[:3]) == ["subject_id", "membrane_integrity_index", "oxidative_index"]
