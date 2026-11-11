from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.clinical.endpoints import build_clinical_endpoints, write_clinical_endpoint_outputs


def _cfg() -> dict:
    return {
        "analysis": {
            "clinical_endpoints": {
                "enabled": True,
                "clinical_table": "data/clinical_data.csv",
                "subject_id_col": "subject_id",
                "moca_baseline_col": "moca_baseline_total",
                "moca_3m_col": "moca_3m_total",
                "rbans_col": "rbans_total",
                "age_col": "age",
                "aspects_col": "aspects_score",
                "sex_col": "sex",
                "diagnosis_col": "diagnosis_category",
                "lesion_location_col": "lesion_location",
                "severity_threshold": 20,
                "decline_threshold": 0,
                "mild_threshold": 25,
                "recovery_threshold": 2,
            }
        }
    }


def _write_csv(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "clinical.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_build_endpoints_maps_current_header_aliases_and_writes_outputs(tmp_path: Path) -> None:
    clinical_csv = _write_csv(
        tmp_path,
        [
            {
                "participant_id": "7",
                "age_years": 65,
                "gender": "female",
                "diagnosis_text": "right hemisphere ischemic stroke",
                "diagnosis_category": "right_hemisphere",
                "rbans_total_3m": 82,
                "moca_total_baseline": 19,
                "moca_total_3m": 22,
                "aspects_score": 10,
            },
            {
                "participant_id": "38",
                "age_years": 56,
                "gender": "female",
                "diagnosis_text": "left cerebellum ischemic stroke",
                "diagnosis_category": "left_cerebellum",
                "rbans_total_3m": 106,
                "moca_total_baseline": 29,
                "moca_total_3m": 21,
                "aspects_score": 9,
            },
            {
                "participant_id": "55",
                "age_years": 61,
                "gender": "male",
                "diagnosis_text": "ischemic stroke in the posterior vascular circulation of the brain",
                "diagnosis_category": "posterior_circulation",
                "rbans_total_3m": 88,
                "moca_total_baseline": 20,
                "moca_total_3m": 20,
                "aspects_score": 8,
            },
        ],
    )

    result = build_clinical_endpoints(cfg=_cfg(), clinical_csv=clinical_csv, output_dir=tmp_path / "out")
    written = write_clinical_endpoint_outputs(result)

    subject_table = result.clinical_subject_table.set_index("subject_id").sort_index()
    assert result.clinical_column_map["subject_id"] == "participant_id"
    assert result.clinical_column_map["moca_baseline"] == "moca_total_baseline"
    assert result.clinical_column_map["lesion_location"] == "diagnosis_category"
    assert float(subject_table.loc["7", "delta_moca"]) == 3.0
    assert float(subject_table.loc["38", "delta_moca"]) == -8.0
    assert float(subject_table.loc["55", "delta_moca"]) == 0.0
    assert str(subject_table.loc["7", "moca_severity_group"]) == "low_moca"
    assert str(subject_table.loc["38", "moca_severity_group"]) == "high_moca"
    assert str(subject_table.loc["55", "moca_severity_group"]) == "high_moca"
    assert str(subject_table.loc["38", "delta_moca_group"]) == "decline"
    assert str(subject_table.loc["7", "lesion_location_group"]) == "right"
    assert str(subject_table.loc["38", "lesion_location_group"]) == "posterior"
    assert str(subject_table.loc["55", "lesion_location_group"]) == "posterior"

    assert str(subject_table.loc["7", "moca_severity_group_extreme"]) == "severe"
    assert str(subject_table.loc["38", "moca_severity_group_extreme"]) == "mild"
    assert pd.isna(subject_table.loc["55", "moca_severity_group_extreme"])
    assert str(subject_table.loc["38", "delta_moca_group_extreme"]) == "decline"
    assert str(subject_table.loc["7", "delta_moca_group_extreme"]) == "recovery"
    assert pd.isna(subject_table.loc["55", "delta_moca_group_extreme"])

    group_counts = result.clinical_group_counts
    assert int(
        group_counts.loc[
            (group_counts["group_variable"] == "lesion_location_group")
            & (group_counts["group_name"] == "posterior"),
            "n_subjects",
        ].iloc[0]
    ) == 2

    assert set(written) == {
        "clinical_endpoint_summary.csv",
        "clinical_subject_table.csv",
        "clinical_missingness_report.csv",
        "clinical_group_counts.csv",
        "clinical_complete_case_comparison.csv",
        "clinical_column_map.json",
    }
    assert json.loads((tmp_path / "out" / "clinical_column_map.json").read_text(encoding="utf-8"))["rbans_total"] == "rbans_total_3m"


def test_build_endpoints_rejects_duplicate_subject_ids(tmp_path: Path) -> None:
    clinical_csv = _write_csv(
        tmp_path,
        [
            {
                "participant_id": "7",
                "age_years": 65,
                "gender": "female",
                "diagnosis_category": "right_hemisphere",
                "rbans_total_3m": 82,
                "moca_total_baseline": 19,
                "moca_total_3m": 22,
                "aspects_score": 10,
            },
            {
                "participant_id": "7",
                "age_years": 56,
                "gender": "male",
                "diagnosis_category": "left_hemisphere",
                "rbans_total_3m": 70,
                "moca_total_baseline": 18,
                "moca_total_3m": 17,
                "aspects_score": 8,
            },
        ],
    )

    with pytest.raises(ValueError, match="duplicate subject_id"):
        build_clinical_endpoints(cfg=_cfg(), clinical_csv=clinical_csv, output_dir=tmp_path / "out")


def test_build_endpoints_fails_on_missing_clinical_csv(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError, match="missing.csv"):
        build_clinical_endpoints(cfg=_cfg(), clinical_csv=missing, output_dir=tmp_path / "out")


def test_build_endpoints_rejects_missing_required_column_mapping(tmp_path: Path) -> None:
    clinical_csv = _write_csv(
        tmp_path,
        [
            {
                "participant_id": "7",
                "age_years": 65,
                "gender": "female",
                "diagnosis_category": "right_hemisphere",
                "moca_total_baseline": 19,
                "moca_total_3m": 22,
                "aspects_score": 10,
            }
        ],
    )

    with pytest.raises(ValueError, match="rbans_total"):
        build_clinical_endpoints(cfg=_cfg(), clinical_csv=clinical_csv, output_dir=tmp_path / "out")


def test_build_endpoints_rejects_non_numeric_moca_columns(tmp_path: Path) -> None:
    clinical_csv = _write_csv(
        tmp_path,
        [
            {
                "participant_id": "7",
                "age_years": 65,
                "gender": "female",
                "diagnosis_category": "right_hemisphere",
                "rbans_total_3m": 82,
                "moca_total_baseline": "bad",
                "moca_total_3m": "also_bad",
                "aspects_score": 10,
            }
        ],
    )

    with pytest.raises(ValueError, match="moca_baseline"):
        build_clinical_endpoints(cfg=_cfg(), clinical_csv=clinical_csv, output_dir=tmp_path / "out")


def test_extreme_group_boundaries_and_counts(tmp_path: Path) -> None:
    clinical_csv = _write_csv(
        tmp_path,
        [
            {
                "participant_id": "A",
                "age_years": 60,
                "gender": "male",
                "diagnosis_category": "right_hemisphere",
                "rbans_total_3m": 80,
                "moca_total_baseline": 20,
                "moca_total_3m": 24,
                "aspects_score": 9,
            },
            {
                "participant_id": "B",
                "age_years": 61,
                "gender": "female",
                "diagnosis_category": "right_hemisphere",
                "rbans_total_3m": 81,
                "moca_total_baseline": 25,
                "moca_total_3m": 25,
                "aspects_score": 9,
            },
            {
                "participant_id": "C",
                "age_years": 62,
                "gender": "male",
                "diagnosis_category": "right_hemisphere",
                "rbans_total_3m": 82,
                "moca_total_baseline": 26,
                "moca_total_3m": 29,
                "aspects_score": 9,
            },
            {
                "participant_id": "D",
                "age_years": 63,
                "gender": "female",
                "diagnosis_category": "right_hemisphere",
                "rbans_total_3m": 83,
                "moca_total_baseline": 22,
                "moca_total_3m": 20,
                "aspects_score": 9,
            },
            {
                "participant_id": "E",
                "age_years": 64,
                "gender": "male",
                "diagnosis_category": "right_hemisphere",
                "rbans_total_3m": 84,
                "moca_total_baseline": 24,
                "moca_total_3m": 27,
                "aspects_score": 9,
            },
        ],
    )

    result = build_clinical_endpoints(cfg=_cfg(), clinical_csv=clinical_csv, output_dir=tmp_path / "out")
    table = result.clinical_subject_table.set_index("subject_id")

    assert pd.isna(table.loc["A", "moca_severity_group_extreme"])
    assert pd.isna(table.loc["B", "moca_severity_group_extreme"])
    assert str(table.loc["C", "moca_severity_group_extreme"]) == "mild"
    assert str(table.loc["D", "delta_moca_group_extreme"]) == "decline"
    assert str(table.loc["E", "delta_moca_group_extreme"]) == "recovery"

    counts = result.clinical_group_counts
    assert "moca_severity_group_extreme" in set(counts["group_variable"])
    assert "delta_moca_group_extreme" in set(counts["group_variable"])
