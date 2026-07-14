from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.disease_subgroups import run_disease_subgroups


def _cfg() -> dict:
    return {
        "analysis": {
            "disease_subgroups": {
                "enabled": True,
                "min_subjects": 8,
                "min_clusters": 2,
                "max_clusters": 3,
                "n_bootstrap": 6,
                "random_state": 42,
            }
        },
        "whole_spectra": {
            "bands": {
                "amide_iii_1258": [1238.0, 1252.0],
                "ch2_sym_stretch_2853": [2838.0, 2852.0],
            }
        },
    }


def _feature_matrix() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(16):
        high = index < 8
        rows.append(
            {
                "subject_id": f"D{index:02d}",
                "condition": "Disease",
                "is_control": False,
                "1300__1240.0": 8.0 if high else 1.0,
                "1300__1245.0": 8.5 if high else 1.2,
                "1300__1250.0": 9.0 if high else 1.4,
                "2850__2840.0": 1.0 if high else 7.0,
                "2850__2845.0": 1.2 if high else 7.5,
                "2850__2850.0": 1.4 if high else 8.0,
            }
        )
    return pd.DataFrame(rows)


def _integrated_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_region_id": ["1300_R1", "2850_R1"],
            "spectral_block": ["1300", "2850"],
            "start_cm-1": [1238.0, 2838.0],
            "end_cm-1": [1252.0, 2852.0],
            "evidence_status": [
                "cluster_supported_exploratory",
                "cluster_supported_exploratory",
            ],
        }
    )


def test_run_disease_subgroups_writes_expected_outputs(tmp_path: Path) -> None:
    matrix_path = tmp_path / "subject_feature_matrix_whole.parquet"
    integrated_path = tmp_path / "integrated_candidate_regions.csv"
    _feature_matrix().to_parquet(matrix_path, index=False)
    _integrated_candidates().to_csv(integrated_path, index=False)

    outputs = run_disease_subgroups(
        feature_matrix_path=matrix_path,
        integrated_candidates_path=integrated_path,
        output_dir=tmp_path / "12_disease_subgroups",
        cfg=_cfg(),
    )

    assignments = pd.read_csv(outputs["disease_subgroup_assignments"])
    gap = pd.read_csv(outputs["disease_subgroup_gap_statistic"])
    stability = pd.read_csv(outputs["disease_subgroup_stability"])
    features = pd.read_csv(outputs["disease_subgroup_feature_matrix"])

    assert {
        "disease_subgroup_assignments",
        "disease_subgroup_gap_statistic",
        "disease_subgroup_stability",
        "disease_subgroup_feature_matrix",
        "subgroup_mean_spectra",
    }.issubset(set(outputs))
    assert (tmp_path / "12_disease_subgroups" / "subgroup_mean_spectra.csv").exists()
    assert len(assignments) == 16
    assert assignments["subgroup_label"].nunique() >= 2
    assert set(gap["n_clusters"]) == {2, 3}
    assert int(stability["n_bootstrap_requested"].iloc[0]) == 6
    assert "region__1300_R1" in features.columns


def test_run_disease_subgroups_disabled_returns_empty(tmp_path: Path) -> None:
    outputs = run_disease_subgroups(
        output_dir=tmp_path,
        cfg={"analysis": {"disease_subgroups": {"enabled": False}}},
    )
    assert outputs == {}


def test_run_disease_subgroups_skips_writes_when_insufficient_subjects(tmp_path: Path) -> None:
    matrix_path = tmp_path / "subject_feature_matrix_whole.parquet"
    integrated_path = tmp_path / "integrated_candidate_regions.csv"
    # Fewer subjects than min_subjects=8 in _cfg().
    _feature_matrix().iloc[:4].to_parquet(matrix_path, index=False)
    _integrated_candidates().to_csv(integrated_path, index=False)

    out_dir = tmp_path / "12_disease_subgroups"
    outputs = run_disease_subgroups(
        feature_matrix_path=matrix_path,
        integrated_candidates_path=integrated_path,
        output_dir=out_dir,
        cfg=_cfg(),
    )

    assert set(outputs) == {"_status"}
    assert outputs["_status"]["phase_key"] == "disease_subgroups"
    assert outputs["_status"]["status"] == "insufficient_n"
    assert not out_dir.exists() or not any(out_dir.glob("*.csv"))
    assert not out_dir.exists() or not any(out_dir.glob("figures/*.png"))
