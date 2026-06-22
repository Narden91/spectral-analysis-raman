from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.disease_stratification import run_disease_stratification


def _cfg() -> dict:
    return {
        "analysis": {
            "disease_stratification": {
                "enabled": True,
                "min_subjects_per_group": 5,
                "cluster_alpha": 0.1,
                "n_permutations": 8,
                "random_state": 42,
                "candidate_region_min_width_cm": 1.0,
                "candidate_region_merge_gap_cm": 5.0,
            }
        }
    }


def _feature_matrix() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(12):
        low_moca = index < 6
        decline = index % 2 == 0
        lesion = "other" if index < 3 else ("left" if index < 6 else ("right" if index < 9 else "posterior"))
        rows.append(
            {
                "subject_id": f"S{index:02d}",
                "condition": "disease",
                "is_control": False,
                "moca_severity_group": "low_moca" if low_moca else "high_moca",
                "delta_moca_group": "decline" if decline else "stable_or_improved",
                "lesion_location_group": lesion,
                "1300__1240.0": 5.0 if low_moca else 1.0,
                "1300__1245.0": 4.0 if low_moca else 1.0,
                "1300__1250.0": 3.0 if low_moca else 1.0,
                "2850__2840.0": 2.0 if decline else 0.5,
                "2850__2845.0": 2.2 if decline else 0.5,
                "2850__2850.0": 2.4 if decline else 0.5,
            }
        )
    return pd.DataFrame(rows)


def test_run_disease_stratification_writes_expected_outputs(tmp_path: Path) -> None:
    matrix_path = tmp_path / "subject_feature_matrix_whole.parquet"
    _feature_matrix().to_parquet(matrix_path, index=False)

    written = run_disease_stratification(
        feature_matrix_path=matrix_path,
        output_dir=tmp_path / "disease_stratification",
        cfg=_cfg(),
    )

    tests = pd.read_csv(written["stratification_wavenumber_tests"])
    counts = pd.read_csv(written["stratification_group_counts"])
    candidates = pd.read_csv(written["stratification_candidate_regions"])

    assert "low_vs_high_moca" in set(tests["contrast_name"])
    assert "decline_vs_stable_or_improved" in set(tests["contrast_name"])
    assert "mannwhitney" in set(tests["test"].dropna())
    assert "kruskal" in set(tests["test"].dropna())
    assert "insufficient sample size" in set(tests["status"])
    assert "lesion_location_group" in set(counts["group_variable"])
    assert (written["stratification_effect_plots"] / "low_vs_high_moca.png").exists()
    assert not candidates.empty


def test_run_disease_stratification_rejects_duplicate_subject_rows(tmp_path: Path) -> None:
    frame = _feature_matrix()
    frame.loc[len(frame)] = frame.iloc[0]
    matrix_path = tmp_path / "subject_feature_matrix_whole.parquet"
    frame.to_parquet(matrix_path, index=False)

    try:
        run_disease_stratification(
            feature_matrix_path=matrix_path,
            output_dir=tmp_path / "disease_stratification",
            cfg=_cfg(),
        )
    except ValueError as exc:
        assert "one row per subject" in str(exc)
    else:
        raise AssertionError("Expected duplicate-subject validation to fail")
