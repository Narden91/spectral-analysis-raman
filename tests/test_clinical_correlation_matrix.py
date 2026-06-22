from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analysis.clinical_correlation_matrix import run_clinical_correlation_matrix


def _cfg() -> dict:
    return {
        "analysis": {
            "clinical_correlation_matrix": {
                "enabled": True,
                "min_subjects": 5,
                "fdr_alpha": 0.05,
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
        delta = float(index - 5)
        baseline = float(18 + index)
        moca_3m = float(baseline + delta)
        group = "decline" if delta < 0 else "stable_or_improved"
        rows.append(
            {
                "subject_id": f"D{index:02d}",
                "condition": "Disease",
                "is_control": False,
                "moca_baseline": baseline,
                "moca_3m": moca_3m,
                "delta_moca": delta,
                "rbans_total": float(80 + index * 2),
                "age": float(50 + index),
                "aspects_score": float(10 - (index % 3)) if index != 3 else None,
                "delta_moca_group": group,
                "1300__1240.0": float(10 + delta * 2.0),
                "1300__1245.0": float(11 + delta * 2.2),
                "1300__1250.0": float(12 + delta * 2.4),
                "2850__2840.0": float(7 - delta * 1.8),
                "2850__2845.0": float(8 - delta * 2.0),
                "2850__2850.0": float(9 - delta * 2.2),
            }
        )
    for index in range(4):
        rows.append(
            {
                "subject_id": f"C{index:02d}",
                "condition": "Control",
                "is_control": True,
                "moca_baseline": float(26 + index),
                "moca_3m": float(27 + index),
                "delta_moca": 1.0,
                "rbans_total": float(95 + index),
                "age": float(45 + index),
                "aspects_score": 10.0,
                "delta_moca_group": "stable_or_improved",
                "1300__1240.0": 100.0 + index,
                "1300__1245.0": 101.0 + index,
                "1300__1250.0": 102.0 + index,
                "2850__2840.0": 90.0 + index,
                "2850__2845.0": 91.0 + index,
                "2850__2850.0": 92.0 + index,
            }
        )
    return pd.DataFrame(rows)


def test_run_clinical_correlation_matrix_writes_expected_outputs(tmp_path: Path) -> None:
    matrix_path = tmp_path / "subject_feature_matrix_whole.parquet"
    _feature_matrix().to_parquet(matrix_path, index=False)

    written = run_clinical_correlation_matrix(
        feature_matrix_path=matrix_path,
        output_dir=tmp_path / "clinical_correlations",
        cfg=_cfg(),
    )

    correlations = pd.read_csv(written["clinical_correlation_matrix"])
    candidates = pd.read_csv(written["clinical_correlation_candidate_regions"])
    missingness = pd.read_csv(written["endpoint_missingness_report"])

    assert set(written) == {
        "clinical_correlation_matrix",
        "clinical_correlation_heatmap",
        "clinical_correlation_candidate_regions",
        "endpoint_missingness_report",
    }
    assert set(correlations["target"]) == {
        "moca_baseline",
        "moca_3m",
        "delta_moca",
        "rbans_total",
        "age",
        "aspects_score",
    }
    assert correlations.loc[correlations["target"] == "moca_baseline", "n"].nunique() == 1
    assert int(correlations.loc[correlations["target"] == "moca_baseline", "n"].iloc[0]) == 12
    assert int(missingness.loc[missingness["endpoint"] == "aspects_score", "n_missing"].iloc[0]) == 1
    assert (tmp_path / "clinical_correlations" / "clinical_correlation_heatmap.png").exists()
    assert not candidates.empty
    assert "delta_moca" in set(candidates["target"])


def test_run_clinical_correlation_matrix_rejects_duplicate_subject_rows(tmp_path: Path) -> None:
    frame = _feature_matrix()
    frame.loc[len(frame)] = frame.iloc[0]
    matrix_path = tmp_path / "subject_feature_matrix_whole.parquet"
    frame.to_parquet(matrix_path, index=False)

    with pytest.raises(ValueError, match="one row per subject"):
        run_clinical_correlation_matrix(
            feature_matrix_path=matrix_path,
            output_dir=tmp_path / "clinical_correlations",
            cfg=_cfg(),
        )
