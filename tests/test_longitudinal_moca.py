from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.longitudinal_moca import run_longitudinal_moca


def _cfg() -> dict:
    return {
        "analysis": {
            "longitudinal_moca": {
                "enabled": True,
                "min_subjects": 5,
                "fdr_alpha": 0.05,
                "cluster_alpha": 0.1,
                "n_permutations": 8,
                "random_state": 42,
                "candidate_region_min_width_cm": 1.0,
                "candidate_region_merge_gap_cm": 5.0,
                "top_scatterplot_regions": 2,
            }
        }
    }


def _feature_matrix() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(12):
        delta = float(index - 6)
        baseline = float(18 + index)
        group = "decline" if delta < 0 else "stable_or_improved"
        rows.append(
            {
                "subject_id": f"D{index:02d}",
                "condition": "Disease",
                "is_control": False,
                "moca_baseline": baseline,
                "moca_3m": float(baseline + delta),
                "delta_moca": delta,
                "rbans_total": float(80 + index * 2),
                "age": float(50 + index),
                "aspects_score": float(10 - (index % 3)),
                "delta_moca_group": group,
                "1300__1240.0": float(10 + delta * 2.5),
                "1300__1245.0": float(11 + delta * 2.7),
                "1300__1250.0": float(12 + delta * 2.9),
                "2850__2840.0": float(9 - delta * 1.8),
                "2850__2845.0": float(10 - delta * 2.0),
                "2850__2850.0": float(11 - delta * 2.2),
            }
        )
    for index in range(3):
        rows.append(
            {
                "subject_id": f"C{index:02d}",
                "condition": "Control",
                "is_control": True,
                "moca_baseline": 28.0,
                "moca_3m": 29.0,
                "delta_moca": 1.0,
                "rbans_total": 100.0,
                "age": 46.0,
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


def test_run_longitudinal_moca_writes_expected_outputs(tmp_path: Path) -> None:
    matrix_path = tmp_path / "subject_feature_matrix_whole.parquet"
    _feature_matrix().to_parquet(matrix_path, index=False)

    written = run_longitudinal_moca(
        feature_matrix_path=matrix_path,
        output_dir=tmp_path / "longitudinal_moca",
        cfg=_cfg(),
    )

    associations = pd.read_csv(written["delta_moca_associations"])
    candidates = pd.read_csv(written["delta_moca_candidate_regions"])

    delta_rows = associations.loc[associations["analysis_name"] == "delta_moca_association"]
    group_rows = associations.loc[associations["analysis_name"] == "decline_group_comparison"]

    assert int(delta_rows["n"].iloc[0]) == 12
    assert set(delta_rows["target"]) == {"delta_moca"}
    assert "decline_vs_stable_or_improved" in set(group_rows["contrast_name"])
    ok_rows = group_rows.loc[group_rows["status"] == "ok"]
    assert int(ok_rows["n_group_a"].iloc[0]) == 6
    assert int(ok_rows["n_group_b"].iloc[0]) == 6
    assert not candidates.empty
    assert set(candidates["analysis_name"]).issubset(
        {"delta_moca_association", "decline_group_comparison"}
    )
    assert "selection_basis" in candidates.columns
    scatter_files = list((tmp_path / "longitudinal_moca" / "delta_moca_scatterplots").glob("*.png"))
    assert scatter_files
