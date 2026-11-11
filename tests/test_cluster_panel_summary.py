from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.cluster_panel_summary import summarize_panel_by_subgroup


def test_summarize_panel_by_subgroup_reports_features_and_endpoints() -> None:
    clinical = pd.DataFrame(
        {
            "subject_id": ["S1", "S2", "S3", "S4"],
            "moca_baseline": [18, 20, 26, 28],
            "delta_moca": [-2, -1, 3, 4],
        }
    )
    panel = pd.DataFrame(
        {
            "subject_id": ["S1", "S2", "S3", "S4"],
            "carotenoid_to_carbonyl_ratio": [0.2, 0.4, 1.0, 1.2],
            "ch_bulk_to_carbonyl_ratio": [2.0, 2.2, 3.0, 3.2],
        }
    )
    assignments = pd.DataFrame(
        {
            "subject_id": ["S1", "S2", "S3", "S4"],
            "subgroup_label": ["subgroup_1", "subgroup_1", "subgroup_2", "subgroup_2"],
        }
    )

    summary = summarize_panel_by_subgroup(
        clinical=clinical,
        panel=panel,
        assignments=assignments,
        endpoints=["moca_baseline", "delta_moca"],
    )

    assert {"subgroup_label", "variable_name", "variable_type", "n", "mean", "median"}.issubset(summary.columns)
    assert set(summary["variable_type"]) == {"panel_feature", "endpoint"}
    oxidative_s1 = summary.loc[
        (summary["subgroup_label"] == "subgroup_1")
        & (summary["variable_name"] == "carotenoid_to_carbonyl_ratio")
    ].iloc[0]
    assert oxidative_s1["n"] == 2
    assert oxidative_s1["median"] == pytest.approx(0.3)
