from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.spectral_regions import build_candidate_regions, run_candidate_region_detector


def test_build_candidate_regions_returns_expected_window() -> None:
    tests = pd.DataFrame(
        {
            "target": ["delta_moca"] * 5,
            "spectral_block": ["1300"] * 5,
            "wavenumber": [1240.0, 1245.0, 1250.0, 1300.0, 1305.0],
            "rho": [0.1, 0.8, 0.9, 0.2, 0.1],
            "fdr_q": [0.5, 0.01, 0.02, 0.6, 0.7],
            "significant_fdr": [False, True, True, False, False],
            "cluster_p_value": [1.0, 0.02, 0.02, 1.0, 1.0],
            "status": ["ok"] * 5,
        }
    )

    regions = build_candidate_regions(
        tests,
        group_columns=("target",),
        effect_column="rho",
        min_width_cm=1.0,
        merge_gap_cm=6.0,
        fallback_quantile=0.9,
    )

    assert len(regions) == 1
    assert float(regions.loc[0, "start_cm-1"]) == 1245.0
    assert float(regions.loc[0, "end_cm-1"]) == 1250.0
    assert regions.loc[0, "selection_basis"] == "significant_or_cluster"


def test_run_candidate_region_detector_writes_unified_table(tmp_path: Path) -> None:
    clinical_dir = tmp_path / "clinical"
    long_dir = tmp_path / "long"
    strat_dir = tmp_path / "strat"
    clinical_dir.mkdir()
    long_dir.mkdir()
    strat_dir.mkdir()

    clinical_path = clinical_dir / "clinical_correlation_candidate_regions.csv"
    longitudinal_path = long_dir / "delta_moca_candidate_regions.csv"
    stratification_path = strat_dir / "stratification_candidate_regions.csv"

    pd.DataFrame(
        {
            "target": ["delta_moca"],
            "spectral_block": ["1300"],
            "candidate_region_id": ["1300_R1"],
            "selection_basis": ["significant_or_cluster"],
            "start_cm-1": [1240.0],
            "end_cm-1": [1250.0],
            "width_cm-1": [10.0],
            "n_features": [3],
            "min_fdr_q": [0.01],
            "min_cluster_p_value": [0.02],
            "n_fdr_significant": [2],
            "n_cluster_significant": [2],
            "max_abs_rho": [0.8],
            "mean_abs_rho": [0.75],
            "integrated_abs_rho": [7.2],
        }
    ).to_csv(clinical_path, index=False)
    pd.DataFrame(
        {
            "analysis_name": ["decline_group_comparison"],
            "contrast_name": ["decline_vs_stable_or_improved"],
            "spectral_block": ["2850"],
            "candidate_region_id": ["2850_R1"],
            "selection_basis": ["top_abs_effect_fallback"],
            "start_cm-1": [2840.0],
            "end_cm-1": [2850.0],
            "width_cm-1": [10.0],
            "n_features": [3],
            "min_fdr_q": [0.5],
            "min_cluster_p_value": [0.3],
            "n_fdr_significant": [0],
            "n_cluster_significant": [0],
            "max_abs_effect": [0.4],
            "mean_abs_effect": [0.35],
            "integrated_abs_effect": [3.5],
            "effect_metric": ["rank_biserial"],
        }
    ).to_csv(longitudinal_path, index=False)
    pd.DataFrame(
        {
            "contrast_name": ["low_vs_high_moca"],
            "group_variable": ["moca_severity_group"],
            "spectral_block": ["1300"],
            "candidate_region_id": ["1300_R2"],
            "selection_basis": ["top_abs_effect_fallback"],
            "start_cm-1": [1300.0],
            "end_cm-1": [1310.0],
            "width_cm-1": [10.0],
            "n_features": [3],
            "min_fdr_q": [0.6],
            "min_cluster_p_value": [0.4],
            "max_abs_effect": [0.3],
            "mean_abs_effect": [0.25],
            "integrated_abs_effect": [2.5],
        }
    ).to_csv(stratification_path, index=False)

    written = run_candidate_region_detector(
        clinical_outputs={"clinical_correlation_candidate_regions": clinical_path},
        longitudinal_outputs={"delta_moca_candidate_regions": longitudinal_path},
        stratification_outputs={"stratification_candidate_regions": stratification_path},
        output_dir=tmp_path,
        cfg={"analysis": {"candidate_regions": {"enabled": True}}},
    )

    combined = pd.read_csv(written["candidate_regions_cluster_corrected"])
    assert set(combined["source_analysis"]) == {
        "clinical_correlation_matrix",
        "longitudinal_moca",
        "disease_stratification",
    }
    assert "effect_metric" in combined.columns
    assert "candidate_regions_cluster_corrected" in written
