"""Tests for the Phase 11 integrated candidate evidence table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analysis.integrated_candidates import run_integrated_candidates


def _feature_matrix() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(16):
        delta = float(index - 7)
        group = "decline" if delta < 0 else "stable_or_improved"
        severity = "low_moca" if index < 8 else "high_moca"
        rows.append(
            {
                "subject_id": f"D{index:02d}",
                "condition": "Disease",
                "is_control": False,
                "age": float(50 + index),
                "sex": "female" if index % 2 == 0 else "male",
                "moca_baseline": float(18 + index),
                "moca_3m": float(20 + index + delta),
                "delta_moca": delta,
                "moca_severity_group": severity,
                "delta_moca_group": group,
                "1300__1240.0": float(10 + delta * 2.0 + (index - 8) * 0.5),
                "1300__1245.0": float(11 + delta * 2.2 + (index - 8) * 0.5),
                "1300__1250.0": float(12 + delta * 2.4 + (index - 8) * 0.5),
                "2850__2840.0": float(7 - delta * 1.4),
                "2850__2850.0": float(8 - delta * 1.6),
                "2850__2930.0": float(9 - delta * 1.8),
            }
        )
    return pd.DataFrame(rows)


def _candidate_regions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_analysis": [
                "clinical_correlation_matrix",
                "disease_stratification",
                "longitudinal_moca",
            ],
            "analysis_name": [
                "clinical_correlation_matrix",
                "low_vs_high_moca",
                "delta_moca_association",
            ],
            "target": ["delta_moca", pd.NA, "delta_moca"],
            "contrast_name": ["delta_moca", "decline_vs_stable_or_improved", "delta_moca"],
            "group_variable": [pd.NA, "delta_moca_group", pd.NA],
            "group_a": [pd.NA, "decline", pd.NA],
            "group_b": [pd.NA, "stable_or_improved", pd.NA],
            "spectral_block": ["1300", "1300", "2850"],
            "candidate_region_id": ["1300_R1", "1300_R2", "2850_R1"],
            "selection_basis": [
                "significant_or_cluster",
                "significant_or_cluster",
                "top_abs_effect_fallback",
            ],
            "start_cm-1": [1240.0, 1240.0, 2840.0],
            "end_cm-1": [1250.0, 1250.0, 2850.0],
            "width_cm-1": [10.0, 10.0, 10.0],
            "n_features": [3, 3, 2],
            "min_fdr_q": [0.01, 0.04, 0.4],
            "min_cluster_p_value": [0.02, 0.03, 0.3],
            "n_fdr_significant": [2, 1, 0],
            "n_cluster_significant": [2, 1, 0],
            "max_abs_effect": [0.8, 0.5, 0.3],
            "mean_abs_effect": [0.75, 0.45, 0.25],
            "integrated_abs_effect": [7.0, 4.0, 2.0],
            "effect_metric": ["spearman_rho", "rank_biserial", "spearman_rho"],
        }
    )


def _covariate_models() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_analysis": ["clinical_correlation_matrix", "disease_stratification"],
            "analysis_name": ["clinical_correlation_matrix", "low_vs_high_moca"],
            "target": ["delta_moca", pd.NA],
            "contrast_name": ["delta_moca", "decline_vs_stable_or_improved"],
            "group_variable": [pd.NA, "delta_moca_group"],
            "predictor_type": ["region", "region"],
            "predictor_name": ["1300_R1", "1300_R2"],
            "candidate_region_id": ["1300_R1", "1300_R2"],
            "status": ["ok", "ok"],
            "coefficient": [0.5, -0.3],
            "coef_p_value": [0.001, 0.004],
            "coef_fdr_q": [0.002, 0.004],
            "coef_significant_fdr": [True, True],
        }
    )


def _pls_vip() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": ["delta_moca"] * 6,
            "feature_name": [
                "1300__1240.0",
                "1300__1245.0",
                "1300__1250.0",
                "2850__2840.0",
                "2850__2850.0",
                "2850__2930.0",
            ],
            "spectral_block": ["1300"] * 3 + ["2850"] * 3,
            "wavenumber": [1240.0, 1245.0, 1250.0, 2840.0, 2850.0, 2930.0],
            "vip_score": [1.5, 1.3, 1.1, 0.8, 0.9, 1.2],
            "above_threshold": [True, True, True, False, False, True],
        }
    )


def _splsda_bands() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group_variable": ["delta_moca_group"] * 6,
            "feature_name": [
                "1300__1240.0",
                "1300__1245.0",
                "1300__1250.0",
                "2850__2840.0",
                "2850__2850.0",
                "2850__2930.0",
            ],
            "spectral_block": ["1300"] * 3 + ["2850"] * 3,
            "wavenumber": [1240.0, 1245.0, 1250.0, 2840.0, 2850.0, 2930.0],
            "loading_weight": [0.5, 0.3, 0.0, 0.0, 0.2, 0.0],
            "selected": [True, True, False, False, True, False],
        }
    )


@pytest.fixture
def input_paths(tmp_path: Path) -> dict[str, Path]:
    matrix_path = tmp_path / "subject_feature_matrix_whole.parquet"
    _feature_matrix().to_parquet(matrix_path, index=False)

    candidate_path = tmp_path / "candidate_regions_cluster_corrected.csv"
    _candidate_regions().to_csv(candidate_path, index=False)

    covariate_path = tmp_path / "covariate_adjusted_region_models.csv"
    _covariate_models().to_csv(covariate_path, index=False)

    vip_path = tmp_path / "pls_vip_bands.csv"
    _pls_vip().to_csv(vip_path, index=False)

    splsda_path = tmp_path / "pls_l1da_selected_bands.csv"
    _splsda_bands().to_csv(splsda_path, index=False)

    return {
        "feature_matrix": matrix_path,
        "candidate_regions": candidate_path,
        "covariate_models": covariate_path,
        "pls_vip": vip_path,
        "splsda_bands": splsda_path,
    }


def test_run_integrated_candidates_produces_expected_outputs(input_paths: dict[str, Path], tmp_path: Path) -> None:
    outputs = run_integrated_candidates(
        feature_matrix_path=input_paths["feature_matrix"],
        candidate_regions_path=input_paths["candidate_regions"],
        covariate_models_path=input_paths["covariate_models"],
        pls_vip_path=input_paths["pls_vip"],
        splsda_bands_path=input_paths["splsda_bands"],
        output_dir=tmp_path,
        cfg={
            "analysis": {"integrated_candidates": {"enabled": True, "vip_threshold": 1.0}},
            "whole_spectra": {
                "bands": {
                    "amide_iii_1258": [1235.0, 1255.0],
                    "ch2_sym_stretch_2853": [2840.0, 2855.0],
                    "ch3_asym_stretch_2937": [2925.0, 2935.0],
                },
                "biochemical_classes": {
                    "protein": ["amide_iii_1258"],
                    "lipid": ["ch2_sym_stretch_2853", "ch3_asym_stretch_2937"],
                },
            },
        },
    )

    assert "integrated_candidate_regions" in outputs
    df = pd.read_csv(outputs["integrated_candidate_regions"])
    assert len(df) == 3
    assert set(df["candidate_region_id"]) == {"1300_R1", "1300_R2", "2850_R1"}
    assert "evidence_status" in df.columns
    assert "source_region_id" in df.columns
    assert "n_overlapping_analyses" in df.columns
    assert "n_supporting_sources" not in df.columns
    assert "selection_basis" in df.columns
    assert "significance_supported" in df.columns
    assert "interpretation_note" in df.columns
    assert "biochemical_class" in df.columns

    supported = df.loc[df["candidate_region_id"] == "1300_R1"]
    assert supported["evidence_status"].iloc[0] == "cluster_supported_exploratory"
    assert bool(supported["covariate_persisted"].iloc[0]) is True
    assert bool(supported["pls_vip_coverage"].iloc[0]) is True
    assert supported["inference_role"].iloc[0] == "exploratory"

    fallback = df.loc[df["candidate_region_id"] == "2850_R1"]
    assert bool(fallback["significance_supported"].iloc[0]) is False


def test_run_integrated_candidates_disabled_returns_empty(tmp_path: Path) -> None:
    outputs = run_integrated_candidates(
        output_dir=tmp_path,
        cfg={"analysis": {"integrated_candidates": {"enabled": False}}},
    )
    assert outputs == {}


def test_run_integrated_candidates_skips_write_when_no_candidate_regions(tmp_path: Path) -> None:
    # No candidate_regions_path is supplied and none exists at the resolved
    # default location, so run_integrated_candidates must skip the CSV write
    # (previously this wrote a header-only integrated_candidate_regions.csv).
    outputs = run_integrated_candidates(
        candidate_regions_path=tmp_path / "does_not_exist.csv",
        output_dir=tmp_path / "out",
        cfg={"analysis": {"integrated_candidates": {"enabled": True}}},
    )

    assert set(outputs) == {"_status"}
    assert outputs["_status"]["phase_key"] == "integrated_candidates"
    assert outputs["_status"]["status"] == "missing_required_data"
    assert not (tmp_path / "out" / "integrated_candidate_regions.csv").exists()
