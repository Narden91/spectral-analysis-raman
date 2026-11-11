from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.covariate_models import _fit_logistic, _fit_ols, run_covariate_region_models


def test_standard_covariate_models_return_predictor_inference() -> None:
    rng = np.random.default_rng(42)
    predictor = pd.Series(rng.normal(size=80))
    covariates = pd.DataFrame({"age": rng.normal(65, 6, size=80)})
    continuous = pd.Series(0.7 * predictor + 0.03 * covariates["age"] + rng.normal(scale=0.4, size=80))
    coefficient, _intercept, r_squared, p_value = _fit_ols(continuous, predictor, covariates)
    assert coefficient > 0
    assert r_squared > 0
    assert p_value < 0.05

    logits = -0.4 + 1.2 * predictor + 0.02 * covariates["age"]
    binary = pd.Series(rng.binomial(1, 1.0 / (1.0 + np.exp(-logits))))
    coefficient, _intercept, accuracy, p_value = _fit_logistic(binary, predictor, covariates)
    assert coefficient > 0
    assert accuracy > 0.5
    assert p_value < 0.05


def _feature_matrix() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(12):
        delta = float(index - 5)
        group = "decline" if delta < 0 else "stable_or_improved"
        rows.append(
            {
                "subject_id": f"D{index:02d}",
                "condition": "Disease",
                "is_control": False,
                "age": float(50 + index),
                "delta_moca": delta,
                "moca_baseline": float(20 + index),
                "delta_moca_group": group,
                "1300__1240.0": float(10 + delta * 2.0),
                "1300__1245.0": float(11 + delta * 2.2),
                "1300__1250.0": float(12 + delta * 2.4),
                "2850__2840.0": float(7 - delta * 1.4),
                "2850__2850.0": float(8 - delta * 1.6),
                "2850__2930.0": float(9 - delta * 1.8),
            }
        )
    return pd.DataFrame(rows)


def _candidate_regions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_analysis": ["clinical_correlation_matrix", "disease_stratification"],
            "analysis_name": ["clinical_correlation_matrix", "low_vs_high_moca"],
            "target": ["delta_moca", pd.NA],
            "contrast_name": ["delta_moca", "decline_vs_stable_or_improved"],
            "group_variable": [pd.NA, "delta_moca_group"],
            "group_a": [pd.NA, "decline"],
            "group_b": [pd.NA, "stable_or_improved"],
            "spectral_block": ["1300", "2850"],
            "candidate_region_id": ["1300_R1", "2850_R1"],
            "selection_basis": ["significant_or_cluster", "top_abs_effect_fallback"],
            "start_cm-1": [1240.0, 2840.0],
            "end_cm-1": [1250.0, 2850.0],
            "width_cm-1": [10.0, 10.0],
            "n_features": [3, 2],
            "min_fdr_q": [0.01, 0.4],
            "min_cluster_p_value": [0.02, 0.3],
            "n_fdr_significant": [2, 0],
            "n_cluster_significant": [2, 0],
            "max_abs_effect": [0.8, 0.4],
            "mean_abs_effect": [0.75, 0.35],
            "integrated_abs_effect": [7.0, 3.0],
            "effect_metric": ["spearman_rho", "rank_biserial"],
        }
    )


def test_run_covariate_region_models_reports_missing_covariates_and_formulas(tmp_path: Path) -> None:
    matrix_path = tmp_path / "subject_feature_matrix_whole.parquet"
    candidate_path = tmp_path / "candidate_regions_cluster_corrected.csv"
    _feature_matrix().to_parquet(matrix_path, index=False)
    _candidate_regions().to_csv(candidate_path, index=False)

    written = run_covariate_region_models(
        feature_matrix_path=matrix_path,
        candidate_regions_path=candidate_path,
        output_dir=tmp_path,
        cfg={
            "analysis": {
                "covariate_models": {
                    "enabled": True,
                    "covariates": ["age", "sex"],
                    "min_subjects": 6,
                    "random_state": 42,
                }
            },
            "whole_spectra": {
                "bands": {
                    "amide_iii_1258": [1235.0, 1255.0],
                    "ch2_sym_stretch_2853": [2840.0, 2855.0],
                    "ch3_asym_stretch_2937": [2925.0, 2935.0],
                },
                "ratio_pairs": [
                    ["amide_iii_1258", "ch2_sym_stretch_2853"],
                    ["ch2_sym_stretch_2853", "ch3_asym_stretch_2937"],
                ],
            },
        },
    )

    models = pd.read_csv(written["covariate_adjusted_region_models"])
    availability = pd.read_csv(written["covariate_availability_report"])

    assert set(models["predictor_type"]) == {"region", "ratio"}
    assert models["formula"].str.contains("age").all()
    assert not models["formula"].str.contains("sex").any()
    assert models["n_subjects"].ge(6).all()
    assert availability["missing_covariates"].str.contains("sex").any()
    assert availability["available_covariates"].str.contains("age").all()


def test_run_covariate_region_models_skips_writes_when_no_candidates(tmp_path: Path) -> None:
    matrix_path = tmp_path / "subject_feature_matrix_whole.parquet"
    candidate_path = tmp_path / "candidate_regions_cluster_corrected.csv"
    _feature_matrix().to_parquet(matrix_path, index=False)
    # Empty candidate table (Phase 6 produced zero rows): only a header, no
    # data rows for run_covariate_region_models to iterate over.
    pd.DataFrame(columns=["source_analysis", "analysis_name", "target", "group_variable"]).to_csv(
        candidate_path, index=False
    )

    outputs = run_covariate_region_models(
        feature_matrix_path=matrix_path,
        candidate_regions_path=candidate_path,
        output_dir=tmp_path / "out",
        cfg={"analysis": {"covariate_models": {"enabled": True}}},
    )

    assert set(outputs) == {"_status"}
    assert outputs["_status"]["phase_key"] == "covariate_models"
    assert outputs["_status"]["status"] == "missing_required_data"
    out_dir = tmp_path / "out"
    assert not out_dir.exists() or not any(out_dir.glob("*.csv"))
