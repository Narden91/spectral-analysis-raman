from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src
import src.analysis.biological_pca as biological_pca
import src.analysis.subject_features as subject_features
from src import get_single_range_phase_dirs
from src.analysis.biological_pca import (
    SubjectInputs,
    build_loading_region_summary,
    build_region_contributions,
    build_subject_biochemical_features,
    compute_loading_regions,
    fit_biological_pca_models,
    run_biological_pca,
    select_subjects,
    validate_biological_pca_config,
    validate_outputs,
)
from src.analysis.joint_dr import joint_pca_mfa
from src.analysis.subject_features import build_subject_feature_matrices, write_subject_feature_exports


def test_single_range_phase_numbering_matches_analysis_mode() -> None:
    control = get_single_range_phase_dirs("disease_vs_control")
    disease_only = get_single_range_phase_dirs("disease_only")

    assert list(control.values()) == [
        "01_group_comparisons",
        "02_area_level_pca_qc",
        "03_known_band_context",
        "04_variance_context",
        "05_region_robustness",
        "06_biological_pca",
    ]
    assert disease_only["intra_disease"] == "05_intra_disease"
    assert disease_only["biological_pca"] == "06_biological_pca"
    assert disease_only["robustness"] == "07_region_robustness"


def _cfg() -> dict:
    return {
        "analysis": {"random_state": 42},
        "whole_spectra": {
            "bands": {
                "amide_iii_1258": [1240, 1280],
                "ch2_bend_1300": [1290, 1320],
                "ch2_sym_stretch_2853": [2840, 2865],
                "ch3_asym_stretch_2937": [2920, 2945],
            }
        },
        "biological_pca": {
            "n_components": 3,
            "aggregation": "median",
            "scale": False,
            "loading_region_quantile": 0.8,
            "loading_region_min_width_cm": 8.0,
            "loading_region_merge_gap_cm": 6.0,
            "models": {
                "PCA_1330": {
                    "spectral_blocks": ["1300"],
                    "window": "full",
                    "bands": ["amide_iii_1258", "ch2_bend_1300"],
                },
                "PCA_2850": {
                    "spectral_blocks": ["2850"],
                    "window": "full",
                    "bands": ["ch2_sym_stretch_2853", "ch3_asym_stretch_2937"],
                },
                "PCA_whole": {
                    "spectral_blocks": ["1300", "2850"],
                    "window": "full",
                    "bands": [
                        "amide_iii_1258",
                        "ch2_bend_1300",
                        "ch2_sym_stretch_2853",
                        "ch3_asym_stretch_2937",
                    ],
                },
            },
            "biochemical_classes": {
                "lipid": ["ch2_bend_1300", "ch2_sym_stretch_2853"],
                "protein": ["amide_iii_1258", "ch3_asym_stretch_2937"],
                "nucleic_acid": [],
                "mixed": [],
            },
        },
    }


def test_biological_pca_filters_cohort_before_subject_limit() -> None:
    index = pd.DataFrame(
        {
            "subject_id": ["C01", "D01", "D02", "U01", "X01", "X01"],
            "is_control": [True, False, False, np.nan, True, False],
        }
    )
    disease_statuses, excluded = biological_pca._eligible_subject_control_statuses(index, "disease_only")
    comparison_statuses, _ = biological_pca._eligible_subject_control_statuses(index, "disease_vs_control")

    assert select_subjects(disease_statuses, max_subjects=1) == ["D01"]
    assert disease_statuses == {"D01": False, "D02": False}
    assert comparison_statuses == {"C01": True, "D01": False, "D02": False}
    assert {item["subject_id"] for item in excluded} == {"U01", "X01"}


def _inputs(n_subjects: int = 4) -> SubjectInputs:
    subjects = [f"S{i:02d}" for i in range(n_subjects)]
    axes = {
        "1300": np.array([1240.0, 1260.0, 1280.0, 1300.0, 1320.0]),
        "2850": np.array([2840.0, 2860.0, 2920.0, 2945.0]),
    }
    matrices = {
        "1300": np.vstack([np.arange(5, dtype=float) + i for i in range(n_subjects)]),
        "2850": np.vstack([np.arange(4, dtype=float) * 0.5 + i for i in range(n_subjects)]),
    }
    subject_spectra = pd.DataFrame(
        {
            "subject_id": subjects,
            "n_measurements": [2] * n_subjects,
            "n_areas": [2] * n_subjects,
            "n_total_pixels": [10] * n_subjects,
            "n_retained_pixels": [10] * n_subjects,
            **{f"1300__{x:.1f}": matrices["1300"][:, i] for i, x in enumerate(axes["1300"])},
            **{f"2850__{x:.1f}": matrices["2850"][:, i] for i, x in enumerate(axes["2850"])},
        }
    )
    return SubjectInputs(
        subject_spectra=subject_spectra,
        subject_region_counts=pd.DataFrame(
            {
                "subject_id": np.repeat(subjects, 2),
                "spectral_range": ["1300", "2850"] * n_subjects,
                "n_measurements": [1] * (n_subjects * 2),
                "n_areas": [1] * (n_subjects * 2),
                "n_total_pixels": [5] * (n_subjects * 2),
                "n_retained_pixels": [5] * (n_subjects * 2),
            }
        ),
        input_provenance=pd.DataFrame(
            columns=[
                "subject_id",
                "sample_id",
                "spot_id",
                "measurement_id",
                "x_coord",
                "y_coord",
                "spectral_range",
                "filepath",
                "retained_after_qc",
            ]
        ),
        axes=axes,
        matrices=matrices,
        subjects=subjects,
    )


def test_max_subjects_is_sorted_and_deterministic() -> None:
    assert select_subjects(["S03", "S01", "S02"], max_subjects=2) == ["S01", "S02"]


def test_config_validation_rejects_missing_model() -> None:
    cfg = _cfg()
    del cfg["biological_pca"]["models"]["PCA_whole"]
    with pytest.raises(ValueError, match="Missing Biological PCA model"):
        validate_biological_pca_config(cfg)


def test_config_validation_rejects_duplicate_class_mapping() -> None:
    cfg = _cfg()
    cfg["biological_pca"]["biochemical_classes"]["protein"].append("ch2_bend_1300")
    with pytest.raises(ValueError, match="multiple biochemical classes"):
        validate_biological_pca_config(cfg)


def test_exact_component_validation_rejects_undersized_data() -> None:
    cfg = _cfg()
    cfg["biological_pca"]["n_components"] = 5
    with pytest.raises(ValueError, match="requires n_components=5"):
        fit_biological_pca_models(_inputs(n_subjects=4), cfg=cfg)


def test_full_axis_single_range_loadings_are_preserved() -> None:
    cfg = _cfg()
    results = fit_biological_pca_models(_inputs(), cfg=cfg)
    fingerprint = next(result for result in results if result.name == "PCA_1330")
    assert np.array_equal(
        fingerprint.loadings["raman_shift"].to_numpy(dtype=float),
        _inputs().axes["1300"],
    )


def test_whole_loadings_preserve_block_identity_and_unique_pairs() -> None:
    cfg = _cfg()
    results = fit_biological_pca_models(_inputs(), cfg=cfg)
    whole = next(result for result in results if result.name == "PCA_whole")
    assert "spectral_block" in whole.loadings.columns
    assert set(whole.loadings["spectral_block"]) == {"1300", "2850"}
    assert not whole.loadings.duplicated(subset=["spectral_block", "raman_shift"]).any()
    assert whole.method_metadata["method"] == "mfa_first_eigenvalue_scaling_v1"


def test_mfa_scaled_blocks_have_unit_first_eigenvalue() -> None:
    rng = np.random.default_rng(42)
    left = rng.normal(scale=20.0, size=(30, 8))
    right = rng.normal(scale=0.001, size=(30, 5))
    result = joint_pca_mfa(np.hstack([left, right]), split_index=8, n_components=3)
    for block in result["scaled_blocks"].values():
        eigenvalue = float(np.linalg.svd(block, full_matrices=False, compute_uv=False)[0] ** 2 / (len(block) - 1))
        assert np.isclose(eigenvalue, 1.0, rtol=1e-10, atol=1e-12)


def test_mfa_rejects_degenerate_block() -> None:
    rng = np.random.default_rng(42)
    with pytest.raises(ValueError, match="degenerate"):
        joint_pca_mfa(
            np.hstack([rng.normal(size=(10, 4)), np.zeros((10, 3))]),
            split_index=4,
            n_components=2,
        )


def test_loading_region_extraction_normalizes_relative_contributions() -> None:
    cfg = _cfg()
    loadings = pd.DataFrame(
        {
            "raman_shift": [1240.0, 1260.0, 1280.0, 1300.0, 1320.0],
            "PC1_loading": [1.0, 1.0, 1.0, 2.0, 2.0],
            "PC2_loading": [0.0, 0.0, 0.0, 0.0, 0.0],
            "region_model": "PCA_1330",
        }
    )
    regions = compute_loading_regions("PCA_1330", loadings, cfg)
    contributions = build_region_contributions(regions)
    assert regions["start_cm-1"].min() == 1300.0
    assert regions["end_cm-1"].max() == 1320.0
    pc1 = contributions[contributions["PC"] == "PC1"]
    assert abs(pc1["relative_absolute_contribution"].sum() - 1.0) < 1e-6


def test_interpretation_text_is_conservative() -> None:
    contributions = pd.DataFrame(
        {
            "region_model": ["PCA_1330", "PCA_1330"],
            "PC": ["PC1", "PC1"],
            "spectral_block": ["single", "single"],
            "region_id": ["PC1_R1", "PC1_R2"],
            "region_label": ["1240.0-1280.0 cm-1", "1290.0-1320.0 cm-1"],
            "absolute_integrated_loading": [0.7, 0.3],
            "relative_absolute_contribution": [0.7, 0.3],
            "global_rank": [1, 2],
            "block_rank": [1, 2],
            "dominant_sign": ["positive", "negative"],
        }
    )
    annotations = pd.DataFrame(
        {
            "region_id": ["PC1_R1", "PC1_R2"],
            "biochemical_context": ["protein_associated_region", "mixed_context_region"],
        }
    )
    summary = build_loading_region_summary("PCA_1330", contributions, annotations)
    sentence = summary.loc[0, "interpretation_sentence"].lower()
    assert "loading-derived intervals" in sentence or "broad spectral structure" in sentence
    assert "increased" not in sentence
    assert "diagnoses" not in sentence
    assert "disease separation" not in sentence


def test_output_uniqueness_validation_rejects_duplicate_subject_features() -> None:
    features = pd.DataFrame({"subject_id": ["S01", "S01"], "PCA_1330_PC1": [0.1, 0.2]})
    with pytest.raises(ValueError, match="duplicated subject_id"):
        validate_outputs(_inputs(), [], features, cfg=_cfg())


def test_fit_outputs_have_unique_subject_rows_and_feature_columns() -> None:
    inputs = _inputs()
    results = fit_biological_pca_models(inputs, cfg=_cfg())
    features = build_subject_biochemical_features(results)
    validate_outputs(inputs, results, features, cfg=_cfg())
    assert "1300__1240.0" in inputs.subject_spectra.columns
    assert "2850__2840.0" in inputs.subject_spectra.columns
    assert not features["subject_id"].duplicated().any()


def test_subject_feature_matrices_preserve_block_identity(monkeypatch) -> None:
    clinical = pd.DataFrame(
        {
            "subject_id": ["S00", "S01", "S02", "S03"],
            "condition": ["Disease"] * 4,
            "is_control": [False] * 4,
            "gender": ["female", "male", "female", "male"],
            "moca_baseline_total": [19.0, 23.0, 20.0, 24.0],
        }
    )
    endpoints = pd.DataFrame(
        {
            "subject_id": ["S00", "S01", "S02", "S03"],
            "moca_severity_group": ["low_moca", "high_moca", "high_moca", "high_moca"],
            "delta_moca_group": ["decline", "stable_or_improved", "decline", "stable_or_improved"],
            "lesion_location_group": ["left", "right", "posterior", "left"],
        }
    )
    monkeypatch.setattr(subject_features, "load_clinical_subjects", lambda *args, **kwargs: clinical)
    monkeypatch.setattr(
        subject_features,
        "build_clinical_endpoints",
        lambda cfg=None: type("EndpointResult", (), {"clinical_subject_table": endpoints})(),
    )
    matrices, provenance = build_subject_feature_matrices(_inputs(), cfg=_cfg())
    assert "1300__1240.0" in matrices["1300"].columns
    assert "2850__2840.0" in matrices["2850"].columns
    assert "1300__1240.0" in matrices["whole"].columns
    assert "2850__2840.0" in matrices["whole"].columns
    assert not matrices["whole"]["subject_id"].duplicated().any()
    assert "analysis_mode" in provenance.columns


def test_subject_feature_export_writes_required_outputs(monkeypatch, tmp_path: Path) -> None:
    clinical = pd.DataFrame(
        {
            "subject_id": ["S00", "S01", "S02", "S03"],
            "condition": ["Disease"] * 4,
            "is_control": [False] * 4,
            "gender": ["female", "male", "female", "male"],
        }
    )
    endpoints = pd.DataFrame({"subject_id": ["S00", "S01", "S02", "S03"]})
    monkeypatch.setattr(subject_features, "load_clinical_subjects", lambda *args, **kwargs: clinical)
    monkeypatch.setattr(
        subject_features,
        "build_clinical_endpoints",
        lambda cfg=None: type("EndpointResult", (), {"clinical_subject_table": endpoints})(),
    )
    written = write_subject_feature_exports(_inputs(), output_dir=tmp_path, cfg=_cfg())
    assert (tmp_path / "subject_feature_matrix_1300.parquet").exists()
    assert (tmp_path / "subject_feature_matrix_2850.parquet").exists()
    assert (tmp_path / "subject_feature_matrix_whole.parquet").exists()
    assert (tmp_path / "subject_feature_provenance.csv").exists()
    roundtrip = pd.read_parquet(written["subject_feature_matrix_whole"])
    assert roundtrip["subject_id"].astype(str).nunique() == 4


def test_known_bands_only_annotate_after_loading_regions_exist() -> None:
    cfg = _cfg()
    results = fit_biological_pca_models(_inputs(), cfg=cfg)
    fingerprint = next(result for result in results if result.name == "PCA_1330")
    assert np.array_equal(
        fingerprint.loadings["raman_shift"].to_numpy(dtype=float),
        _inputs().axes["1300"],
    )
    assert not fingerprint.loading_regions.empty
    assert "overlapping_bands" in fingerprint.region_annotation_candidates.columns


def test_whole_loading_regions_preserve_block_identity() -> None:
    cfg = _cfg()
    results = fit_biological_pca_models(_inputs(), cfg=cfg)
    whole = next(result for result in results if result.name == "PCA_whole")
    assert set(whole.loading_regions["spectral_block"]) == {"1300", "2850"}
    assert {"start_cm-1", "end_cm-1", "PC", "absolute_integrated_loading"}.issubset(
        whole.loading_regions.columns
    )


def test_default_output_uses_only_canonical_dirs(monkeypatch, tmp_path) -> None:
    cfg = _cfg()
    expected_root = tmp_path / "Results_control_vs_disease"
    inputs = _inputs()
    results = fit_biological_pca_models(inputs, cfg=cfg)
    features = build_subject_biochemical_features(results)

    monkeypatch.setattr(
        biological_pca, "build_subject_inputs", lambda max_subjects=None, cfg=None, **kwargs: inputs
    )
    monkeypatch.setattr(biological_pca, "fit_biological_pca_models", lambda inputs, cfg=None: results)
    monkeypatch.setattr(biological_pca, "build_subject_biochemical_features", lambda results: features)
    monkeypatch.setattr(
        biological_pca,
        "write_subject_feature_exports",
        lambda inputs, output_dir=None, cfg=None, **kwargs: {
            "subject_feature_matrix_1300": expected_root / "04_cross_region_summary" / "01_subject_features" / "subject_feature_matrix_1300.parquet",
            "subject_feature_matrix_2850": expected_root / "04_cross_region_summary" / "01_subject_features" / "subject_feature_matrix_2850.parquet",
            "subject_feature_matrix_whole": expected_root / "04_cross_region_summary" / "01_subject_features" / "subject_feature_matrix_whole.parquet",
            "subject_feature_provenance": expected_root / "04_cross_region_summary" / "01_subject_features" / "subject_feature_provenance.csv",
        },
    )

    def _canonical_dir(model: str) -> Path:
        _phase_dirs = get_single_range_phase_dirs()
        _bio = _phase_dirs["biological_pca"]
        path = expected_root / {
            "PCA_1330": f"01_fingerprint_1300/{_bio}",
            "PCA_2850": f"02_ch_stretch_2850/{_bio}",
            "PCA_whole": "03_whole_spectrum_1300_plus_2850/04_biological_pca",
        }[model]
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _cross_region_dir() -> Path:
        path = expected_root / "04_cross_region_summary"
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(src, "_output_dir", lambda key: expected_root)
    monkeypatch.setattr(src, "get_biological_pca_canonical_dir", _canonical_dir)
    monkeypatch.setattr(src, "get_cross_region_summary_dir", _cross_region_dir)
    monkeypatch.setattr(biological_pca, "get_biological_pca_canonical_dir", src.get_biological_pca_canonical_dir)
    monkeypatch.setattr(biological_pca, "get_cross_region_summary_dir", src.get_cross_region_summary_dir)

    out = run_biological_pca(cfg=cfg)
    assert out["PCA_1330"] == _canonical_dir("PCA_1330")
    assert out["cross_region_summary"] == _cross_region_dir()
    assert (_canonical_dir("PCA_1330") / "scores.csv").exists()
    assert (_canonical_dir("PCA_1330") / "scores_pc1_pc3.png").exists()
    assert (_canonical_dir("PCA_1330") / "scores_pc2_pc3.png").exists()
    assert (_canonical_dir("PCA_1330") / "scores_pc_pairgrid.png").exists()
    assert (_canonical_dir("PCA_1330") / "loadings_pc1_pc2_pc3.png").exists()
    assert (_canonical_dir("PCA_1330") / "loading_regions_pc1_pc2_pc3.png").exists()
    assert (expected_root / "04_cross_region_summary" / "02_biological_pca_summary" / "subject_biological_pca_scores.csv").exists()
    assert not (expected_root / "biological_pca").exists()


def test_comparison_summary_and_consistency_include_pc3() -> None:
    cfg = _cfg()
    results = fit_biological_pca_models(_inputs(), cfg=cfg)
    comparison = biological_pca._build_pca_comparison_summary(results)
    assert set(comparison["PC"]) == {"PC1", "PC2", "PC3"}
