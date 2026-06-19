from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.biological_pca import (
    ModelResult,
    SubjectInputs,
    biochemical_summary,
    build_subject_biochemical_features,
    compute_band_contributions,
    fit_biological_pca_models,
    select_subjects,
    validate_biological_pca_config,
    validate_outputs,
)


def _cfg() -> dict:
    return {
        "analysis": {"random_state": 42},
        "whole_spectra": {
            "bands": {
                "amide_iii_1260": [1240, 1280],
                "ch2_bend_1300": [1290, 1320],
                "ch2_sym_stretch_2850": [2840, 2865],
                "ch3_sym_stretch_2930": [2920, 2945],
            }
        },
        "biological_pca": {
            "n_components": 2,
            "aggregation": "median",
            "scale": False,
            "models": {
                "PCA_1330": {
                    "spectral_blocks": ["1300"],
                    "window": [1240, 1320],
                    "bands": ["amide_iii_1260", "ch2_bend_1300"],
                },
                "PCA_2850": {
                    "spectral_blocks": ["2850"],
                    "window": [2840, 2945],
                    "bands": ["ch2_sym_stretch_2850", "ch3_sym_stretch_2930"],
                },
                "PCA_whole": {
                    "spectral_blocks": ["1300", "2850"],
                    "window": "full",
                    "bands": [
                        "amide_iii_1260",
                        "ch2_bend_1300",
                        "ch2_sym_stretch_2850",
                        "ch3_sym_stretch_2930",
                    ],
                },
            },
            "biochemical_classes": {
                "lipid": ["ch2_bend_1300", "ch2_sym_stretch_2850"],
                "protein": ["amide_iii_1260", "ch3_sym_stretch_2930"],
                "nucleic_acid": [],
                "mixed": [],
            },
        },
    }


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
            "n_samples": [2] * n_subjects,
            "n_spots": [2] * n_subjects,
            "n_pixels": [10] * n_subjects,
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
                "n_samples": [1] * (n_subjects * 2),
                "n_spots": [1] * (n_subjects * 2),
                "n_pixels": [5] * (n_subjects * 2),
                "n_retained_pixels": [5] * (n_subjects * 2),
            }
        ),
        input_provenance=pd.DataFrame(
            columns=[
                "subject_id",
                "sample_id",
                "spot_id",
                "pixel_id",
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


def test_whole_loadings_preserve_block_identity_and_unique_pairs() -> None:
    cfg = _cfg()
    results = fit_biological_pca_models(_inputs(), cfg=cfg)
    whole = next(result for result in results if result.name == "PCA_whole")
    assert "spectral_block" in whole.loadings.columns
    assert set(whole.loadings["spectral_block"]) == {"1300", "2850"}
    assert not whole.loadings.duplicated(subset=["spectral_block", "raman_shift"]).any()


def test_band_integration_records_represented_overlap_and_normalizes() -> None:
    cfg = _cfg()
    loadings = pd.DataFrame(
        {
            "raman_shift": [1240.0, 1260.0, 1280.0, 1300.0, 1320.0],
            "PC1_loading": [1.0, 1.0, 1.0, 2.0, 2.0],
            "PC2_loading": [0.0, 0.0, 0.0, 0.0, 0.0],
            "region_model": "PCA_1330",
        }
    )
    contributions = compute_band_contributions(
        "PCA_1330",
        cfg["biological_pca"]["models"]["PCA_1330"],
        loadings,
        cfg,
    )
    assert set(contributions["represented_window_min"]) == {1240.0, 1300.0}
    pc1 = contributions[contributions["PC"] == "PC1"]
    assert abs(pc1["relative_absolute_contribution"].sum() - 1.0) < 1e-6
    pc2 = contributions[contributions["PC"] == "PC2"]
    assert pc2["relative_absolute_contribution"].isna().all()


def test_interpretation_text_is_conservative() -> None:
    contributions = pd.DataFrame(
        {
            "region_model": ["PCA_1330", "PCA_1330"],
            "PC": ["PC1", "PC1"],
            "biochemical_class": ["lipid", "protein"],
            "relative_absolute_contribution": [0.7, 0.3],
        }
    )
    summary = biochemical_summary("PCA_1330", contributions)
    sentence = summary.loc[0, "interpretation_sentence"].lower()
    assert (
        "associated raman loading contributions" in sentence
        or "associated loading contributions" in sentence
        or "regional raman loading pattern" in sentence
    )
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
