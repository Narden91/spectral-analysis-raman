from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src
import src.analysis.biological_pca as biological_pca
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
            "loading_region_quantile": 0.8,
            "loading_region_min_width_cm": 8.0,
            "loading_region_merge_gap_cm": 6.0,
            "models": {
                "PCA_1330": {
                    "spectral_blocks": ["1300"],
                    "window": "full",
                    "bands": ["amide_iii_1260", "ch2_bend_1300"],
                },
                "PCA_2850": {
                    "spectral_blocks": ["2850"],
                    "window": "full",
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


def test_default_output_uses_index_and_canonical_dirs(monkeypatch, tmp_path) -> None:
    cfg = _cfg()
    expected_root = tmp_path / "Results_control_vs_disease"
    inputs = _inputs()
    results = fit_biological_pca_models(inputs, cfg=cfg)
    features = build_subject_biochemical_features(results)

    monkeypatch.setattr(biological_pca, "build_subject_inputs", lambda max_subjects=None, cfg=None: inputs)
    monkeypatch.setattr(biological_pca, "fit_biological_pca_models", lambda inputs, cfg=None: results)
    monkeypatch.setattr(biological_pca, "build_subject_biochemical_features", lambda results: features)

    def _canonical_dir(model: str) -> Path:
        path = expected_root / {
            "PCA_1330": "01_fingerprint_1300/06_biological_pca",
            "PCA_2850": "02_ch_stretch_2850/06_biological_pca",
            "PCA_whole": "03_whole_spectrum_1300_plus_2850/04_biological_pca",
        }[model]
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _cross_region_dir() -> Path:
        path = expected_root / "04_cross_region_summary"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _index_dir() -> Path:
        path = expected_root / "biological_pca"
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(src, "_output_dir", lambda key: expected_root)
    monkeypatch.setattr(src, "get_biological_pca_canonical_dir", _canonical_dir)
    monkeypatch.setattr(src, "get_cross_region_summary_dir", _cross_region_dir)
    monkeypatch.setattr(src, "get_biological_pca_index_dir", _index_dir)
    monkeypatch.setattr(biological_pca, "get_biological_pca_canonical_dir", src.get_biological_pca_canonical_dir)
    monkeypatch.setattr(biological_pca, "get_cross_region_summary_dir", src.get_cross_region_summary_dir)
    monkeypatch.setattr(biological_pca, "get_biological_pca_index_dir", src.get_biological_pca_index_dir)

    out = run_biological_pca(cfg=cfg)
    assert out == expected_root / "biological_pca"
    assert (_canonical_dir("PCA_1330") / "scores.csv").exists()
    assert (expected_root / "04_cross_region_summary" / "subject_biological_pca_scores.csv").exists()
    assert (out / "index.csv").exists()
    assert not (out / "PCA_1330" / "scores.csv").exists()
