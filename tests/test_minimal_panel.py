"""Tests for the Phase 13 minimal candidate Raman panel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.minimal_panel import run_minimal_panel


def _feature_matrix() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_subjects = 20
    rows: list[dict[str, object]] = []
    for index in range(n_subjects):
        delta = float(index - 9)
        group = "decline" if delta < 0 else "stable_or_improved"
        severity = "low_moca" if index < 10 else "high_moca"
        # signal feature correlates with delta; noise does not
        signal = float(delta * 2.0 + rng.normal(0, 0.5))
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
                "1300__1240.0": signal,
                "1300__1245.0": signal + rng.normal(0, 0.2),
                "1300__1250.0": signal + rng.normal(0, 0.2),
                "1300__1500.0": float(rng.normal(0, 1)),
                "2850__2840.0": float(7 - delta * 1.4 + rng.normal(0, 0.3)),
                "2850__2850.0": float(8 - delta * 1.6 + rng.normal(0, 0.3)),
                "2850__2930.0": float(rng.normal(0, 1)),
            }
        )
    return pd.DataFrame(rows)


def _integrated_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_region_id": ["1300_R1", "2850_R1"],
            "spectral_block": ["1300", "2850"],
            "start_cm-1": [1238.0, 2838.0],
            "end_cm-1": [1252.0, 2852.0],
            "evidence_level": ["High", "Medium"],
            "covariate_persisted": [True, False],
            "pls_vip_coverage": [True, False],
        }
    )


@pytest.fixture
def input_paths(tmp_path: Path) -> dict[str, Path]:
    matrix_path = tmp_path / "subject_feature_matrix_whole.parquet"
    _feature_matrix().to_parquet(matrix_path, index=False)

    integrated_path = tmp_path / "integrated_candidate_regions.csv"
    _integrated_candidates().to_csv(integrated_path, index=False)

    return {"feature_matrix": matrix_path, "integrated_candidates": integrated_path}


def test_run_minimal_panel_produces_expected_outputs(input_paths: dict[str, Path], tmp_path: Path) -> None:
    outputs = run_minimal_panel(
        feature_matrix_path=input_paths["feature_matrix"],
        integrated_candidates_path=input_paths["integrated_candidates"],
        output_dir=tmp_path,
        cfg={
            "analysis": {
                "minimal_panel": {
                    "enabled": True,
                    "max_features": 5,
                    "min_features": 2,
                    "n_outer_splits": 3,
                    "n_inner_splits": 2,
                    "n_permutations": 10,
                    "n_bootstrap": 20,
                    "n_jobs": 1,
                    "detect_gpu": False,
                    "random_state": 42,
                    "continuous_targets": ["delta_moca"],
                    "binary_targets": [
                        {"group_variable": "delta_moca_group", "ordered_groups": ["decline", "stable_or_improved"]}
                    ],
                }
            },
            "whole_spectra": {
                "bands": {
                    "amide_iii_1260": [1235.0, 1255.0],
                    "ch2_sym_stretch_2850": [2840.0, 2855.0],
                    "ch3_sym_stretch_2930": [2925.0, 2935.0],
                },
                "ratio_pairs": [
                    ["amide_iii_1260", "ch2_sym_stretch_2850"],
                ],
                "biochemical_classes": {
                    "protein": ["amide_iii_1260"],
                    "lipid": ["ch2_sym_stretch_2850", "ch3_sym_stretch_2930"],
                },
            },
        },
    )

    assert "minimal_candidate_panel" in outputs
    assert "minimal_panel_cv_performance" in outputs
    assert "minimal_panel_coefficients" in outputs

    panel = pd.read_csv(outputs["minimal_candidate_panel"])
    perf = pd.read_csv(outputs["minimal_panel_cv_performance"])
    coef = pd.read_csv(outputs["minimal_panel_coefficients"])

    assert not panel.empty
    assert "panel_selected" in panel.columns
    assert panel["panel_selected"].any()
    for endpoint, group in panel.groupby("endpoint"):
        selected_count = int(group["panel_selected"].sum())
        assert selected_count <= 5
        assert selected_count >= 2

    assert not perf.empty
    assert "permutation_p_value" in perf.columns

    assert not coef.empty
    assert "ci_low" in coef.columns
    assert "ci_high" in coef.columns


def test_run_minimal_panel_accepts_preregistered_panel_source(input_paths: dict[str, Path], tmp_path: Path) -> None:
    panel_matrix = _feature_matrix()[["subject_id"]].copy()
    panel_matrix["oxidative_index"] = _feature_matrix()["delta_moca"] + np.linspace(0, 1, len(panel_matrix))
    panel_matrix["membrane_integrity_index"] = -_feature_matrix()["delta_moca"]
    panel_matrix["protein_remodeling_index"] = np.linspace(0, 1, len(panel_matrix))
    panel_path = tmp_path / "panel_feature_matrix.csv"
    panel_matrix.to_csv(panel_path, index=False)

    outputs = run_minimal_panel(
        feature_matrix_path=input_paths["feature_matrix"],
        panel_matrix_path=panel_path,
        output_dir=tmp_path / "confirmatory",
        cfg={
            "analysis": {
                "minimal_panel": {
                    "enabled": True,
                    "feature_source": "preregistered_panel",
                    "max_features": 3,
                    "min_features": 2,
                    "n_outer_splits": 3,
                    "n_inner_splits": 2,
                    "n_permutations": 10,
                    "n_bootstrap": 10,
                    "n_jobs": 1,
                    "detect_gpu": False,
                    "random_state": 42,
                    "continuous_targets": ["delta_moca"],
                    "binary_targets": [],
                },
                "preregistered_panel": {
                    "indices": {
                        "oxidative_index": ["carotenoid_1525", "lipid_carbonyl_1720"],
                        "membrane_integrity_index": ["ch_bulk_2885", "lipid_carbonyl_1720"],
                        "protein_remodeling_index": ["protein_1555", "structural_protein_925", "structural_protein_980"],
                    },
                    "integrals": {},
                },
            }
        },
    )

    assert "panel_minimal_model_performance" in outputs
    assert "panel_minimal_model_coefficients" in outputs
    assert outputs["panel_minimal_model_performance"].name == "panel_minimal_model_performance.csv"
    features = pd.read_csv(outputs["panel_minimal_model_features"])
    assert set(features["feature_source"]) <= {"preregistered_index"}


def test_run_minimal_panel_disabled_returns_empty(tmp_path: Path) -> None:
    outputs = run_minimal_panel(
        output_dir=tmp_path,
        cfg={"analysis": {"minimal_panel": {"enabled": False}}},
    )
    assert outputs == {}
