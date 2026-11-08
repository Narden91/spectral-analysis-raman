"""Tests for the Phase 13 minimal candidate Raman panel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.minimal_panel import _permutation_test_binary, _permutation_test_continuous, run_minimal_panel


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
            "evidence_status": [
                "cluster_supported_exploratory",
                "cluster_supported_exploratory",
            ],
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
                    "amide_iii_1258": [1235.0, 1255.0],
                    "ch2_sym_stretch_2853": [2840.0, 2855.0],
                    "ch3_asym_stretch_2937": [2925.0, 2935.0],
                },
                "ratio_pairs": [
                    ["amide_iii_1258", "ch2_sym_stretch_2853"],
                ],
                "biochemical_classes": {
                    "protein": ["amide_iii_1258"],
                    "lipid": ["ch2_sym_stretch_2853", "ch3_asym_stretch_2937"],
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
    assert "stability_interval_low" in coef.columns
    assert "stability_interval_high" in coef.columns
    assert "bootstrap_selection_frequency" in coef.columns
    assert not coef["validated"].any()


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
                        "oxidative_index": ["carotenoid_1525", "lipid_carbonyl_1721"],
                        "membrane_integrity_index": ["ch_bulk_2885", "lipid_carbonyl_1721"],
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


def test_candidate_panel_skips_when_cluster_support_gate_fails(
    input_paths: dict[str, Path],
    tmp_path: Path,
) -> None:
    integrated = _integrated_candidates()
    integrated["evidence_status"] = "effect_only_fallback"
    integrated.to_csv(input_paths["integrated_candidates"], index=False)

    outputs = run_minimal_panel(
        feature_matrix_path=input_paths["feature_matrix"],
        integrated_candidates_path=input_paths["integrated_candidates"],
        output_dir=tmp_path / "gated",
        cfg={
            "analysis": {"minimal_panel": {"enabled": True}},
            "whole_spectra": {
                "bands": {"amide_iii_1258": [1235.0, 1255.0]},
                "ratio_pairs": [],
            },
        },
    )

    # The whole-phase input pool is empty (no cluster-supported region), so no
    # CSVs are written at all; the phase reports a status dict instead.
    assert set(outputs) == {"_status"}
    assert outputs["_status"]["status"] == "not_tested_gate"
    assert outputs["_status"]["phase_key"] == "minimal_panel"
    assert not (tmp_path / "gated" / "minimal_candidate_panel.csv").exists()
    assert not (tmp_path / "gated" / "minimal_panel_cv_performance.csv").exists()


def test_run_minimal_panel_disabled_returns_empty(tmp_path: Path) -> None:
    outputs = run_minimal_panel(
        output_dir=tmp_path,
        cfg={"analysis": {"minimal_panel": {"enabled": False}}},
    )
    assert outputs == {}


def _panel_permutation_inputs(rng: np.random.Generator, n: int = 24, n_features: int = 4) -> tuple[np.ndarray, list[str]]:
    X = rng.normal(size=(n, n_features))
    feature_names = [f"feat_{i}" for i in range(n_features)]
    return X, feature_names


def test_permutation_test_continuous_respects_permutation_max() -> None:
    """Each draw here refits a nested-CV panel selection, which is expensive;
    the stopping rule must never exceed the configured ceiling regardless of
    whether the permutation test happens to resolve early on this small sample.
    """
    rng = np.random.default_rng(31)
    X, feature_names = _panel_permutation_inputs(rng)
    y = rng.normal(size=X.shape[0])  # pure noise target: R2 reference distribution centers near 0

    p_value, _mean, _std, n_used = _permutation_test_continuous(
        X,
        y,
        feature_names,
        n_outer_splits=3,
        n_inner_splits=2,
        n_permutations=30,
        random_state=42,
        observed_r2=0.0,
        n_jobs=1,
        max_features=3,
        min_features=1,
        panel_cfg={"permutation_min": 10, "permutation_check_interval": 10},
    )
    assert n_used <= 30
    assert 0.0 <= p_value <= 1.0


def test_permutation_test_binary_respects_permutation_max() -> None:
    rng = np.random.default_rng(32)
    X, feature_names = _panel_permutation_inputs(rng)
    y = np.array([0] * 12 + [1] * 12)

    p_value, _mean, _std, n_used = _permutation_test_binary(
        X,
        y,
        feature_names,
        n_outer_splits=3,
        n_inner_splits=2,
        n_permutations=30,
        random_state=42,
        observed_balanced_accuracy=0.5,
        n_jobs=1,
        max_features=3,
        min_features=1,
        panel_cfg={"permutation_min": 10, "permutation_check_interval": 10},
    )
    assert n_used <= 30
    assert 0.0 <= p_value <= 1.0
