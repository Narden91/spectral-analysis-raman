"""Tests for the Phase 9 PLS-L1DA Exploratory Classifier."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from src.analysis.splsda import _permutation_test, run_splsda_classifier


def test_splsda_permutation_test_rejects_all_undefined_scores() -> None:
    with patch("src.analysis.splsda._permuted_splsda_score", return_value=np.nan):
        with pytest.raises(ValueError, match="Permutation scores must be finite"):
            _permutation_test(
                np.ones((4, 2)), np.array([0, 0, 1, 1]), None, None,
                [1], [1.0], 2, 0.75, 200, 42, n_jobs=1,
            )


@pytest.fixture
def mock_splsda_cfg():
    return {
        "analysis": {
            "splsda": {
                "enabled": True,
                "gate_on_pls_p_value": 0.05,
                "targets": [
                    {"group_variable": "moca_severity_group", "ordered_groups": ["low_moca", "high_moca"]},
                    {"group_variable": "delta_moca_group", "ordered_groups": ["decline", "stable_or_improved"]}
                ],
                "component_grid": [1, 2],
                "c_range": [0.1, 1.0],
                "n_outer_splits": 3,
                "n_inner_splits": 2,
                "min_subjects_per_group": 3,
                "n_permutations": 5,
                "random_state": 42,
                "emit_oplsda": True,
            }
        }
    }


@pytest.fixture
def synthetic_feature_matrix():
    n_subjects = 20
    np.random.seed(42)

    data = {
        "subject_id": [f"S{i:02d}" for i in range(n_subjects)],
        "moca_severity_group": ["low_moca"] * 10 + ["high_moca"] * 10,
        "delta_moca_group": ["decline"] * 8 + ["stable_or_improved"] * 12,
    }

    for j in range(5):
        data[f"1300__{1000 + j * 10}.0"] = np.random.normal(0, 1, n_subjects)

    df = pd.DataFrame(data)

    meta_data = {
        "feature_name": [f"1300__{1000 + j * 10}.0" for j in range(5)],
        "spectral_block": ["1300"] * 5,
        "wavenumber": [1000.0 + j * 10.0 for j in range(5)],
    }
    meta_df = pd.DataFrame(meta_data)

    return df, meta_df


@pytest.fixture
def passing_pls_outputs(tmp_path):
    perm_df = pd.DataFrame({
        "target": ["moca_baseline"],
        "permutation_p_value": [0.01]  # Passes gate (< 0.05)
    })
    perm_path = tmp_path / "pls_perm.csv"
    perm_df.to_csv(perm_path, index=False)
    return {"pls_permutation_results": perm_path}


@pytest.fixture
def failing_pls_outputs(tmp_path):
    perm_df = pd.DataFrame({
        "target": ["moca_baseline"],
        "permutation_p_value": [0.1]  # Fails gate (>= 0.05)
    })
    perm_path = tmp_path / "pls_perm.csv"
    perm_df.to_csv(perm_path, index=False)
    return {"pls_permutation_results": perm_path}


@patch("src.analysis.splsda.load_disease_feature_frame")
def test_splsda_produces_expected_outputs(mock_load, synthetic_feature_matrix, mock_splsda_cfg, passing_pls_outputs, tmp_path):
    df, meta_df = synthetic_feature_matrix
    mock_load.return_value = (df, meta_df)

    out_dir = tmp_path / "splsda_out"
    outputs = run_splsda_classifier(
        feature_matrix_path="dummy.parquet",
        pls_outputs=passing_pls_outputs,
        output_dir=out_dir,
        cfg=mock_splsda_cfg,
    )

    assert len(outputs) == 4
    for key in ["pls_l1da_cv_performance", "pls_l1da_selected_bands", "pls_l1da_permutation_results", "oplsda_results"]:
        assert key in outputs
        assert outputs[key].exists()

    perf_df = pd.read_csv(outputs["pls_l1da_cv_performance"])
    assert len(perf_df) == 2
    assert "method_label" in perf_df.columns
    assert perf_df["method_label"].iloc[0] == "PLS-L1DA"
    assert "group_variable" in perf_df.columns
    assert set(perf_df["group_variable"]) == {"moca_severity_group", "delta_moca_group"}
    assert set(perf_df["inference_role"]) == {"conditional_exploratory"}
    bands_df = pd.read_csv(outputs["pls_l1da_selected_bands"])
    assert {"estimation_scope", "validated", "selection_frequency", "sign_stability"}.issubset(bands_df.columns)
    assert not bands_df["validated"].any()

    oplsda_df = pd.read_csv(outputs["oplsda_results"])
    assert len(oplsda_df) == 2


@patch("src.analysis.splsda.load_disease_feature_frame")
def test_splsda_gates_on_pls_p_value(mock_load, synthetic_feature_matrix, mock_splsda_cfg, failing_pls_outputs, tmp_path):
    df, meta_df = synthetic_feature_matrix
    mock_load.return_value = (df, meta_df)

    out_dir = tmp_path / "splsda_out"
    outputs = run_splsda_classifier(
        feature_matrix_path="dummy.parquet",
        pls_outputs=failing_pls_outputs,
        output_dir=out_dir,
        cfg=mock_splsda_cfg,
    )

    # bands_path carries zero rows in the gated case (no band is ever scored),
    # so it is no longer written at all; only the per-target status rows in
    # perf/perm remain.
    assert set(outputs) == {
        "pls_l1da_cv_performance",
        "pls_l1da_permutation_results",
        "_status",
    }
    assert outputs["_status"]["status"] == "not_tested_gate"
    assert outputs["_status"]["n_artifacts"] == 2
    perf_df = pd.read_csv(outputs["pls_l1da_cv_performance"])
    assert set(perf_df["status"]) == {"not_tested_gate"}


@patch("src.analysis.splsda.load_disease_feature_frame")
def test_splsda_skips_insufficient_class_size(mock_load, synthetic_feature_matrix, mock_splsda_cfg, passing_pls_outputs, tmp_path):
    df, meta_df = synthetic_feature_matrix
    # Make delta_moca_group have < 3 subjects in one class
    df.loc[:1, "delta_moca_group"] = "decline"
    df.loc[2:, "delta_moca_group"] = "stable_or_improved"  # Only 2 'decline'

    mock_load.return_value = (df, meta_df)

    out_dir = tmp_path / "splsda_out"
    outputs = run_splsda_classifier(
        feature_matrix_path="dummy.parquet",
        pls_outputs=passing_pls_outputs,
        output_dir=out_dir,
        cfg=mock_splsda_cfg,
    )

    assert "pls_l1da_cv_performance" in outputs
    perf_df = pd.read_csv(outputs["pls_l1da_cv_performance"])
    assert len(perf_df) == 2
    status = perf_df.set_index("group_variable")["status"]
    assert status["moca_severity_group"] == "ok"
    assert status["delta_moca_group"] == "insufficient_n"


def test_splsda_skips_when_disabled(tmp_path):
    cfg = {"analysis": {"splsda": {"enabled": False}}}
    outputs = run_splsda_classifier(output_dir=tmp_path, cfg=cfg)
    assert outputs == {}


def test_splsda_permutation_test_stops_early_on_unambiguous_null():
    rng = np.random.default_rng(21)
    X = rng.normal(size=(24, 4))
    y = np.array([0] * 12 + [1] * 12)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Balanced-accuracy 0.5 is chance level for a balanced binary target, so a
    # weak/noise classifier's permutation null should resolve as clearly not
    # significant well before the 2000-draw ceiling.
    p_value, n_used = _permutation_test(
        X,
        y,
        cv,
        None,
        [1, 2],
        [0.1, 1.0],
        2,
        0.5,
        2000,
        42,
        n_jobs=1,
        splsda_cfg={"permutation_min": 40, "permutation_check_interval": 20},
    )
    assert n_used < 2000
    assert 0.0 <= p_value <= 1.0


def test_splsda_permutation_test_returns_full_budget_when_never_resolved():
    rng = np.random.default_rng(22)
    X = rng.normal(size=(24, 4))
    y = np.array([0] * 12 + [1] * 12)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    _p_value, n_used = _permutation_test(
        X,
        y,
        cv,
        None,
        [1, 2],
        [0.1, 1.0],
        2,
        0.5,
        10,
        42,
        n_jobs=1,
        splsda_cfg={"permutation_min": 10, "permutation_check_interval": 10, "permutation_confidence": 0.999999},
    )
    assert n_used == 10
