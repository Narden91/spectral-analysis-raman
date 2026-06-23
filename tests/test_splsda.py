"""Tests for the Phase 9 PLS-L1DA Exploratory Classifier."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.analysis.splsda import run_splsda_classifier


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
    
    assert set(outputs) == {
        "pls_l1da_cv_performance",
        "pls_l1da_selected_bands",
        "pls_l1da_permutation_results",
    }
    bands_df = pd.read_csv(outputs["pls_l1da_selected_bands"])
    assert bands_df.empty
    assert "status" in bands_df.columns


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
    assert len(perf_df) == 1
    assert perf_df["group_variable"].iloc[0] == "moca_severity_group"


def test_splsda_skips_when_disabled(tmp_path):
    cfg = {"analysis": {"splsda": {"enabled": False}}}
    outputs = run_splsda_classifier(output_dir=tmp_path, cfg=cfg)
    assert outputs == {}
