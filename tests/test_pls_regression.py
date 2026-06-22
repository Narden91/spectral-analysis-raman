"""Tests for the Phase 8 PLS1 Regression module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.analysis.pls_regression import run_pls_regression


@pytest.fixture
def mock_pls_cfg():
    return {
        "analysis": {
            "pls_regression": {
                "enabled": True,
                "targets": ["moca_baseline", "delta_moca"],
                "component_grid": [1, 2, 3],
                "tune_components_inside_cv": True,
                "n_cv_splits": 5,
                "min_subjects": 5,
                "vip_threshold": 1.0,
                "n_permutations": 10,
                "random_state": 42,
            }
        }
    }


@pytest.fixture
def synthetic_feature_matrix(tmp_path):
    n_subjects = 20
    np.random.seed(42)
    
    data = {
        "subject_id": [f"S{i:02d}" for i in range(n_subjects)],
        "moca_baseline": np.random.normal(25, 3, n_subjects),
        "delta_moca": np.random.normal(-1, 2, n_subjects),
    }
    
    # 5 features
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


@patch("src.analysis.pls_regression.load_disease_feature_frame")
def test_pls_regression_produces_expected_outputs(mock_load, synthetic_feature_matrix, mock_pls_cfg, tmp_path):
    df, meta_df = synthetic_feature_matrix
    mock_load.return_value = (df, meta_df)
    
    out_dir = tmp_path / "pls_out"
    outputs = run_pls_regression(
        feature_matrix_path="dummy.parquet",
        output_dir=out_dir,
        cfg=mock_pls_cfg,
    )
    
    assert len(outputs) == 4
    for key in ["pls_regression_performance", "pls_vip_bands", "pls_coefficients", "pls_permutation_results"]:
        assert key in outputs
        assert outputs[key].exists()
        
    perf_df = pd.read_csv(outputs["pls_regression_performance"])
    assert len(perf_df) == 2
    assert "target" in perf_df.columns
    assert "high_variance_flag" in perf_df.columns
    assert "selected_n_components" in perf_df.columns
    assert perf_df["high_variance_flag"].iloc[0] == False
    assert perf_df["cv_strategy"].iloc[0] == "KFold"

    vip_df = pd.read_csv(outputs["pls_vip_bands"])
    assert len(vip_df) == 10  # 2 targets * 5 features
    assert "vip_score" in vip_df.columns


@patch("src.analysis.pls_regression.load_disease_feature_frame")
def test_pls_regression_loo_fallback(mock_load, synthetic_feature_matrix, mock_pls_cfg, tmp_path):
    df, meta_df = synthetic_feature_matrix
    df_small = df.iloc[:12].copy()  # < 15 subjects
    mock_load.return_value = (df_small, meta_df)
    
    out_dir = tmp_path / "pls_out"
    outputs = run_pls_regression(
        feature_matrix_path="dummy.parquet",
        output_dir=out_dir,
        cfg=mock_pls_cfg,
    )
    
    perf_df = pd.read_csv(outputs["pls_regression_performance"])
    assert perf_df["high_variance_flag"].iloc[0] == True
    assert perf_df["cv_strategy"].iloc[0] == "LeaveOneOut"


@patch("src.analysis.pls_regression.load_disease_feature_frame")
def test_pls_regression_permutation_deterministic(mock_load, synthetic_feature_matrix, mock_pls_cfg, tmp_path):
    df, meta_df = synthetic_feature_matrix
    mock_load.return_value = (df, meta_df)
    
    out_dir1 = tmp_path / "pls_out1"
    outputs1 = run_pls_regression(output_dir=out_dir1, cfg=mock_pls_cfg)
    
    out_dir2 = tmp_path / "pls_out2"
    outputs2 = run_pls_regression(output_dir=out_dir2, cfg=mock_pls_cfg)
    
    perm1 = pd.read_csv(outputs1["pls_permutation_results"])
    perm2 = pd.read_csv(outputs2["pls_permutation_results"])
    
    pd.testing.assert_frame_equal(perm1, perm2)


def test_pls_regression_skips_when_disabled(tmp_path):
    cfg = {"analysis": {"pls_regression": {"enabled": False}}}
    outputs = run_pls_regression(output_dir=tmp_path, cfg=cfg)
    assert outputs == {}
