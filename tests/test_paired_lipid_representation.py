"""Tests for reproducible paired Raman representation learning and lipid evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src.analysis.paired_lipid_representation import (
    CovariatePreprocessor,
    PairedLipidConfig,
    PairedMoments,
    WeightedPLSResult,
    WeightedSpectralPreprocessor,
    aggregate_samples_to_participants,
    build_paired_cohort_dataset,
    compute_discriminability_and_retrieval,
    compute_endpoint_summary,
    compute_icc_a1,
    compute_subspace_principal_angles,
    evaluate_bootstrap_calibration,
    evaluate_nested_cohort,
    extract_sample_band_matrix,
    fit_ridge_grid,
    generate_synthetic_cohort,
    load_clinical_master_verified,
    map_subspace_to_metric_orthonormal_basis,
    parse_sample_label,
    predict_ridge_grid,
    run_multiplicity_bootstrap,
    summarize_anchor_associations,
    wilson_score_interval,
)


def test_parse_sample_label_memoized_and_deterministic() -> None:
    parse_sample_label.cache_clear()
    assert parse_sample_label("sample_42") == 42.0
    assert parse_sample_label("66'") == 66.0
    assert parse_sample_label("109\u2019") == 109.0
    assert np.isnan(parse_sample_label(None))
    assert np.isnan(parse_sample_label("no_digits_here"))
    info = parse_sample_label.cache_info()
    assert info.currsize >= 4


def test_load_clinical_master_verified_on_repo_master() -> None:
    master_path = Path("data/clinical_master.csv")
    if not master_path.exists():
        pytest.skip("data/clinical_master.csv not present in workspace")
    frame, audit = load_clinical_master_verified(master_path)
    assert audit["stroke_with_spectra"] == 97
    assert audit["stroke_identity_verified"] == 92
    assert audit["control_with_spectra"] == 50
    stroke_verified = frame.loc[(frame["cohort"] == "stroke") & frame["identity_verified"]]
    assert len(stroke_verified) == 92
    assert stroke_verified["ldl_mmol_l"].notna().sum() == 92


def test_hierarchy_exclusions_and_subset_modes() -> None:
    dataset_all, _ = generate_synthetic_cohort(
        n_participants=24,
        n_single_sample=3,
        maps_per_sample=3,
        seed=11,
        compact_grid=True,
    )
    assert dataset_all.n_participants == 24
    assert dataset_all.manifest_summary["n_paired_participants"] == 21
    assert dataset_all.manifest_summary["n_single_sample_participants"] == 3

    # Verify sample-to-participant aggregation equals manual mean per participant
    sample_bands, band_names = extract_sample_band_matrix(dataset_all)
    assert len(band_names) == 21
    # Calling a second time hits the dataset cache
    sample_bands_cached, _ = extract_sample_band_matrix(dataset_all)
    assert sample_bands_cached is sample_bands

    part_avg = aggregate_samples_to_participants(sample_bands, dataset_all, single_sample_only=False)
    part_single = aggregate_samples_to_participants(sample_bands, dataset_all, single_sample_only=True)
    assert part_avg.shape == (24, 21)
    assert part_single.shape == (24, 21)
    # For single-sample participants, average and single-sample selection must be identical
    single_p_idx = np.flatnonzero(dataset_all.participant_sample_counts == 1)
    np.testing.assert_allclose(part_avg[single_p_idx], part_single[single_p_idx])


def test_multi_alpha_ridge_matches_sklearn_exactly() -> None:
    rng = np.random.default_rng(7)
    n, p_cov, p_spec = 40, 4, 3
    X_cov = rng.normal(size=(n, p_cov))
    X_cov = (X_cov - X_cov.mean(axis=0)) / X_cov.std(axis=0)
    Z_spec = rng.normal(loc=2.0, scale=3.0, size=(n, p_spec))
    y = 1.5 + X_cov @ np.array([0.4, -0.2, 0.1, 0.3]) + Z_spec @ np.array([0.5, -0.3, 0.2]) + rng.normal(0, 0.2, n)

    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    weights = np.ones(n, dtype=float)
    fit = fit_ridge_grid(X_cov, Z_spec, y, weights, alphas)
    preds_all, spec_all = predict_ridge_grid(fit, X_cov, Z_spec)

    Z_std = (Z_spec - Z_spec.mean(axis=0)) / Z_spec.std(axis=0)
    X_full = np.hstack([X_cov, Z_std])
    for idx, alpha in enumerate(alphas):
        # Objective is (1/n) ||y - X b - b0||^2 + alpha ||b||^2 -> sklearn alpha = n * alpha
        sk = Ridge(alpha=n * alpha, fit_intercept=True, solver="svd")
        sk.fit(X_full, y)
        np.testing.assert_allclose(preds_all[:, idx], sk.predict(X_full), rtol=1e-10, atol=1e-10)
        pred_single, spec_single = predict_ridge_grid(fit, X_cov, Z_spec, alpha_idx=idx)
        np.testing.assert_allclose(pred_single, preds_all[:, idx], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(spec_single, spec_all[:, idx], rtol=1e-12, atol=1e-12)


def test_nested_pca_slicing_and_paired_eigensolver_properties() -> None:
    dataset, _ = generate_synthetic_cohort(
        n_participants=30,
        n_single_sample=2,
        seed=19,
        compact_grid=True,
    )
    train_idx = np.arange(24, dtype=int)
    mult = np.ones(24, dtype=float)
    cfg = PairedLipidConfig()

    spec_prep = WeightedSpectralPreprocessor.fit(dataset, train_idx, mult, cfg=cfg)
    coords_max = spec_prep.transform_pca(dataset.sample_1300, dataset.sample_2850)
    coords_5 = spec_prep.transform_pca(dataset.sample_1300, dataset.sample_2850, k=5)
    np.testing.assert_allclose(coords_max[:, :5], coords_5, rtol=1e-12, atol=1e-12)

    moments_max = PairedMoments.from_sample_coordinates(coords_max, dataset, train_idx, mult)
    moments_5 = PairedMoments.from_sample_coordinates(coords_5, dataset, train_idx, mult)
    np.testing.assert_allclose(moments_max.W_max[:5, :5], moments_5.W_max, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(moments_max.T_max[:5, :5], moments_5.T_max, rtol=1e-12, atol=1e-12)

    V, eigvals, tau = moments_max.solve_directions(10, lam=0.1)
    assert tau > 0
    assert np.all(np.diff(eigvals) <= 1e-10)
    denom = moments_max.W_max[:10, :10] + 0.1 * tau * np.eye(10)
    metric_gram = V.T @ denom @ V
    np.testing.assert_allclose(metric_gram, np.eye(10), rtol=1e-10, atol=1e-10)

    # Check subspace metric orthonormalization and zero self-angles
    Q = map_subspace_to_metric_orthonormal_basis(
        spec_prep, 10, V[:, :3], dataset.wavenumbers_1300.size, dataset.wavenumbers_2850.size
    )
    np.testing.assert_allclose(Q.T @ Q, np.eye(3), rtol=1e-10, atol=1e-10)
    angles_df = compute_subspace_principal_angles([Q, Q])
    assert len(angles_df) == 1
    assert angles_df["max_principal_angle_deg"].iloc[0] == pytest.approx(0.0, abs=1e-6)


def test_icc_a1_and_discriminability_properties() -> None:
    rng = np.random.default_rng(123)
    # High agreement case
    true_sig = rng.normal(0.0, 2.0, size=25)
    y_a = true_sig + rng.normal(0.0, 0.1, size=25)
    y_b = true_sig + rng.normal(0.0, 0.1, size=25)
    icc_high = compute_icc_a1(y_a, y_b)
    assert icc_high["estimable"] is True
    assert icc_high["icc_a1"] > 0.95

    # Systematic position shift reduces absolute agreement ICC(A,1)
    icc_shifted = compute_icc_a1(y_a, y_b + 3.0)
    assert icc_shifted["icc_a1"] < icc_high["icc_a1"]

    # Negative ICC is retained (not clipped to zero) when within-pair disagreement exceeds between-subject variance
    v = np.linspace(-1.0, 1.0, 10)
    icc_neg = compute_icc_a1(v, -v)
    assert icc_neg["estimable"] is True
    assert icc_neg["icc_a1"] < 0.0

    # Discriminability and retrieval on well-separated pairs
    z_a = rng.normal(0.0, 3.0, size=(15, 3))
    z_b = z_a + rng.normal(0.0, 0.05, size=(15, 3))
    disc = compute_discriminability_and_retrieval(
        z_a, z_b, np.arange(15, dtype=float), np.arange(15, dtype=float) + 0.5
    )
    assert disc["discriminability"] > 0.98
    assert disc["retrieval_top1"] > 0.95
    assert disc["retrieval_chance"] == pytest.approx(1.0 / 15.0)


def test_nested_evaluation_recovers_signal_and_detects_nonlinear_proxy() -> None:
    # 1. Genuine reproducible biological signal improves held-out LDL-C prediction over covariates
    ds_sig, _ = generate_synthetic_cohort(
        n_participants=45,
        n_single_sample=3,
        seed=101,
        compact_grid=True,
        signal_strength=1.1,
    )
    res_sig = evaluate_nested_cohort(ds_sig, include_band_paired=True)
    delta_paired = res_sig["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"]
    assert delta_paired > 0.15
    assert "band_paired_ridge" in res_sig["comparisons_vs_covariate_ridge"]
    assert len(res_sig["fold_settings"]) == 5
    assert not res_sig["subspace_angles"].empty

    # Anchor association summary remains continuous and descriptive
    anchor_df = summarize_anchor_associations(ds_sig, res_sig, n_resamples=50)
    assert set(anchor_df["anchor"]) == {"ldl_mmol_l", "hdl_mmol_l", "triglycerides_mmol_l", "plt_10e9_l"}
    assert (anchor_df["attribution_status"] == "continuous_exploratory_summary_only").all()

    # 2. Failure mode: spectra proxy nonlinear age^2 without independent biological effect on y
    ds_proxy, _ = generate_synthetic_cohort(
        n_participants=50,
        n_single_sample=2,
        seed=202,
        compact_grid=True,
        signal_strength=0.0,
        nonlinear_covariate_proxy=True,
    )
    res_linear_c = evaluate_nested_cohort(ds_proxy, cfg=PairedLipidConfig(covariate_preset="primary"))
    res_quad_c = evaluate_nested_cohort(
        ds_proxy, cfg=PairedLipidConfig(covariate_preset="quadratic_age_and_label")
    )
    delta_linear = res_linear_c["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"]
    delta_quad = res_quad_c["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"]
    # Adding quadratic age terms to the covariate model removes the artificial proxy advantage
    assert delta_linear > delta_quad


def test_multiplicity_bootstrap_and_calibration_gate() -> None:
    ds, _ = generate_synthetic_cohort(
        n_participants=30,
        n_single_sample=2,
        seed=303,
        compact_grid=True,
        signal_strength=0.9,
    )
    boot = run_multiplicity_bootstrap(ds, n_bootstraps=12, bootstrap_seed=99)
    assert boot["interval_procedure_succeeded"] is True
    assert boot["n_bootstraps_succeeded"] == 12
    assert boot["n_failed_refits"] == 0
    lo, hi = boot["ci_95_delta_mse"]
    assert np.isfinite(lo) and np.isfinite(hi) and lo < hi

    cal = evaluate_bootstrap_calibration(
        n_repetitions=4,
        n_bootstraps=8,
        n_participants=30,
        signal_strength=0.9,
        base_seed=500,
    )
    assert cal["dataset_failure_rate"] == 0.0
    assert 0.0 <= cal["unconditional_coverage"] <= 1.0
    w_lo, w_hi = wilson_score_interval(960, 1000)
    assert w_lo > 0.925 and w_hi < 1.0


def test_manuscript_ml_scripts_and_approval_gate(tmp_path: Path) -> None:
    from manuscript_ml import resolve_experiment_dataset
    from manuscript_ml.run_all import run_all_pipeline

    with pytest.raises(PermissionError, match="approve-cohort-evaluation"):
        resolve_experiment_dataset(use_synthetic=False, approve_cohort_evaluation=False)

    out_dir = tmp_path / "manuscript_outputs"
    outputs = run_all_pipeline(
        use_synthetic=True,
        quick=True,
        output_dir=out_dir,
        n_bootstraps=4,
    )
    expected_keys = {
        "readiness_manifest",
        "limitations_table",
        "simulation_scenarios_summary",
        "simulation_calibration_report",
        "eligibility_and_split_manifest",
        "nested_held_out_predictions",
        "comparator_errors_and_differences",
        "fold_selected_settings",
        "baseline_repeatability_report",
        "subspace_principal_angles",
        "covariate_and_subset_sensitivity",
        "outer_partition_seed_sensitivity",
        "secondary_endpoints_summary",
        "anchor_associations_continuous",
    }
    assert expected_keys.issubset(set(outputs.keys()))
    for path in outputs.values():
        assert path.is_file()


