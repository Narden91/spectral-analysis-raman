"""Tests for reproducible paired Raman representation learning and lipid evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src.analysis.paired_lipid.cohort import (
    PairedLipidConfig,
    aggregate_samples_to_participants,
    build_paired_cohort_dataset,
    extract_sample_band_matrix,
    load_clinical_master_verified,
    parse_sample_label,
)
from src.analysis.paired_lipid.models import (
    PairedMoments,
    WeightedSpectralPreprocessor,
    fit_ridge_grid,
    predict_ridge_grid,
)
from src.analysis.paired_lipid.nested_cv import (
    compute_discriminability_and_retrieval,
    compute_icc_a1,
    compute_subspace_principal_angles,
    evaluate_nested_cohort,
    map_subspace_to_metric_orthonormal_basis,
    run_multiplicity_bootstrap,
    summarize_anchor_associations,
)
from src.analysis.paired_lipid.simulation import (
    check_gate_attainable,
    evaluate_bootstrap_calibration,
    generate_synthetic_cohort,
    min_reps_for_gate,
    wilson_score_interval,
)


def test_calibration_gate_minimum_repetitions() -> None:
    # 100% coverage gives Wilson lower n/(n + z^2): 48 is the first n reaching 0.925.
    assert min_reps_for_gate(0.925) == 48
    assert wilson_score_interval(48, 48)[0] >= 0.925 > wilson_score_interval(47, 47)[0]
    assert check_gate_attainable(48, "validation", 0.925) == (True, 48)
    assert check_gate_attainable(20, "development", 0.925) == (False, 48)
    with pytest.raises(ValueError, match=">= 48 repetitions"):
        check_gate_attainable(47, "validation", 0.925)


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


def test_clinical_ids_link_as_strings_and_exclusions_name_the_cause(tmp_path: Path) -> None:
    # A "not_available" id once made pandas read ids as floats ("7.0"), so no participant linked.
    master = tmp_path / "master.csv"
    master.write_text(
        "participant_key,cohort,spectral_subject_id,has_spectra,spectral_link_status,qc_flags,ldl_mmol_l\n"
        "stroke_7,stroke,7,1,matched,none,3.1\n"
        "stroke_8,stroke,8,1,matched,SEX_FOLDER_CONFLICT,2.9\n"
        "stroke_11,stroke,not_available,0,unmatched,none,2.5\n",
        encoding="utf-8",
    )
    frame, _ = load_clinical_master_verified(master)
    assert frame.loc[0, "spectral_subject_id"] == "7"

    rng = np.random.default_rng(0)
    maps = pd.DataFrame(
        [
            {"participant_id": "7", "sample_id": sid, "spectral_range": rng_code, "row_index": 3 * s + m}
            for s, sid in enumerate(("12", "13"))
            for m in range(3)
            for rng_code in ("1300", "2850")
        ]
    )
    ds = build_paired_cohort_dataset(
        frame, maps, rng.normal(size=(6, 4)), rng.normal(size=(6, 3)), np.arange(4.0), np.arange(3.0)
    )
    assert ds.participant_ids.tolist() == ["7"]
    assert ds.manifest_summary["n_paired_participants"] == 1
    reasons = {e["participant_id"]: e["reason"] for e in ds.exclusions}
    assert reasons == {"8": "identity_review_required", "stroke_11": "no_spectra"}


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

    # n=30 draws can leave an inner division too few paired participants for k=5, which fails the
    # interval by design; n=40 keeps this check about calibration bookkeeping, not small-n failure.
    cal = evaluate_bootstrap_calibration(
        n_repetitions=4,
        n_bootstraps=8,
        n_participants=40,
        signal_strength=0.9,
        base_seed=500,
    )
    assert cal["dataset_failure_rate"] == 0.0
    assert 0.0 <= cal["unconditional_coverage"] <= 1.0
    w_lo, w_hi = wilson_score_interval(960, 1000)
    assert w_lo > 0.925 and w_hi < 1.0


def _gpu_available() -> bool:
    from src.analysis.paired_lipid.batched import _get_gpu_xp

    return _get_gpu_xp() is not None


@pytest.mark.parametrize(
    "use_gpu",
    [False, pytest.param(True, marks=pytest.mark.skipif(not _gpu_available(), reason="CuPy/CUDA unavailable"))],
)
def test_batched_bootstrap_draws_match_sequential_refits(use_gpu: bool) -> None:
    # The batched engine re-derives ridge, PLS and the paired eigensolver in closed form;
    # any drift from the sequential reference silently changes every bootstrap interval.
    from src.analysis.paired_lipid.batched import (
        _check_fast_batch_eligibility,
        _get_dataset_gram_cache,
        _run_batched_multiplicity_draws,
    )
    from src.analysis.paired_lipid.nested_cv import make_nested_splits

    # n=70 keeps resampled divisions above the k<=15 cap, so every draw stays on the batched path.
    ds, _ = generate_synthetic_cohort(n_participants=70, n_single_sample=2, seed=404, compact_grid=True)
    cfg = PairedLipidConfig()
    n_p = ds.n_participants
    outer, inner = make_nested_splits(ds, cfg, cfg.primary_seed)
    rng = np.random.default_rng(5)
    M = np.vstack(
        [np.ones(n_p)] + [np.bincount(rng.integers(0, n_p, n_p), minlength=n_p).astype(float) for _ in range(3)]
    )
    gcache = _get_dataset_gram_cache(ds, cfg)
    assert _check_fast_batch_eligibility(ds, outer, inner, M, cfg, gcache).all()

    tol = 1e-8
    d_pair, d_comp = _run_batched_multiplicity_draws(ds, ds.outcomes[cfg.primary_endpoint], outer, inner, M, cfg, gcache, use_gpu=use_gpu)
    for b in range(M.shape[0]):
        # include_band_paired disables the batched fast inner path, so the reference is fully sequential.
        ref = evaluate_nested_cohort(
            ds, cfg=cfg, participant_multiplicities=M[b], precomputed_splits=(outer, inner), include_band_paired=True
        )
        assert d_pair[b] == pytest.approx(ref["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"], abs=tol)
        assert d_comp[b] == pytest.approx(ref["paired_vs_strongest_comparator"]["delta_mse"], abs=tol)


def test_fast_inner_selection_matches_sequential_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    # Primary tables use the sequential path (include_band_paired); sensitivity and anchor tables use
    # the fast inner path. Both must select the same settings and give the same held-out predictions.
    import src.analysis.paired_lipid.nested_cv as nested_cv_mod
    from src.analysis.paired_lipid.batched import _check_fast_batch_eligibility, _get_dataset_gram_cache

    def _fail_if_sequential(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("default evaluation fell back to sequential inner selection")

    ds, _ = generate_synthetic_cohort(n_participants=70, n_single_sample=2, seed=606, compact_grid=True)
    cfg = PairedLipidConfig()
    rng = np.random.default_rng(6)
    M = np.vstack(
        [np.ones(ds.n_participants)]
        + [np.bincount(rng.integers(0, ds.n_participants, ds.n_participants), minlength=ds.n_participants) for _ in range(2)]
    ).astype(float)
    splits = evaluate_nested_cohort(ds, cfg=cfg)["splits"]
    assert _check_fast_batch_eligibility(ds, *splits, M, cfg, _get_dataset_gram_cache(ds, cfg)).all()

    for mult in M:
        with monkeypatch.context() as mp:
            mp.setattr(nested_cv_mod, "_select_inner_sequential", _fail_if_sequential)
            fast = evaluate_nested_cohort(ds, cfg=cfg, participant_multiplicities=mult, precomputed_splits=splits)
        seq = evaluate_nested_cohort(
            ds, cfg=cfg, participant_multiplicities=mult, precomputed_splits=splits, include_band_paired=True
        )
        settings = [c for c in fast["fold_settings"].columns if c != "dropped_covariates"]
        pd.testing.assert_frame_equal(fast["fold_settings"][settings], seq["fold_settings"][settings], rtol=1e-8)
        for model, preds in fast["oof_predictions"].items():
            np.testing.assert_allclose(preds, seq["oof_predictions"][model], rtol=0, atol=1e-8)


def test_missing_secondary_endpoint_equals_zero_weight() -> None:
    # A missing endpoint value must act exactly like weight 0: its placeholder never enters a fit or error.
    from dataclasses import replace

    ds, _ = generate_synthetic_cohort(n_participants=40, n_single_sample=2, seed=505, compact_grid=True)
    missing = np.array([3, 11, 27])
    y = ds.outcomes["hdl_mmol_l"].copy()
    y_nan, y_far = y.copy(), y.copy()
    y_nan[missing] = np.nan
    y_far[missing] = 1e3
    ds_nan = replace(ds, outcomes={**ds.outcomes, "hdl_mmol_l": y_nan})
    ds_far = replace(ds, outcomes={**ds.outcomes, "hdl_mmol_l": y_far})
    mult = np.ones(ds.n_participants)
    mult[missing] = 0.0

    res_nan = evaluate_nested_cohort(ds_nan, endpoint="hdl_mmol_l", allow_missing_outcome=True)
    res_far = evaluate_nested_cohort(ds_far, endpoint="hdl_mmol_l", participant_multiplicities=mult)
    assert res_nan["n_missing_outcome"] == 3
    for model in ("covariate_ridge", "paired_linear_ridge", "strongest_comparator"):
        a = res_nan["comparisons_vs_covariate_ridge"][model]["delta_mse"]
        b = res_far["comparisons_vs_covariate_ridge"][model]["delta_mse"]
        assert a == pytest.approx(b, abs=1e-10)
    with pytest.raises(ValueError, match="missing values"):
        evaluate_nested_cohort(ds_nan, endpoint="hdl_mmol_l")


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


