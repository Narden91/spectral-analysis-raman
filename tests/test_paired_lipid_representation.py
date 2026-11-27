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


def test_manuscript_ml_scripts_and_approval_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import manuscript_ml.run_all as run_all_module
    import src.data_preparation.materialize as materialize
    from manuscript_ml import resolve_experiment_dataset
    from manuscript_ml.audit_readiness import run_readiness_audit
    from manuscript_ml.run_all import run_all_pipeline

    monkeypatch.setattr(materialize, "store_dir", lambda *_args: tmp_path / "missing_store")
    monkeypatch.setattr(
        run_all_module, "run_readiness_audit",
        lambda **kwargs: run_readiness_audit(clinical_master_path=tmp_path / "missing_clinical.csv", **kwargs),
    )
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
        if path.suffix == ".csv":
            assert not any(str(column).startswith("Unnamed:") for column in pd.read_csv(path).columns)


def test_bootstrap_counts_nonfinite_fallback_and_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    import src.analysis.paired_lipid.nested_cv as module

    dataset = SimpleNamespace(n_participants=2, outcomes={"ldl_mmol_l": np.array([1.0, 2.0])})
    monkeypatch.setattr(module, "make_nested_splits", lambda *_args: ([], []))
    monkeypatch.setattr(module, "_get_dataset_gram_cache", lambda *_args: {})
    monkeypatch.setattr(module, "_check_fast_batch_eligibility", lambda *_args: np.zeros(3, dtype=bool))
    calls = 0

    def evaluate(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ValueError("refit failed")
        value = 1.0 if calls == 1 else np.nan
        return {
            "comparisons_vs_covariate_ridge": {"paired_linear_ridge": {"delta_mse": value}},
            "paired_vs_strongest_comparator": {"delta_mse": value},
        }

    monkeypatch.setattr(module, "evaluate_nested_cohort", evaluate)
    result = module.run_multiplicity_bootstrap(dataset, n_bootstraps=2, use_gpu=False)
    assert result["n_failed_refits"] == 2
    assert result["n_bootstraps_succeeded"] == 0
    assert result["interval_procedure_succeeded"] is False
    assert set(result["failure_reasons"]) == {"refit failed", "nonfinite_bootstrap_estimate"}


def test_bootstrap_records_requested_gpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    import src.analysis.paired_lipid.batched as module

    monkeypatch.setattr(module, "_get_gpu_xp", lambda: None)
    execution = {}
    module._run_batched_multiplicity_draws(
        SimpleNamespace(), np.array([1.0, 2.0]), [], [], np.ones((1, 2)),
        PairedLipidConfig(), {"part_band_mat": np.ones((2, 1))},
        use_gpu=True, execution_metadata=execution,
    )
    assert execution["device"] == "cpu"
    assert execution["gpu_fallback"] is True
    assert execution["gpu_requested"] is True


def test_manuscript_provenance_saves_config_and_data_manifest(tmp_path: Path) -> None:
    import json

    from manuscript_ml import write_run_provenance

    config_path = tmp_path / "settings.yaml"
    config_text = "paired_lipid: {}\n"
    config_path.write_text(config_text, encoding="utf-8")
    store_path = tmp_path / "store"
    store_path.mkdir()
    manifest = {"manifest_hash": "fixture", "schema_version": 1}
    (store_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    path = write_run_provenance(
        tmp_path / "output", "fixture", config_path=config_path, cfg=PairedLipidConfig(),
        settings={}, source_meta={"store_path": str(store_path)},
    )
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["config_text"] == config_path.read_bytes().decode("utf-8")
    assert result["data_manifest"] == manifest
    assert result["data_manifest_status"] == "saved"


@pytest.mark.parametrize("preset", ["primray", ""])
def test_covariate_preprocessor_rejects_unknown_preset(preset: str) -> None:
    from src.analysis.paired_lipid.models import CovariatePreprocessor

    with pytest.raises(ValueError, match="Unknown covariate preset"):
        CovariatePreprocessor._build_raw_design(pd.DataFrame(), preset=preset)


def test_paired_lipid_config_validates_invariants() -> None:
    with pytest.raises(ValueError, match="Unknown covariate preset"):
        PairedLipidConfig(covariate_preset="invalid_preset")
    with pytest.raises(ValueError, match="pca_dimensions"):
        PairedLipidConfig(pca_dimensions=(0, 5))
    with pytest.raises(ValueError, match="lambda_grid"):
        PairedLipidConfig(lambda_grid=(-0.01, 0.1))
    with pytest.raises(ValueError, match="ridge_alpha_grid"):
        PairedLipidConfig(ridge_alpha_grid=(-0.1, 1.0))
    with pytest.raises(ValueError, match="n_outer_splits"):
        PairedLipidConfig(n_outer_splits=1)
    with pytest.raises(ValueError, match="n_inner_splits"):
        PairedLipidConfig(n_inner_splits=1)


def test_covariate_preprocessor_excludes_zero_weight_participants() -> None:
    from src.analysis.paired_lipid.models import CovariatePreprocessor

    df = pd.DataFrame(
        {
            "age_years": [60.0, 65.0, 70.0, 75.0, np.nan, 1.0e6],
            "sex": ["female", "male", "female", "male", "female", "male"],
            "statin": [0, 0, 0, 0, 0, 1],
            "sample_label_trend": [10.0, 20.0, 30.0, 40.0, 50.0, -9999.0],
        }
    )
    all_idx = np.arange(6, dtype=int)
    weights = np.array([1.0, 2.0, 1.0, 1.0, 0.0, 0.0], dtype=float)
    active_idx = np.arange(4, dtype=int)

    prep_all = CovariatePreprocessor.fit(df, all_idx, weights, preset="primary")
    prep_act = CovariatePreprocessor.fit(df, active_idx, weights[:4], preset="primary")

    assert prep_all.feature_names == prep_act.feature_names
    assert prep_all.dropped_columns == prep_act.dropped_columns
    assert "statin" in prep_all.dropped_columns
    assert "age_years__missing" not in prep_all.feature_names
    prep_with_missing = CovariatePreprocessor.fit(df, np.arange(5, dtype=int), np.ones(5, dtype=float), preset="primary")
    assert "age_years__missing" in prep_with_missing.feature_names
    np.testing.assert_allclose(
        prep_all.transform(df, active_idx),
        prep_act.transform(df, active_idx),
        rtol=1e-12,
        atol=1e-12,
    )


def test_compute_diagnostics_false_preserves_predictions_and_contributions() -> None:
    ds, _ = generate_synthetic_cohort(n_participants=36, n_single_sample=2, seed=707, compact_grid=True)
    res_full = evaluate_nested_cohort(ds, compute_diagnostics=True)
    res_fast = evaluate_nested_cohort(ds, compute_diagnostics=False)

    assert res_fast["fold_repeatability"] == []
    assert res_fast["subspace_angles"].empty
    pd.testing.assert_frame_equal(res_fast["fold_settings"], res_full["fold_settings"])
    for model, preds in res_full["oof_predictions"].items():
        np.testing.assert_allclose(res_fast["oof_predictions"][model], preds, rtol=0, atol=1e-12)
    for model, contrib in res_full["oof_spectral_contributions"].items():
        np.testing.assert_allclose(res_fast["oof_spectral_contributions"][model], contrib, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "use_gpu",
    [False, pytest.param(True, marks=pytest.mark.skipif(not _gpu_available(), reason="CuPy/CUDA unavailable"))],
)
def test_batched_chefsi_and_cpu_match_sequential_at_cohort_scale(use_gpu: bool) -> None:
    # n=92 (~180 samples) keeps active sample Gram sizes > 32 in both inner and outer divisions,
    # exercising the GPU CheFSI polynomial eigensolver and fused subspace projection.
    from src.analysis.paired_lipid.batched import (
        _check_fast_batch_eligibility,
        _get_dataset_gram_cache,
        _run_batched_multiplicity_draws,
    )
    from src.analysis.paired_lipid.nested_cv import make_nested_splits

    ds, _ = generate_synthetic_cohort(n_participants=92, n_single_sample=4, seed=808, compact_grid=True)
    cfg = PairedLipidConfig()
    n_p = ds.n_participants
    outer, inner = make_nested_splits(ds, cfg, cfg.primary_seed)
    rng = np.random.default_rng(17)
    M = np.vstack(
        [np.ones(n_p)] + [np.bincount(rng.integers(0, n_p, n_p), minlength=n_p).astype(float) for _ in range(2)]
    )
    gcache = _get_dataset_gram_cache(ds, cfg)
    assert _check_fast_batch_eligibility(ds, outer, inner, M, cfg, gcache).all()

    d_pair, d_comp = _run_batched_multiplicity_draws(
        ds, ds.outcomes[cfg.primary_endpoint], outer, inner, M, cfg, gcache, use_gpu=use_gpu
    )
    for b in range(M.shape[0]):
        ref = evaluate_nested_cohort(
            ds,
            cfg=cfg,
            participant_multiplicities=M[b],
            precomputed_splits=(outer, inner),
            include_band_paired=True,
            compute_diagnostics=False,
        )
        assert d_pair[b] == pytest.approx(
            ref["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"], abs=1e-8
        )
        assert d_comp[b] == pytest.approx(ref["paired_vs_strongest_comparator"]["delta_mse"], abs=1e-8)


def test_calibration_two_way_se_and_scenario_suite() -> None:
    from src.analysis.paired_lipid.simulation import evaluate_scenario_calibration_suite

    with pytest.raises(ValueError, match="base_seed"):
        evaluate_bootstrap_calibration(n_repetitions=2, n_bootstraps=4, base_seed=10_000)

    scenarios = [
        {"scenario": "linear_signal", "covariate_preset": "primary", "signal_strength": 0.9},
        {
            "scenario": "nonlinear_proxy_quad",
            "covariate_preset": "quadratic_age_and_label",
            "signal_strength": 0.0,
            "nonlinear_covariate_proxy": True,
        },
    ]
    df = evaluate_scenario_calibration_suite(
        scenarios=scenarios,
        n_repetitions=3,
        n_bootstraps=6,
        n_participants=42,
        n_test_participants=120,
        base_seed=9100,
        compact_grid=True,
    )
    assert list(df["scenario"]) == ["linear_signal", "nonlinear_proxy_quad"]
    assert list(df["covariate_preset"]) == ["primary", "quadratic_age_and_label"]
    assert np.all(np.isfinite(df["target_monte_carlo_se"])) and (df["target_monte_carlo_se"] >= 0.0).all()
    assert np.all(np.isfinite(df["target_test_cohort_se"])) and (df["target_test_cohort_se"] >= 0.0).all()
    assert np.all(np.isfinite(df["target_total_se"])) and (df["target_total_se"] >= 0.0).all()


@pytest.mark.skipif(not _gpu_available(), reason="CuPy/CUDA unavailable")
def test_gpu_chefsi_eigensolver_matches_exact_eigh_to_fp64_precision() -> None:
    from src.analysis.paired_lipid.batched import _chefsi_top_k_gpu, _exact_top_k_eigh_active, _get_gpu_xp

    cp = _get_gpu_xp()
    assert cp is not None
    rng = np.random.default_rng(2026)
    n_s, p_dim, n_draws = 96, 128, 12
    # Power-law spectral decay mimicking Raman covariance eigen-spectra
    scales = np.exp(-0.12 * np.arange(p_dim))
    X = rng.normal(size=(n_s, p_dim)) * scales[None, :]
    K = X @ X.T

    M_s = np.vstack([np.ones(n_s)] + [rng.binomial(2, 0.65, size=n_s).astype(float) for _ in range(n_draws - 1)])
    sw = np.sqrt(M_s / np.maximum(np.sum(M_s, axis=1, keepdims=True), 1.0))
    G_batch = K[None, :, :] * (sw[:, :, None] * sw[:, None, :])
    G_batch = 0.5 * (G_batch + np.swapaxes(G_batch, -1, -2))

    _, U0 = np.linalg.eigh(G_batch[0])
    Q0_comb = U0[:, -32:] / np.maximum(sw[0, :, None], 1e-12)

    G_gpu = cp.asarray(G_batch)
    sw_gpu = cp.asarray(sw)
    Q0_gpu = cp.asarray(Q0_comb)

    ev_ref, u_ref = _exact_top_k_eigh_active(cp, G_gpu, 15, sw_gpu)
    ev_chefsi, u_chefsi = _chefsi_top_k_gpu(cp, G_gpu, 15, sw_gpu, Q0_gpu)

    np.testing.assert_allclose(cp.asnumpy(ev_chefsi), cp.asnumpy(ev_ref), rtol=1e-12, atol=1e-12)
    dots = np.abs(np.sum(cp.asnumpy(u_chefsi) * cp.asnumpy(u_ref), axis=1))
    np.testing.assert_allclose(dots, np.ones_like(dots), rtol=1e-11, atol=1e-11)


def test_cohort_dataset_replace_isolates_cache_and_band_matrix_is_readonly() -> None:
    from dataclasses import replace

    ds, _ = generate_synthetic_cohort(n_participants=12, n_single_sample=2, seed=1011, compact_grid=True)
    mat_1, names_1 = extract_sample_band_matrix(ds)
    assert mat_1.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        mat_1[0, 0] = 0.0

    mat_2, names_2 = extract_sample_band_matrix(ds)
    assert mat_2 is mat_1
    assert names_2 == names_1
    assert names_2 is not names_1
    names_1.append("mutated_band")
    _, names_3 = extract_sample_band_matrix(ds)
    assert names_3 == names_2
    assert "mutated_band" not in names_3

    ds_copy = replace(ds, outcomes={**ds.outcomes})
    assert ds_copy._cache is not ds._cache
    assert ds_copy._cache == {}
    ds_copy._cache["isolated_probe"] = True
    assert "isolated_probe" not in ds._cache

    mat_copy, _ = extract_sample_band_matrix(ds_copy)
    assert mat_copy is not mat_1
    assert mat_copy.flags.writeable is False
    np.testing.assert_array_equal(mat_copy, mat_1)


def test_config_coercion_validation_and_subset_mode_rejection() -> None:
    cfg = PairedLipidConfig(
        pca_dimensions=[4, 8],  # type: ignore[arg-type]
        retained_directions=[1, 2],  # type: ignore[arg-type]
        lambda_grid=[0.01, 0.5],  # type: ignore[arg-type]
        ridge_alpha_grid=[0.1, 10.0],  # type: ignore[arg-type]
        pls_component_grid=[1, 3],  # type: ignore[arg-type]
        sensitivity_seeds=[11, 22],  # type: ignore[arg-type]
        secondary_endpoints=["hdl_mmol_l"],  # type: ignore[arg-type]
    )
    assert cfg.pca_dimensions == (4, 8) and isinstance(cfg.pca_dimensions, tuple)
    assert cfg.retained_directions == (1, 2) and isinstance(cfg.retained_directions, tuple)
    assert cfg.lambda_grid == (0.01, 0.5) and isinstance(cfg.lambda_grid, tuple)
    assert cfg.ridge_alpha_grid == (0.1, 10.0) and isinstance(cfg.ridge_alpha_grid, tuple)
    assert cfg.pls_component_grid == (1, 3) and isinstance(cfg.pls_component_grid, tuple)
    assert cfg.sensitivity_seeds == (11, 22) and isinstance(cfg.sensitivity_seeds, tuple)
    assert cfg.secondary_endpoints == ("hdl_mmol_l",) and isinstance(cfg.secondary_endpoints, tuple)

    for bad_kwargs in (
        {"lambda_grid": (0.01, float("nan"))},
        {"lambda_grid": (0.01, float("inf"))},
        {"ridge_alpha_grid": (0.1, float("nan"))},
        {"ridge_alpha_grid": (0.1, float("inf"))},
        {"pca_dimensions": (5, float("nan"))},
        {"rank_tolerance": float("nan")},
        {"eigenvalue_tolerance": float("inf")},
    ):
        with pytest.raises(ValueError, match="finite"):
            PairedLipidConfig(**bad_kwargs)  # type: ignore[arg-type]

    for bad_kwargs in (
        {"pca_dimensions": (10, 5)},
        {"pca_dimensions": (5, 5)},
        {"retained_directions": (2, 1)},
        {"retained_directions": (2, 2)},
        {"lambda_grid": (1.0, 0.1)},
        {"lambda_grid": (0.1, 0.1)},
        {"ridge_alpha_grid": (10.0, 1.0)},
        {"ridge_alpha_grid": (1.0, 1.0)},
        {"pls_component_grid": (3, 2)},
        {"pls_component_grid": (2, 2)},
    ):
        with pytest.raises(ValueError, match="strictly ascending"):
            PairedLipidConfig(**bad_kwargs)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown subset_mode"):
        build_paired_cohort_dataset(
            pd.DataFrame(),
            pd.DataFrame(),
            np.empty((0, 2)),
            np.empty((0, 2)),
            np.arange(2.0),
            np.arange(2.0),
            subset_mode="invalid_mode",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="Unknown subset_mode"):
        generate_synthetic_cohort(n_participants=10, compact_grid=True, subset_mode="invalid_mode")  # type: ignore[arg-type]


def test_covariate_whitespace_readonly_matrix_and_weight_validation() -> None:
    from src.analysis.paired_lipid.models import CovariatePreprocessor, WeightedPLSResult

    df = pd.DataFrame(
        {
            "age_years": [60.0, 65.0, 70.0, 75.0],
            "sex": [" female ", "  MALE  ", "Female ", " male"],
            "statin": [0, 1, 0, 1],
            "sample_label_trend": [1.0, 2.0, 3.0, 4.0],
            "smoking_status": [" former ", " NEVER ", " current ", " Former"],
        }
    )
    cache: dict[str, object] = {}
    raw_mat, raw_cols = CovariatePreprocessor.raw_design_matrix(
        df, preset="pre_event_risk_factors", raw_cache=cache
    )
    assert raw_mat.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        raw_mat[0, 0] = 99.0

    raw_mat_2, raw_cols_2 = CovariatePreprocessor.raw_design_matrix(
        df, preset="pre_event_risk_factors", raw_cache=cache
    )
    assert raw_mat_2 is raw_mat
    assert raw_cols_2 == raw_cols

    col_idx = {name: i for i, name in enumerate(raw_cols)}
    np.testing.assert_array_equal(raw_mat[:, col_idx["sex_female"]], np.array([1.0, 0.0, 1.0, 0.0]))
    np.testing.assert_array_equal(raw_mat[:, col_idx["smoking_former"]], np.array([1.0, 0.0, 0.0, 1.0]))
    np.testing.assert_array_equal(raw_mat[:, col_idx["smoking_current"]], np.array([0.0, 0.0, 1.0, 0.0]))

    rng = np.random.default_rng(14)
    X_cov = rng.normal(size=(8, 2))
    Z_spec = rng.normal(size=(8, 3))
    y = rng.normal(size=8)
    bad_weights = (
        np.zeros(8, dtype=float),
        np.array([1.0, -0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        np.array([1.0, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        np.array([1.0, np.inf, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    )
    for weights in bad_weights:
        with pytest.raises(ValueError, match="weights"):
            fit_ridge_grid(X_cov, Z_spec, y, weights, [0.1, 1.0])
        with pytest.raises(ValueError, match="weights"):
            WeightedPLSResult.fit(X_cov, Z_spec, y, weights, max_components=2)


def test_weighted_spectral_preprocessor_matches_primal_svd_reference() -> None:
    from src.analysis.paired_lipid.cohort import CohortDataset

    rng = np.random.default_rng(1301)
    n_p, n_paired = 20, 17
    p1, p2 = 35, 25
    counts = np.array([2] * n_paired + [1] * (n_p - n_paired), dtype=int)
    n_s = int(np.sum(counts))
    sample_p_idx = np.repeat(np.arange(n_p, dtype=int), counts)

    # Smooth exponential spectral decay gives strictly separated singular values for exact loading parity.
    r_latent = 14
    q1, _ = np.linalg.qr(rng.normal(size=(p1, r_latent)))
    q2, _ = np.linalg.qr(rng.normal(size=(p2, r_latent)))
    scores = rng.normal(size=(n_s, r_latent)) * np.exp(-0.35 * np.arange(r_latent))[None, :]
    sample_1300 = scores @ q1.T + 0.05
    sample_2850 = scores @ q2.T - 0.03

    paired_p = np.arange(n_paired, dtype=int)
    ds = CohortDataset(
        participant_ids=np.arange(n_p).astype(str),
        outcomes={"ldl_mmol_l": rng.normal(3.0, 0.5, size=n_p)},
        covariate_frame=pd.DataFrame({"age_years": np.linspace(55.0, 80.0, n_p)}),
        wavenumbers_1300=np.linspace(750.0, 1750.0, p1),
        wavenumbers_2850=np.linspace(2500.0, 3150.0, p2),
        sample_1300=sample_1300,
        sample_2850=sample_2850,
        sample_participant_idx=sample_p_idx,
        sample_ids=np.array([f"s_{i}" for i in range(n_s)]),
        sample_parsed_labels=np.arange(1, n_s + 1, dtype=float),
        participant_sample_counts=counts,
        paired_participant_indices=paired_p,
        pair_sample_a_indices=2 * paired_p,
        pair_sample_b_indices=2 * paired_p + 1,
    )

    train_p = np.arange(18, dtype=int)
    mult = np.array([2.0, 0.0, 1.0, 3.0, 1.0, 2.0, 0.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0])
    cfg = PairedLipidConfig(pca_dimensions=(5, 10))
    prep = WeightedSpectralPreprocessor.fit(ds, train_p, mult, cfg=cfg)

    # Explicit primal SVD reference
    keep = mult > 0.0
    active_p = train_p[keep]
    active_m = mult[keep]
    p_to_m = np.zeros(n_p, dtype=float)
    p_to_m[active_p] = active_m
    s_idx = np.flatnonzero(p_to_m[sample_p_idx] > 0.0)
    s_p = sample_p_idx[s_idx]
    w_s = p_to_m[s_p] / (float(np.sum(active_m)) * counts[s_p].astype(float))
    w_s /= float(np.sum(w_s))

    X1, X2 = sample_1300[s_idx], sample_2850[s_idx]
    mean1 = w_s @ X1
    mean2 = w_s @ X2
    m_total = float(np.sum(p_to_m[s_p]))
    sqrt_w = np.sqrt(w_s * (m_total / max(m_total - 1.0, 1.0)))[:, None]
    SW1 = sqrt_w * (X1 - mean1)
    SW2 = sqrt_w * (X2 - mean2)
    _, s1_ref, _ = np.linalg.svd(SW1, full_matrices=False)
    _, s2_ref, _ = np.linalg.svd(SW2, full_matrices=False)
    scale1_ref = 1.0 / float(s1_ref[0])
    scale2_ref = 1.0 / float(s2_ref[0])

    SW_comb = np.hstack([SW1 * scale1_ref, SW2 * scale2_ref])
    _, s_ref, vt_ref = np.linalg.svd(SW_comb, full_matrices=False)
    k = prep.pca_components.shape[0]
    loadings_ref = vt_ref[:k].copy()
    for i in range(k):
        if loadings_ref[i, int(np.argmax(np.abs(loadings_ref[i])))] < 0.0:
            loadings_ref[i] *= -1.0

    np.testing.assert_allclose(prep.mean_1300, mean1, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(prep.mean_2850, mean2, rtol=1e-12, atol=1e-12)
    assert prep.scale_1300 == pytest.approx(scale1_ref, rel=1e-12, abs=1e-12)
    assert prep.scale_2850 == pytest.approx(scale2_ref, rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(prep.pca_singular_values, s_ref[:k], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(prep.pca_components, loadings_ref, rtol=1e-12, atol=1e-12)


def test_two_way_anova_exact_icc_scale_invariance_and_antisymmetric_degeneracy() -> None:
    from src.analysis.paired_lipid.metrics import two_way_anova_ms

    mat = np.array([[1.0, 3.0], [2.0, 6.0], [3.0, 9.0]], dtype=float)
    ms_row, ms_col, ms_err = two_way_anova_ms(mat)
    assert ms_row == pytest.approx(8.0, abs=1e-14)
    assert ms_col == pytest.approx(24.0, abs=1e-14)
    assert ms_err == pytest.approx(2.0, abs=1e-14)

    with pytest.raises(ValueError, match="2D matrix"):
        two_way_anova_ms(np.array([1.0, 2.0, 3.0]))

    base_icc = compute_icc_a1(mat[:, 0], mat[:, 1])
    assert base_icc["estimable"] is True
    assert base_icc["icc_a1"] == pytest.approx(9.0 / 37.0, rel=1e-12, abs=1e-12)

    for scale in (1e-8, 1e8):
        scaled_icc = compute_icc_a1(scale * mat[:, 0], scale * mat[:, 1])
        assert scaled_icc["estimable"] is True
        assert scaled_icc["icc_a1"] == pytest.approx(base_icc["icc_a1"], rel=1e-12, abs=1e-12)

    # n=2 anti-symmetric pairs have MS_R = MS_C = 0 and denom = MS_R + MS_C = 0.
    for shift in (0.0, 1e4):
        deg = compute_icc_a1(np.array([1.0 + shift, -1.0 + shift]), np.array([-1.0 + shift, 1.0 + shift]))
        assert deg["estimable"] is False
        assert deg["reason"] == "degenerate_denominator"
        assert np.isnan(deg["icc_a1"])


def test_discriminability_exact_ties_and_nan_label_exclusion() -> None:
    # Identical embeddings yield exact 0.5 discriminability and chance retrieval via tie credit.
    zeros = np.zeros((4, 2), dtype=float)
    all_ties = compute_discriminability_and_retrieval(zeros, zeros)
    assert all_ties["discriminability"] == pytest.approx(0.5, abs=1e-12)
    assert all_ties["retrieval_top1"] == pytest.approx(0.25, abs=1e-12)

    # Partial exact ties: pairs 0 and 1 coincide at 0.0, pair 2 is isolated at 10.0.
    z_tied = np.array([[0.0], [0.0], [10.0]], dtype=float)
    partial = compute_discriminability_and_retrieval(z_tied, z_tied)
    assert partial["discriminability"] == pytest.approx(5.0 / 6.0, abs=1e-12)
    assert partial["retrieval_top1"] == pytest.approx(2.0 / 3.0, abs=1e-12)

    # NaN labels must be excluded from local candidate windows and local retrieval scoring.
    z_a = np.array([[0.0], [10.0], [20.0], [30.0], [40.0], [50.0]], dtype=float)
    z_b = np.array([[0.2], [10.2], [20.2], [30.2], [0.05], [50.2]], dtype=float)
    labels_a = np.array([1.0, 2.0, 3.0, 4.0, np.nan, 6.0], dtype=float)
    labels_b = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan], dtype=float)
    res = compute_discriminability_and_retrieval(z_a, z_b, labels_a, labels_b, local_window=1)
    # Globally, pair 4's B sample (0.05) hijacks pair 0's A->B match; locally, pairs 4 and 5 are excluded.
    assert res["retrieval_top1"] < 1.0
    assert res["local_retrieval_top1"] == pytest.approx(1.0, abs=1e-12)
    # 4 finite-label pairs with window=1 have candidate counts [2, 3, 3, 2] -> mean chance = 5/12.
    assert res["local_retrieval_chance"] == pytest.approx(5.0 / 12.0, abs=1e-12)

    few_finite = compute_discriminability_and_retrieval(
        z_a[:3], z_b[:3], np.array([1.0, np.nan, np.nan]), np.array([1.0, np.nan, np.nan]), local_window=1
    )
    assert np.isnan(few_finite["local_retrieval_top1"])
    assert np.isnan(few_finite["local_retrieval_chance"])


def test_subspace_principal_angles_matches_scipy_across_regimes() -> None:
    from scipy import linalg

    rng = np.random.default_rng(77)

    def _assert_matches_scipy(base_a: np.ndarray, base_b: np.ndarray) -> None:
        df = compute_subspace_principal_angles([base_a, base_b])
        assert len(df) == 1
        row = df.iloc[0]
        assert int(row["dim_a"]) == base_a.shape[1]
        assert int(row["dim_b"]) == base_b.shape[1]
        ref_rad = linalg.subspace_angles(base_a, base_b)
        ref_deg = np.degrees(ref_rad)
        assert row["max_principal_angle_deg"] == pytest.approx(float(np.max(ref_deg)), rel=1e-12, abs=1e-12)
        assert row["mean_principal_angle_deg"] == pytest.approx(float(np.mean(ref_deg)), rel=1e-12, abs=1e-12)
        assert row["min_cosine"] == pytest.approx(float(np.min(np.cos(ref_rad))), rel=1e-12, abs=1e-12)

    # Equal-dim (including non-orthonormal input)
    A_eq = rng.normal(size=(40, 3))
    B_eq = rng.normal(size=(40, 3))
    _assert_matches_scipy(A_eq, B_eq)

    # Unequal-dim in both orders
    Q_2, _ = linalg.qr(rng.normal(size=(40, 2)), mode="economic")
    Q_4, _ = linalg.qr(rng.normal(size=(40, 4)), mode="economic")
    _assert_matches_scipy(Q_2, Q_4)
    _assert_matches_scipy(Q_4, Q_2)

    # Tiny-angle regime exercising Bjorck-Golub sine refinement
    Q_full, _ = linalg.qr(rng.normal(size=(50, 6)), mode="economic")
    Q_base = Q_full[:, :3]
    Q_pert, _ = linalg.qr(Q_base + 1e-9 * Q_full[:, 3:6], mode="economic")
    _assert_matches_scipy(Q_base, Q_pert)

    # Metric-weighted bases with p_1300 != p_2850
    ds, _ = generate_synthetic_cohort(n_participants=30, n_single_sample=2, seed=78, compact_grid=True)
    p1, p2 = ds.wavenumbers_1300.size, ds.wavenumbers_2850.size
    assert p1 != p2
    cfg = PairedLipidConfig()
    tr_a = np.arange(0, 24, dtype=int)
    tr_b = np.arange(6, 30, dtype=int)
    prep_a = WeightedSpectralPreprocessor.fit(ds, tr_a, np.ones(tr_a.size), cfg=cfg)
    prep_b = WeightedSpectralPreprocessor.fit(ds, tr_b, np.ones(tr_b.size), cfg=cfg)
    mom_a = PairedMoments.from_sample_coordinates(
        prep_a.transform_pca(ds.sample_1300, ds.sample_2850), ds, tr_a, np.ones(tr_a.size)
    )
    mom_b = PairedMoments.from_sample_coordinates(
        prep_b.transform_pca(ds.sample_1300, ds.sample_2850), ds, tr_b, np.ones(tr_b.size)
    )
    Va, _, _ = mom_a.solve_directions(10, lam=0.1)
    Vb, _, _ = mom_b.solve_directions(10, lam=0.1)
    Qa = map_subspace_to_metric_orthonormal_basis(prep_a, 10, Va[:, :2], p1, p2)
    Qb = map_subspace_to_metric_orthonormal_basis(prep_b, 10, Vb[:, :3], p1, p2)
    np.testing.assert_allclose(Qa.T @ Qa, np.eye(2), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(Qb.T @ Qb, np.eye(3), rtol=0.0, atol=1e-12)
    _assert_matches_scipy(Qa, Qb)


def test_anchor_associations_handles_collinear_covariates_under_missingness() -> None:
    from dataclasses import replace

    ds, _ = generate_synthetic_cohort(n_participants=30, n_single_sample=2, seed=812, compact_grid=True)
    cov = ds.covariate_frame.copy()
    # Globally full-rank covariates that drop rank by 2 once participant 0 is masked out by missing HDL.
    cov["statin"] = 0
    cov.loc[0, "statin"] = 1
    cov["age_years"] = 60.0 + 0.5 * cov["sample_label_trend"]
    cov.loc[0, "age_years"] = 85.0

    hdl = ds.outcomes["hdl_mmol_l"].copy()
    hdl[0] = np.nan
    ds_col = replace(ds, covariate_frame=cov, outcomes={**ds.outcomes, "hdl_mmol_l": hdl})

    rng = np.random.default_rng(813)
    spec_contrib = 0.6 * np.where(np.isfinite(hdl), hdl, 1.5) + 0.3 * cov["age_years"].to_numpy() + rng.normal(
        0.0, 0.25, size=ds_col.n_participants
    )
    eval_stub = {"oof_spectral_contributions": {"paired_linear_ridge": spec_contrib}}
    df = summarize_anchor_associations(
        ds_col, eval_stub, anchors=("ldl_mmol_l", "hdl_mmol_l"), n_resamples=40, seed=7
    )
    row_hdl = df.loc[df["anchor"] == "hdl_mmol_l"].iloc[0]
    assert int(row_hdl["n_participants"]) == ds_col.n_participants - 1
    assert np.isfinite(row_hdl["partial_pearson_r"])
    assert np.isfinite(row_hdl["partial_spearman_rho"])
    assert np.isfinite(row_hdl["resample_ci_low"]) and np.isfinite(row_hdl["resample_ci_high"])
    assert row_hdl["attribution_status"] == "continuous_exploratory_summary_only"

    # Independent reduced full-rank reference on the observed HDL subset (intercept, sex_female, label_trend).
    valid = np.arange(1, ds_col.n_participants, dtype=int)
    sex_bin = (cov["sex"].astype(str).str.strip().str.lower() == "female").to_numpy(dtype=float)[valid]
    lbl_v = cov["sample_label_trend"].to_numpy(dtype=float)[valid]
    X_ref = np.column_stack([np.ones(valid.size), sex_bin, lbl_v])
    s_v = spec_contrib[valid]
    y_v = hdl[valid]
    s_res = s_v - X_ref @ np.linalg.lstsq(X_ref, s_v, rcond=None)[0]
    y_res = y_v - X_ref @ np.linalg.lstsq(X_ref, y_v, rcond=None)[0]
    expected_r = float(np.dot(s_res, y_res) / (np.linalg.norm(s_res) * np.linalg.norm(y_res)))
    assert row_hdl["partial_pearson_r"] == pytest.approx(expected_r, rel=0.0, abs=1e-12)


@pytest.mark.parametrize(
    "use_gpu",
    [False, pytest.param(True, marks=pytest.mark.skipif(not _gpu_available(), reason="CuPy/CUDA unavailable"))],
)
def test_batched_pca_schur_sse_matches_batched_ridge(use_gpu: bool) -> None:
    from src.analysis.paired_lipid.batched import (
        _get_dataset_gram_cache,
        _get_gpu_xp,
        _precompute_division_constants,
        _to_np,
        batched_division_prep,
        batched_pca_schur_sse,
        batched_ridge_sse_and_preds,
    )
    from src.analysis.paired_lipid.nested_cv import make_nested_splits

    xp = _get_gpu_xp() if use_gpu else np
    assert xp is not None
    ds, _ = generate_synthetic_cohort(n_participants=60, n_single_sample=2, seed=901, compact_grid=True)
    cfg = PairedLipidConfig(pca_dimensions=(5, 10, 15))
    outer, inner = make_nested_splits(ds, cfg, cfg.primary_seed)
    tr_outer, _ = outer[0]
    in_tr_rel, in_val_rel = inner[0][0]
    tr_p, val_p = tr_outer[in_tr_rel], tr_outer[in_val_rel]

    gcache = _get_dataset_gram_cache(ds, cfg)
    dconst = _precompute_division_constants(xp, gcache, ds, tr_p, val_p, cfg)

    rng = np.random.default_rng(902)
    M = np.vstack(
        [np.ones(ds.n_participants)]
        + [np.bincount(rng.integers(0, ds.n_participants, ds.n_participants), minlength=ds.n_participants) for _ in range(5)]
    ).astype(np.float64)
    M_xp = xp.asarray(M, dtype=xp.float64)
    y_xp = xp.asarray(ds.outcomes[cfg.primary_endpoint], dtype=xp.float64)
    alphas_xp = xp.asarray(list(cfg.ridge_alpha_grid), dtype=xp.float64)

    div = batched_division_prep(
        xp, gcache, dconst, M_xp[:, tr_p], M_xp[:, val_p], y_xp, alphas_xp, cfg, compute_pca_suff=True
    )
    k_list = list(cfg.pca_dimensions)
    sse_schur = _to_np(xp, batched_pca_schur_sse(xp, div, k_list, alphas_xp))
    val_w_sum = np.sum(M[:, val_p], axis=1, keepdims=True)

    for k_idx, k in enumerate(k_list):
        sse_ref, _ = batched_ridge_sse_and_preds(
            xp,
            div["X_cov_tr"],
            div["X_cov_val"],
            div["part_pca_max"][:, tr_p, :k],
            div["part_pca_max"][:, val_p, :k],
            div["y_tr"],
            div["y_val"],
            div["w_tr"],
            div["w_val"],
            alphas_xp,
        )
        sse_ref_np = _to_np(xp, sse_ref)
        np.testing.assert_allclose(sse_schur[:, k_idx, :], sse_ref_np, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(
            sse_schur[:, k_idx, :] / val_w_sum,
            sse_ref_np / val_w_sum,
            rtol=0.0,
            atol=1e-11,
        )


@pytest.mark.parametrize(
    "use_gpu",
    [False, pytest.param(True, marks=pytest.mark.skipif(not _gpu_available(), reason="CuPy/CUDA unavailable"))],
)
def test_rayleigh_ritz_32_cholesky_qr_fallback_on_rank_deficient_batch(use_gpu: bool) -> None:
    from scipy import linalg

    from src.analysis.paired_lipid.batched import _get_gpu_xp, _rayleigh_ritz_32, _to_np

    xp = _get_gpu_xp() if use_gpu else np
    assert xp is not None
    rng = np.random.default_rng(915)
    n_s, q = 48, 32

    # Build a 4-draw symmetric PSD batch with full-rank, exact rank-deficient, near-singular, and zero-col Y.
    A = rng.normal(size=(4, n_s, 20))
    G_np = np.einsum("bik,bjk->bij", A, A)
    G_np = 0.5 * (G_np + np.swapaxes(G_np, -1, -2))

    Y_np = rng.normal(size=(4, n_s, q))
    Y_np[1, :, 16:] = Y_np[1, :, :16]
    Y_np[2, :, 31] = Y_np[2, :, 0] + 1e-14 * rng.normal(size=n_s)
    Y_np[3, :, 31] = 0.0

    for subset in ([0, 1, 2, 3], [0, 2]):
        G_sub = xp.asarray(G_np[subset], dtype=xp.float64)
        Y_sub = xp.asarray(Y_np[subset], dtype=xp.float64)
        theta_xp, U_xp, GU_xp = _rayleigh_ritz_32(xp, G_sub, Y_sub)
        theta = _to_np(xp, theta_xp)
        U = _to_np(xp, U_xp)
        GU = _to_np(xp, GU_xp)
        assert np.all(np.isfinite(theta))
        assert np.all(np.isfinite(U))
        assert np.all(np.isfinite(GU))

        for loc_i, orig_b in enumerate(subset):
            G_b = G_np[orig_b]
            np.testing.assert_allclose(GU[loc_i], G_b @ U[loc_i], rtol=1e-11, atol=1e-11)
            np.testing.assert_allclose(U[loc_i].T @ G_b @ U[loc_i], np.diag(theta[loc_i]), rtol=1e-11, atol=1e-11)

            Q_ref = linalg.orth(Y_np[orig_b])
            r_eff = Q_ref.shape[1]
            ev_ref = linalg.eigvalsh(Q_ref.T @ G_b @ Q_ref)
            np.testing.assert_allclose(theta[loc_i, -r_eff:], ev_ref, rtol=1e-11, atol=1e-11)

    # Empty batch edge case
    theta_e, U_e, GU_e = _rayleigh_ritz_32(
        xp, xp.zeros((0, n_s, n_s), dtype=xp.float64), xp.zeros((0, n_s, q), dtype=xp.float64)
    )
    assert theta_e.shape == (0, q)
    assert U_e.shape == (0, n_s, q)
    assert GU_e.shape == (0, n_s, q)


def test_noncontiguous_pls_grid_parity_between_sequential_and_batched() -> None:
    from src.analysis.paired_lipid.batched import (
        _check_fast_batch_eligibility,
        _get_dataset_gram_cache,
        _run_batched_multiplicity_draws,
    )
    from src.analysis.paired_lipid.nested_cv import make_nested_splits

    ds, _ = generate_synthetic_cohort(n_participants=70, n_single_sample=2, seed=1010, compact_grid=True)
    cfg = PairedLipidConfig(pls_component_grid=(2, 4))
    splits = make_nested_splits(ds, cfg, cfg.primary_seed)

    fast = evaluate_nested_cohort(ds, cfg=cfg, precomputed_splits=splits, compute_diagnostics=False)
    seq = evaluate_nested_cohort(
        ds, cfg=cfg, precomputed_splits=splits, include_band_paired=True, compute_diagnostics=False
    )
    assert set(fast["fold_settings"]["pls_components"]).issubset({2, 4})
    assert set(seq["fold_settings"]["pls_components"]).issubset({2, 4})
    cols = [c for c in fast["fold_settings"].columns if c != "dropped_covariates"]
    pd.testing.assert_frame_equal(fast["fold_settings"][cols], seq["fold_settings"][cols], rtol=1e-8)
    for model, preds in fast["oof_predictions"].items():
        np.testing.assert_allclose(preds, seq["oof_predictions"][model], rtol=0.0, atol=1e-8)

    rng = np.random.default_rng(1011)
    n_p = ds.n_participants
    M = np.vstack(
        [np.ones(n_p)] + [np.bincount(rng.integers(0, n_p, n_p), minlength=n_p) for _ in range(2)]
    ).astype(float)
    gcache = _get_dataset_gram_cache(ds, cfg)
    assert _check_fast_batch_eligibility(ds, *splits, M, cfg, gcache).all()
    d_pair, d_comp = _run_batched_multiplicity_draws(
        ds, ds.outcomes[cfg.primary_endpoint], *splits, M, cfg, gcache, use_gpu=False
    )
    for b in range(M.shape[0]):
        ref = evaluate_nested_cohort(
            ds,
            cfg=cfg,
            participant_multiplicities=M[b],
            precomputed_splits=splits,
            include_band_paired=True,
            compute_diagnostics=False,
        )
        assert d_pair[b] == pytest.approx(
            ref["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"], abs=1e-8
        )
        assert d_comp[b] == pytest.approx(ref["paired_vs_strongest_comparator"]["delta_mse"], abs=1e-8)


def test_twelve_decimal_comparator_tie_breaking_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.analysis.paired_lipid.batched as batched_mod
    import src.analysis.paired_lipid.nested_cv as nested_mod

    ds, _ = generate_synthetic_cohort(n_participants=36, n_single_sample=2, seed=1020, compact_grid=True)
    cfg = PairedLipidConfig()
    outer, inner = nested_mod.make_nested_splits(ds, cfg, cfg.primary_seed)
    tr_idx, te_idx = outer[0]
    y_all = ds.outcomes[cfg.primary_endpoint]
    mult = np.ones(ds.n_participants, dtype=float)

    # 1. Sequential/outer comparator tie-breaking at 12 decimals (band -> pca -> pls).
    for band_mse, pca_mse, pls_mse, expected_name in (
        (1.0, 1.0 - 1e-14, 1.0 - 2e-14, "fixed_band_ridge"),
        (1.1, 1.0, 1.0 - 2e-14, "pca_ridge"),
    ):
        sel_stub = nested_mod._InnerSelection(
            cov_alpha_idx=0,
            band_alpha_idx=0,
            band_mse=band_mse,
            pca_setting=(5, 0),
            pca_mse=pca_mse,
            pls_components=1,
            pls_mse=pls_mse,
            paired_setting=(5, cfg.lambda_grid[0], 1, 0),
            invalid_candidates=[],
        )
        with monkeypatch.context() as mp:
            mp.setattr(nested_mod, "_select_inner_settings", lambda *_a, _s=sel_stub, **_kw: _s)
            fold_out = nested_mod._select_inner_models_and_fit_outer(
                ds, y_all, tr_idx, te_idx, inner[0], mult, cfg, compute_diagnostics=False
            )
        assert fold_out["selected_settings"]["strongest_comparator_name"] == expected_name

    # 2. Batched inner selection comparator tie-breaking at 12 decimals.
    n_b, n_a = 2, len(cfg.ridge_alpha_grid)
    n_k, n_lam, n_r = len(cfg.pca_dimensions), len(cfg.lambda_grid), len(cfg.retained_directions)
    max_pls = max(cfg.pls_component_grid)
    dummy_div = {
        "rank_ok": np.ones(n_b, dtype=bool),
        "w_val": np.full((n_b, 10), 1.0),
        "sse_0": np.full((n_b, n_a), 20.0),
        "X_cov_tr": np.zeros((n_b, 1, 1)),
        "X_cov_val": np.zeros((n_b, 1, 1)),
        "y_tr": np.zeros(1),
        "y_val": np.zeros(1),
        "w_tr": np.ones((n_b, 1)),
        "W_max": np.zeros((n_b, 1, 1)),
        "T_max": np.zeros((n_b, 1, 1)),
    }
    band_sse = np.full((n_b, n_a), 20.0)
    band_sse[0, 0] = 10.0  # MSE = 1.0
    band_sse[1, 0] = 11.0  # MSE = 1.1
    pca_sse = np.full((n_b, n_k, n_a), 20.0)
    pca_sse[0, 0, 0] = 10.0 - 1e-13  # MSE = 1.0 - 1e-14 -> ties band at 12 decimals
    pca_sse[1, 0, 0] = 10.0  # MSE = 1.0
    pls_sse = np.full((n_b, max_pls), 20.0)
    pls_sse[0, 0] = 10.0 - 2e-13  # MSE = 1.0 - 2e-14 -> ties band and pca at 12 decimals
    pls_sse[1, 0] = 10.0 - 2e-13  # MSE = 1.0 - 2e-14 -> ties pca at 12 decimals

    with monkeypatch.context() as mp:
        mp.setattr(batched_mod, "batched_division_prep", lambda *_a, **_kw: dummy_div)
        mp.setattr(batched_mod, "batched_ridge_sse_and_preds", lambda *_a, **_kw: (band_sse, None))
        mp.setattr(batched_mod, "batched_pca_schur_sse", lambda *_a, **_kw: pca_sse)
        mp.setattr(batched_mod, "batched_pls_sse_and_preds", lambda *_a, **_kw: (pls_sse, None))
        mp.setattr(batched_mod, "batched_solve_directions_grid", lambda *_a, **_kw: {})
        mp.setattr(
            batched_mod,
            "batched_paired_schur_sse",
            lambda *_a, **_kw: np.full((n_b, n_r, n_k * n_lam, n_a), 10.0),
        )
        sel = batched_mod.batched_select_inner_settings(
            np,
            {},
            [{"tr_p": np.array([0]), "val_p": np.array([1])}],
            np.ones((n_b, 2)),
            np.zeros(2),
            np.asarray(cfg.ridge_alpha_grid, dtype=float),
            np.zeros((2, 1)),
            cfg,
        )
    assert sel["best_comp_idx"].tolist() == [0, 1]


def test_inner_fold_rank_deficiency_triggers_fallback_draws_out() -> None:
    from dataclasses import replace

    from src.analysis.paired_lipid.batched import (
        _check_fast_batch_eligibility,
        _get_dataset_gram_cache,
        _run_batched_multiplicity_draws,
    )
    from src.analysis.paired_lipid.nested_cv import make_nested_splits

    ds, _ = generate_synthetic_cohort(n_participants=60, n_single_sample=0, seed=1012, compact_grid=True)
    cfg = PairedLipidConfig(pca_dimensions=(5, 10))
    outer, inner = make_nested_splits(ds, cfg, cfg.primary_seed)
    tr_outer_0, _ = outer[0]
    in_tr_0 = tr_outer_0[inner[0][0][0]]
    special_p = in_tr_0[:3]

    # Restrict non-special samples inside inner fold 0's training split to a joint rank-7 subspace so
    # zeroing only special_p causes rank deficiency (< 10) inside inner fold 0 while draw 0 keeps full rank.
    p1 = ds.sample_1300.shape[1]
    s1 = ds.sample_1300.copy()
    s2 = ds.sample_2850.copy()
    restrict_p = np.setdiff1d(in_tr_0, special_p)
    restrict_mask = np.isin(ds.sample_participant_idx, restrict_p)
    comb_r = np.hstack([s1[restrict_mask], s2[restrict_mask]])
    uc, svc, vtc = np.linalg.svd(comb_r, full_matrices=False)
    comb_low = (uc[:, :7] * svc[:7]) @ vtc[:7]
    s1[restrict_mask] = comb_low[:, :p1]
    s2[restrict_mask] = comb_low[:, p1:]
    ds_part_rank = replace(ds, sample_1300=s1, sample_2850=s2)

    m_full = np.ones(ds.n_participants, dtype=float)
    m_def = np.ones(ds.n_participants, dtype=float)
    m_def[special_p] = 0.0
    M = np.vstack([m_full, m_def])

    gcache = _get_dataset_gram_cache(ds_part_rank, cfg)
    assert _check_fast_batch_eligibility(ds_part_rank, outer, inner, M, cfg, gcache).all()
    fallback_out: set[int] = set()
    d_pair, d_comp = _run_batched_multiplicity_draws(
        ds_part_rank,
        ds_part_rank.outcomes[cfg.primary_endpoint],
        outer,
        inner,
        M,
        cfg,
        gcache,
        use_gpu=False,
        fallback_draws_out=fallback_out,
    )
    assert fallback_out == {1}
    assert np.isfinite(d_pair[0]) and np.isfinite(d_comp[0])
    assert np.isnan(d_pair[1])

    # End-to-end bootstrap recovery: globally joint rank-8 spectra with max_k=10 fall back cleanly to sequential.
    comb_all = np.hstack([ds.sample_1300, ds.sample_2850])
    u_all, sv_all, vt_all = np.linalg.svd(comb_all, full_matrices=False)
    comb_rank8 = (u_all[:, :8] * sv_all[:8]) @ vt_all[:8]
    ds_rank8 = replace(ds, sample_1300=comb_rank8[:, :p1], sample_2850=comb_rank8[:, p1:])
    sp_rank8 = WeightedSpectralPreprocessor.fit(ds_rank8, np.arange(48, dtype=int), np.ones(48, dtype=float), cfg=cfg)
    assert sp_rank8.numerical_rank == 8
    assert sp_rank8.pca_components.shape[0] == 8
    boot = run_multiplicity_bootstrap(ds_rank8, cfg=cfg, n_bootstraps=3, bootstrap_seed=42, use_gpu=False)
    assert boot["interval_procedure_succeeded"] is True
    assert boot["n_bootstraps_succeeded"] == 3
    assert boot["n_fallback_draws"] == 3
    assert np.isfinite(boot["point_estimate_delta_mse"])


@pytest.mark.parametrize(
    "use_gpu",
    [False, pytest.param(True, marks=pytest.mark.skipif(not _gpu_available(), reason="CuPy/CUDA unavailable"))],
)
def test_mixed_batch_degenerate_outer_isolates_fallback_draws(
    use_gpu: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.analysis.paired_lipid.batched as batched_mod
    from src.analysis.paired_lipid.nested_cv import make_nested_splits

    ds, _ = generate_synthetic_cohort(n_participants=60, n_single_sample=0, seed=1050, compact_grid=True)
    cfg = PairedLipidConfig(pca_dimensions=(5, 10))
    outer, inner = make_nested_splits(ds, cfg, cfg.primary_seed)
    n_p = ds.n_participants
    rng = np.random.default_rng(1051)
    M = np.vstack(
        [
            np.ones(n_p),
            np.bincount(rng.integers(0, n_p, n_p), minlength=n_p),
            np.bincount(rng.integers(0, n_p, n_p), minlength=n_p),
        ]
    ).astype(np.float64)
    gcache = batched_mod._get_dataset_gram_cache(ds, cfg)
    assert batched_mod._check_fast_batch_eligibility(ds, outer, inner, M, cfg, gcache).all()

    ref_0 = evaluate_nested_cohort(
        ds,
        cfg=cfg,
        participant_multiplicities=M[0],
        precomputed_splits=(outer, inner),
        include_band_paired=True,
        compute_diagnostics=False,
    )
    ref_2 = evaluate_nested_cohort(
        ds,
        cfg=cfg,
        participant_multiplicities=M[2],
        precomputed_splits=(outer, inner),
        include_band_paired=True,
        compute_diagnostics=False,
    )
    orig_prep = batched_mod.batched_division_prep

    # 1. Degenerate outer T_max on draw 1 produces non-finite V_outer_sel without crashing CPU/GPU batch.
    def _prep_degenerate_t(xp, *args, **kwargs):
        out = orig_prep(xp, *args, **kwargs)
        if kwargs.get("compute_cov_preds"):
            out["T_max"][1] = 0.0
        return out

    fb_t: set[int] = set()
    with monkeypatch.context() as mp:
        mp.setattr(batched_mod, "batched_division_prep", _prep_degenerate_t)
        dp_t, dc_t = batched_mod._run_batched_multiplicity_draws(
            ds,
            ds.outcomes[cfg.primary_endpoint],
            outer,
            inner,
            M,
            cfg,
            gcache,
            use_gpu=use_gpu,
            fallback_draws_out=fb_t,
        )
    assert fb_t == {1}
    assert np.isnan(dp_t[1])
    assert dp_t[0] == pytest.approx(ref_0["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"], abs=1e-8)
    assert dc_t[0] == pytest.approx(ref_0["paired_vs_strongest_comparator"]["delta_mse"], abs=1e-8)
    assert dp_t[2] == pytest.approx(ref_2["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"], abs=1e-8)
    assert dc_t[2] == pytest.approx(ref_2["paired_vs_strongest_comparator"]["delta_mse"], abs=1e-8)

    # 2. Failed outer rank_ok (~batch_ok) with poisoned part_pca_max on draw 1 must stay isolated from draws 0 and 2.
    def _prep_bad_rank(xp, *args, **kwargs):
        out = orig_prep(xp, *args, **kwargs)
        if kwargs.get("compute_cov_preds"):
            out["rank_ok"][1] = False
            out["part_pca_max"][1] = xp.nan
        return out

    fb_r: set[int] = set()
    with monkeypatch.context() as mp:
        mp.setattr(batched_mod, "batched_division_prep", _prep_bad_rank)
        dp_r, dc_r = batched_mod._run_batched_multiplicity_draws(
            ds,
            ds.outcomes[cfg.primary_endpoint],
            outer,
            inner,
            M,
            cfg,
            gcache,
            use_gpu=use_gpu,
            fallback_draws_out=fb_r,
        )
    assert fb_r == {1}
    assert np.isnan(dp_r[1])
    assert dp_r[0] == pytest.approx(dp_t[0], abs=1e-12)
    assert dc_r[0] == pytest.approx(dc_t[0], abs=1e-12)
    assert dp_r[2] == pytest.approx(dp_t[2], abs=1e-12)
    assert dc_r[2] == pytest.approx(dc_t[2], abs=1e-12)


def test_multiplicity_bootstrap_allows_missing_secondary_endpoint() -> None:
    from dataclasses import replace

    ds, _ = generate_synthetic_cohort(n_participants=60, n_single_sample=2, seed=1013, compact_grid=True)
    y_hdl = ds.outcomes["hdl_mmol_l"].copy()
    missing_idx = np.array([2, 15, 31, 47])
    y_hdl[missing_idx] = np.nan
    ds_nan = replace(ds, outcomes={**ds.outcomes, "hdl_mmol_l": y_hdl})

    with pytest.raises(ValueError, match="missing values"):
        run_multiplicity_bootstrap(ds_nan, endpoint="hdl_mmol_l", n_bootstraps=4, use_gpu=False)

    boot = run_multiplicity_bootstrap(
        ds_nan,
        endpoint="hdl_mmol_l",
        n_bootstraps=6,
        bootstrap_seed=77,
        use_gpu=False,
        allow_missing_outcome=True,
    )
    assert boot["interval_procedure_succeeded"] is True
    assert boot["n_bootstraps_succeeded"] == 6
    assert boot["n_failed_refits"] == 0
    assert boot["mean_distinct_participants"] <= ds_nan.n_participants - missing_idx.size
    lo, hi = boot["ci_95_delta_mse"]
    assert np.isfinite(lo) and np.isfinite(hi) and lo < hi

    ref = evaluate_nested_cohort(ds_nan, endpoint="hdl_mmol_l", allow_missing_outcome=True, compute_diagnostics=False)
    assert boot["point_estimate_delta_mse"] == pytest.approx(
        ref["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"], abs=1e-8
    )
    assert boot["point_estimate_vs_comparator"] == pytest.approx(
        ref["paired_vs_strongest_comparator"]["delta_mse"], abs=1e-8
    )


def test_synthetic_cohort_subset_modes_and_prepared_external_test() -> None:
    from src.analysis.paired_lipid.nested_cv import PreparedExternalTest, score_external_primary_pair

    ds_all, _ = generate_synthetic_cohort(
        n_participants=30, n_single_sample=5, seed=1101, compact_grid=True, subset_mode="all_eligible"
    )
    ds_paired, truth_paired = generate_synthetic_cohort(
        n_participants=30, n_single_sample=5, seed=1101, compact_grid=True, subset_mode="paired_only"
    )
    assert ds_paired.manifest_summary["subset_mode"] == "paired_only"
    assert ds_paired.n_participants == 25
    assert ds_paired.manifest_summary["n_single_sample_participants"] == 0
    assert truth_paired["z_bio"].shape == (25, 2)
    assert any(e["reason"] == "unpaired_in_paired_only_mode" for e in ds_paired.exclusions)

    ds_single, truth_single = generate_synthetic_cohort(
        n_participants=30, n_single_sample=5, seed=1101, compact_grid=True, subset_mode="single_sample"
    )
    assert ds_single.manifest_summary["subset_mode"] == "single_sample"
    assert ds_single.n_participants == 30
    assert truth_single["z_bio"].shape == (30, 2)
    paired_p = ds_single.paired_participant_indices
    np.testing.assert_allclose(
        ds_single.covariate_frame["sample_label_trend"].to_numpy()[paired_p],
        ds_single.sample_parsed_labels[ds_single.pair_sample_a_indices],
        rtol=0.0,
        atol=1e-12,
    )

    ds_3map, truth_3map = generate_synthetic_cohort(
        n_participants=30, n_single_sample=5, seed=1101, compact_grid=True, subset_mode="three_map_complete"
    )
    assert ds_3map.manifest_summary["subset_mode"] == "three_map_complete"
    assert ds_3map.manifest_summary["min_maps_per_block"] == 3
    assert ds_3map.n_samples < ds_all.n_samples
    assert truth_3map["z_bio"].shape == (ds_3map.n_participants, 2)
    assert any(e["reason"] == "fewer_than_3_maps" for e in ds_3map.exclusions)

    train_ds, _ = generate_synthetic_cohort(n_participants=40, n_single_sample=2, seed=1102, compact_grid=True)
    test_ds, _ = generate_synthetic_cohort(n_participants=80, n_single_sample=4, seed=1103, compact_grid=True)
    cfg = PairedLipidConfig()
    prepared = PreparedExternalTest.from_dataset(test_ds, cfg.primary_endpoint, preset=cfg.covariate_preset)

    ref = evaluate_nested_cohort(train_ds, cfg=cfg, external_test=test_ds, compute_diagnostics=False)
    fast = score_external_primary_pair(train_ds, prepared, cfg=cfg)
    via_eval = evaluate_nested_cohort(train_ds, cfg=cfg, external_test=prepared, compute_diagnostics=False)

    assert fast["endpoint"] == ref["endpoint"]
    assert fast["external_delta_mse"] == pytest.approx(ref["external_delta_mse"], rel=0.0, abs=1e-12)
    np.testing.assert_allclose(
        fast["external_participant_deltas"], ref["external_participant_deltas"], rtol=0.0, atol=1e-12
    )
    assert via_eval["external_delta_mse"] == pytest.approx(ref["external_delta_mse"], rel=0.0, abs=1e-12)
    np.testing.assert_allclose(
        via_eval["external_participant_deltas"], ref["external_participant_deltas"], rtol=0.0, atol=1e-12
    )

    wrong_ep = PreparedExternalTest.from_dataset(test_ds, "hdl_mmol_l", preset=cfg.covariate_preset)
    with pytest.raises(ValueError, match="does not match requested endpoint"):
        score_external_primary_pair(train_ds, wrong_ep, cfg=cfg)


def test_audit_readiness_count_mismatch_and_missing_summary_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    from types import SimpleNamespace

    import manuscript_ml.audit_readiness as audit_mod
    import src.data_preparation.materialize as materialize
    import src.data_preparation.store as store

    master_path = tmp_path / "clinical_master.csv"
    master_path.write_text("participant_key\nstroke_1\n", encoding="utf-8")
    fake_store = tmp_path / "store"
    fake_store.mkdir()
    (fake_store / "manifest.json").write_text("{}", encoding="utf-8")
    source = SimpleNamespace(source_path=fake_store, manifest_hash="fixture_hash")

    monkeypatch.setattr(materialize, "store_dir", lambda *_args: fake_store)
    monkeypatch.setattr(store, "get_active_data_source", lambda: source)
    monkeypatch.setattr(
        audit_mod,
        "load_clinical_master_verified",
        lambda _p: (pd.DataFrame(), {"stroke_identity_verified": 92}),
    )

    # 1. Observed counts mismatch expected counts
    mismatch_ds = SimpleNamespace(
        manifest_summary={
            "n_eligible_participants": 91,
            "n_paired_participants": 87,
            "n_single_sample_participants": 4,
            "n_eligible_samples": 178,
        },
        exclusions=[],
    )
    monkeypatch.setattr(audit_mod, "load_paired_cohort_from_store", lambda *_a, **_kw: (mismatch_ds, {}))
    out_mismatch = audit_mod.run_readiness_audit(
        clinical_master_path=master_path, output_dir=tmp_path / "audit_mismatch"
    )
    manifest_mismatch = json.loads(out_mismatch["readiness_manifest"].read_text(encoding="utf-8"))
    mr = manifest_mismatch["measurement_readiness"]
    assert mr["identity_verified_matches_expected"] is True
    assert mr["cohort_join"]["status"] == "count_mismatch"
    assert mr["ready_for_cohort_join"] is False
    assert mr["ready_for_cohort_evaluation"] is False

    # 2. Missing summary keys must report count_mismatch without raising KeyError
    partial_ds = SimpleNamespace(manifest_summary={"n_eligible_participants": 92}, exclusions=[])
    monkeypatch.setattr(audit_mod, "load_paired_cohort_from_store", lambda *_a, **_kw: (partial_ds, {}))
    out_partial = audit_mod.run_readiness_audit(
        clinical_master_path=master_path, output_dir=tmp_path / "audit_partial"
    )
    manifest_partial = json.loads(out_partial["readiness_manifest"].read_text(encoding="utf-8"))
    mr_p = manifest_partial["measurement_readiness"]
    assert mr_p["cohort_join"]["status"] == "count_mismatch"
    assert mr_p["ready_for_cohort_join"] is False
    assert mr_p["ready_for_cohort_evaluation"] is False
    assert mr_p["cohort_join"]["observed"] == {"n_eligible_participants": 92}


def test_sensitivity_eval_memo_and_config_scenario_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import copy

    import yaml

    import manuscript_ml.run_sensitivity_and_secondary as sens_mod
    from manuscript_ml import DEFAULT_CONFIG_PATH, clear_dataset_cache, load_manuscript_config

    clear_dataset_cache()
    calls: list[tuple[object, ...]] = []

    def _recording_eval(
        ds: object,
        *,
        cfg: PairedLipidConfig = PairedLipidConfig(),
        endpoint: str | None = None,
        outer_seed: int | None = None,
        single_sample_downstream: bool = False,
        allow_missing_outcome: bool = False,
        compute_diagnostics: bool = True,
        **_kwargs: object,
    ) -> dict[str, object]:
        del compute_diagnostics
        calls.append(
            (
                ds.manifest_summary.get("subset_mode"),  # type: ignore[attr-defined]
                cfg.covariate_preset,
                endpoint or cfg.primary_endpoint,
                cfg.primary_seed if outer_seed is None else int(outer_seed),
                bool(single_sample_downstream),
                bool(allow_missing_outcome),
            )
        )
        n_p = ds.n_participants  # type: ignore[attr-defined]
        stub_metric = {
            "delta_mse": 0.12,
            "relative_mse_reduction": 0.06,
            "rmse_model": 0.55,
            "mae_model": 0.42,
        }
        return {
            "n_missing_outcome": 0,
            "comparisons_vs_covariate_ridge": {"paired_linear_ridge": stub_metric},
            "paired_vs_strongest_comparator": {"delta_mse": 0.03},
            "oof_spectral_contributions": {"paired_linear_ridge": np.linspace(-1.0, 1.0, n_p)},
        }

    monkeypatch.setattr(sens_mod, "evaluate_nested_cohort", _recording_eval)
    try:
        outputs = sens_mod.run_sensitivity_and_secondary(
            use_synthetic=True,
            compact_grid=True,
            output_dir=tmp_path / "sens_memo",
        )
    finally:
        clear_dataset_cache()

    # 5 covariate presets + 3 non-all_eligible subsets + 3 sensitivity seeds + 3 secondary endpoints = 14 unique calls.
    assert len(calls) == 14
    assert len(set(calls)) == 14
    primary_key = ("all_eligible", "primary", "ldl_mmol_l", 42, False, False)
    assert calls.count(primary_key) == 1
    assert len(pd.read_csv(outputs["covariate_and_subset_sensitivity"])) == 9
    assert len(pd.read_csv(outputs["outer_partition_seed_sensitivity"])) == 4
    assert len(pd.read_csv(outputs["secondary_endpoints_summary"])) == 3
    assert len(pd.read_csv(outputs["anchor_associations_continuous"])) == 4

    # load_manuscript_config must reject unknown/reserved scenario kwargs and invalid presets upfront.
    base_raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    for bad_update, pattern in (
        ({"unknown_scenario_arg": 1.0}, "Unknown or reserved generator argument"),
        ({"seed": 999}, "Unknown or reserved generator argument"),
        ({"n_participants": 50}, "Unknown or reserved generator argument"),
        ({"covariate_preset": "unknown_preset"}, "Unknown covariate_preset"),
    ):
        mutated = copy.deepcopy(base_raw)
        mutated["simulation"]["scenarios"][0].update(bad_update)
        bad_cfg_path = tmp_path / f"bad_{next(iter(bad_update))}.yaml"
        bad_cfg_path.write_text(yaml.safe_dump(mutated), encoding="utf-8")
        with pytest.raises(ValueError, match=pattern):
            load_manuscript_config(bad_cfg_path)


def test_band_paired_and_band_discriminability_are_scale_invariant_across_blocks() -> None:
    from dataclasses import replace

    # Real stores use SNV for 1300 (~SD 0.5) and trapezoidal area for 2850 (~SD 1e-3),
    # a ~2e5 variance ratio that previously erased C-H bands from band_paired_ridge and band_discriminability.
    ds_base, _ = generate_synthetic_cohort(
        n_participants=40, n_single_sample=2, seed=1201, compact_grid=True, signal_strength=0.9
    )
    ds_scaled = replace(ds_base, sample_2850=ds_base.sample_2850 * 1e-3)

    res_base = evaluate_nested_cohort(ds_base, include_band_paired=True, compute_diagnostics=True)
    res_scaled = evaluate_nested_cohort(ds_scaled, include_band_paired=True, compute_diagnostics=True)

    np.testing.assert_allclose(
        res_base["oof_predictions"]["band_paired_ridge"],
        res_scaled["oof_predictions"]["band_paired_ridge"],
        rtol=1e-9,
        atol=1e-9,
    )
    for rep_a, rep_b in zip(res_base["fold_repeatability"], res_scaled["fold_repeatability"]):
        bd_a = rep_a["band_discriminability"]
        bd_b = rep_b["band_discriminability"]
        assert bd_a["discriminability"] == pytest.approx(bd_b["discriminability"], abs=1e-12)
        assert bd_a["retrieval_top1"] == pytest.approx(bd_b["retrieval_top1"], abs=1e-12)
        assert bd_a["local_retrieval_top1"] == pytest.approx(bd_b["local_retrieval_top1"], abs=1e-12)
        assert bd_a["global_candidate_size"] == float(rep_a["n_paired_test"])
        assert np.isfinite(bd_a["mean_local_candidate_size"]) and bd_a["mean_local_candidate_size"] >= 2.0


def test_external_comparator_scoring_and_calibration_parity() -> None:
    from src.analysis.paired_lipid.calibration import evaluate_scenario_calibration_suite
    from src.analysis.paired_lipid.nested_cv import PreparedExternalTest, score_external_primary_pair

    train_ds, _ = generate_synthetic_cohort(n_participants=45, n_single_sample=3, seed=1202, compact_grid=True)
    test_ds, _ = generate_synthetic_cohort(n_participants=100, n_single_sample=7, seed=1203, compact_grid=True)
    cfg = PairedLipidConfig()
    prepared = PreparedExternalTest.from_dataset(test_ds, cfg.primary_endpoint, preset=cfg.covariate_preset)

    ref = evaluate_nested_cohort(train_ds, cfg=cfg, external_test=test_ds, compute_diagnostics=False)
    fast = score_external_primary_pair(train_ds, prepared, cfg=cfg)
    assert fast["external_delta_mse"] == pytest.approx(ref["external_delta_mse"], abs=1e-12)
    assert fast["external_delta_vs_comparator"] == pytest.approx(ref["external_delta_vs_comparator"], abs=1e-12)
    np.testing.assert_allclose(
        fast["external_participant_deltas_vs_comparator"],
        ref["external_participant_deltas_vs_comparator"],
        rtol=0.0,
        atol=1e-12,
    )

    cal = evaluate_bootstrap_calibration(
        n_repetitions=3,
        n_bootstraps=6,
        n_participants=42,
        signal_strength=0.85,
        base_seed=9200,
        n_test_participants=120,
        compact_grid=True,
    )
    for key in (
        "target_delta_vs_comparator",
        "target_monte_carlo_se_vs_comparator",
        "target_test_cohort_se_vs_comparator",
        "target_total_se_vs_comparator",
        "mean_point_estimate_vs_comparator",
        "bias_vs_comparator",
        "unconditional_coverage_vs_comparator",
        "conditional_coverage_vs_comparator",
        "target_sensitivity_min_coverage",
        "target_sensitivity_min_coverage_vs_comparator",
        "wilson_95_lower_vs_comparator",
        "wilson_95_upper_vs_comparator",
        "mean_interval_width_vs_comparator",
    ):
        assert np.isfinite(cal[key]), f"Expected finite calibration metric {key}, got {cal[key]}"
    assert cal["target_sensitivity_min_coverage"] <= cal["unconditional_coverage"] + 1e-12
    assert cal["target_sensitivity_min_coverage_vs_comparator"] <= cal["unconditional_coverage_vs_comparator"] + 1e-12

    with pytest.raises(ValueError, match="disjoint from replicate seeds"):
        evaluate_scenario_calibration_suite(
            scenarios=[
                {"scenario": "s1", "covariate_preset": "primary"},
                {"scenario": "s2", "covariate_preset": "primary"},
            ],
            n_repetitions=2,
            n_bootstraps=4,
            base_seed=9999,
        )


def test_synthetic_label_drift_confounds_ldl_and_is_controlled_by_label_covariate() -> None:
    ds, _ = generate_synthetic_cohort(
        n_participants=60,
        n_single_sample=2,
        seed=1204,
        compact_grid=True,
        signal_strength=0.0,
        ab_shift_scale=0.0,
        shared_prep_scale=0.0,
        label_drift_scale=0.9,
    )
    lbl = ds.covariate_frame["sample_label_trend"].to_numpy(dtype=float)
    ldl = ds.outcomes["ldl_mmol_l"]
    corr = float(np.corrcoef(lbl, ldl)[0, 1])
    assert corr > 0.6

    res_adj = evaluate_nested_cohort(
        ds, cfg=PairedLipidConfig(covariate_preset="primary"), compute_diagnostics=False
    )
    res_unadj = evaluate_nested_cohort(
        ds, cfg=PairedLipidConfig(covariate_preset="no_label_trend"), compute_diagnostics=False
    )
    delta_adj = res_adj["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"]
    delta_unadj = res_unadj["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"]
    assert delta_unadj > delta_adj + 0.1


def test_singular_lambda_zero_denominator_recorded_invalid_without_crash() -> None:
    from dataclasses import replace

    # Identical A/B pairs make W = 0, so lam = 0.0 triggers scipy.linalg.LinAlgError while lam = 0.1 succeeds.
    ds, _ = generate_synthetic_cohort(n_participants=36, n_single_sample=0, seed=1205, compact_grid=True)
    s1 = ds.sample_1300.copy()
    s2 = ds.sample_2850.copy()
    s1[ds.pair_sample_b_indices] = s1[ds.pair_sample_a_indices]
    s2[ds.pair_sample_b_indices] = s2[ds.pair_sample_a_indices]
    ds_zero_w = replace(ds, sample_1300=s1, sample_2850=s2)

    cfg = PairedLipidConfig(pca_dimensions=(5,), retained_directions=(1, 2), lambda_grid=(0.0, 0.1))
    res = evaluate_nested_cohort(ds_zero_w, cfg=cfg, include_band_paired=True, compute_diagnostics=False)
    assert (res["fold_settings"]["paired_lambda"] == 0.1).all()
    assert (res["fold_settings"]["n_invalid_candidates"] >= 1).all()


def test_primary_evaluation_outputs_unambiguous_baseline_and_candidate_sizes(tmp_path: Path) -> None:
    import json

    from manuscript_ml.run_primary_evaluation import run_primary_evaluation

    outputs = run_primary_evaluation(
        use_synthetic=True,
        output_dir=tmp_path / "prim_schema",
        n_bootstraps=4,
        compact_grid=True,
        use_gpu=False,
    )
    comp_df = pd.read_csv(outputs["comparator_errors_and_differences"])
    assert "baseline_model" in comp_df.columns
    assert "mse_baseline" in comp_df.columns
    comp_row = comp_df.loc[comp_df["comparison"] == "paired_linear_ridge_vs_strongest_comparator"].iloc[0]
    assert comp_row["baseline_model"] == "strongest_comparator"
    assert np.isfinite(comp_row["mse_baseline"])
    assert pd.isna(comp_row["mse_covariate"])

    rep_df = pd.read_csv(outputs["baseline_repeatability_report"])
    for key in ("paired_discriminability", "pca_discriminability", "band_discriminability"):
        assert f"{key}_global_candidate_size" in rep_df.columns
        assert f"{key}_mean_local_candidate_size" in rep_df.columns

    manifest = json.loads(outputs["eligibility_and_split_manifest"].read_text(encoding="utf-8"))
    boot_m = manifest["multiplicity_bootstrap"]
    assert boot_m["parity_within_1e6"] is True
    assert "monte_carlo_se_of_mean_vs_comparator" in boot_m


def test_band_paired_ablation_handles_singular_w_without_crashing_primary() -> None:
    # With n_paired_train < 21 and lam=0.0, W_21 is singular even when k=5 paired_linear_ridge succeeds.
    ds, _ = generate_synthetic_cohort(n_participants=26, n_single_sample=4, seed=1206, compact_grid=True)
    cfg = PairedLipidConfig(pca_dimensions=(5,), retained_directions=(1, 2), lambda_grid=(0.0,))
    res = evaluate_nested_cohort(ds, cfg=cfg, include_band_paired=True, compute_diagnostics=False)
    assert np.isfinite(res["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"])
    # Secondary ablation must be reported non-estimable (NaN) rather than crashing or dropping folds.
    assert np.isnan(res["comparisons_vs_covariate_ridge"]["band_paired_ridge"]["delta_mse"])
    assert np.isnan(res["comparisons_vs_covariate_ridge"]["band_paired_ridge"]["mse_model"])


def test_prepared_external_test_missing_band_matrix_fails_closed() -> None:
    from src.analysis.paired_lipid.nested_cv import PreparedExternalTest, score_external_primary_pair

    ds_tr, _ = generate_synthetic_cohort(n_participants=45, n_single_sample=2, seed=2, compact_grid=True)
    ds_te, _ = generate_synthetic_cohort(n_participants=60, n_single_sample=2, seed=3, compact_grid=True)
    bad_ext = PreparedExternalTest(
        endpoint="ldl_mmol_l",
        spec_1300=ds_te.sample_1300[:60],
        spec_2850=ds_te.sample_2850[:60],
        raw_cov_frame=ds_te.covariate_frame,
        y_ext=ds_te.outcomes["ldl_mmol_l"],
        n_participants=60,
        part_band_mat=None,
    )
    with pytest.raises(ValueError, match="part_band_mat is required"):
        score_external_primary_pair(ds_tr, bad_ext, cfg=PairedLipidConfig())


@pytest.mark.parametrize(
    "use_gpu",
    [False, pytest.param(True, marks=pytest.mark.skipif(not _gpu_available(), reason="CuPy/CUDA unavailable"))],
)
def test_batched_covariate_ptp_and_zero_cov_parity_with_sequential(use_gpu: bool) -> None:
    from dataclasses import replace

    from src.analysis.paired_lipid.batched import (
        _get_dataset_gram_cache,
        _run_batched_multiplicity_draws,
    )
    from src.analysis.paired_lipid.nested_cv import make_nested_splits

    ds, _ = generate_synthetic_cohort(n_participants=70, n_single_sample=2, seed=1207, compact_grid=True)
    cfg = PairedLipidConfig()
    outer, inner = make_nested_splits(ds, cfg, cfg.primary_seed)
    M = np.ones((2, 70), dtype=float)
    M[1, 0] = 2.0
    M[1, 1] = 0.0

    # 1. Small-scale covariate with ptp=1e-6 (> 1e-12) but variance=2.5e-13 (<= 1e-12)
    cov_tiny = ds.covariate_frame.copy()
    parity = np.where(np.arange(70) % 2 == 0, 1.0, -1.0)
    cov_tiny["age_years"] = 65.0 + 0.5e-6 * parity
    y_par = 3.0 + 2.0 * parity + np.random.default_rng(1208).normal(0.0, 0.2, size=70)
    ds_tiny = replace(ds, covariate_frame=cov_tiny, outcomes={**ds.outcomes, "ldl_mmol_l": y_par})
    gc_tiny = _get_dataset_gram_cache(ds_tiny, cfg)
    dp_t, dc_t = _run_batched_multiplicity_draws(ds_tiny, y_par, outer, inner, M, cfg, gc_tiny, use_gpu=use_gpu)
    for b in range(2):
        ref = evaluate_nested_cohort(
            ds_tiny, cfg=cfg, participant_multiplicities=M[b], precomputed_splits=(outer, inner),
            include_band_paired=True, compute_diagnostics=False,
        )
        assert dp_t[b] == pytest.approx(
            ref["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"], abs=1e-10
        )
        assert dc_t[b] == pytest.approx(ref["paired_vs_strongest_comparator"]["delta_mse"], abs=1e-10)

    # 2. All covariates constant (p_cov == 0 after dropping)
    cov_const = ds.covariate_frame.copy()
    cov_const["age_years"] = 65.0
    cov_const["sex"] = "female"
    cov_const["statin"] = 0
    cov_const["sample_label_trend"] = 10.0
    ds_zero = replace(ds, covariate_frame=cov_const)
    gc_zero = _get_dataset_gram_cache(ds_zero, cfg)
    dp_z, dc_z = _run_batched_multiplicity_draws(
        ds_zero, ds_zero.outcomes["ldl_mmol_l"], outer, inner, M, cfg, gc_zero, use_gpu=use_gpu
    )
    ref0 = evaluate_nested_cohort(
        ds_zero, cfg=cfg, participant_multiplicities=M[0], precomputed_splits=(outer, inner),
        include_band_paired=True, compute_diagnostics=False,
    )
    assert dp_z[0] == pytest.approx(
        ref0["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"], abs=1e-10
    )
    assert dc_z[0] == pytest.approx(ref0["paired_vs_strongest_comparator"]["delta_mse"], abs=1e-10)


def test_fused_pls_predict_from_raw_blocks_matches_unfused_predict() -> None:
    from src.analysis.paired_lipid.models import CovariatePreprocessor, WeightedPLSResult

    ds, _ = generate_synthetic_cohort(n_participants=45, n_single_sample=2, seed=1209, compact_grid=True)
    cfg = PairedLipidConfig()
    tr = np.arange(35, dtype=int)
    te = np.arange(35, 45, dtype=int)
    w_tr = np.ones(tr.size, dtype=float)
    cov_prep = CovariatePreprocessor.fit(ds.covariate_frame, tr, w_tr, preset=cfg.covariate_preset)
    X_cov_tr = cov_prep.transform(ds.covariate_frame, tr)
    X_cov_te = cov_prep.transform(ds.covariate_frame, te)
    spec_prep = WeightedSpectralPreprocessor.fit(ds, tr, w_tr, cfg=cfg)
    part_1300 = aggregate_samples_to_participants(ds.sample_1300, ds)
    part_2850 = aggregate_samples_to_participants(ds.sample_2850, ds)
    X_spec_tr = spec_prep.transform_scaled_blocks(part_1300[tr], part_2850[tr])
    X_spec_te = spec_prep.transform_scaled_blocks(part_1300[te], part_2850[te])

    pls = WeightedPLSResult.fit(X_cov_tr, X_spec_tr, ds.outcomes["ldl_mmol_l"][tr], w_tr, max_components=4)
    p_all_ref, s_all_ref = pls.predict(X_cov_te, X_spec_te)
    p_all_fus, s_all_fus = pls.predict_from_raw_blocks(spec_prep, X_cov_te, part_1300[te], part_2850[te])
    np.testing.assert_allclose(p_all_fus, p_all_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(s_all_fus, s_all_ref, rtol=1e-12, atol=1e-12)
    for c in range(1, 5):
        p_ref, s_ref = pls.predict(X_cov_te, X_spec_te, n_components=c)
        p_fus, s_fus = pls.predict_from_raw_blocks(
            spec_prep, X_cov_te, part_1300[te], part_2850[te], n_components=c
        )
        np.testing.assert_allclose(p_fus, p_ref, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(s_fus, s_ref, rtol=1e-12, atol=1e-12)


def test_anchor_associations_reresidualizes_per_bootstrap_resample() -> None:
    from scipy import linalg

    from src.analysis.paired_lipid.models import CovariatePreprocessor

    ds, _ = generate_synthetic_cohort(n_participants=40, n_single_sample=2, seed=1210, compact_grid=True)
    cfg = PairedLipidConfig(covariate_preset="pre_event_risk_factors")
    rng_s = np.random.default_rng(1211)
    spec_c = rng_s.normal(size=ds.n_participants)
    df = summarize_anchor_associations(
        ds,
        {"oof_spectral_contributions": {"paired_linear_ridge": spec_c}},
        anchors=("ldl_mmol_l",),
        n_resamples=60,
        seed=99,
        cfg=cfg,
    )
    row = df.iloc[0]

    # Independent direct re-residualization reference with the exact same seed
    all_idx = np.arange(ds.n_participants, dtype=int)
    cp = CovariatePreprocessor.fit(ds.covariate_frame, all_idx, np.ones(ds.n_participants), preset=cfg.covariate_preset)
    X_cov = cp.transform(ds.covariate_frame, all_idx)
    design = np.column_stack([np.ones(ds.n_participants, dtype=float), X_cov])
    y_a = ds.outcomes["ldl_mmol_l"]
    rng = np.random.default_rng(99)
    idx_mat = rng.integers(0, ds.n_participants, size=(60, ds.n_participants))
    ref_rs = []
    for b_idx in idx_mat:
        d_b = design[b_idx]
        u_b, sv_b, _ = linalg.svd(d_b, full_matrices=False)
        uk_b = u_b[:, sv_b > max(d_b.shape) * np.finfo(float).eps * sv_b[0]]
        sb = spec_c[b_idx] - uk_b @ (uk_b.T @ spec_c[b_idx])
        yb = y_a[b_idx] - uk_b @ (uk_b.T @ y_a[b_idx])
        ref_rs.append(float(np.dot(sb, yb) / (linalg.norm(sb) * linalg.norm(yb))))
    assert row["resample_ci_low"] == pytest.approx(float(np.percentile(ref_rs, 2.5)), abs=1e-12)
    assert row["resample_ci_high"] == pytest.approx(float(np.percentile(ref_rs, 97.5)), abs=1e-12)


def test_calibration_checkpoint_resume_sufficient_stats_and_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.analysis.paired_lipid.calibration as cal_mod
    from manuscript_ml.run_simulation_calibration import run_simulation_and_calibration
    from src.analysis.paired_lipid.calibration import (
        _two_way_anova_ms_from_sufficient_stats,
        evaluate_scenario_calibration_suite,
    )
    from src.analysis.paired_lipid.metrics import two_way_anova_ms

    # 1. Exact O(R + C) two-way ANOVA parity against dense matrix ANOVA
    rng = np.random.default_rng(1212)
    D = rng.normal(loc=0.5, scale=1.2, size=(15, 200))
    ms_r_ref, ms_c_ref, ms_e_ref = two_way_anova_ms(D)
    row_means = np.mean(D, axis=1)
    col_sums = np.sum(D, axis=0)
    row_ss = np.sum((D - row_means[:, None]) ** 2, axis=1)
    ms_r, ms_c, ms_e = _two_way_anova_ms_from_sufficient_stats(row_means, col_sums, row_ss, D.shape[1])
    assert ms_r == pytest.approx(ms_r_ref, rel=1e-12, abs=1e-12)
    assert ms_c == pytest.approx(ms_c_ref, rel=1e-12, abs=1e-12)
    assert ms_e == pytest.approx(ms_e_ref, rel=1e-12, abs=1e-12)

    # 2. Input validation rejects degenerate calibration parameters upfront
    with pytest.raises(ValueError, match="n_bootstraps"):
        evaluate_bootstrap_calibration(n_repetitions=2, n_bootstraps=0, n_participants=40)
    with pytest.raises(ValueError, match="n_repetitions"):
        evaluate_bootstrap_calibration(n_repetitions=0, n_bootstraps=4, n_participants=40)
    with pytest.raises(ValueError, match="base_seed"):
        evaluate_bootstrap_calibration(n_repetitions=2, n_bootstraps=4, base_seed=-1)

    # 3. Chunked checkpoint save/resume matches uncheckpointed run and catches config mismatch
    ckpt_file = tmp_path / "cal_ckpt.npz"
    ref = evaluate_bootstrap_calibration(
        n_repetitions=4,
        n_bootstraps=6,
        n_participants=42,
        signal_strength=0.85,
        base_seed=9250,
        n_test_participants=80,
        compact_grid=True,
        use_gpu=False,
        n_jobs=1,
    )
    assert ref["conditional_coverage"] + ref["lower_noncoverage_rate"] + ref["upper_noncoverage_rate"] == pytest.approx(
        1.0, abs=1e-12
    )
    assert (
        ref["conditional_coverage_vs_comparator"]
        + ref["lower_noncoverage_rate_vs_comparator"]
        + ref["upper_noncoverage_rate_vs_comparator"]
        == pytest.approx(1.0, abs=1e-12)
    )
    assert np.isfinite(ref["empirical_sd_point_estimate"])
    assert np.isfinite(ref["empirical_sd_point_estimate_vs_comparator"])

    res_ckpt = evaluate_bootstrap_calibration(
        n_repetitions=4,
        n_bootstraps=6,
        n_participants=42,
        signal_strength=0.85,
        base_seed=9250,
        n_test_participants=80,
        compact_grid=True,
        use_gpu=False,
        n_jobs=1,
        checkpoint_path=ckpt_file,
        checkpoint_chunk_size=2,
    )
    assert ckpt_file.is_file()
    for key in (
        "target_delta_mse",
        "target_test_cohort_se",
        "target_total_se",
        "unconditional_coverage",
        "lower_noncoverage_rate",
        "upper_noncoverage_rate",
        "target_delta_vs_comparator",
        "target_total_se_vs_comparator",
        "unconditional_coverage_vs_comparator",
    ):
        assert res_ckpt[key] == pytest.approx(ref[key], rel=1e-12, abs=1e-12)

    # Simulate real mid-run crash on rep index 2 (after chunk 0 [reps 0, 1] flushes to disk),
    # then resume and verify only reps 2 and 3 execute and match the uncheckpointed run to 1e-12.
    ckpt_mid = tmp_path / "cal_mid_crash.npz"
    orig_eval_rep = cal_mod._eval_single_calibration_rep
    calls_before_crash = 0

    def _crash_on_third(*args: object, **kwargs: object) -> object:
        nonlocal calls_before_crash
        calls_before_crash += 1
        if calls_before_crash == 3:
            raise RuntimeError("simulated mid-campaign interruption")
        return orig_eval_rep(*args, **kwargs)  # type: ignore[arg-type]

    with monkeypatch.context() as mp:
        mp.setattr(cal_mod, "_eval_single_calibration_rep", _crash_on_third)
        with pytest.raises(RuntimeError, match="simulated mid-campaign interruption"):
            evaluate_bootstrap_calibration(
                n_repetitions=4,
                n_bootstraps=6,
                n_participants=42,
                signal_strength=0.85,
                base_seed=9250,
                n_test_participants=80,
                compact_grid=True,
                use_gpu=False,
                n_jobs=1,
                checkpoint_path=ckpt_mid,
                checkpoint_chunk_size=2,
            )
    assert ckpt_mid.is_file()
    with np.load(ckpt_mid, allow_pickle=False) as partial_state:
        assert int(np.sum(partial_state["completed"])) == 2

    calls_on_resume = 0

    def _counting_eval(*args: object, **kwargs: object) -> object:
        nonlocal calls_on_resume
        calls_on_resume += 1
        return orig_eval_rep(*args, **kwargs)  # type: ignore[arg-type]

    with monkeypatch.context() as mp:
        mp.setattr(cal_mod, "_eval_single_calibration_rep", _counting_eval)
        res_resumed = evaluate_bootstrap_calibration(
            n_repetitions=4,
            n_bootstraps=6,
            n_participants=42,
            signal_strength=0.85,
            base_seed=9250,
            n_test_participants=80,
            compact_grid=True,
            use_gpu=False,
            n_jobs=1,
            checkpoint_path=ckpt_mid,
            checkpoint_chunk_size=2,
        )
    assert calls_on_resume == 2
    for key in (
        "target_delta_mse",
        "target_test_cohort_se",
        "target_total_se",
        "unconditional_coverage",
        "lower_noncoverage_rate",
        "upper_noncoverage_rate",
        "target_delta_vs_comparator",
        "target_total_se_vs_comparator",
        "unconditional_coverage_vs_comparator",
    ):
        assert res_resumed[key] == pytest.approx(ref[key], rel=1e-12, abs=1e-12)

    with pytest.raises(ValueError, match="fingerprint"):
        evaluate_bootstrap_calibration(
            n_repetitions=4,
            n_bootstraps=6,
            n_participants=42,
            signal_strength=0.5,  # mismatched config!
            base_seed=9250,
            n_test_participants=80,
            compact_grid=True,
            use_gpu=False,
            n_jobs=1,
            checkpoint_path=ckpt_file,
        )

    # 4. Filtered scenario calibration preserves canonical seed offsets and writes checkpoints
    out_dir = tmp_path / "filtered_cal"
    outs = run_simulation_and_calibration(
        output_dir=out_dir,
        n_calibration_reps=2,
        n_bootstraps=4,
        n_participants=42,
        n_test_participants=60,
        compact_grid=True,
        use_gpu=False,
        n_jobs=1,
        calibrate_all_scenarios=True,
        scenario_names=["no_association_reference"],
        skip_scenario_suite=True,
        resume=True,
        checkpoint_chunk_size=1,
    )
    assert "simulation_scenarios_summary" not in outs
    cal_df = pd.read_csv(outs["simulation_calibration_report"])
    assert list(cal_df["scenario"]) == ["no_association_reference"]
    # no_association_reference is scenario index 1 in config/manuscript_ml.yaml
    assert (out_dir / "calibration_checkpoints" / "scenario_01_no_association_reference.npz").is_file()
    assert (out_dir / "calibration_checkpoints" / "partial_calibration_report.csv").is_file()
    # Direct suite check with explicit canonical offset matches identically
    direct_df = evaluate_scenario_calibration_suite(
        scenarios=[{"scenario": "no_association_reference", "covariate_preset": "primary", "signal_strength": 0.0, "ab_shift_scale": 0.15, "label_drift_scale": 0.0}],
        n_repetitions=2,
        n_bootstraps=4,
        n_participants=42,
        base_seed=9000,
        n_test_participants=60,
        compact_grid=True,
        use_gpu=False,
        n_jobs=1,
        base_seed_offsets=[1],
    )
    assert cal_df["target_delta_mse"].iloc[0] == pytest.approx(direct_df["target_delta_mse"].iloc[0], abs=1e-12)


def test_studentized_t_bootstrap_batched_vs_sequential_parity_cpu_and_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.analysis.paired_lipid.nested_cv as nested_mod
    from src.analysis.paired_lipid.calibration import evaluate_bootstrap_calibration

    ds, _ = generate_synthetic_cohort(n_participants=48, seed=2101, compact_grid=True)
    cfg = PairedLipidConfig()

    boot_cpu = run_multiplicity_bootstrap(
        ds, cfg=cfg, n_bootstraps=6, bootstrap_seed=701, interval_method="studentized_t", use_gpu=False
    )
    assert boot_cpu["interval_method"] == "studentized_t"
    assert boot_cpu["interval_procedure_succeeded"] is True
    assert boot_cpu["n_failed_refits"] == 0
    assert boot_cpu["mean_distinct_participants"] > 0.0
    assert 0 < boot_cpu["min_distinct_participants"] <= boot_cpu["max_distinct_participants"] <= 48
    for mc_key in (
        "mc_se_ci_95_low_delta_mse",
        "mc_se_ci_95_high_delta_mse",
        "mc_se_ci_95_low_vs_comparator",
        "mc_se_ci_95_high_vs_comparator",
        "mc_se_ci_95_low_strongest_vs_covariate",
        "mc_se_ci_95_high_strongest_vs_covariate",
    ):
        assert np.isfinite(boot_cpu[mc_key]) and boot_cpu[mc_key] >= 0.0

    # Force sequential fallback for all draws and verify exact parity (<= 1e-12) across means, SDs, and CIs.
    with monkeypatch.context() as mp:
        mp.setattr(
            nested_mod,
            "_check_fast_batch_eligibility",
            lambda _ds, _os, _is, m_mat, _cfg, _gc: np.zeros(m_mat.shape[0], dtype=bool),
        )
        boot_seq = run_multiplicity_bootstrap(
            ds, cfg=cfg, n_bootstraps=6, bootstrap_seed=701, interval_method="studentized_t", use_gpu=False
        )
    assert boot_seq["n_fallback_draws"] == 6
    for arr_key in (
        "bootstrap_draws_delta_mse",
        "bootstrap_draws_vs_comparator",
        "bootstrap_draws_strongest_vs_covariate",
        "bootstrap_oof_sds_delta_mse",
        "bootstrap_oof_sds_vs_comparator",
        "bootstrap_oof_sds_strongest_vs_covariate",
    ):
        assert np.max(np.abs(boot_cpu[arr_key] - boot_seq[arr_key])) <= 1e-12
    for ci_key in (
        "ci_95_delta_mse",
        "ci_95_vs_comparator",
        "ci_95_strongest_vs_covariate",
        "ci_95_percentile_delta_mse",
        "ci_95_percentile_vs_comparator",
        "ci_95_percentile_strongest_vs_covariate",
    ):
        assert boot_cpu[ci_key][0] == pytest.approx(boot_seq[ci_key][0], abs=1e-12)
        assert boot_cpu[ci_key][1] == pytest.approx(boot_seq[ci_key][1], abs=1e-12)

    if _gpu_available():
        boot_gpu = run_multiplicity_bootstrap(
            ds, cfg=cfg, n_bootstraps=6, bootstrap_seed=701, interval_method="studentized_t", use_gpu=True
        )
        for arr_key in (
            "bootstrap_draws_delta_mse",
            "bootstrap_draws_vs_comparator",
            "bootstrap_draws_strongest_vs_covariate",
            "bootstrap_oof_sds_delta_mse",
            "bootstrap_oof_sds_vs_comparator",
            "bootstrap_oof_sds_strongest_vs_covariate",
        ):
            assert np.max(np.abs(boot_gpu[arr_key] - boot_seq[arr_key])) <= 1e-12

    # Degenerate OOF SD (<= 1e-12) must fail closed under studentized_t.
    stub_zero_sd = {
        "comparisons_vs_covariate_ridge": {
            "paired_linear_ridge": {"delta_mse": 0.0, "oof_loss_diff_sd": 0.0},
            "strongest_comparator": {"delta_mse": 0.0, "oof_loss_diff_sd": 0.0},
        },
        "paired_vs_strongest_comparator": {"delta_mse": 0.0, "oof_loss_diff_sd": 0.0},
    }
    with monkeypatch.context() as mp:
        mp.setattr(
            nested_mod,
            "_check_fast_batch_eligibility",
            lambda _ds, _os, _is, m_mat, _cfg, _gc: np.zeros(m_mat.shape[0], dtype=bool),
        )
        mp.setattr(nested_mod, "evaluate_nested_cohort", lambda *_a, **_kw: stub_zero_sd)
        deg = run_multiplicity_bootstrap(
            ds, cfg=cfg, n_bootstraps=3, bootstrap_seed=702, interval_method="studentized_t", use_gpu=False
        )
    assert deg["interval_procedure_succeeded"] is False
    assert deg["n_failed_refits"] == 3
    assert any("degenerate" in r for r in deg["failure_reasons"])

    # Calibration checkpoint fingerprint must reject mismatched interval_method.
    ckpt_path = tmp_path / "cal_im.npz"
    cal_st = evaluate_bootstrap_calibration(
        n_repetitions=2,
        n_bootstraps=4,
        n_participants=42,
        signal_strength=0.85,
        base_seed=9310,
        n_test_participants=60,
        compact_grid=True,
        interval_method="studentized_t",
        use_gpu=False,
        n_jobs=1,
        checkpoint_path=ckpt_path,
    )
    assert cal_st["interval_method"] == "studentized_t"
    with pytest.raises(ValueError, match="fingerprint"):
        evaluate_bootstrap_calibration(
            n_repetitions=2,
            n_bootstraps=4,
            n_participants=42,
            signal_strength=0.85,
            base_seed=9310,
            n_test_participants=60,
            compact_grid=True,
            interval_method="percentile",
            use_gpu=False,
            n_jobs=1,
            checkpoint_path=ckpt_path,
        )


def test_engine_edge_cases_constant_nan_covariate_gram_rank_and_anchor_guard() -> None:
    import warnings
    from dataclasses import replace

    from src.analysis.paired_lipid._batched_models import _prep_mfa_pca_batch
    from src.analysis.paired_lipid._gram_cache import _get_dataset_gram_cache, _precompute_division_constants
    from src.analysis.paired_lipid.models import CovariatePreprocessor, WeightedSpectralPreprocessor
    from src.analysis.paired_lipid.nested_cv import summarize_anchor_associations

    # 1. Constant observed numeric covariate with NaNs must retain its {col}__missing indicator.
    cov_df = pd.DataFrame(
        {
            "age_years": [60.0, 65.0, 70.0, 75.0, 80.0, 85.0],
            "sex": ["female", "male", "female", "male", "female", "male"],
            "statin": [1.0, 1.0, np.nan, 1.0, np.nan, 1.0],
            "sample_label_trend": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    idx_all = np.arange(6, dtype=int)
    w_all = np.ones(6, dtype=float)
    prep = CovariatePreprocessor.fit(cov_df, idx_all, w_all, preset="primary")
    assert "statin" in prep.dropped_columns
    assert "statin__missing" in prep.feature_names
    x_mat = prep.transform(cov_df, idx_all)
    assert x_mat.shape == (6, len(prep.feature_names))
    assert np.all(np.isfinite(x_mat))

    # 2. Gram, primal SVD, and batched MFA-PCA agree on numerical_rank.
    ds, _ = generate_synthetic_cohort(n_participants=36, seed=2102, compact_grid=True)
    cfg = PairedLipidConfig()
    tr_idx = np.arange(28, dtype=int)
    te_idx = np.arange(28, 36, dtype=int)
    w_tr = np.ones(tr_idx.size, dtype=float)
    spec_gram = WeightedSpectralPreprocessor.fit(ds, tr_idx, w_tr, cfg=cfg)
    gcache = _get_dataset_gram_cache(ds, cfg)
    dconst = _precompute_division_constants(np, gcache, ds, tr_idx, te_idx, cfg)
    m_tr = np.ones((1, tr_idx.size), dtype=float)
    w_tr_sum = np.array([[float(tr_idx.size)]], dtype=float)
    batch_prep = _prep_mfa_pca_batch(np, gcache, dconst, m_tr, w_tr_sum, cfg, compute_full_rank=True)
    assert int(batch_prep["numerical_rank"][0]) == int(spec_gram.numerical_rank)

    # 3. summarize_anchor_associations: constant input emits zero ConstantInputWarning, tags target_role & per-fold stats.
    ds_miss, _ = generate_synthetic_cohort(
        n_participants=42, seed=2103, compact_grid=True, covariate_missing_rate=0.15
    )
    eval_res = evaluate_nested_cohort(ds_miss, cfg=cfg, compute_diagnostics=False)
    # Build a synthetic eval dict with constant spectral contributions to verify Spearman constant guard.
    eval_const = dict(eval_res)
    eval_const["oof_spectral_contributions"] = {
        "paired_linear_ridge": np.zeros(ds_miss.n_participants, dtype=float)
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        anchor_const = summarize_anchor_associations(ds_miss, eval_const, n_resamples=20, seed=11, cfg=cfg)
    assert not any("ConstantInput" in str(w.category.__name__) or "constant" in str(w.message).lower() for w in caught)
    assert (anchor_const["partial_spearman_rho"] == 0.0).all()

    anchor_df = summarize_anchor_associations(ds_miss, eval_res, n_resamples=30, seed=11, cfg=cfg)
    assert set(
        ["target_role", "fold_median_partial_r", "fold_min_partial_r", "fold_max_partial_r"]
    ).issubset(anchor_df.columns)
    roles = dict(zip(anchor_df["anchor"], anchor_df["target_role"]))
    assert roles["ldl_mmol_l"] == "primary_supervised_target"
    assert roles["hdl_mmol_l"] == "unseen_secondary_anchor"
    assert roles["triglycerides_mmol_l"] == "unseen_secondary_anchor"
    assert roles["plt_10e9_l"] == "unseen_secondary_anchor"
    assert np.all(np.isfinite(anchor_df["fold_median_partial_r"]))
    _ = replace(cfg, primary_endpoint="ldl_mmol_l")


def test_readonly_cohort_dataset_manifest_hash_and_dictionary_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    from types import SimpleNamespace

    import manuscript_ml as runner
    import manuscript_ml.audit_readiness as audit_mod
    import src.data_preparation.materialize as materialize
    import src.data_preparation.store as store

    ds, _ = generate_synthetic_cohort(n_participants=24, seed=2104, compact_grid=True)
    with pytest.raises(ValueError, match="read-only"):
        ds.sample_1300[0, 0] = 999.0
    with pytest.raises(ValueError, match="read-only"):
        ds.outcomes["ldl_mmol_l"][0] = 999.0
    with pytest.raises(ValueError, match="read-only"):
        ds.paired_participant_indices[0] = 0

    # Active source manifest_hash disagreeing with disk manifest.json must raise RuntimeError.
    fake_store = tmp_path / "store"
    fake_store.mkdir()
    (fake_store / "manifest.json").write_text(json.dumps({"manifest_hash": "disk_hash_abc"}), encoding="utf-8")
    master_path = tmp_path / "clinical_master.csv"
    master_path.write_text("participant_key,age_years\nstroke_1,60\n", encoding="utf-8")
    dict_path = tmp_path / "clinical_master_dictionary.csv"
    dict_path.write_text("column,applies_to\nparticipant_key,both\nwrong_col,both\n", encoding="utf-8")

    bad_source = SimpleNamespace(source_path=fake_store, manifest_hash="active_hash_xyz")
    monkeypatch.setattr(store, "get_active_data_source", lambda: bad_source)
    runner.clear_dataset_cache()
    with pytest.raises(RuntimeError, match="manifest_hash"):
        runner.resolve_experiment_dataset(
            use_synthetic=False,
            approve_cohort_evaluation=True,
            clinical_master_path=master_path,
            readiness={"expected_identity_verified": 1, "expected_cohort_join": {"n_eligible_participants": 1}},
        )
    runner.clear_dataset_cache()

    # Mismatched clinical_master_dictionary.csv columns must block ready_for_cohort_evaluation.
    good_source = SimpleNamespace(source_path=fake_store, manifest_hash="disk_hash_abc")
    monkeypatch.setattr(materialize, "store_dir", lambda *_a: fake_store)
    monkeypatch.setattr(store, "get_active_data_source", lambda: good_source)
    monkeypatch.setattr(
        audit_mod, "load_clinical_master_verified", lambda _p: (pd.DataFrame(), {"stroke_identity_verified": 92})
    )
    ok_ds = SimpleNamespace(
        manifest_summary={
            "n_eligible_participants": 92,
            "n_paired_participants": 88,
            "n_single_sample_participants": 4,
            "n_eligible_samples": 180,
        },
        exclusions=[],
    )
    monkeypatch.setattr(audit_mod, "load_paired_cohort_from_store", lambda *_a, **_kw: (ok_ds, {}))
    out_bad = audit_mod.run_readiness_audit(clinical_master_path=master_path, output_dir=tmp_path / "audit_dict_bad")
    m_bad = json.loads(out_bad["readiness_manifest"].read_text(encoding="utf-8"))
    assert m_bad["measurement_readiness"]["dictionary_columns_match_exact"] is False
    assert m_bad["measurement_readiness"]["ready_for_cohort_evaluation"] is False

    dict_path.write_text("column,applies_to\nparticipant_key,both\nage_years,both\n", encoding="utf-8")
    out_ok = audit_mod.run_readiness_audit(
        clinical_master_path=master_path,
        output_dir=tmp_path / "audit_dict_ok",
        measured_runtime_seconds=10.19,
    )
    m_ok = json.loads(out_ok["readiness_manifest"].read_text(encoding="utf-8"))
    assert m_ok["measurement_readiness"]["dictionary_columns_match_exact"] is True
    assert m_ok["measurement_readiness"]["ready_for_cohort_evaluation"] is True
    assert m_ok["computational_readiness"]["status"] == "measured_in_current_audit"
    assert m_ok["computational_readiness"]["total_division_candidate_fits_per_refit"] == 5350


def test_single_sample_autodetect_new_synthetic_regimes_and_mde_smoke(tmp_path: Path) -> None:
    from manuscript_ml.run_simulation_calibration import run_simulation_and_calibration
    from src.analysis.paired_lipid.synthetic import DEFAULT_N_SINGLE_SAMPLE

    # 1. Default n_single_sample == 4 and randomized positions across 0..n_participants-1.
    ds_def, _ = generate_synthetic_cohort(n_participants=92, seed=2105, compact_grid=True)
    assert DEFAULT_N_SINGLE_SAMPLE == 4
    assert ds_def.manifest_summary["n_single_sample_participants"] == 4
    single_pos = np.flatnonzero(ds_def.participant_sample_counts == 1)
    assert single_pos.size == 4
    assert not np.array_equal(single_pos, np.arange(88, 92))

    # 2. single_sample auto-detection in evaluate_nested_cohort & run_multiplicity_bootstrap.
    ds_ss, _ = generate_synthetic_cohort(n_participants=36, seed=2106, compact_grid=True, subset_mode="single_sample")
    cfg = PairedLipidConfig()
    ev_auto = evaluate_nested_cohort(ds_ss, cfg=cfg, compute_diagnostics=False)
    ev_expl = evaluate_nested_cohort(ds_ss, cfg=cfg, single_sample_downstream=True, compute_diagnostics=False)
    assert ev_auto["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"] == pytest.approx(
        ev_expl["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"], abs=1e-12
    )
    b_auto = run_multiplicity_bootstrap(ds_ss, cfg=cfg, n_bootstraps=3, bootstrap_seed=77, use_gpu=False)
    b_expl = run_multiplicity_bootstrap(
        ds_ss, cfg=cfg, n_bootstraps=3, bootstrap_seed=77, single_sample_downstream=True, use_gpu=False
    )
    assert b_auto["n_fallback_draws"] == 3
    assert b_auto["point_estimate_delta_mse"] == pytest.approx(b_expl["point_estimate_delta_mse"], abs=1e-12)
    assert np.max(np.abs(b_auto["bootstrap_draws_delta_mse"] - b_expl["bootstrap_draws_delta_mse"])) <= 1e-12

    # 3. Formula-linked LDL includes label drift and new regimes produce valid finite cohorts.
    ds_f0, _ = generate_synthetic_cohort(
        n_participants=60, seed=2107, compact_grid=True, formula_linked_ldl=True, label_drift_scale=0.0
    )
    ds_f1, _ = generate_synthetic_cohort(
        n_participants=60, seed=2107, compact_grid=True, formula_linked_ldl=True, label_drift_scale=0.9
    )
    corr_f1 = float(np.corrcoef(ds_f1.covariate_frame["sample_label_trend"], ds_f1.outcomes["ldl_mmol_l"])[0, 1])
    corr_f0 = float(np.corrcoef(ds_f0.covariate_frame["sample_label_trend"], ds_f0.outcomes["ldl_mmol_l"])[0, 1])
    assert corr_f1 > corr_f0 + 0.3

    ds_qc, _ = generate_synthetic_cohort(
        n_participants=60, n_single_sample=0, seed=2108, compact_grid=True, qc_sample_dropout_fraction=0.2
    )
    assert ds_qc.manifest_summary["n_single_sample_participants"] > 0
    assert len(ds_qc.exclusions) == ds_qc.manifest_summary["n_single_sample_participants"]

    for kwargs in (
        {"n_nuisance_modes": 12, "nuisance_part_scale": 0.9, "nuisance_sample_scale": 0.9, "include_shoulder_in_bio": True},
        {"n_nuisance_modes": 6, "nuisance_sample_scale": 0.8, "correlated_tech_mode_rho": 0.65},
        {"unequal_b_error_scale": 0.8},
        {"nonlinear_spectral_signal": True},
        {"map_spatial_corr": 0.85, "map_spatial_scale": 0.5},
        {"covariate_missing_rate": 0.15},
    ):
        ds_reg, _ = generate_synthetic_cohort(n_participants=36, seed=2109, compact_grid=True, **kwargs)
        assert np.all(np.isfinite(ds_reg.sample_1300))
        assert np.all(np.isfinite(ds_reg.sample_2850))
        assert np.all(np.isfinite(ds_reg.outcomes["ldl_mmol_l"]))

    # 4. MDE curve smoke via run_simulation_and_calibration.
    mde_outs = run_simulation_and_calibration(
        output_dir=tmp_path / "mde_smoke",
        quick=True,
        n_calibration_reps=2,
        n_bootstraps=4,
        n_jobs=1,
        use_gpu=False,
        skip_scenario_suite=True,
        run_mde_curve=True,
    )
    assert "simulation_mde_curve" in mde_outs
    mde_df = pd.read_csv(mde_outs["simulation_mde_curve"])
    assert len(mde_df) == 2
    assert set(["signal_strength", "detection_rate_delta_mse", "interval_method"]).issubset(mde_df.columns)
    assert (mde_df["interval_method"] == "studentized_t").all()


def test_expanded_diagnostics_weights_identity_artifact_builder_and_cli_parity(tmp_path: Path) -> None:
    import json

    from manuscript_ml.build_manuscript_artifacts import build_manuscript_artifacts
    from manuscript_ml.run_all import _parse_args as parse_run_all_args
    from manuscript_ml.run_primary_evaluation import run_primary_evaluation
    from manuscript_ml.run_sensitivity_and_secondary import run_sensitivity_and_secondary

    stage_dir = tmp_path / "stage_out"
    prim_outs = run_primary_evaluation(
        use_synthetic=True,
        output_dir=stage_dir,
        n_bootstraps=4,
        compact_grid=True,
        use_gpu=False,
    )
    sens_outs = run_sensitivity_and_secondary(
        use_synthetic=True,
        output_dir=stage_dir,
        compact_grid=True,
        n_bootstraps=3,
        use_gpu=False,
    )

    # 1. Expanded primary schemas & bootstrap_draws.npz
    assert prim_outs["fold_spectral_weights"].is_file()
    assert prim_outs["bootstrap_draws"].is_file()
    manifest = json.loads(prim_outs["eligibility_and_split_manifest"].read_text(encoding="utf-8"))
    mb = manifest["multiplicity_bootstrap"]
    for key in (
        "interval_method",
        "ci_95_strongest_vs_covariate",
        "ci_95_percentile_delta_mse",
        "mean_distinct_participants",
        "mc_se_ci_95_low_delta_mse",
        "mc_se_ci_95_high_delta_mse",
    ):
        assert key in mb
    with np.load(prim_outs["bootstrap_draws"]) as npz:
        assert set(npz.files) == {
            "delta_mse",
            "vs_comparator",
            "strongest_vs_covariate",
            "oof_sds_delta_mse",
            "oof_sds_vs_comparator",
            "oof_sds_strongest_vs_covariate",
        }

    rep_df = pd.read_csv(prim_outs["baseline_repeatability_report"])
    for col in (
        "strongest_comparator_name",
        "pca_full_icc_estimate",
        "pca_spectral_icc_estimate",
        "pls_full_icc_estimate",
        "pls_spectral_icc_estimate",
        "strongest_comparator_spectral_icc_estimate",
        "pls_discriminability_score",
        "paired_sd_spectral_contrib",
        "paired_spectral_variance_share",
        "paired_spectral_icc_ms_between_participants",
        "paired_spectral_icc_ms_error",
    ):
        assert col in rep_df.columns

    fs_df = pd.read_csv(prim_outs["fold_selected_settings"])
    for col in (
        "tr_T",
        "tr_W",
        "tr_W_over_tr_T",
        "mean_ab_shift_share_of_tr_W",
        "top_rhos_json",
        "inner_mse_covariate_ridge",
        "inner_mse_paired_linear_ridge",
        "inner_min_paired_distinct",
        "inner_min_numerical_rank",
    ):
        assert col in fs_df.columns

    ang_df = pd.read_csv(prim_outs["subspace_principal_angles"])
    assert set(ang_df["model"].unique()) == {"paired_linear_ridge", "pca_ridge"}
    assert {"angle_1_deg", "angle_2_deg", "angle_3_deg"}.issubset(ang_df.columns)

    # 2. Exact identity check: canonical fold_spectral_weights reconstructs OOF spectral contributions (up to intercept offset).
    w_df = pd.read_csv(prim_outs["fold_spectral_weights"])
    pred_df = pd.read_csv(prim_outs["nested_held_out_predictions"])
    syn_seed = int(manifest["source_metadata"]["synthetic_seed"])
    ds_syn, _ = generate_synthetic_cohort(n_participants=92, seed=syn_seed, compact_grid=True)
    p1300 = aggregate_samples_to_participants(ds_syn.sample_1300, ds_syn)
    p2850 = aggregate_samples_to_participants(ds_syn.sample_2850, ds_syn)
    p_all = np.hstack([p1300, p2850])
    for fold_id in range(5):
        te_mask = pred_df["outer_fold"].to_numpy() == fold_id
        w_vec = w_df[f"paired_weight_fold_{fold_id}"].to_numpy(dtype=float)
        s_hat = p_all[te_mask] @ w_vec
        s_ref = pred_df.loc[te_mask, "spectral_contrib_paired_linear_ridge"].to_numpy(dtype=float)
        # Constant intercept offset cancels when centered across test participants.
        diff_centered = (s_hat - np.mean(s_hat)) - (s_ref - np.mean(s_ref))
        assert float(np.max(np.abs(diff_centered))) <= 1e-11
    for col in ("paired_weight_mean", "paired_weight_sd", "paired_weight_abs_mean", "pca_weight_mean", "pls_weight_mean"):
        assert col in w_df.columns and np.all(np.isfinite(w_df[col]))

    # 3. Expanded sensitivity columns with optional bootstrap CIs.
    sens_df = pd.read_csv(sens_outs["covariate_and_subset_sensitivity"])
    for col in (
        "mse_covariate",
        "mse_paired",
        "mse_pca",
        "mse_pls",
        "mse_fixed_band",
        "strongest_comparator_name",
        "attenuation_vs_primary_delta_mse",
        "ci_95_delta_mse_low",
        "ci_95_delta_mse_high",
    ):
        assert col in sens_df.columns

    # 4. build_manuscript_artifacts refuses synthetic outputs unless --allow-synthetic-preview is passed.
    art_dir = tmp_path / "artifacts"
    with pytest.raises(PermissionError, match="allow-synthetic-preview"):
        build_manuscript_artifacts(input_dir=stage_dir, output_dir=art_dir)

    art_outs = build_manuscript_artifacts(
        input_dir=stage_dir, output_dir=art_dir, allow_synthetic_preview=True
    )
    for key in (
        "table_primary_results",
        "table_repeatability_and_angles",
        "table_sensitivity_and_secondary",
        "fig_repeatability_icc_and_scale",
        "fig_spectral_weights_profile",
        "fig_sensitivity_forest",
        "artifact_build_manifest",
    ):
        assert key in art_outs and art_outs[key].is_file()
    assert "SYNTHETIC PREVIEW" in art_outs["table_primary_results"].read_text(encoding="utf-8")

    # 5. run_all CLI parity check
    parsed = parse_run_all_args(
        [
            "--synthetic",
            "--cohort-grid",
            "--participants",
            "92",
            "--n-test-participants",
            "25000",
            "--scenarios",
            "linear_biological_signal,no_association_reference",
            "--skip-scenario-suite",
            "--run-mde-curve",
            "--resume",
            "--checkpoint-chunk-size",
            "8",
            "--batch-size",
            "32",
            "--interval-method",
            "studentized_t",
            "--sensitivity-bootstraps",
            "10",
        ]
    )
    assert parsed.compact_grid is False
    assert parsed.participants == 92
    assert parsed.n_test_participants == 25000
    assert parsed.scenarios == "linear_biological_signal,no_association_reference"
    assert parsed.skip_scenario_suite is True
    assert parsed.run_mde_curve is True
    assert parsed.resume is True
    assert parsed.checkpoint_chunk_size == 8
    assert parsed.batch_size == 32
    assert parsed.interval_method == "studentized_t"
    assert parsed.sensitivity_bootstraps == 10


def test_review_round2_hardening_guards_and_latex_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import warnings
    from dataclasses import replace
    from types import SimpleNamespace

    import manuscript_ml.audit_readiness as audit_mod
    import src.data_preparation.materialize as materialize
    import src.data_preparation.store as store
    from manuscript_ml.build_manuscript_artifacts import build_manuscript_artifacts
    from src.analysis.paired_lipid.nested_cv import (
        _InnerSelection,
        evaluate_nested_cohort,
        summarize_anchor_associations,
    )

    # 1. _InnerSelection default inner_dropped_covariates must be an immutable tuple.
    sel_stub = _InnerSelection(0, 0, 1.0, (5, 0), 1.0, 1, 1.0, (5, 0.1, 1, 0), [])
    assert isinstance(sel_stub.inner_dropped_covariates, tuple)
    assert sel_stub.inner_dropped_covariates == ()

    # 2. audit_readiness must fail closed when clinical_master_dictionary.csv is absent.
    master_path = tmp_path / "clinical_master.csv"
    master_path.write_text("participant_key,age_years\nstroke_1,65\n", encoding="utf-8")
    fake_store = tmp_path / "store"
    fake_store.mkdir()
    (fake_store / "manifest.json").write_text("{}", encoding="utf-8")
    source = SimpleNamespace(source_path=fake_store, manifest_hash="fixture_hash")
    ok_ds = SimpleNamespace(
        manifest_summary={
            "n_eligible_participants": 92,
            "n_paired_participants": 88,
            "n_single_sample_participants": 4,
            "n_eligible_samples": 180,
        },
        exclusions=[],
    )
    monkeypatch.setattr(materialize, "store_dir", lambda *_args: fake_store)
    monkeypatch.setattr(store, "get_active_data_source", lambda: source)
    monkeypatch.setattr(
        audit_mod,
        "load_clinical_master_verified",
        lambda _p: (pd.DataFrame(), {"stroke_identity_verified": 92}),
    )
    monkeypatch.setattr(audit_mod, "load_paired_cohort_from_store", lambda *_a, **_kw: (ok_ds, {}))

    out_missing_dict = audit_mod.run_readiness_audit(
        clinical_master_path=master_path,
        output_dir=tmp_path / "audit_missing_dict",
    )
    m_missing = json.loads(out_missing_dict["readiness_manifest"].read_text(encoding="utf-8"))
    assert m_missing["measurement_readiness"]["clinical_dictionary"]["present"] is False
    assert m_missing["measurement_readiness"]["ready_for_cohort_evaluation"] is False

    # 3. summarize_anchor_associations must skip saturated folds (residual df < 2) rather than injecting 0.0.
    ds_sat, _ = generate_synthetic_cohort(
        n_participants=35, n_single_sample=2, seed=2201, compact_grid=True, covariate_missing_rate=0.15
    )
    cfg_rf = replace(PairedLipidConfig(), covariate_preset="pre_event_risk_factors")
    ev_sat = evaluate_nested_cohort(ds_sat, cfg=cfg_rf, compute_diagnostics=False)
    adf_sat = summarize_anchor_associations(ds_sat, ev_sat, n_resamples=10, cfg=cfg_rf)
    assert not adf_sat.empty
    assert adf_sat["fold_median_partial_r"].isna().all()
    assert adf_sat["fold_min_partial_r"].isna().all()
    assert adf_sat["fold_max_partial_r"].isna().all()

    # 4. build_manuscript_artifacts must enforce synthetic guard on standalone sensitivity/anchor CSVs
    #    and handle empty optional CSVs cleanly with zero Matplotlib warnings and valid tabularx X specs.
    seed_only = tmp_path / "seed_only"
    seed_only.mkdir()
    pd.DataFrame(
        [
            {
                "outer_seed": 42,
                "role": "primary_locked",
                "mse_covariate": 1.0,
                "delta_mse_paired": 0.1,
                "delta_vs_strongest_comparator": 0.0,
                "attenuation_vs_primary_delta_mse": 0.0,
            }
        ]
    ).to_csv(seed_only / "outer_partition_seed_sensitivity.csv", index=False)
    with pytest.raises(PermissionError, match="allow-synthetic-preview"):
        build_manuscript_artifacts(input_dir=seed_only, output_dir=tmp_path / "out_seed_blocked")

    empty_ang_dir = tmp_path / "empty_angles"
    empty_ang_dir.mkdir()
    pd.DataFrame(
        [
            {
                "outer_fold": 0,
                "n_paired_test": 10,
                "paired_full_icc_estimate": 0.5,
                "paired_spectral_icc_estimate": 0.4,
                "paired_sd_spectral_contrib": 0.1,
                "strongest_comparator_spectral_icc_estimate": 0.3,
                "paired_discriminability_score": 0.6,
                "paired_discriminability_retrieval_top1": 0.5,
            }
        ]
    ).to_csv(empty_ang_dir / "baseline_repeatability_report.csv", index=False)
    (empty_ang_dir / "subspace_principal_angles.csv").write_text("\n", encoding="utf-8")

    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        art_empty = build_manuscript_artifacts(
            input_dir=empty_ang_dir,
            output_dir=tmp_path / "out_empty_angles",
            allow_synthetic_preview=True,
        )
    assert len(wrec) == 0
    rep_tex = art_empty["table_repeatability_and_angles"].read_text(encoding="utf-8")
    assert r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}X" in rep_tex


def test_optimization_vectorization_caching_and_edge_cases(tmp_path: Path) -> None:
    from dataclasses import replace

    from scipy import linalg

    from manuscript_ml.build_manuscript_artifacts import build_manuscript_artifacts
    from src.analysis.paired_lipid._eigensolvers import (
        _fast_top1_eigvalsh,
        _fast_top1_eigvalsh_from_gram,
    )
    from src.analysis.paired_lipid._gram_cache import (
        _get_dataset_gram_cache,
        _precompute_division_constants,
    )
    from src.analysis.paired_lipid.cohort import CohortDataset
    from src.analysis.paired_lipid.models import (
        CovariatePreprocessor,
        PairedMoments,
        WeightedSpectralPreprocessor,
        fit_ridge_grid,
        predict_ridge_grid,
    )
    from src.analysis.paired_lipid.nested_cv import (
        _batched_anchor_bootstrap_partial_r,
        _predict_fused_spectral_ridge,
        _residualize_and_partial_r,
    )

    # 1. Matrix-free top-1 eigenvalue matches materialized _fast_top1_eigvalsh on CPU (and GPU if available).
    rng = np.random.default_rng(3101)
    n_s, q = 36, 8
    A = rng.normal(size=(n_s, 10))
    K = 0.5 * (A @ A.T + (A @ A.T).T)
    M = rng.multinomial(n_s, np.full(n_s, 1.0 / n_s), size=6).astype(np.float64)
    M[0] = 1.0
    w_s = M / np.sum(M, axis=1, keepdims=True)
    sw = np.sqrt(w_s * (n_s / max(n_s - 1.0, 1.0)))
    k_mu = w_s @ K
    mu_sq = np.sum(w_s * k_mu, axis=1)
    Q0 = rng.normal(size=(n_s, q))
    G_mat = (K[None, :, :] - (k_mu[:, :, None] + k_mu[:, None, :]) + mu_sq[:, None, None]) * (
        sw[:, :, None] * sw[:, None, :]
    )
    ev_mat = _fast_top1_eigvalsh(np, G_mat, sw, Q0)
    ev_mf = _fast_top1_eigvalsh_from_gram(np, K, k_mu, mu_sq, sw, Q0)
    np.testing.assert_allclose(ev_mf, ev_mat, rtol=1e-12, atol=1e-12)
    if _gpu_available():
        from src.analysis.paired_lipid._backend import _get_gpu_xp, _to_np

        cp = _get_gpu_xp()
        assert cp is not None
        ev_mf_gpu = _to_np(
            cp,
            _fast_top1_eigvalsh_from_gram(
                cp, cp.asarray(K), cp.asarray(k_mu), cp.asarray(mu_sq), cp.asarray(sw), cp.asarray(Q0)
            ),
        )
        np.testing.assert_allclose(ev_mf_gpu, ev_mat, rtol=1e-12, atol=1e-12)

    # 2. WeightedSpectralPreprocessor.fit cached sample Gram matches primal path, low-rank fallback,
    #    and ignores NaNs in non-training samples.
    ds, _ = generate_synthetic_cohort(
        n_participants=40,
        n_single_sample=2,
        seed=3102,
        compact_grid=True,
        n_nuisance_modes=8,
        nuisance_part_scale=0.5,
        nuisance_sample_scale=1.0,
    )
    cfg = PairedLipidConfig()
    tr_idx = np.arange(32, dtype=int)
    w_tr = np.ones(tr_idx.size, dtype=float)
    sp_cached = WeightedSpectralPreprocessor.fit(ds, tr_idx, w_tr, cfg=cfg)
    assert "sample_gram_blocks" in ds._cache

    # Inject a NaN into an unused test sample and verify training fit is completely unaffected.
    s1300_nan = ds.sample_1300.copy()
    unused_s = int(np.flatnonzero(ds.sample_participant_idx == 39)[0])
    s1300_nan[unused_s, 0] = np.nan
    ds_nan_unused = CohortDataset(
        participant_ids=ds.participant_ids.copy(),
        outcomes={k: v.copy() for k, v in ds.outcomes.items()},
        covariate_frame=ds.covariate_frame.copy(),
        wavenumbers_1300=ds.wavenumbers_1300.copy(),
        wavenumbers_2850=ds.wavenumbers_2850.copy(),
        sample_1300=s1300_nan,
        sample_2850=ds.sample_2850.copy(),
        sample_participant_idx=ds.sample_participant_idx.copy(),
        sample_ids=ds.sample_ids.copy(),
        sample_parsed_labels=ds.sample_parsed_labels.copy(),
        participant_sample_counts=ds.participant_sample_counts.copy(),
        paired_participant_indices=ds.paired_participant_indices.copy(),
        pair_sample_a_indices=ds.pair_sample_a_indices.copy(),
        pair_sample_b_indices=ds.pair_sample_b_indices.copy(),
    )
    sp_unused_nan = WeightedSpectralPreprocessor.fit(ds_nan_unused, tr_idx, w_tr, cfg=cfg)
    np.testing.assert_allclose(sp_unused_nan.pca_singular_values, sp_cached.pca_singular_values, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sp_unused_nan.pca_components, sp_cached.pca_components, rtol=1e-12, atol=1e-12)

    # Low-rank dataset triggers primal SVD fallback cleanly even when sample_gram_blocks is cached.
    u_low = rng.normal(size=(ds.n_samples, 3))
    s1300_low = u_low @ rng.normal(size=(3, ds.sample_1300.shape[1]))
    s2850_low = u_low @ rng.normal(size=(3, ds.sample_2850.shape[1]))
    ds_low = CohortDataset(
        participant_ids=ds.participant_ids.copy(),
        outcomes={k: v.copy() for k, v in ds.outcomes.items()},
        covariate_frame=ds.covariate_frame.copy(),
        wavenumbers_1300=ds.wavenumbers_1300.copy(),
        wavenumbers_2850=ds.wavenumbers_2850.copy(),
        sample_1300=s1300_low,
        sample_2850=s2850_low,
        sample_participant_idx=ds.sample_participant_idx.copy(),
        sample_ids=ds.sample_ids.copy(),
        sample_parsed_labels=ds.sample_parsed_labels.copy(),
        participant_sample_counts=ds.participant_sample_counts.copy(),
        paired_participant_indices=ds.paired_participant_indices.copy(),
        pair_sample_a_indices=ds.pair_sample_a_indices.copy(),
        pair_sample_b_indices=ds.pair_sample_b_indices.copy(),
    )
    sp_low = WeightedSpectralPreprocessor.fit(ds_low, tr_idx, w_tr, cfg=cfg)
    assert sp_low.numerical_rank == 3
    assert sp_low.pca_components.shape[0] == 3

    # Huge constant offset triggers the primal-centering cancellation guard and preserves full accuracy.
    ds_shifted = CohortDataset(
        participant_ids=ds.participant_ids.copy(),
        outcomes={k: v.copy() for k, v in ds.outcomes.items()},
        covariate_frame=ds.covariate_frame.copy(),
        wavenumbers_1300=ds.wavenumbers_1300.copy(),
        wavenumbers_2850=ds.wavenumbers_2850.copy(),
        sample_1300=ds.sample_1300 + 1e7,
        sample_2850=ds.sample_2850 + 1e7,
        sample_participant_idx=ds.sample_participant_idx.copy(),
        sample_ids=ds.sample_ids.copy(),
        sample_parsed_labels=ds.sample_parsed_labels.copy(),
        participant_sample_counts=ds.participant_sample_counts.copy(),
        paired_participant_indices=ds.paired_participant_indices.copy(),
        pair_sample_a_indices=ds.pair_sample_a_indices.copy(),
        pair_sample_b_indices=ds.pair_sample_b_indices.copy(),
    )
    sp_shifted = WeightedSpectralPreprocessor.fit(ds_shifted, tr_idx, w_tr, cfg=cfg)
    np.testing.assert_allclose(sp_shifted.pca_singular_values, sp_cached.pca_singular_values, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(sp_shifted.pca_components, sp_cached.pca_components, rtol=1e-6, atol=1e-6)

    # 3. PairedMoments.solve_directions cached W_k factorization matches direct scipy.linalg.eigh
    #    across lambda sweeps, respects changing eigenvalue_tolerance, and fails closed on singular W.
    sample_pca = sp_cached.transform_pca(ds.sample_1300, ds.sample_2850)
    pm = PairedMoments.from_sample_coordinates(sample_pca, ds, tr_idx, w_tr)
    for k_test in (5, 10):
        W_k = pm.W_max[:k_test, :k_test]
        T_k = pm.T_max[:k_test, :k_test]
        tau_ref = float(np.trace(T_k)) / float(k_test)
        for lam in (0.001, 0.1, 10.0):
            vecs, vals, tau = pm.solve_directions(k_test, lam)
            assert tau == pytest.approx(tau_ref, rel=1e-13, abs=1e-13)
            denom = 0.5 * (W_k + W_k.T) + lam * tau_ref * np.eye(k_test)
            ref_vals, ref_vecs = linalg.eigh(T_k, denom)
            order = np.argsort(ref_vals)[::-1]
            ref_vals = ref_vals[order]
            ref_vecs = ref_vecs[:, order]
            np.testing.assert_allclose(vals, ref_vals, rtol=1e-11, atol=1e-11)
            # Check eigenvector alignment up to sign on distinct eigenvalues
            dots = np.abs(np.sum(vecs * (denom @ ref_vecs), axis=0))
            norms_ref = np.sqrt(np.sum(ref_vecs * (denom @ ref_vecs), axis=0))
            np.testing.assert_allclose(dots / norms_ref, np.ones(k_test), rtol=1e-9, atol=1e-9)
    with pytest.raises(ValueError, match="tau"):
        pm.solve_directions(5, 0.1, eigenvalue_tolerance=1e12)
    pm_sing = PairedMoments(
        W_max=np.zeros((5, 5), dtype=float),
        T_max=np.eye(5, dtype=float),
        mean_ab_shift=np.zeros(5, dtype=float),
        n_paired_multiplicity=10.0,
        n_paired_distinct=10,
    )
    with pytest.raises(linalg.LinAlgError):
        pm_sing.solve_directions(5, 0.0)

    # 4. Preset-independent spectral division cache shares K1_div across presets and upgrades cleanly to GPU.
    ds_hi, _ = generate_synthetic_cohort(
        n_participants=72,
        n_single_sample=2,
        seed=3103,
        compact_grid=True,
        n_nuisance_modes=16,
        nuisance_part_scale=0.6,
        nuisance_sample_scale=1.5,
    )
    cfg_prim = PairedLipidConfig(covariate_preset="primary")
    cfg_rf = replace(cfg_prim, covariate_preset="pre_event_risk_factors")
    tr_hi = np.arange(56, dtype=int)
    val_hi = np.arange(56, 72, dtype=int)
    gc_prim = _get_dataset_gram_cache(ds_hi, cfg_prim)
    gc_rf = _get_dataset_gram_cache(ds_hi, cfg_rf)
    dc_prim_cpu = _precompute_division_constants(np, gc_prim, ds_hi, tr_hi, val_hi, cfg_prim)
    dc_rf_cpu = _precompute_division_constants(np, gc_rf, ds_hi, tr_hi, val_hi, cfg_rf)
    assert dc_prim_cpu["K1_div"] is dc_rf_cpu["K1_div"]
    assert dc_prim_cpu["Q0_comb"] is None
    assert dc_rf_cpu["X_raw_tr"].shape[1] > dc_prim_cpu["X_raw_tr"].shape[1]
    if _gpu_available():
        from src.analysis.paired_lipid._backend import _get_gpu_xp

        cp = _get_gpu_xp()
        assert cp is not None
        dc_rf_gpu = _precompute_division_constants(cp, gc_rf, ds_hi, tr_hi, val_hi, cfg_rf)
        dc_prim_gpu = _precompute_division_constants(cp, gc_prim, ds_hi, tr_hi, val_hi, cfg_prim)
        assert dc_rf_gpu["Q0_comb"] is not None
        assert dc_prim_gpu["Q0_comb"] is not None

    # 5. Fused external spectral ridge scoring matches unfused transform + predict_ridge_grid to 1e-12.
    cov_p = CovariatePreprocessor.fit(ds.covariate_frame, tr_idx, w_tr, preset="primary", raw_cache=ds._cache)
    X_cov_tr = cov_p.transform(ds.covariate_frame, tr_idx, raw_cache=ds._cache)
    te_idx = np.arange(32, 40, dtype=int)
    X_cov_te = cov_p.transform(ds.covariate_frame, te_idx, raw_cache=ds._cache)
    y_tr = ds.outcomes["ldl_mmol_l"][tr_idx]
    part_1300 = aggregate_samples_to_participants(ds.sample_1300, ds)
    part_2850 = aggregate_samples_to_participants(ds.sample_2850, ds)
    part_pca_10 = sp_cached.transform_pca(part_1300, part_2850, k=10)
    fit_pca = fit_ridge_grid(X_cov_tr, part_pca_10[tr_idx], y_tr, w_tr, [1.0])
    pred_pca_ref, _ = predict_ridge_grid(fit_pca, X_cov_te, part_pca_10[te_idx], alpha_idx=0)
    pred_pca_fused = _predict_fused_spectral_ridge(
        sp_cached, fit_pca, X_cov_te, part_1300[te_idx], part_2850[te_idx], 10
    )
    np.testing.assert_allclose(pred_pca_fused, pred_pca_ref, rtol=1e-12, atol=1e-12)

    V_10, _, _ = pm.solve_directions(10, 0.1)
    V_r = V_10[:, :3]
    part_z = sp_cached.project_linear_subspace(part_1300, part_2850, 10, V_r)
    fit_pair = fit_ridge_grid(X_cov_tr, part_z[tr_idx], y_tr, w_tr, [1.0])
    pred_pair_ref, _ = predict_ridge_grid(fit_pair, X_cov_te, part_z[te_idx], alpha_idx=0)
    pred_pair_fused = _predict_fused_spectral_ridge(
        sp_cached, fit_pair, X_cov_te, part_1300[te_idx], part_2850[te_idx], 10, V_r
    )
    np.testing.assert_allclose(pred_pair_fused, pred_pair_ref, rtol=1e-12, atol=1e-12)

    # 6. Vectorized anchor bootstrap handles collinear, near-collinear (1e-10), rare binary, and empty edge cases.
    n_v = 24
    rare_bin = np.zeros(n_v, dtype=float)
    rare_bin[:2] = 1.0
    col_x = rng.normal(size=n_v)
    col_near = col_x + 1e-10 * rng.normal(size=n_v)
    design_edge = np.column_stack([np.ones(n_v), col_x, 2.0 * col_x, col_near, rare_bin])
    s_edge = rng.normal(size=n_v)
    y_edge = rng.normal(size=n_v)
    idx_mat = rng.integers(0, n_v, size=(120, n_v))
    eps_f = float(np.finfo(float).eps)
    vec_rs = _batched_anchor_bootstrap_partial_r(design_edge, s_edge, y_edge, idx_mat, eps_f)
    ref_rs = []
    for b_idx in idx_mat:
        r_b, _, _, dn_b, rank_b = _residualize_and_partial_r(design_edge[b_idx], s_edge[b_idx], y_edge[b_idx], eps_f)
        if n_v - rank_b >= 2 and dn_b > 1e-12:
            ref_rs.append(r_b)
    np.testing.assert_allclose(vec_rs, np.asarray(ref_rs, dtype=float), rtol=1e-11, atol=1e-11)
    assert _batched_anchor_bootstrap_partial_r(design_edge, s_edge, y_edge, np.empty((0, n_v), dtype=int), eps_f).size == 0
    assert _batched_anchor_bootstrap_partial_r(design_edge, np.ones(n_v), y_edge, idx_mat[:10], eps_f).size == 0

    # 7. build_manuscript_artifacts populates ICC column from mean_spectral_icc_a1.
    scen_dir = tmp_path / "scen_icc"
    scen_dir.mkdir()
    pd.DataFrame(
        [
            {
                "scenario": "linear_biological_signal",
                "n_participants": 92,
                "covariate_preset": "primary",
                "mean_delta_mse_paired": 0.234,
                "mean_delta_vs_strongest_comparator": 0.056,
                "mean_spectral_icc_a1": 0.812,
                "mean_recovery_r2_ldl_factor": 0.678,
            }
        ]
    ).to_csv(scen_dir / "simulation_scenarios_summary.csv", index=False)
    scen_arts = build_manuscript_artifacts(input_dir=scen_dir, output_dir=tmp_path / "scen_arts")
    scen_tex = scen_arts["table_synthetic_scenarios"].read_text(encoding="utf-8")
    assert "0.812" in scen_tex
    assert " & -- & " not in scen_tex

    # 8. Memoized synthetic cohort generator is bit-for-bit reproducible across calls and edge n_single_sample=0.
    ds_a, _ = generate_synthetic_cohort(
        n_participants=30,
        n_single_sample=0,
        seed=3104,
        compact_grid=True,
        n_nuisance_modes=16,
        nuisance_part_scale=0.5,
        nuisance_sample_scale=1.0,
        map_spatial_scale=0.3,
        map_spatial_corr=0.7,
    )
    ds_b, _ = generate_synthetic_cohort(
        n_participants=30,
        n_single_sample=0,
        seed=3104,
        compact_grid=True,
        n_nuisance_modes=16,
        nuisance_part_scale=0.5,
        nuisance_sample_scale=1.0,
        map_spatial_scale=0.3,
        map_spatial_corr=0.7,
    )
    np.testing.assert_array_equal(ds_a.sample_1300, ds_b.sample_1300)
    np.testing.assert_array_equal(ds_a.sample_2850, ds_b.sample_2850)
    np.testing.assert_array_equal(ds_a.pair_sample_a_indices, ds_b.pair_sample_a_indices)
    np.testing.assert_array_equal(ds_a.pair_sample_b_indices, ds_b.pair_sample_b_indices)
    assert ds_a.paired_participant_indices.size == 30


def test_review_hardening_numerical_stability_gpu_and_io(tmp_path: Path) -> None:
    import warnings
    from collections.abc import Mapping
    from concurrent.futures import ThreadPoolExecutor
    from typing import Any

    from scipy import linalg

    from src.analysis.io_utils import write_npz_atomic, write_text_atomic
    from src.analysis.paired_lipid import calibration as cal_mod
    from src.analysis.paired_lipid._batched_models import (
        batched_ridge_preds_single_alpha,
        batched_ridge_sse_and_preds,
    )
    from src.analysis.paired_lipid.cohort import CohortDataset
    from src.analysis.paired_lipid.models import PairedMoments, WeightedSpectralPreprocessor
    from src.analysis.paired_lipid.nested_cv import (
        _batched_anchor_bootstrap_partial_r,
        _residualize_and_partial_r,
    )

    cfg = PairedLipidConfig()

    # 1. Pre-centered sample Gram cache preserves full precision on moderate offsets (+10, +50),
    #    raises cleanly on training-sample NaN, and bypasses cache above 2048 samples.
    ds, _ = generate_synthetic_cohort(n_participants=40, n_single_sample=2, seed=3201, compact_grid=True)
    tr_idx = np.arange(32, dtype=int)
    w_tr = np.ones(tr_idx.size, dtype=float)
    sp_ref = WeightedSpectralPreprocessor.fit(ds, tr_idx, w_tr, cfg=cfg)
    for offset in (10.0, 50.0, 1e4):
        ds_off = CohortDataset(
            participant_ids=ds.participant_ids.copy(),
            outcomes={k: v.copy() for k, v in ds.outcomes.items()},
            covariate_frame=ds.covariate_frame.copy(),
            wavenumbers_1300=ds.wavenumbers_1300.copy(),
            wavenumbers_2850=ds.wavenumbers_2850.copy(),
            sample_1300=ds.sample_1300 + offset,
            sample_2850=ds.sample_2850 + offset,
            sample_participant_idx=ds.sample_participant_idx.copy(),
            sample_ids=ds.sample_ids.copy(),
            sample_parsed_labels=ds.sample_parsed_labels.copy(),
            participant_sample_counts=ds.participant_sample_counts.copy(),
            paired_participant_indices=ds.paired_participant_indices.copy(),
            pair_sample_a_indices=ds.pair_sample_a_indices.copy(),
            pair_sample_b_indices=ds.pair_sample_b_indices.copy(),
        )
        sp_off = WeightedSpectralPreprocessor.fit(ds_off, tr_idx, w_tr, cfg=cfg)
        np.testing.assert_allclose(sp_off.pca_singular_values, sp_ref.pca_singular_values, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(
            np.abs(sp_off.pca_components), np.abs(sp_ref.pca_components), rtol=1e-9, atol=1e-9
        )

    s1300_nan_tr = ds.sample_1300.copy()
    s1300_nan_tr[0, 0] = np.nan
    ds_nan_tr = CohortDataset(
        participant_ids=ds.participant_ids.copy(),
        outcomes={k: v.copy() for k, v in ds.outcomes.items()},
        covariate_frame=ds.covariate_frame.copy(),
        wavenumbers_1300=ds.wavenumbers_1300.copy(),
        wavenumbers_2850=ds.wavenumbers_2850.copy(),
        sample_1300=s1300_nan_tr,
        sample_2850=ds.sample_2850.copy(),
        sample_participant_idx=ds.sample_participant_idx.copy(),
        sample_ids=ds.sample_ids.copy(),
        sample_parsed_labels=ds.sample_parsed_labels.copy(),
        participant_sample_counts=ds.participant_sample_counts.copy(),
        paired_participant_indices=ds.paired_participant_indices.copy(),
        pair_sample_a_indices=ds.pair_sample_a_indices.copy(),
        pair_sample_b_indices=ds.pair_sample_b_indices.copy(),
    )
    with pytest.raises(ValueError, match="Non-finite"):
        WeightedSpectralPreprocessor.fit(ds_nan_tr, tr_idx, w_tr, cfg=cfg)

    ds_large, _ = generate_synthetic_cohort(n_participants=1050, n_single_sample=0, seed=3202, compact_grid=True)
    assert ds_large.n_samples == 2100
    sp_large = WeightedSpectralPreprocessor.fit(ds_large, np.arange(80, dtype=int), np.ones(80, dtype=float), cfg=cfg)
    assert "sample_gram_blocks" not in ds_large._cache
    assert sp_large.pca_singular_values.size > 0

    # 2. PairedMoments.solve_directions is scale-invariant on tiny PSD matrices and raises ValueError on NaN lambda.
    W_small = 1e-15 * np.eye(6, dtype=float)
    T_small = 2e-15 * np.eye(6, dtype=float)
    pm_small = PairedMoments(
        W_max=W_small,
        T_max=T_small,
        mean_ab_shift=np.zeros(6, dtype=float),
        n_paired_multiplicity=10.0,
        n_paired_distinct=10,
    )
    ev_ref, _ = linalg.eigh(T_small, W_small + 0.1 * (np.trace(T_small) / 6.0) * np.eye(6, dtype=float))
    _, ev_small, _ = pm_small.solve_directions(6, 0.1, eigenvalue_tolerance=0.0)
    np.testing.assert_allclose(ev_small, ev_ref[::-1], rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="Non-finite"):
        pm_small.solve_directions(6, float("nan"), eigenvalue_tolerance=0.0)

    # 3. _batched_anchor_bootstrap_partial_r guards normal-equation amplification (kappa ~ 1e5,
    #    large signal projection), handles p == 0 empty design, and chunks across > 512 resamples.
    rng = np.random.default_rng(3203)
    n_v = 40
    eps_f = float(np.finfo(float).eps)
    x1 = rng.normal(size=n_v)
    x2 = x1 + 1e-5 * rng.normal(size=n_v)
    design_ill = np.column_stack([np.ones(n_v), x1, x2])
    s_ill = 1e5 * x1 + 1e5 * (x2 - x1) / 1e-5 + rng.normal(scale=1e-2, size=n_v)
    y_ill = 1e5 * x1 - 1e5 * (x2 - x1) / 1e-5 + rng.normal(scale=1e-2, size=n_v)
    idx_ill = rng.integers(0, n_v, size=(560, n_v))
    vec_ill = _batched_anchor_bootstrap_partial_r(design_ill, s_ill, y_ill, idx_ill, eps_f)
    ref_ill = np.asarray(
        [
            r_b
            for b_idx in idx_ill
            for r_b, _, _, dn_b, rank_b in [
                _residualize_and_partial_r(design_ill[b_idx], s_ill[b_idx], y_ill[b_idx], eps_f)
            ]
            if n_v - rank_b >= 2 and dn_b > 1e-12
        ],
        dtype=float,
    )
    np.testing.assert_allclose(vec_ill, ref_ill, rtol=1e-11, atol=1e-11)

    d_zero_cols = np.empty((n_v, 0), dtype=float)
    vec_p0 = _batched_anchor_bootstrap_partial_r(d_zero_cols, s_ill, y_ill, idx_ill[:24], eps_f)
    ref_p0 = np.asarray(
        [_residualize_and_partial_r(d_zero_cols[b], s_ill[b], y_ill[b], eps_f)[0] for b in idx_ill[:24]],
        dtype=float,
    )
    np.testing.assert_allclose(vec_p0, ref_p0, rtol=1e-12, atol=1e-12)

    # 4. batched_ridge_preds_single_alpha matches multi-alpha ridge on CPU/GPU and handles collinear alpha=0 cleanly.
    B_r, n_tr_r, n_val_r, p_cov_r, p_z_r = 12, 36, 10, 4, 15
    X_cov_tr = rng.normal(size=(B_r, n_tr_r, p_cov_r))
    X_cov_val = rng.normal(size=(B_r, n_val_r, p_cov_r))
    Z_tr = rng.normal(size=(n_tr_r, p_z_r))
    Z_val = rng.normal(size=(n_val_r, p_z_r))
    y_tr_r = rng.normal(size=n_tr_r)
    y_val_r = rng.normal(size=n_val_r)
    w_tr_r = np.full((B_r, n_tr_r), 1.0 / n_tr_r)
    w_val_r = np.full((B_r, n_val_r), 1.0 / n_val_r)
    alphas_r = np.array([1e-3, 1e-1, 1.0, 10.0, 100.0])
    sel_a = rng.integers(0, alphas_r.size, size=B_r)
    Z_col = Z_tr.copy()
    Z_col[:, -1] = Z_col[:, 0]
    alphas_col = np.where(np.arange(B_r) == 0, 0.0, 1.0)

    backends: list[tuple[str, Any]] = [("cpu", np)]
    if _gpu_available():
        from src.analysis.paired_lipid._backend import _get_gpu_xp

        cp_mod = _get_gpu_xp()
        assert cp_mod is not None
        backends.append(("gpu", cp_mod))

    for _, xp in backends:
        _, pr_all = batched_ridge_sse_and_preds(
            xp,
            xp.asarray(X_cov_tr),
            xp.asarray(X_cov_val),
            xp.asarray(Z_tr),
            xp.asarray(Z_val),
            xp.asarray(y_tr_r),
            xp.asarray(y_val_r),
            xp.asarray(w_tr_r),
            xp.asarray(w_val_r),
            xp.asarray(alphas_r),
        )
        pr_ref = pr_all[xp.arange(B_r), :, xp.asarray(sel_a)]
        pr_one = batched_ridge_preds_single_alpha(
            xp,
            xp.asarray(X_cov_tr),
            xp.asarray(X_cov_val),
            xp.asarray(Z_tr),
            xp.asarray(Z_val),
            xp.asarray(y_tr_r),
            xp.asarray(w_tr_r),
            xp.asarray(alphas_r[sel_a]),
        )
        np.testing.assert_allclose(
            np.asarray(xp.asnumpy(pr_one) if xp is not np else pr_one),
            np.asarray(xp.asnumpy(pr_ref) if xp is not np else pr_ref),
            rtol=1e-12,
            atol=1e-12,
        )
        with warnings.catch_warnings(record=True) as wrec:
            warnings.simplefilter("always")
            pr_col = batched_ridge_preds_single_alpha(
                xp,
                xp.asarray(X_cov_tr),
                xp.asarray(X_cov_val),
                xp.asarray(Z_col),
                xp.asarray(Z_val),
                xp.asarray(y_tr_r),
                xp.asarray(w_tr_r),
                xp.asarray(alphas_col),
            )
        assert len(wrec) == 0
        assert bool(np.all(np.isfinite(xp.asnumpy(pr_col) if xp is not np else pr_col)))

    # 5. Adaptive GPU CheFSI continuation and fused Gram assembly match CPU on 16-nuisance synthetic cohort.
    if _gpu_available():
        ds_nuis, _ = generate_synthetic_cohort(
            n_participants=92,
            n_single_sample=2,
            seed=3204,
            compact_grid=True,
            n_nuisance_modes=16,
            nuisance_part_scale=0.5,
            nuisance_sample_scale=1.0,
        )
        b_cpu = run_multiplicity_bootstrap(ds_nuis, cfg=cfg, n_bootstraps=12, bootstrap_seed=3205, use_gpu=False)
        b_gpu = run_multiplicity_bootstrap(ds_nuis, cfg=cfg, n_bootstraps=12, bootstrap_seed=3205, use_gpu=True)
        np.testing.assert_allclose(
            b_gpu["bootstrap_draws_delta_mse"], b_cpu["bootstrap_draws_delta_mse"], rtol=1e-11, atol=1e-11
        )
        np.testing.assert_allclose(
            b_gpu["bootstrap_draws_vs_comparator"], b_cpu["bootstrap_draws_vs_comparator"], rtol=1e-11, atol=1e-11
        )

    # 6. Thread-safe atomic writers leave no stray temporary files and preserve integrity.
    io_dir = tmp_path / "atomic_threads"
    io_dir.mkdir()
    npz_target = io_dir / "shared.npz"
    txt_target = io_dir / "shared.txt"

    def _write_worker(idx: int) -> None:
        write_npz_atomic(npz_target, {"arr": np.full(8, float(idx))})
        write_text_atomic(txt_target, f"worker-{idx}\n")

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(_write_worker, range(16)))
    assert npz_target.is_file() and txt_target.is_file()
    assert not list(io_dir.glob("*.tmp*"))
    with np.load(npz_target, allow_pickle=False) as loaded:
        assert loaded["arr"].shape == (8,)

    # 7. Multi-worker calibration checkpoint crash/resume matches unbroken run exactly.
    ckpt_ref = tmp_path / "cal_ref.npz"
    ckpt_resume = tmp_path / "cal_resume.npz"
    res_ref = cal_mod.evaluate_bootstrap_calibration(
        n_repetitions=4,
        n_bootstraps=60,
        n_participants=60,
        n_test_participants=120,
        n_jobs=2,
        use_gpu=False,
        compact_grid=True,
        checkpoint_path=ckpt_ref,
        checkpoint_chunk_size=2,
    )
    orig_save = cal_mod._save_calibration_checkpoint
    call_counter = {"n": 0}

    def _fail_second_save(path: Path, state: Mapping[str, Any]) -> None:
        call_counter["n"] += 1
        if call_counter["n"] == 2:
            raise RuntimeError("simulated checkpoint failure")
        orig_save(path, state)

    cal_mod._save_calibration_checkpoint = _fail_second_save  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match="simulated checkpoint failure"):
            cal_mod.evaluate_bootstrap_calibration(
                n_repetitions=4,
                n_bootstraps=60,
                n_participants=60,
                n_test_participants=120,
                n_jobs=2,
                use_gpu=False,
                compact_grid=True,
                checkpoint_path=ckpt_resume,
                checkpoint_chunk_size=2,
            )
    finally:
        cal_mod._save_calibration_checkpoint = orig_save

    res_resumed = cal_mod.evaluate_bootstrap_calibration(
        n_repetitions=4,
        n_bootstraps=60,
        n_participants=60,
        n_test_participants=120,
        n_jobs=2,
        use_gpu=False,
        compact_grid=True,
        checkpoint_path=ckpt_resume,
        checkpoint_chunk_size=2,
    )
    for key in (
        "target_delta_mse",
        "mean_point_estimate",
        "unconditional_coverage",
        "mean_interval_width",
        "target_delta_vs_comparator",
    ):
        np.testing.assert_allclose(res_resumed[key], res_ref[key], rtol=0.0, atol=1e-14)


def test_missing_covariates_and_single_sample_batched_parity() -> None:
    from dataclasses import replace

    from src.analysis.paired_lipid._batched_models import _prep_covariates_batch
    from src.analysis.paired_lipid._gram_cache import (
        _get_dataset_gram_cache,
        _precompute_division_constants,
    )
    from src.analysis.paired_lipid.models import CovariatePreprocessor, WeightedSpectralPreprocessor
    from src.analysis.paired_lipid.nested_cv import (
        _batched_anchor_bootstrap_partial_r_missing,
        _residualize_and_partial_r,
        _select_inner_fast,
        _select_inner_sequential,
        make_nested_splits,
    )

    cfg = PairedLipidConfig()

    # 1. Missing covariates: zero fallback draws on CPU & GPU, exact parity vs sequential evaluate_nested_cohort.
    ds_miss, _ = generate_synthetic_cohort(
        n_participants=92,
        n_single_sample=4,
        seed=4101,
        compact_grid=True,
        covariate_missing_rate=0.18,
    )
    splits_m = make_nested_splits(ds_miss, cfg, cfg.primary_seed)
    boot_cpu = run_multiplicity_bootstrap(
        ds_miss, cfg=cfg, n_bootstraps=12, bootstrap_seed=4102, use_gpu=False
    )
    assert boot_cpu["execution"]["device"] == "cpu"
    assert boot_cpu["n_fallback_draws"] == 0

    rng_m = np.random.default_rng(4102)
    M_m = np.zeros((12, ds_miss.n_participants), dtype=float)
    dm = rng_m.integers(0, ds_miss.n_participants, size=(12, ds_miss.n_participants))
    np.add.at(M_m, (np.arange(12)[:, None], dm), 1.0)

    ev_base_m = evaluate_nested_cohort(ds_miss, cfg=cfg, precomputed_splits=splits_m, compute_diagnostics=False)
    assert boot_cpu["point_estimate_delta_mse"] == pytest.approx(
        ev_base_m["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"], abs=1e-10
    )
    assert boot_cpu["point_estimate_vs_comparator"] == pytest.approx(
        ev_base_m["paired_vs_strongest_comparator"]["delta_mse"], abs=1e-10
    )
    for b in range(4):
        seq_ev = evaluate_nested_cohort(
            ds_miss,
            cfg=cfg,
            participant_multiplicities=M_m[b],
            precomputed_splits=splits_m,
            compute_diagnostics=False,
        )
        assert boot_cpu["bootstrap_draws_delta_mse"][b] == pytest.approx(
            seq_ev["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"], abs=1e-10
        )
        assert boot_cpu["bootstrap_draws_vs_comparator"][b] == pytest.approx(
            seq_ev["paired_vs_strongest_comparator"]["delta_mse"], abs=1e-10
        )

    # _select_inner_fast vs _select_inner_sequential parity on missing covariates (including dropped_covariates).
    tr0, te0 = splits_m[0][0]
    in0 = splits_m[1][0]
    w_all = np.ones(ds_miss.n_participants, dtype=float)
    outer_sp = WeightedSpectralPreprocessor.fit(ds_miss, tr0, w_all[tr0], cfg=cfg)
    sel_fast = _select_inner_fast(ds_miss, ds_miss.outcomes[cfg.primary_endpoint], tr0, in0, w_all, cfg)
    sel_seq = _select_inner_sequential(
        ds_miss,
        ds_miss.outcomes[cfg.primary_endpoint],
        tr0,
        in0,
        w_all,
        cfg,
        outer_sp,
        single_sample_downstream=False,
        include_band_paired=False,
    )
    assert sel_fast is not None
    assert sel_fast.paired_setting == sel_seq.paired_setting
    assert sel_fast.pca_setting == sel_seq.pca_setting
    assert sel_fast.pls_components == sel_seq.pls_components
    assert sel_fast.inner_dropped_covariates == sel_seq.inner_dropped_covariates
    assert sel_fast.paired_mse == pytest.approx(sel_seq.paired_mse, abs=1e-10)

    if _gpu_available():
        boot_gpu = run_multiplicity_bootstrap(
            ds_miss, cfg=cfg, n_bootstraps=12, bootstrap_seed=4102, use_gpu=True
        )
        assert boot_gpu["execution"]["device"] == "gpu"
        assert boot_gpu["n_fallback_draws"] == 0
        np.testing.assert_allclose(
            boot_gpu["bootstrap_draws_delta_mse"], boot_cpu["bootstrap_draws_delta_mse"], rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            boot_gpu["bootstrap_draws_vs_comparator"],
            boot_cpu["bootstrap_draws_vs_comparator"],
            rtol=1e-10,
            atol=1e-10,
        )

    # 2. Missing-covariate edge cases: constant+NaN, validation-only NaN, all-NaN col, all-active-NaN draw, 50/50 median.
    cov_edge = pd.DataFrame(
        {
            "age_years": [60.0, 65.0, 70.0, 75.0, 80.0, 85.0, np.nan, 72.0],
            "sex": ["female", "male", "female", "male", "female", "male", "female", "male"],
            "statin": [1.0, 1.0, np.nan, 1.0, np.nan, 1.0, 1.0, np.nan],
            "sample_label_trend": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 5.0, 6.0],
        }
    )
    # Put validation-only NaN in age_years (row 6) and all-NaN in sample_label_trend across tr_p (rows 0..5).
    tr_e = np.arange(6, dtype=int)
    val_e = np.arange(6, 8, dtype=int)
    raw_edge, cols_edge = CovariatePreprocessor.raw_design_matrix(cov_edge, preset="primary")
    ds_edge, _ = generate_synthetic_cohort(n_participants=8, n_single_sample=0, seed=4103, compact_grid=True)
    ds_edge_mod = replace(ds_edge, covariate_frame=cov_edge)
    gc_e = _get_dataset_gram_cache(ds_edge_mod, cfg)
    dc_e = _precompute_division_constants(np, gc_e, ds_edge_mod, tr_e, val_e, cfg)
    # Draw 0: exact 50/50 split on sex (3 female, 3 male), statin constant+NaN; Draw 1: only non-missing statin rows active;
    # Draw 2: only missing statin rows active (all-active-NaN for statin!).
    M_tr_e = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 2.0, 0.0, 3.0, 0.0],
        ],
        dtype=np.float64,
    )
    M_val_e = np.ones((3, 2), dtype=np.float64)
    _, _, _, X_tr_b, X_val_b, valid_c_b = _prep_covariates_batch(np, dc_e, M_tr_e, M_val_e, return_valid_mask=True)
    for b in range(3):
        cp_ref = CovariatePreprocessor.fit(cov_edge, tr_e, M_tr_e[b], preset="primary")
        X_tr_ref = cp_ref.transform(cov_edge, tr_e)
        X_val_ref = cp_ref.transform(cov_edge, val_e)
        # Non-zero columns in batched output must match CovariatePreprocessor.transform identically.
        nz_cols = np.any((X_tr_b[b] != 0.0), axis=0) | np.any((X_val_b[b] != 0.0), axis=0)
        np.testing.assert_allclose(X_tr_b[b][:, nz_cols], X_tr_ref, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(X_val_b[b][:, nz_cols], X_val_ref, rtol=1e-12, atol=1e-12)
        dropped_b = set(dc_e["dropped_cov_cols_base"]) | {
            col for col, ok in zip(dc_e["kept_cov_cols"], valid_c_b[b]) if not ok
        }
        assert dropped_b == set(cp_ref.dropped_columns)
    assert raw_edge.shape[1] == len(cols_edge)

    # Collinear missingness indicators (quadratic_age_and_label with missing age_years).
    cfg_quad = replace(cfg, covariate_preset="quadratic_age_and_label")
    boot_quad = run_multiplicity_bootstrap(
        ds_miss, cfg=cfg_quad, n_bootstraps=6, bootstrap_seed=4104, use_gpu=False
    )
    assert boot_quad["n_fallback_draws"] == 0
    ev_quad = evaluate_nested_cohort(ds_miss, cfg=cfg_quad, compute_diagnostics=False)
    assert boot_quad["point_estimate_delta_mse"] == pytest.approx(
        ev_quad["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"], abs=1e-10
    )

    # 3. single_sample_downstream (both subset_mode="single_sample" and combined with missing covariates).
    ds_ss, _ = generate_synthetic_cohort(
        n_participants=92,
        n_single_sample=4,
        seed=4105,
        compact_grid=True,
        subset_mode="single_sample",
    )
    splits_s = make_nested_splits(ds_ss, cfg, cfg.primary_seed)
    boot_ss_cpu = run_multiplicity_bootstrap(
        ds_ss, cfg=cfg, n_bootstraps=12, bootstrap_seed=4106, use_gpu=False
    )
    assert boot_ss_cpu["execution"]["device"] == "cpu"
    assert boot_ss_cpu["n_fallback_draws"] == 0
    ev_ss = evaluate_nested_cohort(ds_ss, cfg=cfg, precomputed_splits=splits_s, compute_diagnostics=False)
    assert boot_ss_cpu["point_estimate_delta_mse"] == pytest.approx(
        ev_ss["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"], abs=1e-10
    )
    assert boot_ss_cpu["point_estimate_vs_comparator"] == pytest.approx(
        ev_ss["paired_vs_strongest_comparator"]["delta_mse"], abs=1e-10
    )

    rng_s = np.random.default_rng(4106)
    M_s = np.zeros((12, ds_ss.n_participants), dtype=float)
    dm_s = rng_s.integers(0, ds_ss.n_participants, size=(12, ds_ss.n_participants))
    np.add.at(M_s, (np.arange(12)[:, None], dm_s), 1.0)
    for b in range(4):
        seq_ss = evaluate_nested_cohort(
            ds_ss,
            cfg=cfg,
            participant_multiplicities=M_s[b],
            precomputed_splits=splits_s,
            single_sample_downstream=True,
            compute_diagnostics=False,
        )
        assert boot_ss_cpu["bootstrap_draws_delta_mse"][b] == pytest.approx(
            seq_ss["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"], abs=1e-10
        )
        assert boot_ss_cpu["bootstrap_draws_vs_comparator"][b] == pytest.approx(
            seq_ss["paired_vs_strongest_comparator"]["delta_mse"], abs=1e-10
        )

    if _gpu_available():
        boot_ss_gpu = run_multiplicity_bootstrap(
            ds_ss, cfg=cfg, n_bootstraps=12, bootstrap_seed=4106, use_gpu=True
        )
        assert boot_ss_gpu["execution"]["device"] == "gpu"
        assert boot_ss_gpu["n_fallback_draws"] == 0
        np.testing.assert_allclose(
            boot_ss_gpu["bootstrap_draws_delta_mse"],
            boot_ss_cpu["bootstrap_draws_delta_mse"],
            rtol=1e-10,
            atol=1e-10,
        )

    # Combined single_sample_downstream + missing covariates simultaneously.
    ds_combo, _ = generate_synthetic_cohort(
        n_participants=92,
        n_single_sample=4,
        seed=4107,
        compact_grid=True,
        subset_mode="single_sample",
        covariate_missing_rate=0.15,
    )
    boot_combo = run_multiplicity_bootstrap(
        ds_combo, cfg=cfg, n_bootstraps=8, bootstrap_seed=4108, use_gpu=False
    )
    assert boot_combo["n_fallback_draws"] == 0
    ev_combo = evaluate_nested_cohort(ds_combo, cfg=cfg, compute_diagnostics=False)
    assert boot_combo["point_estimate_delta_mse"] == pytest.approx(
        ev_combo["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"], abs=1e-10
    )

    # 4. Anchor missing-covariate vectorized bootstrap parity (primary, collinear quadratic, all-NaN, empty).
    eps_f = float(np.finfo(float).eps)
    s_anc = ev_base_m["oof_spectral_contributions"]["paired_linear_ridge"]
    y_anc = ds_miss.outcomes["ldl_mmol_l"]
    idx_anc = np.random.default_rng(4109).integers(0, ds_miss.n_participants, size=(60, ds_miss.n_participants))
    for preset_name in ("primary", "quadratic_age_and_label"):
        raw_p, _ = CovariatePreprocessor.raw_design_matrix(
            ds_miss.covariate_frame, preset=preset_name, raw_cache=ds_miss._cache
        )
        vec_r = _batched_anchor_bootstrap_partial_r_missing(raw_p, s_anc, y_anc, idx_anc, eps_f)
        ref_r: list[float] = []
        valid_i = np.arange(ds_miss.n_participants, dtype=int)
        for b_idx in idx_anc:
            bm = np.bincount(b_idx, minlength=ds_miss.n_participants).astype(float)
            cp_b = CovariatePreprocessor.fit(
                ds_miss.covariate_frame, valid_i, bm, preset=preset_name, raw_cache=ds_miss._cache
            )
            Xb = cp_b.transform(ds_miss.covariate_frame, b_idx, raw_cache=ds_miss._cache)
            db = np.column_stack([np.ones(ds_miss.n_participants), Xb])
            rb, _, _, dnb, rkb = _residualize_and_partial_r(db, s_anc[b_idx], y_anc[b_idx], eps_f)
            if ds_miss.n_participants - rkb >= 2 and dnb > 1e-12:
                ref_r.append(rb)
        np.testing.assert_allclose(vec_r, np.asarray(ref_r), rtol=1e-11, atol=1e-11)

    all_nan_cov = np.full((ds_miss.n_participants, 2), np.nan, dtype=float)
    vec_all_nan = _batched_anchor_bootstrap_partial_r_missing(all_nan_cov, s_anc, y_anc, idx_anc[:10], eps_f)
    assert vec_all_nan.size == 10 and np.all(np.isfinite(vec_all_nan))
    assert (
        _batched_anchor_bootstrap_partial_r_missing(
            all_nan_cov, s_anc, y_anc, np.empty((0, ds_miss.n_participants), dtype=int), eps_f
        ).size
        == 0
    )


def test_batched_pls_early_stop_subset_shift_fallback_and_ipc_blas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pickle
    from dataclasses import replace
    from typing import Any

    import src.analysis.paired_lipid.calibration as cal_mod
    import src.analysis.paired_lipid.nested_cv as ncv_mod
    from src.analysis.paired_lipid._backend import _get_threadpool_controller
    from src.analysis.paired_lipid._batched_models import _prep_mfa_pca_batch, batched_pls_sse_and_preds
    from src.analysis.paired_lipid._gram_cache import (
        _get_dataset_gram_cache,
        _precompute_division_constants,
    )
    from src.analysis.paired_lipid.models import WeightedPLSResult
    from src.analysis.paired_lipid.nested_cv import PreparedExternalTest, score_external_primary_pair

    # 1. batched_pls_sse_and_preds freezes components on w_norm <= 1e-12 and t_sq <= 1e-12 just like WeightedPLSResult.fit.
    rng = np.random.default_rng(4201)
    n_tr, n_val, p_cov, p_spec = 24, 8, 2, 10
    X_cov_tr_0 = rng.normal(size=(n_tr, p_cov))
    X_cov_val_0 = rng.normal(size=(n_val, p_cov))
    X1_tr = rng.normal(size=(n_tr, p_spec))
    X1_val = rng.normal(size=(n_val, p_spec))
    # Construct rank-1 spectral block (plus tiny 1e-10 residual direction for t_sq <= 1e-12).
    u_tr = rng.normal(size=(n_tr, 1))
    u_val = rng.normal(size=(n_val, 1))
    v_dir = rng.normal(size=(1, p_spec))
    v_tiny = rng.normal(size=(1, p_spec))
    v_tiny /= np.linalg.norm(v_tiny)
    X1_rank1_tr = u_tr @ v_dir + 1e-10 * (rng.normal(size=(n_tr, 1)) @ v_tiny)
    X1_rank1_val = u_val @ v_dir + 1e-10 * (rng.normal(size=(n_val, 1)) @ v_tiny)

    w_tr_1d = np.ones(n_tr, dtype=float) / float(n_tr)
    w_val_1d = np.ones(n_val, dtype=float)
    y_const = np.full(n_tr, 3.25, dtype=float)
    y_const_val = np.full(n_val, 3.25, dtype=float)
    # Exact rank-1 target (w_norm hits 0 after component 1).
    y_rank1_tr = 2.0 * u_tr[:, 0] + 1.5
    y_rank1_val = 2.0 * u_val[:, 0] + 1.5
    # Large target on near-rank-1 X with tiny 1e-10 second direction (triggers t_sq <= 1e-12 after component 1).
    y_tsq_tr = 1e5 * rng.normal(size=n_tr)
    y_tsq_val = 1e5 * rng.normal(size=n_val)

    backends: list[Any] = [np]
    if _gpu_available():
        from src.analysis.paired_lipid._backend import _get_gpu_xp

        cp_mod = _get_gpu_xp()
        assert cp_mod is not None
        backends.append(cp_mod)

    for xp in backends:
        for case_name, Xc_tr, Xc_val, Xs_tr, Xs_val, yt, yv in (
            ("const_y", X_cov_tr_0, X_cov_val_0, X1_tr, X1_val, y_const, y_const_val),
            (
                "rank1_exact",
                np.zeros((n_tr, 0)),
                np.zeros((n_val, 0)),
                u_tr @ v_dir,
                u_val @ v_dir,
                y_rank1_tr,
                y_rank1_val,
            ),
            (
                "tiny_t_sq",
                np.zeros((n_tr, 0)),
                np.zeros((n_val, 0)),
                X1_rank1_tr,
                X1_rank1_val,
                y_tsq_tr,
                y_tsq_val,
            ),
        ):
            ref_pls = WeightedPLSResult.fit(Xc_tr, Xs_tr, yt, np.ones(n_tr), max_components=5)
            ref_preds, _ = ref_pls.predict(Xc_val, Xs_val)
            K1_tr = Xs_tr @ Xs_tr.T
            K1_vt = Xs_val @ Xs_tr.T
            div = {
                "w_tr": xp.asarray(w_tr_1d[None, :], dtype=xp.float64),
                "w_val": xp.asarray(w_val_1d[None, :], dtype=xp.float64),
                "y_tr": xp.asarray(yt, dtype=xp.float64),
                "y_val": xp.asarray(yv, dtype=xp.float64),
                "X_cov_tr": xp.asarray(Xc_tr[None, :, :], dtype=xp.float64),
                "X_cov_val": xp.asarray(Xc_val[None, :, :], dtype=xp.float64),
                "scale1_sq": xp.asarray([1.0], dtype=xp.float64),
                "scale2_sq": xp.asarray([0.0], dtype=xp.float64),
                "K_p1_tr": xp.asarray(K1_tr, dtype=xp.float64),
                "K_p1_vt": xp.asarray(K1_vt, dtype=xp.float64),
                "K_p2_tr": xp.zeros_like(xp.asarray(K1_tr, dtype=xp.float64)),
                "K_p2_vt": xp.zeros_like(xp.asarray(K1_vt, dtype=xp.float64)),
            }
            _, batch_preds = batched_pls_sse_and_preds(xp, div, max_comp=5)
            bp_np = np.asarray(xp.asnumpy(batch_preds) if xp is not np else batch_preds)[0]
            np.testing.assert_allclose(bp_np, ref_preds, rtol=1e-10, atol=1e-10, err_msg=case_name)
            if case_name == "const_y":
                np.testing.assert_allclose(bp_np, float(np.mean(yt)), rtol=0.0, atol=1e-12)
            else:
                # Subsequent components must stay frozen at component 1.
                for c in range(1, 5):
                    np.testing.assert_allclose(bp_np[:, c], bp_np[:, 0], rtol=0.0, atol=1e-12)

        # Mixed batch: draw 0 active on all rows (varying y), draw 1 active only on rows with constant y.
        y_mix_tr = rng.normal(size=n_tr)
        y_mix_tr[:6] = 5.0
        y_mix_val = rng.normal(size=n_val)
        w_mix = np.zeros((2, n_tr), dtype=float)
        w_mix[0] = 1.0 / float(n_tr)
        w_mix[1, :6] = 1.0 / 6.0
        K1_mix_tr = X1_tr @ X1_tr.T
        K1_mix_vt = X1_val @ X1_tr.T
        div_mix = {
            "w_tr": xp.asarray(w_mix, dtype=xp.float64),
            "w_val": xp.asarray(np.vstack([w_val_1d, w_val_1d]), dtype=xp.float64),
            "y_tr": xp.asarray(y_mix_tr, dtype=xp.float64),
            "y_val": xp.asarray(y_mix_val, dtype=xp.float64),
            "X_cov_tr": xp.asarray(np.stack([X_cov_tr_0, X_cov_tr_0], axis=0), dtype=xp.float64),
            "X_cov_val": xp.asarray(np.stack([X_cov_val_0, X_cov_val_0], axis=0), dtype=xp.float64),
            "scale1_sq": xp.asarray([1.0, 1.0], dtype=xp.float64),
            "scale2_sq": xp.asarray([0.0, 0.0], dtype=xp.float64),
            "K_p1_tr": xp.asarray(K1_mix_tr, dtype=xp.float64),
            "K_p1_vt": xp.asarray(K1_mix_vt, dtype=xp.float64),
            "K_p2_tr": xp.zeros_like(xp.asarray(K1_mix_tr, dtype=xp.float64)),
            "K_p2_vt": xp.zeros_like(xp.asarray(K1_mix_vt, dtype=xp.float64)),
        }
        _, mix_preds = batched_pls_sse_and_preds(xp, div_mix, max_comp=5)
        mix_np = np.asarray(xp.asnumpy(mix_preds) if xp is not np else mix_preds)
        ref_0, _ = WeightedPLSResult.fit(X_cov_tr_0, X1_tr, y_mix_tr, w_mix[0], max_components=5).predict(
            X_cov_val_0, X1_val
        )
        np.testing.assert_allclose(mix_np[0], ref_0, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(mix_np[1], 5.0, rtol=0.0, atol=1e-12)

    # 2. _prep_mfa_pca_batch marks rank_ok=False when subset-mean shift risks Gram cancellation (mu1_sq > 1e2 * tr(G1)).
    cfg = PairedLipidConfig()
    ds_shift, _ = generate_synthetic_cohort(n_participants=50, n_single_sample=2, seed=4202, compact_grid=True)
    # Shift one subset of participants in sample_1300 by +1e4 so the remaining subset has huge mean shift
    # relative to the full-cohort centered Gram matrix while maintaining O(1) internal variance.
    s1300_shift = ds_shift.sample_1300.copy()
    hi_mask = ds_shift.sample_participant_idx >= 42
    s1300_shift[hi_mask] += 1e4
    ds_shifted = replace(ds_shift, sample_1300=s1300_shift)
    tr_lo = np.arange(40, dtype=int)
    val_hi = np.arange(40, 50, dtype=int)
    gc_sh = _get_dataset_gram_cache(ds_shifted, cfg)
    dc_sh = _precompute_division_constants(np, gc_sh, ds_shifted, tr_lo, val_hi, cfg)
    mfa_sh = _prep_mfa_pca_batch(
        np, gc_sh, dc_sh, np.ones((1, tr_lo.size), dtype=np.float64), np.array([[float(tr_lo.size)]]), cfg
    )
    assert bool(mfa_sh["rank_ok"][0]) is False

    # End-to-end bootstrap routes shifted draws to primal fallback and matches evaluate_nested_cohort.
    boot_sh = run_multiplicity_bootstrap(ds_shifted, cfg=cfg, n_bootstraps=3, bootstrap_seed=4203, use_gpu=False)
    ev_sh = evaluate_nested_cohort(ds_shifted, cfg=cfg, compute_diagnostics=False)
    assert boot_sh["point_estimate_delta_mse"] == pytest.approx(
        ev_sh["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"], abs=1e-10
    )

    # 3. PreparedExternalTest drops raw_cov_frame when preset is given, reduces pickle size, and fails closed on mismatch.
    ds_tr, _ = generate_synthetic_cohort(n_participants=40, n_single_sample=2, seed=4204, compact_grid=True)
    ds_te, _ = generate_synthetic_cohort(n_participants=600, n_single_sample=10, seed=4205, compact_grid=True)
    prep_preset = PreparedExternalTest.from_dataset(ds_te, cfg.primary_endpoint, preset=cfg.covariate_preset)
    prep_full = PreparedExternalTest.from_dataset(ds_te, cfg.primary_endpoint, preset=None)
    assert prep_preset.raw_cov_frame is None
    assert prep_full.raw_cov_frame is not None
    assert len(pickle.dumps(prep_preset)) < len(pickle.dumps(prep_full))
    ext_a = score_external_primary_pair(ds_tr, prep_preset, cfg=cfg)
    ext_b = score_external_primary_pair(ds_tr, prep_full, cfg=cfg)
    assert ext_a["external_delta_mse"] == pytest.approx(ext_b["external_delta_mse"], abs=1e-12)
    cfg_nolbl = replace(cfg, covariate_preset="no_label_trend")
    with pytest.raises(ValueError, match="no cached raw covariates for preset"):
        score_external_primary_pair(ds_tr, prep_preset, cfg=cfg_nolbl)
    # preset=None retains raw_cov_frame and supports any preset dynamically.
    ext_nolbl = score_external_primary_pair(ds_tr, prep_full, cfg=cfg_nolbl)
    assert np.isfinite(ext_nolbl["external_delta_mse"])

    # 4. BLAS thread pools are capped at 1 inside score_external_primary_pair and parallel calibration workers.
    tpc = _get_threadpool_controller()
    if tpc is not None:
        observed_threads: list[int] = []
        orig_score_fold = ncv_mod._score_external_fold

        def _spy_score_fold(*args: Any, **kwargs: Any) -> Any:
            for lib in tpc.lib_controllers:
                if lib.user_api == "blas":
                    observed_threads.append(int(lib.num_threads))
            return orig_score_fold(*args, **kwargs)

        monkeypatch.setattr(ncv_mod, "_score_external_fold", _spy_score_fold)
        score_external_primary_pair(ds_tr, prep_preset, cfg=cfg)
        assert observed_threads and all(t == 1 for t in observed_threads)

        worker_threads: list[int] = []
        orig_gen = cal_mod.generate_synthetic_cohort

        def _spy_gen(*args: Any, **kwargs: Any) -> Any:
            for lib in tpc.lib_controllers:
                if lib.user_api == "blas":
                    worker_threads.append(int(lib.num_threads))
            return orig_gen(*args, **kwargs)

        monkeypatch.setattr(cal_mod, "generate_synthetic_cohort", _spy_gen)
        cal_mod._eval_single_scenario_rep(
            seed=4206,
            n_participants=36,
            compact_grid=True,
            spec={"scenario": "s", "covariate_preset": "primary"},
            cfg=cfg,
            single_thread_blas=True,
        )
        assert worker_threads and all(t == 1 for t in worker_threads)
        worker_threads.clear()
        cal_mod._eval_single_calibration_rep(
            rep_seed=4208,
            n_participants=36,
            gen_kwargs={"signal_strength": 0.5},
            n_bootstraps=2,
            cfg=cfg,
            use_gpu=False,
            test_ds=prep_preset,
            compact_grid=True,
            single_thread_blas=True,
        )
        assert worker_threads and all(t == 1 for t in worker_threads)


def test_backend_and_triangular_fallback_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    import types
    from typing import Any

    import src.analysis.paired_lipid._backend as backend_mod
    import src.analysis.paired_lipid._batched_models as bm_mod
    import src.analysis.paired_lipid._eigensolvers as eig_mod
    from src.analysis.paired_lipid._gram_cache import (
        _get_dataset_gram_cache,
        _precompute_division_constants,
    )

    def _clear_backend_caches() -> None:
        backend_mod._probe_cupy.cache_clear()
        backend_mod._get_cupyx_linalg.cache_clear()
        backend_mod._cupyx_supports_batched_triangular.cache_clear()
        backend_mod._get_fused_mfa_gram_kernel.cache_clear()
        backend_mod._get_threadpool_controller.cache_clear()

    orig_import = builtins.__import__
    try:
        # 1. CuPy unavailable fallback paths in _backend.py.
        _clear_backend_caches()

        def _block_cupy(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "cupy" or name.startswith("cupy.") or name.startswith("cupyx"):
                raise ImportError("simulated missing cupy")
            return orig_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_cupy)
        assert backend_mod._probe_cupy() is None
        assert backend_mod._get_cupyx_linalg() is None
        assert backend_mod._cupyx_supports_batched_triangular() is False
        assert backend_mod._get_fused_mfa_gram_kernel() is None
        xp_fb, meta_fb = backend_mod.resolve_array_backend(True, 16)
        assert xp_fb is np
        assert meta_fb["gpu_fallback"] is True
        assert meta_fb["gpu_fallback_reason"] == "CuPy/CUDA device unavailable"
        assert backend_mod.suggest_batch_size(np, 1000, 180) == 128

        # 2. threadpoolctl unavailable fallback path in _backend.py.
        monkeypatch.setattr(builtins, "__import__", orig_import)
        _clear_backend_caches()

        def _block_tpc(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "threadpoolctl":
                raise ImportError("simulated missing threadpoolctl")
            return orig_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_tpc)
        assert backend_mod._get_threadpool_controller() is None
        with backend_mod._single_thread_blas():
            pass
    finally:
        monkeypatch.setattr(builtins, "__import__", orig_import)
        _clear_backend_caches()

    # 3. _cupyx_supports_batched_triangular() == False and runtime exception fallbacks in
    #    batched_pca_schur_sse, _solve_lower_triangular_batch, batched_ridge_sse_and_preds,
    #    and batched_ridge_preds_single_alpha (tested via fake non-NumPy proxy so it runs on CPU-only hosts too).
    fake_xp = types.SimpleNamespace(**{name: getattr(np, name) for name in dir(np) if not name.startswith("__")})

    class _ForbidTriLinalg:
        def solve_triangular(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("solve_triangular must not be called when batched triangular is unsupported")

    class _RaiseTriLinalg:
        def solve_triangular(self, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("simulated runtime solve_triangular failure")

    rng = np.random.default_rng(4301)
    B, k_dim = 4, 5
    A_mat = rng.normal(size=(B, k_dim, k_dim))
    L_tri = np.linalg.cholesky(A_mat @ np.swapaxes(A_mat, -1, -2) + 2.0 * np.eye(k_dim)[None, :, :])
    rhs_tri = rng.normal(size=(B, k_dim, 2))
    sol_ref = np.linalg.solve(L_tri, rhs_tri)

    cfg = PairedLipidConfig()
    ds, _ = generate_synthetic_cohort(n_participants=44, n_single_sample=2, seed=4302, compact_grid=True)
    tr_p = np.arange(34, dtype=int)
    val_p = np.arange(34, 44, dtype=int)
    gc = _get_dataset_gram_cache(ds, cfg)
    dc = _precompute_division_constants(np, gc, ds, tr_p, val_p, cfg)
    M_b = np.ones((2, ds.n_participants), dtype=np.float64)
    M_b[1, 0] = 2.0
    alphas = np.asarray(cfg.ridge_alpha_grid, dtype=np.float64)
    div_cpu = bm_mod.batched_division_prep(
        np, gc, dc, M_b[:, tr_p], M_b[:, val_p], ds.outcomes[cfg.primary_endpoint], alphas, cfg
    )
    pca_sse_ref = bm_mod.batched_pca_schur_sse(np, div_cpu, list(cfg.pca_dimensions), alphas)

    with monkeypatch.context() as mp:
        mp.setattr(eig_mod, "_cupyx_supports_batched_triangular", lambda: False)
        mp.setattr(eig_mod, "_get_cupyx_linalg", lambda: _ForbidTriLinalg())
        mp.setattr(bm_mod, "_cupyx_supports_batched_triangular", lambda: False)
        mp.setattr(bm_mod, "_get_cupyx_linalg", lambda: _ForbidTriLinalg())

        sol_guard = eig_mod._solve_lower_triangular_batch(fake_xp, L_tri, rhs_tri)
        np.testing.assert_allclose(sol_guard, sol_ref, rtol=1e-12, atol=1e-12)
        sse_guard = bm_mod.batched_pca_schur_sse(fake_xp, div_cpu, list(cfg.pca_dimensions), alphas)
        np.testing.assert_allclose(sse_guard, pca_sse_ref, rtol=1e-12, atol=1e-12)

        Z_tr_big = rng.normal(size=(2, tr_p.size, 14))
        Z_val_big = rng.normal(size=(2, val_p.size, 14))
        sse_r_ref, pr_r_ref = bm_mod.batched_ridge_sse_and_preds(
            np, div_cpu["X_cov_tr"], div_cpu["X_cov_val"], Z_tr_big, Z_val_big,
            div_cpu["y_tr"], div_cpu["y_val"], div_cpu["w_tr"], div_cpu["w_val"], alphas,
        )
        sse_r_fb, pr_r_fb = bm_mod.batched_ridge_sse_and_preds(
            fake_xp, div_cpu["X_cov_tr"], div_cpu["X_cov_val"], Z_tr_big, Z_val_big,
            div_cpu["y_tr"], div_cpu["y_val"], div_cpu["w_tr"], div_cpu["w_val"], alphas,
        )
        np.testing.assert_allclose(sse_r_fb, sse_r_ref, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(pr_r_fb, pr_r_ref, rtol=1e-12, atol=1e-12)

        pr_one_fb = bm_mod.batched_ridge_preds_single_alpha(
            fake_xp, div_cpu["X_cov_tr"], div_cpu["X_cov_val"], Z_tr_big, Z_val_big,
            div_cpu["y_tr"], div_cpu["w_tr"], np.array([alphas[1], alphas[3]]),
        )
        np.testing.assert_allclose(pr_one_fb[0], pr_r_ref[0, :, 1], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(pr_one_fb[1], pr_r_ref[1, :, 3], rtol=1e-12, atol=1e-12)

    # Runtime exception inside solve_triangular falls back cleanly to xp.linalg.solve.
    with monkeypatch.context() as mp:
        mp.setattr(eig_mod, "_cupyx_supports_batched_triangular", lambda: True)
        mp.setattr(eig_mod, "_get_cupyx_linalg", lambda: _RaiseTriLinalg())
        mp.setattr(bm_mod, "_cupyx_supports_batched_triangular", lambda: True)
        mp.setattr(bm_mod, "_get_cupyx_linalg", lambda: _RaiseTriLinalg())
        np.testing.assert_allclose(
            eig_mod._solve_lower_triangular_batch(fake_xp, L_tri, rhs_tri), sol_ref, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            bm_mod.batched_pca_schur_sse(fake_xp, div_cpu, list(cfg.pca_dimensions), alphas),
            pca_sse_ref,
            rtol=1e-12,
            atol=1e-12,
        )

    # 4. Real GPU fallback when _get_fused_mfa_gram_kernel() returns None or batched triangular is disabled.
    if _gpu_available():
        b_ref_gpu = run_multiplicity_bootstrap(ds, cfg=cfg, n_bootstraps=6, bootstrap_seed=4303, use_gpu=True)
        with monkeypatch.context() as mp:
            mp.setattr(bm_mod, "_get_fused_mfa_gram_kernel", lambda: None)
            mp.setattr(bm_mod, "_cupyx_supports_batched_triangular", lambda: False)
            mp.setattr(eig_mod, "_cupyx_supports_batched_triangular", lambda: False)
            b_fb_gpu = run_multiplicity_bootstrap(ds, cfg=cfg, n_bootstraps=6, bootstrap_seed=4303, use_gpu=True)
        np.testing.assert_allclose(
            b_fb_gpu["bootstrap_draws_delta_mse"],
            b_ref_gpu["bootstrap_draws_delta_mse"],
            rtol=1e-11,
            atol=1e-11,
        )
        np.testing.assert_allclose(
            b_fb_gpu["bootstrap_draws_vs_comparator"],
            b_ref_gpu["bootstrap_draws_vs_comparator"],
            rtol=1e-11,
            atol=1e-11,
        )


def _loky_blas_probe_worker(seed: int) -> list[int]:
    """Top-level picklable worker verifying _single_thread_blas() caps BLAS pools inside loky subprocesses."""
    from threadpoolctl import threadpool_info

    from src.analysis.paired_lipid._backend import _single_thread_blas

    with _single_thread_blas():
        ds, _ = generate_synthetic_cohort(n_participants=36, n_single_sample=2, seed=seed, compact_grid=True)
        _ = evaluate_nested_cohort(ds, cfg=PairedLipidConfig(), compute_diagnostics=False)
        return [int(info["num_threads"]) for info in threadpool_info() if info.get("user_api") == "blas"]


def test_review_round2_external_single_sample_dual_pls_schur_and_loky_blas() -> None:
    from dataclasses import replace
    from typing import Any

    from joblib import Parallel, delayed

    from src.analysis.paired_lipid._backend import _get_gpu_xp, _get_threadpool_controller
    from src.analysis.paired_lipid._batched_models import (
        _prep_covariates_batch,
        batched_division_prep,
        batched_paired_schur_sse,
        batched_pca_schur_sse,
        batched_pls_sse_and_preds,
    )
    from src.analysis.paired_lipid._eigensolvers import batched_solve_directions_grid
    from src.analysis.paired_lipid._gram_cache import (
        _get_dataset_gram_cache,
        _precompute_division_constants,
    )
    from src.analysis.paired_lipid.calibration import evaluate_bootstrap_calibration
    from src.analysis.paired_lipid.models import CovariatePreprocessor, WeightedPLSResult
    from src.analysis.paired_lipid.nested_cv import (
        PreparedExternalTest,
        _batched_anchor_bootstrap_partial_r_missing,
        _residualize_and_partial_r,
        make_nested_splits,
        score_external_primary_pair,
    )

    cfg = PairedLipidConfig()

    # 1. PreparedExternalTest auto-detects subset_mode="single_sample", matches raw CohortDataset scoring
    #    (including with covariate missingness), and fails closed when single_sample_downstream disagrees.
    ds_tr, _ = generate_synthetic_cohort(
        n_participants=40,
        n_single_sample=4,
        seed=4401,
        compact_grid=True,
        subset_mode="single_sample",
        covariate_missing_rate=0.15,
    )
    ds_te, _ = generate_synthetic_cohort(
        n_participants=80,
        n_single_sample=4,
        seed=4402,
        compact_grid=True,
        subset_mode="single_sample",
        covariate_missing_rate=0.15,
    )
    prep_auto = PreparedExternalTest.from_dataset(ds_te, cfg.primary_endpoint, preset=cfg.covariate_preset)
    prep_full = PreparedExternalTest.from_dataset(ds_te, cfg.primary_endpoint, preset=None)
    assert prep_auto.single_sample_downstream is True
    assert prep_full.single_sample_downstream is True
    ext_raw = score_external_primary_pair(ds_tr, ds_te, cfg=cfg)
    ext_auto = score_external_primary_pair(ds_tr, prep_auto, cfg=cfg)
    ext_full = score_external_primary_pair(ds_tr, prep_full, cfg=cfg)
    assert ext_auto["external_delta_mse"] == pytest.approx(ext_raw["external_delta_mse"], abs=1e-12)
    assert ext_full["external_delta_mse"] == pytest.approx(ext_raw["external_delta_mse"], abs=1e-12)
    ev_raw = evaluate_nested_cohort(ds_tr, cfg=cfg, external_test=ds_te, compute_diagnostics=False)
    ev_auto = evaluate_nested_cohort(ds_tr, cfg=cfg, external_test=prep_auto, compute_diagnostics=False)
    assert ev_auto["external_delta_mse"] == pytest.approx(ev_raw["external_delta_mse"], abs=1e-12)

    prep_paired = PreparedExternalTest.from_dataset(
        ds_te, cfg.primary_endpoint, single_sample_downstream=False, preset=cfg.covariate_preset
    )
    assert prep_paired.single_sample_downstream is False
    with pytest.raises(ValueError, match="single_sample_downstream"):
        score_external_primary_pair(ds_tr, prep_paired, cfg=cfg)
    with pytest.raises(ValueError, match="single_sample_downstream"):
        evaluate_nested_cohort(ds_tr, cfg=cfg, external_test=prep_paired, compute_diagnostics=False)

    cal_ss = evaluate_bootstrap_calibration(
        n_repetitions=2,
        n_bootstraps=3,
        n_participants=44,
        n_test_participants=60,
        base_seed=440,
        cfg=cfg,
        n_jobs=1,
        use_gpu=False,
        compact_grid=True,
        scenario_kwargs={"subset_mode": "single_sample", "covariate_missing_rate": 0.12},
    )
    assert np.isfinite(cal_ss["target_delta_mse"])

    # 2. Explicit single_sample_downstream=False override on a subset_mode="single_sample" cohort
    #    must propagate to fallback draws so batched and fallback draws share the same estimand.
    ds_ovr, _ = generate_synthetic_cohort(
        n_participants=52, n_single_sample=4, seed=4403, compact_grid=True, subset_mode="single_sample"
    )
    splits_ovr = make_nested_splits(ds_ovr, cfg, cfg.primary_seed)
    boot_ovr = run_multiplicity_bootstrap(
        ds_ovr, cfg=cfg, n_bootstraps=8, bootstrap_seed=4404, single_sample_downstream=False, use_gpu=False
    )
    assert boot_ovr["n_fallback_draws"] > 0
    rng_ovr = np.random.default_rng(4404)
    M_ovr = np.zeros((8, ds_ovr.n_participants), dtype=float)
    np.add.at(M_ovr, (np.arange(8)[:, None], rng_ovr.integers(0, ds_ovr.n_participants, size=(8, ds_ovr.n_participants))), 1.0)
    for b in range(8):
        ev_b = evaluate_nested_cohort(
            ds_ovr,
            cfg=cfg,
            participant_multiplicities=M_ovr[b],
            precomputed_splits=splits_ovr,
            single_sample_downstream=False,
            compute_diagnostics=False,
        )
        assert boot_ovr["bootstrap_draws_delta_mse"][b] == pytest.approx(
            ev_b["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"], abs=1e-10
        )

    # 3. Dual PLS roundoff floor prevents fitting Gram deflation noise on large-scale rank-deficient X
    #    while preserving genuine weak components on both CPU and GPU.
    rng_pls = np.random.default_rng(4405)
    n_tr, n_val, p_spec = 30, 10, 40
    u_tr = rng_pls.normal(size=(n_tr, 1))
    u_val = rng_pls.normal(size=(n_val, 1))
    v_dir = rng_pls.normal(size=(1, p_spec))
    X_big_tr = 1e4 * (u_tr @ v_dir)
    X_big_val = 1e4 * (u_val @ v_dir)
    y_noisy_tr = 2.0 * u_tr[:, 0] + 5.0 * rng_pls.normal(size=n_tr)
    y_noisy_val = 2.0 * u_val[:, 0] + 5.0 * rng_pls.normal(size=n_val)
    Xc0_tr = np.zeros((n_tr, 0))
    Xc0_val = np.zeros((n_val, 0))
    ref_big_p, _ = WeightedPLSResult.fit(Xc0_tr, X_big_tr, y_noisy_tr, np.ones(n_tr), max_components=5).predict(
        Xc0_val, X_big_val
    )
    backends: list[Any] = [np]
    cp_mod = _get_gpu_xp()
    if cp_mod is not None:
        backends.append(cp_mod)
    for xp in backends:
        div_pls = {
            "w_tr": xp.asarray(np.ones((1, n_tr)) / float(n_tr), dtype=xp.float64),
            "w_val": xp.asarray(np.ones((1, n_val)), dtype=xp.float64),
            "y_tr": xp.asarray(y_noisy_tr, dtype=xp.float64),
            "y_val": xp.asarray(y_noisy_val, dtype=xp.float64),
            "X_cov_tr": xp.asarray(Xc0_tr[None], dtype=xp.float64),
            "X_cov_val": xp.asarray(Xc0_val[None], dtype=xp.float64),
            "scale1_sq": xp.asarray([1.0], dtype=xp.float64),
            "scale2_sq": xp.asarray([0.0], dtype=xp.float64),
            "K_p1_tr": xp.asarray(X_big_tr @ X_big_tr.T, dtype=xp.float64),
            "K_p1_vt": xp.asarray(X_big_val @ X_big_tr.T, dtype=xp.float64),
            "K_p2_tr": xp.zeros((n_tr, n_tr), dtype=xp.float64),
            "K_p2_vt": xp.zeros((n_val, n_tr), dtype=xp.float64),
        }
        _, bp = batched_pls_sse_and_preds(xp, div_pls, max_comp=5)
        bp_np = np.asarray(xp.asnumpy(bp) if xp is not np else bp)[0]
        np.testing.assert_allclose(bp_np, ref_big_p, rtol=1e-10, atol=1e-10)

    # 4. batched_pca_schur_sse and batched_paired_schur_sse isolate non-PD / NaN draws without crashing valid draws.
    ds_sch, _ = generate_synthetic_cohort(n_participants=60, n_single_sample=2, seed=4406, compact_grid=True)
    tr_s = np.arange(48, dtype=int)
    val_s = np.arange(48, 60, dtype=int)
    gc_s = _get_dataset_gram_cache(ds_sch, cfg)
    dc_s = _precompute_division_constants(np, gc_s, ds_sch, tr_s, val_s, cfg)
    M2 = np.ones((2, 60), dtype=np.float64)
    alphas = np.asarray(cfg.ridge_alpha_grid, dtype=np.float64)
    div_ref = batched_division_prep(
        np, gc_s, dc_s, M2[:1, tr_s], M2[:1, val_s], ds_sch.outcomes[cfg.primary_endpoint], alphas, cfg
    )
    pca_ref = batched_pca_schur_sse(np, div_ref, list(cfg.pca_dimensions), alphas)[0]
    for bad_fill in ("indefinite", "nan"):
        div_bad = batched_division_prep(
            np, gc_s, dc_s, M2[:, tr_s], M2[:, val_s], ds_sch.outcomes[cfg.primary_endpoint], alphas, cfg
        )
        if bad_fill == "indefinite":
            div_bad["Schur_p"][1] = -1e6 * np.eye(div_bad["Schur_p"].shape[-1])[None, :, :]
        else:
            div_bad["Schur_p"][1] = np.nan
        sse_pca_bad = batched_pca_schur_sse(np, div_bad, list(cfg.pca_dimensions), alphas)
        np.testing.assert_allclose(sse_pca_bad[0], pca_ref, rtol=1e-11, atol=1e-11)
        assert np.all(np.isnan(sse_pca_bad[1]))
        V_grid_bad = batched_solve_directions_grid(
            np, div_bad["W_max"], div_bad["T_max"], list(cfg.pca_dimensions), cfg.lambda_grid
        )
        sse_pair_bad = batched_paired_schur_sse(np, div_bad, V_grid_bad, list(cfg.pca_dimensions), alphas)
        assert np.all(np.isfinite(sse_pair_bad[0]))
        assert np.all(np.isnan(sse_pair_bad[1]))

    # 5. Missing-cov stress edges: np.inf/-np.inf, p_b==0 with p_m>0 in full bootstrap, single all-NaN val row,
    #    and anchor missing-cov parity with inf/-np.inf and collinear quadratic indicators.
    ds_inf, _ = generate_synthetic_cohort(
        n_participants=92, n_single_sample=4, seed=4407, compact_grid=True, covariate_missing_rate=0.12
    )
    cov_inf = ds_inf.covariate_frame.copy()
    cov_inf.loc[0, "age_years"] = np.inf
    cov_inf.loc[3, "age_years"] = -np.inf
    cov_inf.loc[91, ["age_years", "sex", "statin", "sample_label_trend"]] = np.nan
    ds_inf_mod = replace(ds_inf, covariate_frame=cov_inf)
    ev_inf = evaluate_nested_cohort(ds_inf_mod, cfg=cfg, compute_diagnostics=False)
    b_inf = run_multiplicity_bootstrap(ds_inf_mod, cfg=cfg, n_bootstraps=4, bootstrap_seed=4408, use_gpu=False)
    assert b_inf["n_fallback_draws"] == 0
    assert b_inf["point_estimate_delta_mse"] == pytest.approx(
        ev_inf["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"], abs=1e-10
    )

    cov_pb0 = pd.DataFrame(
        {
            "age_years": [65.0 if i % 4 != 0 else np.nan for i in range(92)],
            "sex": ["female" if i % 5 != 0 else np.nan for i in range(92)],
            "statin": [1.0] * 92,
            "sample_label_trend": [10.0] * 92,
        }
    )
    ds_pb0 = replace(ds_inf, covariate_frame=cov_pb0)
    ev_pb0 = evaluate_nested_cohort(ds_pb0, cfg=cfg, compute_diagnostics=False)
    b_pb0 = run_multiplicity_bootstrap(ds_pb0, cfg=cfg, n_bootstraps=4, bootstrap_seed=4409, use_gpu=False)
    assert b_pb0["n_fallback_draws"] == 0
    assert b_pb0["point_estimate_delta_mse"] == pytest.approx(
        ev_pb0["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"], abs=1e-10
    )

    gc_inf = _get_dataset_gram_cache(ds_inf_mod, cfg)
    dc_1val = _precompute_division_constants(
        np, gc_inf, ds_inf_mod, np.arange(91, dtype=int), np.array([91], dtype=int), cfg
    )
    _, _, _, _, Xval_1 = _prep_covariates_batch(np, dc_1val, np.ones((1, 91)), np.ones((1, 1)))
    cp_1 = CovariatePreprocessor.fit(cov_inf, np.arange(91, dtype=int), np.ones(91), preset="primary")
    np.testing.assert_allclose(Xval_1[0], cp_1.transform(cov_inf, np.array([91], dtype=int)), rtol=1e-12, atol=1e-12)

    eps_f = float(np.finfo(float).eps)
    raw_q_inf, _ = CovariatePreprocessor.raw_design_matrix(cov_inf, preset="quadratic_age_and_label")
    idx_inf = np.random.default_rng(4410).integers(0, 92, size=(30, 92))
    s_inf = ev_inf["oof_spectral_contributions"]["paired_linear_ridge"]
    y_inf = ds_inf_mod.outcomes["ldl_mmol_l"]
    vec_inf_r = _batched_anchor_bootstrap_partial_r_missing(raw_q_inf, s_inf, y_inf, idx_inf, eps_f)
    ref_inf_r: list[float] = []
    vi = np.arange(92, dtype=int)
    for b_idx in idx_inf:
        bm = np.bincount(b_idx, minlength=92).astype(float)
        cp_b = CovariatePreprocessor.fit(cov_inf, vi, bm, preset="quadratic_age_and_label")
        Xb = cp_b.transform(cov_inf, b_idx)
        rb, _, _, dnb, rkb = _residualize_and_partial_r(
            np.column_stack([np.ones(92), Xb]), s_inf[b_idx], y_inf[b_idx], eps_f
        )
        if 92 - rkb >= 2 and dnb > 1e-12:
            ref_inf_r.append(rb)
    np.testing.assert_allclose(vec_inf_r, np.asarray(ref_inf_r), rtol=1e-11, atol=1e-11)

    # 6. Spawned loky child workers actually cap OpenBLAS thread pools to 1 inside _single_thread_blas().
    if _get_threadpool_controller() is not None:
        worker_blas_threads = Parallel(n_jobs=2)(delayed(_loky_blas_probe_worker)(4420 + i) for i in range(2))
        assert len(worker_blas_threads) == 2
        for thread_counts in worker_blas_threads:
            assert thread_counts and all(t == 1 for t in thread_counts)


def _loky_bootstrap_nested_guard_worker(seed_val: int) -> tuple[int, list[int], float]:
    from src.analysis.paired_lipid._backend import _get_threadpool_controller, _single_thread_blas
    from src.analysis.paired_lipid.batched import _resolve_cpu_bootstrap_workers

    ds, _ = generate_synthetic_cohort(n_participants=64, n_single_sample=2, seed=seed_val, compact_grid=True)
    cfg = PairedLipidConfig(pca_dimensions=(5, 10))
    resolved_workers = _resolve_cpu_bootstrap_workers(300, -1, min_parallel_draws=256)
    res = run_multiplicity_bootstrap(
        ds, cfg=cfg, n_bootstraps=260, bootstrap_seed=seed_val + 1, use_gpu=False, n_jobs=-1
    )
    threads: list[int] = []
    with _single_thread_blas():
        ctrl = _get_threadpool_controller()
        if ctrl is not None:
            threads = [int(info["num_threads"]) for info in ctrl.info() if info.get("user_api") == "blas"]
    return resolved_workers, threads, float(res["bootstrap_sd_delta_mse"])


def test_gpu_zero_sync_chefsi_and_flat_tail_parity() -> None:
    from src.analysis.paired_lipid.batched import (
        _chefsi_top_k_gpu,
        _exact_top_k_eigh_active,
        _get_dataset_gram_cache,
        _get_gpu_xp,
        _precompute_division_constants,
        _rayleigh_ritz_32,
    )
    from src.analysis.paired_lipid.splits import make_nested_splits

    cp = _get_gpu_xp()
    if cp is None:
        pytest.skip("CUDA/CuPy unavailable")

    cfg = PairedLipidConfig()
    # 1. Verify _rayleigh_ritz_32 branchless check_spd_sync=False matches check_spd_sync=True on SPD draws
    #    and correctly flags rank-deficient draws in spd mask.
    rng = np.random.default_rng(7101)
    B_rr, n_s_rr, q_rr = 6, 48, 32
    A_rr = rng.normal(size=(B_rr, n_s_rr, 10))
    G_rr = A_rr @ np.swapaxes(A_rr, -1, -2)
    G_rr = 0.5 * (G_rr + np.swapaxes(G_rr, -1, -2))
    Y_rr = rng.normal(size=(B_rr, n_s_rr, q_rr))
    Y_rr[2, :, 16:] = 0.0  # rank-deficient draw
    G_gpu = cp.asarray(G_rr, dtype=cp.float64)
    Y_gpu = cp.asarray(Y_rr, dtype=cp.float64)
    th_sync, u_sync, gu_sync = _rayleigh_ritz_32(cp, G_gpu, Y_gpu)
    th_ns, u_ns, gu_ns, spd_ns = _rayleigh_ritz_32(cp, G_gpu, Y_gpu, check_spd_sync=False, return_spd=True)
    spd_host = cp.asnumpy(spd_ns)
    assert bool(spd_host[0]) and not bool(spd_host[2])
    np.testing.assert_allclose(cp.asnumpy(th_ns[0]), cp.asnumpy(th_sync[0]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cp.asnumpy(u_ns[0]), cp.asnumpy(u_sync[0]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cp.asnumpy(gu_ns[0]), cp.asnumpy(gu_sync[0]), rtol=1e-12, atol=1e-12)

    # 2. Verify flat-tail cohort (n_nuisance_modes=0 => Q0_comb_ratio < 2.0) populates Q0_comb + Q0_comb_ratio
    #    and _chefsi_top_k_gpu(n_cycles=5, poly_deg=5, max_extra_cycles=4) + rank-deficient n_act<32 fallback
    #    matches _exact_top_k_eigh_active to fp64 precision.
    ds_flat, _ = generate_synthetic_cohort(
        n_participants=92, n_single_sample=4, seed=101, compact_grid=True, n_nuisance_modes=0
    )
    gc_flat = _get_dataset_gram_cache(ds_flat, cfg)
    outer_splits, inner_splits = make_nested_splits(ds_flat, cfg, cfg.primary_seed)
    tr_idx, _ = outer_splits[0]
    in_tr_rel, in_val_rel = inner_splits[0][0]
    dc_gpu = _precompute_division_constants(
        cp, gc_flat, ds_flat, tr_idx[in_tr_rel], tr_idx[in_val_rel], cfg
    )
    assert dc_gpu["Q0_comb"] is not None
    assert dc_gpu["Q0_comb_ratio"] is not None
    assert float(dc_gpu["Q0_comb_ratio"]) < 2.0

    n_s = int(dc_gpu["K1_div"].shape[0])
    sw_host = np.zeros((10, n_s), dtype=np.float64)
    for b in range(10):
        n_keep = 20 if b == 3 else n_s - (b % 5) * 4
        sw_host[b, :n_keep] = 1.0 / float(n_keep)
    sw_gpu = cp.asarray(sw_host, dtype=cp.float64)
    G_flat = (dc_gpu["K1_div"] + dc_gpu["K2_div"])[None, :, :] * (sw_gpu[:, :, None] * sw_gpu[:, None, :])
    G_flat = 0.5 * (G_flat + cp.swapaxes(G_flat, -1, -2))
    ev_chefsi, u_chefsi = _chefsi_top_k_gpu(
        cp, G_flat, 15, sw_gpu, dc_gpu["Q0_comb"], n_cycles=5, poly_deg=5, max_extra_cycles=4
    )
    ev_exact, u_exact = _exact_top_k_eigh_active(cp, G_flat, 15, sw_gpu)
    np.testing.assert_allclose(cp.asnumpy(ev_chefsi), cp.asnumpy(ev_exact), rtol=1e-12, atol=1e-12)
    # Check subspace alignment per eigenvalue multiplicity (up to sign).
    u_c_np = cp.asnumpy(u_chefsi)
    u_e_np = cp.asnumpy(u_exact)
    ev_e_np = cp.asnumpy(ev_exact)
    for b in range(10):
        # Verify residual ||G u - ev u|| / ev_0 <= 1e-11 for every draw including n_act<32 draw 3.
        G_b = cp.asnumpy(G_flat[b])
        res = G_b @ u_c_np[b] - u_c_np[b] * ev_e_np[b][None, :]
        assert float(np.max(np.linalg.norm(res, axis=0)) / max(ev_e_np[b, 0], 1e-12)) <= 1e-11
        assert float(np.min(np.abs(np.sum(u_c_np[b][:, :3] * u_e_np[b][:, :3], axis=0)))) >= 0.999999

    # 3. End-to-end GPU vs sequential (include_band_paired=True) parity across resampled draws on n_nuis=0.
    y_all = np.asarray(ds_flat.outcomes[cfg.primary_endpoint], dtype=float)
    rng_m = np.random.default_rng(7102)
    M_draws = np.zeros((4, ds_flat.n_participants), dtype=np.float64)
    drawn = rng_m.integers(0, ds_flat.n_participants, size=(4, ds_flat.n_participants))
    np.add.at(M_draws, (np.arange(4)[:, None], drawn), 1.0)
    from src.analysis.paired_lipid.batched import _run_batched_multiplicity_draws

    batched_gpu = _run_batched_multiplicity_draws(
        ds_flat, y_all, outer_splits, inner_splits, M_draws, cfg, gc_flat, use_gpu=True, return_oof_sd=True
    )
    assert isinstance(batched_gpu, dict)
    for b in range(4):
        seq_res = evaluate_nested_cohort(
            ds_flat,
            cfg=cfg,
            participant_multiplicities=M_draws[b],
            precomputed_splits=(outer_splits, inner_splits),
            include_band_paired=True,
            compute_diagnostics=False,
        )
        seq_pair = float(seq_res["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"])
        seq_comp = float(seq_res["paired_vs_strongest_comparator"]["delta_mse"])
        assert float(batched_gpu["delta_pair"][b]) == pytest.approx(seq_pair, abs=1e-12)
        assert float(batched_gpu["delta_comp"][b]) == pytest.approx(seq_comp, abs=1e-12)


def test_cpu_bootstrap_draw_chunk_parallelism_and_nested_guard() -> None:
    import os

    from joblib import Parallel, delayed

    from src.analysis.paired_lipid.batched import (
        _get_dataset_gram_cache,
        _resolve_cpu_bootstrap_workers,
        _run_batched_multiplicity_draws,
    )
    from src.analysis.paired_lipid.splits import make_nested_splits

    assert _resolve_cpu_bootstrap_workers(255, -1, min_parallel_draws=256) == 1
    assert _resolve_cpu_bootstrap_workers(512, 1, min_parallel_draws=256) == 1
    prev = os.environ.get("_PAIRED_LIPID_IN_BOOTSTRAP_WORKER")
    os.environ["_PAIRED_LIPID_IN_BOOTSTRAP_WORKER"] = "1"
    try:
        assert _resolve_cpu_bootstrap_workers(512, -1, min_parallel_draws=256) == 1
    finally:
        if prev is None:
            os.environ.pop("_PAIRED_LIPID_IN_BOOTSTRAP_WORKER", None)
        else:
            os.environ["_PAIRED_LIPID_IN_BOOTSTRAP_WORKER"] = prev

    if (os.cpu_count() or 1) >= 2:
        assert _resolve_cpu_bootstrap_workers(512, -1, min_parallel_draws=256) >= 2

    ds, _ = generate_synthetic_cohort(n_participants=72, n_single_sample=2, seed=7201, compact_grid=True)
    cfg = PairedLipidConfig(pca_dimensions=(5, 10))
    gcache = _get_dataset_gram_cache(ds, cfg)
    outer_splits, inner_splits = make_nested_splits(ds, cfg, cfg.primary_seed)
    y_all = np.asarray(ds.outcomes[cfg.primary_endpoint], dtype=np.float64)

    n_draws = 272
    rng = np.random.default_rng(7202)
    M_sub = np.zeros((n_draws, ds.n_participants), dtype=np.float64)
    drawn = rng.integers(0, ds.n_participants, size=(n_draws, ds.n_participants))
    np.add.at(M_sub, (np.arange(n_draws)[:, None], drawn), 1.0)

    fb_serial: set[int] = set()
    meta_serial: dict[str, object] = {}
    out_serial = _run_batched_multiplicity_draws(
        ds,
        y_all,
        outer_splits,
        inner_splits,
        M_sub,
        cfg,
        gcache,
        use_gpu=False,
        execution_metadata=meta_serial,
        fallback_draws_out=fb_serial,
        return_oof_sd=True,
        n_jobs=1,
    )
    fb_par: set[int] = set()
    meta_par: dict[str, object] = {}
    out_par = _run_batched_multiplicity_draws(
        ds,
        y_all,
        outer_splits,
        inner_splits,
        M_sub,
        cfg,
        gcache,
        use_gpu=False,
        execution_metadata=meta_par,
        fallback_draws_out=fb_par,
        return_oof_sd=True,
        n_jobs=-1,
    )
    assert fb_serial == fb_par
    assert isinstance(out_serial, dict) and isinstance(out_par, dict)
    for key in ("delta_pair", "delta_comp", "delta_strong", "sd_pair", "sd_comp", "sd_strong"):
        np.testing.assert_allclose(out_par[key], out_serial[key], rtol=0.0, atol=1e-13, err_msg=key)

    # Nested-pool guard inside active loky Parallel pool must resolve to 1 worker and 1 BLAS thread.
    worker_results = Parallel(n_jobs=2)(delayed(_loky_bootstrap_nested_guard_worker)(7210 + i) for i in range(2))
    for w_count, blas_threads, sd_val in worker_results:
        assert w_count == 1
        assert np.isfinite(sd_val) and sd_val > 0.0
        if blas_threads:
            assert all(t == 1 for t in blas_threads)


def test_batched_pls_matrix_free_cpu_parity_and_trimming() -> None:
    from src.analysis.paired_lipid._batched_models import batched_pls_sse_and_preds
    from src.analysis.paired_lipid._gram_cache import _MAX_DCONST_CACHE_ENTRIES
    from src.analysis.paired_lipid.models import WeightedPLSResult

    assert _MAX_DCONST_CACHE_ENTRIES == 256

    rng = np.random.default_rng(7301)
    n_tr, n_val = 28, 12
    B = 12  # >= 8 triggers matrix-free CPU path

    X_cov_tr = rng.normal(size=(B, n_tr, 3))
    X_cov_val = rng.normal(size=(B, n_val, 3))
    X1_tr = rng.normal(size=(n_tr, 24))
    X1_val = rng.normal(size=(n_val, 24))
    X2_tr = rng.normal(size=(n_tr, 18))
    X2_val = rng.normal(size=(n_val, 18))
    w_tr = rng.uniform(0.5, 2.0, size=(B, n_tr))
    w_tr /= np.sum(w_tr, axis=1, keepdims=True)
    w_val = rng.uniform(0.5, 2.0, size=(B, n_val))
    y_tr = rng.normal(size=n_tr)
    y_val = rng.normal(size=n_val)
    scale1_sq = rng.uniform(0.4, 2.5, size=B)
    scale2_sq = rng.uniform(0.4, 2.5, size=B)

    div_cpu = {
        "w_tr": w_tr,
        "w_val": w_val,
        "y_tr": y_tr,
        "y_val": y_val,
        "X_cov_tr": X_cov_tr,
        "X_cov_val": X_cov_val,
        "scale1_sq": scale1_sq,
        "scale2_sq": scale2_sq,
        "K_p1_tr": X1_tr @ X1_tr.T,
        "K_p1_vt": X1_val @ X1_tr.T,
        "K_p2_tr": X2_tr @ X2_tr.T,
        "K_p2_vt": X2_val @ X2_tr.T,
    }
    sse_mf, preds_mf = batched_pls_sse_and_preds(np, div_cpu, max_comp=5)
    assert preds_mf is not None and sse_mf is not None
    batch_keys = {"w_tr", "w_val", "X_cov_tr", "X_cov_val", "scale1_sq", "scale2_sq"}
    # Compare matrix-free (B=12) against materialized small-batch path (B=1 per draw) and WeightedPLSResult.fit.
    for b in range(B):
        div_b1 = {
            k: (v[b : b + 1] if k in batch_keys else v)
            for k, v in div_cpu.items()
        }
        sse_ref_b, preds_ref_b = batched_pls_sse_and_preds(np, div_b1, max_comp=5)
        np.testing.assert_allclose(sse_mf[b : b + 1], sse_ref_b, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(preds_mf[b : b + 1], preds_ref_b, rtol=1e-13, atol=1e-13)

        X_spec_tr_b = np.hstack([np.sqrt(scale1_sq[b]) * X1_tr, np.sqrt(scale2_sq[b]) * X2_tr])
        X_spec_val_b = np.hstack([np.sqrt(scale1_sq[b]) * X1_val, np.sqrt(scale2_sq[b]) * X2_val])
        pr_fit, _ = WeightedPLSResult.fit(X_cov_tr[b], X_spec_tr_b, y_tr, w_tr[b], max_components=5).predict(
            X_cov_val[b], X_spec_val_b
        )
        np.testing.assert_allclose(preds_mf[b], pr_fit, rtol=1e-11, atol=1e-11)

    # Check return_preds=False, target_comp_idx + return_sse=False, and p_cov=0.
    sse_only, none_preds = batched_pls_sse_and_preds(np, div_cpu, max_comp=5, return_preds=False)
    assert none_preds is None
    np.testing.assert_allclose(sse_only, sse_mf, rtol=0.0, atol=1e-15)

    target_idx = np.array([0, 1, 2, 3, 4, 0, 2, 4, 1, 3, 0, 2], dtype=np.int64)
    none_sse, trimmed_preds = batched_pls_sse_and_preds(
        np, div_cpu, max_comp=int(np.max(target_idx)) + 1, return_sse=False, target_comp_idx=target_idx
    )
    assert none_sse is None
    assert trimmed_preds.shape == (B, n_val)
    np.testing.assert_allclose(
        trimmed_preds, preds_mf[np.arange(B), :, target_idx], rtol=0.0, atol=1e-14
    )

    div_nocov = {
        **div_cpu,
        "X_cov_tr": np.zeros((B, n_tr, 0), dtype=np.float64),
        "X_cov_val": np.zeros((B, n_val, 0), dtype=np.float64),
    }
    sse_nc, preds_nc = batched_pls_sse_and_preds(np, div_nocov, max_comp=4)
    for b in range(4):
        div_nc1 = {
            k: (v[b : b + 1] if k in batch_keys else v)
            for k, v in div_nocov.items()
        }
        sse_nc1, preds_nc1 = batched_pls_sse_and_preds(np, div_nc1, max_comp=4)
        np.testing.assert_allclose(sse_nc[b : b + 1], sse_nc1, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(preds_nc[b : b + 1], preds_nc1, rtol=1e-13, atol=1e-13)

    # Check scale=1e4 rank-1 exact + tiny_t_sq + const_y inside B=10 matrix-free batch.
    u_r1 = rng.normal(size=(n_tr, 1)) * 1e4
    v_r1 = rng.normal(size=(1, 16))
    X_r1_tr = u_r1 @ v_r1
    X_r1_val = (rng.normal(size=(n_val, 1)) * 1e4) @ v_r1
    X_cov_r1_tr = np.zeros((10, n_tr, 2), dtype=np.float64)
    X_cov_r1_val = np.zeros((10, n_val, 2), dtype=np.float64)
    y_r1 = rng.normal(size=n_tr) * 1e4
    w_eq = np.full((10, n_tr), 1.0 / float(n_tr), dtype=np.float64)
    div_r1 = {
        "w_tr": w_eq,
        "w_val": np.ones((10, n_val), dtype=np.float64),
        "y_tr": y_r1,
        "y_val": rng.normal(size=n_val) * 1e4,
        "X_cov_tr": X_cov_r1_tr,
        "X_cov_val": X_cov_r1_val,
        "scale1_sq": np.ones(10, dtype=np.float64),
        "scale2_sq": np.zeros(10, dtype=np.float64),
        "K_p1_tr": X_r1_tr @ X_r1_tr.T,
        "K_p1_vt": X_r1_val @ X_r1_tr.T,
        "K_p2_tr": np.zeros((n_tr, n_tr), dtype=np.float64),
        "K_p2_vt": np.zeros((n_val, n_tr), dtype=np.float64),
    }
    _, preds_r1 = batched_pls_sse_and_preds(np, div_r1, max_comp=5)
    ref_r1, _ = WeightedPLSResult.fit(X_cov_r1_tr[0], X_r1_tr, y_r1, w_eq[0], max_components=5).predict(
        X_cov_r1_val[0], X_r1_val
    )
    np.testing.assert_allclose(preds_r1[0], ref_r1, rtol=1e-11, atol=1e-11)


def test_spectral_trajectory_reuse_parity_and_fail_closed() -> None:
    from dataclasses import replace

    from src.analysis.paired_lipid.batched import _get_gpu_xp
    from src.analysis.paired_lipid.cohort import COVARIATE_PRESETS
    from src.analysis.paired_lipid.nested_cv import build_spectral_trajectory

    cfg = PairedLipidConfig(pca_dimensions=(5, 10))
    ds, _ = generate_synthetic_cohort(
        n_participants=76,
        n_single_sample=4,
        seed=7401,
        compact_grid=True,
        covariate_missing_rate=0.12,
    )

    # 1. Cross-preset parity (point + bootstrap) on CPU across all 5 COVARIATE_PRESETS with covariate missingness.
    traj_cpu = build_spectral_trajectory(
        ds, cfg=cfg, n_bootstraps=12, bootstrap_seed=7402, use_gpu=False, include_point=True
    )
    for preset in COVARIATE_PRESETS:
        cfg_p = replace(cfg, covariate_preset=preset)
        ev_ref = evaluate_nested_cohort(ds, cfg=cfg_p, compute_diagnostics=True)
        ev_traj = evaluate_nested_cohort(
            ds, cfg=cfg_p, compute_diagnostics=True, spectral_trajectory=traj_cpu
        )
        assert ev_traj["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"] == pytest.approx(
            ev_ref["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"], abs=1e-14
        )
        assert ev_traj["paired_vs_strongest_comparator"]["delta_mse"] == pytest.approx(
            ev_ref["paired_vs_strongest_comparator"]["delta_mse"], abs=1e-14
        )
        np.testing.assert_allclose(
            ev_traj["oof_predictions"]["paired_linear_ridge"],
            ev_ref["oof_predictions"]["paired_linear_ridge"],
            rtol=1e-14,
            atol=1e-14,
        )

        bt_ref = run_multiplicity_bootstrap(
            ds, cfg=cfg_p, n_bootstraps=12, bootstrap_seed=7402, use_gpu=False
        )
        bt_traj = run_multiplicity_bootstrap(
            ds, cfg=cfg_p, n_bootstraps=12, bootstrap_seed=7402, use_gpu=False, spectral_trajectory=traj_cpu
        )
        assert bt_traj["bootstrap_sd_delta_mse"] == pytest.approx(bt_ref["bootstrap_sd_delta_mse"], abs=1e-14)
        assert bt_traj["ci_95_delta_mse"] == pytest.approx(bt_ref["ci_95_delta_mse"], abs=1e-14)
        assert bt_traj["ci_95_vs_comparator"] == pytest.approx(bt_ref["ci_95_vs_comparator"], abs=1e-14)
        assert bt_traj["n_fallback_draws"] == bt_ref["n_fallback_draws"]

    # 2. GPU trajectory parity (if CUDA available).
    if _get_gpu_xp() is not None:
        traj_gpu = build_spectral_trajectory(
            ds, cfg=cfg, n_bootstraps=12, bootstrap_seed=7402, use_gpu=True, include_point=True
        )
        for preset in ("primary", "pre_event_risk_factors"):
            cfg_p = replace(cfg, covariate_preset=preset)
            bt_ref_g = run_multiplicity_bootstrap(
                ds, cfg=cfg_p, n_bootstraps=12, bootstrap_seed=7402, use_gpu=True
            )
            bt_traj_g = run_multiplicity_bootstrap(
                ds, cfg=cfg_p, n_bootstraps=12, bootstrap_seed=7402, use_gpu=True, spectral_trajectory=traj_gpu
            )
            assert bt_traj_g["bootstrap_sd_delta_mse"] == pytest.approx(
                bt_ref_g["bootstrap_sd_delta_mse"], abs=1e-12
            )
            assert bt_traj_g["ci_95_delta_mse"] == pytest.approx(bt_ref_g["ci_95_delta_mse"], abs=1e-12)

    # 3. single_sample_downstream=True and explicit False parity on subset_mode="single_sample".
    ds_ss, _ = generate_synthetic_cohort(
        n_participants=76, n_single_sample=0, seed=7403, compact_grid=True, subset_mode="single_sample"
    )
    for ss_flag in (True, False):
        traj_ss = build_spectral_trajectory(
            ds_ss,
            cfg=cfg,
            single_sample_downstream=ss_flag,
            n_bootstraps=8,
            bootstrap_seed=7404,
            use_gpu=False,
        )
        cfg_p = replace(cfg, covariate_preset="quadratic_age_and_label")
        ev_ss_ref = evaluate_nested_cohort(
            ds_ss, cfg=cfg_p, single_sample_downstream=ss_flag, compute_diagnostics=False
        )
        ev_ss_traj = evaluate_nested_cohort(
            ds_ss,
            cfg=cfg_p,
            single_sample_downstream=ss_flag,
            compute_diagnostics=False,
            spectral_trajectory=traj_ss,
        )
        assert ev_ss_traj["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"] == pytest.approx(
            ev_ss_ref["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"], abs=1e-14
        )
        bt_ss_ref = run_multiplicity_bootstrap(
            ds_ss, cfg=cfg_p, single_sample_downstream=ss_flag, n_bootstraps=8, bootstrap_seed=7404, use_gpu=False
        )
        bt_ss_traj = run_multiplicity_bootstrap(
            ds_ss,
            cfg=cfg_p,
            single_sample_downstream=ss_flag,
            n_bootstraps=8,
            bootstrap_seed=7404,
            use_gpu=False,
            spectral_trajectory=traj_ss,
        )
        assert bt_ss_traj["bootstrap_sd_delta_mse"] == pytest.approx(
            bt_ss_ref["bootstrap_sd_delta_mse"], abs=1e-14
        )

    # 4. allow_missing_outcome=True and fallback-draw cohort parity.
    ds_miss, _ = generate_synthetic_cohort(n_participants=60, n_single_sample=2, seed=7405, compact_grid=True)
    y_miss = ds_miss.outcomes[cfg.primary_endpoint].copy()
    y_miss[0] = np.nan
    ds_miss = replace(ds_miss, outcomes={**ds_miss.outcomes, cfg.primary_endpoint: y_miss})
    cfg_fb = PairedLipidConfig(pca_dimensions=(5, 10, 15))
    traj_miss = build_spectral_trajectory(
        ds_miss,
        cfg=cfg_fb,
        allow_missing_outcome=True,
        n_bootstraps=6,
        bootstrap_seed=7406,
        use_gpu=False,
    )
    bt_miss_ref = run_multiplicity_bootstrap(
        ds_miss, cfg=cfg_fb, allow_missing_outcome=True, n_bootstraps=6, bootstrap_seed=7406, use_gpu=False
    )
    bt_miss_traj = run_multiplicity_bootstrap(
        ds_miss,
        cfg=cfg_fb,
        allow_missing_outcome=True,
        n_bootstraps=6,
        bootstrap_seed=7406,
        use_gpu=False,
        spectral_trajectory=traj_miss,
    )
    assert bt_miss_traj["n_fallback_draws"] == bt_miss_ref["n_fallback_draws"]
    assert bt_miss_traj["point_estimate_delta_mse"] == pytest.approx(
        bt_miss_ref["point_estimate_delta_mse"], abs=1e-14
    )

    # 5. Fail-closed validation checks.
    with pytest.raises(ValueError, match="include_band_paired=True"):
        evaluate_nested_cohort(ds, cfg=cfg, include_band_paired=True, spectral_trajectory=traj_cpu)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        evaluate_nested_cohort(ds, cfg=cfg, single_sample_downstream=True, spectral_trajectory=traj_cpu)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        evaluate_nested_cohort(ds, cfg=cfg, outer_seed=999, spectral_trajectory=traj_cpu)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        evaluate_nested_cohort(ds, cfg=replace(cfg, pca_dimensions=(5, 12)), spectral_trajectory=traj_cpu)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        evaluate_nested_cohort(ds_ss, cfg=cfg, spectral_trajectory=traj_cpu)
    with pytest.raises(ValueError, match="bootstrap fingerprint mismatch"):
        run_multiplicity_bootstrap(ds, cfg=cfg, n_bootstraps=12, bootstrap_seed=9999, spectral_trajectory=traj_cpu)
    with pytest.raises(ValueError, match="bootstrap fingerprint mismatch"):
        run_multiplicity_bootstrap(ds, cfg=cfg, n_bootstraps=10, bootstrap_seed=7402, spectral_trajectory=traj_cpu)
    with pytest.raises(ValueError, match="resampled participant_multiplicities"):
        evaluate_nested_cohort(
            ds,
            cfg=cfg,
            participant_multiplicities=np.full(ds.n_participants, 2.0),
            spectral_trajectory=traj_cpu,
        )
    traj_no_pt = build_spectral_trajectory(ds, cfg=cfg, n_bootstraps=0, include_point=False)
    with pytest.raises(ValueError, match="point_artifacts"):
        evaluate_nested_cohort(ds, cfg=cfg, spectral_trajectory=traj_no_pt)
    with pytest.raises(ValueError, match="bootstrap_artifacts"):
        run_multiplicity_bootstrap(ds, cfg=cfg, n_bootstraps=12, bootstrap_seed=7402, spectral_trajectory=traj_no_pt)
    # Different endpoint missingness mask must fail closed.
    y_other = ds_miss.outcomes[cfg.primary_endpoint].copy()
    y_other[1] = np.nan
    ds_miss2 = replace(ds_miss, outcomes={**ds_miss.outcomes, "hdl_mmol_l": y_other})
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        evaluate_nested_cohort(
            ds_miss2, cfg=cfg_fb, endpoint="hdl_mmol_l", allow_missing_outcome=True, spectral_trajectory=traj_miss
        )


def test_batched_build_spectral_division_trajectories_empty_batch() -> None:
    from src.analysis.paired_lipid.batched import (
        SpectralDivisionTrajectory,
        _get_gpu_xp,
        batched_build_spectral_division_trajectories,
    )
    from src.analysis.paired_lipid.splits import make_nested_splits

    ds, _ = generate_synthetic_cohort(n_participants=40, n_single_sample=2, seed=7501, compact_grid=True)
    cfg = PairedLipidConfig(pca_dimensions=(5, 10))
    outer_sp, inner_sp = make_nested_splits(ds, cfg, cfg.primary_seed)
    M_empty = np.empty((0, ds.n_participants), dtype=np.float64)
    max_k = max(cfg.pca_dimensions)
    n_lam = len(cfg.lambda_grid)

    gpu_flags = [False, True] if _get_gpu_xp() is not None else [False]
    for use_gpu in gpu_flags:
        trajs = batched_build_spectral_division_trajectories(
            ds,
            outer_sp,
            inner_sp,
            M_empty,
            cfg,
            use_gpu=use_gpu,
        )
        assert len(trajs) == len(outer_sp)
        for (tr_idx, te_idx), in_splits, (in_trajs, out_traj) in zip(outer_sp, inner_sp, trajs):
            assert len(in_trajs) == len(in_splits)
            for (in_tr_rel, in_val_rel), in_t in zip(in_splits, in_trajs):
                assert isinstance(in_t, SpectralDivisionTrajectory)
                assert in_t.part_pca_tr.shape == (0, int(tr_idx[in_tr_rel].size), max_k)
                assert in_t.part_pca_tr.dtype == np.float64
                assert in_t.part_pca_val.shape == (0, int(tr_idx[in_val_rel].size), max_k)
                assert in_t.part_pca_val.dtype == np.float64
                assert in_t.W_max.shape == (0, max_k, max_k) and in_t.W_max.dtype == np.float64
                assert in_t.T_max.shape == (0, max_k, max_k) and in_t.T_max.dtype == np.float64
                assert in_t.scale1_sq.shape == (0,) and in_t.scale1_sq.dtype == np.float64
                assert in_t.scale2_sq.shape == (0,) and in_t.scale2_sq.dtype == np.float64
                assert in_t.rank_ok.shape == (0,) and in_t.rank_ok.dtype == bool
                assert in_t.V_grid is not None
                assert set(in_t.V_grid.keys()) == {int(k) for k in cfg.pca_dimensions}
                for k_val, v_arr in in_t.V_grid.items():
                    assert v_arr.shape == (0, n_lam, k_val, k_val)
                    assert v_arr.dtype == np.float64
                assert in_t.numerical_rank is None

            assert isinstance(out_traj, SpectralDivisionTrajectory)
            assert out_traj.part_pca_tr.shape == (0, int(tr_idx.size), max_k)
            assert out_traj.part_pca_tr.dtype == np.float64
            assert out_traj.part_pca_val.shape == (0, int(te_idx.size), max_k)
            assert out_traj.part_pca_val.dtype == np.float64
            assert out_traj.W_max.shape == (0, max_k, max_k) and out_traj.W_max.dtype == np.float64
            assert out_traj.T_max.shape == (0, max_k, max_k) and out_traj.T_max.dtype == np.float64
            assert out_traj.scale1_sq.shape == (0,) and out_traj.scale1_sq.dtype == np.float64
            assert out_traj.scale2_sq.shape == (0,) and out_traj.scale2_sq.dtype == np.float64
            assert out_traj.rank_ok.shape == (0,) and out_traj.rank_ok.dtype == bool
            assert out_traj.V_grid is None
            assert out_traj.numerical_rank is None


def _loky_external_memmap_probe_worker(
    prep: object,
) -> tuple[int, bool, bool, bool, bool, int, list[int]]:
    import os

    from src.analysis.paired_lipid._backend import _get_threadpool_controller, _single_thread_blas
    from src.analysis.paired_lipid.batched import _resolve_cpu_bootstrap_workers
    from src.analysis.paired_lipid.nested_cv import PreparedExternalTest

    assert isinstance(prep, PreparedExternalTest)
    assert prep.part_band_mat is not None
    with _single_thread_blas():
        nested_workers = _resolve_cpu_bootstrap_workers(512, -1)
        threads: list[int] = []
        ctrl = _get_threadpool_controller()
        if ctrl is not None:
            threads = [int(info["num_threads"]) for info in ctrl.info() if info.get("user_api") == "blas"]
    return (
        os.getpid(),
        isinstance(prep.spec_1300, np.memmap),
        isinstance(prep.spec_2850, np.memmap),
        isinstance(prep.part_band_mat, np.memmap),
        bool(prep.spec_1300.flags.writeable),
        nested_workers,
        threads,
    )


def test_median_maps_scratch_and_destructive_parity() -> None:
    from src.analysis.paired_lipid.synthetic import _median_maps

    rng = np.random.default_rng(7601)
    for has_drop in (False, True):
        maps = rng.normal(size=(41, 3, 37))
        maps[0, 0, 0] = np.nan
        maps[1, 1, 2] = np.inf
        maps[2, 2, 3] = -np.inf
        maps[3, :, 4] = np.array([0.0, -0.0, 0.0])
        maps_orig = maps.copy()
        drop = np.array([i % 4 == 0 for i in range(41)], dtype=bool) if has_drop else None

        out_no_scratch = np.empty((41, 37), dtype=np.float64)
        _median_maps(maps, out_no_scratch, drop)
        assert np.array_equal(maps, maps_orig, equal_nan=True)

        t1 = np.empty((41, 37), dtype=np.float64)
        t2 = np.empty((41, 37), dtype=np.float64)
        out_scratch = np.empty((41, 37), dtype=np.float64)
        _median_maps(maps, out_scratch, drop, destructive=False, scratch=(t1, t2))
        assert np.array_equal(maps, maps_orig, equal_nan=True)

        maps_mut = maps.copy()
        out_destr = np.empty((41, 37), dtype=np.float64)
        _median_maps(maps_mut, out_destr, drop, destructive=True)

        ref = np.empty((41, 37), dtype=np.float64)
        if drop is not None:
            ref[~drop] = np.median(maps_orig[~drop], axis=1)
            ref[drop] = 0.5 * (maps_orig[drop, 0, :] + maps_orig[drop, 1, :])
        else:
            ref[:] = np.median(maps_orig, axis=1)

        assert np.array_equal(out_no_scratch, ref, equal_nan=True)
        assert np.array_equal(out_scratch, ref, equal_nan=True)
        assert np.array_equal(out_destr, ref, equal_nan=True)

    # Non-3 maps fallback with and without drop.
    maps5 = rng.normal(size=(19, 5, 23))
    maps5_orig = maps5.copy()
    drop5 = np.array([i % 3 == 0 for i in range(19)], dtype=bool)
    out5_drop = np.empty((19, 23), dtype=np.float64)
    _median_maps(maps5, out5_drop, drop5, destructive=False)
    assert np.array_equal(maps5, maps5_orig, equal_nan=True)
    ref5_drop = np.empty((19, 23), dtype=np.float64)
    ref5_drop[~drop5] = np.median(maps5_orig[~drop5], axis=1)
    ref5_drop[drop5] = np.median(maps5_orig[drop5, :-1, :], axis=1)
    assert np.array_equal(out5_drop, ref5_drop, equal_nan=True)

    out5_full = np.empty((19, 23), dtype=np.float64)
    _median_maps(maps5, out5_full, None, destructive=True)
    assert np.array_equal(out5_full, np.median(maps5_orig, axis=1), equal_nan=True)


def test_streaming_synthetic_external_test_parity_across_scenarios_and_modes() -> None:
    import yaml

    from src.analysis.paired_lipid.nested_cv import PreparedExternalTest, _get_participant_spectra
    from src.analysis.paired_lipid.synthetic import generate_synthetic_external_test

    cfg_path = Path(__file__).resolve().parents[1] / "config" / "manuscript_ml.yaml"
    cfg_yaml = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    scenarios = cfg_yaml["simulation"]["scenarios"]
    assert len(scenarios) == 17

    # 1. All 17 manuscript scenarios (compact_grid=True) bit-exact parity.
    for idx, sc in enumerate(scenarios):
        preset = str(sc["covariate_preset"])
        kw = {k: v for k, v in sc.items() if k not in ("scenario", "covariate_preset")}
        ds_ref, _ = generate_synthetic_cohort(
            n_participants=260, n_single_sample=11, seed=8100 + idx, compact_grid=True, **kw
        )
        p_ref = PreparedExternalTest.from_dataset(ds_ref, "ldl_mmol_l", preset=preset)
        p_str = generate_synthetic_external_test(
            n_participants=260,
            n_single_sample=11,
            seed=8100 + idx,
            compact_grid=True,
            endpoint="ldl_mmol_l",
            preset=preset,
            **kw,
        )
        assert p_str.raw_cov_frame is None
        assert not p_str.spec_1300.flags.writeable
        assert not p_str.spec_2850.flags.writeable
        assert p_str.part_band_mat is not None and not p_str.part_band_mat.flags.writeable
        assert not p_str.y_ext.flags.writeable
        assert np.max(np.abs(p_str.spec_1300 - p_ref.spec_1300)) == 0.0
        assert np.max(np.abs(p_str.spec_2850 - p_ref.spec_2850)) == 0.0
        assert np.max(np.abs(p_str.part_band_mat - p_ref.part_band_mat)) == 0.0
        assert np.max(np.abs(p_str.y_ext - p_ref.y_ext)) == 0.0
        mat_ref, cols_ref = p_ref.raw_cache[f"raw_cov_matrix__{preset}"]
        mat_str, cols_str = p_str.raw_cache[f"raw_cov_matrix__{preset}"]
        assert cols_str == cols_ref
        assert np.array_equal(mat_str, mat_ref, equal_nan=True)

    # 2. All 4 subset_modes, N > 900 lexicographical sort, single_sample_downstream overrides, and dropouts.
    for mode in ("all_eligible", "paired_only", "single_sample", "three_map_complete"):
        for ss_ov in (None, True, False):
            ds_ref, _ = generate_synthetic_cohort(
                n_participants=950,
                n_single_sample=37,
                seed=8200,
                compact_grid=True,
                subset_mode=mode,
                map_dropout_fraction=0.15,
                qc_sample_dropout_fraction=0.1,
                n_nuisance_modes=4,
                nuisance_part_scale=0.3,
                nuisance_sample_scale=0.5,
                map_spatial_scale=0.4,
                map_spatial_corr=0.6,
                covariate_missing_rate=0.1,
            )
            p_ref = PreparedExternalTest.from_dataset(
                ds_ref, "ldl_mmol_l", single_sample_downstream=ss_ov, preset="primary"
            )
            p_str = generate_synthetic_external_test(
                n_participants=950,
                n_single_sample=37,
                seed=8200,
                compact_grid=True,
                endpoint="ldl_mmol_l",
                single_sample_downstream=ss_ov,
                preset="primary",
                subset_mode=mode,
                map_dropout_fraction=0.15,
                qc_sample_dropout_fraction=0.1,
                n_nuisance_modes=4,
                nuisance_part_scale=0.3,
                nuisance_sample_scale=0.5,
                map_spatial_scale=0.4,
                map_spatial_corr=0.6,
                covariate_missing_rate=0.1,
            )
            assert p_str.single_sample_downstream == p_ref.single_sample_downstream
            assert p_str.n_participants == p_ref.n_participants
            assert np.max(np.abs(p_str.spec_1300 - p_ref.spec_1300)) == 0.0
            assert np.max(np.abs(p_str.spec_2850 - p_ref.spec_2850)) == 0.0
            assert np.max(np.abs(p_str.part_band_mat - p_ref.part_band_mat)) == 0.0
            assert np.max(np.abs(p_str.y_ext - p_ref.y_ext)) == 0.0
            mat_ref, _ = p_ref.raw_cache["raw_cov_matrix__primary"]
            mat_str, _ = p_str.raw_cache["raw_cov_matrix__primary"]
            assert np.array_equal(mat_str, mat_ref, equal_nan=True)

    # 3. Chunk-boundary straddlers (compact_grid=False) and N > 2048 chunked _get_participant_spectra.
    for n_p, n_ss in ((129, 1), (257, 0), (2050, 5)):
        ds_ref, _ = generate_synthetic_cohort(
            n_participants=n_p, n_single_sample=n_ss, seed=8300, compact_grid=False
        )
        p_ref = PreparedExternalTest.from_dataset(ds_ref, "ldl_mmol_l", preset="primary")
        p_str = generate_synthetic_external_test(
            n_participants=n_p,
            n_single_sample=n_ss,
            seed=8300,
            compact_grid=False,
            endpoint="ldl_mmol_l",
            preset="primary",
        )
        assert np.max(np.abs(p_str.spec_1300 - p_ref.spec_1300)) == 0.0
        assert np.max(np.abs(p_str.spec_2850 - p_ref.spec_2850)) == 0.0
        assert np.max(np.abs(p_str.part_band_mat - p_ref.part_band_mat)) == 0.0
        assert np.max(np.abs(p_str.y_ext - p_ref.y_ext)) == 0.0
        if n_p > 2048:
            p1_chunk, p2_chunk = _get_participant_spectra(ds_ref, single_sample_downstream=False)
            p1_agg = aggregate_samples_to_participants(ds_ref.sample_1300, ds_ref, single_sample_only=False)
            p2_agg = aggregate_samples_to_participants(ds_ref.sample_2850, ds_ref, single_sample_only=False)
            assert np.max(np.abs(p1_chunk - p1_agg)) == 0.0
            assert np.max(np.abs(p2_chunk - p2_agg)) == 0.0

    # 4. Non-3 maps_per_sample streaming path + invalid endpoint fail-closed parity.
    ds_m5, _ = generate_synthetic_cohort(
        n_participants=64,
        n_single_sample=4,
        maps_per_sample=5,
        seed=8350,
        compact_grid=True,
        subset_mode="paired_only",
        map_dropout_fraction=0.2,
    )
    p_m5_ref = PreparedExternalTest.from_dataset(ds_m5, "ldl_mmol_l", preset="primary")
    p_m5_str = generate_synthetic_external_test(
        n_participants=64,
        n_single_sample=4,
        maps_per_sample=5,
        seed=8350,
        compact_grid=True,
        subset_mode="paired_only",
        map_dropout_fraction=0.2,
        endpoint="ldl_mmol_l",
        preset="primary",
    )
    assert np.max(np.abs(p_m5_str.spec_1300 - p_m5_ref.spec_1300)) == 0.0
    assert np.max(np.abs(p_m5_str.spec_2850 - p_m5_ref.spec_2850)) == 0.0
    assert np.max(np.abs(p_m5_str.part_band_mat - p_m5_ref.part_band_mat)) == 0.0

    with pytest.raises(ValueError, match="not present in external test dataset outcomes"):
        generate_synthetic_external_test(n_participants=16, seed=8360, compact_grid=True, endpoint="unknown")


def test_calibration_loky_memmap_and_parallel_parity() -> None:
    import os

    from joblib import Parallel, delayed

    from src.analysis.paired_lipid.synthetic import generate_synthetic_external_test

    # 1. Verify read-only C-contiguous arrays > 1 MiB memmap cleanly across loky workers
    #    (n_participants=7000 ensures part_band_mat is 7000 x 21 x 8 B = 1.12 MiB > 1 MiB).
    prep = generate_synthetic_external_test(
        n_participants=7000,
        n_single_sample=300,
        seed=8401,
        compact_grid=False,
        endpoint="ldl_mmol_l",
        preset="primary",
    )
    n_workers = min(16, os.cpu_count() or 1)
    if n_workers >= 2:
        worker_out = Parallel(n_jobs=n_workers)(
            delayed(_loky_external_memmap_probe_worker)(prep) for _ in range(n_workers)
        )
        assert len(worker_out) == n_workers
        for _, mm1, mm2, mm_b, wr1, nested_w, blas_threads in worker_out:
            assert mm1 and mm2 and mm_b
            assert not wr1
            assert nested_w == 1
            if blas_threads:
                assert all(t == 1 for t in blas_threads)

    # 2. Verify evaluate_bootstrap_calibration serial vs parallel parity using streaming external test.
    cfg = PairedLipidConfig(pca_dimensions=(5, 10))
    cal_ser = evaluate_bootstrap_calibration(
        n_repetitions=4,
        n_bootstraps=50,
        n_participants=60,
        base_seed=8410,
        cfg=cfg,
        n_jobs=1,
        use_gpu=False,
        n_test_participants=300,
        compact_grid=True,
    )
    cal_par = evaluate_bootstrap_calibration(
        n_repetitions=4,
        n_bootstraps=50,
        n_participants=60,
        base_seed=8410,
        cfg=cfg,
        n_jobs=-1,
        use_gpu=False,
        n_test_participants=300,
        compact_grid=True,
    )
    for key in (
        "target_delta_mse",
        "target_delta_vs_comparator",
        "mean_point_estimate",
        "mean_interval_width",
        "unconditional_coverage",
        "dataset_failure_rate",
    ):
        assert cal_par[key] == pytest.approx(cal_ser[key], abs=1e-12, rel=0.0)


def test_streaming_synthetic_external_test_boundary_parity() -> None:
    from src.analysis.paired_lipid.nested_cv import PreparedExternalTest, _prepare_external_test
    from src.analysis.paired_lipid.synthetic import generate_synthetic_external_test

    # 1. All-single-sample and trailing-single-sample boundaries across subset_modes and lex sort > 999.
    boundary_cases = (
        (2, 2, 91, "single_sample"),
        (16, 16, 101, "single_sample"),
        (20, 4, 2, "single_sample"),
        (20, 5, 7, "three_map_complete"),
        (1005, 1005, 1, "single_sample"),
        (1005, 1005, 1, "three_map_complete"),
    )
    for n_p, n_ss, seed, mode in boundary_cases:
        ds_ref, _ = generate_synthetic_cohort(
            n_participants=n_p,
            n_single_sample=n_ss,
            seed=seed,
            compact_grid=True,
            subset_mode=mode,
        )
        p_ref = PreparedExternalTest.from_dataset(ds_ref, "ldl_mmol_l", preset="primary")
        p_str = generate_synthetic_external_test(
            n_participants=n_p,
            n_single_sample=n_ss,
            seed=seed,
            compact_grid=True,
            subset_mode=mode,
            endpoint="ldl_mmol_l",
            preset="primary",
        )
        assert not p_ref.y_ext.flags.writeable
        assert p_str.n_participants == p_ref.n_participants
        assert np.max(np.abs(p_str.spec_1300 - p_ref.spec_1300)) == 0.0
        assert np.max(np.abs(p_str.spec_2850 - p_ref.spec_2850)) == 0.0
        assert np.max(np.abs(p_str.part_band_mat - p_ref.part_band_mat)) == 0.0
        assert np.max(np.abs(p_str.y_ext - p_ref.y_ext)) == 0.0
        mat_ref, cols_ref = p_ref.raw_cache["raw_cov_matrix__primary"]
        mat_str, cols_str = p_str.raw_cache["raw_cov_matrix__primary"]
        assert cols_str == cols_ref
        assert np.array_equal(mat_str, mat_ref, equal_nan=True)

    # 2. raw_cov_frame.equals(...) parity (preset=None) across all 4 subset_modes and missingness rates.
    for miss_rate in (0.0, 0.2):
        for mode in ("all_eligible", "paired_only", "single_sample", "three_map_complete"):
            ds_ref, _ = generate_synthetic_cohort(
                n_participants=20,
                n_single_sample=4,
                seed=2,
                compact_grid=True,
                subset_mode=mode,
                covariate_missing_rate=miss_rate,
            )
            p_ref = PreparedExternalTest.from_dataset(ds_ref, "ldl_mmol_l", preset=None)
            p_str = generate_synthetic_external_test(
                n_participants=20,
                n_single_sample=4,
                seed=2,
                compact_grid=True,
                subset_mode=mode,
                covariate_missing_rate=miss_rate,
                endpoint="ldl_mmol_l",
                preset=None,
            )
            assert p_ref.raw_cov_frame is not None and p_str.raw_cov_frame is not None
            assert p_str.raw_cov_frame.equals(p_ref.raw_cov_frame)

    # 3. Dropout + min_maps_per_block=1 parity across maps_per_sample in (2, 3) + all-excluded ValueError.
    cfg_min1 = PairedLipidConfig(min_maps_per_block=1)
    for maps_per_s in (2, 3):
        for mode in ("all_eligible", "paired_only", "single_sample", "three_map_complete"):
            if maps_per_s == 2 and mode == "three_map_complete":
                continue
            ds_ref, _ = generate_synthetic_cohort(
                n_participants=20,
                n_single_sample=4,
                maps_per_sample=maps_per_s,
                seed=2,
                compact_grid=True,
                subset_mode=mode,
                qc_sample_dropout_fraction=0.5,
                map_dropout_fraction=0.5,
                cfg=cfg_min1,
            )
            p_ref = PreparedExternalTest.from_dataset(ds_ref, "ldl_mmol_l", preset=None)
            p_str = generate_synthetic_external_test(
                n_participants=20,
                n_single_sample=4,
                maps_per_sample=maps_per_s,
                seed=2,
                compact_grid=True,
                subset_mode=mode,
                qc_sample_dropout_fraction=0.5,
                map_dropout_fraction=0.5,
                cfg=cfg_min1,
                endpoint="ldl_mmol_l",
                preset=None,
            )
            assert np.max(np.abs(p_str.spec_1300 - p_ref.spec_1300)) == 0.0
            assert np.max(np.abs(p_str.spec_2850 - p_ref.spec_2850)) == 0.0
            assert np.max(np.abs(p_str.part_band_mat - p_ref.part_band_mat)) == 0.0
            assert np.max(np.abs(p_str.y_ext - p_ref.y_ext)) == 0.0
            assert p_ref.raw_cov_frame is not None and p_str.raw_cov_frame is not None
            assert p_str.raw_cov_frame.equals(p_ref.raw_cov_frame)

    for ex_kw in (
        {"n_participants": 2, "n_single_sample": 2, "subset_mode": "paired_only"},
        {"n_participants": 8, "maps_per_sample": 2, "subset_mode": "three_map_complete"},
    ):
        with pytest.raises(ValueError, match="No eligible stroke participants remained"):
            generate_synthetic_cohort(seed=91, compact_grid=True, **ex_kw)
        with pytest.raises(ValueError, match="No eligible stroke participants remained"):
            generate_synthetic_external_test(seed=91, compact_grid=True, endpoint="ldl_mmol_l", **ex_kw)

    # 4. PreparedExternalTest.from_arrays & _prepare_external_test fail-closed boundary validation.
    valid_cov = pd.DataFrame(
        {
            "age_years": [65.0, 70.0],
            "sex": pd.Series(["female", "male"], dtype=object),
            "statin": [1, 0],
            "sample_label_trend": [1.5, 3.5],
        }
    )
    with pytest.raises(ValueError, match="n_participants >= 1"):
        PreparedExternalTest.from_arrays(
            endpoint="ldl_mmol_l",
            spec_1300=np.zeros((0, 4), dtype=np.float64),
            spec_2850=np.zeros((0, 4), dtype=np.float64),
            cov_frame=valid_cov.iloc[:0],
            y_ext=np.zeros(0, dtype=np.float64),
            part_band_mat=np.zeros((0, 21), dtype=np.float64),
        )
    with pytest.raises(ValueError, match="1D y_ext"):
        PreparedExternalTest.from_arrays(
            endpoint="ldl_mmol_l",
            spec_1300=np.zeros((2, 4), dtype=np.float64),
            spec_2850=np.zeros((2, 4), dtype=np.float64),
            cov_frame=valid_cov,
            y_ext=np.zeros((2, 1), dtype=np.float64),
            part_band_mat=np.zeros((2, 21), dtype=np.float64),
        )
    with pytest.raises(ValueError, match="matching participant counts"):
        PreparedExternalTest.from_arrays(
            endpoint="ldl_mmol_l",
            spec_1300=np.zeros((2, 4), dtype=np.float64),
            spec_2850=np.zeros((3, 4), dtype=np.float64),
            cov_frame=valid_cov,
            y_ext=np.zeros(2, dtype=np.float64),
            part_band_mat=np.zeros((2, 21), dtype=np.float64),
        )
    with pytest.raises(ValueError, match="matching participant counts"):
        PreparedExternalTest.from_arrays(
            endpoint="ldl_mmol_l",
            spec_1300=np.zeros((2, 4), dtype=np.float64),
            spec_2850=np.zeros((2, 4), dtype=np.float64),
            cov_frame=valid_cov.iloc[:1],
            y_ext=np.zeros(2, dtype=np.float64),
            part_band_mat=np.zeros((2, 21), dtype=np.float64),
        )
    empty_direct = PreparedExternalTest(
        endpoint="ldl_mmol_l",
        spec_1300=np.zeros((0, 4), dtype=np.float64),
        spec_2850=np.zeros((0, 4), dtype=np.float64),
        raw_cov_frame=valid_cov.iloc[:0],
        y_ext=np.zeros(0, dtype=np.float64),
        n_participants=0,
        part_band_mat=np.zeros((0, 21), dtype=np.float64),
    )
    with pytest.raises(ValueError, match="at least 1 participant"):
        _prepare_external_test(
            empty_direct, "ldl_mmol_l", single_sample_downstream=False, preset="primary"
        )


def test_calibration_paired_only_n_ext_and_checkpoint_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.analysis.paired_lipid.calibration as cal_mod

    cfg = PairedLipidConfig(pca_dimensions=(5, 10))
    common_kw: dict[str, object] = {
        "n_repetitions": 2,
        "n_bootstraps": 4,
        "n_participants": 30,
        "n_test_participants": 20,
        "base_seed": 440,
        "cfg": cfg,
        "compact_grid": True,
        "n_jobs": 1,
        "use_gpu": False,
        "scenario_kwargs": {"subset_mode": "paired_only", "n_single_sample": 4},
    }
    res_unckpt = evaluate_bootstrap_calibration(**common_kw)  # type: ignore[arg-type]
    for k in ("target_test_cohort_se", "target_total_se", "target_delta_mse"):
        assert np.isfinite(res_unckpt[k])

    ckpt_path = tmp_path / "paired_only_cal.npz"
    orig_eval_rep = cal_mod._eval_single_calibration_rep
    calls_before_crash = 0

    def _crash_on_second(*args: object, **kwargs: object) -> object:
        nonlocal calls_before_crash
        calls_before_crash += 1
        if calls_before_crash == 2:
            raise RuntimeError("simulated paired_only interruption")
        return orig_eval_rep(*args, **kwargs)  # type: ignore[arg-type]

    with monkeypatch.context() as mp:
        mp.setattr(cal_mod, "_eval_single_calibration_rep", _crash_on_second)
        with pytest.raises(RuntimeError, match="simulated paired_only interruption"):
            evaluate_bootstrap_calibration(
                checkpoint_path=ckpt_path,
                checkpoint_chunk_size=1,
                **common_kw,  # type: ignore[arg-type]
            )
    assert ckpt_path.is_file()
    with np.load(ckpt_path, allow_pickle=False) as saved:
        assert int(np.sum(saved["completed"])) == 1
        n_ext_saved = int(saved["ext_col_sum_mse"].size)
        assert n_ext_saved < 20
    assert res_unckpt["n_test_participants"] == n_ext_saved

    calls_partial = 0

    def _counting_eval(*args: object, **kwargs: object) -> object:
        nonlocal calls_partial
        calls_partial += 1
        return orig_eval_rep(*args, **kwargs)  # type: ignore[arg-type]

    with monkeypatch.context() as mp:
        mp.setattr(cal_mod, "_eval_single_calibration_rep", _counting_eval)
        res_partial = evaluate_bootstrap_calibration(
            checkpoint_path=ckpt_path,
            checkpoint_chunk_size=1,
            **common_kw,  # type: ignore[arg-type]
        )
    assert calls_partial == 1

    calls_done = 0

    def _counting_done(*args: object, **kwargs: object) -> object:
        nonlocal calls_done
        calls_done += 1
        return orig_eval_rep(*args, **kwargs)  # type: ignore[arg-type]

    with monkeypatch.context() as mp:
        mp.setattr(cal_mod, "_eval_single_calibration_rep", _counting_done)
        res_done = evaluate_bootstrap_calibration(
            checkpoint_path=ckpt_path,
            checkpoint_chunk_size=1,
            **common_kw,  # type: ignore[arg-type]
        )
    assert calls_done == 0

    for res in (res_partial, res_done):
        assert res["n_test_participants"] == n_ext_saved
        for k in ("target_test_cohort_se", "target_total_se", "target_delta_mse"):
            assert np.isfinite(res[k])
            assert res[k] == pytest.approx(res_unckpt[k], abs=1e-12, rel=0.0)


def test_calibration_serial_n_jobs_and_scenario_offsets_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.analysis.paired_lipid.batched as batched_mod
    import src.analysis.paired_lipid.calibration as cal_mod

    # P2: serial calibration branch forwards n_jobs=1 to run_multiplicity_bootstrap and avoids spawning workers.
    cfg = PairedLipidConfig(pca_dimensions=(5, 10))
    seen_n_jobs: list[object] = []
    orig_boot = cal_mod.run_multiplicity_bootstrap

    def _spy_boot(*args: object, **kwargs: object) -> object:
        seen_n_jobs.append(kwargs.get("n_jobs", "<MISSING>"))
        return orig_boot(*args, **kwargs)  # type: ignore[arg-type]

    def _forbid_parallel(*args: object, **kwargs: object) -> object:
        raise AssertionError("Parallel should not be invoked when n_jobs=1")

    with monkeypatch.context() as mp:
        mp.setattr(cal_mod, "run_multiplicity_bootstrap", _spy_boot)
        mp.setattr(batched_mod, "Parallel", _forbid_parallel)
        cal_mod.evaluate_bootstrap_calibration(
            n_repetitions=1,
            n_bootstraps=2,
            n_participants=36,
            n_test_participants=40,
            base_seed=451,
            cfg=cfg,
            n_jobs=1,
            use_gpu=False,
            compact_grid=True,
        )
    assert seen_n_jobs == [1]
    assert batched_mod._resolve_cpu_bootstrap_workers(301, 1) == 1

    # P3: run_synthetic_scenario_suite preserves canonical seed offsets when scenario_offsets is provided.
    spec_a = {"scenario": "scen_a", "covariate_preset": "primary", "signal_strength": 0.85}
    spec_b = {"scenario": "scen_b", "covariate_preset": "primary", "signal_strength": 0.85, "ab_shift_scale": 0.4}
    df_full = cal_mod.run_synthetic_scenario_suite(
        scenarios=[spec_a, spec_b],
        n_participants_grid=[36],
        n_repetitions=1,
        compact_grid=True,
        base_seed=1000,
        n_jobs=1,
        cfg=cfg,
    )
    df_sub = cal_mod.run_synthetic_scenario_suite(
        scenarios=[spec_b],
        n_participants_grid=[36],
        n_repetitions=1,
        compact_grid=True,
        base_seed=1000,
        n_jobs=1,
        cfg=cfg,
        scenario_offsets=[1],
    )
    row_full = df_full.loc[df_full["scenario"] == "scen_b"].iloc[0]
    row_sub = df_sub.iloc[0]
    assert float(row_sub["mean_delta_mse_paired"]) == float(row_full["mean_delta_mse_paired"])
    assert float(row_sub["mean_recovery_r2_ldl_factor"]) == float(row_full["mean_recovery_r2_ldl_factor"])
    with pytest.raises(ValueError, match="scenario_offsets length"):
        cal_mod.run_synthetic_scenario_suite(
            scenarios=[spec_b],
            n_participants_grid=[36],
            n_repetitions=1,
            compact_grid=True,
            n_jobs=1,
            cfg=cfg,
            scenario_offsets=[0, 1],
        )


def test_studentized_t_missing_outcome_se_scale_and_nonfinite_oof_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    import src.analysis.paired_lipid.nested_cv as ncv_mod
    from src.analysis.paired_lipid.nested_cv import _summarize_bootstrap_contrast

    # P5: unit checks on _summarize_bootstrap_contrast relative SE scaling and zero-weight failure handling.
    draws = np.array([0.10, 0.25, 0.40, 0.55, 0.30, 0.20, 0.45, 0.35], dtype=float)
    draw_sds = np.array([0.8, 0.9, 1.1, 1.0, 0.85, 0.95, 1.05, 0.75], dtype=float)
    w_sums = np.array([40.0, 44.0, 48.0, 52.0, 46.0, 50.0, 42.0, 49.0], dtype=float)
    n_obs = 48.0
    sd_scale = np.sqrt(n_obs / w_sums)

    res_none = _summarize_bootstrap_contrast(0.32, 0.9, draws, draw_sds, interval_method="studentized_t", valid=True)
    res_unit = _summarize_bootstrap_contrast(
        0.32, 0.9, draws, draw_sds, interval_method="studentized_t", valid=True, sd_scale=np.ones_like(draws)
    )
    assert res_none["ci"] == res_unit["ci"]

    res_scaled = _summarize_bootstrap_contrast(
        0.32, 0.9, draws, draw_sds, interval_method="studentized_t", valid=True, sd_scale=sd_scale
    )
    se_0 = 0.9 / np.sqrt(n_obs)
    se_b = draw_sds / np.sqrt(w_sums)
    t_explicit = (draws - float(np.mean(draws))) / se_b
    exp_lo = 0.32 - float(np.percentile(t_explicit, 97.5)) * se_0
    exp_hi = 0.32 - float(np.percentile(t_explicit, 2.5)) * se_0
    assert res_scaled["ci"][0] == pytest.approx(exp_lo, abs=1e-14, rel=0.0)
    assert res_scaled["ci"][1] == pytest.approx(exp_hi, abs=1e-14, rel=0.0)

    bad_scale = sd_scale.copy()
    bad_scale[0] = np.nan
    res_bad = _summarize_bootstrap_contrast(
        0.32, 0.9, draws, draw_sds, interval_method="studentized_t", valid=True, sd_scale=bad_scale
    )
    assert np.isnan(res_bad["ci"][0]) and np.isnan(res_bad["ci"][1])
    assert res_bad["ci_percentile"] == res_none["ci_percentile"]

    # P5 end-to-end with allow_missing_outcome=True.
    cfg = PairedLipidConfig(pca_dimensions=(5, 10))
    ds_base, _ = generate_synthetic_cohort(n_participants=60, seed=452, compact_grid=True)
    y_miss = ds_base.outcomes["hdl_mmol_l"].copy()
    y_miss[:10] = np.nan
    ds_miss = replace(ds_base, outcomes={**ds_base.outcomes, "hdl_mmol_l": y_miss})
    boot_miss = run_multiplicity_bootstrap(
        ds_miss,
        cfg=cfg,
        endpoint="hdl_mmol_l",
        n_bootstraps=6,
        bootstrap_seed=453,
        allow_missing_outcome=True,
        use_gpu=False,
        n_jobs=1,
    )
    assert boot_miss["interval_procedure_succeeded"]
    rng_m = np.random.default_rng(453)
    M_m = np.zeros((6, ds_miss.n_participants), dtype=np.float64)
    drawn_m = rng_m.integers(0, ds_miss.n_participants, size=(6, ds_miss.n_participants))
    np.add.at(M_m, (np.arange(6)[:, None], drawn_m), 1.0)
    M_m *= np.isfinite(y_miss)[None, :]
    n_obs_m = float(np.count_nonzero(np.isfinite(y_miss)))
    se_0_m = float(boot_miss["oof_loss_diff_sd_delta_mse"]) / np.sqrt(n_obs_m)
    se_b_m = boot_miss["bootstrap_oof_sds_delta_mse"] / np.sqrt(M_m.sum(axis=1))
    d_m = boot_miss["bootstrap_draws_delta_mse"]
    t_m = (d_m - float(np.mean(d_m))) / se_b_m
    pt_m = float(boot_miss["point_estimate_delta_mse"])
    assert boot_miss["ci_95_delta_mse"][0] == pytest.approx(pt_m - float(np.percentile(t_m, 97.5)) * se_0_m, abs=1e-14)
    assert boot_miss["ci_95_delta_mse"][1] == pytest.approx(pt_m - float(np.percentile(t_m, 2.5)) * se_0_m, abs=1e-14)

    # P7: non-finite prediction in strongest_comparator or covariate_ridge marks contrast non-estimable.
    orig_fit_outer = ncv_mod._select_inner_models_and_fit_outer
    for inject_key in ("strongest_comparator", "covariate_ridge"):
        called = {"n": 0}

        def _inject_nan(*args: object, _k: str = inject_key, _c: dict[str, int] = called, **kwargs: object) -> dict[str, object]:
            out = orig_fit_outer(*args, **kwargs)  # type: ignore[arg-type]
            if _c["n"] == 0:
                preds = out["predictions"][_k].copy()
                preds[0] = np.nan
                out["predictions"][_k] = preds
            _c["n"] += 1
            return out

        with monkeypatch.context() as mp:
            mp.setattr(ncv_mod, "_select_inner_models_and_fit_outer", _inject_nan)
            ev = evaluate_nested_cohort(ds_base, cfg=cfg, compute_diagnostics=False)
        if inject_key == "strongest_comparator":
            assert np.isnan(ev["paired_vs_strongest_comparator"]["delta_mse"])
            assert np.isfinite(ev["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"])
        else:
            assert np.isnan(ev["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"])


def test_batched_select_inner_settings_nonfinite_mse_routes_to_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.analysis.paired_lipid.batched as batched_mod
    from src.analysis.paired_lipid.batched import _get_dataset_gram_cache, _run_batched_multiplicity_draws
    from src.analysis.paired_lipid.splits import make_nested_splits

    cfg = PairedLipidConfig(pca_dimensions=(5, 10))
    ds, _ = generate_synthetic_cohort(n_participants=72, n_single_sample=2, seed=8101, compact_grid=True)
    outer_sp, inner_sp = make_nested_splits(ds, cfg, cfg.primary_seed)
    gcache = _get_dataset_gram_cache(ds, cfg)
    y_all = np.asarray(ds.outcomes[cfg.primary_endpoint], dtype=float)

    rng = np.random.default_rng(8102)
    M_two = np.ones((2, ds.n_participants), dtype=np.float64)
    M_two[1] = 0.0
    np.add.at(M_two[1], rng.integers(0, ds.n_participants, size=ds.n_participants), 1.0)

    orig_pca_sse = batched_mod.batched_pca_schur_sse

    def _inject_nan(xp: object, div: dict[str, object], k_list: list[int], alphas_xp: object) -> object:
        res = orig_pca_sse(xp, div, k_list, alphas_xp).copy()
        res[0, 1, 0] = np.nan
        return res

    monkeypatch.setattr(batched_mod, "batched_pca_schur_sse", _inject_nan)
    fb: set[int] = set()
    res = _run_batched_multiplicity_draws(
        ds,
        y_all,
        outer_sp,
        inner_sp,
        M_two,
        cfg,
        gcache,
        use_gpu=False,
        fallback_draws_out=fb,
        return_oof_sd=True,
    )
    assert isinstance(res, dict)
    assert fb == {0}
    assert np.isnan(res["delta_pair"][0])
    assert np.isnan(res["delta_comp"][0])

    seq_ref = evaluate_nested_cohort(
        ds,
        cfg=cfg,
        participant_multiplicities=M_two[1],
        precomputed_splits=(outer_sp, inner_sp),
        include_band_paired=True,
        compute_diagnostics=False,
    )
    assert float(res["delta_pair"][1]) == pytest.approx(
        float(seq_ref["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"]),
        abs=1e-13,
    )
    assert float(res["delta_comp"][1]) == pytest.approx(
        float(seq_ref["paired_vs_strongest_comparator"]["delta_mse"]),
        abs=1e-13,
    )


def test_whitened_generalized_eigh_scale_invariant_parity_with_solve_directions() -> None:
    from dataclasses import replace

    from scipy import linalg

    from src.analysis.paired_lipid._backend import _get_gpu_xp
    from src.analysis.paired_lipid._eigensolvers import _whitened_generalized_eigh
    from src.analysis.paired_lipid.batched import _get_dataset_gram_cache, _run_batched_multiplicity_draws
    from src.analysis.paired_lipid.models import PairedMoments
    from src.analysis.paired_lipid.splits import make_nested_splits

    k = 5
    W_ill = np.diag([1e4, 1e2, 1.0, 1e-4, 1e-12])
    T_base = np.diag([1.0, 2.0, 3.0, 4.0, 5.0])
    pm_ill = PairedMoments(
        W_max=W_ill, T_max=T_base, mean_ab_shift=np.zeros(k), n_paired_multiplicity=20.0, n_paired_distinct=20
    )
    with pytest.raises(linalg.LinAlgError):
        pm_ill.solve_directions(k, 0.0)

    W_small = np.eye(k) * 1e-19
    T_small = np.diag([5.0, 4.0, 3.0, 2.0, 1.0]) * 1e-19
    pm_small = PairedMoments(
        W_max=W_small, T_max=T_small, mean_ab_shift=np.zeros(k), n_paired_multiplicity=20.0, n_paired_distinct=20
    )
    ev_seq_sm, _, _ = pm_small.solve_directions(k, 0.01, eigenvalue_tolerance=1e-22)
    assert np.all(np.isfinite(ev_seq_sm))

    W_indef = np.diag([1.0, 1.0, 1.0, 1.0, -1e-15])
    pm_indef = PairedMoments(
        W_max=W_indef, T_max=T_base, mean_ab_shift=np.zeros(k), n_paired_multiplicity=20.0, n_paired_distinct=20
    )
    with pytest.raises(linalg.LinAlgError):
        pm_indef.solve_directions(k, 5e-16)

    cp = _get_gpu_xp()
    backends = [np, cp] if cp is not None else [np]
    for xp in backends:
        for grid_mode in (True, False):
            v_ill = _whitened_generalized_eigh(
                xp, xp.asarray(W_ill[None]), xp.asarray(T_base[None]), xp.array([0.0]), grid_mode=grid_mode
            )
            assert not bool(xp.any(xp.isfinite(v_ill)))
            v_sm = _whitened_generalized_eigh(
                xp,
                xp.asarray(W_small[None]),
                xp.asarray(T_small[None]),
                xp.array([0.01]),
                grid_mode=grid_mode,
                eigenvalue_tolerance=1e-22,
            )
            assert bool(xp.all(xp.isfinite(v_sm)))
            v_ind = _whitened_generalized_eigh(
                xp, xp.asarray(W_indef[None]), xp.asarray(T_base[None]), xp.array([5e-16]), grid_mode=grid_mode
            )
            assert not bool(xp.any(xp.isfinite(v_ind)))
            lam_empty = xp.array([0.01]) if grid_mode else xp.zeros(0, dtype=xp.float64)
            v_empty = _whitened_generalized_eigh(
                xp, xp.zeros((0, k, k)), xp.zeros((0, k, k)), lam_empty, grid_mode=grid_mode
            )
            assert v_empty.shape[0] == 0

    rng = np.random.default_rng(8201)
    ds_l0, _ = generate_synthetic_cohort(
        n_participants=80,
        n_single_sample=0,
        seed=8202,
        compact_grid=True,
        signal_strength=2.5,
        ab_shift_scale=0.0,
        bio_sample_noise=0.0,
        nuisance_sample_scale=0.0,
    )
    X1_l0 = ds_l0.sample_1300.copy()
    X2_l0 = ds_l0.sample_2850.copy()
    ia, ib = ds_l0.pair_sample_a_indices, ds_l0.pair_sample_b_indices
    X1_l0[ib] = X1_l0[ia] + 2e-6 * (X1_l0[ib] - X1_l0[ia])
    X2_l0[ib] = X2_l0[ia] + 2e-6 * (X2_l0[ib] - X2_l0[ia])
    v_dir = rng.normal(size=(1, X1_l0.shape[1]))
    v_dir /= np.linalg.norm(v_dir)
    X1_l0[ib] += (rng.normal(size=(ia.size, 1)) * 5.0) @ v_dir
    ds_l0 = replace(ds_l0, sample_1300=X1_l0, sample_2850=X2_l0)

    cfg_l0 = PairedLipidConfig(pca_dimensions=(5,), retained_directions=(1, 2, 3), lambda_grid=(0.0, 100.0))
    outer_l0, inner_l0 = make_nested_splits(ds_l0, cfg_l0, cfg_l0.primary_seed)
    gc_l0 = _get_dataset_gram_cache(ds_l0, cfg_l0)
    y_l0 = np.asarray(ds_l0.outcomes[cfg_l0.primary_endpoint], dtype=float)
    M_one = np.ones((1, ds_l0.n_participants), dtype=np.float64)

    fb_l0: set[int] = set()
    bat_l0 = _run_batched_multiplicity_draws(
        ds_l0, y_l0, outer_l0, inner_l0, M_one, cfg_l0, gc_l0, use_gpu=False, fallback_draws_out=fb_l0, return_oof_sd=True
    )
    seq_l0 = evaluate_nested_cohort(
        ds_l0, cfg=cfg_l0, precomputed_splits=(outer_l0, inner_l0), include_band_paired=True, compute_diagnostics=False
    )
    assert isinstance(bat_l0, dict)
    assert fb_l0 == set()
    assert float(bat_l0["delta_pair"][0]) == pytest.approx(
        float(seq_l0["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"]),
        abs=1e-13,
    )

    cfg_l0_only = PairedLipidConfig(pca_dimensions=(5,), retained_directions=(1, 2, 3), lambda_grid=(0.0,))
    fb_only: set[int] = set()
    bat_only = _run_batched_multiplicity_draws(
        ds_l0,
        y_l0,
        outer_l0,
        inner_l0,
        M_one,
        cfg_l0_only,
        gc_l0,
        use_gpu=False,
        fallback_draws_out=fb_only,
        return_oof_sd=True,
    )
    assert isinstance(bat_only, dict)
    assert fb_only == {0}
    assert np.isnan(bat_only["delta_pair"][0])


def test_fast_top1_eigvalsh_from_gram_residual_fallback_and_16_nuisance_parity() -> None:
    from scipy import linalg

    from src.analysis.paired_lipid._backend import _get_gpu_xp
    from src.analysis.paired_lipid._eigensolvers import _fast_top1_eigvalsh_from_gram
    from src.analysis.paired_lipid.batched import (
        _get_dataset_gram_cache,
        _precompute_division_constants,
        _run_batched_multiplicity_draws,
    )
    from src.analysis.paired_lipid.splits import make_nested_splits

    cp = _get_gpu_xp()
    rng = np.random.default_rng(8301)

    def _assert_top1_exact(
        K: np.ndarray, Q0: np.ndarray, sw: np.ndarray, k_mu: np.ndarray, mu_sq: np.ndarray
    ) -> None:
        G_all = (K[None, :, :] - k_mu[:, :, None] - k_mu[:, None, :] + mu_sq[:, None, None]) * (
            sw[:, :, None] * sw[:, None, :]
        )
        G_all = 0.5 * (G_all + np.swapaxes(G_all, -1, -2))
        ex_cpu = np.array([linalg.eigvalsh(G_all[b])[-1] for b in range(sw.shape[0])], dtype=np.float64)
        est_cpu = _fast_top1_eigvalsh_from_gram(np, K, k_mu, mu_sq, sw, Q0)
        rel_cpu = np.abs(est_cpu - ex_cpu) / np.maximum(np.abs(ex_cpu), 1e-15)
        assert float(rel_cpu.max()) < 1e-13
        if cp is not None:
            ex_gpu = cp.asnumpy(cp.linalg.eigvalsh(cp.asarray(G_all))[:, -1])
            est_gpu = cp.asnumpy(
                _fast_top1_eigvalsh_from_gram(
                    cp, cp.asarray(K), cp.asarray(k_mu), cp.asarray(mu_sq), cp.asarray(sw), cp.asarray(Q0)
                )
            )
            rel_gpu_cp = np.abs(est_gpu - ex_gpu) / np.maximum(np.abs(ex_cpu), 1e-15)
            rel_gpu_scipy = np.abs(est_gpu - ex_cpu) / np.maximum(np.abs(ex_cpu), 1e-15)
            assert float(rel_gpu_cp.max()) < 1e-13
            assert float(rel_gpu_scipy.max()) < 2e-13

    # 1. Isotropic flat-spectrum random block.
    n_s = 64
    X_flat = rng.normal(size=(n_s, 200))
    X_flat -= X_flat.mean(axis=0)
    K_flat = 0.5 * (X_flat @ X_flat.T + (X_flat @ X_flat.T).T)
    w0 = np.full(n_s, 1.0 / n_s)
    sw0 = np.sqrt(w0 * n_s / (n_s - 1.0))
    k0 = w0 @ K_flat
    m0 = float(w0 @ k0)
    G0 = (K_flat - k0[:, None] - k0[None, :] + m0) * (sw0[:, None] * sw0[None, :])
    _, U0 = linalg.eigh(G0, subset_by_index=[n_s - 8, n_s - 1], driver="evr")
    Q0_flat = U0 / sw0[:, None]
    M_f = rng.multinomial(n_s, w0, size=24).astype(np.float64)
    w_f = M_f / M_f.sum(axis=1, keepdims=True)
    sw_f = np.sqrt(w_f * n_s / (n_s - 1.0))
    k_f = w_f @ K_flat
    m_f = np.sum(w_f * k_f, axis=1)
    _assert_top1_exact(K_flat, Q0_flat, sw_f, k_f, m_f)

    # 2. Near-degenerate lambda1 ~= lambda2 block (gap = 1e-8).
    Q_orth, _ = np.linalg.qr(rng.normal(size=(60, 60)))
    ev_deg = np.concatenate([[1.0, 1.0 - 1e-8, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05], np.linspace(0.04, 0.001, 52)])
    K_deg = (Q_orth * ev_deg[None, :]) @ Q_orth.T
    K_deg = 0.5 * (K_deg + K_deg.T)
    w_d0 = np.full(60, 1.0 / 60.0)
    sw_d0 = np.sqrt(w_d0 * 60.0 / 59.0)
    k_d0 = w_d0 @ K_deg
    m_d0 = float(w_d0 @ k_d0)
    G_d0 = (K_deg - k_d0[:, None] - k_d0[None, :] + m_d0) * (sw_d0[:, None] * sw_d0[None, :])
    _, U_d0 = linalg.eigh(G_d0, subset_by_index=[52, 59], driver="evr")
    Q0_deg = U_d0 / sw_d0[:, None]
    M_d = rng.multinomial(60, w_d0, size=16).astype(np.float64)
    w_d = M_d / M_d.sum(axis=1, keepdims=True)
    sw_d = np.sqrt(w_d * 60.0 / 59.0)
    k_d = w_d @ K_deg
    m_d = np.sum(w_d * k_d, axis=1)
    _assert_top1_exact(K_deg, Q0_deg, sw_d, k_d, m_d)

    # 3. Exact theta1 == theta2 zero-residual case (no 0/0 NaN).
    C60 = np.eye(60) - np.full((60, 60), 1.0 / 60.0)
    Q_c, _ = np.linalg.qr(C60 @ rng.normal(size=(60, 59)))
    ev_eq = np.concatenate([[2.0, 2.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05], np.linspace(0.04, 0.001, 51)])
    K_eq = (Q_c * ev_eq[None, :]) @ Q_c.T
    K_eq = 0.5 * (K_eq + K_eq.T)
    G_eq0 = K_eq * (sw_d0[:, None] * sw_d0[None, :])
    _, U_eq0 = linalg.eigh(G_eq0, subset_by_index=[52, 59], driver="evr")
    Q0_eq = U_eq0 / sw_d0[:, None]
    _assert_top1_exact(K_eq, Q0_eq, sw_d0[None, :], np.zeros((1, 60)), np.zeros(1))

    # 4. Edge cases: B=0, q=1, G=0, and single-slice NaN isolation.
    for xp in ([np, cp] if cp is not None else [np]):
        e_empty = _fast_top1_eigvalsh_from_gram(
            xp, xp.eye(10), xp.zeros((0, 10)), xp.zeros(0), xp.zeros((0, 10)), xp.eye(10)[:, :4]
        )
        assert e_empty.shape == (0,)
        e_q1 = _fast_top1_eigvalsh_from_gram(
            xp, xp.eye(4), xp.zeros((2, 4)), xp.zeros(2), xp.ones((2, 4)) * 0.5, xp.ones((4, 1))
        )
        assert e_q1.shape == (2,) and bool(xp.all(xp.isfinite(e_q1)))
        e_zero = _fast_top1_eigvalsh_from_gram(
            xp, xp.zeros((6, 6)), xp.zeros((2, 6)), xp.zeros(2), xp.ones((2, 6)) * 0.5, xp.eye(6)[:, :4]
        )
        e_zero_np = cp.asnumpy(e_zero) if (cp is not None and xp is cp) else np.asarray(e_zero)
        np.testing.assert_allclose(e_zero_np, 0.0, atol=1e-15)
        sw_nan = np.ones((2, 6), dtype=np.float64) * 0.5
        sw_nan[0] = np.nan
        e_nan = _fast_top1_eigvalsh_from_gram(
            xp, xp.eye(6), xp.zeros((2, 6)), xp.zeros(2), xp.asarray(sw_nan), xp.eye(6)[:, :4]
        )
        e_nan_np = cp.asnumpy(e_nan) if (cp is not None and xp is cp) else np.asarray(e_nan)
        assert np.isnan(e_nan_np[0]) and np.isfinite(e_nan_np[1])

    # 5. 16-nuisance synthetic block & B=8 end-to-end CPU parity vs sequential (include_band_paired=True).
    cfg = PairedLipidConfig()
    ds16, _ = generate_synthetic_cohort(
        n_participants=92,
        n_single_sample=4,
        seed=8302,
        compact_grid=True,
        n_nuisance_modes=16,
        nuisance_part_scale=1.0,
        nuisance_sample_scale=0.5,
        nuisance_decay=0.95,
    )
    gc16 = _get_dataset_gram_cache(ds16, cfg)
    outer16, inner16 = make_nested_splits(ds16, cfg, cfg.primary_seed)
    dc16 = _precompute_division_constants(
        np, gc16, ds16, outer16[0][0][inner16[0][0][0]], outer16[0][0][inner16[0][0][1]], cfg
    )
    M24 = np.zeros((24, ds16.n_participants), dtype=np.float64)
    np.add.at(M24, (np.arange(24)[:, None], rng.integers(0, ds16.n_participants, size=(24, ds16.n_participants))), 1.0)
    M_tr16 = M24[:, dc16["tr_p"]]
    w_s16 = M_tr16[:, dc16["s_rel"]] / (np.sum(M_tr16, axis=1, keepdims=True) * dc16["m_i_div"][None, :])
    w_s16 /= np.maximum(np.sum(w_s16, axis=1, keepdims=True), 1e-12)
    mt16 = np.sum(M_tr16[:, dc16["s_rel"]], axis=1)
    sw16 = np.sqrt(w_s16 * (mt16 / np.maximum(mt16 - 1.0, 1.0))[:, None])
    k16 = w_s16 @ dc16["K1_div"]
    m16 = np.sum(w_s16 * k16, axis=1)
    _assert_top1_exact(dc16["K1_div"], dc16["Q0_1"], sw16, k16, m16)

    y16 = np.asarray(ds16.outcomes[cfg.primary_endpoint], dtype=float)
    M8 = M24[:8]
    fb16: set[int] = set()
    bat16 = _run_batched_multiplicity_draws(
        ds16, y16, outer16, inner16, M8, cfg, gc16, use_gpu=False, fallback_draws_out=fb16, return_oof_sd=True
    )
    assert isinstance(bat16, dict) and fb16 == set()
    for b in range(8):
        seq16 = evaluate_nested_cohort(
            ds16,
            cfg=cfg,
            participant_multiplicities=M8[b],
            precomputed_splits=(outer16, inner16),
            include_band_paired=True,
            compute_diagnostics=False,
        )
        assert float(bat16["delta_pair"][b]) == pytest.approx(
            float(seq16["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"]),
            abs=1e-13,
        )
        assert float(bat16["delta_comp"][b]) == pytest.approx(
            float(seq16["paired_vs_strongest_comparator"]["delta_mse"]),
            abs=1e-13,
        )










