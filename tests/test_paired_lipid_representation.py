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

