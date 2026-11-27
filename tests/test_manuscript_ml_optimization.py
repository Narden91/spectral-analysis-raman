"""Tests for Milestone M2 manuscript_ml preset trajectory sharing, stage reuse, and subset derivation."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from manuscript_ml import (
    _derive_subset_cohort_dataset,
    clear_dataset_cache,
    load_manuscript_config,
    resolve_experiment_dataset,
)
from manuscript_ml.run_all import run_all_pipeline
from manuscript_ml.run_primary_evaluation import run_primary_evaluation
from manuscript_ml.run_sensitivity_and_secondary import run_sensitivity_and_secondary
from src.analysis.paired_lipid._backend import _get_gpu_xp
from src.analysis.paired_lipid.cohort import (
    COVARIATE_PRESETS,
    SUBSET_MODES,
    CohortDataset,
    PairedLipidConfig,
    build_paired_cohort_dataset,
)
from src.analysis.paired_lipid.nested_cv import evaluate_nested_cohort, run_multiplicity_bootstrap
from src.analysis.paired_lipid.synthetic import generate_synthetic_cohort


def _assert_cohort_datasets_equal(actual: CohortDataset, expected: CohortDataset) -> None:
    assert actual.n_participants == expected.n_participants
    assert actual.n_samples == expected.n_samples
    assert np.array_equal(actual.participant_ids, expected.participant_ids)
    assert set(actual.outcomes) == set(expected.outcomes)
    for ep_name, exp_arr in expected.outcomes.items():
        act_arr = actual.outcomes[ep_name]
        assert np.array_equal(act_arr, exp_arr, equal_nan=True)
        assert act_arr.flags.writeable is False
    assert actual.covariate_frame.equals(expected.covariate_frame)
    for arr_attr in (
        "participant_ids",
        "wavenumbers_1300",
        "wavenumbers_2850",
        "sample_1300",
        "sample_2850",
        "sample_participant_idx",
        "sample_ids",
        "sample_parsed_labels",
        "participant_sample_counts",
        "paired_participant_indices",
        "pair_sample_a_indices",
        "pair_sample_b_indices",
    ):
        act_val = getattr(actual, arr_attr)
        exp_val = getattr(expected, arr_attr)
        if np.issubdtype(act_val.dtype, np.floating) or np.issubdtype(exp_val.dtype, np.floating):
            eq = np.array_equal(act_val, exp_val, equal_nan=True)
        else:
            eq = np.array_equal(act_val, exp_val)
        assert eq, f"Mismatch in {arr_attr}"
        assert act_val.flags.writeable is False, f"{arr_attr} must be read-only"
    assert actual.manifest_summary == expected.manifest_summary
    assert actual.exclusions == expected.exclusions


@pytest.mark.parametrize("n_participants", [32, 48, 92])
@pytest.mark.parametrize("first_mode", ["all_eligible", "paired_only", "single_sample"])
def test_synthetic_subset_derivation_parity_both_cache_orders(
    n_participants: int,
    first_mode: str,
) -> None:
    cfg, _ = load_manuscript_config()
    clear_dataset_cache()
    try:
        modes_order = [first_mode] + [
            m for m in ("all_eligible", "paired_only", "single_sample") if m != first_mode
        ]
        resolved: dict[str, tuple[CohortDataset, dict[str, Any]]] = {}
        for mode in modes_order:
            resolved[mode] = resolve_experiment_dataset(
                use_synthetic=True,
                subset_mode=mode,  # type: ignore[arg-type]
                synthetic_participants=n_participants,
                synthetic_seed=17,
                compact_grid=True,
                cfg=cfg,
            )

        base_ds, base_meta = resolved["all_eligible"]
        base_cov_snapshot = base_ds.covariate_frame.copy(deep=True)
        base_manifest_snapshot = copy.deepcopy(base_ds.manifest_summary)
        base_exclusions_snapshot = copy.deepcopy(base_ds.exclusions)

        for mode in ("all_eligible", "paired_only", "single_sample"):
            ds, meta = resolved[mode]
            ref_ds, ref_truth = generate_synthetic_cohort(
                n_participants=n_participants,
                seed=17,
                compact_grid=True,
                subset_mode=mode,  # type: ignore[arg-type]
                cfg=cfg,
            )
            _assert_cohort_datasets_equal(ds, ref_ds)
            assert meta["signal_strength"] == ref_truth["signal_strength"]

        assert base_ds.covariate_frame.equals(base_cov_snapshot)
        assert base_ds.manifest_summary == base_manifest_snapshot
        assert base_ds.exclusions == base_exclusions_snapshot

        # Returned metadata must be isolated from the internal cache entry.
        base_meta["mutated_marker"] = True
        _, base_meta_again = resolve_experiment_dataset(
            use_synthetic=True,
            subset_mode="all_eligible",
            synthetic_participants=n_participants,
            synthetic_seed=17,
            compact_grid=True,
            cfg=cfg,
        )
        assert "mutated_marker" not in base_meta_again
    finally:
        clear_dataset_cache()


def _build_synthetic_store_fixture() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20261126)
    wn_1300 = np.linspace(734.0, 1788.0, 16)
    wn_2850 = np.linspace(2451.0, 3197.0, 12)

    # Participant ids intentionally cover:
    # P01: identity review exclusion, P02: no_spectra exclusion, P03: unpaired (single sample),
    # P04: no_eligible_samples (<2 maps on its only sample), P05: paired (3 samples -> capped at 2),
    # P06: single sample with non-finite label, P07: paired, P08: no_eligible_samples (0 maps),
    # P09: paired, P10: paired.
    clin_rows = [
        {"participant_id": "P01", "cohort": "stroke", "identity_verified": False, "has_spectra": 1, "age_years": 61.0, "sex": "female", "statin": 1, "ldl_mmol_l": 2.4, "hdl_mmol_l": 1.2, "triglycerides_mmol_l": 1.5, "plt_10e9_l": 210.0},
        {"participant_id": "P02", "cohort": "stroke", "identity_verified": False, "has_spectra": 0, "participant_key": "K02", "age_years": 63.0, "sex": "male", "statin": 0, "ldl_mmol_l": 2.8, "hdl_mmol_l": 1.1, "triglycerides_mmol_l": 1.6, "plt_10e9_l": 220.0},
        {"participant_id": "P03", "cohort": "stroke", "identity_verified": True, "has_spectra": 1, "age_years": 65.0, "sex": "female", "statin": 1, "ldl_mmol_l": 2.6, "hdl_mmol_l": 1.3, "triglycerides_mmol_l": 1.4, "plt_10e9_l": 230.0},
        {"participant_id": "P04", "cohort": "stroke", "identity_verified": True, "has_spectra": 1, "age_years": 67.0, "sex": "male", "statin": 0, "ldl_mmol_l": 3.1, "hdl_mmol_l": 1.0, "triglycerides_mmol_l": 1.8, "plt_10e9_l": 240.0},
        {"participant_id": "P05", "cohort": "stroke", "identity_verified": True, "has_spectra": 1, "age_years": 69.0, "sex": "female", "statin": 1, "ldl_mmol_l": 2.2, "hdl_mmol_l": 1.4, "triglycerides_mmol_l": 1.3, "plt_10e9_l": 250.0},
        {"participant_id": "P06", "cohort": "stroke", "identity_verified": True, "has_spectra": 1, "age_years": 71.0, "sex": "male", "statin": 1, "ldl_mmol_l": 2.9, "hdl_mmol_l": 1.2, "triglycerides_mmol_l": 1.7, "plt_10e9_l": 205.0},
        {"participant_id": "P07", "cohort": "stroke", "identity_verified": True, "has_spectra": 1, "age_years": 73.0, "sex": "female", "statin": 0, "ldl_mmol_l": 3.4, "hdl_mmol_l": 1.1, "triglycerides_mmol_l": 1.9, "plt_10e9_l": 215.0},
        {"participant_id": "P08", "cohort": "stroke", "identity_verified": True, "has_spectra": 1, "age_years": 75.0, "sex": "male", "statin": 0, "ldl_mmol_l": 2.5, "hdl_mmol_l": 1.3, "triglycerides_mmol_l": 1.2, "plt_10e9_l": 225.0},
        {"participant_id": "P09", "cohort": "stroke", "identity_verified": True, "has_spectra": 1, "age_years": 77.0, "sex": "female", "statin": 1, "ldl_mmol_l": 3.0, "hdl_mmol_l": 1.2, "triglycerides_mmol_l": 1.5, "plt_10e9_l": 235.0},
        {"participant_id": "P10", "cohort": "stroke", "identity_verified": True, "has_spectra": 1, "age_years": 79.0, "sex": "male", "statin": 1, "ldl_mmol_l": 2.7, "hdl_mmol_l": 1.4, "triglycerides_mmol_l": 1.1, "plt_10e9_l": 245.0},
    ]
    clin_df = pd.DataFrame(clin_rows)

    sample_map_counts = [
        ("P03", "sample_1", 3),
        ("P04", "sample_2", 1),  # triggers fewer_than_2_maps and no_eligible_samples
        ("P05", "sample_3", 3),
        ("P05", "sample_4", 3),
        ("P05", "sample_5", 3),  # triggers beyond_first_two_samples
        ("P06", "sample_bad", 3),  # non-finite label on single-sample participant
        ("P07", "sample_7", 3),
        ("P07", "sample_8", 3),
        ("P09", "sample_9", 2),
        ("P09", "sample_10", 3),
        ("P10", "sample_11", 3),
        ("P10", "sample_12", 3),
    ]
    map_rows: list[dict[str, Any]] = []
    row_idx = 0
    for pid, sid, n_maps in sample_map_counts:
        for _ in range(n_maps):
            map_rows.append({"participant_id": pid, "sample_id": sid, "spectral_range": "1300", "row_index": row_idx, "retained": True})
            map_rows.append({"participant_id": pid, "sample_id": sid, "spectral_range": "2850", "row_index": row_idx, "retained": True})
            row_idx += 1
    map_df = pd.DataFrame(map_rows)
    spectra_1300 = rng.normal(size=(row_idx, wn_1300.size))
    spectra_2850 = rng.normal(size=(row_idx, wn_2850.size))
    return clin_df, map_df, spectra_1300, spectra_2850, wn_1300, wn_2850


def test_mocked_real_store_derivation_parity_and_single_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import manuscript_ml as mml
    import src.data_preparation.store as store

    clin_df, map_df, s1300, s2850, wn1300, wn2850 = _build_synthetic_store_fixture()
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    master_path = tmp_path / "synthetic_master.csv"
    master_path.write_text("synthetic_fixture\n", encoding="utf-8")
    source = SimpleNamespace(source_path=tmp_path, manifest_hash=None)
    monkeypatch.setattr(store, "get_active_data_source", lambda: source)

    cfg = PairedLipidConfig()
    readiness = {
        "expected_identity_verified": 6,
        "expected_cohort_join": {"n_eligible_participants": 6},
    }
    load_calls: list[str] = []

    def _fake_load_store(*, clinical_master_path: Any, data_source: Any, cfg: PairedLipidConfig, subset_mode: str):
        del clinical_master_path, data_source
        load_calls.append(subset_mode)
        ds = build_paired_cohort_dataset(
            clin_df,
            map_df,
            s1300,
            s2850,
            wn1300,
            wn2850,
            cfg=cfg,
            subset_mode=subset_mode,  # type: ignore[arg-type]
        )
        return ds, {"stroke_identity_verified": 6}

    monkeypatch.setattr(mml, "load_paired_cohort_from_store", _fake_load_store)

    clear_dataset_cache()
    try:
        ds_all, _ = resolve_experiment_dataset(
            use_synthetic=False,
            approve_cohort_evaluation=True,
            clinical_master_path=master_path,
            subset_mode="all_eligible",
            cfg=cfg,
            readiness=readiness,
        )
        ds_paired, _ = resolve_experiment_dataset(
            use_synthetic=False,
            approve_cohort_evaluation=True,
            clinical_master_path=master_path,
            subset_mode="paired_only",
            cfg=cfg,
            readiness=readiness,
        )
        ds_single, _ = resolve_experiment_dataset(
            use_synthetic=False,
            approve_cohort_evaluation=True,
            clinical_master_path=master_path,
            subset_mode="single_sample",
            cfg=cfg,
            readiness=readiness,
        )
        assert load_calls == ["all_eligible"]

        ds_3m, _ = resolve_experiment_dataset(
            use_synthetic=False,
            approve_cohort_evaluation=True,
            clinical_master_path=master_path,
            subset_mode="three_map_complete",
            cfg=cfg,
            readiness=readiness,
        )
        assert load_calls == ["all_eligible", "three_map_complete"]
        assert ds_3m.manifest_summary["min_maps_per_block"] == 3
        assert ds_3m.paired_participant_indices.size < ds_paired.paired_participant_indices.size

        ref_all = build_paired_cohort_dataset(clin_df, map_df, s1300, s2850, wn1300, wn2850, cfg=cfg, subset_mode="all_eligible")
        ref_paired = build_paired_cohort_dataset(clin_df, map_df, s1300, s2850, wn1300, wn2850, cfg=cfg, subset_mode="paired_only")
        ref_single = build_paired_cohort_dataset(clin_df, map_df, s1300, s2850, wn1300, wn2850, cfg=cfg, subset_mode="single_sample")
        _assert_cohort_datasets_equal(ds_all, ref_all)
        _assert_cohort_datasets_equal(ds_paired, ref_paired)
        _assert_cohort_datasets_equal(ds_single, ref_single)

        # Requesting paired_only before all_eligible must also hit load_store only once.
        load_calls.clear()
        clear_dataset_cache()
        ds_paired_first, _ = resolve_experiment_dataset(
            use_synthetic=False,
            approve_cohort_evaluation=True,
            clinical_master_path=master_path,
            subset_mode="paired_only",
            cfg=cfg,
            readiness=readiness,
        )
        ds_all_second, _ = resolve_experiment_dataset(
            use_synthetic=False,
            approve_cohort_evaluation=True,
            clinical_master_path=master_path,
            subset_mode="all_eligible",
            cfg=cfg,
            readiness=readiness,
        )
        assert load_calls == ["all_eligible"]
        _assert_cohort_datasets_equal(ds_paired_first, ref_paired)
        _assert_cohort_datasets_equal(ds_all_second, ref_all)
    finally:
        clear_dataset_cache()


def test_derive_subset_zero_paired_participants_raises_value_error() -> None:
    ds_all, _ = generate_synthetic_cohort(n_participants=20, seed=7, compact_grid=True, subset_mode="all_eligible")
    keep_first = np.cumsum(ds_all.participant_sample_counts) - ds_all.participant_sample_counts
    single_only_ds = CohortDataset(
        participant_ids=ds_all.participant_ids,
        outcomes=ds_all.outcomes,
        covariate_frame=ds_all.covariate_frame.copy(),
        wavenumbers_1300=ds_all.wavenumbers_1300,
        wavenumbers_2850=ds_all.wavenumbers_2850,
        sample_1300=ds_all.sample_1300[keep_first],
        sample_2850=ds_all.sample_2850[keep_first],
        sample_participant_idx=np.arange(ds_all.n_participants, dtype=int),
        sample_ids=ds_all.sample_ids[keep_first],
        sample_parsed_labels=ds_all.sample_parsed_labels[keep_first],
        participant_sample_counts=np.ones(ds_all.n_participants, dtype=int),
        paired_participant_indices=np.empty(0, dtype=int),
        pair_sample_a_indices=np.empty(0, dtype=int),
        pair_sample_b_indices=np.empty(0, dtype=int),
        manifest_summary={**ds_all.manifest_summary, "n_paired_participants": 0},
        exclusions=[],
    )
    with pytest.raises(ValueError, match="No eligible stroke participants remained after linkage and spectral checks."):
        _derive_subset_cohort_dataset(single_only_ds, "paired_only")


@pytest.mark.parametrize("use_gpu", [False, True])
def test_sensitivity_cross_preset_trajectory_sharing_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_gpu: bool,
) -> None:
    if use_gpu and _get_gpu_xp() is None:
        pytest.skip("CUDA/CuPy device not available")

    import manuscript_ml.run_sensitivity_and_secondary as sens_mod

    traj_calls: list[dict[str, Any]] = []
    eval_calls: list[tuple[str, str, bool]] = []
    boot_calls: list[tuple[str, str, bool]] = []

    orig_build_traj = sens_mod.build_spectral_trajectory
    orig_eval = sens_mod.evaluate_nested_cohort
    orig_boot = sens_mod.run_multiplicity_bootstrap

    def _spy_build_traj(*args: Any, **kwargs: Any) -> Any:
        traj_calls.append(dict(kwargs))
        return orig_build_traj(*args, **kwargs)

    def _spy_eval(ds: CohortDataset, *args: Any, **kwargs: Any) -> dict[str, Any]:
        cfg = kwargs["cfg"]
        eval_calls.append(
            (
                str(ds.manifest_summary.get("subset_mode")),
                str(cfg.covariate_preset),
                kwargs.get("spectral_trajectory") is not None,
            )
        )
        return orig_eval(ds, *args, **kwargs)

    def _spy_boot(ds: CohortDataset, *args: Any, **kwargs: Any) -> dict[str, Any]:
        cfg = kwargs["cfg"]
        boot_calls.append(
            (
                str(ds.manifest_summary.get("subset_mode")),
                str(cfg.covariate_preset),
                kwargs.get("spectral_trajectory") is not None,
            )
        )
        return orig_boot(ds, *args, **kwargs)

    monkeypatch.setattr(sens_mod, "build_spectral_trajectory", _spy_build_traj)
    monkeypatch.setattr(sens_mod, "evaluate_nested_cohort", _spy_eval)
    monkeypatch.setattr(sens_mod, "run_multiplicity_bootstrap", _spy_boot)

    n_boot = 12
    clear_dataset_cache()
    try:
        out = sens_mod.run_sensitivity_and_secondary(
            use_synthetic=True,
            compact_grid=True,
            n_bootstraps=n_boot,
            use_gpu=use_gpu,
            output_dir=tmp_path / f"sens_gpu_{use_gpu}",
        )
    finally:
        clear_dataset_cache()

    assert len(traj_calls) == 1
    assert all(has_traj for s_mode, _, has_traj in eval_calls[:5] if s_mode == "all_eligible")
    assert all(not has_traj for s_mode, _, has_traj in eval_calls if s_mode != "all_eligible")
    assert len(boot_calls) == 8  # 5 presets + 3 non-all_eligible subsets
    assert sum(1 for _, _, has_traj in boot_calls if has_traj) == 5

    sens_df = pd.read_csv(out["covariate_and_subset_sensitivity"])
    cfg_base, raw_cfg = load_manuscript_config()
    boot_seed = int(raw_cfg["seeds"]["bootstrap_seed"])
    interval_method = raw_cfg["calibration"].get("interval_method", "studentized_t")

    ds_all, _ = generate_synthetic_cohort(n_participants=92, seed=42, compact_grid=True, subset_mode="all_eligible", cfg=cfg_base)
    for preset in COVARIATE_PRESETS:
        cfg_p = replace(cfg_base, covariate_preset=preset)
        ref_eval = evaluate_nested_cohort(ds_all, cfg=cfg_p, compute_diagnostics=False)
        ref_boot = run_multiplicity_bootstrap(
            ds_all,
            cfg=cfg_p,
            n_bootstraps=n_boot,
            bootstrap_seed=boot_seed,
            interval_method=interval_method,
            use_gpu=use_gpu,
        )
        row = sens_df.loc[(sens_df["sensitivity_axis"] == "covariate_preset") & (sens_df["setting"] == preset)].iloc[0]
        assert abs(float(row["delta_mse_paired"]) - ref_eval["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"]) <= 1e-13
        assert abs(float(row["delta_vs_strongest_comparator"]) - ref_eval["paired_vs_strongest_comparator"]["delta_mse"]) <= 1e-13
        assert abs(float(row["ci_95_delta_mse_low"]) - ref_boot["ci_95_delta_mse"][0]) <= 1e-13
        assert abs(float(row["ci_95_delta_mse_high"]) - ref_boot["ci_95_delta_mse"][1]) <= 1e-13
        assert abs(float(row["ci_95_vs_comparator_low"]) - ref_boot["ci_95_vs_comparator"][0]) <= 1e-13
        assert abs(float(row["ci_95_vs_comparator_high"]) - ref_boot["ci_95_vs_comparator"][1]) <= 1e-13
        assert abs(float(row["ci_95_strongest_vs_covariate_low"]) - ref_boot["ci_95_strongest_vs_covariate"][0]) <= 1e-13
        assert abs(float(row["ci_95_strongest_vs_covariate_high"]) - ref_boot["ci_95_strongest_vs_covariate"][1]) <= 1e-13

    for s_mode in SUBSET_MODES:
        if s_mode == "all_eligible":
            ds_sub = ds_all
        else:
            ds_sub, _ = generate_synthetic_cohort(n_participants=92, seed=42, compact_grid=True, subset_mode=s_mode, cfg=cfg_base)
        is_single = s_mode == "single_sample"
        ref_eval = evaluate_nested_cohort(ds_sub, cfg=cfg_base, single_sample_downstream=is_single, compute_diagnostics=False)
        ref_boot = run_multiplicity_bootstrap(
            ds_sub,
            cfg=cfg_base,
            n_bootstraps=n_boot,
            bootstrap_seed=boot_seed,
            interval_method=interval_method,
            single_sample_downstream=is_single,
            use_gpu=use_gpu,
        )
        row = sens_df.loc[(sens_df["sensitivity_axis"] == "subset_mode") & (sens_df["setting"] == s_mode)].iloc[0]
        assert abs(float(row["delta_mse_paired"]) - ref_eval["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"]) <= 1e-13
        assert abs(float(row["ci_95_delta_mse_low"]) - ref_boot["ci_95_delta_mse"][0]) <= 1e-13
        assert abs(float(row["ci_95_delta_mse_high"]) - ref_boot["ci_95_delta_mse"][1]) <= 1e-13


def test_run_all_stage_reuse_parity_and_memo_skip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import manuscript_ml.run_sensitivity_and_secondary as sens_mod

    absent_master = tmp_path / "absent_clinical_master.csv"
    eval_calls: list[tuple[str, str, str, int]] = []
    boot_calls: list[tuple[str, str, int]] = []
    orig_eval = sens_mod.evaluate_nested_cohort
    orig_boot = sens_mod.run_multiplicity_bootstrap

    def _spy_eval(ds: CohortDataset, *args: Any, **kwargs: Any) -> dict[str, Any]:
        cfg_arg = kwargs["cfg"]
        ep_arg = str(kwargs.get("endpoint") or cfg_arg.primary_endpoint)
        seed_arg = int(kwargs["outer_seed"] if kwargs.get("outer_seed") is not None else cfg_arg.primary_seed)
        eval_calls.append(
            (
                str(ds.manifest_summary.get("subset_mode")),
                str(cfg_arg.covariate_preset),
                ep_arg,
                seed_arg,
            )
        )
        return orig_eval(ds, *args, **kwargs)

    def _spy_boot(ds: CohortDataset, *args: Any, **kwargs: Any) -> dict[str, Any]:
        boot_calls.append((str(ds.manifest_summary.get("subset_mode")), str(kwargs["cfg"].covariate_preset), int(kwargs["n_bootstraps"])))
        return orig_boot(ds, *args, **kwargs)

    monkeypatch.setattr(sens_mod, "evaluate_nested_cohort", _spy_eval)
    monkeypatch.setattr(sens_mod, "run_multiplicity_bootstrap", _spy_boot)

    # Case 1: matching n_bootstraps == sensitivity_bootstraps == 4 reuses both eval_res and boot_res.
    clear_dataset_cache()
    try:
        out_all = run_all_pipeline(
            use_synthetic=True,
            clinical_master_path=absent_master,
            quick=True,
            output_dir=tmp_path / "all_match",
            n_bootstraps=4,
            sensitivity_bootstraps=4,
            n_scenario_reps=1,
            n_calibration_reps=1,
            n_participants=40,
            n_test_participants=80,
            scenario_names=["linear_biological_signal"],
            skip_scenario_suite=True,
            n_jobs=1,
            use_gpu=False,
        )
    finally:
        clear_dataset_cache()

    assert ("all_eligible", "primary", "ldl_mmol_l", 42) not in eval_calls
    assert ("all_eligible", "primary", 4) not in boot_calls
    assert len(eval_calls) == 13  # 14 - 1 reused primary evaluation
    assert len(boot_calls) == 7   # 8 - 1 reused primary bootstrap

    # Standalone stages for reference parity
    monkeypatch.setattr(sens_mod, "evaluate_nested_cohort", orig_eval)
    monkeypatch.setattr(sens_mod, "run_multiplicity_bootstrap", orig_boot)
    sep_dir = tmp_path / "sep_match"
    out_p = run_primary_evaluation(
        use_synthetic=True,
        clinical_master_path=absent_master,
        output_dir=sep_dir,
        compact_grid=True,
        n_bootstraps=4,
        use_gpu=False,
    )
    out_s = run_sensitivity_and_secondary(
        use_synthetic=True,
        clinical_master_path=absent_master,
        output_dir=sep_dir,
        compact_grid=True,
        n_bootstraps=4,
        use_gpu=False,
    )
    df_all_p = pd.read_csv(out_all["comparator_errors_and_differences"])
    df_sep_p = pd.read_csv(out_p["comparator_errors_and_differences"])
    assert np.max(np.abs(df_all_p["delta_mse"].to_numpy() - df_sep_p["delta_mse"].to_numpy())) <= 1e-13

    df_all_s = pd.read_csv(out_all["covariate_and_subset_sensitivity"])
    df_sep_s = pd.read_csv(out_s["covariate_and_subset_sensitivity"])
    for col in ("delta_mse_paired", "ci_95_delta_mse_low", "ci_95_delta_mse_high", "ci_95_vs_comparator_low", "ci_95_vs_comparator_high"):
        assert np.max(np.abs(df_all_s[col].to_numpy() - df_sep_s[col].to_numpy())) <= 1e-13

    # Case 2: differing bootstrap counts (6 vs 2) still reuses eval_res but runs sensitivity bootstrap at B=2.
    eval_calls.clear()
    boot_calls.clear()
    monkeypatch.setattr(sens_mod, "evaluate_nested_cohort", _spy_eval)
    monkeypatch.setattr(sens_mod, "run_multiplicity_bootstrap", _spy_boot)
    clear_dataset_cache()
    try:
        out_diff = run_all_pipeline(
            use_synthetic=True,
            clinical_master_path=absent_master,
            quick=True,
            output_dir=tmp_path / "all_diff",
            n_bootstraps=6,
            sensitivity_bootstraps=2,
            n_scenario_reps=1,
            n_calibration_reps=1,
            n_participants=40,
            n_test_participants=80,
            scenario_names=["linear_biological_signal"],
            skip_scenario_suite=True,
            n_jobs=1,
            use_gpu=False,
        )
    finally:
        clear_dataset_cache()

    assert ("all_eligible", "primary", "ldl_mmol_l", 42) not in eval_calls
    assert ("all_eligible", "primary", 2) in boot_calls
    assert len(boot_calls) == 8

    monkeypatch.setattr(sens_mod, "evaluate_nested_cohort", orig_eval)
    monkeypatch.setattr(sens_mod, "run_multiplicity_bootstrap", orig_boot)
    out_s2 = run_sensitivity_and_secondary(
        use_synthetic=True,
        clinical_master_path=absent_master,
        output_dir=tmp_path / "sep_diff",
        compact_grid=True,
        n_bootstraps=2,
        use_gpu=False,
    )
    df_all_s2 = pd.read_csv(out_diff["covariate_and_subset_sensitivity"])
    df_sep_s2 = pd.read_csv(out_s2["covariate_and_subset_sensitivity"])
    for col in ("delta_mse_paired", "ci_95_delta_mse_low", "ci_95_delta_mse_high"):
        assert np.max(np.abs(df_all_s2[col].to_numpy() - df_sep_s2[col].to_numpy())) <= 1e-13


def test_sensitivity_primary_state_fail_closed_validation(tmp_path: Path) -> None:
    valid_state: dict[str, Any] = {}
    run_primary_evaluation(
        use_synthetic=True,
        compact_grid=True,
        n_bootstraps=4,
        use_gpu=False,
        output_dir=tmp_path / "prim",
        state_out=valid_state,
    )

    bad_cases: list[tuple[str, dict[str, Any]]] = []

    missing_key = dict(valid_state)
    missing_key.pop("eval_res")
    bad_cases.append(("missing required key", missing_key))

    wrong_cfg = dict(valid_state)
    wrong_cfg["cfg"] = replace(valid_state["cfg"], covariate_preset="no_label_trend")
    bad_cases.append(("wrong cfg", wrong_cfg))

    wrong_meta = dict(valid_state)
    wrong_meta["source_meta"] = {**valid_state["source_meta"], "synthetic_seed": 999}
    bad_cases.append(("wrong source_meta", wrong_meta))

    ds_paired, _ = generate_synthetic_cohort(n_participants=92, seed=42, compact_grid=True, subset_mode="paired_only")
    wrong_ds = dict(valid_state)
    wrong_ds["dataset"] = ds_paired
    bad_cases.append(("wrong dataset", wrong_ds))

    wrong_seed = dict(valid_state)
    wrong_seed["outer_seed"] = 101
    bad_cases.append(("wrong outer_seed", wrong_seed))

    wrong_endpoint = dict(valid_state)
    wrong_endpoint["endpoint"] = "hdl_mmol_l"
    bad_cases.append(("wrong endpoint", wrong_endpoint))

    wrong_ss = dict(valid_state)
    wrong_ss["single_sample_downstream"] = True
    bad_cases.append(("single_sample_downstream=True", wrong_ss))

    wrong_missing = dict(valid_state)
    wrong_missing["allow_missing_outcome"] = True
    bad_cases.append(("allow_missing_outcome=True", wrong_missing))

    bad_eval = dict(valid_state)
    bad_eval["eval_res"] = {k: v for k, v in valid_state["eval_res"].items() if k != "oof_spectral_contributions"}
    bad_cases.append(("missing eval_res key", bad_eval))

    bad_boot = dict(valid_state)
    bad_boot["boot_res"] = {k: v for k, v in valid_state["boot_res"].items() if k != "ci_95_delta_mse"}
    bad_cases.append(("missing boot_res key", bad_boot))

    for label, bad_state in bad_cases:
        with pytest.raises(ValueError):
            run_sensitivity_and_secondary(
                use_synthetic=True,
                compact_grid=True,
                n_bootstraps=4,
                use_gpu=False,
                output_dir=tmp_path / f"bad_{abs(hash(label))}",
                primary_state=bad_state,
            )


def test_synthetic_subset_label_scale_and_custom_cfg_parity() -> None:
    from src.analysis.paired_lipid.nested_cv import PreparedExternalTest
    from src.analysis.paired_lipid.synthetic import LABEL_RANGE, generate_synthetic_external_test

    # P1: fixed-range label_scale holds across all subset modes for n != 92 and matches streaming external test.
    for mode in SUBSET_MODES:
        ds_tr, _ = generate_synthetic_cohort(
            n_participants=40,
            seed=21,
            compact_grid=True,
            subset_mode=mode,
            label_drift_scale=0.35,
        )
        tr_lbl = ds_tr.covariate_frame["sample_label_trend"].to_numpy(dtype=float)
        assert float(np.max(tr_lbl)) <= LABEL_RANGE + 1e-12
        assert float(np.max(ds_tr.sample_parsed_labels)) <= LABEL_RANGE + 1e-12

        ds_ext, _ = generate_synthetic_cohort(
            n_participants=200,
            seed=22,
            compact_grid=True,
            subset_mode=mode,
            label_drift_scale=0.35,
        )
        prep_ref = PreparedExternalTest.from_dataset(ds_ext, "ldl_mmol_l", preset="primary")
        prep_str = generate_synthetic_external_test(
            n_participants=200,
            seed=22,
            compact_grid=True,
            subset_mode=mode,
            label_drift_scale=0.35,
            endpoint="ldl_mmol_l",
            preset="primary",
        )
        ext_lbl = prep_str.raw_cache["raw_cov_matrix__primary"][0][:, 3]
        assert float(np.max(ext_lbl)) <= LABEL_RANGE + 1e-12
        assert np.array_equal(
            prep_str.raw_cache["raw_cov_matrix__primary"][0],
            prep_ref.raw_cache["raw_cov_matrix__primary"][0],
            equal_nan=True,
        )

    # n=92 preserves label_scale == 1.0 bitwise.
    ds_92_p, _ = generate_synthetic_cohort(n_participants=92, seed=42, compact_grid=True, subset_mode="paired_only")
    assert np.allclose(ds_92_p.sample_parsed_labels, np.round(ds_92_p.sample_parsed_labels), atol=0.0, rtol=0.0)

    # P4: custom PairedLipidConfig bypasses synthetic subset-derivation shortcut and matches direct generation.
    custom_cfg = PairedLipidConfig(min_maps_per_block=3, secondary_endpoints=("hdl_mmol_l",))
    clear_dataset_cache()
    try:
        for mode in ("paired_only", "single_sample"):
            ds_res, _ = resolve_experiment_dataset(
                use_synthetic=True,
                subset_mode=mode,  # type: ignore[arg-type]
                synthetic_participants=32,
                synthetic_seed=17,
                compact_grid=True,
                cfg=custom_cfg,
            )
            ds_ref, _ = generate_synthetic_cohort(
                n_participants=32,
                seed=17,
                compact_grid=True,
                subset_mode=mode,  # type: ignore[arg-type]
                cfg=custom_cfg,
            )
            _assert_cohort_datasets_equal(ds_res, ds_ref)
    finally:
        clear_dataset_cache()

