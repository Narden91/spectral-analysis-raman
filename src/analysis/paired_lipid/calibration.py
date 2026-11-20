"""Bootstrap interval calibration, feasibility gate checks, and synthetic scenario suites."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.paired_lipid.batched import resolve_array_backend
from src.analysis.paired_lipid.cohort import CohortDataset, PairedLipidConfig
from src.analysis.paired_lipid.metrics import two_way_anova_ms
from src.analysis.paired_lipid.nested_cv import (
    PreparedExternalTest,
    evaluate_nested_cohort,
    run_multiplicity_bootstrap,
    score_external_primary_pair,
)
from src.analysis.paired_lipid.synthetic import generate_synthetic_cohort

SeedSet = Literal["development", "validation"]
# Disjoint replicate-seed ranges; the independent test cohort uses base_seed (< 10_000).
CALIBRATION_SEED_RANGES: dict[SeedSet, tuple[int, int]] = {
    "development": (10_000, 5_000_000),
    "validation": (5_000_000, 10_000_000),
}
_SCENARIO_RESERVED_KEYS: frozenset[str] = frozenset(
    {"scenario", "covariate_preset", "n_participants", "seed", "compact_grid"}
)
_SCENARIO_META_KEYS: tuple[str, ...] = ("scenario", "covariate_preset")


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute the Wilson score interval for a binomial proportion."""
    if total <= 0:
        return (float("nan"), float("nan"))
    z = float(stats.norm.ppf(0.5 + 0.5 * confidence))
    phat = float(successes) / float(total)
    denom = 1.0 + (z**2) / float(total)
    center = (phat + (z**2) / (2.0 * float(total))) / denom
    half = (z / denom) * math.sqrt(
        (phat * (1.0 - phat) / float(total)) + ((z**2) / (4.0 * (float(total) ** 2)))
    )
    return (max(0.0, center - half), min(1.0, center + half))


def min_reps_for_gate(wilson_lower_min: float, confidence: float = 0.95) -> int:
    """Smallest repetition count whose Wilson lower bound at 100% coverage reaches the gate."""
    # At n/n coverage the Wilson lower bound is n / (n + z^2).
    z_sq = float(stats.norm.ppf(0.5 + 0.5 * confidence)) ** 2
    return math.ceil(wilson_lower_min * z_sq / (1.0 - wilson_lower_min))


def check_gate_attainable(n_repetitions: int, seed_set: str, wilson_lower_min: float) -> tuple[bool, int]:
    """Return (attainable, minimum reps); raise for a validation run that cannot pass the gate."""
    if seed_set not in CALIBRATION_SEED_RANGES:
        raise ValueError(
            f"Unknown seed_set: {seed_set!r}. Expected one of {tuple(CALIBRATION_SEED_RANGES)}."
        )
    min_reps = min_reps_for_gate(wilson_lower_min)
    if seed_set == "validation" and n_repetitions < min_reps:
        # Even 100% coverage cannot reach the Wilson lower bound, so refuse before any compute.
        raise ValueError(
            f"Validation calibration needs >= {min_reps} repetitions to be able to pass the gate "
            f"(Wilson lower >= {wilson_lower_min}); got {n_repetitions}."
        )
    return n_repetitions >= min_reps, min_reps


def _eval_single_calibration_rep(
    rep_seed: int,
    n_participants: int,
    gen_kwargs: dict[str, Any],
    n_bootstraps: int,
    cfg: PairedLipidConfig,
    use_gpu: bool | None,
    test_ds: CohortDataset | PreparedExternalTest,
    compact_grid: bool,
) -> tuple[float, float, float, bool, float, np.ndarray]:
    """Run one synthetic calibration replicate: bootstrap interval and external target score."""
    ds, _ = generate_synthetic_cohort(
        n_participants=n_participants,
        seed=rep_seed,
        compact_grid=compact_grid,
        **gen_kwargs,
    )
    boot = run_multiplicity_bootstrap(
        ds,
        cfg=cfg,
        n_bootstraps=n_bootstraps,
        bootstrap_seed=rep_seed + 1,
        use_gpu=use_gpu,
    )
    ext_eval = score_external_primary_pair(ds, test_ds, cfg=cfg)
    return (
        float(boot["point_estimate_delta_mse"]),
        float(boot["ci_95_delta_mse"][0]),
        float(boot["ci_95_delta_mse"][1]),
        bool(boot["interval_procedure_succeeded"]),
        float(ext_eval["external_delta_mse"]),
        np.asarray(ext_eval["external_participant_deltas"], dtype=float),
    )


def evaluate_bootstrap_calibration(
    *,
    n_repetitions: int = 20,
    n_bootstraps: int = 100,
    n_participants: int = 60,
    signal_strength: float = 0.8,
    base_seed: int = 9000,
    cfg: PairedLipidConfig = PairedLipidConfig(),
    n_jobs: int = -1,
    use_gpu: bool | None = None,
    n_test_participants: int = 5000,
    seed_set: SeedSet = "development",
    compact_grid: bool = True,
    gate_wilson_lower_min: float = 0.925,
    gate_max_failure_rate: float = 0.01,
    scenario_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate bootstrap interval coverage, width, bias, and failure rate in synthetic repetitions.

    Target: the expected held-out risk difference MSE_cov - MSE_paired of the procedure at this
    training size, estimated as the replicate mean of each replicate's outer-fold models scored on
    one independent test cohort (seed ``base_seed``, below the replicate seed range).
    Development and validation replicates draw seeds from disjoint ranges; only the validation
    set gives a confirmatory gate result.
    """
    min_rep_seed = CALIBRATION_SEED_RANGES["development"][0]
    if base_seed >= min_rep_seed:
        raise ValueError(
            f"base_seed ({base_seed}) must be < {min_rep_seed} to stay disjoint from replicate seeds."
        )
    gate_attainable, min_reps = check_gate_attainable(n_repetitions, seed_set, gate_wilson_lower_min)
    eff_cfg = (
        replace(cfg, covariate_preset=str(scenario_kwargs["covariate_preset"]))
        if scenario_kwargs is not None and "covariate_preset" in scenario_kwargs
        else cfg
    )
    gen_kwargs: dict[str, Any] = {"signal_strength": float(signal_strength)}
    if scenario_kwargs is not None:
        for k, v in scenario_kwargs.items():
            if k not in _SCENARIO_RESERVED_KEYS:
                gen_kwargs[k] = v

    lo_seed, hi_seed = CALIBRATION_SEED_RANGES[seed_set]
    n_slots = (hi_seed - lo_seed) // 2
    if n_repetitions < 0 or n_repetitions > n_slots:
        raise ValueError(
            f"n_repetitions ({n_repetitions}) must be between 0 and {n_slots} for seed_set={seed_set!r}."
        )
    rng = np.random.default_rng(base_seed)
    # Draw unique even offsets so rep_seed and bootstrap_seed = rep_seed + 1 can never collide.
    rep_seeds = lo_seed + 2 * rng.choice(n_slots, size=n_repetitions, replace=False)

    test_ds, _ = generate_synthetic_cohort(
        n_participants=n_test_participants,
        seed=base_seed,
        compact_grid=compact_grid,
        **gen_kwargs,
    )
    prepared_test = PreparedExternalTest.from_dataset(
        test_ds,
        eff_cfg.primary_endpoint,
        preset=eff_cfg.covariate_preset,
    )
    n_ext = prepared_test.n_participants
    del test_ds

    # For small-to-medium bootstrap counts across multiple replicates, parallel CPU workers
    # outperform serializing replicates through small-batch GPU kernel launches.
    prefer_parallel_cpu = (
        use_gpu is None and n_repetitions >= 4 and n_bootstraps <= 256 and n_jobs != 1
    )
    effective_use_gpu: bool | None = False if prefer_parallel_cpu else use_gpu
    _, gpu_meta = resolve_array_backend(effective_use_gpu, n_bootstraps)
    gpu_active = bool(gpu_meta["gpu_active"])

    if not gpu_active and n_jobs != 1 and n_repetitions > 1 and (n_repetitions * max(n_bootstraps, 1) >= 200):
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs)(
            delayed(_eval_single_calibration_rep)(
                int(rep_seeds[rep_idx]),
                n_participants,
                gen_kwargs,
                n_bootstraps,
                eff_cfg,
                False,
                prepared_test,
                compact_grid,
            )
            for rep_idx in range(n_repetitions)
        )
    else:
        results = [
            _eval_single_calibration_rep(
                int(rep_seeds[rep_idx]),
                n_participants,
                gen_kwargs,
                n_bootstraps,
                eff_cfg,
                effective_use_gpu,
                prepared_test,
                compact_grid,
            )
            for rep_idx in range(n_repetitions)
        ]

    point_estimates = [r[0] for r in results]
    ci_lows = [r[1] for r in results]
    ci_highs = [r[2] for r in results]
    succeeded_flags = [r[3] for r in results]
    external_deltas = np.asarray([r[4] for r in results], dtype=float)

    target_delta = float(np.mean(external_deltas)) if n_repetitions > 0 else float("nan")
    target_mc_se = (
        float(np.std(external_deltas, ddof=1) / math.sqrt(n_repetitions))
        if n_repetitions > 1
        else float("nan")
    )
    if n_repetitions > 0 and n_ext > 1:
        D_mat = np.vstack([r[5] for r in results])
        ms_row, ms_col, ms_err = two_way_anova_ms(D_mat)
        denom_scale = float(n_repetitions * n_ext)
        target_test_se = float(math.sqrt(max(0.0, ms_col) / denom_scale))
        if n_repetitions > 1:
            var_total = max(0.0, (ms_row + ms_col - ms_err) / denom_scale)
            target_total_se = float(math.sqrt(var_total))
        else:
            target_total_se = float("nan")
    else:
        target_test_se = float("nan")
        target_total_se = float("nan")

    uncond_covered = 0
    n_success = 0
    widths: list[float] = []

    for lo, hi, ok in zip(ci_lows, ci_highs, succeeded_flags):
        if ok and np.isfinite(lo) and np.isfinite(hi):
            n_success += 1
            widths.append(hi - lo)
            if lo <= target_delta <= hi:
                uncond_covered += 1

    uncond_cov = float(uncond_covered) / float(n_repetitions) if n_repetitions > 0 else float("nan")
    cond_cov = float(uncond_covered) / float(n_success) if n_success > 0 else float("nan")
    fail_rate = float(n_repetitions - n_success) / float(n_repetitions) if n_repetitions > 0 else float("nan")
    wilson_lo, wilson_hi = wilson_score_interval(uncond_covered, n_repetitions, confidence=0.95)
    passes_gate = bool(wilson_lo >= gate_wilson_lower_min and fail_rate < gate_max_failure_rate)

    finite_points = [v for v in point_estimates if np.isfinite(v)]
    mean_point_est = float(np.mean(finite_points)) if finite_points else float("nan")

    return {
        "seed_set": seed_set,
        "gate_is_confirmatory": seed_set == "validation",
        "n_repetitions": n_repetitions,
        "n_bootstraps": n_bootstraps,
        "gpu_requested": use_gpu,
        "gpu_active": gpu_active,
        "gpu_fallback": bool(gpu_meta["gpu_fallback"]),
        "target_delta_mse": target_delta,
        "target_monte_carlo_se": target_mc_se,
        "target_test_cohort_se": target_test_se,
        "target_total_se": target_total_se,
        "n_test_participants": n_test_participants,
        "compact_grid": compact_grid,
        "mean_point_estimate": mean_point_est,
        "bias": float(mean_point_est - target_delta),
        "unconditional_coverage": uncond_cov,
        "conditional_coverage": cond_cov,
        "wilson_95_lower": wilson_lo,
        "wilson_95_upper": wilson_hi,
        "dataset_failure_rate": fail_rate,
        "mean_interval_width": float(np.mean(widths)) if widths else float("nan"),
        "passes_provisional_gate": passes_gate,
        "gate_attainable": gate_attainable,
        "min_reps_for_gate": min_reps,
    }


def evaluate_scenario_calibration_suite(
    *,
    scenarios: Sequence[Mapping[str, Any]],
    n_repetitions: int = 20,
    n_bootstraps: int = 100,
    n_participants: int = 92,
    base_seed: int = 9000,
    cfg: PairedLipidConfig = PairedLipidConfig(),
    n_jobs: int = -1,
    use_gpu: bool | None = None,
    n_test_participants: int = 5000,
    seed_set: SeedSet = "development",
    compact_grid: bool = True,
    gate_wilson_lower_min: float = 0.925,
    gate_max_failure_rate: float = 0.01,
) -> pd.DataFrame:
    """Evaluate bootstrap interval calibration across multiple synthetic data-generating scenarios."""
    rows: list[dict[str, Any]] = []
    for s_idx, spec in enumerate(scenarios):
        scenario_name = str(spec.get("scenario", f"scenario_{s_idx}"))
        preset_name = str(spec.get("covariate_preset", cfg.covariate_preset))
        sig = float(spec.get("signal_strength", 0.8))
        summary = evaluate_bootstrap_calibration(
            n_repetitions=n_repetitions,
            n_bootstraps=n_bootstraps,
            n_participants=n_participants,
            signal_strength=sig,
            base_seed=base_seed + s_idx,
            cfg=cfg,
            n_jobs=n_jobs,
            use_gpu=use_gpu,
            n_test_participants=n_test_participants,
            seed_set=seed_set,
            compact_grid=compact_grid,
            gate_wilson_lower_min=gate_wilson_lower_min,
            gate_max_failure_rate=gate_max_failure_rate,
            scenario_kwargs=spec,
        )
        rows.append(
            {
                "scenario": scenario_name,
                "covariate_preset": preset_name,
                **summary,
            }
        )
    return pd.DataFrame(rows)


def _ldl_factor_recovery_r2(eval_res: dict[str, Any], z_ldl: np.ndarray) -> float:
    """Mean over outer folds of the held-out R^2 of the true LDL-C factor on the learned paired scores."""
    r2s: list[float] = []
    for te_idx, scores in eval_res["fold_test_paired_scores"]:
        if te_idx.size == 0:
            continue
        y = z_ldl[te_idx]
        X = np.column_stack([np.ones(te_idx.size, dtype=float), scores])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2s.append(1.0 - float(np.sum((y - X @ beta) ** 2)) / ss_tot if ss_tot > 0.0 else float("nan"))
    if not r2s:
        return float("nan")
    arr = np.asarray(r2s, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if finite.size > 0 else float("nan")


class _ScenarioRepResult(NamedTuple):
    """Per-repetition summary metrics for a synthetic scenario evaluation."""

    delta_mse_paired: float
    delta_mse_pca: float
    delta_mse_pls: float
    delta_mse_fixed_band: float
    delta_vs_strongest: float
    mean_spectral_icc: float | None
    recovery_r2_ldl_factor: float
    corr_ldl_tg: float


def _eval_single_scenario_rep(
    seed: int,
    n_participants: int,
    compact_grid: bool,
    spec: Mapping[str, Any],
    cfg: PairedLipidConfig,
) -> _ScenarioRepResult:
    """Evaluate one synthetic scenario replicate on the locked nested-CV pipeline."""
    gen_kwargs = {k: v for k, v in spec.items() if k not in _SCENARIO_META_KEYS}
    ds, truth = generate_synthetic_cohort(
        n_participants=n_participants, seed=seed, compact_grid=compact_grid, **gen_kwargs
    )
    res = evaluate_nested_cohort(ds, cfg=cfg)
    comp = res["comparisons_vs_covariate_ridge"]
    fold_iccs = [
        float(r["paired_spectral_icc"]["icc_a1"])
        for r in res["fold_repeatability"]
        if "paired_spectral_icc" in r and r["paired_spectral_icc"]["estimable"]
    ]
    mean_icc = float(np.mean(fold_iccs)) if fold_iccs else None
    return _ScenarioRepResult(
        delta_mse_paired=float(comp["paired_linear_ridge"]["delta_mse"]),
        delta_mse_pca=float(comp["pca_ridge"]["delta_mse"]),
        delta_mse_pls=float(comp["pls_regression"]["delta_mse"]),
        delta_mse_fixed_band=float(comp["fixed_band_ridge"]["delta_mse"]),
        delta_vs_strongest=float(res["paired_vs_strongest_comparator"]["delta_mse"]),
        mean_spectral_icc=mean_icc,
        recovery_r2_ldl_factor=_ldl_factor_recovery_r2(res, truth["z_bio"][:, 0]),
        corr_ldl_tg=float(np.corrcoef(ds.outcomes["ldl_mmol_l"], ds.outcomes["triglycerides_mmol_l"])[0, 1]),
    )


def run_synthetic_scenario_suite(
    *,
    scenarios: Sequence[Mapping[str, Any]],
    n_participants_grid: Sequence[int],
    n_repetitions: int = 5,
    compact_grid: bool = True,
    base_seed: int = 1000,
    n_jobs: int = -1,
    cfg: PairedLipidConfig = PairedLipidConfig(),
) -> pd.DataFrame:
    """Evaluate every configured synthetic scenario at every participant count across repetitions."""
    tasks = [
        (
            s_idx,
            n_idx,
            base_seed + 1_000_000 * n_idx + 10_000 * s_idx + rep,
            int(n_participants_grid[n_idx]),
            dict(spec),
            replace(cfg, covariate_preset=str(spec["covariate_preset"])),
        )
        for n_idx in range(len(n_participants_grid))
        for s_idx, spec in enumerate(scenarios)
        for rep in range(n_repetitions)
    ]

    if n_jobs > 1 and len(tasks) > 1:
        eff_jobs = min(int(n_jobs), len(tasks))
    elif n_jobs < 0 and (len(tasks) >= 32 or (not compact_grid and len(tasks) >= 8)):
        eff_jobs = min(8, max(1, (os.cpu_count() or 4) // 2), len(tasks))
    else:
        eff_jobs = 1

    if eff_jobs > 1:
        from joblib import Parallel, delayed

        task_results: list[_ScenarioRepResult] = Parallel(n_jobs=eff_jobs)(
            delayed(_eval_single_scenario_rep)(seed, n_p, compact_grid, spec, task_cfg)
            for _, _, seed, n_p, spec, task_cfg in tasks
        )
    else:
        task_results = [
            _eval_single_scenario_rep(seed, n_p, compact_grid, spec, task_cfg)
            for _, _, seed, n_p, spec, task_cfg in tasks
        ]

    grouped: dict[tuple[int, int], list[_ScenarioRepResult]] = {}
    for (s_idx, n_idx, *_), result in zip(tasks, task_results):
        grouped.setdefault((n_idx, s_idx), []).append(result)

    rows: list[dict[str, Any]] = []
    for (n_idx, s_idx), s_res in grouped.items():
        spec = scenarios[s_idx]
        deltas_paired = [r.delta_mse_paired for r in s_res]
        spec_iccs = [r.mean_spectral_icc for r in s_res if r.mean_spectral_icc is not None]
        rec_r2s = [r.recovery_r2_ldl_factor for r in s_res if np.isfinite(r.recovery_r2_ldl_factor)]
        rows.append(
            {
                "scenario": spec["scenario"],
                "covariate_preset": spec["covariate_preset"],
                "n_repetitions": n_repetitions,
                "n_participants": int(n_participants_grid[n_idx]),
                "mean_delta_mse_paired": float(np.mean(deltas_paired)),
                "sd_delta_mse_paired": float(np.std(deltas_paired, ddof=1)) if n_repetitions > 1 else 0.0,
                "mean_delta_vs_strongest_comparator": float(np.mean([r.delta_vs_strongest for r in s_res])),
                "mean_delta_mse_pca": float(np.mean([r.delta_mse_pca for r in s_res])),
                "mean_delta_mse_pls": float(np.mean([r.delta_mse_pls for r in s_res])),
                "mean_delta_mse_fixed_band": float(np.mean([r.delta_mse_fixed_band for r in s_res])),
                "mean_spectral_icc_a1": float(np.mean(spec_iccs)) if spec_iccs else float("nan"),
                "mean_recovery_r2_ldl_factor": float(np.mean(rec_r2s)) if rec_r2s else float("nan"),
                "mean_corr_ldl_tg": float(np.mean([r.corr_ldl_tg for r in s_res])),
            }
        )
    return pd.DataFrame(rows)
