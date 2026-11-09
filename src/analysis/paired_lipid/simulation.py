"""Synthetic cohort generator and bootstrap-interval calibration."""

from __future__ import annotations
import math
from typing import Any, Literal
import numpy as np
import pandas as pd
from scipy import linalg, stats

from src.analysis.paired_lipid.cohort import (
    CohortDataset,
    PairedLipidConfig,
)
from src.analysis.paired_lipid.batched import (
    _get_gpu_xp,
)
from src.analysis.paired_lipid.nested_cv import (
    evaluate_nested_cohort,
    run_multiplicity_bootstrap,
)

LABEL_RANGE = 184.0  # 2 samples x 92 participants: synthetic labels match the cohort's label scale
# Disjoint replicate-seed ranges; the independent test cohort uses base_seed (< 10_000).
CALIBRATION_SEED_RANGES: dict[str, tuple[int, int]] = {
    "development": (10_000, 5_000_000),
    "validation": (5_000_000, 10_000_000),
}


def generate_synthetic_cohort(
    *,
    n_participants: int = 92,
    n_single_sample: int = 2,
    maps_per_sample: int = 3,
    seed: int = 42,
    compact_grid: bool = False,
    signal_strength: float = 0.8,
    ab_shift_scale: float = 0.15,
    nonlinear_covariate_proxy: bool = False,
    formula_linked_ldl: bool = False,
    shared_prep_scale: float = 0.3,
    label_drift_scale: float = 0.1,
    tg_ldl_loading: float = 0.35,
    map_dropout_fraction: float = 0.0,
) -> tuple[CohortDataset, dict[str, Any]]:
    """Generate a synthetic paired Raman cohort with known latent biological and technical factors.

    ``tg_ldl_loading`` sets how strongly triglycerides load on the LDL-C biological factor (anchor
    correlation). ``map_dropout_fraction`` is the share of samples that lose one map to QC.
    """
    if n_participants < 2 or not 0 <= n_single_sample <= n_participants or maps_per_sample < 2:
        raise ValueError("Need n_participants >= 2, 0 <= n_single_sample <= n_participants, maps_per_sample >= 2.")
    rng = np.random.default_rng(seed)
    step = 10.0 if compact_grid else 1.0
    wn_1300 = np.arange(734.0, 1788.0 + 0.5 * step, step, dtype=float)
    wn_2850 = np.arange(2451.0, 3197.0 + 0.5 * step, step, dtype=float)

    def _gaussian_peak(axis: np.ndarray, center: float, width: float) -> np.ndarray:
        v = np.exp(-0.5 * ((axis - center) / width) ** 2)
        return v / max(float(linalg.norm(v)), 1e-12)

    # Smooth biological basis functions overlapping lipid-associated and protein-associated regions
    bio_1300_1 = _gaussian_peak(wn_1300, 1440.0, 14.0) + 0.7 * _gaussian_peak(wn_1300, 1301.0, 16.0)
    bio_2850_1 = _gaussian_peak(wn_2850, 2853.0, 12.0) + 0.6 * _gaussian_peak(wn_2850, 2890.0, 18.0)

    bio_1300_2 = _gaussian_peak(wn_1300, 1657.0, 18.0) - 0.5 * _gaussian_peak(wn_1300, 1089.0, 12.0)
    bio_2850_2 = _gaussian_peak(wn_2850, 2937.0, 15.0) - 0.4 * _gaussian_peak(wn_2850, 2853.0, 12.0)

    # Technical nuisance/disagreement basis functions (orthogonalizable or distinct peaks)
    tech_1300 = _gaussian_peak(wn_1300, 1156.0, 20.0) - 0.6 * _gaussian_peak(wn_1300, 980.0, 15.0)
    tech_2850 = _gaussian_peak(wn_2850, 3015.0, 22.0)

    age = rng.normal(68.0, 11.0, size=n_participants)
    sex_female = rng.binomial(1, 0.48, size=n_participants)
    statin = rng.binomial(1, 0.35, size=n_participants)

    # Latent biological factors z_bio (n_participants x 2)
    z_bio = rng.normal(0.0, 1.0, size=(n_participants, 2))
    # Population constants, not sample moments: cohorts drawn with different seeds or sizes
    # (calibration training vs independent test cohort) must share one data-generating process.
    age_std = (age - 68.0) / 11.0
    if nonlinear_covariate_proxy:
        # Failure mode test: spectral factor proxies nonlinear age^2 without independent biological effect on y
        z_bio[:, 0] = 0.85 * (age_std**2 - 1.0) + 0.5 * rng.normal(0.0, 1.0, size=n_participants)

    shared_prep = shared_prep_scale * rng.normal(0.0, 1.0, size=n_participants)

    hdl = np.clip(1.3 + 0.25 * z_bio[:, 1] + 0.1 * sex_female + rng.normal(0.0, 0.2, size=n_participants), 0.5, 3.0)
    tg = np.clip(1.5 + tg_ldl_loading * z_bio[:, 0] - 0.2 * z_bio[:, 1] + rng.normal(0.0, 0.3, size=n_participants), 0.4, 5.0)
    plt_cnt = np.clip(240.0 + 25.0 * shared_prep + rng.normal(0.0, 35.0, size=n_participants), 100.0, 500.0)

    linear_cov = 3.1 - 0.18 * age_std - 0.45 * statin + 0.12 * sex_female
    if nonlinear_covariate_proxy:
        ldl = linear_cov + 0.45 * (age_std**2 - 1.0) + rng.normal(0.0, 0.45, size=n_participants)
    elif formula_linked_ldl:
        total_chol = 4.8 + signal_strength * z_bio[:, 0] - 0.4 * statin + rng.normal(0.0, 0.4, size=n_participants)
        ldl = total_chol - hdl - tg / 2.2
    else:
        ldl = (
            linear_cov
            + signal_strength * z_bio[:, 0]
            + 0.25 * signal_strength * z_bio[:, 1]
            + label_drift_scale * shared_prep
            + rng.normal(0.0, 0.45, size=n_participants)
        )

    single_set = set(range(n_participants - n_single_sample, n_participants))
    sample_counter = 1
    # Labels span a fixed range whatever n is, so label covariates never extrapolate between cohorts.
    label_scale = LABEL_RANGE / float(2 * n_participants)

    n_paired = n_participants - n_single_sample
    n_samples = 2 * n_paired + n_single_sample
    sample_1300 = np.empty((n_samples, wn_1300.size), dtype=float)
    sample_2850 = np.empty((n_samples, wn_2850.size), dtype=float)
    sample_p_idx = np.empty(n_samples, dtype=int)
    sample_ids: list[str] = []
    sample_lbls = np.empty(n_samples, dtype=float)
    sample_counts = np.empty(n_participants, dtype=int)
    paired_p_indices = np.arange(n_paired, dtype=int)
    pair_a_indices = 2 * np.arange(n_paired, dtype=int)
    pair_b_indices = pair_a_indices + 1

    p_ids = np.array([str(100 + i) for i in range(n_participants)], dtype=str)
    sex_arr = np.where(sex_female == 1, "female", "male").astype(object)
    htn_arr = np.empty(n_participants, dtype=np.int64)
    dm_arr = np.empty(n_participants, dtype=np.int64)
    af_arr = np.empty(n_participants, dtype=np.int64)
    cad_arr = np.empty(n_participants, dtype=np.int64)
    smk_list: list[str] = []
    apt_arr = np.empty(n_participants, dtype=np.int64)
    aco_arr = np.empty(n_participants, dtype=np.int64)
    mean_lbl_arr = np.empty(n_participants, dtype=float)

    s_1300_buf = np.empty((maps_per_sample, wn_1300.size), dtype=float)
    s_2850_buf = np.empty((maps_per_sample, wn_2850.size), dtype=float)
    s_cursor = 0

    for i in range(n_participants):
        htn_arr[i] = int(rng.binomial(1, 0.6))
        dm_arr[i] = int(rng.binomial(1, 0.25))
        af_arr[i] = int(rng.binomial(1, 0.2))
        cad_arr[i] = int(rng.binomial(1, 0.18))
        smk_list.append(str(rng.choice(["never", "former", "current"], p=[0.5, 0.3, 0.2])))
        apt_arr[i] = int(rng.binomial(1, 0.3))
        aco_arr[i] = int(rng.binomial(1, 0.15))

        m_i = 1 if i in single_set else 2
        sample_counts[i] = m_i
        lbl_sum = 0.0
        for s_pos in range(m_i):
            lbl_val = sample_counter * label_scale
            sid = f"sample_{sample_counter}"
            lbl_drift = label_drift_scale * (lbl_val / LABEL_RANGE)
            sample_counter += 1

            ab_shift = ab_shift_scale if s_pos == 1 else -ab_shift_scale
            sample_err = rng.normal(0.0, 0.65) + ab_shift + lbl_drift
            base_1300 = (
                (z_bio[i, 0] + shared_prep[i]) * bio_1300_1
                + z_bio[i, 1] * bio_1300_2
                + sample_err * tech_1300
            )
            base_2850 = (
                (z_bio[i, 0] + shared_prep[i]) * bio_2850_1
                + z_bio[i, 1] * bio_2850_2
                + sample_err * tech_2850
            )
            lbl_sum += lbl_val

            for map_idx in range(maps_per_sample):
                s_1300_buf[map_idx] = base_1300 + rng.normal(0.0, 0.04, size=wn_1300.size)
                s_2850_buf[map_idx] = base_2850 + rng.normal(0.0, 0.04, size=wn_2850.size)
            n_kept = maps_per_sample
            if map_dropout_fraction > 0.0 and rng.random() < map_dropout_fraction:
                n_kept -= 1

            sample_1300[s_cursor] = np.median(s_1300_buf[:n_kept], axis=0)
            sample_2850[s_cursor] = np.median(s_2850_buf[:n_kept], axis=0)
            sample_p_idx[s_cursor] = i
            sample_ids.append(sid)
            sample_lbls[s_cursor] = lbl_val
            s_cursor += 1

        mean_lbl_arr[i] = lbl_sum / float(m_i)

    cov_df = pd.DataFrame(
        {
            "age_years": age.astype(float),
            "sex": sex_arr,
            "statin": statin.astype(np.int64),
            "sample_label_trend": mean_lbl_arr,
            "hypertension": htn_arr,
            "diabetes": dm_arr,
            "atrial_fibrillation": af_arr,
            "coronary_artery_disease": cad_arr,
            "smoking_status": pd.Series(smk_list, dtype=object),
            "antiplatelet": apt_arr,
            "anticoagulant": aco_arr,
        }
    )
    dataset = CohortDataset(
        participant_ids=p_ids,
        outcomes={
            "ldl_mmol_l": ldl.astype(float),
            "hdl_mmol_l": hdl.astype(float),
            "triglycerides_mmol_l": tg.astype(float),
            "plt_10e9_l": plt_cnt.astype(float),
        },
        covariate_frame=cov_df,
        wavenumbers_1300=wn_1300,
        wavenumbers_2850=wn_2850,
        sample_1300=sample_1300,
        sample_2850=sample_2850,
        sample_participant_idx=sample_p_idx,
        sample_ids=np.asarray(sample_ids, dtype=str),
        sample_parsed_labels=sample_lbls,
        participant_sample_counts=sample_counts,
        paired_participant_indices=paired_p_indices,
        pair_sample_a_indices=pair_a_indices,
        pair_sample_b_indices=pair_b_indices,
        manifest_summary={
            "subset_mode": "all_eligible",
            "min_maps_per_block": 2,
            "n_eligible_participants": n_participants,
            "n_paired_participants": n_paired,
            "n_single_sample_participants": n_single_sample,
            "n_eligible_samples": n_samples,
            "n_exclusions": 0,
        },
        exclusions=[],
    )

    truth = {
        "bio_1300": np.column_stack([bio_1300_1, bio_1300_2]),
        "bio_2850": np.column_stack([bio_2850_1, bio_2850_2]),
        "z_bio": z_bio,
        "signal_strength": signal_strength,
        "nonlinear_covariate_proxy": nonlinear_covariate_proxy,
    }
    return dataset, truth


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute the Wilson score interval for a binomial proportion."""
    if total <= 0:
        return (float("nan"), float("nan"))
    z = float(stats.norm.ppf(0.5 + 0.5 * confidence))
    phat = float(successes) / float(total)
    denom = 1.0 + (z**2) / float(total)
    center = (phat + (z**2) / (2.0 * float(total))) / denom
    half = (z / denom) * math.sqrt((phat * (1.0 - phat) / float(total)) + ((z**2) / (4.0 * (float(total) ** 2))))
    return (max(0.0, center - half), min(1.0, center + half))


def min_reps_for_gate(wilson_lower_min: float, confidence: float = 0.95) -> int:
    """Smallest repetition count whose Wilson lower bound at 100% coverage reaches the gate."""
    # At n/n coverage the Wilson lower bound is n / (n + z^2).
    z_sq = float(stats.norm.ppf(0.5 + 0.5 * confidence)) ** 2
    return math.ceil(wilson_lower_min * z_sq / (1.0 - wilson_lower_min))


def check_gate_attainable(n_repetitions: int, seed_set: str, wilson_lower_min: float) -> tuple[bool, int]:
    """Return (attainable, minimum reps); raise for a validation run that cannot pass the gate."""
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
    signal_strength: float,
    n_bootstraps: int,
    cfg: PairedLipidConfig,
    use_gpu: bool | None,
    test_ds: CohortDataset,
    compact_grid: bool,
) -> tuple[float, float, float, bool, float]:
    ds, _ = generate_synthetic_cohort(
        n_participants=n_participants,
        seed=rep_seed,
        compact_grid=compact_grid,
        signal_strength=signal_strength,
    )
    boot = run_multiplicity_bootstrap(
        ds,
        cfg=cfg,
        n_bootstraps=n_bootstraps,
        bootstrap_seed=rep_seed + 1,
        use_gpu=use_gpu,
    )
    external_delta = evaluate_nested_cohort(ds, cfg=cfg, external_test=test_ds)["external_delta_mse"]
    return (
        float(boot["point_estimate_delta_mse"]),
        float(boot["ci_95_delta_mse"][0]),
        float(boot["ci_95_delta_mse"][1]),
        bool(boot["interval_procedure_succeeded"]),
        float(external_delta),
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
    seed_set: Literal["development", "validation"] = "development",
    compact_grid: bool = True,
    gate_wilson_lower_min: float = 0.925,
    gate_max_failure_rate: float = 0.01,
) -> dict[str, Any]:
    """Evaluate bootstrap interval coverage, width, bias, and failure rate in synthetic repetitions.

    Target: the expected held-out risk difference MSE_cov - MSE_paired of the procedure at this
    training size, estimated as the replicate mean of each replicate's outer-fold models scored on
    one independent test cohort (seed ``base_seed``, below the replicate seed range).
    Development and validation replicates draw seeds from disjoint ranges; only the validation
    set gives a confirmatory gate result.
    """
    gate_attainable, min_reps = check_gate_attainable(n_repetitions, seed_set, gate_wilson_lower_min)
    lo_seed, hi_seed = CALIBRATION_SEED_RANGES[seed_set]
    rng = np.random.default_rng(base_seed)
    rep_seeds = rng.integers(lo_seed, hi_seed, size=n_repetitions)
    test_ds, _ = generate_synthetic_cohort(
        n_participants=n_test_participants,
        seed=base_seed,
        compact_grid=compact_grid,
        signal_strength=signal_strength,
    )

    gpu_active = (_get_gpu_xp() is not None) and (use_gpu is True or (use_gpu is None and n_bootstraps >= 32))
    if not gpu_active and n_jobs != 1 and (n_repetitions * max(n_bootstraps, 1) >= 200):
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs)(
            delayed(_eval_single_calibration_rep)(
                int(rep_seeds[rep_idx]),
                n_participants,
                signal_strength,
                n_bootstraps,
                cfg,
                False,
                test_ds,
                compact_grid,
            )
            for rep_idx in range(n_repetitions)
        )
    else:
        results = [
            _eval_single_calibration_rep(
                int(rep_seeds[rep_idx]),
                n_participants,
                signal_strength,
                n_bootstraps,
                cfg,
                use_gpu,
                test_ds,
                compact_grid,
            )
            for rep_idx in range(n_repetitions)
        ]

    point_estimates = [r[0] for r in results]
    ci_lows = [r[1] for r in results]
    ci_highs = [r[2] for r in results]
    succeeded_flags = [r[3] for r in results]
    external_deltas = np.asarray([r[4] for r in results], dtype=float)

    target_delta = float(np.mean(external_deltas))
    target_mc_se = float(np.std(external_deltas, ddof=1) / np.sqrt(n_repetitions)) if n_repetitions > 1 else float("nan")
    uncond_covered = 0
    cond_covered = 0
    n_success = 0
    widths: list[float] = []

    for est, lo, hi, ok in zip(point_estimates, ci_lows, ci_highs, succeeded_flags):
        if ok and np.isfinite(lo) and np.isfinite(hi):
            n_success += 1
            widths.append(hi - lo)
            if lo <= target_delta <= hi:
                uncond_covered += 1
                cond_covered += 1

    uncond_cov = float(uncond_covered) / float(n_repetitions) if n_repetitions > 0 else float("nan")
    cond_cov = float(cond_covered) / float(n_success) if n_success > 0 else float("nan")
    fail_rate = float(n_repetitions - n_success) / float(n_repetitions) if n_repetitions > 0 else float("nan")
    wilson_lo, wilson_hi = wilson_score_interval(uncond_covered, n_repetitions, confidence=0.95)
    passes_gate = bool(wilson_lo >= gate_wilson_lower_min and fail_rate < gate_max_failure_rate)

    return {
        "seed_set": seed_set,
        "gate_is_confirmatory": seed_set == "validation",
        "n_repetitions": n_repetitions,
        "n_bootstraps": n_bootstraps,
        "target_delta_mse": target_delta,
        "target_monte_carlo_se": target_mc_se,
        "n_test_participants": n_test_participants,
        "compact_grid": compact_grid,
        "mean_point_estimate": float(np.nanmean(point_estimates)),
        "bias": float(np.nanmean(point_estimates) - target_delta),
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
