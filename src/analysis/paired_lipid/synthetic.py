"""Lightweight synthetic paired Raman cohort generator with known latent factors."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg

from src.analysis.paired_lipid.cohort import (
    SUBSET_MODES,
    CohortDataset,
    PairedLipidConfig,
    SubsetMode,
    build_paired_cohort_dataset,
)

LABEL_RANGE = 184.0  # 2 samples x 92 participants: synthetic labels match the cohort's label scale
DEFAULT_N_SINGLE_SAMPLE = 4  # matches post-QC 88 paired / 4 single-sample cohort mixture
_MAP_CHUNK_SAMPLES = 256


def _gaussian_peak(axis: np.ndarray, center: float, width: float) -> np.ndarray:
    v = np.exp(-0.5 * ((axis - center) / width) ** 2)
    return v / max(float(linalg.norm(v)), 1e-12)


def _build_nuisance_basis(
    wn_1300: np.ndarray,
    wn_2850: np.ndarray,
    n_modes: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth physically motivated Raman nuisance modes across fingerprint and C-H blocks."""
    specs_1300 = (
        ((1445.0, 26.0, 1.0), (1305.0, 24.0, -0.8), (1156.0, 22.0, 0.6)),
        ((1420.0, 95.0, 1.0), (950.0, 80.0, -0.7)),
        ((1156.0, 22.0, 1.0), (980.0, 18.0, -0.6), (1525.0, 20.0, 0.45)),
        ((1580.0, 35.0, 1.0), (1350.0, 45.0, 0.8), (1100.0, 30.0, -0.5)),
        ((1660.0, 26.0, 1.0), (1250.0, 28.0, 0.7), (1555.0, 16.0, -0.5)),
        ((1455.0, 24.0, 1.0), (1310.0, 22.0, -0.8), (1725.0, 20.0, 0.5)),
        ((850.0, 25.0, 1.0), (1045.0, 24.0, 0.7), (786.0, 18.0, -0.6)),
        ((1440.0, 18.0, 0.8), (1657.0, 20.0, -0.7), (1301.0, 18.0, 0.6)),
        ((1180.0, 110.0, 1.0), (1620.0, 90.0, -0.9)),
        ((821.0, 18.0, 0.9), (925.0, 20.0, -0.7), (1060.0, 20.0, 0.5)),
        ((1465.0, 16.0, 0.8), (1258.0, 22.0, -0.7), (1744.0, 15.0, 0.5)),
        ((775.0, 30.0, 0.9), (1380.0, 35.0, -0.6), (1700.0, 30.0, 0.6)),
        ((1200.0, 40.0, 0.7), (1490.0, 35.0, -0.8), (1600.0, 30.0, 0.6)),
        ((890.0, 28.0, 0.8), (1280.0, 25.0, 0.6), (1540.0, 22.0, -0.7)),
        ((1015.0, 22.0, 0.9), (1330.0, 30.0, -0.6), (1690.0, 25.0, 0.5)),
        ((755.0, 20.0, 0.8), (1130.0, 25.0, -0.7), (1410.0, 28.0, 0.6)),
    )
    specs_2850 = (
        ((2858.0, 20.0, 1.0), (2895.0, 22.0, -0.75)),
        ((2880.0, 110.0, 1.0), (2600.0, 90.0, -0.6)),
        ((3015.0, 25.0, 1.0), (2960.0, 22.0, -0.5)),
        ((2937.0, 24.0, 1.0), (2850.0, 24.0, -0.7)),
        ((2895.0, 28.0, 1.0), (2975.0, 30.0, 0.6)),
        ((2855.0, 22.0, -0.8), (2910.0, 26.0, 0.9)),
        ((2740.0, 65.0, 1.0), (3050.0, 55.0, -0.7)),
        ((2865.0, 20.0, 0.8), (2945.0, 22.0, -0.8)),
        ((2900.0, 95.0, 1.0), (2550.0, 80.0, -0.8)),
        ((2845.0, 18.0, 0.8), (2925.0, 20.0, -0.7)),
        ((2885.0, 24.0, 0.7), (2965.0, 25.0, -0.8)),
        ((2660.0, 50.0, 0.9), (3080.0, 50.0, -0.7)),
        ((2810.0, 35.0, 0.8), (2990.0, 35.0, -0.7)),
        ((2700.0, 40.0, 0.8), (2915.0, 25.0, -0.7)),
        ((2780.0, 30.0, 0.9), (3030.0, 35.0, -0.6)),
        ((2620.0, 45.0, 0.7), (2875.0, 30.0, 0.8)),
    )
    b1: list[np.ndarray] = []
    b2: list[np.ndarray] = []
    for m in range(n_modes):
        v1 = sum(
            (w * _gaussian_peak(wn_1300, c, sig) for c, sig, w in specs_1300[m % len(specs_1300)]),
            np.zeros_like(wn_1300),
        )
        v2 = sum(
            (w * _gaussian_peak(wn_2850, c, sig) for c, sig, w in specs_2850[m % len(specs_2850)]),
            np.zeros_like(wn_2850),
        )
        b1.append(v1 / max(float(linalg.norm(v1)), 1e-12))
        b2.append(v2 / max(float(linalg.norm(v2)), 1e-12))
    return np.vstack(b1), np.vstack(b2)


def _median_maps(maps: np.ndarray, out: np.ndarray, drop: np.ndarray | None) -> None:
    """Compute per-sample map median along axis 1, dropping the last map where requested."""
    if maps.shape[1] == 3:
        if drop is not None and np.any(drop):
            kept = ~drop
            if np.any(kept):
                m0 = maps[kept, 0, :]
                m1 = maps[kept, 1, :]
                m2 = maps[kept, 2, :]
                out[kept] = np.maximum(np.minimum(m0, m1), np.minimum(np.maximum(m0, m1), m2))
            out[drop] = 0.5 * (maps[drop, 0, :] + maps[drop, 1, :])
        else:
            m0, m1, m2 = maps[:, 0, :], maps[:, 1, :], maps[:, 2, :]
            np.maximum(np.minimum(m0, m1), np.minimum(np.maximum(m0, m1), m2), out=out)
        return
    if drop is not None and np.any(drop):
        kept = ~drop
        if np.any(kept):
            out[kept] = np.median(maps[kept], axis=1)
        out[drop] = np.median(maps[drop, :-1, :], axis=1)
    else:
        out[:] = np.median(maps, axis=1)


def generate_synthetic_cohort(
    *,
    n_participants: int = 92,
    n_single_sample: int = DEFAULT_N_SINGLE_SAMPLE,
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
    subset_mode: SubsetMode = "all_eligible",
    cfg: PairedLipidConfig | None = None,
    n_nuisance_modes: int = 0,
    nuisance_part_scale: float = 0.0,
    nuisance_sample_scale: float = 0.0,
    nuisance_decay: float = 0.88,
    bio_loading_scale: float = 1.0,
    bio_sample_noise: float = 0.0,
    include_shoulder_in_bio: bool = False,
    correlated_tech_mode_rho: float = 0.0,
    unequal_b_error_scale: float = 0.0,
    nonlinear_spectral_signal: bool = False,
    map_spatial_corr: float = 0.0,
    map_spatial_scale: float = 0.0,
    covariate_missing_rate: float = 0.0,
    qc_sample_dropout_fraction: float = 0.0,
) -> tuple[CohortDataset, dict[str, Any]]:
    """Generate a synthetic paired Raman cohort with known latent biological and technical factors.

    ``tg_ldl_loading`` sets how strongly triglycerides load on the LDL-C biological factor (anchor
    correlation). ``map_dropout_fraction`` is the share of samples that lose one map to QC.
    ``qc_sample_dropout_fraction`` is the share of paired participants whose second technical sample
    falls below the 2-map QC threshold and becomes single-sample. ``subset_mode`` applies cohort
    subsetting directly at raw-map level when not ``all_eligible``.
    """
    if subset_mode not in SUBSET_MODES:
        raise ValueError(f"Unknown subset_mode: {subset_mode!r}. Expected one of {SUBSET_MODES}.")
    if n_participants < 2 or not 0 <= n_single_sample <= n_participants or maps_per_sample < 2:
        raise ValueError("Need n_participants >= 2, 0 <= n_single_sample <= n_participants, maps_per_sample >= 2.")
    if n_nuisance_modes < 0:
        raise ValueError(f"n_nuisance_modes must be non-negative; got {n_nuisance_modes}.")
    if not (0.0 <= covariate_missing_rate < 1.0):
        raise ValueError(f"covariate_missing_rate must lie in [0, 1); got {covariate_missing_rate}.")
    if not (0.0 <= qc_sample_dropout_fraction < 1.0):
        raise ValueError(
            f"qc_sample_dropout_fraction must lie in [0, 1); got {qc_sample_dropout_fraction}."
        )

    rng = np.random.default_rng(seed)
    step = 10.0 if compact_grid else 1.0
    wn_1300 = np.arange(734.0, 1788.0 + 0.5 * step, step, dtype=float)
    wn_2850 = np.arange(2451.0, 3197.0 + 0.5 * step, step, dtype=float)

    # Smooth biological basis functions overlapping lipid-associated and protein-associated regions.
    if include_shoulder_in_bio:
        b1_raw = (
            _gaussian_peak(wn_1300, 1440.0, 14.0)
            + 0.7 * _gaussian_peak(wn_1300, 1301.0, 16.0)
            + 0.65 * _gaussian_peak(wn_1300, 1412.0, 14.0)
            - 0.6 * _gaussian_peak(wn_1300, 1355.0, 15.0)
            + 0.5 * _gaussian_peak(wn_1300, 1618.0, 16.0)
        )
        b2_raw = (
            _gaussian_peak(wn_2850, 2853.0, 12.0)
            + 0.6 * _gaussian_peak(wn_2850, 2890.0, 18.0)
            - 0.65 * _gaussian_peak(wn_2850, 2918.0, 14.0)
            + 0.55 * _gaussian_peak(wn_2850, 2832.0, 12.0)
        )
        bio_1300_1 = b1_raw / max(float(linalg.norm(b1_raw)), 1e-12) * math.sqrt(1.49)
        bio_2850_1 = b2_raw / max(float(linalg.norm(b2_raw)), 1e-12) * math.sqrt(1.36)
    else:
        bio_1300_1 = _gaussian_peak(wn_1300, 1440.0, 14.0) + 0.7 * _gaussian_peak(wn_1300, 1301.0, 16.0)
        bio_2850_1 = _gaussian_peak(wn_2850, 2853.0, 12.0) + 0.6 * _gaussian_peak(wn_2850, 2890.0, 18.0)

    bio_1300_2 = _gaussian_peak(wn_1300, 1657.0, 18.0) - 0.5 * _gaussian_peak(wn_1300, 1089.0, 12.0)
    bio_2850_2 = _gaussian_peak(wn_2850, 2937.0, 15.0) - 0.4 * _gaussian_peak(wn_2850, 2853.0, 12.0)

    # Technical nuisance/disagreement basis functions (orthogonalizable or distinct peaks).
    tech_1300 = _gaussian_peak(wn_1300, 1156.0, 20.0) - 0.6 * _gaussian_peak(wn_1300, 980.0, 15.0)
    tech_2850 = _gaussian_peak(wn_2850, 3015.0, 22.0)

    age = rng.normal(68.0, 11.0, size=n_participants)
    sex_female = rng.binomial(1, 0.48, size=n_participants)
    statin = rng.binomial(1, 0.35, size=n_participants)

    # Latent biological factors z_bio (n_participants x 2).
    z_bio = rng.normal(0.0, 1.0, size=(n_participants, 2))
    # Population constants, not sample moments: cohorts drawn with different seeds or sizes
    # (calibration training vs independent test cohort) must share one data-generating process.
    age_std = (age - 68.0) / 11.0
    if nonlinear_covariate_proxy:
        # Failure mode test: spectral factor proxies nonlinear age^2 without independent biological effect on y.
        z_bio[:, 0] = 0.85 * (age_std**2 - 1.0) + 0.5 * rng.normal(0.0, 1.0, size=n_participants)

    shared_prep = shared_prep_scale * rng.normal(0.0, 1.0, size=n_participants)

    hdl = np.clip(1.3 + 0.25 * z_bio[:, 1] + 0.1 * sex_female + rng.normal(0.0, 0.2, size=n_participants), 0.5, 3.0)
    tg = np.clip(1.5 + tg_ldl_loading * z_bio[:, 0] - 0.2 * z_bio[:, 1] + rng.normal(0.0, 0.3, size=n_participants), 0.4, 5.0)
    plt_cnt = np.clip(240.0 + 25.0 * shared_prep + rng.normal(0.0, 35.0, size=n_participants), 100.0, 500.0)

    p_ids = np.arange(100, 100 + n_participants, dtype=int).astype(str)

    # Randomize single-sample participant positions using a dedicated SeedSequence stream so layout
    # randomization stays independent of the main biological/outcome RNG stream across seeds.
    layout_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 1009]))
    sample_counts = np.full(n_participants, 2, dtype=int)
    if n_single_sample > 0:
        sample_counts[layout_rng.choice(n_participants, size=n_single_sample, replace=False)] = 1

    qc_dropped_paired_p: np.ndarray = np.empty(0, dtype=int)
    exclusions: list[dict[str, Any]] = []
    if qc_sample_dropout_fraction > 0.0:
        paired_init = np.flatnonzero(sample_counts == 2)
        if paired_init.size > 2:
            qc_drop_flags = layout_rng.random(paired_init.size) < qc_sample_dropout_fraction
            cand_drop = paired_init[qc_drop_flags]
            max_drop = max(0, int(paired_init.size) - 2)
            if cand_drop.size > max_drop:
                cand_drop = cand_drop[:max_drop]
            qc_dropped_paired_p = cand_drop
            if subset_mode == "all_eligible" and qc_dropped_paired_p.size > 0:
                sample_counts[qc_dropped_paired_p] = 1
                for p_i in qc_dropped_paired_p:
                    exclusions.append(
                        {
                            "participant_id": str(p_ids[int(p_i)]),
                            "sample_id": f"qc_dropped_sample_{int(p_i)}",
                            "level": "sample",
                            "reason": "fewer_than_2_maps",
                        }
                    )

    n_samples = int(np.sum(sample_counts))
    sample_p_idx = np.repeat(np.arange(n_participants, dtype=int), sample_counts)
    paired_p_list: list[int] = []
    pair_a_list: list[int] = []
    pair_b_list: list[int] = []
    single_s_list: list[int] = []
    pos = 0
    for p_i in range(n_participants):
        if sample_counts[p_i] == 2:
            paired_p_list.append(p_i)
            pair_a_list.append(pos)
            pair_b_list.append(pos + 1)
            pos += 2
        else:
            single_s_list.append(pos)
            pos += 1
    paired_p_indices = np.asarray(paired_p_list, dtype=int)
    pair_a_indices = np.asarray(pair_a_list, dtype=int)
    pair_b_indices = np.asarray(pair_b_list, dtype=int)
    single_s_indices = np.asarray(single_s_list, dtype=int)
    n_paired = int(paired_p_indices.size)
    n_single_actual = n_participants - n_paired

    # Labels span a fixed range whatever n is, so label covariates never extrapolate between cohorts.
    label_scale = LABEL_RANGE / float(2 * n_participants)
    sample_lbls = np.arange(1, n_samples + 1, dtype=float) * label_scale
    mean_lbl_arr = np.empty(n_participants, dtype=float)
    if paired_p_indices.size > 0:
        mean_lbl_arr[paired_p_indices] = 0.5 * (sample_lbls[pair_a_indices] + sample_lbls[pair_b_indices])
    if single_s_indices.size > 0:
        mean_lbl_arr[np.flatnonzero(sample_counts == 1)] = sample_lbls[single_s_indices]
    # Unit-variance centered label trend independent of n_participants so train and test cohorts
    # share one data-generating process while coupling technical label drift to LDL-C.
    label_trend_std = math.sqrt(12.0) * (mean_lbl_arr / LABEL_RANGE - 0.5)

    linear_cov = 3.1 - 0.18 * age_std - 0.45 * statin + 0.12 * sex_female
    if nonlinear_covariate_proxy:
        ldl = linear_cov + 0.45 * (age_std**2 - 1.0) + rng.normal(0.0, 0.45, size=n_participants)
    elif nonlinear_spectral_signal:
        nl_sig = (z_bio[:, 0] ** 2 - 1.0) / math.sqrt(2.0) + 0.7 * (z_bio[:, 0] * z_bio[:, 1])
        ldl = (
            linear_cov
            + signal_strength * nl_sig
            + label_drift_scale * label_trend_std
            + rng.normal(0.0, 0.45, size=n_participants)
        )
    elif formula_linked_ldl:
        total_chol = (
            4.8
            + signal_strength * z_bio[:, 0]
            - 0.4 * statin
            + label_drift_scale * label_trend_std
            + rng.normal(0.0, 0.4, size=n_participants)
        )
        ldl = total_chol - hdl - tg / 2.2
    else:
        ldl = (
            linear_cov
            + signal_strength * z_bio[:, 0]
            + 0.25 * signal_strength * z_bio[:, 1]
            + label_drift_scale * label_trend_std
            + rng.normal(0.0, 0.45, size=n_participants)
        )

    sex_arr = np.where(sex_female == 1, "female", "male").astype(object)
    age_cov = age.astype(float).copy()
    statin_cov = statin.astype(float).copy()
    if covariate_missing_rate > 0.0:
        miss_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 2017]))
        age_cov[miss_rng.random(n_participants) < covariate_missing_rate] = np.nan
        statin_cov[miss_rng.random(n_participants) < covariate_missing_rate] = np.nan
        sex_arr[miss_rng.random(n_participants) < (0.5 * covariate_missing_rate)] = np.nan

    htn_arr = rng.binomial(1, 0.6, size=n_participants).astype(np.int64)
    dm_arr = rng.binomial(1, 0.25, size=n_participants).astype(np.int64)
    af_arr = rng.binomial(1, 0.2, size=n_participants).astype(np.int64)
    cad_arr = rng.binomial(1, 0.18, size=n_participants).astype(np.int64)
    smk_arr = rng.choice(
        np.array(["never", "former", "current"], dtype=object),
        size=n_participants,
        p=[0.5, 0.3, 0.2],
    )
    apt_arr = rng.binomial(1, 0.3, size=n_participants).astype(np.int64)
    aco_arr = rng.binomial(1, 0.15, size=n_participants).astype(np.int64)

    sample_ids = np.array([f"sample_{k}" for k in range(1, n_samples + 1)], dtype=str)

    ab_shifts = np.zeros(n_samples, dtype=float)
    ab_shifts[pair_a_indices] = -ab_shift_scale
    ab_shifts[pair_b_indices] = ab_shift_scale
    if single_s_indices.size > 0:
        ab_shifts[single_s_indices] = (
            layout_rng.choice(np.array([-1.0, 1.0]), size=single_s_indices.size) * ab_shift_scale
        )
    lbl_drift = label_drift_scale * math.sqrt(12.0) * (sample_lbls / LABEL_RANGE - 0.5)
    sample_err = rng.normal(0.0, 0.65, size=n_samples) + ab_shifts + lbl_drift
    dropout_mask = (rng.random(n_samples) < map_dropout_fraction) if map_dropout_fraction > 0.0 else None

    z0_base = (z_bio[:, 0] + shared_prep)[sample_p_idx]
    z1_base = z_bio[sample_p_idx, 1]
    if bio_sample_noise > 0.0:
        z0_s = bio_loading_scale * (z0_base + rng.normal(0.0, bio_sample_noise, size=n_samples))
        z1_s = bio_loading_scale * (z1_base + rng.normal(0.0, bio_sample_noise, size=n_samples))
    else:
        z0_s = bio_loading_scale * z0_base
        z1_s = bio_loading_scale * z1_base
    if unequal_b_error_scale > 0.0 and pair_b_indices.size > 0:
        z0_s = z0_s.copy()
        z1_s = z1_s.copy()
        z0_s[pair_b_indices] += rng.normal(0.0, unequal_b_error_scale, size=pair_b_indices.size)
        z1_s[pair_b_indices] += rng.normal(0.0, unequal_b_error_scale, size=pair_b_indices.size)

    factors_list: list[np.ndarray] = [z0_s, z1_s, sample_err]
    b1_list: list[np.ndarray] = [bio_1300_1, bio_1300_2, tech_1300]
    b2_list: list[np.ndarray] = [bio_2850_1, bio_2850_2, tech_2850]

    if n_nuisance_modes > 0 and (nuisance_part_scale > 0.0 or nuisance_sample_scale > 0.0):
        nuis_b1, nuis_b2 = _build_nuisance_basis(wn_1300, wn_2850, n_modes=n_nuisance_modes)
        decay_vec = nuisance_decay ** np.arange(n_nuisance_modes, dtype=float)
        nuis_p = (
            rng.normal(0.0, 1.0, size=(n_participants, n_nuisance_modes))
            * (nuisance_part_scale * decay_vec)[None, :]
        )
        nuis_s = (
            rng.normal(0.0, 1.0, size=(n_samples, n_nuisance_modes))
            * (nuisance_sample_scale * decay_vec)[None, :]
        )
        if abs(correlated_tech_mode_rho) > 0.0 and pair_a_indices.size > 0:
            rho = float(np.clip(correlated_tech_mode_rho, -0.99, 0.99))
            corr_shared = (
                rng.normal(0.0, 1.0, size=(paired_p_indices.size, n_nuisance_modes))
                * (nuisance_sample_scale * decay_vec)[None, :]
            )
            nuis_s[pair_a_indices] = (
                math.sqrt(abs(rho)) * corr_shared + math.sqrt(1.0 - abs(rho)) * nuis_s[pair_a_indices]
            )
            nuis_s[pair_b_indices] = (
                math.copysign(1.0, rho) * math.sqrt(abs(rho)) * corr_shared
                + math.sqrt(1.0 - abs(rho)) * nuis_s[pair_b_indices]
            )
        if unequal_b_error_scale > 0.0 and pair_b_indices.size > 0:
            nuis_s[pair_b_indices] *= 1.0 + unequal_b_error_scale
        factors_list.append(nuis_p[sample_p_idx] + nuis_s)
        b1_list.append(nuis_b1)
        b2_list.append(nuis_b2)

    sample_factors = np.column_stack(factors_list)
    basis_1300 = np.vstack(b1_list)
    basis_2850 = np.vstack(b2_list)

    if map_spatial_scale > 0.0:
        nuis_b1_m, nuis_b2_m = _build_nuisance_basis(wn_1300, wn_2850, n_modes=4)
    else:
        nuis_b1_m = nuis_b2_m = None

    cov_df = pd.DataFrame(
        {
            "age_years": age_cov,
            "sex": sex_arr,
            "statin": statin_cov if covariate_missing_rate > 0.0 else statin.astype(np.int64),
            "sample_label_trend": mean_lbl_arr,
            "hypertension": htn_arr,
            "diabetes": dm_arr,
            "atrial_fibrillation": af_arr,
            "coronary_artery_disease": cad_arr,
            "smoking_status": pd.Series(smk_arr, dtype=object),
            "antiplatelet": apt_arr,
            "anticoagulant": aco_arr,
        }
    )

    if subset_mode == "all_eligible":
        sample_1300 = np.empty((n_samples, wn_1300.size), dtype=float)
        sample_2850 = np.empty((n_samples, wn_2850.size), dtype=float)
        for s0 in range(0, n_samples, _MAP_CHUNK_SAMPLES):
            s1 = min(s0 + _MAP_CHUNK_SAMPLES, n_samples)
            c_len = s1 - s0
            base_1300 = sample_factors[s0:s1] @ basis_1300
            base_2850 = sample_factors[s0:s1] @ basis_2850
            # Median is translation-invariant: compute it on zero-mean map noise, then add base in 2D.
            noise_1300 = rng.normal(0.0, 0.04, size=(c_len, maps_per_sample, wn_1300.size))
            noise_2850 = rng.normal(0.0, 0.04, size=(c_len, maps_per_sample, wn_2850.size))
            if map_spatial_scale > 0.0 and nuis_b1_m is not None and nuis_b2_m is not None:
                r_mc = float(np.clip(map_spatial_corr, 0.0, 0.99))
                z_shared = rng.normal(0.0, map_spatial_scale, size=(c_len, 1, 4))
                z_indep = rng.normal(0.0, map_spatial_scale, size=(c_len, maps_per_sample, 4))
                z_map = math.sqrt(r_mc) * z_shared + math.sqrt(1.0 - r_mc) * z_indep
                noise_1300 += z_map @ nuis_b1_m
                noise_2850 += z_map @ nuis_b2_m
            drop_local = dropout_mask[s0:s1] if dropout_mask is not None else None
            _median_maps(noise_1300, sample_1300[s0:s1], drop_local)
            sample_1300[s0:s1] += base_1300
            _median_maps(noise_2850, sample_2850[s0:s1], drop_local)
            sample_2850[s0:s1] += base_2850

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
            sample_ids=sample_ids,
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
                "n_single_sample_participants": n_single_actual,
                "n_eligible_samples": n_samples,
                "n_exclusions": len(exclusions),
            },
            exclusions=exclusions,
        )
        z_bio_out = z_bio
    else:
        n_maps_per_sample = np.full(n_samples, maps_per_sample, dtype=int)
        if dropout_mask is not None:
            n_maps_per_sample[dropout_mask] = maps_per_sample - 1
        if qc_dropped_paired_p.size > 0:
            # Drop sample B to 1 map (< min_maps=2) so raw-map cohort assembly excludes it cleanly.
            drop_p_set = set(int(p) for p in qc_dropped_paired_p)
            for p_i, s_b in zip(paired_p_indices.tolist(), pair_b_indices.tolist()):
                if p_i in drop_p_set:
                    n_maps_per_sample[s_b] = 1
        if subset_mode == "three_map_complete" and map_dropout_fraction == 0.0:
            # Deterministic 20% 2-map dropout on raw maps so min_maps=3 exercises exclusions cleanly.
            map_drop_rng = np.random.default_rng(seed + 7)
            two_map_drop = map_drop_rng.random(n_samples) < 0.2
            n_maps_per_sample[two_map_drop] = np.minimum(n_maps_per_sample[two_map_drop], min(2, maps_per_sample))

        raw_chunks_1300: list[np.ndarray] = []
        raw_chunks_2850: list[np.ndarray] = []
        map_idx_axis = np.arange(maps_per_sample, dtype=int)[None, :]
        for s0 in range(0, n_samples, _MAP_CHUNK_SAMPLES):
            s1 = min(s0 + _MAP_CHUNK_SAMPLES, n_samples)
            c_len = s1 - s0
            base_1300 = sample_factors[s0:s1] @ basis_1300
            base_2850 = sample_factors[s0:s1] @ basis_2850
            maps_1300 = base_1300[:, None, :] + rng.normal(
                0.0, 0.04, size=(c_len, maps_per_sample, wn_1300.size)
            )
            maps_2850 = base_2850[:, None, :] + rng.normal(
                0.0, 0.04, size=(c_len, maps_per_sample, wn_2850.size)
            )
            if map_spatial_scale > 0.0 and nuis_b1_m is not None and nuis_b2_m is not None:
                r_mc = float(np.clip(map_spatial_corr, 0.0, 0.99))
                z_shared = rng.normal(0.0, map_spatial_scale, size=(c_len, 1, 4))
                z_indep = rng.normal(0.0, map_spatial_scale, size=(c_len, maps_per_sample, 4))
                z_map = math.sqrt(r_mc) * z_shared + math.sqrt(1.0 - r_mc) * z_indep
                maps_1300 += z_map @ nuis_b1_m
                maps_2850 += z_map @ nuis_b2_m
            keep_map = map_idx_axis < n_maps_per_sample[s0:s1, None]
            raw_chunks_1300.append(maps_1300[keep_map])
            raw_chunks_2850.append(maps_2850[keep_map])

        raw_1300 = np.vstack(raw_chunks_1300)
        raw_2850 = np.vstack(raw_chunks_2850)
        sample_idx_per_map = np.repeat(np.arange(n_samples, dtype=int), n_maps_per_sample)
        total_maps = int(sample_idx_per_map.size)
        map_pids = p_ids[sample_p_idx[sample_idx_per_map]]
        map_sids = sample_ids[sample_idx_per_map]
        row_indices = np.arange(total_maps, dtype=int)
        map_df = pd.DataFrame(
            {
                "participant_id": np.repeat(map_pids, 2),
                "sample_id": np.repeat(map_sids, 2),
                "spectral_range": np.tile(np.array(["1300", "2850"], dtype=object), total_maps),
                "row_index": np.repeat(row_indices, 2),
                "retained": np.ones(2 * total_maps, dtype=bool),
            }
        )

        clin_df = cov_df.copy()
        clin_df["participant_id"] = p_ids
        clin_df["cohort"] = "stroke"
        clin_df["identity_verified"] = True
        clin_df["ldl_mmol_l"] = ldl.astype(float)
        clin_df["hdl_mmol_l"] = hdl.astype(float)
        clin_df["triglycerides_mmol_l"] = tg.astype(float)
        clin_df["plt_10e9_l"] = plt_cnt.astype(float)

        eff_cfg = PairedLipidConfig() if cfg is None else cfg
        dataset = build_paired_cohort_dataset(
            clin_df,
            map_df,
            raw_1300,
            raw_2850,
            wn_1300,
            wn_2850,
            cfg=eff_cfg,
            subset_mode=subset_mode,
        )
        id_to_orig = {pid: idx for idx, pid in enumerate(p_ids.tolist())}
        kept_orig = np.asarray([id_to_orig[str(pid)] for pid in dataset.participant_ids], dtype=int)
        z_bio_out = z_bio[kept_orig]

    truth = {
        "bio_1300": np.column_stack([bio_1300_1, bio_1300_2]),
        "bio_2850": np.column_stack([bio_2850_1, bio_2850_2]),
        "z_bio": z_bio_out,
        "signal_strength": signal_strength,
        "nonlinear_covariate_proxy": nonlinear_covariate_proxy,
        "nonlinear_spectral_signal": nonlinear_spectral_signal,
    }
    return dataset, truth
