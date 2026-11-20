"""Lightweight synthetic paired Raman cohort generator with known latent factors."""

from __future__ import annotations

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
_MAP_CHUNK_SAMPLES = 256


def _gaussian_peak(axis: np.ndarray, center: float, width: float) -> np.ndarray:
    v = np.exp(-0.5 * ((axis - center) / width) ** 2)
    return v / max(float(linalg.norm(v)), 1e-12)


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
    subset_mode: SubsetMode = "all_eligible",
    cfg: PairedLipidConfig | None = None,
) -> tuple[CohortDataset, dict[str, Any]]:
    """Generate a synthetic paired Raman cohort with known latent biological and technical factors.

    ``tg_ldl_loading`` sets how strongly triglycerides load on the LDL-C biological factor (anchor
    correlation). ``map_dropout_fraction`` is the share of samples that lose one map to QC.
    ``subset_mode`` applies cohort subsetting directly at raw-map level when not ``all_eligible``.
    """
    if subset_mode not in SUBSET_MODES:
        raise ValueError(f"Unknown subset_mode: {subset_mode!r}. Expected one of {SUBSET_MODES}.")
    if n_participants < 2 or not 0 <= n_single_sample <= n_participants or maps_per_sample < 2:
        raise ValueError("Need n_participants >= 2, 0 <= n_single_sample <= n_participants, maps_per_sample >= 2.")
    rng = np.random.default_rng(seed)
    step = 10.0 if compact_grid else 1.0
    wn_1300 = np.arange(734.0, 1788.0 + 0.5 * step, step, dtype=float)
    wn_2850 = np.arange(2451.0, 3197.0 + 0.5 * step, step, dtype=float)

    # Smooth biological basis functions overlapping lipid-associated and protein-associated regions.
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

    # Labels span a fixed range whatever n is, so label covariates never extrapolate between cohorts.
    label_scale = LABEL_RANGE / float(2 * n_participants)

    n_paired = n_participants - n_single_sample
    n_samples = 2 * n_paired + n_single_sample
    paired_p_indices = np.arange(n_paired, dtype=int)
    pair_a_indices = 2 * paired_p_indices
    pair_b_indices = pair_a_indices + 1

    p_ids = np.arange(100, 100 + n_participants, dtype=int).astype(str)
    sex_arr = np.where(sex_female == 1, "female", "male").astype(object)
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

    sample_counts = np.empty(n_participants, dtype=int)
    sample_counts[:n_paired] = 2
    sample_counts[n_paired:] = 1
    sample_p_idx = np.repeat(np.arange(n_participants, dtype=int), sample_counts)
    sample_ids = np.array([f"sample_{k}" for k in range(1, n_samples + 1)], dtype=str)
    sample_lbls = np.arange(1, n_samples + 1, dtype=float) * label_scale

    mean_lbl_arr = np.empty(n_participants, dtype=float)
    mean_lbl_arr[:n_paired] = 0.5 * (sample_lbls[pair_a_indices] + sample_lbls[pair_b_indices])
    mean_lbl_arr[n_paired:] = sample_lbls[2 * n_paired :]

    ab_shifts = np.full(n_samples, -ab_shift_scale, dtype=float)
    ab_shifts[pair_b_indices] = ab_shift_scale
    lbl_drift = label_drift_scale * (sample_lbls / LABEL_RANGE)
    sample_err = rng.normal(0.0, 0.65, size=n_samples) + ab_shifts + lbl_drift
    dropout_mask = (rng.random(n_samples) < map_dropout_fraction) if map_dropout_fraction > 0.0 else None

    z0_plus_prep = (z_bio[:, 0] + shared_prep)[sample_p_idx]
    z1_sample = z_bio[sample_p_idx, 1]
    sample_factors = np.column_stack([z0_plus_prep, z1_sample, sample_err])
    basis_1300 = np.vstack([bio_1300_1, bio_1300_2, tech_1300])
    basis_2850 = np.vstack([bio_2850_1, bio_2850_2, tech_2850])

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
                "n_single_sample_participants": n_single_sample,
                "n_eligible_samples": n_samples,
                "n_exclusions": 0,
            },
            exclusions=[],
        )
        z_bio_out = z_bio
    else:
        n_maps_per_sample = np.full(n_samples, maps_per_sample, dtype=int)
        if dropout_mask is not None:
            n_maps_per_sample[dropout_mask] = maps_per_sample - 1
        if subset_mode == "three_map_complete" and map_dropout_fraction == 0.0:
            # Deterministic 20% 2-map dropout on raw maps so min_maps=3 exercises exclusions cleanly.
            map_drop_rng = np.random.default_rng(seed + 7)
            two_map_drop = map_drop_rng.random(n_samples) < 0.2
            n_maps_per_sample[two_map_drop] = min(2, maps_per_sample)

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
    }
    return dataset, truth
