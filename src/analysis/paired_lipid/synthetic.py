"""Lightweight synthetic paired Raman cohort generator with known latent factors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import linalg

from src.analysis.paired_lipid.cohort import (
    SUBSET_MODES,
    CohortDataset,
    PairedLipidConfig,
    SubsetMode,
    _cached_band_slices,
    build_paired_cohort_dataset,
)

if TYPE_CHECKING:
    from src.analysis.paired_lipid.nested_cv import PreparedExternalTest

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


@lru_cache(maxsize=32)
def _cached_spectral_grids_and_bases(
    compact_grid: bool,
    include_shoulder_in_bio: bool,
    n_nuisance_modes: int,
    include_map_spatial_basis: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Memoize wavenumber grids and fixed biological/technical/nuisance spectral basis vectors."""
    step = 10.0 if compact_grid else 1.0
    wn_1300 = np.arange(734.0, 1788.0 + 0.5 * step, step, dtype=float)
    wn_2850 = np.arange(2451.0, 3197.0 + 0.5 * step, step, dtype=float)

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

    tech_1300 = _gaussian_peak(wn_1300, 1156.0, 20.0) - 0.6 * _gaussian_peak(wn_1300, 980.0, 15.0)
    tech_2850 = _gaussian_peak(wn_2850, 3015.0, 22.0)

    nuis_b1: np.ndarray | None = None
    nuis_b2: np.ndarray | None = None
    if n_nuisance_modes > 0:
        nuis_b1, nuis_b2 = _build_nuisance_basis(wn_1300, wn_2850, n_modes=n_nuisance_modes)

    nuis_b1_m: np.ndarray | None = None
    nuis_b2_m: np.ndarray | None = None
    if include_map_spatial_basis:
        if nuis_b1 is not None and nuis_b2 is not None and nuis_b1.shape[0] >= 4:
            nuis_b1_m = np.ascontiguousarray(nuis_b1[:4])
            nuis_b2_m = np.ascontiguousarray(nuis_b2[:4])
        else:
            nuis_b1_m, nuis_b2_m = _build_nuisance_basis(wn_1300, wn_2850, n_modes=4)

    for arr in (
        wn_1300,
        wn_2850,
        bio_1300_1,
        bio_2850_1,
        bio_1300_2,
        bio_2850_2,
        tech_1300,
        tech_2850,
        nuis_b1,
        nuis_b2,
        nuis_b1_m,
        nuis_b2_m,
    ):
        if arr is not None:
            arr.setflags(write=False)

    return (
        wn_1300,
        wn_2850,
        bio_1300_1,
        bio_2850_1,
        bio_1300_2,
        bio_2850_2,
        tech_1300,
        tech_2850,
        nuis_b1,
        nuis_b2,
        nuis_b1_m,
        nuis_b2_m,
    )


def _median_maps(
    maps: np.ndarray,
    out: np.ndarray,
    drop: np.ndarray | None,
    *,
    destructive: bool = False,
    scratch: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    """Compute per-sample map median along axis 1, dropping the last map where requested."""
    if maps.shape[1] == 3:
        drop_idx: np.ndarray | None = None
        drop_vals: np.ndarray | None = None
        if drop is not None and np.any(drop):
            drop_idx = np.flatnonzero(drop)
            drop_vals = 0.5 * (maps[drop_idx, 0, :] + maps[drop_idx, 1, :])
        m0 = maps[:, 0, :]
        m1 = maps[:, 1, :]
        m2 = maps[:, 2, :]
        if destructive:
            np.minimum(m0, m1, out=out)
            np.maximum(m0, m1, out=m0)
            np.minimum(m0, m2, out=m0)
            np.maximum(out, m0, out=out)
        else:
            c_len, _, p = maps.shape
            if (
                scratch is not None
                and scratch[0].shape[0] >= c_len
                and scratch[0].shape[1] == p
                and scratch[1].shape[0] >= c_len
                and scratch[1].shape[1] == p
            ):
                t1 = scratch[0][:c_len]
                t2 = scratch[1][:c_len]
            else:
                t1 = np.empty((c_len, p), dtype=out.dtype)
                t2 = np.empty((c_len, p), dtype=out.dtype)
            np.minimum(m0, m1, out=t1)
            np.maximum(m0, m1, out=t2)
            np.minimum(t2, m2, out=t2)
            np.maximum(t1, t2, out=out)
        if drop_idx is not None and drop_vals is not None:
            out[drop_idx] = drop_vals
        return
    if drop is not None and np.any(drop):
        kept = ~drop
        if np.any(kept):
            out[kept] = np.median(maps[kept], axis=1)
        out[drop] = np.median(maps[drop, :-1, :], axis=1)
    else:
        np.median(maps, axis=1, out=out)


@dataclass(slots=True)
class _SyntheticCohortState:
    rng: np.random.Generator
    wn_1300: np.ndarray
    wn_2850: np.ndarray
    bio_1300_1: np.ndarray
    bio_2850_1: np.ndarray
    bio_1300_2: np.ndarray
    bio_2850_2: np.ndarray
    nuis_b1_m: np.ndarray | None
    nuis_b2_m: np.ndarray | None
    p_ids: np.ndarray
    sample_counts: np.ndarray
    offsets: np.ndarray
    n_samples: int
    sample_p_idx: np.ndarray
    paired_p_indices: np.ndarray
    pair_a_indices: np.ndarray
    pair_b_indices: np.ndarray
    n_paired: int
    n_single_actual: int
    sample_ids: np.ndarray
    sample_lbls: np.ndarray
    z_bio: np.ndarray
    outcomes: dict[str, np.ndarray]
    cov_df: pd.DataFrame
    sample_factors: np.ndarray
    basis_1300: np.ndarray
    basis_2850: np.ndarray
    dropout_mask: np.ndarray | None
    exclusions: list[dict[str, Any]]
    n_maps_per_sample: np.ndarray | None
    eff_cfg: PairedLipidConfig


def _prepare_synthetic_cohort_state(
    *,
    n_participants: int,
    n_single_sample: int,
    maps_per_sample: int,
    seed: int,
    compact_grid: bool,
    signal_strength: float,
    ab_shift_scale: float,
    nonlinear_covariate_proxy: bool,
    formula_linked_ldl: bool,
    shared_prep_scale: float,
    label_drift_scale: float,
    tg_ldl_loading: float,
    map_dropout_fraction: float,
    subset_mode: SubsetMode,
    cfg: PairedLipidConfig | None,
    n_nuisance_modes: int,
    nuisance_part_scale: float,
    nuisance_sample_scale: float,
    nuisance_decay: float,
    bio_loading_scale: float,
    bio_sample_noise: float,
    include_shoulder_in_bio: bool,
    correlated_tech_mode_rho: float,
    unequal_b_error_scale: float,
    nonlinear_spectral_signal: bool,
    map_spatial_corr: float,
    map_spatial_scale: float,
    covariate_missing_rate: float,
    qc_sample_dropout_fraction: float,
) -> _SyntheticCohortState:
    """Validate synthetic DGP arguments and materialize participant/sample state with fixed RNG order."""
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
    has_nuisance = n_nuisance_modes > 0 and (nuisance_part_scale > 0.0 or nuisance_sample_scale > 0.0)
    (
        wn_1300,
        wn_2850,
        bio_1300_1,
        bio_2850_1,
        bio_1300_2,
        bio_2850_2,
        tech_1300,
        tech_2850,
        nuis_b1_cached,
        nuis_b2_cached,
        nuis_b1_m,
        nuis_b2_m,
    ) = _cached_spectral_grids_and_bases(
        bool(compact_grid),
        bool(include_shoulder_in_bio),
        int(n_nuisance_modes) if has_nuisance else 0,
        bool(map_spatial_scale > 0.0),
    )

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
    offsets = np.cumsum(sample_counts) - sample_counts
    is_paired = sample_counts == 2
    paired_p_indices = np.flatnonzero(is_paired)
    pair_a_indices = offsets[paired_p_indices]
    pair_b_indices = pair_a_indices + 1
    single_s_indices = offsets[np.flatnonzero(~is_paired)]
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

    if has_nuisance and nuis_b1_cached is not None and nuis_b2_cached is not None:
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
        b1_list.append(nuis_b1_cached)
        b2_list.append(nuis_b2_cached)

    sample_factors = np.ascontiguousarray(np.column_stack(factors_list), dtype=np.float64)
    basis_1300 = np.ascontiguousarray(np.vstack(b1_list), dtype=np.float64)
    basis_2850 = np.ascontiguousarray(np.vstack(b2_list), dtype=np.float64)

    cov_df = pd.DataFrame(
        {
            "age_years": age_cov,
            "sex": pd.Series(sex_arr, dtype=object),
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

    n_maps_per_sample: np.ndarray | None = None
    if subset_mode != "all_eligible":
        n_maps_per_sample = np.full(n_samples, maps_per_sample, dtype=int)
        if dropout_mask is not None:
            n_maps_per_sample[dropout_mask] = maps_per_sample - 1
        if qc_dropped_paired_p.size > 0:
            # Drop sample B to 1 map (< min_maps=2) so raw-map cohort assembly excludes it cleanly.
            drop_p_set = {int(p) for p in qc_dropped_paired_p}
            for p_i, s_b in zip(paired_p_indices.tolist(), pair_b_indices.tolist()):
                if p_i in drop_p_set:
                    n_maps_per_sample[s_b] = 1
        if subset_mode == "three_map_complete" and map_dropout_fraction == 0.0:
            # Deterministic 20% 2-map dropout on raw maps so min_maps=3 exercises exclusions cleanly.
            map_drop_rng = np.random.default_rng(seed + 7)
            two_map_drop = map_drop_rng.random(n_samples) < 0.2
            n_maps_per_sample[two_map_drop] = np.minimum(n_maps_per_sample[two_map_drop], min(2, maps_per_sample))

    outcomes = {
        "ldl_mmol_l": ldl.astype(float),
        "hdl_mmol_l": hdl.astype(float),
        "triglycerides_mmol_l": tg.astype(float),
        "plt_10e9_l": plt_cnt.astype(float),
    }
    eff_cfg = PairedLipidConfig() if cfg is None else cfg
    return _SyntheticCohortState(
        rng=rng,
        wn_1300=wn_1300,
        wn_2850=wn_2850,
        bio_1300_1=bio_1300_1,
        bio_2850_1=bio_2850_1,
        bio_1300_2=bio_1300_2,
        bio_2850_2=bio_2850_2,
        nuis_b1_m=nuis_b1_m,
        nuis_b2_m=nuis_b2_m,
        p_ids=p_ids,
        sample_counts=sample_counts,
        offsets=offsets,
        n_samples=n_samples,
        sample_p_idx=sample_p_idx,
        paired_p_indices=paired_p_indices,
        pair_a_indices=pair_a_indices,
        pair_b_indices=pair_b_indices,
        n_paired=n_paired,
        n_single_actual=n_single_actual,
        sample_ids=sample_ids,
        sample_lbls=sample_lbls,
        z_bio=z_bio,
        outcomes=outcomes,
        cov_df=cov_df,
        sample_factors=sample_factors,
        basis_1300=basis_1300,
        basis_2850=basis_2850,
        dropout_mask=dropout_mask,
        exclusions=exclusions,
        n_maps_per_sample=n_maps_per_sample,
        eff_cfg=eff_cfg,
    )


def _alloc_chunk_buffers(
    max_c: int,
    maps_per_sample: int,
    p1: int,
    p2: int,
    *,
    has_spatial: bool,
    with_median_scratch: bool = False,
) -> dict[str, np.ndarray | None]:
    """Preallocate reusable chunk-sized buffers for map generation."""
    return {
        "b1_buf": np.empty((max_c, p1), dtype=np.float64),
        "b2_buf": np.empty((max_c, p2), dtype=np.float64),
        "n1_buf": np.empty((max_c, maps_per_sample, p1), dtype=np.float64),
        "n2_buf": np.empty((max_c, maps_per_sample, p2), dtype=np.float64),
        "sp1_buf": np.empty((max_c, maps_per_sample, p1), dtype=np.float64) if has_spatial else None,
        "sp2_buf": np.empty((max_c, maps_per_sample, p2), dtype=np.float64) if has_spatial else None,
        "z_shared_buf": np.empty((max_c, 1, 4), dtype=np.float64) if has_spatial else None,
        "z_indep_buf": np.empty((max_c, maps_per_sample, 4), dtype=np.float64) if has_spatial else None,
        "z_map_buf": np.empty((max_c, maps_per_sample, 4), dtype=np.float64) if has_spatial else None,
        "t1_buf": np.empty((max_c, max(p1, p2)), dtype=np.float64) if with_median_scratch else None,
        "t2_buf": np.empty((max_c, max(p1, p2)), dtype=np.float64) if with_median_scratch else None,
    }


def _fill_map_noise_chunk(
    state: _SyntheticCohortState,
    bufs: dict[str, np.ndarray | None],
    s0: int,
    s1: int,
    *,
    maps_per_sample: int,
    map_spatial_corr: float,
    map_spatial_scale: float,
    add_base_before_spatial: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate one chunk of base spectra and map noise using exact RNG call order."""
    c_len = s1 - s0
    b1_c = bufs["b1_buf"][:c_len]  # type: ignore[index]
    b2_c = bufs["b2_buf"][:c_len]  # type: ignore[index]
    n1_c = bufs["n1_buf"][:c_len]  # type: ignore[index]
    n2_c = bufs["n2_buf"][:c_len]  # type: ignore[index]

    np.matmul(state.sample_factors[s0:s1], state.basis_1300, out=b1_c)
    np.matmul(state.sample_factors[s0:s1], state.basis_2850, out=b2_c)
    state.rng.standard_normal(size=n1_c.shape, dtype=np.float64, out=n1_c)
    n1_c *= 0.04
    state.rng.standard_normal(size=n2_c.shape, dtype=np.float64, out=n2_c)
    n2_c *= 0.04
    if add_base_before_spatial:
        np.add(b1_c[:, None, :], n1_c, out=n1_c)
        np.add(b2_c[:, None, :], n2_c, out=n2_c)
    if map_spatial_scale > 0.0 and state.nuis_b1_m is not None and state.nuis_b2_m is not None:
        r_mc = float(np.clip(map_spatial_corr, 0.0, 0.99))
        z_shared_c = bufs["z_shared_buf"][:c_len]  # type: ignore[index]
        z_indep_c = bufs["z_indep_buf"][:c_len]  # type: ignore[index]
        z_map_c = bufs["z_map_buf"][:c_len]  # type: ignore[index]
        sp1_c = bufs["sp1_buf"][:c_len]  # type: ignore[index]
        sp2_c = bufs["sp2_buf"][:c_len]  # type: ignore[index]
        state.rng.standard_normal(size=(c_len, 1, 4), dtype=np.float64, out=z_shared_c)
        z_shared_c *= map_spatial_scale
        state.rng.standard_normal(size=(c_len, maps_per_sample, 4), dtype=np.float64, out=z_indep_c)
        z_indep_c *= map_spatial_scale
        np.multiply(z_shared_c, math.sqrt(r_mc), out=z_map_c)
        z_indep_c *= math.sqrt(1.0 - r_mc)
        z_map_c += z_indep_c
        np.matmul(z_map_c, state.nuis_b1_m, out=sp1_c)
        n1_c += sp1_c
        np.matmul(z_map_c, state.nuis_b2_m, out=sp2_c)
        n2_c += sp2_c
    return b1_c, b2_c, n1_c, n2_c


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
    state = _prepare_synthetic_cohort_state(
        n_participants=n_participants,
        n_single_sample=n_single_sample,
        maps_per_sample=maps_per_sample,
        seed=seed,
        compact_grid=compact_grid,
        signal_strength=signal_strength,
        ab_shift_scale=ab_shift_scale,
        nonlinear_covariate_proxy=nonlinear_covariate_proxy,
        formula_linked_ldl=formula_linked_ldl,
        shared_prep_scale=shared_prep_scale,
        label_drift_scale=label_drift_scale,
        tg_ldl_loading=tg_ldl_loading,
        map_dropout_fraction=map_dropout_fraction,
        subset_mode=subset_mode,
        cfg=cfg,
        n_nuisance_modes=n_nuisance_modes,
        nuisance_part_scale=nuisance_part_scale,
        nuisance_sample_scale=nuisance_sample_scale,
        nuisance_decay=nuisance_decay,
        bio_loading_scale=bio_loading_scale,
        bio_sample_noise=bio_sample_noise,
        include_shoulder_in_bio=include_shoulder_in_bio,
        correlated_tech_mode_rho=correlated_tech_mode_rho,
        unequal_b_error_scale=unequal_b_error_scale,
        nonlinear_spectral_signal=nonlinear_spectral_signal,
        map_spatial_corr=map_spatial_corr,
        map_spatial_scale=map_spatial_scale,
        covariate_missing_rate=covariate_missing_rate,
        qc_sample_dropout_fraction=qc_sample_dropout_fraction,
    )

    p1 = int(state.wn_1300.size)
    p2 = int(state.wn_2850.size)
    max_c = min(_MAP_CHUNK_SAMPLES, state.n_samples)
    has_spatial = map_spatial_scale > 0.0 and state.nuis_b1_m is not None and state.nuis_b2_m is not None
    bufs = _alloc_chunk_buffers(
        max_c,
        maps_per_sample,
        p1,
        p2,
        has_spatial=has_spatial,
        with_median_scratch=False,
    )

    if subset_mode == "all_eligible":
        sample_1300 = np.empty((state.n_samples, p1), dtype=np.float64)
        sample_2850 = np.empty((state.n_samples, p2), dtype=np.float64)
        for s0 in range(0, state.n_samples, _MAP_CHUNK_SAMPLES):
            s1 = min(s0 + _MAP_CHUNK_SAMPLES, state.n_samples)
            b1_c, b2_c, n1_c, n2_c = _fill_map_noise_chunk(
                state,
                bufs,
                s0,
                s1,
                maps_per_sample=maps_per_sample,
                map_spatial_corr=map_spatial_corr,
                map_spatial_scale=map_spatial_scale,
                add_base_before_spatial=False,
            )
            # Median is translation-invariant: compute it on zero-mean map noise, then add base in 2D.
            drop_local = state.dropout_mask[s0:s1] if state.dropout_mask is not None else None
            _median_maps(n1_c, sample_1300[s0:s1], drop_local, destructive=True)
            sample_1300[s0:s1] += b1_c
            _median_maps(n2_c, sample_2850[s0:s1], drop_local, destructive=True)
            sample_2850[s0:s1] += b2_c

        dataset = CohortDataset(
            participant_ids=state.p_ids,
            outcomes=state.outcomes,
            covariate_frame=state.cov_df,
            wavenumbers_1300=state.wn_1300,
            wavenumbers_2850=state.wn_2850,
            sample_1300=sample_1300,
            sample_2850=sample_2850,
            sample_participant_idx=state.sample_p_idx,
            sample_ids=state.sample_ids,
            sample_parsed_labels=state.sample_lbls,
            participant_sample_counts=state.sample_counts,
            paired_participant_indices=state.paired_p_indices,
            pair_sample_a_indices=state.pair_a_indices,
            pair_sample_b_indices=state.pair_b_indices,
            manifest_summary={
                "subset_mode": "all_eligible",
                "min_maps_per_block": 2,
                "n_eligible_participants": n_participants,
                "n_paired_participants": state.n_paired,
                "n_single_sample_participants": state.n_single_actual,
                "n_eligible_samples": state.n_samples,
                "n_exclusions": len(state.exclusions),
            },
            exclusions=state.exclusions,
        )
        z_bio_out = state.z_bio
    else:
        assert state.n_maps_per_sample is not None
        raw_chunks_1300: list[np.ndarray] = []
        raw_chunks_2850: list[np.ndarray] = []
        map_idx_axis = np.arange(maps_per_sample, dtype=int)[None, :]
        for s0 in range(0, state.n_samples, _MAP_CHUNK_SAMPLES):
            s1 = min(s0 + _MAP_CHUNK_SAMPLES, state.n_samples)
            _, _, maps_1300, maps_2850 = _fill_map_noise_chunk(
                state,
                bufs,
                s0,
                s1,
                maps_per_sample=maps_per_sample,
                map_spatial_corr=map_spatial_corr,
                map_spatial_scale=map_spatial_scale,
                add_base_before_spatial=True,
            )
            keep_map = map_idx_axis < state.n_maps_per_sample[s0:s1, None]
            raw_chunks_1300.append(maps_1300[keep_map])
            raw_chunks_2850.append(maps_2850[keep_map])

        raw_1300 = np.vstack(raw_chunks_1300)
        raw_2850 = np.vstack(raw_chunks_2850)
        sample_idx_per_map = np.repeat(np.arange(state.n_samples, dtype=int), state.n_maps_per_sample)
        total_maps = int(sample_idx_per_map.size)
        map_pids = state.p_ids[state.sample_p_idx[sample_idx_per_map]]
        map_sids = state.sample_ids[sample_idx_per_map]
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

        clin_df = state.cov_df.copy()
        clin_df["participant_id"] = state.p_ids
        clin_df["cohort"] = "stroke"
        clin_df["identity_verified"] = True
        for ep_name, ep_vals in state.outcomes.items():
            clin_df[ep_name] = ep_vals

        dataset = build_paired_cohort_dataset(
            clin_df,
            map_df,
            raw_1300,
            raw_2850,
            state.wn_1300,
            state.wn_2850,
            cfg=state.eff_cfg,
            subset_mode=subset_mode,
        )
        # Preserve fixed-range label_scale = LABEL_RANGE / (2 * n_participants) across all subset
        # modes so synthetic label covariates never extrapolate between training and test sizes.
        label_scale = LABEL_RANGE / float(2 * n_participants)
        scaled_lbls = np.ascontiguousarray(dataset.sample_parsed_labels * label_scale, dtype=np.float64)
        scaled_lbls.setflags(write=False)
        first_idx = np.cumsum(dataset.participant_sample_counts) - dataset.participant_sample_counts
        scaled_trend = scaled_lbls[first_idx].copy()
        if subset_mode != "single_sample" and dataset.paired_participant_indices.size > 0:
            scaled_trend[dataset.paired_participant_indices] = 0.5 * (
                scaled_lbls[dataset.pair_sample_a_indices] + scaled_lbls[dataset.pair_sample_b_indices]
            )
        dataset.sample_parsed_labels = scaled_lbls
        dataset.covariate_frame["sample_label_trend"] = scaled_trend
        id_to_orig = {pid: idx for idx, pid in enumerate(state.p_ids.tolist())}
        kept_orig = np.asarray([id_to_orig[str(pid)] for pid in dataset.participant_ids], dtype=int)
        z_bio_out = state.z_bio[kept_orig]

    truth = {
        "bio_1300": np.column_stack([state.bio_1300_1, state.bio_1300_2]),
        "bio_2850": np.column_stack([state.bio_2850_1, state.bio_2850_2]),
        "z_bio": z_bio_out,
        "signal_strength": signal_strength,
        "nonlinear_covariate_proxy": nonlinear_covariate_proxy,
        "nonlinear_spectral_signal": nonlinear_spectral_signal,
    }
    return dataset, truth


def generate_synthetic_external_test_arrays(
    *,
    endpoint: str = "ldl_mmol_l",
    single_sample_downstream: bool | None = None,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, bool]:
    """Stream synthetic external test data directly to participant-level arrays without sample matrices."""
    state = _prepare_synthetic_cohort_state(
        n_participants=n_participants,
        n_single_sample=n_single_sample,
        maps_per_sample=maps_per_sample,
        seed=seed,
        compact_grid=compact_grid,
        signal_strength=signal_strength,
        ab_shift_scale=ab_shift_scale,
        nonlinear_covariate_proxy=nonlinear_covariate_proxy,
        formula_linked_ldl=formula_linked_ldl,
        shared_prep_scale=shared_prep_scale,
        label_drift_scale=label_drift_scale,
        tg_ldl_loading=tg_ldl_loading,
        map_dropout_fraction=map_dropout_fraction,
        subset_mode=subset_mode,
        cfg=cfg,
        n_nuisance_modes=n_nuisance_modes,
        nuisance_part_scale=nuisance_part_scale,
        nuisance_sample_scale=nuisance_sample_scale,
        nuisance_decay=nuisance_decay,
        bio_loading_scale=bio_loading_scale,
        bio_sample_noise=bio_sample_noise,
        include_shoulder_in_bio=include_shoulder_in_bio,
        correlated_tech_mode_rho=correlated_tech_mode_rho,
        unequal_b_error_scale=unequal_b_error_scale,
        nonlinear_spectral_signal=nonlinear_spectral_signal,
        map_spatial_corr=map_spatial_corr,
        map_spatial_scale=map_spatial_scale,
        covariate_missing_rate=covariate_missing_rate,
        qc_sample_dropout_fraction=qc_sample_dropout_fraction,
    )

    if subset_mode == "all_eligible":
        valid_endpoints = set(state.outcomes.keys())
    else:
        valid_endpoints = {
            ep
            for ep in (state.eff_cfg.primary_endpoint, *state.eff_cfg.secondary_endpoints)
            if ep in state.outcomes
        }
    if endpoint not in valid_endpoints:
        raise ValueError(f"Endpoint '{endpoint}' not present in external test dataset outcomes.")

    eff_single_sample = (
        bool(single_sample_downstream)
        if single_sample_downstream is not None
        else (subset_mode == "single_sample")
    )

    if subset_mode == "all_eligible":
        n_kept = n_participants
        cov_out = state.cov_df
        y_ext = np.ascontiguousarray(state.outcomes[endpoint], dtype=np.float64)
        if eff_single_sample:
            single_src = state.offsets.astype(np.int64, copy=False)
            single_dst = np.arange(n_kept, dtype=np.int64)
            pair_a_src = np.empty(0, dtype=np.int64)
            pair_b_src = np.empty(0, dtype=np.int64)
            pair_dst = np.empty(0, dtype=np.int64)
        else:
            single_p = np.flatnonzero(state.sample_counts == 1)
            single_src = state.offsets[single_p].astype(np.int64, copy=False)
            single_dst = single_p.astype(np.int64, copy=False)
            pair_a_src = state.pair_a_indices.astype(np.int64, copy=False)
            pair_b_src = state.pair_b_indices.astype(np.int64, copy=False)
            pair_dst = state.paired_p_indices.astype(np.int64, copy=False)
    else:
        assert state.n_maps_per_sample is not None
        min_maps = 3 if subset_mode == "three_map_complete" else state.eff_cfg.min_maps_per_block
        surv_sample = state.n_maps_per_sample >= min_maps
        has_s0 = surv_sample[state.offsets]
        has_s1 = (state.sample_counts == 2) & surv_sample[np.minimum(state.offsets + 1, state.n_samples - 1)]
        n_surv = has_s0.astype(np.int64) + has_s1.astype(np.int64)
        kept_mask = (n_surv >= 2) if subset_mode == "paired_only" else (n_surv >= 1)
        if not np.any(kept_mask):
            raise ValueError("No eligible stroke participants remained after linkage and spectral checks.")

        kept_unsorted = np.flatnonzero(kept_mask)
        # Match cohort._filter_clinical_records lexicographical string sort on participant_id.
        lex_order = np.argsort(state.p_ids[kept_unsorted], kind="mergesort")
        kept_orig = kept_unsorted[lex_order]
        n_kept = int(kept_orig.size)

        k_s0 = has_s0[kept_orig]
        k_s1 = has_s1[kept_orig]
        k_off = state.offsets[kept_orig]
        first_s = np.where(k_s0, k_off, np.minimum(k_off + 1, state.n_samples - 1))
        both_surv = k_s0 & k_s1
        parsed_lbls = state.sample_lbls
        mean_labels = parsed_lbls[first_s].copy()
        if subset_mode != "single_sample" and np.any(both_surv):
            both_idx = np.flatnonzero(both_surv)
            k_both = k_off[both_idx]
            mean_labels[both_idx] = 0.5 * (parsed_lbls[k_both] + parsed_lbls[k_both + 1])
        cov_out = state.cov_df.iloc[kept_orig].reset_index(drop=True)
        cov_out["sample_label_trend"] = mean_labels
        y_ext = np.ascontiguousarray(state.outcomes[endpoint][kept_orig], dtype=np.float64)

        if eff_single_sample:
            single_src = first_s.astype(np.int64, copy=False)
            single_dst = np.arange(n_kept, dtype=np.int64)
            pair_a_src = np.empty(0, dtype=np.int64)
            pair_b_src = np.empty(0, dtype=np.int64)
            pair_dst = np.empty(0, dtype=np.int64)
        else:
            is_single_out = ~both_surv
            single_dst = np.flatnonzero(is_single_out).astype(np.int64, copy=False)
            single_src = first_s[single_dst].astype(np.int64, copy=False)
            pair_dst = np.flatnonzero(both_surv).astype(np.int64, copy=False)
            pair_a_src = k_off[pair_dst].astype(np.int64, copy=False)
            pair_b_src = (pair_a_src + 1).astype(np.int64, copy=False)

    sample_role = np.zeros(state.n_samples, dtype=np.int8)
    sample_dst = np.full(state.n_samples, -1, dtype=np.int64)
    if single_src.size > 0:
        sample_role[single_src] = 1
        sample_dst[single_src] = single_dst
    if pair_a_src.size > 0:
        sample_role[pair_a_src] = 2
        sample_dst[pair_a_src] = pair_dst
        sample_role[pair_b_src] = 3
        sample_dst[pair_b_src] = pair_dst

    w1 = np.ascontiguousarray(state.wn_1300, dtype=np.float64).tobytes()
    w2 = np.ascontiguousarray(state.wn_2850, dtype=np.float64).tobytes()
    band_slices = _cached_band_slices(w1, w2)
    for name, block, selector in band_slices:
        if isinstance(selector, np.ndarray) and selector.size == 0:
            raise ValueError(f"Band '{name}' has no channels in block {block} axis.")

    p1 = int(state.wn_1300.size)
    p2 = int(state.wn_2850.size)
    n_bands = len(band_slices)
    part_1300 = np.empty((n_kept, p1), dtype=np.float64)
    part_2850 = np.empty((n_kept, p2), dtype=np.float64)
    part_band_mat = np.empty((n_kept, n_bands), dtype=np.float64)

    max_c = min(_MAP_CHUNK_SAMPLES, state.n_samples)
    has_spatial = map_spatial_scale > 0.0 and state.nuis_b1_m is not None and state.nuis_b2_m is not None
    bufs = _alloc_chunk_buffers(
        max_c,
        maps_per_sample,
        p1,
        p2,
        has_spatial=has_spatial,
        with_median_scratch=False,
    )
    s1_buf = np.zeros((max_c, p1), dtype=np.float64)
    s2_buf = np.zeros((max_c, p2), dtype=np.float64)
    sb_buf = np.empty((max_c, n_bands), dtype=np.float64)
    max_pairs_chunk = max(1, max_c // 2)
    pair_tmp1 = np.empty((max_pairs_chunk, p1), dtype=np.float64)
    pair_tmp2 = np.empty((max_pairs_chunk, p2), dtype=np.float64)
    pair_tmpb = np.empty((max_pairs_chunk, n_bands), dtype=np.float64)
    carry_1 = np.empty(p1, dtype=np.float64)
    carry_2 = np.empty(p2, dtype=np.float64)
    carry_b = np.empty(n_bands, dtype=np.float64)
    carry_dst = -1

    add_base_before = subset_mode != "all_eligible"
    for s0 in range(0, state.n_samples, _MAP_CHUNK_SAMPLES):
        s1 = min(s0 + _MAP_CHUNK_SAMPLES, state.n_samples)
        c_len = s1 - s0
        b1_c, b2_c, n1_c, n2_c = _fill_map_noise_chunk(
            state,
            bufs,
            s0,
            s1,
            maps_per_sample=maps_per_sample,
            map_spatial_corr=map_spatial_corr,
            map_spatial_scale=map_spatial_scale,
            add_base_before_spatial=add_base_before,
        )
        chunk_1300 = s1_buf[:c_len]
        chunk_2850 = s2_buf[:c_len]
        chunk_band = sb_buf[:c_len]
        role_c = sample_role[s0:s1]
        dst_c = sample_dst[s0:s1]

        if subset_mode == "all_eligible":
            drop_local = state.dropout_mask[s0:s1] if state.dropout_mask is not None else None
            _median_maps(n1_c, chunk_1300, drop_local, destructive=True)
            chunk_1300 += b1_c
            _median_maps(n2_c, chunk_2850, drop_local, destructive=True)
            chunk_2850 += b2_c
        else:
            assert state.n_maps_per_sample is not None
            n_maps_c = state.n_maps_per_sample[s0:s1]
            if maps_per_sample == 3:
                k1_idx = np.flatnonzero((role_c != 0) & (n_maps_c == 1))
                k1_vals_1300 = n1_c[k1_idx, 0, :].copy() if k1_idx.size > 0 else None
                k1_vals_2850 = n2_c[k1_idx, 0, :].copy() if k1_idx.size > 0 else None
                drop_c = n_maps_c == 2
                _median_maps(n1_c, chunk_1300, drop_c, destructive=True)
                _median_maps(n2_c, chunk_2850, drop_c, destructive=True)
                if k1_idx.size > 0 and k1_vals_1300 is not None and k1_vals_2850 is not None:
                    chunk_1300[k1_idx] = k1_vals_1300
                    chunk_2850[k1_idx] = k1_vals_2850
            else:
                for k_val in np.unique(n_maps_c[role_c != 0]):
                    rows_k = np.flatnonzero((role_c != 0) & (n_maps_c == int(k_val)))
                    chunk_1300[rows_k] = np.median(n1_c[rows_k, : int(k_val), :], axis=1)
                    chunk_2850[rows_k] = np.median(n2_c[rows_k, : int(k_val), :], axis=1)

        if not np.any(role_c):
            continue

        for j, (_, block, selector) in enumerate(band_slices):
            source = chunk_1300 if block == "1300" else chunk_2850
            np.mean(source[:, selector], axis=1, out=chunk_band[:, j])

        loc_s = np.flatnonzero(role_c == 1)
        if loc_s.size > 0:
            d_s = dst_c[loc_s]
            part_1300[d_s] = chunk_1300[loc_s]
            part_2850[d_s] = chunk_2850[loc_s]
            part_band_mat[d_s] = chunk_band[loc_s]

        if carry_dst >= 0:
            np.add(carry_1, chunk_1300[0], out=part_1300[carry_dst])
            part_1300[carry_dst] *= 0.5
            np.add(carry_2, chunk_2850[0], out=part_2850[carry_dst])
            part_2850[carry_dst] *= 0.5
            np.add(carry_b, chunk_band[0], out=part_band_mat[carry_dst])
            part_band_mat[carry_dst] *= 0.5
            carry_dst = -1

        loc_a = np.flatnonzero(role_c == 2)
        if loc_a.size > 0:
            if int(loc_a[-1]) == c_len - 1:
                carry_dst = int(dst_c[-1])
                carry_1[:] = chunk_1300[-1]
                carry_2[:] = chunk_2850[-1]
                carry_b[:] = chunk_band[-1]
                loc_a = loc_a[:-1]
            if loc_a.size > 0:
                n_pairs_c = int(loc_a.size)
                d_p = dst_c[loc_a]
                loc_b = loc_a + 1
                tmp1 = pair_tmp1[:n_pairs_c]
                tmp2 = pair_tmp2[:n_pairs_c]
                tmpb = pair_tmpb[:n_pairs_c]
                np.add(chunk_1300[loc_a], chunk_1300[loc_b], out=tmp1)
                tmp1 *= 0.5
                part_1300[d_p] = tmp1
                np.add(chunk_2850[loc_a], chunk_2850[loc_b], out=tmp2)
                tmp2 *= 0.5
                part_2850[d_p] = tmp2
                np.add(chunk_band[loc_a], chunk_band[loc_b], out=tmpb)
                tmpb *= 0.5
                part_band_mat[d_p] = tmpb

    part_1300.setflags(write=False)
    part_2850.setflags(write=False)
    part_band_mat.setflags(write=False)
    y_ext.setflags(write=False)
    return part_1300, part_2850, part_band_mat, cov_out, y_ext, eff_single_sample


def generate_synthetic_external_test(
    *,
    endpoint: str = "ldl_mmol_l",
    single_sample_downstream: bool | None = None,
    preset: str | None = None,
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
) -> PreparedExternalTest:
    """Build a read-only streaming PreparedExternalTest directly without sample-level spectra."""
    from src.analysis.paired_lipid.nested_cv import PreparedExternalTest

    part_1300, part_2850, part_band_mat, cov_out, y_ext, eff_single_sample = (
        generate_synthetic_external_test_arrays(
            endpoint=endpoint,
            single_sample_downstream=single_sample_downstream,
            n_participants=n_participants,
            n_single_sample=n_single_sample,
            maps_per_sample=maps_per_sample,
            seed=seed,
            compact_grid=compact_grid,
            signal_strength=signal_strength,
            ab_shift_scale=ab_shift_scale,
            nonlinear_covariate_proxy=nonlinear_covariate_proxy,
            formula_linked_ldl=formula_linked_ldl,
            shared_prep_scale=shared_prep_scale,
            label_drift_scale=label_drift_scale,
            tg_ldl_loading=tg_ldl_loading,
            map_dropout_fraction=map_dropout_fraction,
            subset_mode=subset_mode,
            cfg=cfg,
            n_nuisance_modes=n_nuisance_modes,
            nuisance_part_scale=nuisance_part_scale,
            nuisance_sample_scale=nuisance_sample_scale,
            nuisance_decay=nuisance_decay,
            bio_loading_scale=bio_loading_scale,
            bio_sample_noise=bio_sample_noise,
            include_shoulder_in_bio=include_shoulder_in_bio,
            correlated_tech_mode_rho=correlated_tech_mode_rho,
            unequal_b_error_scale=unequal_b_error_scale,
            nonlinear_spectral_signal=nonlinear_spectral_signal,
            map_spatial_corr=map_spatial_corr,
            map_spatial_scale=map_spatial_scale,
            covariate_missing_rate=covariate_missing_rate,
            qc_sample_dropout_fraction=qc_sample_dropout_fraction,
        )
    )
    return PreparedExternalTest.from_arrays(
        endpoint=endpoint,
        spec_1300=part_1300,
        spec_2850=part_2850,
        cov_frame=cov_out,
        y_ext=y_ext,
        part_band_mat=part_band_mat,
        single_sample_downstream=eff_single_sample,
        preset=preset,
    )
