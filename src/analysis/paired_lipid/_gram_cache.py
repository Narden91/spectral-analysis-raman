"""Dataset spectral Gram memoization and division-level constant precomputation."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
from scipy import linalg

from src.analysis.paired_lipid.cohort import (
    CohortDataset,
    PairedLipidConfig,
    extract_sample_band_matrix,
)
from src.analysis.paired_lipid.models import CovariatePreprocessor

_MAX_DCONST_CACHE_ENTRIES = 64

_HOST_DCONST_KEYS: frozenset[str] = frozenset(
    {
        "tr_p",
        "val_p",
        "s_div",
        "s_rel",
        "p_pair_tr",
        "idx_a_tr",
        "idx_b_tr",
        "pair_rel",
        "base_comb_rank",
        "p_total",
        "kept_cov_cols",
        "dropped_cov_cols_base",
    }
)


def _freeze_readonly(*arrays: np.ndarray) -> None:
    """Mark host arrays read-only so shared spectral caches cannot be mutated."""
    for arr in arrays:
        arr.setflags(write=False)


def _get_spectral_gram_cache(dataset: CohortDataset) -> dict[str, Any]:
    """Memoize preset-independent sample and participant spectral Gram matrices on dataset."""
    cached = dataset._cache.get("spectral_gram_cache")
    if cached is not None:
        return cached

    n_p = dataset.n_participants
    n_s = dataset.n_samples
    X1 = np.asarray(dataset.sample_1300, dtype=np.float64)
    X2 = np.asarray(dataset.sample_2850, dtype=np.float64)
    K1 = X1 @ X1.T
    K2 = X2 @ X2.T

    A_part = np.zeros((n_p, n_s), dtype=np.float64)
    p_for_s = dataset.sample_participant_idx
    A_part[p_for_s, np.arange(n_s, dtype=int)] = 1.0 / dataset.participant_sample_counts[p_for_s].astype(np.float64)

    K_part_1 = A_part @ K1 @ A_part.T
    K_part_2 = A_part @ K2 @ A_part.T

    sample_band_mat, _ = extract_sample_band_matrix(dataset)
    part_band_mat = A_part @ sample_band_mat
    _freeze_readonly(K1, K2, A_part, K_part_1, K_part_2, part_band_mat)

    spec_cache = {
        "K1": K1,
        "K2": K2,
        "A_part": A_part,
        "K_part_1": K_part_1,
        "K_part_2": K_part_2,
        "sample_band_mat": sample_band_mat,
        "part_band_mat": part_band_mat,
        "p_total": int(X1.shape[1] + X2.shape[1]),
    }
    dataset._cache["spectral_gram_cache"] = spec_cache
    return spec_cache


def _get_dataset_gram_cache(dataset: CohortDataset, cfg: PairedLipidConfig) -> dict[str, Any]:
    """Memoize sample/participant Gram matrices and preset-specific raw covariates on dataset."""
    key = f"gram_cache__{cfg.covariate_preset}"
    cached = dataset._cache.get(key)
    if cached is not None:
        return cached

    spec_cache = _get_spectral_gram_cache(dataset)
    raw_cov_fn = getattr(CovariatePreprocessor, "raw_design_matrix", CovariatePreprocessor._get_raw_matrix)
    raw_cov, raw_cols = raw_cov_fn(
        dataset.covariate_frame,
        preset=cfg.covariate_preset,
        raw_cache=dataset._cache,
    )
    raw_cov_arr = np.asarray(raw_cov, dtype=np.float64)

    out = {
        **spec_cache,
        "raw_cov": raw_cov_arr,
        "raw_cov_cols": tuple(str(c) for c in raw_cols),
        "has_missing_cov": bool(not np.all(np.isfinite(raw_cov_arr))),
        "dconst_cpu_cache": OrderedDict(),
    }
    dataset._cache[key] = out
    return out


def _precompute_division_constants(
    xp: Any,
    gcache: dict[str, Any],
    dataset: CohortDataset,
    tr_p: np.ndarray,
    val_p: np.ndarray,
    cfg: PairedLipidConfig,
) -> dict[str, Any]:
    """Precompute division-specific Gram slices and warm-start bases outside the bootstrap loop."""
    cpu_cache = gcache.get("dconst_cpu_cache")
    if cpu_cache is None:
        cpu_cache = OrderedDict()
        gcache["dconst_cpu_cache"] = cpu_cache

    cache_key = (
        np.ascontiguousarray(tr_p, dtype=np.int64).tobytes(),
        np.ascontiguousarray(val_p, dtype=np.int64).tobytes(),
        cfg.pca_dimensions,
        float(cfg.rank_tolerance),
    )
    cpu_out = cpu_cache.get(cache_key)
    if cpu_out is not None:
        if isinstance(cpu_cache, OrderedDict):
            cpu_cache.move_to_end(cache_key)
    else:
        n_tr = tr_p.size
        in_tr_mask = np.isin(dataset.sample_participant_idx, tr_p)
        s_div = np.flatnonzero(in_tr_mask)
        n_s = int(s_div.size)
        p_to_rel = np.full(dataset.n_participants, -1, dtype=int)
        p_to_rel[tr_p] = np.arange(n_tr, dtype=int)
        s_rel = p_to_rel[dataset.sample_participant_idx[s_div]]
        m_i_div = dataset.participant_sample_counts[dataset.sample_participant_idx[s_div]].astype(np.float64)

        raw_cov = np.asarray(gcache["raw_cov"], dtype=np.float64)
        raw_cov_cols = tuple(gcache.get("raw_cov_cols", (f"cov_{i}" for i in range(raw_cov.shape[1]))))
        X_raw_tr = raw_cov[tr_p]
        X_raw_val = raw_cov[val_p]
        ptp_tr = np.ptp(X_raw_tr, axis=0)
        keep_col = ptp_tr > 1e-12
        kept_cov_cols = tuple(c for c, flag in zip(raw_cov_cols, keep_col) if flag)
        dropped_cov_cols_base = tuple(c for c, flag in zip(raw_cov_cols, keep_col) if not flag)
        X_raw_tr = np.ascontiguousarray(X_raw_tr[:, keep_col])
        X_raw_val = np.ascontiguousarray(X_raw_val[:, keep_col])

        A_part_np = gcache["A_part"]
        Kp1_all_div = np.ascontiguousarray(A_part_np @ gcache["K1"][:, s_div], dtype=np.float64)
        K1_div = np.ascontiguousarray(gcache["K1"][np.ix_(s_div, s_div)], dtype=np.float64)
        Kp2_all_div = np.ascontiguousarray(A_part_np @ gcache["K2"][:, s_div], dtype=np.float64)
        K2_div = np.ascontiguousarray(gcache["K2"][np.ix_(s_div, s_div)], dtype=np.float64)

        w_s0 = (1.0 / m_i_div) / np.sum(1.0 / m_i_div)
        sw0 = np.sqrt(w_s0 * (n_s / max(n_s - 1.0, 1.0)))
        sw0_outer = sw0[:, None] * sw0[None, :]
        inv_sw0 = 1.0 / np.maximum(sw0[:, None], 1e-12)
        q_dim_1 = min(8, n_s)

        k1_0 = w_s0 @ K1_div
        m1_0 = float(np.sum(w_s0 * k1_0))
        G1_0 = (K1_div - k1_0[:, None] - k1_0[None, :] + m1_0) * sw0_outer
        G1_0 = 0.5 * (G1_0 + G1_0.T)
        ev1_0, U1_0 = linalg.eigh(
            G1_0, subset_by_index=[n_s - q_dim_1, n_s - 1], driver="evr", check_finite=False
        )

        k2_0 = w_s0 @ K2_div
        m2_0 = float(np.sum(w_s0 * k2_0))
        G2_0 = (K2_div - k2_0[:, None] - k2_0[None, :] + m2_0) * sw0_outer
        G2_0 = 0.5 * (G2_0 + G2_0.T)
        ev2_0, U2_0 = linalg.eigh(
            G2_0, subset_by_index=[n_s - q_dim_1, n_s - 1], driver="evr", check_finite=False
        )

        Q0_1 = np.ascontiguousarray(U1_0 * inv_sw0)
        Q0_2 = np.ascontiguousarray(U2_0 * inv_sw0)

        max_k = max(cfg.pca_dimensions)
        p_total = int(gcache.get("p_total", dataset.sample_1300.shape[1] + dataset.sample_2850.shape[1]))
        Q0_comb: np.ndarray | None = None
        G_comb_0 = G1_0 / max(float(ev1_0[-1]), 1e-12) + G2_0 / max(float(ev2_0[-1]), 1e-12)
        G_comb_0 = 0.5 * (G_comb_0 + G_comb_0.T)
        if n_s > 32 and max_k <= 24:
            ev_comb_0, U_comb_0 = linalg.eigh(
                G_comb_0, subset_by_index=[n_s - 32, n_s - 1], driver="evr", check_finite=False
            )
            # CheFSI (degree 4 x 3 cycles) requires a clear spectral decay ratio between mode max_k
            # and mode 32; flat white-noise Marchenko-Pastur bulks route directly to exact eigh.
            if float(ev_comb_0[-max_k]) / max(float(ev_comb_0[0]), 1e-15) >= 2.0:
                Q0_comb = np.ascontiguousarray(U_comb_0 * inv_sw0)
        else:
            k_check = min(n_s, max(32, max_k))
            ev_comb_0 = linalg.eigh(
                G_comb_0,
                subset_by_index=[n_s - k_check, n_s - 1],
                eigvals_only=True,
                driver="evr",
                check_finite=False,
            )

        s_comb_top = np.sqrt(np.maximum(ev_comb_0, 0.0))
        s_top_val = float(s_comb_top[-1]) if s_comb_top.size else 1.0
        # Account for squared-Gram eigenvalue roundoff floor when checking base numerical rank.
        eps_f = float(np.finfo(float).eps)
        max_dim_0 = float(max(n_s, p_total))
        s0_0 = max(s_top_val, 1.0)
        comb_gram_tol = float(np.sqrt(max_dim_0 * eps_f)) * s0_0
        comb_primal_tol = max(max_dim_0 * eps_f * s0_0, float(cfg.rank_tolerance))
        base_comb_rank = int(np.sum(s_comb_top > max(10.0 * comb_gram_tol, comb_primal_tol)))

        paired_in_tr = np.isin(dataset.paired_participant_indices, tr_p)
        p_pair_tr = dataset.paired_participant_indices[paired_in_tr]
        idx_a_tr = dataset.pair_sample_a_indices[paired_in_tr]
        idx_b_tr = dataset.pair_sample_b_indices[paired_in_tr]
        pair_rel = p_to_rel[p_pair_tr]
        DK1_pair_div = np.ascontiguousarray(
            gcache["K1"][idx_a_tr][:, s_div] - gcache["K1"][idx_b_tr][:, s_div], dtype=np.float64
        )
        DK2_pair_div = np.ascontiguousarray(
            gcache["K2"][idx_a_tr][:, s_div] - gcache["K2"][idx_b_tr][:, s_div], dtype=np.float64
        )

        cpu_out = {
            "tr_p": tr_p,
            "val_p": val_p,
            "s_div": s_div,
            "s_rel": s_rel,
            "m_i_div": m_i_div,
            "X_raw_tr": X_raw_tr,
            "X_raw_val": X_raw_val,
            "kept_cov_cols": kept_cov_cols,
            "dropped_cov_cols_base": dropped_cov_cols_base,
            "Kp1_all_div": Kp1_all_div,
            "K1_div": K1_div,
            "Kp2_all_div": Kp2_all_div,
            "K2_div": K2_div,
            "DK1_pair_div": DK1_pair_div,
            "DK2_pair_div": DK2_pair_div,
            "Q0_1": Q0_1,
            "Q0_2": Q0_2,
            "Q0_comb": Q0_comb,
            "base_comb_rank": base_comb_rank,
            "p_total": p_total,
            "p_pair_tr": p_pair_tr,
            "idx_a_tr": idx_a_tr,
            "idx_b_tr": idx_b_tr,
            "pair_rel": pair_rel,
            "K_p1_tr": np.ascontiguousarray(gcache["K_part_1"][np.ix_(tr_p, tr_p)], dtype=np.float64),
            "K_p1_vt": np.ascontiguousarray(gcache["K_part_1"][np.ix_(val_p, tr_p)], dtype=np.float64),
            "K_p2_tr": np.ascontiguousarray(gcache["K_part_2"][np.ix_(tr_p, tr_p)], dtype=np.float64),
            "K_p2_vt": np.ascontiguousarray(gcache["K_part_2"][np.ix_(val_p, tr_p)], dtype=np.float64),
        }
        if isinstance(cpu_cache, OrderedDict):
            cpu_cache[cache_key] = cpu_out
            while len(cpu_cache) > _MAX_DCONST_CACHE_ENTRIES:
                cpu_cache.popitem(last=False)
        else:
            if len(cpu_cache) >= _MAX_DCONST_CACHE_ENTRIES:
                cpu_cache.pop(next(iter(cpu_cache)))
            cpu_cache[cache_key] = cpu_out

    if xp is np:
        return cpu_out

    return {
        k: (v if (k in _HOST_DCONST_KEYS or v is None) else xp.asarray(v, dtype=xp.float64))
        for k, v in cpu_out.items()
    }
