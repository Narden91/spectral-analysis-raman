"""Dataset spectral Gram memoization and division-level constant precomputation."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
from scipy import linalg

from src.analysis.paired_lipid.cohort import (
    CohortDataset,
    PairedLipidConfig,
    _get_participant_aggregation_plan,
    extract_sample_band_matrix,
)
from src.analysis.paired_lipid.models import (
    CovariatePreprocessor,
    _finite_column_mean,
    _get_sample_gram_blocks,
    _symmetrize,
)

_MAX_DCONST_CACHE_ENTRIES = 256

_HOST_DCONST_KEYS: frozenset[str] = frozenset(
    {
        "s_div",
        "idx_a_tr",
        "idx_b_tr",
        "base_comb_rank",
        "p_total",
        "kept_cov_cols",
        "dropped_cov_cols_base",
        "has_missing_div",
        "single_sample_downstream",
        "Q0_comb_ratio",
    }
)
_DEVICE_INT_DCONST_KEYS: frozenset[str] = frozenset(
    {
        "tr_p",
        "val_p",
        "s_rel",
        "pair_rel",
        "p_pair_tr",
        "cov_sort_ord",
    }
)
_DEVICE_BOOL_DCONST_KEYS: frozenset[str] = frozenset(
    {
        "X_fin_tr",
        "X_fin_val",
    }
)
_MFA_SPECTRAL_DCONST_KEYS: frozenset[str] = frozenset(
    {
        "s_rel",
        "pair_rel",
        "p_pair_tr",
        "m_i_div",
        "Kp1_all_div",
        "K1_div",
        "Kp2_all_div",
        "K2_div",
        "Kp1_pair_div",
        "Kp2_pair_div",
        "DK1_pair_div",
        "DK2_pair_div",
        "Q0_1",
        "Q0_2",
        "Q0_comb",
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
    gram_blocks = _get_sample_gram_blocks(dataset)
    if gram_blocks is not None:
        K1, K2 = gram_blocks
        p_total = int(dataset.sample_1300.shape[1] + dataset.sample_2850.shape[1])
    else:
        X1 = np.asarray(dataset.sample_1300, dtype=np.float64)
        X2 = np.asarray(dataset.sample_2850, dtype=np.float64)
        X1_c = X1 - _finite_column_mean(X1)
        X2_c = X2 - _finite_column_mean(X2)
        K1 = _symmetrize(X1_c @ X1_c.T)
        K2 = _symmetrize(X2_c @ X2_c.T)
        _freeze_readonly(K1, K2)
        p_total = int(X1.shape[1] + X2.shape[1])

    A_part = np.zeros((n_p, n_s), dtype=np.float64)
    p_for_s = dataset.sample_participant_idx
    A_part[p_for_s, np.arange(n_s, dtype=int)] = 1.0 / dataset.participant_sample_counts[p_for_s].astype(np.float64)

    K_ps_1 = A_part @ K1
    K_ps_2 = A_part @ K2
    K_part_1 = _symmetrize(K_ps_1 @ A_part.T)
    K_part_2 = _symmetrize(K_ps_2 @ A_part.T)

    idx_a = dataset.pair_sample_a_indices
    idx_b = dataset.pair_sample_b_indices
    DK1_pair_all = np.ascontiguousarray(K1[idx_a] - K1[idx_b], dtype=np.float64)
    DK2_pair_all = np.ascontiguousarray(K2[idx_a] - K2[idx_b], dtype=np.float64)

    sample_band_mat, _ = extract_sample_band_matrix(dataset)
    part_band_mat = dataset._cache.get("_part_band__single_False")
    if part_band_mat is None:
        part_band_mat = A_part @ sample_band_mat
        part_band_mat.setflags(write=False)
        dataset._cache["_part_band__single_False"] = part_band_mat
    _freeze_readonly(A_part, K_ps_1, K_ps_2, K_part_1, K_part_2, DK1_pair_all, DK2_pair_all)

    spec_cache = {
        "K1": K1,
        "K2": K2,
        "A_part": A_part,
        "K_ps_1": K_ps_1,
        "K_ps_2": K_ps_2,
        "K_part_1": K_part_1,
        "K_part_2": K_part_2,
        "DK1_pair_all": DK1_pair_all,
        "DK2_pair_all": DK2_pair_all,
        "sample_band_mat": sample_band_mat,
        "part_band_mat": part_band_mat,
        "p_total": p_total,
        "spec_dconst_cache": OrderedDict(),
    }
    dataset._cache["spectral_gram_cache"] = spec_cache
    return spec_cache


def _get_single_sample_gram_cache(dataset: CohortDataset, gcache: dict[str, Any]) -> dict[str, Any]:
    """Lazily memoize single-sample downstream participant Gram blocks and band features on dataset."""
    cached = dataset._cache.get("spectral_gram_cache_single")
    if cached is not None:
        return cached

    plan = _get_participant_aggregation_plan(dataset)
    first_idx = np.asarray(plan["first_indices"], dtype=int)
    K1 = gcache["K1"]
    K2 = gcache["K2"]
    K_ps_1_ss = np.ascontiguousarray(K1[first_idx], dtype=np.float64)
    K_ps_2_ss = np.ascontiguousarray(K2[first_idx], dtype=np.float64)
    K_part_1_ss = _symmetrize(K1[np.ix_(first_idx, first_idx)])
    K_part_2_ss = _symmetrize(K2[np.ix_(first_idx, first_idx)])
    _freeze_readonly(K_ps_1_ss, K_ps_2_ss, K_part_1_ss, K_part_2_ss)

    part_band_single = dataset._cache.get("_part_band__single_True")
    if part_band_single is None:
        sample_band_mat = gcache.get("sample_band_mat")
        if sample_band_mat is None:
            sample_band_mat, _ = extract_sample_band_matrix(dataset)
        part_band_single = np.ascontiguousarray(sample_band_mat[first_idx], dtype=np.float64)
        part_band_single.setflags(write=False)
        dataset._cache["_part_band__single_True"] = part_band_single

    ss_cache = {
        "K_ps_1_ss": K_ps_1_ss,
        "K_ps_2_ss": K_ps_2_ss,
        "K_part_1_ss": K_part_1_ss,
        "K_part_2_ss": K_part_2_ss,
        "part_band_mat_single": part_band_single,
    }
    dataset._cache["spectral_gram_cache_single"] = ss_cache
    return ss_cache


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
    *,
    single_sample_downstream: bool = False,
    spectral_only: bool = False,
    skip_mfa_device_transfer: bool = False,
) -> dict[str, Any]:
    """Precompute division-specific Gram slices and warm-start bases outside the bootstrap loop."""
    cpu_cache = None if spectral_only else gcache.get("dconst_cpu_cache")
    if not spectral_only and cpu_cache is None:
        cpu_cache = OrderedDict()
        gcache["dconst_cpu_cache"] = cpu_cache

    spec_cache = gcache.get("spec_dconst_cache")
    if spec_cache is None:
        spec_cache = OrderedDict()
        gcache["spec_dconst_cache"] = spec_cache

    need_q0_comb = xp is not np
    single_flag = bool(single_sample_downstream)
    cache_key = (
        np.ascontiguousarray(tr_p, dtype=np.int64).tobytes(),
        np.ascontiguousarray(val_p, dtype=np.int64).tobytes(),
        cfg.pca_dimensions,
        float(cfg.rank_tolerance),
        single_flag,
    )
    cpu_out = None if cpu_cache is None else cpu_cache.get(cache_key)
    spec_entry = spec_cache.get(cache_key)

    if spec_entry is not None and (not need_q0_comb or spec_entry[1]):
        if isinstance(spec_cache, OrderedDict):
            spec_cache.move_to_end(cache_key)
        spec_out = spec_entry[0]
        # Refresh preset wrapper if spectral cache was upgraded (e.g., CPU -> GPU Q0_comb).
        if cpu_out is not None and cpu_out.get("K1_div") is not spec_out["K1_div"]:
            cpu_out = None
    else:
        n_tr = tr_p.size
        in_tr_mask = np.isin(dataset.sample_participant_idx, tr_p)
        s_div = np.flatnonzero(in_tr_mask)
        n_s = int(s_div.size)
        p_to_rel = np.full(dataset.n_participants, -1, dtype=int)
        p_to_rel[tr_p] = np.arange(n_tr, dtype=int)
        s_rel = p_to_rel[dataset.sample_participant_idx[s_div]]
        m_i_div = dataset.participant_sample_counts[dataset.sample_participant_idx[s_div]].astype(np.float64)

        if single_flag:
            ss_cache = _get_single_sample_gram_cache(dataset, gcache)
            Kp1_all_div = np.ascontiguousarray(ss_cache["K_ps_1_ss"][:, s_div], dtype=np.float64)
            Kp2_all_div = np.ascontiguousarray(ss_cache["K_ps_2_ss"][:, s_div], dtype=np.float64)
            K_part_1_use = ss_cache["K_part_1_ss"]
            K_part_2_use = ss_cache["K_part_2_ss"]
        else:
            if "K_ps_1" in gcache and "K_ps_2" in gcache:
                Kp1_all_div = np.ascontiguousarray(gcache["K_ps_1"][:, s_div], dtype=np.float64)
                Kp2_all_div = np.ascontiguousarray(gcache["K_ps_2"][:, s_div], dtype=np.float64)
            else:
                A_part_np = gcache["A_part"]
                Kp1_all_div = np.ascontiguousarray(A_part_np @ gcache["K1"][:, s_div], dtype=np.float64)
                Kp2_all_div = np.ascontiguousarray(A_part_np @ gcache["K2"][:, s_div], dtype=np.float64)
            K_part_1_use = gcache["K_part_1"]
            K_part_2_use = gcache["K_part_2"]
        K1_div = np.ascontiguousarray(gcache["K1"][np.ix_(s_div, s_div)], dtype=np.float64)
        K2_div = np.ascontiguousarray(gcache["K2"][np.ix_(s_div, s_div)], dtype=np.float64)
        K1_div = 0.5 * (K1_div + K1_div.T)
        K2_div = 0.5 * (K2_div + K2_div.T)

        w_s0 = (1.0 / m_i_div) / np.sum(1.0 / m_i_div)
        sw0 = np.sqrt(w_s0 * (n_s / max(n_s - 1.0, 1.0)))
        sw0_outer = sw0[:, None] * sw0[None, :]
        inv_sw0 = 1.0 / np.maximum(sw0[:, None], 1e-12)
        q_dim_1 = min(8, n_s)

        k1_0 = w_s0 @ K1_div
        m1_0 = float(np.sum(w_s0 * k1_0))
        G1_0 = (K1_div - (k1_0[:, None] + k1_0[None, :]) + m1_0) * sw0_outer
        ev1_0, U1_0 = linalg.eigh(
            G1_0, subset_by_index=[n_s - q_dim_1, n_s - 1], driver="evr", check_finite=False
        )

        k2_0 = w_s0 @ K2_div
        m2_0 = float(np.sum(w_s0 * k2_0))
        G2_0 = (K2_div - (k2_0[:, None] + k2_0[None, :]) + m2_0) * sw0_outer
        ev2_0, U2_0 = linalg.eigh(
            G2_0, subset_by_index=[n_s - q_dim_1, n_s - 1], driver="evr", check_finite=False
        )

        Q0_1 = np.ascontiguousarray(U1_0 * inv_sw0)
        Q0_2 = np.ascontiguousarray(U2_0 * inv_sw0)

        max_k = max(cfg.pca_dimensions)
        p_total = int(gcache.get("p_total", dataset.sample_1300.shape[1] + dataset.sample_2850.shape[1]))
        Q0_comb: np.ndarray | None = None
        q0_comb_ratio: float | None = None
        G_comb_0 = G1_0 / max(float(ev1_0[-1]), 1e-12) + G2_0 / max(float(ev2_0[-1]), 1e-12)
        G_comb_0 = 0.5 * (G_comb_0 + G_comb_0.T)
        if need_q0_comb and n_s > 32 and max_k <= 24:
            ev_comb_0, U_comb_0 = linalg.eigh(
                G_comb_0,
                subset_by_index=[n_s - 32, n_s - 1],
                driver="evr",
                check_finite=False,
                overwrite_a=True,
            )
            q0_comb_ratio = float(ev_comb_0[-max_k]) / max(float(ev_comb_0[0]), 1e-15)
            Q0_comb = np.ascontiguousarray(U_comb_0 * inv_sw0)
        else:
            k_check = min(n_s, max(32, max_k))
            ev_comb_0 = linalg.eigh(
                G_comb_0,
                subset_by_index=[n_s - k_check, n_s - 1],
                eigvals_only=True,
                driver="evr",
                check_finite=False,
                overwrite_a=True,
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
        if "DK1_pair_all" in gcache and "DK2_pair_all" in gcache:
            DK1_pair_div = np.ascontiguousarray(gcache["DK1_pair_all"][paired_in_tr][:, s_div], dtype=np.float64)
            DK2_pair_div = np.ascontiguousarray(gcache["DK2_pair_all"][paired_in_tr][:, s_div], dtype=np.float64)
        else:
            DK1_pair_div = np.ascontiguousarray(
                gcache["K1"][idx_a_tr][:, s_div] - gcache["K1"][idx_b_tr][:, s_div], dtype=np.float64
            )
            DK2_pair_div = np.ascontiguousarray(
                gcache["K2"][idx_a_tr][:, s_div] - gcache["K2"][idx_b_tr][:, s_div], dtype=np.float64
            )

        if single_flag:
            if "K_ps_1" in gcache and "K_ps_2" in gcache:
                Kp1_pair_div: np.ndarray | None = np.ascontiguousarray(
                    gcache["K_ps_1"][p_pair_tr][:, s_div], dtype=np.float64
                )
                Kp2_pair_div: np.ndarray | None = np.ascontiguousarray(
                    gcache["K_ps_2"][p_pair_tr][:, s_div], dtype=np.float64
                )
            else:
                A_part_pair = gcache["A_part"][p_pair_tr]
                Kp1_pair_div = np.ascontiguousarray(A_part_pair @ gcache["K1"][:, s_div], dtype=np.float64)
                Kp2_pair_div = np.ascontiguousarray(A_part_pair @ gcache["K2"][:, s_div], dtype=np.float64)
        else:
            Kp1_pair_div = None
            Kp2_pair_div = None

        spec_out = {
            "tr_p": np.ascontiguousarray(tr_p, dtype=np.int64),
            "val_p": np.ascontiguousarray(val_p, dtype=np.int64),
            "s_div": s_div,
            "s_rel": np.ascontiguousarray(s_rel, dtype=np.int64),
            "m_i_div": m_i_div,
            "Kp1_all_div": Kp1_all_div,
            "K1_div": K1_div,
            "Kp2_all_div": Kp2_all_div,
            "K2_div": K2_div,
            "Kp1_pair_div": Kp1_pair_div,
            "Kp2_pair_div": Kp2_pair_div,
            "DK1_pair_div": DK1_pair_div,
            "DK2_pair_div": DK2_pair_div,
            "Q0_1": Q0_1,
            "Q0_2": Q0_2,
            "Q0_comb": Q0_comb,
            "Q0_comb_ratio": q0_comb_ratio,
            "base_comb_rank": base_comb_rank,
            "p_total": p_total,
            "p_pair_tr": np.ascontiguousarray(p_pair_tr, dtype=np.int64),
            "idx_a_tr": idx_a_tr,
            "idx_b_tr": idx_b_tr,
            "pair_rel": np.ascontiguousarray(pair_rel, dtype=np.int64),
            "K_p1_tr": np.ascontiguousarray(K_part_1_use[np.ix_(tr_p, tr_p)], dtype=np.float64),
            "K_p1_vt": np.ascontiguousarray(K_part_1_use[np.ix_(val_p, tr_p)], dtype=np.float64),
            "K_p2_tr": np.ascontiguousarray(K_part_2_use[np.ix_(tr_p, tr_p)], dtype=np.float64),
            "K_p2_vt": np.ascontiguousarray(K_part_2_use[np.ix_(val_p, tr_p)], dtype=np.float64),
            "single_sample_downstream": single_flag,
        }
        if isinstance(spec_cache, OrderedDict):
            spec_cache[cache_key] = (spec_out, need_q0_comb)
            while len(spec_cache) > _MAX_DCONST_CACHE_ENTRIES:
                spec_cache.popitem(last=False)
        else:
            if len(spec_cache) >= _MAX_DCONST_CACHE_ENTRIES:
                spec_cache.pop(next(iter(spec_cache)))
            spec_cache[cache_key] = (spec_out, need_q0_comb)
        cpu_out = None  # Refresh preset wrapper whenever spectral entry was newly created or upgraded.

    if spectral_only:
        cpu_out = spec_out
    elif cpu_out is not None:
        if isinstance(cpu_cache, OrderedDict):
            cpu_cache.move_to_end(cache_key)
    else:
        raw_cov = np.asarray(gcache["raw_cov"], dtype=np.float64)
        raw_cov_cols = tuple(gcache.get("raw_cov_cols", (f"cov_{i}" for i in range(raw_cov.shape[1]))))
        X_raw_tr_full = raw_cov[tr_p]
        X_raw_val_full = raw_cov[val_p]
        has_missing_cov = bool(gcache.get("has_missing_cov", not np.all(np.isfinite(raw_cov))))
        if not has_missing_cov:
            ptp_tr = np.ptp(X_raw_tr_full, axis=0)
            keep_col = ptp_tr > 1e-12
            kept_cov_cols = tuple(c for c, flag in zip(raw_cov_cols, keep_col) if flag)
            dropped_cov_cols_base = tuple(c for c, flag in zip(raw_cov_cols, keep_col) if not flag)
            X_raw_tr = np.ascontiguousarray(X_raw_tr_full[:, keep_col])
            X_raw_val = np.ascontiguousarray(X_raw_val_full[:, keep_col])
            cov_extra: dict[str, Any] = {"has_missing_div": False}
        else:
            fin_tr_full = np.isfinite(X_raw_tr_full)
            fin_val_full = np.isfinite(X_raw_val_full)
            n_tr_c = int(X_raw_tr_full.shape[0])
            base_idx_list: list[int] = []
            miss_idx_list: list[int] = []
            dropped_base_list: list[str] = []
            for c_i, col_name in enumerate(raw_cov_cols):
                f_tr = fin_tr_full[:, c_i]
                if not np.any(f_tr):
                    dropped_base_list.append(col_name)
                    continue
                if float(np.ptp(X_raw_tr_full[f_tr, c_i])) > 1e-12:
                    base_idx_list.append(c_i)
                else:
                    dropped_base_list.append(col_name)
                if not np.all(f_tr):
                    miss_idx_list.append(c_i)
            base_idx_arr = np.asarray(base_idx_list, dtype=int)
            miss_idx_arr = np.asarray(miss_idx_list, dtype=int)
            kept_cov_cols = tuple(raw_cov_cols[i] for i in base_idx_list)
            dropped_cov_cols_base = tuple(dropped_base_list)
            X_b_tr = (
                X_raw_tr_full[:, base_idx_arr]
                if base_idx_arr.size
                else np.empty((n_tr_c, 0), dtype=np.float64)
            )
            X_b_val = (
                X_raw_val_full[:, base_idx_arr]
                if base_idx_arr.size
                else np.empty((val_p.size, 0), dtype=np.float64)
            )
            fin_b_tr = np.isfinite(X_b_tr)
            fin_b_val = np.isfinite(X_b_val)
            any_div_missing = bool(
                (not np.all(fin_b_tr)) or (not np.all(fin_b_val)) or (miss_idx_arr.size > 0)
            )
            if not any_div_missing:
                X_raw_tr = np.ascontiguousarray(X_b_tr, dtype=np.float64)
                X_raw_val = np.ascontiguousarray(X_b_val, dtype=np.float64)
                cov_extra = {"has_missing_div": False}
            else:
                X_raw_tr = np.ascontiguousarray(np.where(fin_b_tr, X_b_tr, 0.0), dtype=np.float64)
                X_raw_val = np.ascontiguousarray(np.where(fin_b_val, X_b_val, 0.0), dtype=np.float64)
                p_b = int(base_idx_arr.size)
                cov_sort_ord = np.empty((p_b, n_tr_c), dtype=np.int64)
                cov_sort_vals = np.empty((p_b, n_tr_c), dtype=np.float64)
                cov_sort_fin = np.empty((p_b, n_tr_c), dtype=np.float64)
                for j in range(p_b):
                    fi = np.flatnonzero(fin_b_tr[:, j])
                    ni = np.flatnonzero(~fin_b_tr[:, j])
                    so = fi[np.argsort(X_b_tr[fi, j], kind="mergesort")]
                    full_o = np.concatenate([so, ni]).astype(np.int64, copy=False)
                    cov_sort_ord[j] = full_o
                    cov_sort_vals[j] = X_raw_tr[full_o, j]
                    cov_sort_fin[j] = fin_b_tr[full_o, j].astype(np.float64)
                X_miss_tr = (
                    np.ascontiguousarray((~fin_tr_full[:, miss_idx_arr]).astype(np.float64))
                    if miss_idx_arr.size
                    else np.empty((n_tr_c, 0), dtype=np.float64)
                )
                X_miss_val = (
                    np.ascontiguousarray((~fin_val_full[:, miss_idx_arr]).astype(np.float64))
                    if miss_idx_arr.size
                    else np.empty((val_p.size, 0), dtype=np.float64)
                )
                cov_extra = {
                    "has_missing_div": True,
                    "X_fin_tr": np.ascontiguousarray(fin_b_tr, dtype=bool),
                    "X_fin_val": np.ascontiguousarray(fin_b_val, dtype=bool),
                    "cov_sort_ord": np.ascontiguousarray(cov_sort_ord, dtype=np.int64),
                    "cov_sort_vals": np.ascontiguousarray(cov_sort_vals, dtype=np.float64),
                    "cov_sort_fin": np.ascontiguousarray(cov_sort_fin, dtype=np.float64),
                    "X_miss_tr": X_miss_tr,
                    "X_miss_val": X_miss_val,
                }

        cpu_out = {
            **spec_out,
            "X_raw_tr": X_raw_tr,
            "X_raw_val": X_raw_val,
            "kept_cov_cols": kept_cov_cols,
            "dropped_cov_cols_base": dropped_cov_cols_base,
            **cov_extra,
        }
        if isinstance(cpu_cache, OrderedDict):
            cpu_cache[cache_key] = cpu_out
            while len(cpu_cache) > _MAX_DCONST_CACHE_ENTRIES:
                cpu_cache.popitem(last=False)
        elif cpu_cache is not None:
            if len(cpu_cache) >= _MAX_DCONST_CACHE_ENTRIES:
                cpu_cache.pop(next(iter(cpu_cache)))
            cpu_cache[cache_key] = cpu_out

    if xp is np:
        return cpu_out

    out_dev: dict[str, Any] = {}
    for k, v in cpu_out.items():
        if skip_mfa_device_transfer and k in _MFA_SPECTRAL_DCONST_KEYS:
            continue
        if k in _HOST_DCONST_KEYS or v is None:
            out_dev[k] = v
        elif k in _DEVICE_INT_DCONST_KEYS:
            out_dev[k] = xp.asarray(v, dtype=xp.int64)
        elif k in _DEVICE_BOOL_DCONST_KEYS:
            out_dev[k] = xp.asarray(v, dtype=bool)
        else:
            out_dev[k] = xp.asarray(v, dtype=xp.float64)
    return out_dev
