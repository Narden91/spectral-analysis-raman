"""Batched closed-form models: division prep, Schur PCA/paired ridge, multi-alpha ridge, dual PLS."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import linalg

from src.analysis.paired_lipid._backend import (
    _cupyx_supports_batched_triangular,
    _get_cupyx_linalg,
    _get_fused_mfa_gram_kernel,
)
from src.analysis.paired_lipid._eigensolvers import _fast_top1_eigvalsh_from_gram, _top_k_eigh
from src.analysis.paired_lipid.cohort import PairedLipidConfig


def _prep_covariates_batch(
    xp: Any,
    dconst: dict[str, Any],
    M_tr: Any,
    M_val: Any,
    *,
    return_valid_mask: bool = False,
) -> tuple[Any, ...]:
    """Compute normalized participant weights and standardized training/validation covariates."""
    w_tr_sum = xp.sum(M_tr, axis=1, keepdims=True)
    w_tr = M_tr / xp.maximum(w_tr_sum, 1e-12)
    w_val = M_val

    X_raw_tr = dconst["X_raw_tr"]
    X_raw_val = dconst["X_raw_val"]
    B = int(M_tr.shape[0])
    n_tr = int(X_raw_tr.shape[0])
    n_val = int(X_raw_val.shape[0])

    if not dconst.get("has_missing_div", False):
        if int(X_raw_tr.shape[1]) == 0:
            X_cov_tr = xp.zeros((B, n_tr, 0), dtype=xp.float64)
            X_cov_val = xp.zeros((B, n_val, 0), dtype=xp.float64)
            if return_valid_mask:
                return w_tr_sum, w_tr, w_val, X_cov_tr, X_cov_val, xp.zeros((B, 0), dtype=bool)
            return w_tr_sum, w_tr, w_val, X_cov_tr, X_cov_val
        c_mean = w_tr @ X_raw_tr
        C_tr = X_raw_tr[None, :, :] - c_mean[:, None, :]
        c_var = xp.sum(w_tr[:, :, None] * (C_tr * C_tr), axis=1)
        # Match CovariatePreprocessor.fit: drop columns only when active-participant ptp <= 1e-12,
        # then standardize with min_var=1e-12 so small-scale non-constant covariates stay identical.
        active = M_tr > 0.0
        x_hi = xp.where(active[:, :, None], X_raw_tr[None, :, :], -xp.inf)
        x_lo = xp.where(active[:, :, None], X_raw_tr[None, :, :], xp.inf)
        c_ptp = xp.max(x_hi, axis=1) - xp.min(x_lo, axis=1)
        valid_c = xp.isfinite(c_ptp) & (c_ptp > 1e-12)
        c_scale = xp.where(valid_c, xp.sqrt(xp.maximum(c_var, 1e-12)), 1.0)
        X_cov_tr = xp.where(valid_c[:, None, :], C_tr / c_scale[:, None, :], 0.0)
        X_cov_val = xp.where(
            valid_c[:, None, :],
            (X_raw_val[None, :, :] - c_mean[:, None, :]) / c_scale[:, None, :],
            0.0,
        )
        if return_valid_mask:
            return w_tr_sum, w_tr, w_val, X_cov_tr, X_cov_val, valid_c
        return w_tr_sum, w_tr, w_val, X_cov_tr, X_cov_val

    p_b = int(X_raw_tr.shape[1])
    X_miss_tr = dconst["X_miss_tr"]
    X_miss_val = dconst["X_miss_val"]
    p_m = int(X_miss_tr.shape[1])
    active = M_tr > 0.0

    if p_b > 0:
        X_fin_tr = dconst["X_fin_tr"]
        X_fin_val = dconst["X_fin_val"]
        cov_sort_ord = dconst["cov_sort_ord"]
        cov_sort_vals = dconst["cov_sort_vals"]
        cov_sort_fin = dconst["cov_sort_fin"]

        m_sorted = M_tr[:, cov_sort_ord] * cov_sort_fin[None, :, :]
        cdf = xp.cumsum(m_sorted, axis=2)
        total_fin_w = cdf[:, :, -1]
        any_finite = total_fin_w > 0.0
        med_idx = xp.argmax((cdf > 0.0) & (cdf >= (0.5 * total_fin_w)[:, :, None]), axis=2)
        medians = xp.where(any_finite, cov_sort_vals[xp.arange(p_b)[None, :], med_idx], 0.0)

        X_imp_tr = xp.where(X_fin_tr[None, :, :], X_raw_tr[None, :, :], medians[:, None, :])
        X_imp_val = xp.where(X_fin_val[None, :, :], X_raw_val[None, :, :], medians[:, None, :])
        c_mean = (w_tr[:, None, :] @ X_imp_tr)[:, 0, :]
        C_tr = X_imp_tr - c_mean[:, None, :]
        c_var = xp.sum(w_tr[:, :, None] * (C_tr * C_tr), axis=1)
        active_fin = active[:, :, None] & X_fin_tr[None, :, :]
        x_hi = xp.where(active_fin, X_raw_tr[None, :, :], -xp.inf)
        x_lo = xp.where(active_fin, X_raw_tr[None, :, :], xp.inf)
        c_ptp = xp.max(x_hi, axis=1) - xp.min(x_lo, axis=1)
        valid_c = any_finite & xp.isfinite(c_ptp) & (c_ptp > 1e-12)
        c_scale = xp.where(valid_c, xp.sqrt(xp.maximum(c_var, 1e-12)), 1.0)
        X_base_tr = xp.where(valid_c[:, None, :], C_tr / c_scale[:, None, :], 0.0)
        X_base_val = xp.where(
            valid_c[:, None, :],
            (X_imp_val - c_mean[:, None, :]) / c_scale[:, None, :],
            0.0,
        )
    else:
        valid_c = xp.zeros((B, 0), dtype=bool)
        X_base_tr = xp.zeros((B, n_tr, 0), dtype=xp.float64)
        X_base_val = xp.zeros((B, n_val, 0), dtype=xp.float64)

    if p_m > 0:
        active_f = active.astype(xp.float64)
        n_act_miss = active_f @ X_miss_tr
        n_act_tot = xp.sum(active_f, axis=1, keepdims=True)
        valid_miss = (n_act_miss > 0.0) & (n_act_miss < n_act_tot)
        m_mean = w_tr @ X_miss_tr
        M_c_tr = X_miss_tr[None, :, :] - m_mean[:, None, :]
        m_var = xp.sum(w_tr[:, :, None] * (M_c_tr * M_c_tr), axis=1)
        m_scale = xp.where(valid_miss, xp.sqrt(xp.maximum(m_var, 1e-12)), 1.0)
        X_miss_std_tr = xp.where(valid_miss[:, None, :], M_c_tr / m_scale[:, None, :], 0.0)
        X_miss_std_val = xp.where(
            valid_miss[:, None, :],
            (X_miss_val[None, :, :] - m_mean[:, None, :]) / m_scale[:, None, :],
            0.0,
        )
        if p_b > 0:
            X_cov_tr = xp.concatenate([X_base_tr, X_miss_std_tr], axis=2)
            X_cov_val = xp.concatenate([X_base_val, X_miss_std_val], axis=2)
        else:
            X_cov_tr = X_miss_std_tr
            X_cov_val = X_miss_std_val
    else:
        X_cov_tr = X_base_tr
        X_cov_val = X_base_val

    if return_valid_mask:
        return w_tr_sum, w_tr, w_val, X_cov_tr, X_cov_val, valid_c
    return w_tr_sum, w_tr, w_val, X_cov_tr, X_cov_val


def _prep_mfa_pca_batch(
    xp: Any,
    gcache: dict[str, Any],
    dconst: dict[str, Any],
    M_tr: Any,
    w_tr_sum: Any,
    cfg: PairedLipidConfig,
    *,
    compute_full_rank: bool = False,
) -> dict[str, Any]:
    """Compute batched MFA block scales, top-k spectral PCA projections, and rank validity."""
    s_rel = dconst["s_rel"]
    m_i_div = dconst["m_i_div"]
    m_s = M_tr[:, s_rel]
    w_s_raw = m_s / (w_tr_sum * m_i_div[None, :])
    w_s = w_s_raw / xp.maximum(xp.sum(w_s_raw, axis=1, keepdims=True), 1e-12)
    m_total = xp.sum(m_s, axis=1)
    denom_factor = m_total / xp.maximum(m_total - 1.0, 1.0)
    sqrt_w = xp.sqrt(w_s * denom_factor[:, None])

    Kp1_all_div = dconst["Kp1_all_div"]
    K1_div = dconst["K1_div"]
    Kp2_all_div = dconst["Kp2_all_div"]
    K2_div = dconst["K2_div"]

    # Compute leading block eigenvalues matrix-free via shared GEMMs on K1_div/K2_div so no
    # per-block (B, n_s, n_s) Gram buffers are materialized.
    min_eig_clamp = min(1e-12, float(cfg.eigenvalue_tolerance))
    k1_mu_div = w_s @ K1_div
    mu1_sq = xp.sum(w_s * k1_mu_div, axis=1)
    eig1_1300 = _fast_top1_eigvalsh_from_gram(xp, K1_div, k1_mu_div, mu1_sq, sqrt_w, dconst["Q0_1"])
    scale1_sq = 1.0 / xp.maximum(eig1_1300, min_eig_clamp)

    k2_mu_div = w_s @ K2_div
    mu2_sq = xp.sum(w_s * k2_mu_div, axis=1)
    eig1_2850 = _fast_top1_eigvalsh_from_gram(xp, K2_div, k2_mu_div, mu2_sq, sqrt_w, dconst["Q0_2"])
    scale2_sq = 1.0 / xp.maximum(eig1_2850, min_eig_clamp)

    # Match WeightedSpectralPreprocessor.fit: detect large subset-mean shift cancellation
    # directly from the centered block Gram diagonals without materializing (B, n_s, n_s) buffers.
    sw_sq = sqrt_w * sqrt_w
    tr_g1 = xp.maximum(
        xp.sum((xp.diagonal(K1_div)[None, :] - 2.0 * k1_mu_div + mu1_sq[:, None]) * sw_sq, axis=1),
        0.0,
    )
    tr_g2 = xp.maximum(
        xp.sum((xp.diagonal(K2_div)[None, :] - 2.0 * k2_mu_div + mu2_sq[:, None]) * sw_sq, axis=1),
        0.0,
    )
    shift_ok = (
        xp.isfinite(mu1_sq)
        & xp.isfinite(tr_g1)
        & (mu1_sq <= 1e2 * tr_g1)
        & xp.isfinite(mu2_sq)
        & xp.isfinite(tr_g2)
        & (mu2_sq <= 1e2 * tr_g2)
    )
    del sw_sq

    k_mu = scale1_sq[:, None] * k1_mu_div + scale2_sq[:, None] * k2_mu_div
    mu_sq = scale1_sq * mu1_sq + scale2_sq * mu2_sq

    max_k = max(cfg.pca_dimensions)
    p_total = int(dconst.get("p_total", gcache.get("p_total", sqrt_w.shape[1])))
    n_act = xp.sum(sqrt_w > 0, axis=1)
    max_dim = xp.maximum(n_act, float(p_total))
    eps_f = float(np.finfo(float).eps)
    full_numerical_rank = None

    if xp is np:
        # On CPU, assemble only each draw's active (n_act x n_act) sub-Gram in L1 cache rather
        # than allocating a full (B, n_s, n_s) tensor with ~37% zero rows and columns.
        B_n, n_s = sqrt_w.shape
        eigvals_top = np.zeros((B_n, max_k), dtype=np.float64)
        eigvecs_top = np.zeros((B_n, n_s, max_k), dtype=np.float64)
        active_all_ev: list[np.ndarray] | None = [] if compute_full_rank else None
        for b in range(B_n):
            act = np.flatnonzero(sqrt_w[b] > 0.0)
            na = int(act.size)
            kb = min(max_k, na)
            if kb <= 0:
                if active_all_ev is not None:
                    active_all_ev.append(np.empty(0, dtype=np.float64))
                continue
            sw_a = sqrt_w[b, act]
            km_a = k_mu[b, act]
            G_sub = K1_div[act][:, act]
            G_sub *= scale1_sq[b]
            G_sub += scale2_sq[b] * K2_div[act][:, act]
            G_sub -= km_a[:, None]
            G_sub -= km_a[None, :]
            G_sub += mu_sq[b]
            G_sub *= sw_a[:, None]
            G_sub *= sw_a[None, :]
            if active_all_ev is not None:
                active_all_ev.append(linalg.eigvalsh(G_sub, check_finite=False))
            ev_b, u_b = linalg.eigh(
                G_sub,
                subset_by_index=[na - kb, na - 1],
                driver="evr",
                check_finite=False,
                overwrite_a=True,
            )
            eigvals_top[b, :kb] = np.maximum(ev_b[::-1], 0.0)
            eigvecs_top[b, act, :kb] = u_b[:, ::-1]
        s_kept = np.sqrt(eigvals_top)
        s0 = np.maximum(s_kept[:, 0], 1.0)
        gram_s_tol = np.sqrt(max_dim * eps_f) * s0
        primal_tol = np.maximum(max_dim * eps_f * s0, float(cfg.rank_tolerance))
        rank_thresh = np.maximum(10.0 * gram_s_tol, primal_tol)
        if active_all_ev is not None:
            full_numerical_rank = np.array(
                [int(np.sum(np.sqrt(np.maximum(ev_a, 0.0)) > rank_thresh[b])) for b, ev_a in enumerate(active_all_ev)],
                dtype=int,
            )
    else:
        # On GPU, build the fused symmetric (B, n_s, n_s) buffer in a single kernel pass with zero temporaries.
        B_n, n_s = int(sqrt_w.shape[0]), int(sqrt_w.shape[1])
        fused_kernel = _get_fused_mfa_gram_kernel()
        if fused_kernel is not None:
            gram = xp.empty((B_n, n_s, n_s), dtype=xp.float64)
            fused_kernel(
                xp.ascontiguousarray(K1_div, dtype=xp.float64),
                xp.ascontiguousarray(K2_div, dtype=xp.float64),
                xp.ascontiguousarray(scale1_sq, dtype=xp.float64),
                xp.ascontiguousarray(scale2_sq, dtype=xp.float64),
                xp.ascontiguousarray(k_mu, dtype=xp.float64),
                xp.ascontiguousarray(mu_sq, dtype=xp.float64),
                xp.ascontiguousarray(sqrt_w, dtype=xp.float64),
                np.int32(n_s),
                gram,
            )
        else:
            gram = scale1_sq[:, None, None] * K1_div[None, :, :]
            gram += scale2_sq[:, None, None] * K2_div[None, :, :]
            gram -= k_mu[:, :, None]
            gram -= k_mu[:, None, :]
            gram += mu_sq[:, None, None]
            gram *= sqrt_w[:, :, None]
            gram *= sqrt_w[:, None, :]

        eigvals_top, eigvecs_top = _top_k_eigh(
            xp, gram, max_k, sqrt_w, dconst.get("Q0_comb"), is_symmetric=True
        )
        s_kept = xp.sqrt(eigvals_top)
        s0 = xp.maximum(s_kept[:, 0], 1.0)
        gram_s_tol = xp.sqrt(max_dim * eps_f) * s0
        primal_tol = xp.maximum(max_dim * eps_f * s0, float(cfg.rank_tolerance))
        rank_thresh = xp.maximum(10.0 * gram_s_tol, primal_tol)
        if compute_full_rank:
            all_ev = xp.linalg.eigvalsh(gram)
            all_s = xp.sqrt(xp.maximum(all_ev, 0.0))
            full_numerical_rank = xp.sum(all_s > rank_thresh[:, None], axis=1)
        del gram

    # Squared-Gram eigenvalues have an O(max_dim * eps * s0^2) roundoff floor, so singular
    # values at or below 10 * sqrt(max_dim * eps) * s0 must route to the primal SVD fallback,
    # matching WeightedSpectralPreprocessor.fit exactly.
    rank_ok = (
        shift_ok
        & xp.isfinite(s_kept[:, -1])
        & (s_kept[:, -1] > rank_thresh)
        & xp.isfinite(eig1_1300)
        & (eig1_1300 > float(cfg.eigenvalue_tolerance))
        & xp.isfinite(eig1_2850)
        & (eig1_2850 > float(cfg.eigenvalue_tolerance))
    )

    alpha = (sqrt_w[:, :, None] * eigvecs_top) / xp.maximum(s_kept[:, None, :], 1e-12)
    del eigvecs_top, eigvals_top, s_kept, sqrt_w
    alpha_sum = xp.sum(alpha, axis=1)
    kp1_mu_all = w_s @ Kp1_all_div.T
    kp2_mu_all = w_s @ Kp2_all_div.T

    p_term1 = (
        (Kp1_all_div @ alpha)
        - kp1_mu_all[:, :, None] * alpha_sum[:, None, :]
        - xp.sum((k1_mu_div - mu1_sq[:, None])[:, :, None] * alpha, axis=1)[:, None, :]
    )
    p_term2 = (
        (Kp2_all_div @ alpha)
        - kp2_mu_all[:, :, None] * alpha_sum[:, None, :]
        - xp.sum((k2_mu_div - mu2_sq[:, None])[:, :, None] * alpha, axis=1)[:, None, :]
    )
    part_pca_max = scale1_sq[:, None, None] * p_term1 + scale2_sq[:, None, None] * p_term2
    del p_term1, p_term2

    res_mfa: dict[str, Any] = {
        "w_s": w_s,
        "alpha": alpha,
        "alpha_sum": alpha_sum,
        "scale1_sq": scale1_sq,
        "scale2_sq": scale2_sq,
        "part_pca_max": part_pca_max,
        "rank_ok": rank_ok,
    }
    if full_numerical_rank is not None:
        res_mfa["numerical_rank"] = full_numerical_rank
    return res_mfa


def _prep_paired_moments_batch(
    xp: Any,
    dconst: dict[str, Any],
    M_tr: Any,
    w_s: Any,
    alpha: Any,
    alpha_sum: Any,
    scale1_sq: Any,
    scale2_sq: Any,
    part_pca_max: Any,
) -> tuple[Any, Any]:
    """Compute within-pair W_max and between-participant T_max in max-PCA coordinates."""
    DK1_pair_div = dconst["DK1_pair_div"]
    DK2_pair_div = dconst["DK2_pair_div"]
    p_pair_tr = dconst["p_pair_tr"]
    pair_rel = dconst["pair_rel"]

    mult_pair = M_tr[:, pair_rel]
    n_pair_mult = xp.sum(mult_pair, axis=1)
    w_pair = mult_pair / xp.maximum(n_pair_mult[:, None], 1e-12)

    dk1_mu = w_s @ DK1_pair_div.T
    dk2_mu = w_s @ DK2_pair_div.T
    diff_1 = (DK1_pair_div @ alpha) - dk1_mu[:, :, None] * alpha_sum[:, None, :]
    diff_2 = (DK2_pair_div @ alpha) - dk2_mu[:, :, None] * alpha_sum[:, None, :]
    raw_diff = scale1_sq[:, None, None] * diff_1 + scale2_sq[:, None, None] * diff_2
    W_max = 0.5 * (xp.swapaxes(w_pair[:, :, None] * raw_diff, -1, -2) @ raw_diff)
    W_max = 0.5 * (W_max + xp.swapaxes(W_max, -1, -2))

    Kp1_pair_div = dconst.get("Kp1_pair_div")
    Kp2_pair_div = dconst.get("Kp2_pair_div")
    if Kp1_pair_div is not None and Kp2_pair_div is not None:
        # In single_sample_downstream mode, downstream part_pca_max holds first-sample coordinates,
        # while training between-participant covariance T_max still uses paired-sample averages.
        kp1_mu_pair = w_s @ Kp1_pair_div.T
        kp2_mu_pair = w_s @ Kp2_pair_div.T
        p1_pair = (Kp1_pair_div @ alpha) - kp1_mu_pair[:, :, None] * alpha_sum[:, None, :]
        p2_pair = (Kp2_pair_div @ alpha) - kp2_mu_pair[:, :, None] * alpha_sum[:, None, :]
        zbar = scale1_sq[:, None, None] * p1_pair + scale2_sq[:, None, None] * p2_pair
    else:
        zbar = part_pca_max[:, p_pair_tr, :]
    mu_pair = xp.sum(w_pair[:, :, None] * zbar, axis=1)
    z_cent = zbar - mu_pair[:, None, :]
    pair_denom = n_pair_mult / xp.maximum(n_pair_mult - 1.0, 1.0)
    T_max = pair_denom[:, None, None] * (xp.swapaxes(w_pair[:, :, None] * z_cent, -1, -2) @ z_cent)
    T_max = 0.5 * (T_max + xp.swapaxes(T_max, -1, -2))
    return W_max, T_max


def _prep_schur_sufficient_stats(
    xp: Any,
    X_cov_tr: Any,
    X_cov_val: Any,
    y_tr: Any,
    y_val: Any,
    w_tr: Any,
    w_val: Any,
    part_pca_max: Any,
    tr_p: np.ndarray,
    val_p: np.ndarray,
    alphas_xp: Any,
    *,
    compute_pca_suff: bool,
) -> dict[str, Any]:
    """Compute covariate-only ridge predictions and Schur complement stats for max_k PCA."""
    y_mean = xp.sum(w_tr * y_tr[None, :], axis=1)
    yc = y_tr[None, :] - y_mean[:, None]
    cm = (w_tr[:, None, :] @ X_cov_tr)[:, 0, :]
    Cc_tr = X_cov_tr - cm[:, None, :]
    Cc_val = X_cov_val - cm[:, None, :]
    Cw = Cc_tr * w_tr[:, :, None]
    Cw_T = xp.swapaxes(Cw, -1, -2)
    G_cc = Cw_T @ Cc_tr
    G_cc = 0.5 * (G_cc + xp.swapaxes(G_cc, -1, -2))
    g_cy = (Cw_T @ yc[:, :, None])[:, :, 0]

    p_cov = int(X_cov_tr.shape[2])
    if p_cov == 0:
        B_n = int(X_cov_tr.shape[0])
        A_n = int(alphas_xp.shape[0])
        n_val = int(X_cov_val.shape[1])
        inv_c = xp.zeros((B_n, A_n, 0, 0), dtype=xp.float64)
        beta_c0 = xp.zeros((B_n, A_n, 0), dtype=xp.float64)
        preds_cov = xp.broadcast_to(y_mean[:, None, None], (B_n, n_val, A_n)).copy()
    else:
        evals_c, evecs_c = xp.linalg.eigh(G_cc)
        evals_c = xp.maximum(evals_c, 0.0)
        inv_eig = 1.0 / (evals_c[:, None, :] + alphas_xp[None, :, None])
        evecs_c_T = xp.swapaxes(evecs_c, -1, -2)
        inv_c = (evecs_c[:, None, :, :] * inv_eig[:, :, None, :]) @ evecs_c_T[:, None, :, :]
        beta_c0 = (inv_c @ g_cy[:, None, :, None])[:, :, :, 0]
        preds_cov = y_mean[:, None, None] + (Cc_val @ xp.swapaxes(beta_c0, -1, -2))
    stats_out: dict[str, Any] = {"preds_cov": preds_cov}

    if compute_pca_suff:
        r_val_0 = y_val[None, :, None] - preds_cov
        sse_0 = xp.sum(w_val[:, :, None] * (r_val_0 * r_val_0), axis=1)

        pca_tr = part_pca_max[:, tr_p, :]
        pca_val = part_pca_max[:, val_p, :]
        p_mean = (w_tr[:, None, :] @ pca_tr)[:, 0, :]
        Pc_tr = pca_tr - p_mean[:, None, :]
        Pc_val = pca_val - p_mean[:, None, :]
        Pw = Pc_tr * w_tr[:, :, None]
        Pw_T = xp.swapaxes(Pw, -1, -2)

        Sigma_p = Pw_T @ Pc_tr
        G_cp = Cw_T @ Pc_tr
        g_py = (Pw_T @ yc[:, :, None])[:, :, 0]

        H_p = inv_c @ G_cp[:, None, :, :]
        P_tilde_val = Pc_val[:, None, :, :] - (Cc_val[:, None, :, :] @ H_p)
        P_tilde_w = P_tilde_val * w_val[:, None, :, None]
        P_tilde_w_T = xp.swapaxes(P_tilde_w, -1, -2)
        Val_gram_p = P_tilde_w_T @ P_tilde_val
        Val_rhs_p = (P_tilde_w_T @ xp.swapaxes(r_val_0, 1, 2)[:, :, :, None])[:, :, :, 0]
        Schur_p = Sigma_p[:, None, :, :] - (xp.swapaxes(G_cp, -1, -2)[:, None, :, :] @ H_p)
        g_tilde_p = g_py[:, None, :] - (beta_c0 @ G_cp)

        stats_out.update(
            {
                "sse_0": sse_0,
                "Sigma_p": Sigma_p,
                "Schur_p": Schur_p,
                "g_tilde_p": g_tilde_p,
                "Val_gram_p": Val_gram_p,
                "Val_rhs_p": Val_rhs_p,
            }
        )
    return stats_out


def batched_division_prep(
    xp: Any,
    gcache: dict[str, Any],
    dconst: dict[str, Any],
    M_tr: Any,
    M_val: Any,
    y_all_xp: Any,
    alphas_xp: Any,
    cfg: PairedLipidConfig,
    *,
    compute_pca_suff: bool = True,
    compute_cov_preds: bool = False,
    compute_full_rank: bool = False,
) -> dict[str, Any]:
    """Compute batched MFA-PCA coordinates, paired moments, and max_k Schur sufficient statistics."""
    tr_p = dconst["tr_p"]
    val_p = dconst["val_p"]

    if compute_full_rank:
        w_tr_sum, w_tr, w_val, X_cov_tr, X_cov_val, valid_cov_mask = _prep_covariates_batch(
            xp, dconst, M_tr, M_val, return_valid_mask=True
        )
    else:
        w_tr_sum, w_tr, w_val, X_cov_tr, X_cov_val = _prep_covariates_batch(xp, dconst, M_tr, M_val)
        valid_cov_mask = None
    mfa = _prep_mfa_pca_batch(xp, gcache, dconst, M_tr, w_tr_sum, cfg, compute_full_rank=compute_full_rank)
    W_max, T_max = _prep_paired_moments_batch(
        xp,
        dconst,
        M_tr,
        mfa["w_s"],
        mfa["alpha"],
        mfa["alpha_sum"],
        mfa["scale1_sq"],
        mfa["scale2_sq"],
        mfa["part_pca_max"],
    )

    y_tr = y_all_xp[tr_p]
    y_val = y_all_xp[val_p]

    out: dict[str, Any] = {
        "tr_p": tr_p,
        "val_p": val_p,
        "w_tr": w_tr,
        "w_val": w_val,
        "y_tr": y_tr,
        "y_val": y_val,
        "X_cov_tr": X_cov_tr,
        "X_cov_val": X_cov_val,
        "part_pca_max": mfa["part_pca_max"],
        "W_max": W_max,
        "T_max": T_max,
        "scale1_sq": mfa["scale1_sq"],
        "scale2_sq": mfa["scale2_sq"],
        "rank_ok": mfa["rank_ok"],
        "K_p1_tr": dconst["K_p1_tr"],
        "K_p1_vt": dconst["K_p1_vt"],
        "K_p2_tr": dconst["K_p2_tr"],
        "K_p2_vt": dconst["K_p2_vt"],
    }
    if compute_full_rank:
        out["numerical_rank"] = mfa["numerical_rank"]
        out["valid_cov_mask"] = valid_cov_mask

    if compute_pca_suff or compute_cov_preds:
        out.update(
            _prep_schur_sufficient_stats(
                xp,
                X_cov_tr,
                X_cov_val,
                y_tr,
                y_val,
                w_tr,
                w_val,
                mfa["part_pca_max"],
                tr_p,
                val_p,
                alphas_xp,
                compute_pca_suff=compute_pca_suff,
            )
        )
    return out


def batched_pca_schur_sse(
    xp: Any,
    div: dict[str, Any],
    k_list: list[int],
    alphas_xp: Any,
) -> Any:
    """Evaluate all (k, alpha) PCA-ridge validation SSEs via standardized Schur Cholesky solves."""
    Sigma_p = div["Sigma_p"]
    Schur_p = div["Schur_p"]
    g_tilde_p = div["g_tilde_p"]
    Val_gram_p = div["Val_gram_p"]
    Val_rhs_p = div["Val_rhs_p"]
    sse_0 = div["sse_0"]

    B = Sigma_p.shape[0]
    A = alphas_xp.shape[0]
    if not k_list:
        return xp.zeros((B, 0, A), dtype=xp.float64)

    z_scale = xp.sqrt(xp.maximum(xp.diagonal(Sigma_p, axis1=-2, axis2=-1), 1e-12))
    inv_scale = 1.0 / z_scale
    scale_outer = inv_scale[:, None, :, None] * inv_scale[:, None, None, :]
    Schur_std = Schur_p * scale_outer
    Schur_std = 0.5 * (Schur_std + xp.swapaxes(Schur_std, -1, -2))
    g_std = g_tilde_p * inv_scale[:, None, :]
    Val_gram_std = Val_gram_p * scale_outer
    Val_rhs_std = Val_rhs_p * inv_scale[:, None, :]

    cpx_linalg = (
        _get_cupyx_linalg()
        if (xp is not np and _cupyx_supports_batched_triangular())
        else None
    )

    sse_by_k = []
    for k in k_list:
        eye_k = xp.eye(k, dtype=xp.float64)[None, None, :, :]
        Sk = Schur_std[:, :, :k, :k] + alphas_xp[None, :, None, None] * eye_k
        gk = g_std[:, :, :k]
        beta: Any | None = None
        L: Any | None = None
        try:
            L = xp.linalg.cholesky(Sk)
        except Exception:
            L = None
        if L is not None and bool(
            xp.all(xp.isfinite(L)) & xp.all(xp.diagonal(L, axis1=-2, axis2=-1) > 0.0)
        ):
            if cpx_linalg is not None:
                try:
                    L_flat = L.reshape(B * A, k, k)
                    u = cpx_linalg.solve_triangular(
                        L_flat, gk.reshape(B * A, k, 1), lower=True, check_finite=False
                    )
                    beta = cpx_linalg.solve_triangular(
                        L_flat, u, lower=True, trans="T", check_finite=False
                    ).reshape(B, A, k)
                except Exception:
                    beta = None
            if beta is None:
                u = xp.linalg.solve(L, gk[:, :, :, None])
                beta = xp.linalg.solve(xp.swapaxes(L, -1, -2), u)[:, :, :, 0]
        if beta is None:
            # Recover valid draws via symmetrized eigh when any draw in the batch is non-PD or non-finite,
            # propagating NaN only to corrupted slices so per-draw fallback stays isolated.
            sk_fin = xp.all(xp.isfinite(Sk), axis=(-2, -1)) & xp.all(xp.isfinite(gk), axis=-1)
            Sk_safe = xp.where(sk_fin[:, :, None, None], Sk, eye_k)
            Sk_safe = 0.5 * (Sk_safe + xp.swapaxes(Sk_safe, -1, -2))
            gk_safe = xp.where(sk_fin[:, :, None], gk, 0.0)
            ev_s, U_s = xp.linalg.eigh(Sk_safe)
            max_ev = xp.maximum(ev_s[:, :, -1], 1.0)
            psd_ok = sk_fin & (ev_s[:, :, 0] >= -1e-10 * max_ev)
            ev_clamped = xp.maximum(ev_s, 0.0)
            proj_g = (xp.swapaxes(U_s, -1, -2) @ gk_safe[:, :, :, None])[:, :, :, 0]
            scaled_g = xp.where(ev_clamped > 1e-15, proj_g / xp.maximum(ev_clamped, 1e-15), 0.0)
            beta_eigh = (U_s @ scaled_g[:, :, :, None])[:, :, :, 0]
            beta = xp.where(psd_ok[:, :, None], beta_eigh, xp.nan)
        Vg_b = (Val_gram_std[:, :, :k, :k] @ beta[:, :, :, None])[:, :, :, 0]
        quad = xp.sum(beta * Vg_b, axis=-1)
        lin = xp.sum(beta * Val_rhs_std[:, :, :k], axis=-1)
        sse_by_k.append(sse_0 - 2.0 * lin + quad)
    return xp.stack(sse_by_k, axis=1)


def batched_paired_schur_sse(
    xp: Any,
    div: dict[str, Any],
    V_grid: dict[int, Any],
    k_list: list[int],
    alphas_xp: Any,
) -> Any:
    """Evaluate all (k, lam, r=1..3, alpha) validation SSEs simultaneously via 3x3 nested LDL^T."""
    Sigma_p = div["Sigma_p"]
    Schur_p = div["Schur_p"]
    g_tilde_p = div["g_tilde_p"]
    Val_gram_p = div["Val_gram_p"]
    Val_rhs_p = div["Val_rhs_p"]
    sse_0 = div["sse_0"]

    B = Sigma_p.shape[0]
    max_k = Sigma_p.shape[1]
    V_padded_list = []
    for k in k_list:
        Vk = V_grid[k][:, :, :, :3]
        if k < max_k:
            pad = xp.zeros((B, Vk.shape[1], max_k - k, 3), dtype=Vk.dtype)
            Vk = xp.concatenate([Vk, pad], axis=2)
        V_padded_list.append(Vk)
    V_all = xp.concatenate(V_padded_list, axis=1)
    M = V_all.shape[1]
    A = alphas_xp.shape[0]

    V_flat = xp.swapaxes(V_all, 1, 2).reshape(B, max_k, M * 3)
    SV_flat = Sigma_p @ V_flat
    z_var = xp.sum(V_flat * SV_flat, axis=1).reshape(B, M, 3)
    z_scale = xp.sqrt(xp.maximum(z_var, 1e-12))
    V_std = V_all / z_scale[:, :, None, :]
    V_std_flat = xp.swapaxes(V_std, 1, 2).reshape(B, max_k, M * 3)
    V_std_flat_T = xp.swapaxes(V_std_flat, -1, -2)

    g = xp.swapaxes((V_std_flat_T @ xp.swapaxes(g_tilde_p, -1, -2)).reshape(B, M, 3, A), 2, 3)
    V_val_rhs = xp.swapaxes((V_std_flat_T @ xp.swapaxes(Val_rhs_p, -1, -2)).reshape(B, M, 3, A), 2, 3)

    V_std_T = xp.swapaxes(V_std, -1, -2)[:, :, None, :, :]
    Schur_V = xp.moveaxis((Schur_p @ V_std_flat[:, None, :, :]).reshape(B, A, max_k, M, 3), 3, 1)
    S = V_std_T @ Schur_V
    del Schur_V
    eye3 = xp.eye(3, dtype=S.dtype)[None, None, None, :, :]
    S = S + alphas_xp[None, None, :, None, None] * eye3
    Val_V = xp.moveaxis((Val_gram_p @ V_std_flat[:, None, :, :]).reshape(B, A, max_k, M, 3), 3, 1)
    V_val_gram = V_std_T @ Val_V
    del Val_V

    d1_raw = S[:, :, :, 0, 0]
    d1_ok = xp.isfinite(d1_raw) & (d1_raw > 1e-18)
    d1 = xp.where(d1_ok, d1_raw, 1.0)
    L10 = xp.where(d1_ok, S[:, :, :, 1, 0] / d1, 0.0)
    d2_raw = S[:, :, :, 1, 1] - L10 * L10 * d1
    d2_ok = d1_ok & xp.isfinite(d2_raw) & (d2_raw > 1e-18)
    d2 = xp.where(d2_ok, d2_raw, 1.0)
    L20 = xp.where(d1_ok, S[:, :, :, 2, 0] / d1, 0.0)
    L21 = xp.where(d2_ok, (S[:, :, :, 2, 1] - L20 * L10 * d1) / d2, 0.0)
    d3_raw = S[:, :, :, 2, 2] - L20 * L20 * d1 - L21 * L21 * d2
    d3_ok = d2_ok & xp.isfinite(d3_raw) & (d3_raw > 1e-18)
    d3 = xp.where(d3_ok, d3_raw, 1.0)

    z1 = g[:, :, :, 0]
    z2 = g[:, :, :, 1] - L10 * z1
    z3 = g[:, :, :, 2] - L20 * z1 - L21 * z2

    b1_0 = z1 / d1
    b2_1 = z2 / d2
    b2_0 = b1_0 - L10 * b2_1
    b3_2 = z3 / d3
    b3_1 = b2_1 - L21 * b3_2
    b3_0 = b2_0 - (L20 - L21 * L10) * b3_2

    sse_r1 = xp.where(
        d1_ok,
        sse_0[:, None, :] - 2.0 * (b1_0 * V_val_rhs[:, :, :, 0]) + (b1_0 * b1_0 * V_val_gram[:, :, :, 0, 0]),
        xp.nan,
    )

    quad_r2 = (
        b2_0 * b2_0 * V_val_gram[:, :, :, 0, 0]
        + b2_1 * b2_1 * V_val_gram[:, :, :, 1, 1]
        + 2.0 * b2_0 * b2_1 * V_val_gram[:, :, :, 0, 1]
    )
    lin_r2 = b2_0 * V_val_rhs[:, :, :, 0] + b2_1 * V_val_rhs[:, :, :, 1]
    sse_r2 = xp.where(d2_ok, sse_0[:, None, :] - 2.0 * lin_r2 + quad_r2, xp.nan)

    quad_r3 = (
        b3_0 * b3_0 * V_val_gram[:, :, :, 0, 0]
        + b3_1 * b3_1 * V_val_gram[:, :, :, 1, 1]
        + b3_2 * b3_2 * V_val_gram[:, :, :, 2, 2]
        + 2.0
        * (
            b3_0 * b3_1 * V_val_gram[:, :, :, 0, 1]
            + b3_0 * b3_2 * V_val_gram[:, :, :, 0, 2]
            + b3_1 * b3_2 * V_val_gram[:, :, :, 1, 2]
        )
    )
    lin_r3 = b3_0 * V_val_rhs[:, :, :, 0] + b3_1 * V_val_rhs[:, :, :, 1] + b3_2 * V_val_rhs[:, :, :, 2]
    sse_r3 = xp.where(d3_ok, sse_0[:, None, :] - 2.0 * lin_r3 + quad_r3, xp.nan)

    return xp.stack([sse_r1, sse_r2, sse_r3], axis=1)


def _prepare_batched_ridge_inputs(
    xp: Any,
    X_cov_tr: Any,
    X_cov_val: Any,
    Z_tr: Any,
    Z_val: Any,
    y_tr: Any,
    w_tr: Any,
) -> tuple[Any, Any, Any, Any]:
    """Center y and standardize optional spectral block Z for batched weighted ridge."""
    y_mean = xp.sum(w_tr * y_tr[None, :], axis=1)
    y_cent = y_tr[None, :] - y_mean[:, None]

    if Z_tr is None:
        return y_mean, y_cent, X_cov_tr, X_cov_val

    if Z_tr.ndim == 2:
        z_mean = w_tr @ Z_tr
        Z_c = Z_tr[None, :, :] - z_mean[:, None, :]
    else:
        z_mean = (w_tr[:, None, :] @ Z_tr)[:, 0, :]
        Z_c = Z_tr - z_mean[:, None, :]
    z_var = xp.sum(w_tr[:, :, None] * (Z_c * Z_c), axis=1)
    z_scale = xp.sqrt(xp.maximum(z_var, 1e-12))
    Z_std_tr = Z_c / z_scale[:, None, :]
    Z_val_base = Z_val[None, :, :] if Z_val.ndim == 2 else Z_val
    Z_std_val = (Z_val_base - z_mean[:, None, :]) / z_scale[:, None, :]
    X_tr = xp.concatenate([X_cov_tr, Z_std_tr], axis=2)
    X_val = xp.concatenate([X_cov_val, Z_std_val], axis=2)
    return y_mean, y_cent, X_tr, X_val


def batched_ridge_sse_and_preds(
    xp: Any,
    X_cov_tr: Any,
    X_cov_val: Any,
    Z_tr: Any,
    Z_val: Any,
    y_tr: Any,
    y_val: Any,
    w_tr: Any,
    w_val: Any,
    alphas_xp: Any,
) -> tuple[Any, Any]:
    """Batched multi-penalty ridge regression fit and validation prediction."""
    y_mean, y_cent, X_tr, X_val = _prepare_batched_ridge_inputs(
        xp, X_cov_tr, X_cov_val, Z_tr, Z_val, y_tr, w_tr
    )

    if int(X_tr.shape[2]) == 0:
        preds = xp.broadcast_to(
            y_mean[:, None, None], (int(X_tr.shape[0]), int(X_val.shape[1]), int(alphas_xp.shape[0]))
        ).copy()
        sse = xp.sum(w_val[:, :, None] * ((y_val[None, :, None] - preds) ** 2), axis=1)
        return sse, preds

    x_mean = (w_tr[:, None, :] @ X_tr)[:, 0, :]
    X_c = X_tr - x_mean[:, None, :]
    Xw = X_c * w_tr[:, :, None]
    Xw_T = xp.swapaxes(Xw, -1, -2)
    gram = Xw_T @ X_c
    gram = 0.5 * (gram + xp.swapaxes(gram, -1, -2))
    rhs = (Xw_T @ y_cent[:, :, None])[:, :, 0]
    B_n, p_n, _ = gram.shape
    A_n = int(alphas_xp.shape[0])
    coefs: Any | None = None
    if xp is not np and p_n >= 12 and A_n > 0 and _cupyx_supports_batched_triangular():
        cpx_linalg = _get_cupyx_linalg()
        if cpx_linalg is not None:
            try:
                G_reg = gram[:, None, :, :] + alphas_xp[None, :, None, None] * xp.eye(p_n, dtype=xp.float64)[None, None, :, :]
                L_chol = xp.linalg.cholesky(G_reg)
                if bool(
                    xp.all(xp.isfinite(L_chol))
                    & xp.all(xp.diagonal(L_chol, axis1=-2, axis2=-1) > 0.0)
                ):
                    L_flat = L_chol.reshape(B_n * A_n, p_n, p_n)
                    rhs_rep = xp.broadcast_to(rhs[:, None, :, None], (B_n, A_n, p_n, 1)).reshape(B_n * A_n, p_n, 1)
                    u = cpx_linalg.solve_triangular(L_flat, rhs_rep, lower=True, check_finite=False)
                    coefs = cpx_linalg.solve_triangular(
                        L_flat, u, lower=True, trans="T", check_finite=False
                    ).reshape(B_n, A_n, p_n)
            except Exception:
                coefs = None
    if coefs is None:
        evals, evecs = xp.linalg.eigh(gram)
        evals = xp.maximum(evals, 0.0)
        proj_rhs = (xp.swapaxes(evecs, -1, -2) @ rhs[:, :, None])[:, :, 0]
        denom = evals[:, None, :] + alphas_xp[None, :, None]
        scaled = xp.where(denom > 1e-15, proj_rhs[:, None, :] / xp.maximum(denom, 1e-15), 0.0)
        coefs = scaled @ xp.swapaxes(evecs, -1, -2)
    intercepts = y_mean[:, None] - (coefs @ x_mean[:, :, None])[:, :, 0]
    preds = intercepts[:, None, :] + (X_val @ xp.swapaxes(coefs, -1, -2))
    sse = xp.sum(w_val[:, :, None] * ((y_val[None, :, None] - preds) ** 2), axis=1)
    return sse, preds


def batched_ridge_preds_single_alpha(
    xp: Any,
    X_cov_tr: Any,
    X_cov_val: Any,
    Z_tr: Any,
    Z_val: Any,
    y_tr: Any,
    w_tr: Any,
    selected_alphas: Any,
) -> Any:
    """Batched weighted ridge predictions solving only each draw's single selected alpha."""
    y_mean, y_cent, X_tr, X_val = _prepare_batched_ridge_inputs(
        xp, X_cov_tr, X_cov_val, Z_tr, Z_val, y_tr, w_tr
    )
    B_n, _, p_n = X_tr.shape
    if int(p_n) == 0:
        return xp.broadcast_to(y_mean[:, None], (int(B_n), int(X_val.shape[1]))).copy()

    x_mean = (w_tr[:, None, :] @ X_tr)[:, 0, :]
    X_c = X_tr - x_mean[:, None, :]
    Xw = X_c * w_tr[:, :, None]
    Xw_T = xp.swapaxes(Xw, -1, -2)
    gram = Xw_T @ X_c
    gram = 0.5 * (gram + xp.swapaxes(gram, -1, -2))
    rhs = (Xw_T @ y_cent[:, :, None])[:, :, 0]

    coefs: Any | None = None
    try:
        G_reg = gram + selected_alphas[:, None, None] * xp.eye(int(p_n), dtype=xp.float64)[None, :, :]
        L_chol = xp.linalg.cholesky(G_reg)
        if bool(
            xp.all(xp.isfinite(L_chol))
            & xp.all(xp.diagonal(L_chol, axis1=-2, axis2=-1) > 0.0)
        ):
            if xp is not np and _cupyx_supports_batched_triangular():
                cpx_linalg = _get_cupyx_linalg()
                if cpx_linalg is not None:
                    u = cpx_linalg.solve_triangular(L_chol, rhs[:, :, None], lower=True, check_finite=False)
                    coefs = cpx_linalg.solve_triangular(
                        L_chol, u, lower=True, trans="T", check_finite=False
                    )[:, :, 0]
            if coefs is None:
                u = xp.linalg.solve(L_chol, rhs[:, :, None])
                coefs = xp.linalg.solve(xp.swapaxes(L_chol, -1, -2), u)[:, :, 0]
    except Exception:
        coefs = None

    if coefs is None:
        evals, evecs = xp.linalg.eigh(gram)
        evals = xp.maximum(evals, 0.0)
        proj_rhs = (xp.swapaxes(evecs, -1, -2) @ rhs[:, :, None])[:, :, 0]
        denom = evals + selected_alphas[:, None]
        scaled = xp.where(denom > 1e-15, proj_rhs / xp.maximum(denom, 1e-15), 0.0)
        coefs = (evecs @ scaled[:, :, None])[:, :, 0]

    intercepts = y_mean - xp.sum(coefs * x_mean, axis=1)
    return intercepts[:, None] + (X_val @ coefs[:, :, None])[:, :, 0]


def batched_pls_sse_and_preds(xp: Any, div: dict[str, Any], max_comp: int) -> tuple[Any, Any]:
    """Dual Gram NIPALS for batched multi-component weighted PLS regression."""
    w_tr = div["w_tr"]
    w_val = div["w_val"]
    y_tr = div["y_tr"]
    y_val = div["y_val"]
    X_cov_tr = div["X_cov_tr"]
    X_cov_val = div["X_cov_val"]
    scale1_sq = div["scale1_sq"]
    scale2_sq = div["scale2_sq"]

    B = X_cov_tr.shape[0]
    n_val = X_cov_val.shape[1]

    cov_mean = (w_tr[:, None, :] @ X_cov_tr)[:, 0, :]
    C_tr = X_cov_tr - cov_mean[:, None, :]
    C_val = X_cov_val - cov_mean[:, None, :]
    C_tr_T = xp.swapaxes(C_tr, -1, -2)

    K_p1_tr = div["K_p1_tr"]
    K_p1_vt = div["K_p1_vt"]
    K_p2_tr = div["K_p2_tr"]
    K_p2_vt = div["K_p2_vt"]

    # Combine scaled spectral Gram slices before centering once to halve (B, n_tr, n_tr) buffers.
    k_mu_tr = scale1_sq[:, None] * (w_tr @ K_p1_tr) + scale2_sq[:, None] * (w_tr @ K_p2_tr)
    mu_sq = xp.sum(w_tr * k_mu_tr, axis=1)
    k_mu_vt = scale1_sq[:, None] * (w_tr @ K_p1_vt.T) + scale2_sq[:, None] * (w_tr @ K_p2_vt.T)

    K_tr = scale1_sq[:, None, None] * K_p1_tr[None, :, :]
    K_tr += scale2_sq[:, None, None] * K_p2_tr[None, :, :]
    K_tr -= k_mu_tr[:, :, None]
    K_tr -= k_mu_tr[:, None, :]
    K_tr += mu_sq[:, None, None]
    K_tr += C_tr @ C_tr_T

    K_vt = scale1_sq[:, None, None] * K_p1_vt[None, :, :]
    K_vt += scale2_sq[:, None, None] * K_p2_vt[None, :, :]
    K_vt -= k_mu_vt[:, :, None]
    K_vt -= k_mu_tr[:, None, :]
    K_vt += mu_sq[:, None, None]
    K_vt += C_val @ C_tr_T
    del C_tr, C_val, C_tr_T

    # Dual Gram deflation leaves O(n_tr * eps * tr_w(K_tr_0)) floating-point residual in K_tr after
    # exhausting rank(X), so guard t_sq against both the primal 1e-12 floor and Gram roundoff.
    n_tr = int(X_cov_tr.shape[1])
    eps_f = float(np.finfo(float).eps)
    k0_w_trace = xp.sum(w_tr * xp.maximum(xp.diagonal(K_tr, axis1=-2, axis2=-1), 0.0), axis=1)
    t_sq_floor = xp.maximum(1e-12, 1e2 * float(max(n_tr, 1)) * eps_f * k0_w_trace)

    y_mean = xp.sum(w_tr * y_tr[None, :], axis=1)
    yk = y_tr[None, :] - y_mean[:, None]
    accum = xp.broadcast_to(y_mean[:, None], (B, n_val)).copy()
    preds_all = xp.zeros((B, n_val, max_comp), dtype=xp.float64)

    active = xp.ones(B, dtype=bool)
    for c in range(max_comp):
        wy = w_tr * yk
        k_wy = (K_tr @ wy[:, :, None])[:, :, 0]
        w_norm_sq = xp.sum(wy * k_wy, axis=1)
        w_norm = xp.sqrt(xp.maximum(w_norm_sq, 0.0))
        w_ok = active & xp.isfinite(w_norm_sq) & (w_norm_sq > 0.0) & xp.isfinite(w_norm) & (w_norm > 1e-12)
        w_norm_safe = xp.where(w_ok, w_norm, 1.0)
        tk = xp.where(w_ok[:, None], k_wy / w_norm_safe[:, None], 0.0)
        t_val = xp.where(w_ok[:, None], (K_vt @ wy[:, :, None])[:, :, 0] / w_norm_safe[:, None], 0.0)
        t_sq_raw = xp.sum(w_tr * tk * tk, axis=1)
        step_ok = w_ok & xp.isfinite(t_sq_raw) & (t_sq_raw > t_sq_floor)
        active = step_ok
        t_sq_safe = xp.where(step_ok, t_sq_raw, 1.0)
        qk = xp.where(step_ok, xp.sum(w_tr * tk * yk, axis=1) / t_sq_safe, 0.0)
        t_val_act = xp.where(step_ok[:, None], t_val, 0.0)
        tk_act = xp.where(step_ok[:, None], tk, 0.0)
        accum += qk[:, None] * t_val_act
        preds_all[:, :, c] = accum
        if c < max_comp - 1:
            yk -= qk[:, None] * tk_act
            u = xp.where(step_ok[:, None], (w_tr * tk_act) / t_sq_safe[:, None], 0.0)
            kt_u = (K_tr @ u[:, :, None])[:, :, 0]
            u_kt_u = xp.sum(u * kt_u, axis=1)
            K_vt -= t_val_act[:, :, None] * kt_u[:, None, :]
            v_def = kt_u - (0.5 * u_kt_u)[:, None] * tk_act
            K_tr -= tk_act[:, :, None] * v_def[:, None, :]
            K_tr -= v_def[:, :, None] * tk_act[:, None, :]

    sse = xp.sum(w_val[:, :, None] * ((y_val[None, :, None] - preds_all) ** 2), axis=1)
    return sse, preds_all
