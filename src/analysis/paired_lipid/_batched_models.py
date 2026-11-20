"""Batched closed-form models: division prep, Schur PCA/paired ridge, multi-alpha ridge, dual PLS."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.analysis.paired_lipid._eigensolvers import _fast_top1_eigvalsh, _top_k_eigh
from src.analysis.paired_lipid.cohort import PairedLipidConfig


def _prep_covariates_batch(
    xp: Any,
    dconst: dict[str, Any],
    M_tr: Any,
    M_val: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    """Compute normalized participant weights and standardized training/validation covariates."""
    w_tr_sum = xp.sum(M_tr, axis=1, keepdims=True)
    w_tr = M_tr / xp.maximum(w_tr_sum, 1e-12)
    w_val = M_val

    X_raw_tr = dconst["X_raw_tr"]
    X_raw_val = dconst["X_raw_val"]
    c_mean = w_tr @ X_raw_tr
    C_tr = X_raw_tr[None, :, :] - c_mean[:, None, :]
    c_var = xp.sum(w_tr[:, :, None] * (C_tr * C_tr), axis=1)
    valid_c = c_var > 1e-12
    c_scale = xp.where(valid_c, xp.sqrt(xp.maximum(c_var, 1e-12)), 1.0)
    X_cov_tr = xp.where(valid_c[:, None, :], C_tr / c_scale[:, None, :], 0.0)
    X_cov_val = xp.where(
        valid_c[:, None, :],
        (X_raw_val[None, :, :] - c_mean[:, None, :]) / c_scale[:, None, :],
        0.0,
    )
    return w_tr_sum, w_tr, w_val, X_cov_tr, X_cov_val


def _prep_mfa_pca_batch(
    xp: Any,
    gcache: dict[str, Any],
    dconst: dict[str, Any],
    M_tr: Any,
    w_tr_sum: Any,
    cfg: PairedLipidConfig,
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

    # Assemble (B, n_s, n_s) weighted Grams in-place and reuse gram2 for symmetrization so at
    # most two (B, n_s, n_s) buffers stay live simultaneously.
    k1_mu_div = w_s @ K1_div
    mu1_sq = xp.sum(w_s * k1_mu_div, axis=1)
    gram = K1_div[None, :, :] - k1_mu_div[:, :, None]
    gram -= k1_mu_div[:, None, :]
    gram += mu1_sq[:, None, None]
    gram *= sqrt_w[:, :, None]
    gram *= sqrt_w[:, None, :]
    min_eig_clamp = min(1e-12, float(cfg.eigenvalue_tolerance))
    eig1_1300 = _fast_top1_eigvalsh(xp, gram, sqrt_w, dconst["Q0_1"])
    scale1_sq = 1.0 / xp.maximum(eig1_1300, min_eig_clamp)
    gram *= scale1_sq[:, None, None]

    k2_mu_div = w_s @ K2_div
    mu2_sq = xp.sum(w_s * k2_mu_div, axis=1)
    gram2 = K2_div[None, :, :] - k2_mu_div[:, :, None]
    gram2 -= k2_mu_div[:, None, :]
    gram2 += mu2_sq[:, None, None]
    gram2 *= sqrt_w[:, :, None]
    gram2 *= sqrt_w[:, None, :]
    eig1_2850 = _fast_top1_eigvalsh(xp, gram2, sqrt_w, dconst["Q0_2"])
    scale2_sq = 1.0 / xp.maximum(eig1_2850, min_eig_clamp)
    gram2 *= scale2_sq[:, None, None]

    gram += gram2
    xp.copyto(gram2, xp.swapaxes(gram, -1, -2))
    gram += gram2
    del gram2
    gram *= 0.5

    max_k = max(cfg.pca_dimensions)
    eigvals_top, eigvecs_top = _top_k_eigh(
        xp, gram, max_k, sqrt_w, dconst.get("Q0_comb"), is_symmetric=True
    )
    del gram
    s_kept = xp.sqrt(eigvals_top)

    p_total = int(dconst.get("p_total", gcache.get("p_total", sqrt_w.shape[1])))
    n_act = xp.sum(sqrt_w > 0, axis=1)
    max_dim = xp.maximum(n_act, float(p_total))
    # Squared-Gram eigenvalues have an O(max_dim * eps * s0^2) roundoff floor, so singular
    # values at or below sqrt(max_dim * eps) * s0 must route to the primal SVD fallback.
    gram_s_tol = xp.sqrt(max_dim * float(np.finfo(float).eps)) * xp.maximum(s_kept[:, 0], 1.0)
    rank_thresh = xp.maximum(gram_s_tol, float(cfg.rank_tolerance))
    rank_ok = (
        xp.isfinite(s_kept[:, -1])
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

    return {
        "w_s": w_s,
        "alpha": alpha,
        "alpha_sum": alpha_sum,
        "scale1_sq": scale1_sq,
        "scale2_sq": scale2_sq,
        "part_pca_max": part_pca_max,
        "rank_ok": rank_ok,
    }


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
) -> dict[str, Any]:
    """Compute batched MFA-PCA coordinates, paired moments, and max_k Schur sufficient statistics."""
    tr_p = dconst["tr_p"]
    val_p = dconst["val_p"]

    w_tr_sum, w_tr, w_val, X_cov_tr, X_cov_val = _prep_covariates_batch(xp, dconst, M_tr, M_val)
    mfa = _prep_mfa_pca_batch(xp, gcache, dconst, M_tr, w_tr_sum, cfg)
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

    cpx_linalg: Any = None
    if xp is not np:
        try:
            import cupyx.scipy.linalg as cpx_linalg  # type: ignore[import-not-found]
        except Exception:
            cpx_linalg = None

    sse_by_k = []
    for k in k_list:
        eye_k = xp.eye(k, dtype=xp.float64)[None, None, :, :]
        Sk = Schur_std[:, :, :k, :k] + alphas_xp[None, :, None, None] * eye_k
        gk = g_std[:, :, :k]
        L = xp.linalg.cholesky(Sk)
        if cpx_linalg is not None:
            L_flat = L.reshape(B * A, k, k)
            u = cpx_linalg.solve_triangular(L_flat, gk.reshape(B * A, k, 1), lower=True, check_finite=False)
            beta = cpx_linalg.solve_triangular(
                L_flat, u, lower=True, trans="T", check_finite=False
            ).reshape(B, A, k)
        else:
            u = xp.linalg.solve(L, gk[:, :, :, None])
            beta = xp.linalg.solve(xp.swapaxes(L, -1, -2), u)[:, :, :, 0]
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

    Schur_V = xp.moveaxis((Schur_p @ V_std_flat[:, None, :, :]).reshape(B, A, max_k, M, 3), 3, 1)
    Val_V = xp.moveaxis((Val_gram_p @ V_std_flat[:, None, :, :]).reshape(B, A, max_k, M, 3), 3, 1)
    V_std_T = xp.swapaxes(V_std, -1, -2)[:, :, None, :, :]
    S = V_std_T @ Schur_V
    eye3 = xp.eye(3, dtype=S.dtype)[None, None, None, :, :]
    S = S + alphas_xp[None, None, :, None, None] * eye3
    V_val_gram = V_std_T @ Val_V

    d1 = S[:, :, :, 0, 0]
    L10 = S[:, :, :, 1, 0] / d1
    d2 = S[:, :, :, 1, 1] - L10 * L10 * d1
    L20 = S[:, :, :, 2, 0] / d1
    L21 = (S[:, :, :, 2, 1] - L20 * L10 * d1) / d2
    d3 = S[:, :, :, 2, 2] - L20 * L20 * d1 - L21 * L21 * d2

    z1 = g[:, :, :, 0]
    z2 = g[:, :, :, 1] - L10 * z1
    z3 = g[:, :, :, 2] - L20 * z1 - L21 * z2

    b1_0 = z1 / d1
    b2_1 = z2 / d2
    b2_0 = b1_0 - L10 * b2_1
    b3_2 = z3 / d3
    b3_1 = b2_1 - L21 * b3_2
    b3_0 = b2_0 - (L20 - L21 * L10) * b3_2

    sse_r1 = sse_0[:, None, :] - 2.0 * (b1_0 * V_val_rhs[:, :, :, 0]) + (b1_0 * b1_0 * V_val_gram[:, :, :, 0, 0])

    quad_r2 = (
        b2_0 * b2_0 * V_val_gram[:, :, :, 0, 0]
        + b2_1 * b2_1 * V_val_gram[:, :, :, 1, 1]
        + 2.0 * b2_0 * b2_1 * V_val_gram[:, :, :, 0, 1]
    )
    lin_r2 = b2_0 * V_val_rhs[:, :, :, 0] + b2_1 * V_val_rhs[:, :, :, 1]
    sse_r2 = sse_0[:, None, :] - 2.0 * lin_r2 + quad_r2

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
    sse_r3 = sse_0[:, None, :] - 2.0 * lin_r3 + quad_r3

    return xp.stack([sse_r1, sse_r2, sse_r3], axis=1)


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
    y_mean = xp.sum(w_tr * y_tr[None, :], axis=1)
    y_cent = y_tr[None, :] - y_mean[:, None]

    if Z_tr is None:
        X_tr = X_cov_tr
        X_val = X_cov_val
    else:
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

    x_mean = (w_tr[:, None, :] @ X_tr)[:, 0, :]
    X_c = X_tr - x_mean[:, None, :]
    Xw = X_c * w_tr[:, :, None]
    Xw_T = xp.swapaxes(Xw, -1, -2)
    gram = Xw_T @ X_c
    gram = 0.5 * (gram + xp.swapaxes(gram, -1, -2))
    rhs = (Xw_T @ y_cent[:, :, None])[:, :, 0]
    evals, evecs = xp.linalg.eigh(gram)
    evals = xp.maximum(evals, 0.0)
    proj_rhs = (xp.swapaxes(evecs, -1, -2) @ rhs[:, :, None])[:, :, 0]
    scaled = proj_rhs[:, None, :] / (evals[:, None, :] + alphas_xp[None, :, None])
    coefs = scaled @ xp.swapaxes(evecs, -1, -2)
    intercepts = y_mean[:, None] - (coefs @ x_mean[:, :, None])[:, :, 0]
    preds = intercepts[:, None, :] + (X_val @ xp.swapaxes(coefs, -1, -2))
    sse = xp.sum(w_val[:, :, None] * ((y_val[None, :, None] - preds) ** 2), axis=1)
    return sse, preds


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

    y_mean = xp.sum(w_tr * y_tr[None, :], axis=1)
    yk = y_tr[None, :] - y_mean[:, None]
    accum = xp.broadcast_to(y_mean[:, None], (B, n_val)).copy()
    preds_all = xp.zeros((B, n_val, max_comp), dtype=xp.float64)

    for c in range(max_comp):
        wy = w_tr * yk
        k_wy = (K_tr @ wy[:, :, None])[:, :, 0]
        w_norm = xp.sqrt(xp.maximum(xp.sum(wy * k_wy, axis=1), 1e-24))
        tk = k_wy / w_norm[:, None]
        t_val = (K_vt @ wy[:, :, None])[:, :, 0] / w_norm[:, None]
        t_sq = xp.maximum(xp.sum(w_tr * tk * tk, axis=1), 1e-24)
        qk = xp.sum(w_tr * tk * yk, axis=1) / t_sq
        accum += qk[:, None] * t_val
        preds_all[:, :, c] = accum
        if c < max_comp - 1:
            yk -= qk[:, None] * tk
            u = (w_tr * tk) / t_sq[:, None]
            kt_u = (K_tr @ u[:, :, None])[:, :, 0]
            u_kt_u = xp.sum(u * kt_u, axis=1)
            K_vt -= t_val[:, :, None] * kt_u[:, None, :]
            v_def = kt_u - (0.5 * u_kt_u)[:, None] * tk
            K_tr -= tk[:, :, None] * v_def[:, None, :]
            K_tr -= v_def[:, :, None] * tk[:, None, :]

    sse = xp.sum(w_val[:, :, None] * ((y_val[None, :, None] - preds_all) ** 2), axis=1)
    return sse, preds_all
