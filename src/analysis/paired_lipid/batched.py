"""Batched closed-form evaluation of multiplicity bootstrap draws on CPU (NumPy) or GPU (CuPy)."""

from __future__ import annotations
from typing import Any, Sequence
import numpy as np
from scipy import linalg

from src.analysis.paired_lipid.cohort import (
    CohortDataset,
    PairedLipidConfig,
    extract_sample_band_matrix,
)
from src.analysis.paired_lipid.models import (
    CovariatePreprocessor,
)


def _get_gpu_xp() -> Any | None:
    """Return the CuPy module when an NVIDIA CUDA device is available, else None."""
    try:
        import cupy as cp  # type: ignore[import-not-found]

        if cp.cuda.runtime.getDeviceCount() > 0:
            return cp
    except Exception:
        pass
    return None


def _to_np(xp: Any, arr: Any) -> np.ndarray:
    """Convert a NumPy or CuPy array to a host NumPy float64 array."""
    if xp is not np and hasattr(xp, "asnumpy"):
        return np.asarray(xp.asnumpy(arr), dtype=np.float64)
    return np.asarray(arr, dtype=np.float64)


# Tie rules (research_design.md): fewer retained directions, fewer PCA dimensions, then stronger
# regularization. MSEs arrive rounded to 12 decimals, so exact ties are well defined.
def _argmin_last(values: np.ndarray) -> np.ndarray:
    """Argmin over the last axis; an exact tie goes to the last index (the larger ridge alpha)."""
    return values.shape[-1] - 1 - np.argmin(values[..., ::-1], axis=-1)


def _select_pca_setting(pca_r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best (k index, alpha index) per draw from MSEs of shape (B, n_k, n_alphas)."""
    n_alphas = pca_r.shape[-1]
    flat = np.argmin(pca_r[:, :, ::-1].reshape(pca_r.shape[0], -1), axis=1)
    return flat // n_alphas, (n_alphas - 1) - (flat % n_alphas)


def _select_paired_setting(pair_r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Best (r, k, lambda, alpha) indices per draw from MSEs of shape (B, n_r, n_k, n_lam, n_alphas)."""
    n_b, _, n_k, n_lam, n_alphas = pair_r.shape
    rem = np.argmin(pair_r[:, :, :, ::-1, ::-1].reshape(n_b, -1), axis=1)
    a_idx = (n_alphas - 1) - rem % n_alphas
    rem = rem // n_alphas
    lam_idx = (n_lam - 1) - rem % n_lam
    rem = rem // n_lam
    return rem // n_k, rem % n_k, lam_idx, a_idx


def _get_dataset_gram_cache(dataset: CohortDataset, cfg: PairedLipidConfig) -> dict[str, Any]:
    """Memoize sample-level and participant-level Gram matrices and raw covariates on dataset."""
    key = f"gram_cache__{cfg.covariate_preset}"
    cached = dataset._cache.get(key)
    if cached is not None:
        return cached

    n_p = dataset.n_participants
    n_s = dataset.n_samples
    X1 = np.asarray(dataset.sample_1300, dtype=np.float64)
    X2 = np.asarray(dataset.sample_2850, dtype=np.float64)
    K1 = X1 @ X1.T
    K2 = X2 @ X2.T

    A_part = np.zeros((n_p, n_s), dtype=np.float64)
    for s_idx, p_idx in enumerate(dataset.sample_participant_idx):
        A_part[p_idx, s_idx] = 1.0 / float(dataset.participant_sample_counts[p_idx])

    K_part_1 = A_part @ K1 @ A_part.T
    K_part_2 = A_part @ K2 @ A_part.T

    sample_band_mat, _ = extract_sample_band_matrix(dataset)
    part_band_mat = A_part @ sample_band_mat

    CovariatePreprocessor.fit(
        dataset.covariate_frame,
        np.arange(n_p),
        np.ones(n_p),
        preset=cfg.covariate_preset,
        raw_cache=dataset._cache,
    )
    raw_cov, _ = dataset._cache[f"raw_cov_matrix__{cfg.covariate_preset}"]
    raw_cov_arr = np.asarray(raw_cov, dtype=np.float64)

    out = {
        "K1": K1,
        "K2": K2,
        "A_part": A_part,
        "K_part_1": K_part_1,
        "K_part_2": K_part_2,
        "sample_band_mat": sample_band_mat,
        "part_band_mat": part_band_mat,
        "raw_cov": raw_cov_arr,
        "has_missing_cov": bool(not np.all(np.isfinite(raw_cov_arr))),
    }
    dataset._cache[key] = out
    return out


def _fast_top1_eigvalsh(xp: Any, G_batch: Any, sw: Any, Q0: Any) -> Any:
    """QR-free 2-step subspace Rayleigh-Ritz for the leading eigenvalue of a PSD batch."""
    Y = G_batch @ (sw[:, :, None] * Q0[None, :, :])
    Y = G_batch @ Y
    YtY = xp.swapaxes(Y, -1, -2) @ Y
    YtY = 0.5 * (YtY + xp.swapaxes(YtY, -1, -2))
    d, V = xp.linalg.eigh(YtY)
    d_max = xp.maximum(d[:, -1:], 1e-24)
    valid = d > (1e-18 * d_max)
    inv_s = xp.where(valid, 1.0 / xp.sqrt(xp.maximum(d, 1e-30)), 0.0)
    Q = Y @ (V * inv_s[:, None, :])
    GQ = G_batch @ Q
    H = xp.swapaxes(Q, -1, -2) @ GQ
    H = 0.5 * (H + xp.swapaxes(H, -1, -2))
    return xp.linalg.eigvalsh(H)[:, -1]


def _top_k_eigh(xp: Any, G_batch: Any, max_k: int, sw: Any) -> tuple[Any, Any]:
    """Exact fp64 top-max_k eigenpairs: active-sample LAPACK dsyevr on CPU, batched eigh on GPU."""
    G_sym = G_batch + xp.swapaxes(G_batch, -1, -2)
    G_sym *= 0.5  # in place: same result as 0.5 * (G + G^T), one (B, n_s, n_s) temporary fewer
    if xp is np:
        B, n_s, _ = G_sym.shape
        ev_out = np.zeros((B, max_k), dtype=np.float64)
        u_out = np.zeros((B, n_s, max_k), dtype=np.float64)
        for b in range(B):
            act = np.flatnonzero(sw[b] > 0)
            n_act = act.size
            k_b = min(max_k, n_act)
            if k_b <= 0:
                continue
            G_sub = G_sym[b][np.ix_(act, act)]
            ev_b, u_b = linalg.eigh(
                G_sub,
                subset_by_index=[n_act - k_b, n_act - 1],
                driver="evr",
                check_finite=False,
            )
            ev_out[b, :k_b] = np.maximum(ev_b[::-1], 0.0)
            u_out[b, act, :k_b] = u_b[:, ::-1]
        return ev_out, u_out

    return _exact_top_k_eigh_active(xp, G_sym, max_k, sw)


def _exact_top_k_eigh_active(xp: Any, G_sym: Any, max_k: int, sw: Any) -> tuple[Any, Any]:
    """Exact fp64 top-max_k eigenpairs, solved only over each draw's active samples.

    A sample of a participant with bootstrap multiplicity 0 has weight 0, so its gram row and
    column are zero (about 37% of samples per draw). The top eigenpairs of the active block equal
    those of the full matrix; eigh cost grows with the cube of the size, so solving the smaller
    block is the speed-up. Draws are grouped by active count rounded up to a multiple of 8;
    the extra rows in a group are zero rows and add only zero eigenvalues.
    """
    B, n_s, _ = G_sym.shape
    active = sw > 0
    n_act = _to_np(xp, xp.sum(active, axis=1)).astype(int)
    # Unique sort keys: active samples first, each group in original sample order.
    pos = xp.arange(n_s)[None, :]
    order = xp.argsort(xp.where(active, pos, n_s + pos), axis=1)
    sizes = np.minimum(-(-n_act // 8) * 8, n_s)

    ev_out = xp.zeros((B, max_k), dtype=xp.float64)
    u_out = xp.zeros((B, n_s, max_k), dtype=xp.float64)
    for m in np.unique(sizes):
        b_idx = xp.asarray(np.flatnonzero(sizes == m))
        idx = order[b_idx, :m]
        G_sub = xp.take_along_axis(G_sym[b_idx], idx[:, :, None], axis=1)
        G_sub = xp.take_along_axis(G_sub, idx[:, None, :], axis=2)
        ev, u = xp.linalg.eigh(G_sub)
        k = min(max_k, int(m))
        ev_out[b_idx, :k] = xp.maximum(ev[:, -k:][:, ::-1], 0.0)
        u_grp = xp.zeros((b_idx.size, n_s, max_k), dtype=xp.float64)
        u_grp[xp.arange(b_idx.size)[:, None], idx, :k] = u[:, :, -k:][:, :, ::-1]
        u_out[b_idx] = u_grp
    return ev_out, u_out


def _precompute_division_constants(
    xp: Any,
    gcache: dict[str, Any],
    dataset: CohortDataset,
    tr_p: np.ndarray,
    val_p: np.ndarray,
    cfg: PairedLipidConfig,
) -> dict[str, Any]:
    """Precompute division-specific Gram slices and warm-start bases outside the bootstrap loop."""
    n_tr = tr_p.size
    in_tr_mask = np.isin(dataset.sample_participant_idx, tr_p)
    s_div = np.flatnonzero(in_tr_mask)
    p_to_rel = np.full(dataset.n_participants, -1, dtype=int)
    p_to_rel[tr_p] = np.arange(n_tr, dtype=int)
    s_rel = p_to_rel[dataset.sample_participant_idx[s_div]]
    m_i_div = xp.asarray(
        dataset.participant_sample_counts[dataset.sample_participant_idx[s_div]],
        dtype=xp.float64,
    )

    raw_cov = xp.asarray(gcache["raw_cov"], dtype=xp.float64)
    X_raw_tr = raw_cov[tr_p]
    X_raw_val = raw_cov[val_p]
    ptp_tr = xp.max(X_raw_tr, axis=0) - xp.min(X_raw_tr, axis=0)
    keep_col = ptp_tr > 1e-12
    X_raw_tr = X_raw_tr[:, keep_col]
    X_raw_val = X_raw_val[:, keep_col]

    A_part_np = gcache["A_part"]
    Kp1_all_div = xp.asarray(A_part_np @ gcache["K1"][:, s_div], dtype=xp.float64)
    K1_div = xp.asarray(gcache["K1"][np.ix_(s_div, s_div)], dtype=xp.float64)
    Kp2_all_div = xp.asarray(A_part_np @ gcache["K2"][:, s_div], dtype=xp.float64)
    K2_div = xp.asarray(gcache["K2"][np.ix_(s_div, s_div)], dtype=xp.float64)

    w_s0 = (1.0 / m_i_div) / xp.sum(1.0 / m_i_div)
    sw0 = xp.sqrt(w_s0 * (s_div.size / max(s_div.size - 1.0, 1.0)))
    k1_0 = w_s0 @ K1_div
    m1_0 = xp.sum(w_s0 * k1_0)
    G1_0 = (K1_div - k1_0[:, None] - k1_0[None, :] + m1_0) * (sw0[:, None] * sw0[None, :])
    G1_0 = 0.5 * (G1_0 + G1_0.T)
    _, U1_0 = xp.linalg.eigh(G1_0)

    k2_0 = w_s0 @ K2_div
    m2_0 = xp.sum(w_s0 * k2_0)
    G2_0 = (K2_div - k2_0[:, None] - k2_0[None, :] + m2_0) * (sw0[:, None] * sw0[None, :])
    G2_0 = 0.5 * (G2_0 + G2_0.T)
    _, U2_0 = xp.linalg.eigh(G2_0)

    q_dim_1 = min(8, s_div.size)
    Q0_1 = U1_0[:, -q_dim_1:] / xp.maximum(sw0[:, None], 1e-12)
    Q0_2 = U2_0[:, -q_dim_1:] / xp.maximum(sw0[:, None], 1e-12)

    paired_in_tr = np.isin(dataset.paired_participant_indices, tr_p)
    p_pair_tr = dataset.paired_participant_indices[paired_in_tr]
    idx_a_tr = dataset.pair_sample_a_indices[paired_in_tr]
    idx_b_tr = dataset.pair_sample_b_indices[paired_in_tr]
    pair_rel = p_to_rel[p_pair_tr]
    DK1_pair_div = xp.asarray(
        gcache["K1"][idx_a_tr][:, s_div] - gcache["K1"][idx_b_tr][:, s_div],
        dtype=xp.float64,
    )
    DK2_pair_div = xp.asarray(
        gcache["K2"][idx_a_tr][:, s_div] - gcache["K2"][idx_b_tr][:, s_div],
        dtype=xp.float64,
    )

    return {
        "tr_p": tr_p,
        "val_p": val_p,
        "s_div": s_div,
        "s_rel": s_rel,
        "m_i_div": m_i_div,
        "X_raw_tr": X_raw_tr,
        "X_raw_val": X_raw_val,
        "Kp1_all_div": Kp1_all_div,
        "K1_div": K1_div,
        "Kp2_all_div": Kp2_all_div,
        "K2_div": K2_div,
        "DK1_pair_div": DK1_pair_div,
        "DK2_pair_div": DK2_pair_div,
        "Q0_1": Q0_1,
        "Q0_2": Q0_2,
        "p_pair_tr": p_pair_tr,
        "idx_a_tr": idx_a_tr,
        "idx_b_tr": idx_b_tr,
        "pair_rel": pair_rel,
        "K_p1_tr": xp.asarray(gcache["K_part_1"][np.ix_(tr_p, tr_p)], dtype=xp.float64),
        "K_p1_vt": xp.asarray(gcache["K_part_1"][np.ix_(val_p, tr_p)], dtype=xp.float64),
        "K_p2_tr": xp.asarray(gcache["K_part_2"][np.ix_(tr_p, tr_p)], dtype=xp.float64),
        "K_p2_vt": xp.asarray(gcache["K_part_2"][np.ix_(val_p, tr_p)], dtype=xp.float64),
    }


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
    """Compute batched MFA-PCA coordinates, paired moments, and 15x15 Schur sufficient statistics."""
    tr_p = dconst["tr_p"]
    val_p = dconst["val_p"]

    w_tr_sum = xp.sum(M_tr, axis=1, keepdims=True)
    w_tr = M_tr / xp.maximum(w_tr_sum, 1e-12)
    w_val = M_val

    # 1. Covariates
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

    # 2. Spectral MFA-PCA in sample Gram space
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
    DK1_pair_div = dconst["DK1_pair_div"]
    DK2_pair_div = dconst["DK2_pair_div"]

    # The (B, n_s, n_s) arrays below dominate GPU memory. In-place updates keep the operation
    # order of the plain expressions (bitwise-identical results) with fewer live temporaries.
    sw_outer = sqrt_w[:, :, None] * sqrt_w[:, None, :]

    def _weighted_centered_gram(K_div: Any, k_mu: Any, mu_sq: Any) -> Any:
        g = K_div[None, :, :] - k_mu[:, :, None]
        g -= k_mu[:, None, :]
        g += mu_sq[:, None, None]
        g *= sw_outer
        return g

    k1_mu_div = w_s @ K1_div
    mu1_sq = xp.sum(w_s * k1_mu_div, axis=1)
    gram = _weighted_centered_gram(K1_div, k1_mu_div, mu1_sq)
    eig1_1300 = _fast_top1_eigvalsh(xp, gram, sqrt_w, dconst["Q0_1"])
    scale1_sq = 1.0 / xp.maximum(eig1_1300, 1e-12)

    k2_mu_div = w_s @ K2_div
    mu2_sq = xp.sum(w_s * k2_mu_div, axis=1)
    gram2 = _weighted_centered_gram(K2_div, k2_mu_div, mu2_sq)
    del sw_outer
    eig1_2850 = _fast_top1_eigvalsh(xp, gram2, sqrt_w, dconst["Q0_2"])
    scale2_sq = 1.0 / xp.maximum(eig1_2850, 1e-12)

    gram *= scale1_sq[:, None, None]
    gram2 *= scale2_sq[:, None, None]
    gram += gram2
    del gram2
    max_k = max(cfg.pca_dimensions)
    eigvals_top, eigvecs_top = _top_k_eigh(xp, gram, max_k, sqrt_w)
    del gram
    s_kept = xp.sqrt(eigvals_top)

    alpha = (sqrt_w[:, :, None] * eigvecs_top) / xp.maximum(s_kept[:, None, :], 1e-12)
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

    # 3. PairedMoments directly in participant / pair-difference space
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
        "part_pca_max": part_pca_max,
        "W_max": W_max,
        "T_max": T_max,
        "scale1_sq": scale1_sq,
        "scale2_sq": scale2_sq,
        "K_p1_tr": dconst["K_p1_tr"],
        "K_p1_vt": dconst["K_p1_vt"],
        "K_p2_tr": dconst["K_p2_tr"],
        "K_p2_vt": dconst["K_p2_vt"],
    }

    if compute_pca_suff or compute_cov_preds:
        y_mean = xp.sum(w_tr * y_tr[None, :], axis=1)
        yc = y_tr[None, :] - y_mean[:, None]
        cm = xp.sum(w_tr[:, :, None] * X_cov_tr, axis=1)
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
        out["preds_cov"] = preds_cov

    if compute_pca_suff:
        r_val_0 = y_val[None, :, None] - preds_cov
        sse_0 = xp.sum(w_val[:, :, None] * (r_val_0 * r_val_0), axis=1)

        pca_tr = part_pca_max[:, tr_p, :]
        pca_val = part_pca_max[:, val_p, :]
        p_mean = xp.sum(w_tr[:, :, None] * pca_tr, axis=1)
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

        out.update(
            {
                "sse_0": sse_0,
                "Sigma_p": Sigma_p,
                "Schur_p": Schur_p,
                "g_tilde_p": g_tilde_p,
                "Val_gram_p": Val_gram_p,
                "Val_rhs_p": Val_rhs_p,
            }
        )
    return out


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
        z_mean = xp.sum(w_tr[:, :, None] * Z_tr, axis=1)
        Z_c = Z_tr - z_mean[:, None, :]
        z_var = xp.sum(w_tr[:, :, None] * (Z_c * Z_c), axis=1)
        z_scale = xp.sqrt(xp.maximum(z_var, 1e-12))
        Z_std_tr = Z_c / z_scale[:, None, :]
        Z_std_val = (Z_val - z_mean[:, None, :]) / z_scale[:, None, :]
        X_tr = xp.concatenate([X_cov_tr, Z_std_tr], axis=2)
        X_val = xp.concatenate([X_cov_val, Z_std_val], axis=2)

    x_mean = xp.sum(w_tr[:, :, None] * X_tr, axis=1)
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

    cov_mean = xp.sum(w_tr[:, :, None] * X_cov_tr, axis=1)
    C_tr = X_cov_tr - cov_mean[:, None, :]
    C_val = X_cov_val - cov_mean[:, None, :]
    C_tr_T = xp.swapaxes(C_tr, -1, -2)
    K_cov_tr = C_tr @ C_tr_T
    K_cov_vt = C_val @ C_tr_T

    K_p1_tr = div["K_p1_tr"]
    K_p1_vt = div["K_p1_vt"]
    K_p2_tr = div["K_p2_tr"]
    K_p2_vt = div["K_p2_vt"]

    k1_mu_tr = w_tr @ K_p1_tr
    mu1_sq = xp.sum(w_tr * k1_mu_tr, axis=1)
    K1_spec_tr = K_p1_tr[None, :, :] - k1_mu_tr[:, :, None] - k1_mu_tr[:, None, :] + mu1_sq[:, None, None]
    k1_mu_vt = w_tr @ K_p1_vt.T
    K1_spec_vt = K_p1_vt[None, :, :] - k1_mu_vt[:, :, None] - k1_mu_tr[:, None, :] + mu1_sq[:, None, None]

    k2_mu_tr = w_tr @ K_p2_tr
    mu2_sq = xp.sum(w_tr * k2_mu_tr, axis=1)
    K2_spec_tr = K_p2_tr[None, :, :] - k2_mu_tr[:, :, None] - k2_mu_tr[:, None, :] + mu2_sq[:, None, None]
    k2_mu_vt = w_tr @ K_p2_vt.T
    K2_spec_vt = K_p2_vt[None, :, :] - k2_mu_vt[:, :, None] - k2_mu_tr[:, None, :] + mu2_sq[:, None, None]

    K_tr = K_cov_tr + scale1_sq[:, None, None] * K1_spec_tr + scale2_sq[:, None, None] * K2_spec_tr
    K_vt = K_cov_vt + scale1_sq[:, None, None] * K1_spec_vt + scale2_sq[:, None, None] * K2_spec_vt

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
        accum = accum + qk[:, None] * t_val
        preds_all[:, :, c] = accum
        if c < max_comp - 1:
            yk = yk - qk[:, None] * tk
            u = (w_tr * tk) / t_sq[:, None]
            kt_u = (K_tr @ u[:, :, None])[:, :, 0]
            u_kt_u = xp.sum(u * kt_u, axis=1)
            K_vt = K_vt - t_val[:, :, None] * kt_u[:, None, :]
            v_def = kt_u - (0.5 * u_kt_u)[:, None] * tk
            K_tr = K_tr - tk[:, :, None] * v_def[:, None, :] - v_def[:, :, None] * tk[:, None, :]

    sse = xp.sum(w_val[:, :, None] * ((y_val[None, :, None] - preds_all) ** 2), axis=1)
    return sse, preds_all


def batched_solve_directions_grid(
    xp: Any,
    W_max: Any,
    T_max: Any,
    k_list: list[int],
    lam_grid: Sequence[float],
) -> dict[int, Any]:
    """Solve generalized eigenproblems across all (k, lam) grid points simultaneously."""
    lam_xp = xp.asarray(lam_grid, dtype=xp.float64)
    out: dict[int, Any] = {}
    for k in k_list:
        Wk = W_max[:, :k, :k]
        Tk = T_max[:, :k, :k]
        d_w, q_w = xp.linalg.eigh(Wk)
        d_w = xp.maximum(d_w, 0.0)
        tau = xp.diagonal(Tk, axis1=-2, axis2=-1).sum(axis=-1) / float(k)
        T_tilde = xp.swapaxes(q_w, -1, -2) @ Tk @ q_w
        inv_sqrt = 1.0 / xp.sqrt(
            xp.maximum(d_w[:, None, :] + lam_xp[None, :, None] * tau[:, None, None], 1e-18)
        )
        M_batch = T_tilde[:, None, :, :] * (inv_sqrt[:, :, :, None] * inv_sqrt[:, :, None, :])
        M_batch = 0.5 * (M_batch + xp.swapaxes(M_batch, -1, -2))
        _, U_batch = xp.linalg.eigh(M_batch)
        U_desc = U_batch[:, :, :, ::-1]
        V_batch = q_w[:, None, :, :] @ (inv_sqrt[:, :, :, None] * U_desc)
        out[k] = V_batch
    return out


def _check_fast_batch_eligibility(
    dataset: CohortDataset,
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    inner_splits_per_fold: list[list[tuple[np.ndarray, np.ndarray]]],
    M_all: np.ndarray,
    cfg: PairedLipidConfig,
    gcache: dict[str, Any],
) -> np.ndarray:
    """Return boolean mask of bootstrap draws eligible for vectorized full-grid Schur evaluation."""
    n_b = M_all.shape[0]
    if gcache["has_missing_cov"] or list(cfg.retained_directions) != [1, 2, 3]:
        return np.zeros(n_b, dtype=bool)

    max_pls_c = max(cfg.pls_component_grid)
    eligible = np.ones(n_b, dtype=bool)
    p_pair = dataset.paired_participant_indices
    # Every configured k must stay below the design cap (paired participants - 1); otherwise
    # the sequential path caps k, which the fixed-shape batched grid cannot represent.
    min_inner_pairs = max(cfg.pca_dimensions) + 1

    for (tr_idx, te_idx), in_splits in zip(outer_splits, inner_splits_per_fold):
        if tr_idx.size - 2 < max_pls_c:
            return np.zeros(n_b, dtype=bool)
        eligible &= np.sum(M_all[:, te_idx] > 0, axis=1) >= 1
        eligible &= np.sum(M_all[:, tr_idx] > 0, axis=1) >= 3
        pair_otr = p_pair[np.isin(p_pair, tr_idx)]
        n_pair_otr = np.sum(M_all[:, pair_otr] > 0, axis=1)
        eligible &= n_pair_otr >= (min_inner_pairs + 1)

        min_pair_itr = np.full(n_b, 999999, dtype=int)
        for in_tr_rel, in_val_rel in in_splits:
            in_tr = tr_idx[in_tr_rel]
            in_val = tr_idx[in_val_rel]
            if in_tr.size - 2 < max_pls_c:
                return np.zeros(n_b, dtype=bool)
            eligible &= np.sum(M_all[:, in_tr] > 0, axis=1) >= 3
            eligible &= np.sum(M_all[:, in_val] > 0, axis=1) >= 1
            pair_itr = p_pair[np.isin(p_pair, in_tr)]
            n_pair_itr = np.sum(M_all[:, pair_itr] > 0, axis=1)
            eligible &= n_pair_itr >= min_inner_pairs
            min_pair_itr = np.minimum(min_pair_itr, n_pair_itr)
        eligible &= n_pair_otr > min_pair_itr

    return eligible


def _run_batched_multiplicity_draws(
    dataset: CohortDataset,
    y_all: np.ndarray,
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    inner_splits_per_fold: list[list[tuple[np.ndarray, np.ndarray]]],
    M_sub: np.ndarray,
    cfg: PairedLipidConfig,
    gcache: dict[str, Any],
    *,
    use_gpu: bool | None = None,
    batch_size: int | None = None,
    execution_metadata: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate multiplicity bootstrap draws in batches on GPU (CuPy) or CPU (NumPy)."""
    n_sub, n_p = M_sub.shape
    cp_mod = _get_gpu_xp() if use_gpu is not False else None
    xp = cp_mod if (cp_mod is not None and (use_gpu is True or (use_gpu is None and n_sub >= 32))) else np
    if execution_metadata is not None:
        gpu_requested = use_gpu is True or (use_gpu is None and n_sub >= 32)
        execution_metadata.update(
            device="gpu" if xp is not np else "cpu",
            gpu_requested=use_gpu,
            gpu_fallback=bool(gpu_requested and cp_mod is None),
        )
        if gpu_requested and cp_mod is None:
            execution_metadata["gpu_fallback_reason"] = "CuPy/CUDA device unavailable"
    if batch_size is None:
        batch_size = 2048 if xp is not np else 128

    y_all_xp = xp.asarray(y_all, dtype=xp.float64)
    alphas = list(cfg.ridge_alpha_grid)
    alphas_xp = xp.asarray(alphas, dtype=xp.float64)
    n_alphas = len(alphas)
    k_list = list(cfg.pca_dimensions)
    lam_list = list(cfg.lambda_grid)
    lam_xp = xp.asarray(lam_list, dtype=xp.float64)
    r_list = list(cfg.retained_directions)
    max_pls_c = max(cfg.pls_component_grid)
    part_band_xp = xp.asarray(gcache["part_band_mat"], dtype=xp.float64)

    oof_cov = np.zeros((n_sub, n_p), dtype=np.float64)
    oof_paired = np.zeros((n_sub, n_p), dtype=np.float64)
    oof_strongest = np.zeros((n_sub, n_p), dtype=np.float64)

    fold_dconsts = []
    for (tr_idx, te_idx), in_splits in zip(outer_splits, inner_splits_per_fold):
        in_dconsts = [
            _precompute_division_constants(xp, gcache, dataset, tr_idx[in_tr_rel], tr_idx[in_val_rel], cfg)
            for in_tr_rel, in_val_rel in in_splits
        ]
        out_dconst = _precompute_division_constants(xp, gcache, dataset, tr_idx, te_idx, cfg)
        fold_dconsts.append((tr_idx, te_idx, in_dconsts, out_dconst))

    n_k = len(k_list)
    n_lam = len(lam_list)
    n_r = len(r_list)

    for b_start in range(0, n_sub, batch_size):
        b_end = min(b_start + batch_size, n_sub)
        B = b_end - b_start
        M_batch = xp.asarray(M_sub[b_start:b_end], dtype=xp.float64)

        for tr_idx, te_idx, in_dconsts, out_dconst in fold_dconsts:
            inner_divs = []
            total_val_w = xp.zeros(B, dtype=xp.float64)
            for dconst in in_dconsts:
                in_tr = dconst["tr_p"]
                in_val = dconst["val_p"]
                div = batched_division_prep(
                    xp,
                    gcache,
                    dconst,
                    M_batch[:, in_tr],
                    M_batch[:, in_val],
                    y_all_xp,
                    alphas_xp,
                    cfg,
                    compute_pca_suff=True,
                )
                inner_divs.append(div)
                total_val_w = total_val_w + xp.sum(div["w_val"], axis=1)

            cov_sse = sum(div["sse_0"] for div in inner_divs)
            cov_mses = _to_np(xp, cov_sse / total_val_w[:, None])

            band_sse = xp.zeros((B, n_alphas), dtype=xp.float64)
            for div in inner_divs:
                Zb_tr = xp.broadcast_to(
                    part_band_xp[None, div["tr_p"], :], (B, div["tr_p"].size, part_band_xp.shape[1])
                )
                Zb_val = xp.broadcast_to(
                    part_band_xp[None, div["val_p"], :], (B, div["val_p"].size, part_band_xp.shape[1])
                )
                sse_b, _ = batched_ridge_sse_and_preds(
                    xp,
                    div["X_cov_tr"],
                    div["X_cov_val"],
                    Zb_tr,
                    Zb_val,
                    div["y_tr"],
                    div["y_val"],
                    div["w_tr"],
                    div["w_val"],
                    alphas_xp,
                )
                band_sse = band_sse + sse_b
            band_mses = _to_np(xp, band_sse / total_val_w[:, None])

            pca_sse = xp.zeros((B, n_k, n_alphas), dtype=xp.float64)
            for k_idx, k in enumerate(k_list):
                for div in inner_divs:
                    Zp_tr = div["part_pca_max"][:, div["tr_p"], :k]
                    Zp_val = div["part_pca_max"][:, div["val_p"], :k]
                    sse_p, _ = batched_ridge_sse_and_preds(
                        xp,
                        div["X_cov_tr"],
                        div["X_cov_val"],
                        Zp_tr,
                        Zp_val,
                        div["y_tr"],
                        div["y_val"],
                        div["w_tr"],
                        div["w_val"],
                        alphas_xp,
                    )
                    pca_sse[:, k_idx, :] = pca_sse[:, k_idx, :] + sse_p
            pca_mses = _to_np(xp, pca_sse / total_val_w[:, None, None])

            pls_sse = xp.zeros((B, max_pls_c), dtype=xp.float64)
            for div in inner_divs:
                sse_pls, _ = batched_pls_sse_and_preds(xp, div, max_pls_c)
                pls_sse = pls_sse + sse_pls
            pls_mses = _to_np(xp, pls_sse / total_val_w[:, None])

            pair_sse = xp.zeros((B, n_r, n_k * n_lam, n_alphas), dtype=xp.float64)
            for div in inner_divs:
                V_grid = batched_solve_directions_grid(xp, div["W_max"], div["T_max"], k_list, lam_list)
                pair_sse = pair_sse + batched_paired_schur_sse(xp, div, V_grid, k_list, alphas_xp)
            pair_mses = _to_np(xp, pair_sse / total_val_w[:, None, None, None])

            best_cov_a = _argmin_last(np.round(cov_mses, 12))

            best_band_a = _argmin_last(np.round(band_mses, 12))
            best_band_mse = band_mses[np.arange(B), best_band_a]

            pca_r = np.round(pca_mses, 12)
            best_pca_k_idx, best_pca_a = _select_pca_setting(pca_r)
            best_pca_mse = pca_r[np.arange(B), best_pca_k_idx, best_pca_a]

            pls_r = np.round(pls_mses, 12)
            best_pls_c_idx = np.argmin(pls_r, axis=1)
            best_pls_mse = pls_mses[np.arange(B), best_pls_c_idx]

            comp_stack = np.stack([best_band_mse, best_pca_mse, best_pls_mse], axis=1)
            best_comp_idx = np.argmin(comp_stack, axis=1)

            best_r_idx, best_k_idx, best_pair_l_idx, best_pair_a = _select_paired_setting(
                np.round(pair_mses.reshape(B, n_r, n_k, n_lam, n_alphas), 12)
            )

            outer_div = batched_division_prep(
                xp,
                gcache,
                out_dconst,
                M_batch[:, tr_idx],
                M_batch[:, te_idx],
                y_all_xp,
                alphas_xp,
                cfg,
                compute_pca_suff=False,
                compute_cov_preds=True,
            )
            preds_cov_np = _to_np(xp, outer_div["preds_cov"])
            pca_otr = outer_div["part_pca_max"][:, tr_idx, :]
            pca_ote = outer_div["part_pca_max"][:, te_idx, :]

            # Solve outer paired generalized eigensystem only for each draw's selected (k, lam)
            V_outer_sel: dict[int, Any] = {}
            for k_idx, k in enumerate(k_list):
                b_k = np.flatnonzero(best_k_idx == k_idx)
                if b_k.size == 0:
                    continue
                b_k_xp = xp.asarray(b_k)
                W_k = outer_div["W_max"][b_k_xp, :k, :k]
                T_k = outer_div["T_max"][b_k_xp, :k, :k]
                # tau must match PairedMoments.solve_directions: trace of T, not W.
                tr_T = xp.trace(T_k, axis1=-2, axis2=-1)
                tau = xp.maximum(tr_T / float(k), 1e-12)
                lam_sel = lam_xp[xp.asarray(best_pair_l_idx[b_k])]
                w_ev, q_w = xp.linalg.eigh(W_k)
                w_ev = xp.maximum(w_ev, 0.0)
                q_w_T = xp.swapaxes(q_w, -1, -2)
                T_tilde = q_w_T @ T_k @ q_w
                T_tilde = 0.5 * (T_tilde + xp.swapaxes(T_tilde, -1, -2))
                denom = w_ev + (lam_sel * tau)[:, None]
                inv_sqrt = 1.0 / xp.sqrt(xp.maximum(denom, 1e-12))
                M_mat = T_tilde * (inv_sqrt[:, :, None] * inv_sqrt[:, None, :])
                M_mat = 0.5 * (M_mat + xp.swapaxes(M_mat, -1, -2))
                _, U_mat = xp.linalg.eigh(M_mat)
                U_desc = U_mat[:, :, ::-1]
                V_outer_sel[k_idx] = q_w @ (inv_sqrt[:, :, None] * U_desc)

            pred_paired_ote = np.zeros((B, te_idx.size), dtype=np.float64)
            for r_idx, r in enumerate(r_list):
                mask_b = np.flatnonzero(best_r_idx == r_idx)
                if mask_b.size == 0:
                    continue
                Z_otr_sel = xp.zeros((mask_b.size, tr_idx.size, r), dtype=xp.float64)
                Z_ote_sel = xp.zeros((mask_b.size, te_idx.size, r), dtype=xp.float64)
                for k_idx, k in enumerate(k_list):
                    sub_m = np.flatnonzero(best_k_idx[mask_b] == k_idx)
                    if sub_m.size == 0:
                        continue
                    b_ids = mask_b[sub_m]
                    b_ids_xp = xp.asarray(b_ids)
                    b_k_all = np.flatnonzero(best_k_idx == k_idx)
                    pos_in_bk = xp.asarray(np.searchsorted(b_k_all, b_ids))
                    V_sel = V_outer_sel[k_idx][pos_in_bk, :, :r]
                    sub_m_xp = xp.asarray(sub_m)
                    Z_otr_sel[sub_m_xp] = pca_otr[b_ids_xp, :, :k] @ V_sel
                    Z_ote_sel[sub_m_xp] = pca_ote[b_ids_xp, :, :k] @ V_sel

                b_all_xp = xp.asarray(mask_b)
                _, pr_pair_sub = batched_ridge_sse_and_preds(
                    xp,
                    outer_div["X_cov_tr"][b_all_xp],
                    outer_div["X_cov_val"][b_all_xp],
                    Z_otr_sel,
                    Z_ote_sel,
                    outer_div["y_tr"],
                    outer_div["y_val"],
                    outer_div["w_tr"][b_all_xp],
                    outer_div["w_val"][b_all_xp],
                    alphas_xp,
                )
                pr_pair_sub_np = _to_np(xp, pr_pair_sub)
                pred_paired_ote[mask_b] = pr_pair_sub_np[np.arange(mask_b.size), :, best_pair_a[mask_b]]

            pred_comp_sel = np.zeros((B, te_idx.size), dtype=np.float64)
            b_band = np.flatnonzero(best_comp_idx == 0)
            if b_band.size > 0:
                b_band_xp = xp.asarray(b_band)
                Zb_otr = xp.broadcast_to(
                    part_band_xp[None, tr_idx, :], (b_band.size, tr_idx.size, part_band_xp.shape[1])
                )
                Zb_ote = xp.broadcast_to(
                    part_band_xp[None, te_idx, :], (b_band.size, te_idx.size, part_band_xp.shape[1])
                )
                _, pr_b = batched_ridge_sse_and_preds(
                    xp,
                    outer_div["X_cov_tr"][b_band_xp],
                    outer_div["X_cov_val"][b_band_xp],
                    Zb_otr,
                    Zb_ote,
                    outer_div["y_tr"],
                    outer_div["y_val"],
                    outer_div["w_tr"][b_band_xp],
                    outer_div["w_val"][b_band_xp],
                    alphas_xp,
                )
                pr_b_np = _to_np(xp, pr_b)
                pred_comp_sel[b_band] = pr_b_np[np.arange(b_band.size), :, best_band_a[b_band]]

            for k_idx, k in enumerate(k_list):
                b_pca = np.flatnonzero((best_comp_idx == 1) & (best_pca_k_idx == k_idx))
                if b_pca.size > 0:
                    b_pca_xp = xp.asarray(b_pca)
                    _, pr_p = batched_ridge_sse_and_preds(
                        xp,
                        outer_div["X_cov_tr"][b_pca_xp],
                        outer_div["X_cov_val"][b_pca_xp],
                        pca_otr[b_pca_xp, :, :k],
                        pca_ote[b_pca_xp, :, :k],
                        outer_div["y_tr"],
                        outer_div["y_val"],
                        outer_div["w_tr"][b_pca_xp],
                        outer_div["w_val"][b_pca_xp],
                        alphas_xp,
                    )
                    pr_p_np = _to_np(xp, pr_p)
                    pred_comp_sel[b_pca] = pr_p_np[np.arange(b_pca.size), :, best_pca_a[b_pca]]

            b_pls = np.flatnonzero(best_comp_idx == 2)
            if b_pls.size > 0:
                b_pls_xp = xp.asarray(b_pls)
                pls_div = {
                    "w_tr": outer_div["w_tr"][b_pls_xp],
                    "w_val": outer_div["w_val"][b_pls_xp],
                    "y_tr": outer_div["y_tr"],
                    "y_val": outer_div["y_val"],
                    "X_cov_tr": outer_div["X_cov_tr"][b_pls_xp],
                    "X_cov_val": outer_div["X_cov_val"][b_pls_xp],
                    "scale1_sq": outer_div["scale1_sq"][b_pls_xp],
                    "scale2_sq": outer_div["scale2_sq"][b_pls_xp],
                    "K_p1_tr": outer_div["K_p1_tr"],
                    "K_p1_vt": outer_div["K_p1_vt"],
                    "K_p2_tr": outer_div["K_p2_tr"],
                    "K_p2_vt": outer_div["K_p2_vt"],
                }
                _, pr_pls = batched_pls_sse_and_preds(xp, pls_div, max_pls_c)
                pr_pls_np = _to_np(xp, pr_pls)
                pred_comp_sel[b_pls] = pr_pls_np[np.arange(b_pls.size), :, best_pls_c_idx[b_pls]]

            b_arange = np.arange(B)
            oof_cov[b_start:b_end, te_idx] = preds_cov_np[b_arange, :, best_cov_a]
            oof_paired[b_start:b_end, te_idx] = pred_paired_ote
            oof_strongest[b_start:b_end, te_idx] = pred_comp_sel

    w_norm_all = M_sub / np.sum(M_sub, axis=1, keepdims=True)
    mse_cov = np.sum(w_norm_all * ((y_all[None, :] - oof_cov) ** 2), axis=1)
    mse_paired = np.sum(w_norm_all * ((y_all[None, :] - oof_paired) ** 2), axis=1)
    mse_strongest = np.sum(w_norm_all * ((y_all[None, :] - oof_strongest) ** 2), axis=1)
    return mse_cov - mse_paired, mse_strongest - mse_paired
