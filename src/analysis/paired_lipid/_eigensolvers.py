"""Batched exact fp64 eigensolvers for CPU (NumPy/SciPy) and GPU (CuPy)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import linalg

from src.analysis.paired_lipid._backend import (
    _cupyx_supports_batched_triangular,
    _get_cupyx_linalg,
    _to_np,
)


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


def _fast_top1_eigvalsh_from_gram(
    xp: Any,
    K_div: Any,
    k_mu: Any,
    mu_sq: Any,
    sw: Any,
    Q0: Any,
) -> Any:
    """Matrix-free 2-step subspace Rayleigh-Ritz with residual verification and exact per-slice fallback."""
    B, n_s = sw.shape
    if B == 0:
        return xp.empty(0, dtype=xp.float64)
    q = int(Q0.shape[1])
    Q0_t = xp.ascontiguousarray(Q0.T)

    def _apply_g_t(V_t: Any) -> Any:
        U_t = V_t * sw[:, None, :]
        KU_t = (U_t.reshape(B * q, n_s) @ K_div).reshape(B, q, n_s)
        sU = xp.sum(U_t, axis=2)
        kU = xp.sum(U_t * k_mu[:, None, :], axis=2)
        out = KU_t - sU[:, :, None] * k_mu[:, None, :] - (kU - sU * mu_sq[:, None])[:, :, None]
        out *= sw[:, None, :]
        return out

    Y_t = _apply_g_t(Q0_t[None, :, :] * sw[:, None, :])
    Y_t = _apply_g_t(Y_t)
    YtY = Y_t @ xp.swapaxes(Y_t, -1, -2)
    YtY = 0.5 * (YtY + xp.swapaxes(YtY, -1, -2))
    eye_q = xp.eye(q, dtype=xp.float64)[None, :, :]
    yt_fin = xp.all(xp.isfinite(YtY), axis=(-2, -1))
    d, V = xp.linalg.eigh(xp.where(yt_fin[:, None, None], YtY, eye_q))
    d_max = xp.maximum(d[:, -1:], 1e-24)
    valid = d > (1e-18 * d_max)
    inv_s = xp.where(valid, 1.0 / xp.sqrt(xp.maximum(d, 1e-30)), 0.0)
    Q_t = xp.swapaxes(V * inv_s[:, None, :], -1, -2) @ Y_t
    GQ_t = _apply_g_t(Q_t)
    H = Q_t @ xp.swapaxes(GQ_t, -1, -2)
    H = 0.5 * (H + xp.swapaxes(H, -1, -2))
    h_fin = yt_fin & xp.all(xp.isfinite(H), axis=(-2, -1))
    evals_h, evecs_h = xp.linalg.eigh(xp.where(h_fin[:, None, None], H, eye_q))
    theta1 = evals_h[:, -1]
    w_top = evecs_h[:, :, -1]
    u1 = (w_top[:, None, :] @ Q_t)[:, 0, :]
    Gu1 = (w_top[:, None, :] @ GQ_t)[:, 0, :]
    diff = Gu1 - theta1[:, None] * u1
    r_norm = xp.sqrt(xp.sum(diff * diff, axis=1))
    theta_ok = h_fin & xp.isfinite(theta1) & (theta1 > 0.0) & xp.isfinite(r_norm)
    theta1_safe = xp.where(theta_ok, theta1, 1.0)
    lin_bound = r_norm / theta1_safe
    if q >= 2:
        gap = theta1 - evals_h[:, -2]
        gap_pos = theta_ok & xp.isfinite(gap) & (gap > 0.0)
        denom_quad = xp.where(gap_pos, theta1_safe * gap, 1.0)
        quad_bound = xp.where(gap_pos, (r_norm * r_norm) / denom_quad, xp.inf)
        err_bound = xp.fmin(lin_bound, quad_bound)
    else:
        err_bound = lin_bound

    bad_mask = (~theta_ok) | (~xp.isfinite(err_bound)) | (err_bound > 1e-14)
    bad_host = np.flatnonzero(_to_np(xp, bad_mask) > 0.5)
    if bad_host.size == 0:
        return theta1

    out = theta1.copy()
    if xp is np:
        for b in bad_host:
            if not (np.all(np.isfinite(sw[b])) and np.all(np.isfinite(k_mu[b])) and np.isfinite(mu_sq[b])):
                out[b] = np.nan
                continue
            act = np.flatnonzero(sw[b] > 0.0)
            na = int(act.size)
            if na == 0:
                out[b] = 0.0
                continue
            sw_a = sw[b, act]
            km_a = k_mu[b, act]
            G_sub = K_div[np.ix_(act, act)]
            G_sub -= km_a[:, None]
            G_sub -= km_a[None, :]
            G_sub += mu_sq[b]
            G_sub *= sw_a[:, None]
            G_sub *= sw_a[None, :]
            G_sub = 0.5 * (G_sub + G_sub.T)
            if not np.all(np.isfinite(G_sub)):
                out[b] = np.nan
                continue
            out[b] = float(
                linalg.eigvalsh(
                    G_sub,
                    subset_by_index=[na - 1, na - 1],
                    driver="evr",
                    check_finite=False,
                    overwrite_a=True,
                )[-1]
            )
        return out

    bad_xp = xp.asarray(bad_host)
    sw_b = sw[bad_xp]
    k_mu_b = k_mu[bad_xp]
    mu_sq_b = mu_sq[bad_xp]
    G_bad = K_div[None, :, :] - k_mu_b[:, :, None] - k_mu_b[:, None, :] + mu_sq_b[:, None, None]
    G_bad *= sw_b[:, :, None]
    G_bad *= sw_b[:, None, :]
    G_bad = 0.5 * (G_bad + xp.swapaxes(G_bad, -1, -2))
    active_b = sw_b > 0.0
    n_act_b = _to_np(xp, xp.sum(active_b, axis=1)).astype(int)
    pos = xp.arange(n_s)[None, :]
    order = xp.argsort(xp.where(active_b, pos, n_s + pos), axis=1)
    bucket_step = 32 if bad_host.size <= 256 else 16
    sizes = np.minimum(-(-n_act_b // bucket_step) * bucket_step, n_s)
    ev_bad = xp.zeros(bad_host.size, dtype=xp.float64)
    for m in np.unique(sizes):
        if int(m) == 0:
            continue
        loc = xp.asarray(np.flatnonzero(sizes == m))
        idx = order[loc, :m]
        G_sub = xp.take_along_axis(G_bad[loc], idx[:, :, None], axis=1)
        G_sub = xp.take_along_axis(G_sub, idx[:, None, :], axis=2)
        ev_sub = xp.linalg.eigvalsh(G_sub)
        ev_bad[loc] = ev_sub[:, -1]
        del G_sub, ev_sub
    fin_b = (
        xp.all(xp.isfinite(sw_b), axis=1)
        & xp.all(xp.isfinite(k_mu_b), axis=1)
        & xp.isfinite(mu_sq_b)
    )
    out[bad_xp] = xp.where(fin_b, ev_bad, xp.nan)
    del G_bad
    return out


def _orthonormalize_eigh(xp: Any, Yn: Any, YtY: Any) -> Any:
    """Fallback symmetric orthonormalization via eigendecomposition of Yn^T @ Yn."""
    d, V = xp.linalg.eigh(YtY)
    d_max = xp.maximum(d[:, -1:], 1e-30)
    inv_s = xp.where(d > (1e-24 * d_max), 1.0 / xp.sqrt(xp.maximum(d, 1e-30)), 0.0)
    return Yn @ (V * inv_s[:, None, :])


def _solve_lower_triangular_batch(xp: Any, L: Any, rhs: Any) -> Any:
    """Solve L @ X = rhs for a batched lower-triangular Cholesky factor L."""
    if xp is not np and _cupyx_supports_batched_triangular():
        cpx_linalg = _get_cupyx_linalg()
        if cpx_linalg is not None:
            try:
                return cpx_linalg.solve_triangular(L, rhs, lower=True, check_finite=False)
            except Exception:
                pass
    return xp.linalg.solve(L, rhs)


def _rayleigh_ritz_32(
    xp: Any,
    G_sym: Any,
    Y: Any,
    *,
    check_spd_sync: bool = True,
    return_spd: bool = False,
) -> tuple[Any, ...]:
    """Column-normalized 32x32 Rayleigh-Ritz step using batched Cholesky-QR with eigh fallback."""
    norms = xp.sqrt(xp.sum(Y * Y, axis=1, keepdims=True))
    Yn = Y / xp.maximum(norms, 1e-30)
    YtY = xp.swapaxes(Yn, -1, -2) @ Yn
    YtY = 0.5 * (YtY + xp.swapaxes(YtY, -1, -2))

    if Y.shape[0] == 0:
        H = xp.zeros((0, Y.shape[2], Y.shape[2]), dtype=xp.float64)
        theta, W = xp.linalg.eigh(H)
        if return_spd:
            return theta, Yn @ W, (G_sym @ Yn) @ W, xp.ones(0, dtype=bool)
        return theta, Yn @ W, (G_sym @ Yn) @ W

    Q: Any | None = None
    spd: Any
    try:
        L = xp.linalg.cholesky(YtY)
    except Exception:
        L = None

    if L is not None:
        spd = xp.all(xp.isfinite(L), axis=(1, 2)) & (
            xp.all(xp.diagonal(L, axis1=-2, axis2=-1) > 1e-12, axis=1)
        )
        if not check_spd_sync:
            eye_q = xp.eye(Y.shape[2], dtype=xp.float64)[None, :, :]
            L_safe = xp.where(spd[:, None, None], L, eye_q)
            Qt = _solve_lower_triangular_batch(xp, L_safe, xp.swapaxes(Yn, -1, -2))
            Q = xp.swapaxes(Qt, -1, -2)
        elif bool(xp.all(spd)):
            Qt = _solve_lower_triangular_batch(xp, L, xp.swapaxes(Yn, -1, -2))
            Q = xp.swapaxes(Qt, -1, -2)
        else:
            Q = xp.empty_like(Yn)
            good = xp.flatnonzero(spd)
            if int(good.size) > 0:
                Qt_good = _solve_lower_triangular_batch(xp, L[good], xp.swapaxes(Yn[good], -1, -2))
                Q[good] = xp.swapaxes(Qt_good, -1, -2)
            bad = xp.flatnonzero(~spd)
            if int(bad.size) > 0:
                Q[bad] = _orthonormalize_eigh(xp, Yn[bad], YtY[bad])
    else:
        Q = _orthonormalize_eigh(xp, Yn, YtY)
        spd = xp.all(xp.isfinite(Q), axis=(1, 2))

    GQ = G_sym @ Q
    H = xp.swapaxes(Q, -1, -2) @ GQ
    H = 0.5 * (H + xp.swapaxes(H, -1, -2))
    theta, W = xp.linalg.eigh(H)
    if return_spd:
        return theta, Q @ W, GQ @ W, spd
    return theta, Q @ W, GQ @ W


def _chefsi_top_k_gpu(
    xp: Any,
    G_sym: Any,
    max_k: int,
    sw: Any,
    Q0_comb: Any,
    *,
    n_cycles: int = 3,
    poly_deg: int = 4,
    residual_tol: float = 1e-12,
    max_extra_cycles: int = 3,
) -> tuple[Any, Any]:
    """Warm-started Chebyshev-filtered subspace iteration (q=32) with adaptive residual verification.

    Uses batched GEMMs, branchless batched Cholesky-QR, and 32x32 warp-batched ``eigh`` with a
    single host residual check after ``n_cycles`` initial cycles. Any still-unconverged subset
    receives up to ``max_extra_cycles`` additional cycles as long as residuals decrease by at least
    2x; remaining draws (or draws with fewer than 32 active samples or non-SPD Cholesky factors) are
    solved exactly via ``_exact_top_k_eigh_active`` so fp64 accuracy is strictly preserved.
    """
    B, n_s, _ = G_sym.shape
    if B == 0 or n_s < 32:
        return _exact_top_k_eigh_active(xp, G_sym, max_k, sw)

    spd_init = xp.sum(sw > 0, axis=1) >= 32

    def _run_cycle(
        G_cur: Any, theta_cur: Any, U_cur: Any, GU_cur: Any, spd_cur: Any
    ) -> tuple[Any, Any, Any, Any]:
        b_cut = xp.maximum(theta_cur[:, 0], 1e-15)
        inv_e = 2.0 / b_cut[:, None, None]
        Y_prev = U_cur
        Y_curr = inv_e * GU_cur - U_cur
        del GU_cur
        for _ in range(1, poly_deg):
            GY = G_cur @ Y_curr
            Y_next = 2.0 * (inv_e * GY - Y_curr) - Y_prev
            del GY
            Y_prev = Y_curr
            Y_curr = Y_next
        del Y_prev
        theta_next, U_next, GU_next, spd_step = _rayleigh_ritz_32(
            xp, G_cur, Y_curr, check_spd_sync=False, return_spd=True
        )
        return theta_next, U_next, GU_next, spd_cur & spd_step

    def _check_residuals(
        theta_cur: Any, U_cur: Any, GU_cur: Any, spd_cur: Any
    ) -> tuple[Any, Any, np.ndarray, np.ndarray]:
        ev_k = xp.maximum(theta_cur[:, -max_k:][:, ::-1], 0.0)
        u_k = U_cur[:, :, -max_k:][:, :, ::-1]
        gu_k = GU_cur[:, :, -max_k:][:, :, ::-1]
        diff = gu_k - u_k * ev_k[:, None, :]
        res_norm = xp.max(xp.sqrt(xp.sum(diff * diff, axis=1)), axis=1)
        rel_res_xp = xp.where(spd_cur, res_norm / xp.maximum(ev_k[:, 0], 1e-12), xp.inf)
        rel_res = _to_np(xp, rel_res_xp)
        conv = np.isfinite(rel_res) & (rel_res <= residual_tol)
        return ev_k, u_k, rel_res, conv

    Y0 = G_sym @ (sw[:, :, None] * Q0_comb[None, :, :])
    theta, U, GU, spd_step0 = _rayleigh_ritz_32(xp, G_sym, Y0, check_spd_sync=False, return_spd=True)
    spd_ok = spd_init & spd_step0
    del Y0
    for _ in range(n_cycles):
        theta, U, GU, spd_ok = _run_cycle(G_sym, theta, U, GU, spd_ok)

    ev_k, u_k, rel_res, converged_sub = _check_residuals(theta, U, GU, spd_ok)
    if np.all(converged_sub):
        return ev_k, u_k

    ev_out = xp.zeros((B, max_k), dtype=xp.float64)
    u_out = xp.zeros((B, n_s, max_k), dtype=xp.float64)
    conv_global = np.zeros(B, dtype=bool)
    if np.any(converged_sub):
        ok_idx = xp.asarray(np.flatnonzero(converged_sub))
        ev_out[ok_idx] = ev_k[ok_idx]
        u_out[ok_idx] = u_k[ok_idx]
        conv_global[converged_sub] = True
    del ev_k, u_k

    # Adaptively continue only on draws that are finite/SPD and still unconverged.
    rem_mask = (~converged_sub) & np.isfinite(rel_res)
    if max_extra_cycles > 0 and np.any(rem_mask):
        rem_res = rel_res[rem_mask]
        prev_med = float(np.median(rem_res)) if rem_res.size > 0 else float("inf")
        if np.isfinite(prev_med):
            rem_host = np.flatnonzero(rem_mask)
            if rem_host.size < B:
                rem_xp = xp.asarray(rem_host)
                G_r = G_sym[rem_xp]
                theta_r = theta[rem_xp]
                U_r = U[rem_xp]
                GU_r = GU[rem_xp]
                spd_r = spd_ok[rem_xp]
            else:
                G_r = G_sym
                theta_r = theta
                U_r = U
                GU_r = GU
                spd_r = spd_ok
            del theta, U, GU, spd_ok
            for _ in range(max_extra_cycles):
                theta_r, U_r, GU_r, spd_r = _run_cycle(G_r, theta_r, U_r, GU_r, spd_r)
                ev_r, u_r, rel_r, conv_r = _check_residuals(theta_r, U_r, GU_r, spd_r)
                if np.any(conv_r):
                    ok_loc = xp.asarray(np.flatnonzero(conv_r))
                    ok_glob = xp.asarray(rem_host[conv_r])
                    ev_out[ok_glob] = ev_r[ok_loc]
                    u_out[ok_glob] = u_r[ok_loc]
                    conv_global[rem_host[conv_r]] = True
                del ev_r, u_r
                still_mask = (~conv_r) & np.isfinite(rel_r)
                if not np.any(still_mask):
                    break
                cur_med = float(np.median(rel_r[still_mask]))
                if not np.isfinite(cur_med) or cur_med > 0.5 * prev_med:
                    break
                prev_med = cur_med
                still_loc = np.flatnonzero(still_mask)
                if still_loc.size < rem_host.size:
                    still_xp = xp.asarray(still_loc)
                    rem_host = rem_host[still_loc]
                    G_r = G_r[still_xp]
                    theta_r = theta_r[still_xp]
                    U_r = U_r[still_xp]
                    GU_r = GU_r[still_xp]
                    spd_r = spd_r[still_xp]
            del theta_r, U_r, GU_r, G_r, spd_r
        else:
            del theta, U, GU, spd_ok
    else:
        del theta, U, GU, spd_ok

    bad_host = np.flatnonzero(~conv_global)
    if bad_host.size > 0:
        bad_xp = xp.asarray(bad_host)
        ev_fb, u_fb = _exact_top_k_eigh_active(xp, G_sym[bad_xp], max_k, sw[bad_xp])
        ev_out[bad_xp] = ev_fb
        u_out[bad_xp] = u_fb
    return ev_out, u_out


def _exact_top_k_eigh_active(xp: Any, G_sym: Any, max_k: int, sw: Any) -> tuple[Any, Any]:
    """Exact fp64 top-max_k eigenpairs, solved only over each draw's active samples.

    A sample of a participant with bootstrap multiplicity 0 has weight 0, so its gram row and
    column are zero (about 37% of samples per draw). The top eigenpairs of the active block equal
    those of the full matrix; eigh cost grows with the cube of the size, so solving the smaller
    block is the speed-up. Draws are grouped by active count rounded up to a bucket step;
    the extra rows in a group are zero rows and add only zero eigenvalues.
    """
    B, n_s, _ = G_sym.shape
    active = sw > 0
    n_act = _to_np(xp, xp.sum(active, axis=1)).astype(int)
    pos = xp.arange(n_s)[None, :]
    order = xp.argsort(xp.where(active, pos, n_s + pos), axis=1)
    # Keep bucket_step=32 for small batches and 16 for larger batches to balance cuSOLVER launches.
    bucket_step = 32 if B <= 256 else 16
    sizes = np.minimum(-(-n_act // bucket_step) * bucket_step, n_s)

    ev_out = xp.zeros((B, max_k), dtype=xp.float64)
    u_out = xp.zeros((B, n_s, max_k), dtype=xp.float64)
    for m in np.unique(sizes):
        b_idx = xp.asarray(np.flatnonzero(sizes == m))
        idx = order[b_idx, :m]
        G_sub = xp.take_along_axis(G_sym[b_idx], idx[:, :, None], axis=1)
        G_sub = xp.take_along_axis(G_sub, idx[:, None, :], axis=2)
        ev, u = xp.linalg.eigh(G_sub)
        del G_sub
        k = min(max_k, int(m))
        ev_out[b_idx, :k] = xp.maximum(ev[:, -k:][:, ::-1], 0.0)
        u_out[b_idx[:, None], idx, :k] = u[:, :, -k:][:, :, ::-1]
        del ev, u
    return ev_out, u_out


def _top_k_eigh(
    xp: Any,
    G_batch: Any,
    max_k: int,
    sw: Any,
    Q0_comb: Any | None = None,
    *,
    is_symmetric: bool = False,
    q0_comb_ratio: float | None = None,
) -> tuple[Any, Any]:
    """Exact fp64 top-max_k eigenpairs: active-sample LAPACK dsyevr on CPU, verified CheFSI on GPU."""
    if is_symmetric:
        G_sym = G_batch
    else:
        G_sym = G_batch + xp.swapaxes(G_batch, -1, -2)
        G_sym *= 0.5
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
                overwrite_a=True,
            )
            ev_out[b, :k_b] = np.maximum(ev_b[::-1], 0.0)
            u_out[b, act, :k_b] = u_b[:, ::-1]
        return ev_out, u_out

    B, n_s, _ = G_sym.shape
    if n_s <= 32:
        ev, u = xp.linalg.eigh(G_sym)
        k = min(max_k, n_s)
        ev_desc = xp.maximum(ev[:, -k:][:, ::-1], 0.0)
        u_desc = u[:, :, -k:][:, :, ::-1]
        if k == max_k:
            return ev_desc, u_desc
        ev_pad = xp.zeros((B, max_k), dtype=xp.float64)
        u_pad = xp.zeros((B, n_s, max_k), dtype=xp.float64)
        ev_pad[:, :k] = ev_desc
        u_pad[:, :, :k] = u_desc
        return ev_pad, u_pad

    if Q0_comb is not None and Q0_comb.shape[1] == 32 and max_k <= 24:
        if q0_comb_ratio is not None and q0_comb_ratio < 2.0:
            return _chefsi_top_k_gpu(
                xp, G_sym, max_k, sw, Q0_comb, n_cycles=5, poly_deg=5, max_extra_cycles=4
            )
        return _chefsi_top_k_gpu(xp, G_sym, max_k, sw, Q0_comb)

    return _exact_top_k_eigh_active(xp, G_sym, max_k, sw)


def _whitened_generalized_eigh(
    xp: Any,
    W_k: Any,
    T_k: Any,
    lam: Any,
    *,
    grid_mode: bool,
    eigenvalue_tolerance: float = 1e-10,
) -> Any:
    """Solve T_k v = rho (W_k + lam * tau * I) v via symmetric whitening in O(k^3).

    When ``grid_mode=True``, ``lam`` has shape ``(n_lam,)`` and output has shape ``(B, n_lam, k, k)``.
    When ``grid_mode=False``, ``lam`` has shape ``(B,)`` and output has shape ``(B, k, k)``.
    Draws with ``tau <= eigenvalue_tolerance`` or non-finite inputs propagate ``NaN``.
    """
    k = W_k.shape[-1]
    inputs_finite = xp.all(xp.isfinite(W_k), axis=(-2, -1)) & xp.all(xp.isfinite(T_k), axis=(-2, -1))
    eye_k = xp.eye(k, dtype=xp.float64)
    W_safe = xp.where(inputs_finite[:, None, None], W_k, eye_k[None, :, :])
    W_safe = 0.5 * (W_safe + xp.swapaxes(W_safe, -1, -2))
    T_safe = xp.where(inputs_finite[:, None, None], T_k, eye_k[None, :, :])
    T_safe = 0.5 * (T_safe + xp.swapaxes(T_safe, -1, -2))

    d_w, q_w = xp.linalg.eigh(W_safe)
    tau_raw = xp.diagonal(T_safe, axis1=-2, axis2=-1).sum(axis=-1) / float(k)
    tau_valid = inputs_finite & xp.isfinite(tau_raw) & (tau_raw > float(eigenvalue_tolerance))
    tau_safe = xp.where(tau_valid, tau_raw, 1.0)

    T_tilde = xp.swapaxes(q_w, -1, -2) @ T_safe @ q_w
    T_tilde = 0.5 * (T_tilde + xp.swapaxes(T_tilde, -1, -2))
    if grid_mode:
        denom = d_w[:, None, :] + lam[None, :, None] * tau_safe[:, None, None]
        max_denom = xp.max(denom, axis=-1, keepdims=True)
        denom_ok = xp.all(
            xp.isfinite(denom) & (denom > 0.0) & (denom > (1e-14 * max_denom)),
            axis=-1,
        )
        valid_out = tau_valid[:, None] & denom_ok
        denom_safe = xp.where(xp.isfinite(denom) & (denom > 0.0), denom, 1.0)
        inv_sqrt = xp.where(denom_ok[:, :, None], 1.0 / xp.sqrt(denom_safe), 0.0)
        M_batch = T_tilde[:, None, :, :] * (inv_sqrt[:, :, :, None] * inv_sqrt[:, :, None, :])
        M_batch = 0.5 * (M_batch + xp.swapaxes(M_batch, -1, -2))
        _, U_batch = xp.linalg.eigh(M_batch)
        vecs = q_w[:, None, :, :] @ (inv_sqrt[:, :, :, None] * U_batch[:, :, :, ::-1])
        return xp.where(valid_out[:, :, None, None], vecs, xp.nan)

    denom = d_w + (lam * tau_safe)[:, None]
    max_denom = xp.max(denom, axis=-1, keepdims=True)
    denom_ok = xp.all(
        xp.isfinite(denom) & (denom > 0.0) & (denom > (1e-14 * max_denom)),
        axis=-1,
    )
    valid_out = tau_valid & denom_ok
    denom_safe = xp.where(xp.isfinite(denom) & (denom > 0.0), denom, 1.0)
    inv_sqrt = xp.where(denom_ok[:, None], 1.0 / xp.sqrt(denom_safe), 0.0)
    M_mat = T_tilde * (inv_sqrt[:, :, None] * inv_sqrt[:, None, :])
    M_mat = 0.5 * (M_mat + xp.swapaxes(M_mat, -1, -2))
    _, U_mat = xp.linalg.eigh(M_mat)
    vecs = q_w @ (inv_sqrt[:, :, None] * U_mat[:, :, ::-1])
    return xp.where(valid_out[:, None, None], vecs, xp.nan)


def batched_solve_directions_grid(
    xp: Any,
    W_max: Any,
    T_max: Any,
    k_list: list[int],
    lam_grid: Sequence[float],
    *,
    eigenvalue_tolerance: float = 1e-10,
) -> dict[int, Any]:
    """Solve generalized eigenproblems across all (k, lam) grid points simultaneously."""
    lam_xp = xp.asarray(lam_grid, dtype=xp.float64)
    return {
        k: _whitened_generalized_eigh(
            xp,
            W_max[:, :k, :k],
            T_max[:, :k, :k],
            lam_xp,
            grid_mode=True,
            eigenvalue_tolerance=eigenvalue_tolerance,
        )
        for k in k_list
    }
