"""Batched exact fp64 eigensolvers for CPU (NumPy/SciPy) and GPU (CuPy)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import linalg

from src.analysis.paired_lipid._backend import _to_np


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


def _orthonormalize_eigh(xp: Any, Yn: Any, YtY: Any) -> Any:
    """Fallback symmetric orthonormalization via eigendecomposition of Yn^T @ Yn."""
    d, V = xp.linalg.eigh(YtY)
    d_max = xp.maximum(d[:, -1:], 1e-30)
    inv_s = xp.where(d > (1e-24 * d_max), 1.0 / xp.sqrt(xp.maximum(d, 1e-30)), 0.0)
    return Yn @ (V * inv_s[:, None, :])


def _solve_lower_triangular_batch(xp: Any, L: Any, rhs: Any) -> Any:
    """Solve L @ X = rhs for a batched lower-triangular Cholesky factor L."""
    if xp is not np:
        try:
            import cupyx.scipy.linalg as cpx_linalg  # type: ignore[import-not-found]

            return cpx_linalg.solve_triangular(L, rhs, lower=True, check_finite=False)
        except Exception:
            pass
    return xp.linalg.solve(L, rhs)


def _rayleigh_ritz_32(xp: Any, G_sym: Any, Y: Any) -> tuple[Any, Any, Any]:
    """Column-normalized 32x32 Rayleigh-Ritz step using batched Cholesky-QR with eigh fallback."""
    norms = xp.linalg.norm(Y, axis=1, keepdims=True)
    Yn = Y / xp.maximum(norms, 1e-30)
    YtY = xp.swapaxes(Yn, -1, -2) @ Yn
    YtY = 0.5 * (YtY + xp.swapaxes(YtY, -1, -2))

    if Y.shape[0] == 0:
        H = xp.zeros((0, Y.shape[2], Y.shape[2]), dtype=xp.float64)
        theta, W = xp.linalg.eigh(H)
        return theta, Yn @ W, (G_sym @ Yn) @ W

    Q: Any | None = None
    try:
        L = xp.linalg.cholesky(YtY)
    except Exception:
        L = None

    if L is not None:
        spd = xp.all(xp.isfinite(L), axis=(1, 2)) & (
            xp.all(xp.diagonal(L, axis1=-2, axis2=-1) > 1e-12, axis=1)
        )
        if bool(xp.all(spd)):
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

    GQ = G_sym @ Q
    H = xp.swapaxes(Q, -1, -2) @ GQ
    H = 0.5 * (H + xp.swapaxes(H, -1, -2))
    theta, W = xp.linalg.eigh(H)
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
) -> tuple[Any, Any]:
    """Warm-started Chebyshev-filtered subspace iteration (q=32) with exact residual verification.

    Uses batched GEMMs, batched Cholesky-QR, and 32x32 warp-batched ``eigh``; any draw with
    fewer than 32 active samples or eigenpair residual exceeding ``residual_tol`` is solved
    exactly via ``_exact_top_k_eigh_active`` so fp64 accuracy is strictly preserved.
    """
    B, n_s, _ = G_sym.shape
    n_act = _to_np(xp, xp.sum(sw > 0, axis=1)).astype(int)
    can_chefsi = n_act >= 32
    if not np.any(can_chefsi):
        return _exact_top_k_eigh_active(xp, G_sym, max_k, sw)

    if np.all(can_chefsi):
        G_run = G_sym
        sw_run = sw
    else:
        c_idx = xp.asarray(np.flatnonzero(can_chefsi))
        G_run = G_sym[c_idx]
        sw_run = sw[c_idx]

    Y0 = G_run @ (sw_run[:, :, None] * Q0_comb[None, :, :])
    theta, U, GU = _rayleigh_ritz_32(xp, G_run, Y0)
    del Y0
    for _ in range(n_cycles):
        b_cut = xp.maximum(theta[:, 0], 1e-15)
        inv_e = 2.0 / b_cut[:, None, None]
        Y_prev = U
        Y_curr = inv_e * GU - U
        del GU
        for _ in range(1, poly_deg):
            GY = G_run @ Y_curr
            Y_next = 2.0 * (inv_e * GY - Y_curr) - Y_prev
            del GY
            Y_prev = Y_curr
            Y_curr = Y_next
        del Y_prev
        theta, U, GU = _rayleigh_ritz_32(xp, G_run, Y_curr)
        del Y_curr

    ev_k = xp.maximum(theta[:, -max_k:][:, ::-1], 0.0)
    u_k = U[:, :, -max_k:][:, :, ::-1]
    gu_k = GU[:, :, -max_k:][:, :, ::-1]
    res_norm = xp.max(xp.linalg.norm(gu_k - u_k * ev_k[:, None, :], axis=1), axis=1)
    rel_res = _to_np(xp, res_norm / xp.maximum(ev_k[:, 0], 1e-12))
    converged_sub = np.isfinite(rel_res) & (rel_res <= residual_tol)

    if np.all(can_chefsi) and np.all(converged_sub):
        return ev_k, u_k

    ev_out = xp.zeros((B, max_k), dtype=xp.float64)
    u_out = xp.zeros((B, n_s, max_k), dtype=xp.float64)
    conv_global = np.zeros(B, dtype=bool)
    if np.all(can_chefsi):
        conv_global = converged_sub
        if np.any(converged_sub):
            ok_xp = xp.asarray(np.flatnonzero(converged_sub))
            ev_out[ok_xp] = ev_k[ok_xp]
            u_out[ok_xp] = u_k[ok_xp]
    else:
        c_host = np.flatnonzero(can_chefsi)
        conv_global[c_host] = converged_sub
        if np.any(converged_sub):
            sub_ok = xp.asarray(np.flatnonzero(converged_sub))
            glob_ok = xp.asarray(c_host[converged_sub])
            ev_out[glob_ok] = ev_k[sub_ok]
            u_out[glob_ok] = u_k[sub_ok]

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
    # Keep bucket_step=32 for small batches to minimize cuSOLVER launch overhead.
    bucket_step = 32 if B <= 256 else 8
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
        u_grp = xp.zeros((b_idx.size, n_s, max_k), dtype=xp.float64)
        u_grp[xp.arange(b_idx.size)[:, None], idx, :k] = u[:, :, -k:][:, :, ::-1]
        del ev, u
        u_out[b_idx] = u_grp
        del u_grp
    return ev_out, u_out


def _top_k_eigh(
    xp: Any,
    G_batch: Any,
    max_k: int,
    sw: Any,
    Q0_comb: Any | None = None,
    *,
    is_symmetric: bool = False,
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
    T_safe = xp.where(inputs_finite[:, None, None], T_k, eye_k[None, :, :])

    d_w, q_w = xp.linalg.eigh(W_safe)
    d_w = xp.maximum(d_w, 0.0)
    tau_raw = xp.diagonal(T_safe, axis1=-2, axis2=-1).sum(axis=-1) / float(k)
    tau_valid = inputs_finite & xp.isfinite(tau_raw) & (tau_raw > float(eigenvalue_tolerance))
    tau_safe = xp.where(tau_valid, tau_raw, 1.0)

    T_tilde = xp.swapaxes(q_w, -1, -2) @ T_safe @ q_w
    T_tilde = 0.5 * (T_tilde + xp.swapaxes(T_tilde, -1, -2))
    if grid_mode:
        denom = d_w[:, None, :] + lam[None, :, None] * tau_safe[:, None, None]
        denom_ok = xp.all(xp.isfinite(denom) & (denom > 1e-18), axis=-1)
        valid_out = tau_valid[:, None] & denom_ok
        inv_sqrt = 1.0 / xp.sqrt(xp.maximum(denom, 1e-18))
        M_batch = T_tilde[:, None, :, :] * (inv_sqrt[:, :, :, None] * inv_sqrt[:, :, None, :])
        M_batch = 0.5 * (M_batch + xp.swapaxes(M_batch, -1, -2))
        _, U_batch = xp.linalg.eigh(M_batch)
        vecs = q_w[:, None, :, :] @ (inv_sqrt[:, :, :, None] * U_batch[:, :, :, ::-1])
        return xp.where(valid_out[:, :, None, None], vecs, xp.nan)

    denom = d_w + (lam * tau_safe)[:, None]
    denom_ok = xp.all(xp.isfinite(denom) & (denom > 1e-18), axis=-1)
    valid_out = tau_valid & denom_ok
    inv_sqrt = 1.0 / xp.sqrt(xp.maximum(denom, 1e-18))
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
