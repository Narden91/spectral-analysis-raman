"""Batched closed-form evaluation of multiplicity bootstrap draws on CPU (NumPy) or GPU (CuPy)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from src.analysis.paired_lipid._backend import (
    _get_gpu_xp as _get_gpu_xp,
)
from src.analysis.paired_lipid._backend import (
    _probe_cupy as _probe_cupy,
)
from src.analysis.paired_lipid._backend import (
    _to_np as _to_np,
)
from src.analysis.paired_lipid._backend import (
    resolve_array_backend as _resolve_array_backend_impl,
)
from src.analysis.paired_lipid._backend import (
    suggest_batch_size as suggest_batch_size,
)
from src.analysis.paired_lipid._batched_models import (
    batched_division_prep as batched_division_prep,
)
from src.analysis.paired_lipid._batched_models import (
    batched_paired_schur_sse as batched_paired_schur_sse,
)
from src.analysis.paired_lipid._batched_models import (
    batched_pca_schur_sse as batched_pca_schur_sse,
)
from src.analysis.paired_lipid._batched_models import (
    batched_pls_sse_and_preds as batched_pls_sse_and_preds,
)
from src.analysis.paired_lipid._batched_models import (
    batched_ridge_sse_and_preds as batched_ridge_sse_and_preds,
)
from src.analysis.paired_lipid._eigensolvers import (
    _chefsi_top_k_gpu as _chefsi_top_k_gpu,
)
from src.analysis.paired_lipid._eigensolvers import (
    _exact_top_k_eigh_active as _exact_top_k_eigh_active,
)
from src.analysis.paired_lipid._eigensolvers import (
    _fast_top1_eigvalsh as _fast_top1_eigvalsh,
)
from src.analysis.paired_lipid._eigensolvers import (
    _rayleigh_ritz_32 as _rayleigh_ritz_32,
)
from src.analysis.paired_lipid._eigensolvers import (
    _top_k_eigh as _top_k_eigh,
)
from src.analysis.paired_lipid._eigensolvers import (
    _whitened_generalized_eigh as _whitened_generalized_eigh,
)
from src.analysis.paired_lipid._eigensolvers import (
    batched_solve_directions_grid as batched_solve_directions_grid,
)
from src.analysis.paired_lipid._gram_cache import (
    _get_dataset_gram_cache as _get_dataset_gram_cache,
)
from src.analysis.paired_lipid._gram_cache import (
    _precompute_division_constants as _precompute_division_constants,
)
from src.analysis.paired_lipid.cohort import CohortDataset, PairedLipidConfig


def resolve_array_backend(
    use_gpu: bool | None,
    n_draws: int,
    *,
    min_auto_draws: int = 256,
    gpu_probe: Callable[[], Any | None] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Select CuPy or NumPy backend while honoring monkeypatches to ``batched._get_gpu_xp``."""
    probe = _get_gpu_xp if gpu_probe is None else gpu_probe
    return _resolve_array_backend_impl(use_gpu, n_draws, min_auto_draws=min_auto_draws, gpu_probe=probe)


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
    if (
        gcache["has_missing_cov"]
        or list(cfg.retained_directions) != [1, 2, 3]
        or min(cfg.pca_dimensions) < max(cfg.retained_directions)
    ):
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
        eligible &= np.sum(M_all[:, tr_idx] > 0, axis=1) >= max(3, max_pls_c + 2)
        pair_otr = p_pair[np.isin(p_pair, tr_idx)]
        n_pair_otr = np.sum(M_all[:, pair_otr] > 0, axis=1)
        eligible &= n_pair_otr >= (min_inner_pairs + 1)
        eligible &= np.sum(M_all[:, pair_otr], axis=1) > 1.0

        min_pair_itr = np.full(n_b, 999999, dtype=int)
        for in_tr_rel, in_val_rel in in_splits:
            in_tr = tr_idx[in_tr_rel]
            in_val = tr_idx[in_val_rel]
            if in_tr.size - 2 < max_pls_c:
                return np.zeros(n_b, dtype=bool)
            eligible &= np.sum(M_all[:, in_tr] > 0, axis=1) >= max(3, max_pls_c + 2)
            eligible &= np.sum(M_all[:, in_val] > 0, axis=1) >= 1
            pair_itr = p_pair[np.isin(p_pair, in_tr)]
            n_pair_itr = np.sum(M_all[:, pair_itr] > 0, axis=1)
            eligible &= n_pair_itr >= min_inner_pairs
            eligible &= np.sum(M_all[:, pair_itr], axis=1) > 1.0
            min_pair_itr = np.minimum(min_pair_itr, n_pair_itr)
        eligible &= n_pair_otr > min_pair_itr

    return eligible


def batched_select_inner_settings(
    xp: Any,
    gcache: dict[str, Any],
    in_dconsts: list[dict[str, Any]],
    M_batch: Any,
    y_all_xp: Any,
    alphas_xp: Any,
    part_band_xp: Any,
    cfg: PairedLipidConfig,
) -> dict[str, Any]:
    """Stream inner folds one-at-a-time to select hyperparameters for B draws simultaneously.

    Streaming each inner fold without retaining all ``inner_divs`` simultaneously cuts inner-CV
    peak device/host memory by ``n_inner_splits`` (~3x) while keeping identical operations.
    """
    B = int(M_batch.shape[0])
    n_alphas = int(alphas_xp.shape[0])
    k_list = [int(k) for k in cfg.pca_dimensions]
    lam_list = list(cfg.lambda_grid)
    r_list = list(cfg.retained_directions)
    max_pls_c = max(cfg.pls_component_grid)
    pls_idx = np.asarray(sorted({int(c) - 1 for c in cfg.pls_component_grid}), dtype=int)
    n_k, n_lam, n_r = len(k_list), len(lam_list), len(r_list)

    total_val_w = xp.zeros(B, dtype=xp.float64)
    cov_sse = xp.zeros((B, n_alphas), dtype=xp.float64)
    band_sse = xp.zeros((B, n_alphas), dtype=xp.float64)
    pca_sse = xp.zeros((B, n_k, n_alphas), dtype=xp.float64)
    pls_sse = xp.zeros((B, max_pls_c), dtype=xp.float64)
    pair_sse = xp.zeros((B, n_r, n_k * n_lam, n_alphas), dtype=xp.float64)
    rank_ok_xp = xp.ones(B, dtype=bool)

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
        rank_ok_xp &= div["rank_ok"]
        total_val_w += xp.sum(div["w_val"], axis=1)
        cov_sse += div["sse_0"]

        sse_b, _ = batched_ridge_sse_and_preds(
            xp,
            div["X_cov_tr"],
            div["X_cov_val"],
            part_band_xp[in_tr],
            part_band_xp[in_val],
            div["y_tr"],
            div["y_val"],
            div["w_tr"],
            div["w_val"],
            alphas_xp,
        )
        band_sse += sse_b

        pca_sse += batched_pca_schur_sse(xp, div, k_list, alphas_xp)

        sse_pls, _ = batched_pls_sse_and_preds(xp, div, max_pls_c)
        pls_sse += sse_pls

        V_grid = batched_solve_directions_grid(
            xp,
            div["W_max"],
            div["T_max"],
            k_list,
            lam_list,
            eigenvalue_tolerance=cfg.eigenvalue_tolerance,
        )
        pair_sse += batched_paired_schur_sse(xp, div, V_grid, k_list, alphas_xp)
        del div, V_grid

    cov_mses = _to_np(xp, cov_sse / total_val_w[:, None])
    band_mses = _to_np(xp, band_sse / total_val_w[:, None])
    pca_mses = _to_np(xp, pca_sse / total_val_w[:, None, None])
    pls_mses = _to_np(xp, pls_sse / total_val_w[:, None])
    pair_mses = _to_np(xp, pair_sse / total_val_w[:, None, None, None])
    rank_ok = _to_np(xp, rank_ok_xp) > 0.5

    b_arange = np.arange(B)
    best_cov_a = _argmin_last(np.round(cov_mses, 12))

    best_band_a = _argmin_last(np.round(band_mses, 12))
    best_band_mse = np.round(band_mses[b_arange, best_band_a], 12)

    pca_r = np.round(pca_mses, 12)
    best_pca_k_idx, best_pca_a = _select_pca_setting(pca_r)
    best_pca_mse = pca_r[b_arange, best_pca_k_idx, best_pca_a]

    pls_r_cand = np.round(pls_mses[:, pls_idx], 12)
    best_pls_c_idx = pls_idx[np.argmin(pls_r_cand, axis=1)]
    best_pls_mse = np.round(pls_mses[b_arange, best_pls_c_idx], 12)

    comp_stack = np.stack([best_band_mse, best_pca_mse, best_pls_mse], axis=1)
    best_comp_idx = np.argmin(comp_stack, axis=1)

    pair_r = np.round(pair_mses.reshape(B, n_r, n_k, n_lam, n_alphas), 12)
    pair_finite = np.isfinite(pair_r)
    pair_ok = np.any(pair_finite, axis=(1, 2, 3, 4))
    pair_r_safe = np.where(pair_finite, pair_r, np.inf)
    best_r_idx, best_k_idx, best_pair_l_idx, best_pair_a = _select_paired_setting(pair_r_safe)
    return {
        "best_cov_a": best_cov_a,
        "best_band_a": best_band_a,
        "best_band_mse": best_band_mse,
        "best_pca_k_idx": best_pca_k_idx,
        "best_pca_a": best_pca_a,
        "best_pca_mse": best_pca_mse,
        "best_pls_c_idx": best_pls_c_idx,
        "best_pls_mse": best_pls_mse,
        "best_comp_idx": best_comp_idx,
        "best_r_idx": best_r_idx,
        "best_k_idx": best_k_idx,
        "best_pair_l_idx": best_pair_l_idx,
        "best_pair_a": best_pair_a,
        "rank_ok": rank_ok & pair_ok,
    }


def _predict_grouped_ridge_outer(
    xp: Any,
    outer_div: dict[str, Any],
    b_idx: np.ndarray,
    Z_otr: Any,
    Z_ote: Any,
    alpha_indices: np.ndarray,
    alphas_xp: Any,
    out_preds: Any,
) -> None:
    """Fit outer ridge across all alphas for a draw subset and gather selected predictions on device."""
    if b_idx.size == 0:
        return
    b_xp = xp.asarray(b_idx)
    _, pr = batched_ridge_sse_and_preds(
        xp,
        outer_div["X_cov_tr"][b_xp],
        outer_div["X_cov_val"][b_xp],
        Z_otr,
        Z_ote,
        outer_div["y_tr"],
        outer_div["y_val"],
        outer_div["w_tr"][b_xp],
        outer_div["w_val"][b_xp],
        alphas_xp,
    )
    out_preds[b_xp] = pr[xp.arange(b_idx.size), :, xp.asarray(alpha_indices)]


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
    fallback_draws_out: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate multiplicity bootstrap draws in batches on GPU (CuPy) or CPU (NumPy)."""
    n_sub, n_p = M_sub.shape
    xp, backend_meta = resolve_array_backend(use_gpu, n_sub, gpu_probe=_get_gpu_xp)
    if execution_metadata is not None:
        execution_metadata.update(
            device=backend_meta["device"],
            gpu_requested=backend_meta["gpu_requested"],
            gpu_fallback=backend_meta["gpu_fallback"],
        )
        if "gpu_fallback_reason" in backend_meta:
            execution_metadata["gpu_fallback_reason"] = backend_meta["gpu_fallback_reason"]

    y_all_xp = xp.asarray(y_all, dtype=xp.float64)
    alphas_xp = xp.asarray(list(cfg.ridge_alpha_grid), dtype=xp.float64)
    k_list = [int(k) for k in cfg.pca_dimensions]
    lam_xp = xp.asarray(list(cfg.lambda_grid), dtype=xp.float64)
    r_list = [int(r) for r in cfg.retained_directions]
    max_pls_c = max(cfg.pls_component_grid)
    part_band_xp = xp.asarray(gcache["part_band_mat"], dtype=xp.float64)

    fold_dconsts = []
    for (tr_idx, te_idx), in_splits in zip(outer_splits, inner_splits_per_fold):
        in_dconsts = [
            _precompute_division_constants(xp, gcache, dataset, tr_idx[in_tr_rel], tr_idx[in_val_rel], cfg)
            for in_tr_rel, in_val_rel in in_splits
        ]
        out_dconst = _precompute_division_constants(xp, gcache, dataset, tr_idx, te_idx, cfg)
        fold_dconsts.append((tr_idx, te_idx, in_dconsts, out_dconst))

    if batch_size is None:
        n_samples_outer = max((int(out_dc["K1_div"].shape[0]) for _, _, _, out_dc in fold_dconsts), default=n_p)
        batch_size = suggest_batch_size(xp, n_sub, n_samples_outer)

    oof_cov = xp.zeros((n_sub, n_p), dtype=xp.float64)
    oof_paired = xp.zeros((n_sub, n_p), dtype=xp.float64)
    oof_strongest = xp.zeros((n_sub, n_p), dtype=xp.float64)

    for b_start in range(0, n_sub, batch_size):
        b_end = min(b_start + batch_size, n_sub)
        B = b_end - b_start
        M_batch = xp.asarray(M_sub[b_start:b_end], dtype=xp.float64)
        batch_ok = np.ones(B, dtype=bool)
        b_arange_xp = xp.arange(B)

        for tr_idx, te_idx, in_dconsts, out_dconst in fold_dconsts:
            sel = batched_select_inner_settings(
                xp, gcache, in_dconsts, M_batch, y_all_xp, alphas_xp, part_band_xp, cfg
            )
            batch_ok &= sel["rank_ok"]

            best_cov_a = sel["best_cov_a"]
            best_band_a = sel["best_band_a"]
            best_pca_k_idx = sel["best_pca_k_idx"]
            best_pca_a = sel["best_pca_a"]
            best_pls_c_idx = sel["best_pls_c_idx"]
            best_comp_idx = sel["best_comp_idx"]
            best_r_idx = sel["best_r_idx"]
            best_k_idx = sel["best_k_idx"]
            best_pair_l_idx = sel["best_pair_l_idx"]
            best_pair_a = sel["best_pair_a"]

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
            batch_ok &= _to_np(xp, outer_div["rank_ok"]) > 0.5
            pca_otr = outer_div["part_pca_max"][:, tr_idx, :]
            pca_ote = outer_div["part_pca_max"][:, te_idx, :]

            # Solve outer paired generalized eigensystem only for valid draws' selected (k, lam).
            V_outer_sel: dict[int, Any] = {}
            b_k_by_k: dict[int, np.ndarray] = {}
            for k_idx, k in enumerate(k_list):
                b_k = np.flatnonzero(batch_ok & (best_k_idx == k_idx))
                if b_k.size == 0:
                    continue
                b_k_xp = xp.asarray(b_k)
                lam_sel = lam_xp[xp.asarray(best_pair_l_idx[b_k])]
                V_k = _whitened_generalized_eigh(
                    xp,
                    outer_div["W_max"][b_k_xp, :k, :k],
                    outer_div["T_max"][b_k_xp, :k, :k],
                    lam_sel,
                    grid_mode=False,
                    eigenvalue_tolerance=cfg.eigenvalue_tolerance,
                )
                v_finite = _to_np(xp, xp.all(xp.isfinite(V_k), axis=(-2, -1))) > 0.5
                if not np.all(v_finite):
                    batch_ok[b_k[~v_finite]] = False
                V_outer_sel[k_idx] = V_k
                b_k_by_k[k_idx] = b_k

            pred_paired_ote = xp.zeros((B, te_idx.size), dtype=xp.float64)
            for r_idx, r in enumerate(r_list):
                mask_b = np.flatnonzero(batch_ok & (best_r_idx == r_idx))
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
                    b_k_all = b_k_by_k[k_idx]
                    pos_in_bk = xp.asarray(np.searchsorted(b_k_all, b_ids))
                    V_sel = V_outer_sel[k_idx][pos_in_bk, :, :r]
                    sub_m_xp = xp.asarray(sub_m)
                    Z_otr_sel[sub_m_xp] = pca_otr[b_ids_xp, :, :k] @ V_sel
                    Z_ote_sel[sub_m_xp] = pca_ote[b_ids_xp, :, :k] @ V_sel

                _predict_grouped_ridge_outer(
                    xp, outer_div, mask_b, Z_otr_sel, Z_ote_sel, best_pair_a[mask_b], alphas_xp, pred_paired_ote
                )

            pred_comp_sel = xp.zeros((B, te_idx.size), dtype=xp.float64)
            b_band = np.flatnonzero(batch_ok & (best_comp_idx == 0))
            _predict_grouped_ridge_outer(
                xp,
                outer_div,
                b_band,
                part_band_xp[tr_idx],
                part_band_xp[te_idx],
                best_band_a[b_band],
                alphas_xp,
                pred_comp_sel,
            )

            for k_idx, k in enumerate(k_list):
                b_pca = np.flatnonzero(batch_ok & (best_comp_idx == 1) & (best_pca_k_idx == k_idx))
                if b_pca.size > 0:
                    b_pca_xp = xp.asarray(b_pca)
                    _predict_grouped_ridge_outer(
                        xp,
                        outer_div,
                        b_pca,
                        pca_otr[b_pca_xp, :, :k],
                        pca_ote[b_pca_xp, :, :k],
                        best_pca_a[b_pca],
                        alphas_xp,
                        pred_comp_sel,
                    )

            b_pls = np.flatnonzero(batch_ok & (best_comp_idx == 2))
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
                pred_comp_sel[b_pls_xp] = pr_pls[xp.arange(b_pls.size), :, xp.asarray(best_pls_c_idx[b_pls])]

            te_xp = xp.asarray(te_idx)
            oof_cov[b_start:b_end, te_xp] = outer_div["preds_cov"][b_arange_xp, :, xp.asarray(best_cov_a)]
            oof_paired[b_start:b_end, te_xp] = pred_paired_ote
            oof_strongest[b_start:b_end, te_xp] = pred_comp_sel
            del outer_div

        if not np.all(batch_ok):
            bad_local = np.flatnonzero(~batch_ok)
            oof_paired[xp.asarray(b_start + bad_local)] = xp.nan
            if fallback_draws_out is not None:
                fallback_draws_out.update(int(b_start + idx) for idx in bad_local)

    M_sub_xp = xp.asarray(M_sub, dtype=xp.float64)
    w_norm_all = M_sub_xp / xp.sum(M_sub_xp, axis=1, keepdims=True)
    diff_cov = y_all_xp[None, :] - oof_cov
    diff_pair = y_all_xp[None, :] - oof_paired
    diff_comp = y_all_xp[None, :] - oof_strongest
    mse_cov = xp.sum(w_norm_all * (diff_cov * diff_cov), axis=1)
    mse_paired = xp.sum(w_norm_all * (diff_pair * diff_pair), axis=1)
    mse_strongest = xp.sum(w_norm_all * (diff_comp * diff_comp), axis=1)
    delta_pair = _to_np(xp, mse_cov - mse_paired)
    delta_comp = _to_np(xp, mse_strongest - mse_paired)
    if fallback_draws_out is not None:
        bad_final = np.flatnonzero(~(np.isfinite(delta_pair) & np.isfinite(delta_comp)))
        fallback_draws_out.update(int(idx) for idx in bad_final)
    return delta_pair, delta_comp
