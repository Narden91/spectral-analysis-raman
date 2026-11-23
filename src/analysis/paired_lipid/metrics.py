"""Repeatability, discriminability, subspace principal-angle, and endpoint prediction metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg, spatial

from src.analysis.paired_lipid.cohort import CohortDataset, PairedLipidConfig
from src.analysis.paired_lipid.models import (
    CovariatePreprocessor,
    MultiAlphaRidgeResult,
    WeightedPLSResult,
    WeightedSpectralPreprocessor,
)

_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def two_way_anova_ms(matrix: np.ndarray) -> tuple[float, float, float]:
    """Compute two-way ANOVA mean squares ``(ms_row, ms_col, ms_err)`` for a 2D array."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"two_way_anova_ms requires a 2D matrix, got shape {arr.shape}.")
    n_rows, n_cols = arr.shape
    if n_rows == 0 or n_cols == 0:
        return float("nan"), float("nan"), float("nan")

    grand_mean = float(np.mean(arr))
    centered = arr - grand_mean
    row_means = np.mean(centered, axis=1)
    col_means = np.mean(centered, axis=0)

    ss_row = float(n_cols * np.sum(row_means * row_means))
    ss_col = float(n_rows * np.sum(col_means * col_means))
    ss_tot = float(np.sum(centered * centered))
    ss_err = max(0.0, ss_tot - ss_row - ss_col)

    ms_row = ss_row / float(n_rows - 1) if n_rows >= 2 else float("nan")
    ms_col = ss_col / float(n_cols - 1) if n_cols >= 2 else float("nan")
    ms_err = ss_err / float((n_rows - 1) * (n_cols - 1)) if (n_rows >= 2 and n_cols >= 2) else float("nan")
    return ms_row, ms_col, ms_err


def compute_icc_a1(y_a: np.ndarray, y_b: np.ndarray) -> dict[str, Any]:
    """Two-way mixed-effects, single-measure, absolute-agreement ICC(A,1) per Koo & Li (2016).

    Formula:
        ICC(A,1) = (MS_R - MS_E) / (MS_R + (k - 1)*MS_E + (k / n)*(MS_C - MS_E))
    with k = 2 fixed positions and n random participants. Zero and negative estimates
    are retained; marked non-estimable only for n < 2 or non-positive/degenerate denominator.
    """
    a = np.asarray(y_a, dtype=float)
    b = np.asarray(y_b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    n = int(a.size)
    if n < 2:
        return {
            "icc_a1": float("nan"),
            "estimable": False,
            "reason": "fewer_than_two_paired_participants",
            "n_pairs": n,
            "mean_diff": float("nan"),
            "rmsd": float("nan"),
            "ms_between_participants": float("nan"),
            "ms_between_positions": float("nan"),
            "ms_error": float("nan"),
        }

    k = 2
    ms_r, ms_c, ms_e = two_way_anova_ms(np.column_stack([a, b]))
    denom = ms_r + (k - 1) * ms_e + (float(k) / float(n)) * (ms_c - ms_e)
    diffs = a - b
    mean_diff = float(np.mean(diffs))
    rmsd = float(np.sqrt(np.mean(diffs * diffs)))

    max_ms = max(ms_r, ms_c, ms_e)
    if not np.isfinite(denom) or denom <= 0.0 or denom <= 1e-15 * max_ms:
        return {
            "icc_a1": float("nan"),
            "estimable": False,
            "reason": "degenerate_denominator",
            "n_pairs": n,
            "mean_diff": mean_diff,
            "rmsd": rmsd,
            "ms_between_participants": ms_r,
            "ms_between_positions": ms_c,
            "ms_error": ms_e,
        }

    icc = float((ms_r - ms_e) / denom)
    return {
        "icc_a1": icc,
        "estimable": True,
        "reason": "ok",
        "n_pairs": n,
        "mean_diff": mean_diff,
        "rmsd": rmsd,
        "ms_between_participants": ms_r,
        "ms_between_positions": ms_c,
        "ms_error": ms_e,
    }


def compute_discriminability_and_retrieval(
    z_a: np.ndarray,
    z_b: np.ndarray,
    labels_a: np.ndarray | None = None,
    labels_b: np.ndarray | None = None,
    *,
    local_window: int = 5,
) -> dict[str, float]:
    """Compute within-fold Bridgeford discriminability and bidirectional retrieval accuracy."""
    n = int(z_a.shape[0])
    if n < 2:
        return {
            "n_pairs": float(n),
            "discriminability": float("nan"),
            "retrieval_top1": float("nan"),
            "retrieval_chance": float("nan"),
            "global_candidate_size": float(n),
            "local_retrieval_top1": float("nan"),
            "local_retrieval_chance": float("nan"),
            "mean_local_candidate_size": float("nan"),
        }

    Z = np.vstack([z_a, z_b])  # 2n x r
    dist_mat = spatial.distance.cdist(Z, Z, metric="euclidean")

    # Within-participant distances: d(iA, iB) = dist_mat[i, n + i]
    idx_n = np.arange(n)
    within_d = dist_mat[idx_n, idx_n + n]

    # Discriminability across all 2n samples: compare each sample's partner distance against all 2(n-1) others.
    d_same_all = np.concatenate([within_d, within_d])[:, None]
    other_mask = np.ones((2 * n, 2 * n), dtype=bool)
    other_mask[idx_n, idx_n] = False
    other_mask[idx_n, idx_n + n] = False
    other_mask[idx_n + n, idx_n] = False
    other_mask[idx_n + n, idx_n + n] = False
    wins = float(
        np.sum((d_same_all < dist_mat) & other_mask)
        + 0.5 * np.sum((d_same_all == dist_mat) & other_mask)
    )
    total_comparisons = 2 * n * (2 * n - 2)
    discrim = wins / float(total_comparisons) if total_comparisons > 0 else float("nan")

    # Bidirectional cross-position retrieval (A -> B and B -> A) with fractional tie credit.
    cross_ab = dist_mat[:n, n:]  # n x n, diagonal is true match
    diag_d = np.diag(cross_ab)
    min_ab = np.min(cross_ab, axis=1)
    hits_ab = np.where(
        diag_d == min_ab,
        1.0 / np.maximum(np.sum(cross_ab == min_ab[:, None], axis=1), 1),
        0.0,
    )
    min_ba = np.min(cross_ab, axis=0)
    hits_ba = np.where(
        diag_d == min_ba,
        1.0 / np.maximum(np.sum(cross_ab == min_ba[None, :], axis=0), 1),
        0.0,
    )
    retrieval_top1 = float(0.5 * (np.mean(hits_ab) + np.mean(hits_ba)))
    retrieval_chance = 1.0 / float(n)

    # Local neighbor-label candidate sensitivity over finite-label pairs only, vectorized bidirectionally.
    local_retrieval_top1 = float("nan")
    local_retrieval_chance = float("nan")
    mean_local_candidate_size = float("nan")
    if labels_a is not None and labels_b is not None:
        la = np.asarray(labels_a, dtype=float)
        lb = np.asarray(labels_b, dtype=float)
        # Require both paired sample labels to be finite so unlabelled samples are excluded from local rank windows.
        mean_lbl = 0.5 * (la + lb)
        finite_idx = np.flatnonzero(np.isfinite(mean_lbl))
        m = int(finite_idx.size)
        if m >= 2:
            order = np.argsort(mean_lbl[finite_idx], kind="stable")
            rank_pos = np.empty(m, dtype=int)
            rank_pos[order] = np.arange(m)
            cand_mask = np.abs(rank_pos[:, None] - rank_pos[None, :]) <= local_window
            cand_counts = np.sum(cand_mask, axis=1)
            valid_q = cand_counts >= 2
            if np.any(valid_q):
                cross_sub = cross_ab[np.ix_(finite_idx, finite_idx)]
                diag_sub = diag_d[finite_idx]
                masked_ab = np.where(cand_mask, cross_sub, np.inf)

                # Because cand_mask is symmetric, row i holds A->B candidates and col i holds B->A candidates.
                min_loc_ab = np.min(masked_ab, axis=1)
                hits_loc_ab = np.where(
                    diag_sub == min_loc_ab,
                    1.0 / np.maximum(np.sum(masked_ab == min_loc_ab[:, None], axis=1), 1),
                    0.0,
                )
                min_loc_ba = np.min(masked_ab, axis=0)
                hits_loc_ba = np.where(
                    diag_sub == min_loc_ba,
                    1.0 / np.maximum(np.sum(masked_ab == min_loc_ba[None, :], axis=0), 1),
                    0.0,
                )
                local_retrieval_top1 = float(
                    0.5 * (np.mean(hits_loc_ab[valid_q]) + np.mean(hits_loc_ba[valid_q]))
                )
                local_retrieval_chance = float(np.mean(1.0 / cand_counts[valid_q].astype(float)))
                mean_local_candidate_size = float(np.mean(cand_counts[valid_q].astype(float)))

    return {
        "n_pairs": float(n),
        "discriminability": discrim,
        "retrieval_top1": retrieval_top1,
        "retrieval_chance": retrieval_chance,
        "global_candidate_size": float(n),
        "local_retrieval_top1": local_retrieval_top1,
        "local_retrieval_chance": local_retrieval_chance,
        "mean_local_candidate_size": mean_local_candidate_size,
    }


def map_subspace_to_metric_orthonormal_basis(
    spec_prep: WeightedSpectralPreprocessor,
    pca_k: int,
    eigvecs_V: np.ndarray,
    p_1300: int,
    p_2850: int,
) -> np.ndarray:
    """Map paired directions back to canonical channels and orthonormalize under declared metric.

    Declared metric: equal block weight (1/2 per block) and equal channel weight within each block
    (1 / (2 * p_b)). Returns Q of shape (p_1300 + p_2850, r) with Q^T Q = I_r in metric coordinates.
    """
    pca_loadings_k = spec_prep.pca_components[:pca_k].T  # (p_total, k)
    proj = pca_loadings_k @ eigvecs_V  # (p_total, r)
    if proj.shape[0] != p_1300 + p_2850:
        raise ValueError(
            f"Projected basis row count {proj.shape[0]} does not match p_1300 + p_2850 ({p_1300 + p_2850})."
        )
    scale_1300 = spec_prep.scale_1300 / math.sqrt(2.0 * max(p_1300, 1))
    scale_2850 = spec_prep.scale_2850 / math.sqrt(2.0 * max(p_2850, 1))
    metric_dirs = np.empty_like(proj, dtype=float)
    metric_dirs[:p_1300] = proj[:p_1300] * scale_1300
    metric_dirs[p_1300:] = proj[p_1300:] * scale_2850
    Q, _ = linalg.qr(metric_dirs, mode="economic")
    return Q


def _ensure_orthonormal_basis(basis: np.ndarray) -> np.ndarray:
    """Return an orthonormal column basis, reusing ``basis`` directly when already orthonormal."""
    arr = np.asarray(basis, dtype=float)
    if arr.size == 0:
        return arr
    gram = arr.T @ arr
    if np.allclose(gram, np.eye(arr.shape[1], dtype=float), rtol=0.0, atol=1e-12):
        return arr
    return linalg.orth(arr)


def _principal_angles_from_orthonormal(QA: np.ndarray, QB: np.ndarray) -> np.ndarray:
    """Compute exact Bjorck-Golub principal angles (radians) between orthonormal bases QA and QB."""
    if QA.shape[1] < QB.shape[1]:
        A, B = QB, QA
    else:
        A, B = QA, QB
    M = A.T @ B
    sv = linalg.svdvals(M)
    angles = np.arccos(np.clip(sv, 0.0, 1.0))
    small_mask = sv > _INV_SQRT2
    if np.any(small_mask):
        # Refine angles below pi/4 via sine of the orthogonal residual B - A (A^T B).
        sv_sin = linalg.svdvals(B - A @ M)[::-1]
        angles = np.where(small_mask, np.arcsin(np.clip(sv_sin, 0.0, 1.0)), angles)
    return angles


def compute_subspace_principal_angles(
    bases: Sequence[np.ndarray],
    *,
    model_name: str | None = None,
) -> pd.DataFrame:
    """Compute pairwise principal angles (degrees) between fold-specific metric-orthonormal bases."""
    orth_bases = [_ensure_orthonormal_basis(b) for b in bases]
    rows: list[dict[str, Any]] = []
    for i in range(len(bases)):
        Qi_orig = bases[i]
        Qi = orth_bases[i]
        if Qi_orig.size == 0 or Qi.size == 0:
            continue
        for j in range(i + 1, len(bases)):
            Qj_orig = bases[j]
            Qj = orth_bases[j]
            if Qj_orig.size == 0 or Qj.size == 0:
                continue
            angles_rad = _principal_angles_from_orthonormal(Qi, Qj)
            cosines = np.clip(np.cos(angles_rad), 0.0, 1.0)
            angles_deg = np.sort(np.degrees(angles_rad))
            row: dict[str, Any] = {}
            if model_name is not None:
                row["model"] = str(model_name)
            row.update(
                {
                    "fold_a": i,
                    "fold_b": j,
                    "dim_a": int(Qi_orig.shape[1]),
                    "dim_b": int(Qj_orig.shape[1]),
                    "angle_1_deg": float(angles_deg[0]) if angles_deg.size >= 1 else float("nan"),
                    "angle_2_deg": float(angles_deg[1]) if angles_deg.size >= 2 else float("nan"),
                    "angle_3_deg": float(angles_deg[2]) if angles_deg.size >= 3 else float("nan"),
                    "max_principal_angle_deg": float(np.max(angles_deg)),
                    "mean_principal_angle_deg": float(np.mean(angles_deg)),
                    "min_cosine": float(np.min(cosines)),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def compute_endpoint_summary(
    y_true: np.ndarray,
    y_pred_cov: np.ndarray,
    y_pred_model: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute participant-weighted held-out LDL-C prediction metrics and descriptive calibration."""
    y = np.asarray(y_true, dtype=float)
    yc = np.asarray(y_pred_cov, dtype=float)
    ym = np.asarray(y_pred_model, dtype=float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    valid = np.isfinite(y) & np.isfinite(yc) & np.isfinite(ym) & (w > 0.0)
    if not np.any(valid):
        return {
            "delta_mse": float("nan"),
            "oof_loss_diff_sd": float("nan"),
            "relative_mse_reduction": float("nan"),
            "mse_covariate": float("nan"),
            "mse_model": float("nan"),
            "rmse_model": float("nan"),
            "mae_model": float("nan"),
            "calibration_intercept": float("nan"),
            "calibration_slope": float("nan"),
        }

    y, yc, ym, w = y[valid], yc[valid], ym[valid], w[valid]
    w_norm = w / np.sum(w)

    sq_err_c = (y - yc) ** 2
    sq_err_m = (y - ym) ** 2
    loss_diff = sq_err_c - sq_err_m
    mse_c = float(w_norm @ sq_err_c)
    mse_m = float(w_norm @ sq_err_m)
    delta = float(w_norm @ loss_diff)
    oof_loss_diff_sd = float(math.sqrt(max(float(w_norm @ ((loss_diff - delta) ** 2)), 0.0)))
    rel_red = delta / mse_c if mse_c > 1e-15 else float("nan")
    rmse = float(math.sqrt(mse_m))
    mae = float(w_norm @ np.abs(y - ym))

    ym_bar = float(w_norm @ ym)
    y_bar = float(w_norm @ y)
    ym_c = ym - ym_bar
    var_ym = float(w_norm @ (ym_c * ym_c))
    if var_ym > 1e-12:
        slope = float((w_norm @ (ym_c * (y - y_bar))) / var_ym)
        intercept = y_bar - slope * ym_bar
    else:
        slope = float("nan")
        intercept = float("nan")

    return {
        "delta_mse": delta,
        "oof_loss_diff_sd": oof_loss_diff_sd,
        "relative_mse_reduction": rel_red,
        "mse_covariate": mse_c,
        "mse_model": mse_m,
        "rmse_model": rmse,
        "mae_model": mae,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def _spectral_scale_metrics(
    y_a: np.ndarray,
    y_b: np.ndarray,
    s_a: np.ndarray,
    s_b: np.ndarray,
) -> tuple[float, float]:
    """Compute participant-mean spectral contribution SD and share of full-prediction variance."""
    y_mean = 0.5 * (np.asarray(y_a, dtype=float) + np.asarray(y_b, dtype=float))
    s_mean = 0.5 * (np.asarray(s_a, dtype=float) + np.asarray(s_b, dtype=float))
    valid = np.isfinite(y_mean) & np.isfinite(s_mean)
    if int(np.count_nonzero(valid)) < 2:
        return float("nan"), float("nan")
    y_v = y_mean[valid]
    s_v = s_mean[valid]
    var_s = float(np.var(s_v, ddof=1))
    var_y = float(np.var(y_v, ddof=1))
    sd_s = float(math.sqrt(max(var_s, 0.0)))
    share = float(var_s / var_y) if var_y > 1e-15 else float("nan")
    return sd_s, share


def _predict_paired_ab(
    fit: MultiAlphaRidgeResult,
    X_cov: np.ndarray,
    Z_a: np.ndarray,
    Z_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate paired A/B predictions while computing the shared covariate linear term once."""
    b0 = float(fit.intercepts[0])
    b_cov = fit.coef_cov[0]
    b_spec = fit.coef_spec[0]
    cov_part = (b0 + X_cov @ b_cov) if X_cov.shape[1] > 0 else np.full(X_cov.shape[0], b0, dtype=float)
    if Z_a.shape[1] > 0:
        spec_a = ((Z_a - fit.z_mean) / fit.z_scale) @ b_spec
        spec_b = ((Z_b - fit.z_mean) / fit.z_scale) @ b_spec
    else:
        spec_a = np.zeros(X_cov.shape[0], dtype=float)
        spec_b = np.zeros(X_cov.shape[0], dtype=float)
    return cov_part + spec_a, spec_a, cov_part + spec_b, spec_b


def outer_test_repeatability(
    dataset: CohortDataset,
    cfg: PairedLipidConfig,
    outer_test_idx: np.ndarray,
    outer_cov_prep: CovariatePreprocessor,
    fit_paired_outer: MultiAlphaRidgeResult,
    fit_band_outer: MultiAlphaRidgeResult,
    sample_paired_z: np.ndarray,
    sample_band_mat: np.ndarray,
    sample_pca_sel: np.ndarray,
    *,
    sample_band_discrim: np.ndarray | None = None,
    fit_pca_outer: MultiAlphaRidgeResult | None = None,
    fit_pls_outer: WeightedPLSResult | None = None,
    outer_spec_prep: WeightedSpectralPreprocessor | None = None,
    strongest_comparator_name: str | None = None,
) -> dict[str, Any]:
    """Per-sample (A vs B) ICCs, scale diagnostics, and discriminability on paired outer-test participants."""
    paired_test_pos = np.isin(dataset.paired_participant_indices, outer_test_idx)
    paired_test_p = dataset.paired_participant_indices[paired_test_pos]
    idx_a_te = dataset.pair_sample_a_indices[paired_test_pos]
    idx_b_te = dataset.pair_sample_b_indices[paired_test_pos]

    info: dict[str, Any] = {"n_paired_test": int(paired_test_p.size)}
    if paired_test_p.size < 2:
        return info

    X_cov_pair_te = outer_cov_prep.transform(
        dataset.covariate_frame, paired_test_p, raw_cache=dataset._cache
    )
    y_pair_a, s_pair_a, y_pair_b, s_pair_b = _predict_paired_ab(
        fit_paired_outer, X_cov_pair_te, sample_paired_z[idx_a_te], sample_paired_z[idx_b_te]
    )
    y_band_a, s_band_a, y_band_b, s_band_b = _predict_paired_ab(
        fit_band_outer, X_cov_pair_te, sample_band_mat[idx_a_te], sample_band_mat[idx_b_te]
    )
    lbl_a = dataset.sample_parsed_labels[idx_a_te]
    lbl_b = dataset.sample_parsed_labels[idx_b_te]
    band_discrim_input = sample_band_mat if sample_band_discrim is None else sample_band_discrim

    def _discrim(z: np.ndarray) -> dict[str, float]:
        z_arr = np.asarray(z, dtype=float)
        if z_arr.ndim == 1:
            z_arr = z_arr[:, None]
        return compute_discriminability_and_retrieval(
            z_arr[idx_a_te], z_arr[idx_b_te], lbl_a, lbl_b, local_window=cfg.local_candidate_window
        )

    paired_full_icc = compute_icc_a1(y_pair_a, y_pair_b)
    paired_spec_icc = compute_icc_a1(s_pair_a, s_pair_b)
    paired_sd_s, paired_var_share = _spectral_scale_metrics(y_pair_a, y_pair_b, s_pair_a, s_pair_b)

    band_full_icc = compute_icc_a1(y_band_a, y_band_b)
    band_spec_icc = compute_icc_a1(s_band_a, s_band_b)
    band_sd_s, band_var_share = _spectral_scale_metrics(y_band_a, y_band_b, s_band_a, s_band_b)

    info.update(
        {
            "paired_full_icc": paired_full_icc,
            "paired_spectral_icc": paired_spec_icc,
            "paired_sd_spectral_contrib": paired_sd_s,
            "paired_spectral_variance_share": paired_var_share,
            "fixed_band_full_icc": band_full_icc,
            "fixed_band_spectral_icc": band_spec_icc,
            "fixed_band_sd_spectral_contrib": band_sd_s,
            "fixed_band_spectral_variance_share": band_var_share,
            "paired_discriminability": _discrim(sample_paired_z),
            "pca_discriminability": _discrim(sample_pca_sel),
            "band_discriminability": _discrim(band_discrim_input),
        }
    )

    comp_icc_map: dict[str, tuple[dict[str, Any], dict[str, Any], float, float]] = {
        "fixed_band_ridge": (band_full_icc, band_spec_icc, band_sd_s, band_var_share),
    }

    if fit_pca_outer is not None:
        y_pca_a, s_pca_a, y_pca_b, s_pca_b = _predict_paired_ab(
            fit_pca_outer, X_cov_pair_te, sample_pca_sel[idx_a_te], sample_pca_sel[idx_b_te]
        )
        pca_full_icc = compute_icc_a1(y_pca_a, y_pca_b)
        pca_spec_icc = compute_icc_a1(s_pca_a, s_pca_b)
        pca_sd_s, pca_var_share = _spectral_scale_metrics(y_pca_a, y_pca_b, s_pca_a, s_pca_b)
        info["pca_full_icc"] = pca_full_icc
        info["pca_spectral_icc"] = pca_spec_icc
        info["pca_sd_spectral_contrib"] = pca_sd_s
        info["pca_spectral_variance_share"] = pca_var_share
        comp_icc_map["pca_ridge"] = (pca_full_icc, pca_spec_icc, pca_sd_s, pca_var_share)

    if fit_pls_outer is not None and outer_spec_prep is not None:
        n_comp_pls = int(fit_pls_outer.max_components)
        comp_col = min(max(n_comp_pls, 1), fit_pls_outer.max_components) - 1
        y_pls_a_all, s_pls_a_all = fit_pls_outer.predict_from_raw_blocks(
            outer_spec_prep,
            X_cov_pair_te,
            dataset.sample_1300[idx_a_te],
            dataset.sample_2850[idx_a_te],
            n_components=None,
        )
        y_pls_b_all, s_pls_b_all = fit_pls_outer.predict_from_raw_blocks(
            outer_spec_prep,
            X_cov_pair_te,
            dataset.sample_1300[idx_b_te],
            dataset.sample_2850[idx_b_te],
            n_components=None,
        )
        y_pls_a = y_pls_a_all[:, comp_col]
        s_pls_a = s_pls_a_all[:, comp_col]
        y_pls_b = y_pls_b_all[:, comp_col]
        s_pls_b = s_pls_b_all[:, comp_col]
        pls_full_icc = compute_icc_a1(y_pls_a, y_pls_b)
        pls_spec_icc = compute_icc_a1(s_pls_a, s_pls_b)
        pls_sd_s, pls_var_share = _spectral_scale_metrics(y_pls_a, y_pls_b, s_pls_a, s_pls_b)
        info["pls_full_icc"] = pls_full_icc
        info["pls_spectral_icc"] = pls_spec_icc
        info["pls_sd_spectral_contrib"] = pls_sd_s
        info["pls_spectral_variance_share"] = pls_var_share
        comp_icc_map["pls_regression"] = (pls_full_icc, pls_spec_icc, pls_sd_s, pls_var_share)

        # Sample-level cumulative PLS spectral contributions across 1..c components for discriminability.
        info["pls_discriminability"] = compute_discriminability_and_retrieval(
            s_pls_a_all, s_pls_b_all, lbl_a, lbl_b, local_window=cfg.local_candidate_window
        )

    if strongest_comparator_name is not None and strongest_comparator_name in comp_icc_map:
        sc_full, sc_spec, sc_sd, sc_share = comp_icc_map[strongest_comparator_name]
        info["strongest_comparator_name"] = strongest_comparator_name
        info["strongest_comparator_full_icc"] = sc_full
        info["strongest_comparator_spectral_icc"] = sc_spec
        info["strongest_comparator_sd_spectral_contrib"] = sc_sd
        info["strongest_comparator_spectral_variance_share"] = sc_share

    return info
