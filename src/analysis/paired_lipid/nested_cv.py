"""Nested participant-level cross-validation, repeatability metrics and the multiplicity bootstrap."""

from __future__ import annotations
from typing import Any, NamedTuple, Sequence
import numpy as np
import pandas as pd
from scipy import linalg, stats

from src.analysis.paired_lipid.cohort import (
    CohortDataset,
    PairedLipidConfig,
    aggregate_samples_to_participants,
    extract_sample_band_matrix,
)
from src.analysis.paired_lipid.models import (
    CovariatePreprocessor,
    PairedMoments,
    WeightedPLSResult,
    WeightedSpectralPreprocessor,
    fit_ridge_grid,
    predict_ridge_grid,
)
from src.analysis.paired_lipid.batched import (
    _argmin_last,
    _check_fast_batch_eligibility,
    _get_dataset_gram_cache,
    _precompute_division_constants,
    _run_batched_multiplicity_draws,
    _select_paired_setting,
    _select_pca_setting,
    batched_division_prep,
    batched_paired_schur_sse,
    batched_pls_sse_and_preds,
    batched_ridge_sse_and_preds,
    batched_solve_directions_grid,
)


def make_participant_splits(
    n_participants: int,
    paired_mask: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create deterministic participant-grouped folds stratified by pair availability."""
    rng = np.random.default_rng(seed)
    paired_idx = np.flatnonzero(paired_mask)
    single_idx = np.flatnonzero(~paired_mask)
    paired_perm = rng.permutation(paired_idx)
    single_perm = rng.permutation(single_idx)

    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for i, idx in enumerate(paired_perm):
        folds[i % n_splits].append(int(idx))
    for i, idx in enumerate(single_perm):
        folds[(i + paired_perm.size) % n_splits].append(int(idx))

    all_idx = np.arange(n_participants, dtype=int)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for f in folds:
        test_idx = np.asarray(sorted(f), dtype=int)
        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=True)
        splits.append((train_idx, test_idx))
    return splits


def make_nested_splits(
    dataset: CohortDataset, cfg: PairedLipidConfig, seed: int
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[list[tuple[np.ndarray, np.ndarray]]]]:
    """Locked outer folds and, per outer fold f, inner folds seeded with seed + 1000 * (f + 1)."""
    n_p = dataset.n_participants
    paired_mask = np.isin(np.arange(n_p), dataset.paired_participant_indices)
    outer = make_participant_splits(n_p, paired_mask, cfg.n_outer_splits, seed)
    inner = [
        make_participant_splits(tr_idx.size, paired_mask[tr_idx], cfg.n_inner_splits, seed + 1000 * (fold_idx + 1))
        for fold_idx, (tr_idx, _) in enumerate(outer)
    ]
    return outer, inner


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
        }

    Y = np.column_stack([a, b])
    k = 2
    grand_mean = float(np.mean(Y))
    row_means = np.mean(Y, axis=1)
    col_means = np.mean(Y, axis=0)

    ss_r = float(k * np.sum((row_means - grand_mean) ** 2))
    ss_c = float(n * np.sum((col_means - grand_mean) ** 2))
    ss_tot = float(np.sum((Y - grand_mean) ** 2))
    ss_e = max(0.0, ss_tot - ss_r - ss_c)

    ms_r = ss_r / float(n - 1)
    ms_c = ss_c / float(k - 1)
    ms_e = ss_e / float((n - 1) * (k - 1))

    denom = ms_r + (k - 1) * ms_e + (float(k) / float(n)) * (ms_c - ms_e)
    diffs = a - b
    mean_diff = float(np.mean(diffs))
    rmsd = float(np.sqrt(np.mean(diffs**2)))

    if not np.isfinite(denom) or denom <= 1e-15:
        return {
            "icc_a1": float("nan"),
            "estimable": False,
            "reason": "degenerate_denominator",
            "n_pairs": n,
            "mean_diff": mean_diff,
            "rmsd": rmsd,
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
            "local_retrieval_top1": float("nan"),
            "local_retrieval_chance": float("nan"),
        }

    Z = np.vstack([z_a, z_b])  # 2n x r
    diffs = Z[:, None, :] - Z[None, :, :]
    dist_mat = np.linalg.norm(diffs, axis=2)

    # Within-participant distances: d(iA, iB) = dist_mat[i, n + i]
    within_d = dist_mat[np.arange(n), np.arange(n) + n]

    # Discriminability across all 2n samples: for each sample s of participant i,
    # compare its distance to its partner s' against distances to all 2(n-1) other samples
    Wins = 0.0
    total_comparisons = 0
    for i in range(n):
        d_same = within_d[i]
        other_mask = np.ones(2 * n, dtype=bool)
        other_mask[i] = False
        other_mask[n + i] = False
        for anchor_idx in (i, n + i):
            d_others = dist_mat[anchor_idx, other_mask]
            Wins += float(np.sum(d_same < d_others) + 0.5 * np.sum(d_same == d_others))
            total_comparisons += int(d_others.size)

    discrim = Wins / float(total_comparisons) if total_comparisons > 0 else float("nan")

    # Bidirectional cross-position retrieval (A -> B and B -> A)
    cross_ab = dist_mat[:n, n:]  # n x n, diagonal is true match
    diag_d = np.diag(cross_ab)
    # Fractional tie credit if minimum distance is shared
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

    # Local neighbor-label candidate sensitivity, bidirectional like the global retrieval
    local_hits: list[float] = []
    local_chances: list[float] = []
    if labels_a is not None and labels_b is not None:
        mean_lbl = 0.5 * (np.asarray(labels_a, dtype=float) + np.asarray(labels_b, dtype=float))
        order = np.argsort(np.where(np.isfinite(mean_lbl), mean_lbl, np.inf))
        rank_pos = np.empty(n, dtype=int)
        rank_pos[order] = np.arange(n)
        for i in range(n):
            cand = np.flatnonzero(np.abs(rank_pos - rank_pos[i]) <= local_window)
            if cand.size < 2:
                continue
            for d_cand in (cross_ab[i, cand], cross_ab[cand, i]):  # A -> B, then B -> A
                m_d = np.min(d_cand)
                local_hits.append(float(1.0 / np.sum(d_cand == m_d)) if diag_d[i] == m_d else 0.0)
            local_chances.append(1.0 / float(cand.size))

    return {
        "n_pairs": float(n),
        "discriminability": discrim,
        "retrieval_top1": retrieval_top1,
        "retrieval_chance": retrieval_chance,
        "local_retrieval_top1": float(np.mean(local_hits)) if local_hits else float("nan"),
        "local_retrieval_chance": float(np.mean(local_chances)) if local_chances else float("nan"),
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
    # Raw channel projection matrix: (p_1300 + p_2850, r)
    pca_loadings_k = spec_prep.pca_components[:pca_k].T  # (p_total, k)
    block_scales = np.concatenate(
        [
            np.full(p_1300, spec_prep.scale_1300, dtype=float),
            np.full(p_2850, spec_prep.scale_2850, dtype=float),
        ]
    )
    raw_dirs = block_scales[:, None] * (pca_loadings_k @ eigvecs_V)

    metric_sqrt = np.concatenate(
        [
            np.full(p_1300, 1.0 / np.sqrt(2.0 * max(p_1300, 1)), dtype=float),
            np.full(p_2850, 1.0 / np.sqrt(2.0 * max(p_2850, 1)), dtype=float),
        ]
    )
    metric_dirs = metric_sqrt[:, None] * raw_dirs
    Q, _ = linalg.qr(metric_dirs, mode="economic")
    return Q


def compute_subspace_principal_angles(bases: Sequence[np.ndarray]) -> pd.DataFrame:
    """Compute pairwise principal angles (degrees) between fold-specific metric-orthonormal bases."""
    rows: list[dict[str, Any]] = []
    for i in range(len(bases)):
        for j in range(i + 1, len(bases)):
            Qi, Qj = bases[i], bases[j]
            if Qi.size == 0 or Qj.size == 0:
                continue
            angles_rad = linalg.subspace_angles(Qi, Qj)
            cosines = np.clip(np.cos(angles_rad), 0.0, 1.0)
            angles_deg = np.degrees(angles_rad)
            rows.append(
                {
                    "fold_a": i,
                    "fold_b": j,
                    "dim_a": int(Qi.shape[1]),
                    "dim_b": int(Qj.shape[1]),
                    "max_principal_angle_deg": float(np.max(angles_deg)),
                    "mean_principal_angle_deg": float(np.mean(angles_deg)),
                    "min_cosine": float(np.min(cosines)),
                }
            )
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
    valid = np.isfinite(y) & np.isfinite(yc) & np.isfinite(ym) & (w > 0)
    if not np.any(valid):
        return {
            "delta_mse": float("nan"),
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
    mse_c = float(np.sum(w_norm * sq_err_c))
    mse_m = float(np.sum(w_norm * sq_err_m))
    delta = float(np.sum(w_norm * (sq_err_c - sq_err_m)))
    rel_red = delta / mse_c if mse_c > 1e-15 else float("nan")
    rmse = float(np.sqrt(mse_m))
    mae = float(np.sum(w_norm * np.abs(y - ym)))

    ym_bar = float(np.sum(w_norm * ym))
    y_bar = float(np.sum(w_norm * y))
    var_ym = float(np.sum(w_norm * ((ym - ym_bar) ** 2)))
    if var_ym > 1e-12:
        slope = float(np.sum(w_norm * (ym - ym_bar) * (y - y_bar)) / var_ym)
        intercept = y_bar - slope * ym_bar
    else:
        slope = float("nan")
        intercept = float("nan")

    return {
        "delta_mse": delta,
        "relative_mse_reduction": rel_red,
        "mse_covariate": mse_c,
        "mse_model": mse_m,
        "rmse_model": rmse,
        "mae_model": mae,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


class _InnerSelection(NamedTuple):
    """Inner-CV choices for one outer fold; alpha entries index cfg.ridge_alpha_grid."""

    cov_alpha_idx: int
    band_alpha_idx: int
    band_mse: float
    pca_setting: tuple[int, int]
    pca_mse: float
    pls_components: int
    pls_mse: float
    paired_setting: tuple[int, float, int, int]
    invalid_candidates: list[dict[str, Any]]


def _select_inner_fast(
    dataset: CohortDataset,
    y_all: np.ndarray,
    outer_train_idx: np.ndarray,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    mult_all: np.ndarray,
    cfg: PairedLipidConfig,
) -> _InnerSelection:
    """Select every comparator's settings with the closed-form batched engine (one draw)."""
    alphas = list(cfg.ridge_alpha_grid)
    n_alphas = len(alphas)
    max_pls_c = max(cfg.pls_component_grid)
    gcache = _get_dataset_gram_cache(dataset, cfg)
    M_1 = mult_all[None, :]
    alphas_np = np.asarray(alphas, dtype=np.float64)
    k_list = [int(k) for k in cfg.pca_dimensions]
    lam_list = list(cfg.lambda_grid)
    r_list = list(cfg.retained_directions)
    n_k, n_lam, n_r = len(k_list), len(lam_list), len(r_list)
    part_band_np = gcache["part_band_mat"]

    inner_divs = []
    total_val_weight = 0.0
    for in_tr_rel, in_val_rel in inner_splits:
        in_tr = outer_train_idx[in_tr_rel]
        in_val = outer_train_idx[in_val_rel]
        dconst = _precompute_division_constants(np, gcache, dataset, in_tr, in_val, cfg)
        div = batched_division_prep(
            np,
            gcache,
            dconst,
            M_1[:, in_tr],
            M_1[:, in_val],
            y_all,
            alphas_np,
            cfg,
            compute_pca_suff=True,
        )
        inner_divs.append(div)
        total_val_weight += float(np.sum(div["w_val"]))

    cov_sse = sum(div["sse_0"][0] for div in inner_divs)
    cov_mses = cov_sse / total_val_weight
    best_cov_alpha_idx = int(_argmin_last(np.round(cov_mses, 12)))

    band_sse = np.zeros(n_alphas, dtype=np.float64)
    for div in inner_divs:
        Zb_tr = part_band_np[None, div["tr_p"], :]
        Zb_val = part_band_np[None, div["val_p"], :]
        sse_b, _ = batched_ridge_sse_and_preds(
            np,
            div["X_cov_tr"],
            div["X_cov_val"],
            Zb_tr,
            Zb_val,
            div["y_tr"],
            div["y_val"],
            div["w_tr"],
            div["w_val"],
            alphas_np,
        )
        band_sse += sse_b[0]
    band_mses = band_sse / total_val_weight
    best_band_alpha_idx = int(_argmin_last(np.round(band_mses, 12)))
    best_band_mse = float(band_mses[best_band_alpha_idx])

    pca_sse = np.zeros((n_k, n_alphas), dtype=np.float64)
    for k_idx, k in enumerate(k_list):
        for div in inner_divs:
            Zp_tr = div["part_pca_max"][:, div["tr_p"], :k]
            Zp_val = div["part_pca_max"][:, div["val_p"], :k]
            sse_p, _ = batched_ridge_sse_and_preds(
                np,
                div["X_cov_tr"],
                div["X_cov_val"],
                Zp_tr,
                Zp_val,
                div["y_tr"],
                div["y_val"],
                div["w_tr"],
                div["w_val"],
                alphas_np,
            )
            pca_sse[k_idx] += sse_p[0]
    pca_mses = pca_sse / total_val_weight
    pca_r = np.round(pca_mses, 12)
    k_sel, a_sel = _select_pca_setting(pca_r[None])
    best_pca_k_idx, best_pca_a = int(k_sel[0]), int(a_sel[0])
    best_pca_setting = (int(k_list[best_pca_k_idx]), int(best_pca_a))
    best_pca_mse = float(pca_r[best_pca_k_idx, best_pca_a])

    pls_sse = np.zeros(max_pls_c, dtype=np.float64)
    for div in inner_divs:
        sse_pls, _ = batched_pls_sse_and_preds(np, div, max_pls_c)
        pls_sse += sse_pls[0]
    pls_mses = pls_sse / total_val_weight
    best_pls_c = int(np.argmin(np.round(pls_mses, 12))) + 1
    best_pls_mse = float(pls_mses[best_pls_c - 1])

    pair_sse = np.zeros((n_r, n_k * n_lam, n_alphas), dtype=np.float64)
    for div in inner_divs:
        V_grid = batched_solve_directions_grid(np, div["W_max"], div["T_max"], k_list, lam_list)
        pair_sse += batched_paired_schur_sse(np, div, V_grid, k_list, alphas_np)[0]
    pair_mses = pair_sse / total_val_weight
    sel = _select_paired_setting(np.round(pair_mses.reshape(1, n_r, n_k, n_lam, n_alphas), 12))
    best_r_idx, best_k_idx, best_pair_l_idx, best_pair_a = (int(s[0]) for s in sel)
    best_paired_setting: tuple[int, float, int, int] | None = (
        int(k_list[best_k_idx]),
        float(lam_list[best_pair_l_idx]),
        int(r_list[best_r_idx]),
        int(best_pair_a),
    )
    return _InnerSelection(
        best_cov_alpha_idx,
        best_band_alpha_idx,
        best_band_mse,
        best_pca_setting,
        best_pca_mse,
        best_pls_c,
        best_pls_mse,
        best_paired_setting,
        [],
    )


def _select_inner_sequential(
    dataset: CohortDataset,
    y_all: np.ndarray,
    outer_train_idx: np.ndarray,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    mult_all: np.ndarray,
    cfg: PairedLipidConfig,
    outer_spec_prep: WeightedSpectralPreprocessor,
    *,
    single_sample_downstream: bool,
    include_band_paired: bool,
) -> _InnerSelection:
    """Select every comparator's settings by refitting each candidate model on each inner division."""
    alphas = list(cfg.ridge_alpha_grid)
    n_alphas = len(alphas)
    max_pls_c = max(cfg.pls_component_grid)
    invalid_candidates: list[dict[str, Any]] = []
    sample_band_mat, _ = extract_sample_band_matrix(dataset)
    part_band_mat = aggregate_samples_to_participants(
        sample_band_mat, dataset, single_sample_only=single_sample_downstream
    )
    # Precompute inner division preprocessors once per inner fold
    inner_cache: list[dict[str, Any]] = []
    for in_tr_rel, in_val_rel in inner_splits:
        in_tr = outer_train_idx[in_tr_rel]
        in_val = outer_train_idx[in_val_rel]
        w_tr = mult_all[in_tr]
        w_val = mult_all[in_val]
        if np.sum(w_tr > 0) < 3 or np.sum(w_val > 0) < 1:
            raise ValueError("Inner division lacks sufficient distinct weighted participants.")

        cov_prep = CovariatePreprocessor.fit(
            dataset.covariate_frame, in_tr, w_tr, preset=cfg.covariate_preset, raw_cache=dataset._cache
        )
        X_cov_tr = cov_prep.transform(dataset.covariate_frame, in_tr, raw_cache=dataset._cache)
        X_cov_val = cov_prep.transform(dataset.covariate_frame, in_val, raw_cache=dataset._cache)

        spec_prep = WeightedSpectralPreprocessor.fit(dataset, in_tr, w_tr, cfg=cfg)
        sample_pca_max = spec_prep.transform_pca(dataset.sample_1300, dataset.sample_2850)
        part_pca_max = aggregate_samples_to_participants(
            sample_pca_max, dataset, single_sample_only=single_sample_downstream
        )
        paired_moments = PairedMoments.from_sample_coordinates(sample_pca_max, dataset, in_tr, w_tr)

        sample_scaled = spec_prep.transform_scaled_blocks(dataset.sample_1300, dataset.sample_2850)
        part_scaled = aggregate_samples_to_participants(
            sample_scaled, dataset, single_sample_only=single_sample_downstream
        )

        entry: dict[str, Any] = {
            "in_tr": in_tr,
            "in_val": in_val,
            "w_tr": w_tr,
            "w_val": w_val,
            "y_tr": y_all[in_tr],
            "y_val": y_all[in_val],
            "X_cov_tr": X_cov_tr,
            "X_cov_val": X_cov_val,
            "spec_prep": spec_prep,
            "part_pca_max": part_pca_max,
            "paired_moments": paired_moments,
            "part_scaled": part_scaled,
        }
        if include_band_paired:
            entry["band_moments"] = PairedMoments.from_sample_coordinates(
                sample_band_mat, dataset, in_tr, w_tr
            )
        inner_cache.append(entry)

    total_val_weight = sum(float(np.sum(ic["w_val"])) for ic in inner_cache)

    # 1. Covariate-only ridge inner selection
    cov_sse = np.zeros(n_alphas, dtype=float)
    for ic in inner_cache:
        fit_c = fit_ridge_grid(ic["X_cov_tr"], None, ic["y_tr"], ic["w_tr"], alphas)
        preds_c, _ = predict_ridge_grid(fit_c, ic["X_cov_val"], None)
        cov_sse += np.sum(ic["w_val"][:, None] * ((ic["y_val"][:, None] - preds_c) ** 2), axis=0)
    cov_mses = cov_sse / total_val_weight
    # Tie rule: stronger regularization (larger alpha) wins exact ties
    best_cov_alpha_idx = int(np.lexsort((-np.arange(n_alphas), np.round(cov_mses, 12)))[0])

    # 2. Fixed band summaries + ridge inner selection
    band_sse = np.zeros(n_alphas, dtype=float)
    for ic in inner_cache:
        fit_b = fit_ridge_grid(
            ic["X_cov_tr"],
            part_band_mat[ic["in_tr"]],
            ic["y_tr"],
            ic["w_tr"],
            alphas,
        )
        preds_b, _ = predict_ridge_grid(fit_b, ic["X_cov_val"], part_band_mat[ic["in_val"]])
        band_sse += np.sum(ic["w_val"][:, None] * ((ic["y_val"][:, None] - preds_b) ** 2), axis=0)
    band_mses = band_sse / total_val_weight
    best_band_alpha_idx = int(np.lexsort((-np.arange(n_alphas), np.round(band_mses, 12)))[0])
    best_band_mse = float(band_mses[best_band_alpha_idx])

    # Determine common valid PCA dimensions across inner and outer divisions
    valid_k_sets = [set(ic["spec_prep"].candidate_pca_dims(cfg)) for ic in inner_cache]
    valid_k_sets.append(set(outer_spec_prep.candidate_pca_dims(cfg)))
    candidate_k = sorted(set.intersection(*valid_k_sets)) if valid_k_sets else []
    if not candidate_k:
        raise ValueError("No valid candidate PCA dimensions available across training divisions.")

    # 3. Block-scaled PCA + ridge inner selection
    best_pca_key: tuple[float, int, float] | None = None
    best_pca_setting = (candidate_k[0], 0)
    for k in candidate_k:
        pca_sse_1d = np.zeros(n_alphas, dtype=float)
        for ic in inner_cache:
            Z_tr = ic["part_pca_max"][ic["in_tr"], :k]
            Z_val = ic["part_pca_max"][ic["in_val"], :k]
            fit_p = fit_ridge_grid(ic["X_cov_tr"], Z_tr, ic["y_tr"], ic["w_tr"], alphas)
            preds_p, _ = predict_ridge_grid(fit_p, ic["X_cov_val"], Z_val)
            pca_sse_1d += np.sum(ic["w_val"][:, None] * ((ic["y_val"][:, None] - preds_p) ** 2), axis=0)
        pca_mses_1d = pca_sse_1d / total_val_weight
        for a_idx, mse_val in enumerate(pca_mses_1d):
            # Tie rule: lower MSE, fewer PCA dimensions k, stronger ridge penalty alpha
            cand_key = (round(float(mse_val), 12), int(k), -float(alphas[a_idx]))
            if best_pca_key is None or cand_key < best_pca_key:
                best_pca_key = cand_key
                best_pca_setting = (int(k), int(a_idx))
    best_pca_mse = float(best_pca_key[0]) if best_pca_key is not None else float("inf")

    # 4. PLS regression inner selection
    pls_sse = np.zeros(max_pls_c, dtype=float)
    valid_pls_c = max_pls_c
    for ic in inner_cache:
        pls_fit = WeightedPLSResult.fit(
            ic["X_cov_tr"],
            ic["part_scaled"][ic["in_tr"]],
            ic["y_tr"],
            ic["w_tr"],
            max_pls_c,
        )
        valid_pls_c = min(valid_pls_c, pls_fit.max_components)
        preds_pls, _ = pls_fit.predict(ic["X_cov_val"], ic["part_scaled"][ic["in_val"]])
        pls_sse[:valid_pls_c] += np.sum(
            ic["w_val"][:, None] * ((ic["y_val"][:, None] - preds_pls[:, :valid_pls_c]) ** 2),
            axis=0,
        )
    pls_mses = pls_sse[:valid_pls_c] / total_val_weight
    best_pls_c = int(np.lexsort((np.arange(1, valid_pls_c + 1), np.round(pls_mses, 12)))[0]) + 1
    best_pls_mse = float(pls_mses[best_pls_c - 1])

    # 5. Paired linear representation + ridge inner selection
    best_paired_key: tuple[float, int, int, float, float] | None = None
    best_paired_setting = None

    for k in candidate_k:
        for lam in cfg.lambda_grid:
            # Solve generalized eigenproblem once per inner fold for (k, lam)
            fold_V: list[np.ndarray] = []
            failed_reason: str | None = None
            for ic in inner_cache:
                try:
                    V_k, _, _ = ic["paired_moments"].solve_directions(
                        k, lam, eigenvalue_tolerance=cfg.eigenvalue_tolerance
                    )
                    fold_V.append(V_k)
                except ValueError as exc:
                    failed_reason = str(exc)
                    break
            if failed_reason is not None:
                invalid_candidates.append({"k": k, "lambda": lam, "reason": failed_reason})
                continue

            for r in cfg.retained_directions:
                if r > k:
                    continue
                pair_sse_1d = np.zeros(n_alphas, dtype=float)
                for ic, V_k in zip(inner_cache, fold_V):
                    V_r = V_k[:, :r]
                    Z_tr = ic["part_pca_max"][ic["in_tr"], :k] @ V_r
                    Z_val = ic["part_pca_max"][ic["in_val"], :k] @ V_r
                    fit_pr = fit_ridge_grid(ic["X_cov_tr"], Z_tr, ic["y_tr"], ic["w_tr"], alphas)
                    preds_pr, _ = predict_ridge_grid(fit_pr, ic["X_cov_val"], Z_val)
                    pair_sse_1d += np.sum(
                        ic["w_val"][:, None] * ((ic["y_val"][:, None] - preds_pr) ** 2), axis=0
                    )
                pair_mses_1d = pair_sse_1d / total_val_weight
                for a_idx, mse_val in enumerate(pair_mses_1d):
                    # Exact tie rule: lowest MSE, fewer retained directions r, fewer PCA dims k,
                    # stronger lambda, stronger alpha
                    cand_key = (
                        round(float(mse_val), 12),
                        int(r),
                        int(k),
                        -float(lam),
                        -float(alphas[a_idx]),
                    )
                    if best_paired_key is None or cand_key < best_paired_key:
                        best_paired_key = cand_key
                        best_paired_setting = (int(k), float(lam), int(r), int(a_idx))

    if best_paired_setting is None:
        raise ValueError("No valid paired linear candidate found in inner selection.")

    return _InnerSelection(
        best_cov_alpha_idx,
        best_band_alpha_idx,
        best_band_mse,
        best_pca_setting,
        best_pca_mse,
        best_pls_c,
        best_pls_mse,
        best_paired_setting,
        invalid_candidates,
    )


def _select_inner_models_and_fit_outer(
    dataset: CohortDataset,
    y_all: np.ndarray,
    outer_train_idx: np.ndarray,
    outer_test_idx: np.ndarray,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    participant_multiplicities: np.ndarray,
    cfg: PairedLipidConfig,
    *,
    single_sample_downstream: bool = False,
    include_band_paired: bool = False,
    external: tuple[CohortDataset, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Fit inner CV selection and outer-fold models with memoized spectral SVD & multi-alpha ridge."""
    mult_all = np.asarray(participant_multiplicities, dtype=float)
    sample_band_mat, _ = extract_sample_band_matrix(dataset)
    part_band_mat = aggregate_samples_to_participants(
        sample_band_mat, dataset, single_sample_only=single_sample_downstream
    )
    outer_w_tr = mult_all[outer_train_idx]
    outer_spec_prep = WeightedSpectralPreprocessor.fit(dataset, outer_train_idx, outer_w_tr, cfg=cfg)
    alphas = list(cfg.ridge_alpha_grid)
    gcache = _get_dataset_gram_cache(dataset, cfg)
    # The fast path evaluates every configured k unchanged. The eligibility check sends divisions
    # below the design's cap (paired training participants - 1) to the sequential path, which caps k.
    max_k = max(cfg.pca_dimensions)

    can_fast_inner = (
        not single_sample_downstream
        and not include_band_paired
        and not gcache["has_missing_cov"]
        and list(cfg.retained_directions) == [1, 2, 3]
        and outer_spec_prep.numerical_rank >= max_k
        and bool(
            _check_fast_batch_eligibility(
                dataset,
                [(outer_train_idx, outer_test_idx)],
                [inner_splits],
                mult_all[None, :],
                cfg,
                gcache,
            )[0]
        )
    )

    if can_fast_inner:
        selection = _select_inner_fast(dataset, y_all, outer_train_idx, inner_splits, mult_all, cfg)
    else:
        selection = _select_inner_sequential(
            dataset,
            y_all,
            outer_train_idx,
            inner_splits,
            mult_all,
            cfg,
            outer_spec_prep,
            single_sample_downstream=single_sample_downstream,
            include_band_paired=include_band_paired,
        )
    (
        best_cov_alpha_idx,
        best_band_alpha_idx,
        best_band_mse,
        best_pca_setting,
        best_pca_mse,
        best_pls_c,
        best_pls_mse,
        best_paired_setting,
        invalid_candidates,
    ) = selection

    # Refit all selected models on the complete outer training division
    outer_cov_prep = CovariatePreprocessor.fit(
        dataset.covariate_frame, outer_train_idx, outer_w_tr, preset=cfg.covariate_preset, raw_cache=dataset._cache
    )
    X_cov_otr = outer_cov_prep.transform(dataset.covariate_frame, outer_train_idx, raw_cache=dataset._cache)
    X_cov_ote = outer_cov_prep.transform(dataset.covariate_frame, outer_test_idx, raw_cache=dataset._cache)
    y_otr = y_all[outer_train_idx]

    # Outer Covariate-only Ridge
    fit_cov_outer = fit_ridge_grid(X_cov_otr, None, y_otr, outer_w_tr, [alphas[best_cov_alpha_idx]])
    pred_cov_ote, _ = predict_ridge_grid(fit_cov_outer, X_cov_ote, None, alpha_idx=0)

    # Outer Fixed Band + Ridge
    fit_band_outer = fit_ridge_grid(
        X_cov_otr,
        part_band_mat[outer_train_idx],
        y_otr,
        outer_w_tr,
        [alphas[best_band_alpha_idx]],
    )
    pred_band_ote, spec_band_ote = predict_ridge_grid(
        fit_band_outer, X_cov_ote, part_band_mat[outer_test_idx], alpha_idx=0
    )

    # Outer PCA + Ridge
    sel_pca_k, sel_pca_a_idx = best_pca_setting
    sample_pca_outer = outer_spec_prep.transform_pca(dataset.sample_1300, dataset.sample_2850)
    part_pca_outer = aggregate_samples_to_participants(
        sample_pca_outer, dataset, single_sample_only=single_sample_downstream
    )
    fit_pca_outer = fit_ridge_grid(
        X_cov_otr,
        part_pca_outer[outer_train_idx, :sel_pca_k],
        y_otr,
        outer_w_tr,
        [alphas[sel_pca_a_idx]],
    )
    pred_pca_ote, spec_pca_ote = predict_ridge_grid(
        fit_pca_outer, X_cov_ote, part_pca_outer[outer_test_idx, :sel_pca_k], alpha_idx=0
    )

    # Outer PLS Regression
    sample_scaled_outer = outer_spec_prep.transform_scaled_blocks(dataset.sample_1300, dataset.sample_2850)
    part_scaled_outer = aggregate_samples_to_participants(
        sample_scaled_outer, dataset, single_sample_only=single_sample_downstream
    )
    fit_pls_outer = WeightedPLSResult.fit(
        X_cov_otr,
        part_scaled_outer[outer_train_idx],
        y_otr,
        outer_w_tr,
        best_pls_c,
    )
    pred_pls_ote, spec_pls_ote = fit_pls_outer.predict(
        X_cov_ote, part_scaled_outer[outer_test_idx], n_components=best_pls_c
    )

    # Outer Paired Linear + Ridge
    sel_k, sel_lam, sel_r, sel_a_idx = best_paired_setting
    outer_moments = PairedMoments.from_sample_coordinates(
        sample_pca_outer, dataset, outer_train_idx, outer_w_tr
    )
    V_outer_k, eigvals_outer, tau_outer = outer_moments.solve_directions(
        sel_k, sel_lam, eigenvalue_tolerance=cfg.eigenvalue_tolerance
    )
    V_outer_r = V_outer_k[:, :sel_r]
    sample_paired_z = sample_pca_outer[:, :sel_k] @ V_outer_r
    part_paired_z = aggregate_samples_to_participants(
        sample_paired_z, dataset, single_sample_only=single_sample_downstream
    )
    fit_paired_outer = fit_ridge_grid(
        X_cov_otr,
        part_paired_z[outer_train_idx],
        y_otr,
        outer_w_tr,
        [alphas[sel_a_idx]],
    )
    pred_paired_ote, spec_paired_ote = predict_ridge_grid(
        fit_paired_outer, X_cov_ote, part_paired_z[outer_test_idx], alpha_idx=0
    )

    # Select strongest established comparator using inner validation MSE only
    comp_inner_mses = {
        "fixed_band_ridge": best_band_mse,
        "pca_ridge": best_pca_mse,
        "pls_regression": best_pls_mse,
    }
    strongest_comp_name = min(comp_inner_mses, key=lambda name: comp_inner_mses[name])
    comp_preds_map = {
        "fixed_band_ridge": pred_band_ote,
        "pca_ridge": pred_pca_ote,
        "pls_regression": pred_pls_ote,
    }

    repeatability_info = _outer_test_repeatability(
        dataset,
        cfg,
        outer_test_idx,
        outer_cov_prep,
        fit_paired_outer,
        fit_band_outer,
        sample_paired_z,
        sample_band_mat,
        sample_pca_outer[:, :sel_pca_k],
    )

    subspace_basis = map_subspace_to_metric_orthonormal_basis(
        outer_spec_prep,
        sel_k,
        V_outer_r,
        dataset.wavenumbers_1300.size,
        dataset.wavenumbers_2850.size,
    )

    out = {
        "predictions": {
            "covariate_ridge": pred_cov_ote,
            "fixed_band_ridge": pred_band_ote,
            "pca_ridge": pred_pca_ote,
            "pls_regression": pred_pls_ote,
            "paired_linear_ridge": pred_paired_ote,
            "strongest_comparator": comp_preds_map[strongest_comp_name],
        },
        "spectral_contributions": {
            "fixed_band_ridge": spec_band_ote,
            "pca_ridge": spec_pca_ote,
            "pls_regression": spec_pls_ote,
            "paired_linear_ridge": spec_paired_ote,
        },
        "selected_settings": {
            "covariate_ridge_alpha": float(alphas[best_cov_alpha_idx]),
            "fixed_band_ridge_alpha": float(alphas[best_band_alpha_idx]),
            "pca_ridge_k": int(sel_pca_k),
            "pca_ridge_alpha": float(alphas[sel_pca_a_idx]),
            "pls_components": int(best_pls_c),
            "paired_k": int(sel_k),
            "paired_lambda": float(sel_lam),
            "paired_r": int(sel_r),
            "paired_alpha": float(alphas[sel_a_idx]),
            "strongest_comparator_name": strongest_comp_name,
            "n_paired_distinct_train": int(outer_moments.n_paired_distinct),
            "n_paired_multiplicity_train": float(outer_moments.n_paired_multiplicity),
            "numerical_rank_train": int(outer_spec_prep.numerical_rank),
            "mean_ab_shift_l2": float(linalg.norm(outer_moments.mean_ab_shift[:sel_k])),
            "tau": float(tau_outer),
            "top_rho": float(eigvals_outer[0]),
            "dropped_covariates": outer_cov_prep.dropped_columns,
            "n_invalid_candidates": len(invalid_candidates),
        },
        "repeatability": repeatability_info,
        "subspace_basis": subspace_basis,
        "sample_paired_z": sample_paired_z,
        "part_paired_z": part_paired_z,
    }

    if external is not None:
        # Score the already-fitted fold models on an independent cohort (calibration target); no refit.
        ext_ds, y_ext = external
        X_cov_ext = outer_cov_prep.transform(
            ext_ds.covariate_frame, np.arange(ext_ds.n_participants), raw_cache=ext_ds._cache
        )
        pred_cov_ext, _ = predict_ridge_grid(fit_cov_outer, X_cov_ext, None, alpha_idx=0)
        ext_sample_pca = outer_spec_prep.transform_pca(ext_ds.sample_1300, ext_ds.sample_2850)
        ext_part_z = aggregate_samples_to_participants(
            ext_sample_pca[:, :sel_k] @ V_outer_r, ext_ds, single_sample_only=single_sample_downstream
        )
        pred_pair_ext, _ = predict_ridge_grid(fit_paired_outer, X_cov_ext, ext_part_z, alpha_idx=0)
        out["external_mse"] = {
            "covariate_ridge": float(np.mean((y_ext - pred_cov_ext) ** 2)),
            "paired_linear_ridge": float(np.mean((y_ext - pred_pair_ext) ** 2)),
        }

    if include_band_paired:
        out["predictions"]["band_paired_ridge"] = _band_paired_ablation_predictions(
            dataset,
            cfg,
            sample_band_mat,
            outer_train_idx,
            outer_test_idx,
            outer_w_tr,
            X_cov_otr,
            X_cov_ote,
            y_otr,
            sel_lam,
            sel_r,
            alphas[sel_a_idx],
            single_sample_downstream=single_sample_downstream,
        )

    return out


def _outer_test_repeatability(
    dataset: CohortDataset,
    cfg: PairedLipidConfig,
    outer_test_idx: np.ndarray,
    outer_cov_prep: CovariatePreprocessor,
    fit_paired_outer: Any,
    fit_band_outer: Any,
    sample_paired_z: np.ndarray,
    sample_band_mat: np.ndarray,
    sample_pca_sel: np.ndarray,
) -> dict[str, Any]:
    """Per-sample (A vs B) ICCs and discriminability on the paired outer-test participants."""
    paired_test_pos = np.isin(dataset.paired_participant_indices, outer_test_idx)
    paired_test_p = dataset.paired_participant_indices[paired_test_pos]
    idx_a_te = dataset.pair_sample_a_indices[paired_test_pos]
    idx_b_te = dataset.pair_sample_b_indices[paired_test_pos]

    info: dict[str, Any] = {"n_paired_test": int(paired_test_p.size)}
    if paired_test_p.size < 2:
        return info
    X_cov_pair_te = outer_cov_prep.transform(dataset.covariate_frame, paired_test_p)
    y_pair_a, s_pair_a = predict_ridge_grid(fit_paired_outer, X_cov_pair_te, sample_paired_z[idx_a_te], alpha_idx=0)
    y_pair_b, s_pair_b = predict_ridge_grid(fit_paired_outer, X_cov_pair_te, sample_paired_z[idx_b_te], alpha_idx=0)
    y_band_a, s_band_a = predict_ridge_grid(fit_band_outer, X_cov_pair_te, sample_band_mat[idx_a_te], alpha_idx=0)
    y_band_b, s_band_b = predict_ridge_grid(fit_band_outer, X_cov_pair_te, sample_band_mat[idx_b_te], alpha_idx=0)
    lbl_a = dataset.sample_parsed_labels[idx_a_te]
    lbl_b = dataset.sample_parsed_labels[idx_b_te]

    def _discrim(z: np.ndarray) -> dict[str, float]:
        return compute_discriminability_and_retrieval(
            z[idx_a_te], z[idx_b_te], lbl_a, lbl_b, local_window=cfg.local_candidate_window
        )

    info.update(
        {
            "paired_full_icc": compute_icc_a1(y_pair_a, y_pair_b),
            "paired_spectral_icc": compute_icc_a1(s_pair_a, s_pair_b),
            "fixed_band_full_icc": compute_icc_a1(y_band_a, y_band_b),
            "fixed_band_spectral_icc": compute_icc_a1(s_band_a, s_band_b),
            "paired_discriminability": _discrim(sample_paired_z),
            "pca_discriminability": _discrim(sample_pca_sel),
            "band_discriminability": _discrim(sample_band_mat),
        }
    )
    return info


def _band_paired_ablation_predictions(
    dataset: CohortDataset,
    cfg: PairedLipidConfig,
    sample_band_mat: np.ndarray,
    outer_train_idx: np.ndarray,
    outer_test_idx: np.ndarray,
    outer_w_tr: np.ndarray,
    X_cov_otr: np.ndarray,
    X_cov_ote: np.ndarray,
    y_otr: np.ndarray,
    lam: float,
    r: int,
    alpha: float,
    *,
    single_sample_downstream: bool,
) -> np.ndarray:
    """Secondary band-input paired ablation; reuses lambda, r and alpha of the PCA-input model."""
    band_moments = PairedMoments.from_sample_coordinates(sample_band_mat, dataset, outer_train_idx, outer_w_tr)
    k_band = sample_band_mat.shape[1]
    V_b, _, _ = band_moments.solve_directions(k_band, lam, eigenvalue_tolerance=cfg.eigenvalue_tolerance)
    Z_b_part = aggregate_samples_to_participants(
        sample_band_mat @ V_b[:, : min(r, k_band)], dataset, single_sample_only=single_sample_downstream
    )
    fit_bp = fit_ridge_grid(X_cov_otr, Z_b_part[outer_train_idx], y_otr, outer_w_tr, [alpha])
    pred_bp, _ = predict_ridge_grid(fit_bp, X_cov_ote, Z_b_part[outer_test_idx], alpha_idx=0)
    return pred_bp


def evaluate_nested_cohort(
    dataset: CohortDataset,
    *,
    cfg: PairedLipidConfig = PairedLipidConfig(),
    endpoint: str | None = None,
    outer_seed: int | None = None,
    participant_multiplicities: np.ndarray | None = None,
    precomputed_splits: tuple[
        list[tuple[np.ndarray, np.ndarray]],
        list[list[tuple[np.ndarray, np.ndarray]]],
    ]
    | None = None,
    single_sample_downstream: bool = False,
    include_band_paired: bool = False,
    external_test: CohortDataset | None = None,
    allow_missing_outcome: bool = False,
) -> dict[str, Any]:
    """Execute the locked 5x3 nested cross-validation procedure on the cohort.

    ``external_test`` (synthetic calibration only) is scored by every fitted outer-fold model;
    the result then carries ``external_delta_mse``, the fold-mean of MSE_cov - MSE_paired on it.
    ``allow_missing_outcome`` keeps the dataset's population and folds and gives participants with a
    missing endpoint weight 0, which removes them from every training fit and test error.
    """
    target_col = endpoint or cfg.primary_endpoint
    if target_col not in dataset.outcomes:
        raise ValueError(f"Endpoint '{target_col}' not present in dataset outcomes.")
    y_all = dataset.outcomes[target_col]
    external = None if external_test is None else (external_test, external_test.outcomes[target_col])
    external_fold_deltas: list[float] = []
    fold_test_scores: list[tuple[np.ndarray, np.ndarray]] = []
    observed = np.isfinite(y_all)
    if not observed.all():
        if not allow_missing_outcome:
            raise ValueError(f"Endpoint '{target_col}' contains missing values among eligible participants.")
        y_all = np.where(observed, y_all, 0.0)  # placeholder; weight 0 keeps it out of every weighted sum

    n_p = dataset.n_participants
    mult = (
        np.ones(n_p, dtype=float)
        if participant_multiplicities is None
        else np.asarray(participant_multiplicities, dtype=float)
    ) * observed

    seed_to_use = cfg.primary_seed if outer_seed is None else int(outer_seed)
    outer_splits, inner_splits_per_fold = (
        make_nested_splits(dataset, cfg, seed_to_use) if precomputed_splits is None else precomputed_splits
    )

    model_names = [
        "covariate_ridge",
        "fixed_band_ridge",
        "pca_ridge",
        "pls_regression",
        "paired_linear_ridge",
        "strongest_comparator",
    ]
    if include_band_paired:
        model_names.append("band_paired_ridge")

    oof_preds = {m: np.full(n_p, np.nan, dtype=float) for m in model_names}
    oof_spec = {
        m: np.full(n_p, np.nan, dtype=float)
        for m in ("fixed_band_ridge", "pca_ridge", "pls_regression", "paired_linear_ridge")
    }
    fold_settings: list[dict[str, Any]] = []
    fold_repeatability: list[dict[str, Any]] = []
    subspace_bases: list[np.ndarray] = []

    for fold_idx, ((tr_idx, te_idx), in_splits) in enumerate(zip(outer_splits, inner_splits_per_fold)):
        if np.sum(mult[te_idx] > 0) == 0:
            raise ValueError(f"Outer fold {fold_idx} has zero weighted test participants.")
        fold_out = _select_inner_models_and_fit_outer(
            dataset,
            y_all,
            tr_idx,
            te_idx,
            in_splits,
            mult,
            cfg,
            single_sample_downstream=single_sample_downstream,
            include_band_paired=include_band_paired,
            external=external,
        )
        if external is not None:
            ext_mse = fold_out["external_mse"]
            external_fold_deltas.append(ext_mse["covariate_ridge"] - ext_mse["paired_linear_ridge"])
        fold_test_scores.append((te_idx, fold_out["part_paired_z"][te_idx]))
        for m in model_names:
            oof_preds[m][te_idx] = fold_out["predictions"][m]
        for m in oof_spec:
            oof_spec[m][te_idx] = fold_out["spectral_contributions"][m]

        settings_row = {"outer_fold": fold_idx, **fold_out["selected_settings"]}
        fold_settings.append(settings_row)
        rep_row = {"outer_fold": fold_idx, **fold_out["repeatability"]}
        fold_repeatability.append(rep_row)
        subspace_bases.append(fold_out["subspace_basis"])

    # Compute endpoint summaries for every model relative to covariate-only ridge
    comparisons: dict[str, dict[str, float]] = {}
    for m in model_names:
        comparisons[m] = compute_endpoint_summary(
            y_all, oof_preds["covariate_ridge"], oof_preds[m], weights=mult
        )

    # Also compute direct paired_linear_ridge vs strongest_comparator difference
    vs_strongest = compute_endpoint_summary(
        y_all, oof_preds["strongest_comparator"], oof_preds["paired_linear_ridge"], weights=mult
    )

    angles_df = compute_subspace_principal_angles(subspace_bases)
    extra = {} if external is None else {"external_delta_mse": float(np.mean(external_fold_deltas))}
    return {
        **extra,
        "endpoint": target_col,
        "n_missing_outcome": int(np.sum(~observed)),
        "fold_test_paired_scores": fold_test_scores,
        "outer_seed": seed_to_use,
        "oof_predictions": oof_preds,
        "oof_spectral_contributions": oof_spec,
        "comparisons_vs_covariate_ridge": comparisons,
        "paired_vs_strongest_comparator": vs_strongest,
        "fold_settings": pd.DataFrame(fold_settings),
        "fold_repeatability": fold_repeatability,
        "subspace_angles": angles_df,
        "splits": (outer_splits, inner_splits_per_fold),
    }


def run_multiplicity_bootstrap(
    dataset: CohortDataset,
    *,
    cfg: PairedLipidConfig = PairedLipidConfig(),
    n_bootstraps: int = 2000,
    bootstrap_seed: int = 20261106,
    endpoint: str | None = None,
    use_gpu: bool | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Participant-level multiplicity bootstrap keeping original fold memberships fixed."""
    target_col = endpoint or cfg.primary_endpoint
    if target_col not in dataset.outcomes:
        raise ValueError(f"Endpoint '{target_col}' not present in dataset outcomes.")
    y_all = dataset.outcomes[target_col]
    n_p = dataset.n_participants
    splits = make_nested_splits(dataset, cfg, cfg.primary_seed)
    outer_splits, inner_splits_per_fold = splits

    rng = np.random.default_rng(bootstrap_seed)
    M_all = np.zeros((n_bootstraps, n_p), dtype=np.float64)
    distinct_counts: list[int] = []
    for b in range(n_bootstraps):
        drawn = rng.integers(0, n_p, size=n_p)
        mult = np.bincount(drawn, minlength=n_p).astype(np.float64)
        M_all[b] = mult
        distinct_counts.append(int(np.sum(mult > 0)))

    gcache = _get_dataset_gram_cache(dataset, cfg)
    M_with_base = np.vstack([np.ones((1, n_p), dtype=np.float64), M_all])
    elig_with_base = _check_fast_batch_eligibility(
        dataset, outer_splits, inner_splits_per_fold, M_with_base, cfg, gcache
    )
    base_elig = bool(elig_with_base[0])
    fast_mask = elig_with_base[1:]
    fast_idx = np.flatnonzero(fast_mask)
    fallback_idx = np.flatnonzero(~fast_mask)

    d_pair_arr = np.full(n_bootstraps, np.nan, dtype=np.float64)
    d_comp_arr = np.full(n_bootstraps, np.nan, dtype=np.float64)
    failed_draws = 0
    failure_reasons: list[str] = []
    point_est_delta: float | None = None
    point_est_comp: float | None = None

    # No blanket fallback: a batched-engine error would otherwise rerun every draw sequentially (~40x slower).
    if fast_idx.size > 0 or base_elig:
        M_run = np.vstack([M_with_base[:1], M_all[fast_idx]]) if base_elig else M_all[fast_idx]
        dp_fast, dc_fast = _run_batched_multiplicity_draws(
            dataset,
            y_all,
            outer_splits,
            inner_splits_per_fold,
            M_run,
            cfg,
            gcache,
            use_gpu=use_gpu,
            batch_size=batch_size,
        )
        if base_elig:
            point_est_delta = float(dp_fast[0])
            point_est_comp = float(dc_fast[0])
            d_pair_arr[fast_idx] = dp_fast[1:]
            d_comp_arr[fast_idx] = dc_fast[1:]
        else:
            d_pair_arr[fast_idx] = dp_fast
            d_comp_arr[fast_idx] = dc_fast

    if point_est_delta is None or point_est_comp is None:
        base_eval = evaluate_nested_cohort(
            dataset, cfg=cfg, endpoint=endpoint, precomputed_splits=splits
        )
        point_est_delta = float(
            base_eval["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"]
        )
        point_est_comp = float(base_eval["paired_vs_strongest_comparator"]["delta_mse"])

    for b in fallback_idx:
        mult = M_all[b]
        try:
            b_res = evaluate_nested_cohort(
                dataset,
                cfg=cfg,
                endpoint=endpoint,
                participant_multiplicities=mult,
                precomputed_splits=splits,
            )
            d_pair_arr[b] = float(
                b_res["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"]
            )
            d_comp_arr[b] = float(b_res["paired_vs_strongest_comparator"]["delta_mse"])
        except Exception as exc:
            failed_draws += 1
            failure_reasons.append(str(exc))

    delta_paired: list[float] = []
    delta_vs_comp: list[float] = []
    for b in range(n_bootstraps):
        dp = float(d_pair_arr[b])
        dc = float(d_comp_arr[b])
        if np.isfinite(dp) and np.isfinite(dc):
            delta_paired.append(dp)
            delta_vs_comp.append(dc)
        elif b not in fallback_idx:
            failed_draws += 1
            failure_reasons.append("nonfinite_bootstrap_estimate")

    # Per study design: if any bootstrap refit cannot produce the locked comparison,
    # mark that dataset's interval procedure as failed rather than discarding resamples.
    interval_valid = failed_draws == 0 and len(delta_paired) == n_bootstraps
    if interval_valid:
        arr_p = np.asarray(delta_paired, dtype=float)
        arr_c = np.asarray(delta_vs_comp, dtype=float)
        ci_paired = (float(np.percentile(arr_p, 2.5)), float(np.percentile(arr_p, 97.5)))
        ci_vs_comp = (float(np.percentile(arr_c, 2.5)), float(np.percentile(arr_c, 97.5)))
        mc_se = float(np.std(arr_p, ddof=1) / np.sqrt(n_bootstraps))
    else:
        ci_paired = (float("nan"), float("nan"))
        ci_vs_comp = (float("nan"), float("nan"))
        mc_se = float("nan")

    return {
        "point_estimate_delta_mse": point_est_delta,
        "point_estimate_vs_comparator": point_est_comp,
        "ci_95_delta_mse": ci_paired,
        "ci_95_vs_comparator": ci_vs_comp,
        "interval_procedure_succeeded": interval_valid,
        "n_bootstraps_requested": n_bootstraps,
        "n_bootstraps_succeeded": len(delta_paired),
        "n_failed_refits": failed_draws,
        "n_fallback_draws": int(fallback_idx.size),
        "failure_reasons": failure_reasons[:10],
        "mean_distinct_participants": float(np.mean(distinct_counts)) if distinct_counts else float("nan"),
        # SE of the bootstrap mean, not of the 2.5/97.5 percentile limits.
        "monte_carlo_se_of_mean": mc_se,
        "bootstrap_draws_delta_mse": np.asarray(delta_paired, dtype=float),
    }


def summarize_anchor_associations(
    dataset: CohortDataset,
    eval_result: dict[str, Any],
    *,
    anchors: Sequence[str] = ("ldl_mmol_l", "hdl_mmol_l", "triglycerides_mmol_l", "plt_10e9_l"),
    n_resamples: int = 200,
    seed: int = 42,
    cfg: PairedLipidConfig = PairedLipidConfig(),
) -> pd.DataFrame:
    """Compute continuous anchor associations and resample stability without categorical source claims."""
    spec_contrib = eval_result["oof_spectral_contributions"]["paired_linear_ridge"]
    cov_prep = CovariatePreprocessor.fit(
        dataset.covariate_frame,
        np.arange(dataset.n_participants),
        np.ones(dataset.n_participants),
        preset=cfg.covariate_preset,
    )
    X_cov = cov_prep.transform(dataset.covariate_frame, np.arange(dataset.n_participants))
    rng = np.random.default_rng(seed)

    rows: list[dict[str, Any]] = []
    for anchor in anchors:
        if anchor not in dataset.outcomes:
            continue
        y_a = dataset.outcomes[anchor]
        valid = np.isfinite(spec_contrib) & np.isfinite(y_a)
        if np.sum(valid) < 5:
            continue
        s_v = spec_contrib[valid]
        y_v = y_a[valid]
        C_v = X_cov[valid]

        # Residualize both s_v and y_v on standardized covariates + intercept
        design = np.column_stack([np.ones(s_v.size), C_v])
        q_mat, _ = linalg.qr(design, mode="economic")
        s_res = s_v - q_mat @ (q_mat.T @ s_v)
        y_res = y_v - q_mat @ (q_mat.T @ y_v)

        denom = float(linalg.norm(s_res) * linalg.norm(y_res))
        partial_r = float(np.dot(s_res, y_res) / denom) if denom > 1e-12 else 0.0
        spearman_rho = float(stats.spearmanr(s_res, y_res).statistic)

        boot_r: list[float] = []
        n_v = s_v.size
        for _ in range(n_resamples):
            idx = rng.integers(0, n_v, size=n_v)
            sr = s_res[idx] - np.mean(s_res[idx])
            yr = y_res[idx] - np.mean(y_res[idx])
            dn = float(linalg.norm(sr) * linalg.norm(yr))
            if dn > 1e-12:
                boot_r.append(float(np.dot(sr, yr) / dn))
        arr_r = np.asarray(boot_r, dtype=float)
        sign_stability = float(np.mean(np.sign(arr_r) == np.sign(partial_r))) if arr_r.size else float("nan")

        rows.append(
            {
                "anchor": anchor,
                "n_participants": int(n_v),
                "partial_pearson_r": partial_r,
                "partial_spearman_rho": spearman_rho,
                "resample_ci_low": float(np.percentile(arr_r, 2.5)) if arr_r.size else float("nan"),
                "resample_ci_high": float(np.percentile(arr_r, 97.5)) if arr_r.size else float("nan"),
                "sign_stability": sign_stability,
                "attribution_status": "continuous_exploratory_summary_only",
            }
        )
    return pd.DataFrame(rows)
