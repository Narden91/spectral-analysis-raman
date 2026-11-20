"""Nested participant-level cross-validation, external scoring, and multiplicity bootstrap."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from scipy import linalg, stats

from src.analysis.paired_lipid.batched import (
    _check_fast_batch_eligibility,
    _get_dataset_gram_cache,
    _precompute_division_constants,
    _run_batched_multiplicity_draws,
    batched_select_inner_settings,
)
from src.analysis.paired_lipid.cohort import (
    CohortDataset,
    PairedLipidConfig,
    aggregate_samples_to_participants,
    extract_sample_band_matrix,
)
from src.analysis.paired_lipid.metrics import (
    compute_discriminability_and_retrieval as compute_discriminability_and_retrieval,
)
from src.analysis.paired_lipid.metrics import (
    compute_endpoint_summary as compute_endpoint_summary,
)
from src.analysis.paired_lipid.metrics import (
    compute_icc_a1 as compute_icc_a1,
)
from src.analysis.paired_lipid.metrics import (
    compute_subspace_principal_angles as compute_subspace_principal_angles,
)
from src.analysis.paired_lipid.metrics import (
    map_subspace_to_metric_orthonormal_basis as map_subspace_to_metric_orthonormal_basis,
)
from src.analysis.paired_lipid.metrics import (
    outer_test_repeatability as outer_test_repeatability,
)
from src.analysis.paired_lipid.models import (
    CovariatePreprocessor,
    MultiAlphaRidgeResult,
    PairedMoments,
    WeightedPLSResult,
    WeightedSpectralPreprocessor,
    fit_ridge_grid,
    predict_ridge_grid,
)
from src.analysis.paired_lipid.splits import (
    make_nested_splits as make_nested_splits,
)
from src.analysis.paired_lipid.splits import (
    make_participant_splits as make_participant_splits,
)

_outer_test_repeatability = outer_test_repeatability


def _get_participant_spectra(
    dataset: CohortDataset,
    *,
    single_sample_downstream: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cached participant-averaged (1300, 2850) spectral matrices."""
    key = f"_part_spectra__single_{single_sample_downstream}"
    cached = dataset._cache.get(key)
    if cached is not None:
        return cached
    part_1300 = aggregate_samples_to_participants(
        dataset.sample_1300, dataset, single_sample_only=single_sample_downstream
    )
    part_2850 = aggregate_samples_to_participants(
        dataset.sample_2850, dataset, single_sample_only=single_sample_downstream
    )
    part_1300.setflags(write=False)
    part_2850.setflags(write=False)
    pair = (part_1300, part_2850)
    dataset._cache[key] = pair
    return pair


@dataclass(frozen=True, slots=True)
class PreparedExternalTest:
    """Pre-aggregated independent test cohort for fast repeated external scoring."""

    endpoint: str
    spec_1300: np.ndarray
    spec_2850: np.ndarray
    raw_cov_frame: pd.DataFrame
    y_ext: np.ndarray
    n_participants: int
    raw_cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    all_indices: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "all_indices", np.arange(self.n_participants, dtype=int))

    @classmethod
    def from_dataset(
        cls,
        ext_ds: CohortDataset,
        endpoint: str,
        *,
        single_sample_downstream: bool = False,
        preset: str | None = None,
    ) -> PreparedExternalTest:
        """Precompute participant-averaged spectra and optional raw covariates once."""
        if endpoint not in ext_ds.outcomes:
            raise ValueError(f"Endpoint '{endpoint}' not present in external test dataset outcomes.")
        part_1300, part_2850 = _get_participant_spectra(
            ext_ds, single_sample_downstream=single_sample_downstream
        )
        raw_cache: dict[str, Any] = {}
        if preset is not None:
            CovariatePreprocessor.raw_design_matrix(
                ext_ds.covariate_frame, preset=preset, raw_cache=raw_cache
            )
        return cls(
            endpoint=endpoint,
            spec_1300=part_1300,
            spec_2850=part_2850,
            raw_cov_frame=ext_ds.covariate_frame,
            y_ext=np.asarray(ext_ds.outcomes[endpoint], dtype=float),
            n_participants=ext_ds.n_participants,
            raw_cache=raw_cache,
        )


def _prepare_external_test(
    external_test: CohortDataset | PreparedExternalTest | None,
    target_col: str,
    *,
    single_sample_downstream: bool,
    preset: str,
) -> PreparedExternalTest | None:
    """Normalize a CohortDataset or PreparedExternalTest for repeated fold scoring."""
    if external_test is None:
        return None
    if isinstance(external_test, PreparedExternalTest):
        if external_test.endpoint != target_col:
            raise ValueError(
                f"PreparedExternalTest endpoint '{external_test.endpoint}' "
                f"does not match requested endpoint '{target_col}'."
            )
        return external_test
    if target_col not in external_test.outcomes:
        raise ValueError(f"Endpoint '{target_col}' not present in external test dataset outcomes.")
    part_1300, part_2850 = _get_participant_spectra(
        external_test, single_sample_downstream=single_sample_downstream
    )
    CovariatePreprocessor.raw_design_matrix(
        external_test.covariate_frame, preset=preset, raw_cache=external_test._cache
    )
    return PreparedExternalTest(
        endpoint=target_col,
        spec_1300=part_1300,
        spec_2850=part_2850,
        raw_cov_frame=external_test.covariate_frame,
        y_ext=np.asarray(external_test.outcomes[target_col], dtype=float),
        n_participants=external_test.n_participants,
        raw_cache=external_test._cache,
    )


def _score_external_fold(
    prepared_ext: PreparedExternalTest,
    outer_cov_prep: CovariatePreprocessor,
    outer_spec_prep: WeightedSpectralPreprocessor,
    fit_cov_outer: MultiAlphaRidgeResult,
    fit_paired_outer: MultiAlphaRidgeResult,
    sel_k: int,
    V_outer_r: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    """Score already-fitted outer covariate and paired models on a prepared external cohort."""
    X_cov_ext = outer_cov_prep.transform(
        prepared_ext.raw_cov_frame, prepared_ext.all_indices, raw_cache=prepared_ext.raw_cache
    )
    pred_cov_ext, _ = predict_ridge_grid(fit_cov_outer, X_cov_ext, None, alpha_idx=0)
    ext_part_z = outer_spec_prep.project_linear_subspace(
        prepared_ext.spec_1300, prepared_ext.spec_2850, sel_k, V_outer_r
    )
    pred_pair_ext, _ = predict_ridge_grid(fit_paired_outer, X_cov_ext, ext_part_z, alpha_idx=0)
    sq_cov_ext = (prepared_ext.y_ext - pred_cov_ext) ** 2
    sq_pair_ext = (prepared_ext.y_ext - pred_pair_ext) ** 2
    ext_mse = {
        "covariate_ridge": float(np.mean(sq_cov_ext)),
        "paired_linear_ridge": float(np.mean(sq_pair_ext)),
    }
    return ext_mse, sq_cov_ext - sq_pair_ext


@dataclass(slots=True)
class _InnerFoldCache:
    """Precomputed primal representations for one inner cross-validation division."""

    in_tr: np.ndarray
    in_val: np.ndarray
    w_tr: np.ndarray
    w_val: np.ndarray
    y_tr: np.ndarray
    y_val: np.ndarray
    X_cov_tr: np.ndarray
    X_cov_val: np.ndarray
    spec_prep: WeightedSpectralPreprocessor
    part_pca_max: np.ndarray
    paired_moments: PairedMoments
    part_scaled_tr: np.ndarray
    part_scaled_val: np.ndarray


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
    *,
    gcache: dict[str, Any] | None = None,
    in_dconsts: list[dict[str, Any]] | None = None,
) -> _InnerSelection | None:
    """Select every comparator's settings via the shared closed-form batched engine (B=1)."""
    if gcache is None:
        gcache = _get_dataset_gram_cache(dataset, cfg)
    if in_dconsts is None:
        in_dconsts = [
            _precompute_division_constants(
                np, gcache, dataset, outer_train_idx[in_tr_rel], outer_train_idx[in_val_rel], cfg
            )
            for in_tr_rel, in_val_rel in inner_splits
        ]
    sel = batched_select_inner_settings(
        np,
        gcache,
        in_dconsts,
        mult_all[None, :],
        np.asarray(y_all, dtype=np.float64),
        np.asarray(cfg.ridge_alpha_grid, dtype=np.float64),
        gcache["part_band_mat"],
        cfg,
    )
    if not bool(sel["rank_ok"][0]):
        return None
    k_list = [int(k) for k in cfg.pca_dimensions]
    lam_list = list(cfg.lambda_grid)
    r_list = list(cfg.retained_directions)
    best_pca_k_idx = int(sel["best_pca_k_idx"][0])
    best_pca_a = int(sel["best_pca_a"][0])
    best_k_idx = int(sel["best_k_idx"][0])
    best_pair_l_idx = int(sel["best_pair_l_idx"][0])
    best_r_idx = int(sel["best_r_idx"][0])
    best_pair_a = int(sel["best_pair_a"][0])
    return _InnerSelection(
        int(sel["best_cov_a"][0]),
        int(sel["best_band_a"][0]),
        float(sel["best_band_mse"][0]),
        (k_list[best_pca_k_idx], best_pca_a),
        float(sel["best_pca_mse"][0]),
        int(sel["best_pls_c_idx"][0]) + 1,
        float(sel["best_pls_mse"][0]),
        (k_list[best_k_idx], float(lam_list[best_pair_l_idx]), int(r_list[best_r_idx]), best_pair_a),
        [],
    )


def _score_ridge_inner_folds(
    inner_cache: Sequence[_InnerFoldCache],
    z_getter: Callable[[int, _InnerFoldCache], tuple[np.ndarray | None, np.ndarray | None]],
    alphas: Sequence[float],
    total_val_weight: float,
) -> np.ndarray:
    """Compute participant-weighted inner-CV validation MSE across all ridge alphas."""
    sse = np.zeros(len(alphas), dtype=float)
    for fold_idx, ic in enumerate(inner_cache):
        Z_tr, Z_val = z_getter(fold_idx, ic)
        fit = fit_ridge_grid(ic.X_cov_tr, Z_tr, ic.y_tr, ic.w_tr, alphas)
        preds, _ = predict_ridge_grid(fit, ic.X_cov_val, Z_val)
        sse += np.sum(ic.w_val[:, None] * ((ic.y_val[:, None] - preds) ** 2), axis=0)
    return sse / total_val_weight


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
    part_band_mat: np.ndarray | None = None,
    part_spectra: tuple[np.ndarray, np.ndarray] | None = None,
) -> _InnerSelection:
    """Select every comparator's settings using independent primal NumPy/SciPy inner refits."""
    del include_band_paired
    alphas = list(cfg.ridge_alpha_grid)
    n_alphas = len(alphas)
    max_pls_c = max(cfg.pls_component_grid)
    invalid_candidates: list[dict[str, Any]] = []
    if part_band_mat is None:
        sample_band_mat, _ = extract_sample_band_matrix(dataset)
        part_band_mat = aggregate_samples_to_participants(
            sample_band_mat, dataset, single_sample_only=single_sample_downstream
        )
    if part_spectra is None:
        part_spectra = _get_participant_spectra(
            dataset, single_sample_downstream=single_sample_downstream
        )
    part_1300, part_2850 = part_spectra

    # Precompute inner division preprocessors once per inner fold.
    inner_cache: list[_InnerFoldCache] = []
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

        part_scaled_tr = spec_prep.transform_scaled_blocks(part_1300[in_tr], part_2850[in_tr])
        part_scaled_val = spec_prep.transform_scaled_blocks(part_1300[in_val], part_2850[in_val])

        inner_cache.append(
            _InnerFoldCache(
                in_tr=in_tr,
                in_val=in_val,
                w_tr=w_tr,
                w_val=w_val,
                y_tr=y_all[in_tr],
                y_val=y_all[in_val],
                X_cov_tr=X_cov_tr,
                X_cov_val=X_cov_val,
                spec_prep=spec_prep,
                part_pca_max=part_pca_max,
                paired_moments=paired_moments,
                part_scaled_tr=part_scaled_tr,
                part_scaled_val=part_scaled_val,
            )
        )

    total_val_weight = sum(float(np.sum(ic.w_val)) for ic in inner_cache)

    # 1. Covariate-only ridge inner selection (tie rule: stronger alpha wins exact ties).
    cov_mses = _score_ridge_inner_folds(
        inner_cache, lambda _i, _ic: (None, None), alphas, total_val_weight
    )
    best_cov_alpha_idx = int(np.lexsort((-np.arange(n_alphas), np.round(cov_mses, 12)))[0])

    # 2. Fixed band summaries + ridge inner selection.
    band_mses = _score_ridge_inner_folds(
        inner_cache,
        lambda _i, ic: (part_band_mat[ic.in_tr], part_band_mat[ic.in_val]),
        alphas,
        total_val_weight,
    )
    best_band_alpha_idx = int(np.lexsort((-np.arange(n_alphas), np.round(band_mses, 12)))[0])
    best_band_mse = float(band_mses[best_band_alpha_idx])

    # Determine common PCA dimension cap across all inner and outer divisions.
    all_preps = [ic.spec_prep for ic in inner_cache]
    all_preps.append(outer_spec_prep)
    common_cap = min(
        min(sp.numerical_rank, sp.n_distinct_paired_train - 1, sp.pca_components.shape[0])
        for sp in all_preps
    )
    candidate_k: list[int] = []
    if common_cap >= 1:
        for d in cfg.pca_dimensions:
            capped = min(int(d), common_cap)
            if capped >= 1 and capped not in candidate_k:
                candidate_k.append(capped)
    if not candidate_k:
        raise ValueError("No valid candidate PCA dimensions available across training divisions.")

    # 3. Block-scaled PCA + ridge inner selection.
    best_pca_key: tuple[float, int, float] | None = None
    best_pca_setting = (candidate_k[0], 0)
    best_pca_mse = float("inf")
    for k in candidate_k:
        pca_mses_1d = _score_ridge_inner_folds(
            inner_cache,
            lambda _i, ic, k_cur=k: (ic.part_pca_max[ic.in_tr, :k_cur], ic.part_pca_max[ic.in_val, :k_cur]),
            alphas,
            total_val_weight,
        )
        for a_idx, mse_val in enumerate(pca_mses_1d):
            # Tie rule: lower MSE, fewer PCA dimensions k, stronger ridge penalty alpha.
            cand_key = (round(float(mse_val), 12), int(k), -float(alphas[a_idx]))
            if best_pca_key is None or cand_key < best_pca_key:
                best_pca_key = cand_key
                best_pca_setting = (int(k), int(a_idx))
                best_pca_mse = float(mse_val)

    # 4. PLS regression inner selection over configured component grid.
    pls_sse = np.zeros(max_pls_c, dtype=float)
    valid_pls_c = max_pls_c
    for ic in inner_cache:
        pls_fit = WeightedPLSResult.fit(
            ic.X_cov_tr,
            ic.part_scaled_tr,
            ic.y_tr,
            ic.w_tr,
            max_pls_c,
        )
        valid_pls_c = min(valid_pls_c, pls_fit.max_components)
        preds_pls, _ = pls_fit.predict(ic.X_cov_val, ic.part_scaled_val)
        pls_sse[:valid_pls_c] += np.sum(
            ic.w_val[:, None] * ((ic.y_val[:, None] - preds_pls[:, :valid_pls_c]) ** 2),
            axis=0,
        )
    pls_mses = pls_sse[:valid_pls_c] / total_val_weight
    valid_pls_comps = [int(c) for c in cfg.pls_component_grid if int(c) <= valid_pls_c]
    if not valid_pls_comps:
        if valid_pls_c < 1:
            raise ValueError("No valid PLS component counts available across inner divisions.")
        valid_pls_comps = [int(valid_pls_c)]
    pls_comp_indices = np.asarray([c - 1 for c in valid_pls_comps], dtype=int)
    pls_cand_mses = pls_mses[pls_comp_indices]
    best_pls_pos = int(
        np.lexsort((np.asarray(valid_pls_comps, dtype=int), np.round(pls_cand_mses, 12)))[0]
    )
    best_pls_c = int(valid_pls_comps[best_pls_pos])
    best_pls_mse = float(pls_cand_mses[best_pls_pos])

    # 5. Paired linear representation + ridge inner selection.
    best_paired_key: tuple[float, int, int, float, float] | None = None
    best_paired_setting: tuple[int, float, int, int] | None = None

    for k in candidate_k:
        valid_r_list = [int(r) for r in cfg.retained_directions if int(r) <= k]
        if not valid_r_list:
            continue
        max_r = max(valid_r_list)
        for lam in cfg.lambda_grid:
            # Solve generalized eigenproblem once per inner fold for (k, lam).
            fold_z_tr: list[np.ndarray] = []
            fold_z_val: list[np.ndarray] = []
            failed_reason: str | None = None
            for ic in inner_cache:
                try:
                    V_k, _, _ = ic.paired_moments.solve_directions(
                        k, lam, eigenvalue_tolerance=cfg.eigenvalue_tolerance
                    )
                    V_top = V_k[:, :max_r]
                    fold_z_tr.append(ic.part_pca_max[ic.in_tr, :k] @ V_top)
                    fold_z_val.append(ic.part_pca_max[ic.in_val, :k] @ V_top)
                except ValueError as exc:
                    failed_reason = str(exc)
                    break
            if failed_reason is not None:
                invalid_candidates.append({"k": k, "lambda": lam, "reason": failed_reason})
                continue

            for r in valid_r_list:
                pair_mses_1d = _score_ridge_inner_folds(
                    inner_cache,
                    lambda idx, _ic, r_cur=r, zt=fold_z_tr, zv=fold_z_val: (
                        zt[idx][:, :r_cur],
                        zv[idx][:, :r_cur],
                    ),
                    alphas,
                    total_val_weight,
                )
                for a_idx, mse_val in enumerate(pair_mses_1d):
                    # Exact tie rule: lowest MSE, fewer retained directions r, fewer PCA dims k,
                    # stronger lambda, stronger alpha.
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


def _select_inner_settings(
    dataset: CohortDataset,
    y_all: np.ndarray,
    outer_train_idx: np.ndarray,
    outer_test_idx: np.ndarray,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    mult_all: np.ndarray,
    cfg: PairedLipidConfig,
    outer_spec_prep: WeightedSpectralPreprocessor,
    *,
    single_sample_downstream: bool,
    include_band_paired: bool,
    part_band_mat: np.ndarray | None = None,
    part_spectra: tuple[np.ndarray, np.ndarray] | None = None,
) -> _InnerSelection:
    """Dispatch inner-CV selection to the batched fast path when rank and eligibility allow."""
    max_k = max(cfg.pca_dimensions)
    selection: _InnerSelection | None = None
    if (
        not single_sample_downstream
        and not include_band_paired
        and list(cfg.retained_directions) == [1, 2, 3]
        and outer_spec_prep.numerical_rank >= max_k
    ):
        gcache = _get_dataset_gram_cache(dataset, cfg)
        if not gcache["has_missing_cov"] and bool(
            _check_fast_batch_eligibility(
                dataset,
                [(outer_train_idx, outer_test_idx)],
                [inner_splits],
                mult_all[None, :],
                cfg,
                gcache,
            )[0]
        ):
            in_dconsts = [
                _precompute_division_constants(
                    np, gcache, dataset, outer_train_idx[in_tr_rel], outer_train_idx[in_val_rel], cfg
                )
                for in_tr_rel, in_val_rel in inner_splits
            ]
            if all(int(dc["base_comb_rank"]) >= max_k for dc in in_dconsts):
                selection = _select_inner_fast(
                    dataset,
                    y_all,
                    outer_train_idx,
                    inner_splits,
                    mult_all,
                    cfg,
                    gcache=gcache,
                    in_dconsts=in_dconsts,
                )

    if selection is None:
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
            part_band_mat=part_band_mat,
            part_spectra=part_spectra,
        )
    return selection


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
    external: PreparedExternalTest | tuple[CohortDataset, np.ndarray] | None = None,
    compute_diagnostics: bool = True,
    sample_band_mat: np.ndarray | None = None,
    part_band_mat: np.ndarray | None = None,
    part_spectra: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Fit inner CV selection and outer-fold models with memoized spectral SVD & multi-alpha ridge."""
    mult_all = np.asarray(participant_multiplicities, dtype=float)
    if sample_band_mat is None:
        sample_band_mat, _ = extract_sample_band_matrix(dataset)
    if part_band_mat is None:
        part_band_mat = aggregate_samples_to_participants(
            sample_band_mat, dataset, single_sample_only=single_sample_downstream
        )
    if part_spectra is None:
        part_spectra = _get_participant_spectra(
            dataset, single_sample_downstream=single_sample_downstream
        )
    part_1300, part_2850 = part_spectra

    outer_w_tr = mult_all[outer_train_idx]
    outer_spec_prep = WeightedSpectralPreprocessor.fit(dataset, outer_train_idx, outer_w_tr, cfg=cfg)
    alphas = list(cfg.ridge_alpha_grid)

    selection = _select_inner_settings(
        dataset,
        y_all,
        outer_train_idx,
        outer_test_idx,
        inner_splits,
        mult_all,
        cfg,
        outer_spec_prep,
        single_sample_downstream=single_sample_downstream,
        include_band_paired=include_band_paired,
        part_band_mat=part_band_mat,
        part_spectra=part_spectra,
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

    # Refit all selected models on the complete outer training division.
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

    # Outer PLS Regression (affine scaling commutes with equal participant averaging).
    part_scaled_outer = outer_spec_prep.transform_scaled_blocks(part_1300, part_2850)
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

    # Select strongest established comparator using 12-decimal rounded inner MSE with fixed tie order.
    comp_candidates = (
        ("fixed_band_ridge", round(float(best_band_mse), 12)),
        ("pca_ridge", round(float(best_pca_mse), 12)),
        ("pls_regression", round(float(best_pls_mse), 12)),
    )
    strongest_comp_name = min(enumerate(comp_candidates), key=lambda item: (item[1][1], item[0]))[1][0]
    comp_preds_map = {
        "fixed_band_ridge": pred_band_ote,
        "pca_ridge": pred_pca_ote,
        "pls_regression": pred_pls_ote,
    }

    if compute_diagnostics:
        repeatability_info = outer_test_repeatability(
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
    else:
        repeatability_info = {}
        subspace_basis = np.empty((0, 0), dtype=float)

    out: dict[str, Any] = {
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
        if isinstance(external, PreparedExternalTest):
            prepared_ext = external
        else:
            ext_ds, y_ext = external
            ext_p1, ext_p2 = _get_participant_spectra(
                ext_ds, single_sample_downstream=single_sample_downstream
            )
            CovariatePreprocessor.raw_design_matrix(
                ext_ds.covariate_frame, preset=cfg.covariate_preset, raw_cache=ext_ds._cache
            )
            prepared_ext = PreparedExternalTest(
                endpoint=cfg.primary_endpoint,
                spec_1300=ext_p1,
                spec_2850=ext_p2,
                raw_cov_frame=ext_ds.covariate_frame,
                y_ext=np.asarray(y_ext, dtype=float),
                n_participants=ext_ds.n_participants,
                raw_cache=ext_ds._cache,
            )
        ext_mse, ext_deltas = _score_external_fold(
            prepared_ext,
            outer_cov_prep,
            outer_spec_prep,
            fit_cov_outer,
            fit_paired_outer,
            sel_k,
            V_outer_r,
        )
        out["external_mse"] = ext_mse
        out["external_participant_deltas"] = ext_deltas

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


def score_external_primary_pair(
    dataset: CohortDataset,
    external_test: CohortDataset | PreparedExternalTest,
    *,
    cfg: PairedLipidConfig = PairedLipidConfig(),
    splits: tuple[
        list[tuple[np.ndarray, np.ndarray]],
        list[list[tuple[np.ndarray, np.ndarray]]],
    ]
    | None = None,
    endpoint: str | None = None,
    single_sample_downstream: bool = False,
    participant_multiplicities: np.ndarray | None = None,
    allow_missing_outcome: bool = False,
) -> dict[str, Any]:
    """Fit only outer covariate and paired models per fold and score an independent external cohort."""
    target_col = endpoint or cfg.primary_endpoint
    if target_col not in dataset.outcomes:
        raise ValueError(f"Endpoint '{target_col}' not present in dataset outcomes.")
    y_all = dataset.outcomes[target_col]
    observed = np.isfinite(y_all)
    if not observed.all():
        if not allow_missing_outcome:
            raise ValueError(f"Endpoint '{target_col}' contains missing values among eligible participants.")
        y_all = np.where(observed, y_all, 0.0)

    prepared_ext = _prepare_external_test(
        external_test,
        target_col,
        single_sample_downstream=single_sample_downstream,
        preset=cfg.covariate_preset,
    )
    assert prepared_ext is not None

    n_p = dataset.n_participants
    mult = (
        np.ones(n_p, dtype=float)
        if participant_multiplicities is None
        else np.asarray(participant_multiplicities, dtype=float)
    ) * observed

    outer_splits, inner_splits_per_fold = (
        make_nested_splits(dataset, cfg, cfg.primary_seed) if splits is None else splits
    )
    alphas = list(cfg.ridge_alpha_grid)
    external_fold_deltas: list[float] = []
    external_part_delta_sum = np.zeros(prepared_ext.n_participants, dtype=float)

    for fold_idx, ((tr_idx, te_idx), in_splits) in enumerate(zip(outer_splits, inner_splits_per_fold)):
        if np.sum(mult[te_idx] > 0) == 0:
            raise ValueError(f"Outer fold {fold_idx} has zero weighted test participants.")
        outer_w_tr = mult[tr_idx]
        outer_spec_prep = WeightedSpectralPreprocessor.fit(dataset, tr_idx, outer_w_tr, cfg=cfg)
        selection = _select_inner_settings(
            dataset,
            y_all,
            tr_idx,
            te_idx,
            in_splits,
            mult,
            cfg,
            outer_spec_prep,
            single_sample_downstream=single_sample_downstream,
            include_band_paired=False,
        )
        outer_cov_prep = CovariatePreprocessor.fit(
            dataset.covariate_frame, tr_idx, outer_w_tr, preset=cfg.covariate_preset, raw_cache=dataset._cache
        )
        X_cov_otr = outer_cov_prep.transform(dataset.covariate_frame, tr_idx, raw_cache=dataset._cache)
        y_otr = y_all[tr_idx]

        fit_cov_outer = fit_ridge_grid(
            X_cov_otr, None, y_otr, outer_w_tr, [alphas[selection.cov_alpha_idx]]
        )

        sel_k, sel_lam, sel_r, sel_a_idx = selection.paired_setting
        sample_pca_outer = outer_spec_prep.transform_pca(
            dataset.sample_1300, dataset.sample_2850, k=sel_k
        )
        outer_moments = PairedMoments.from_sample_coordinates(
            sample_pca_outer, dataset, tr_idx, outer_w_tr
        )
        V_outer_k, _, _ = outer_moments.solve_directions(
            sel_k, sel_lam, eigenvalue_tolerance=cfg.eigenvalue_tolerance
        )
        V_outer_r = V_outer_k[:, :sel_r]
        sample_paired_z = sample_pca_outer @ V_outer_r
        part_paired_z = aggregate_samples_to_participants(
            sample_paired_z, dataset, single_sample_only=single_sample_downstream
        )
        fit_paired_outer = fit_ridge_grid(
            X_cov_otr,
            part_paired_z[tr_idx],
            y_otr,
            outer_w_tr,
            [alphas[sel_a_idx]],
        )

        ext_mse, ext_deltas = _score_external_fold(
            prepared_ext,
            outer_cov_prep,
            outer_spec_prep,
            fit_cov_outer,
            fit_paired_outer,
            sel_k,
            V_outer_r,
        )
        external_fold_deltas.append(ext_mse["covariate_ridge"] - ext_mse["paired_linear_ridge"])
        external_part_delta_sum += ext_deltas

    n_folds_ext = max(len(external_fold_deltas), 1)
    return {
        "endpoint": target_col,
        "external_delta_mse": float(np.mean(external_fold_deltas)),
        "external_participant_deltas": external_part_delta_sum / float(n_folds_ext),
    }


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
    external_test: CohortDataset | PreparedExternalTest | None = None,
    allow_missing_outcome: bool = False,
    compute_diagnostics: bool = True,
) -> dict[str, Any]:
    """Execute the locked 5x3 nested cross-validation procedure on the cohort.

    ``external_test`` (synthetic calibration only) is scored by every fitted outer-fold model;
    the result then carries ``external_delta_mse``, the fold-mean of MSE_cov - MSE_paired on it.
    ``allow_missing_outcome`` keeps the dataset's population and folds and gives participants with a
    missing endpoint weight 0, which removes them from every training fit and test error.
    ``compute_diagnostics=False`` skips repeatability ICC/discriminability and channel-space QR/SVD
    principal angles for callers that consume only predictions and error summaries.
    """
    target_col = endpoint or cfg.primary_endpoint
    if target_col not in dataset.outcomes:
        raise ValueError(f"Endpoint '{target_col}' not present in dataset outcomes.")
    y_all = dataset.outcomes[target_col]
    prepared_ext = _prepare_external_test(
        external_test,
        target_col,
        single_sample_downstream=single_sample_downstream,
        preset=cfg.covariate_preset,
    )
    external_fold_deltas: list[float] = []
    external_part_delta_sum: np.ndarray | None = (
        np.zeros(prepared_ext.n_participants, dtype=float) if prepared_ext is not None else None
    )
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

    sample_band_mat, _ = extract_sample_band_matrix(dataset)
    part_band_mat = aggregate_samples_to_participants(
        sample_band_mat, dataset, single_sample_only=single_sample_downstream
    )
    part_spectra = _get_participant_spectra(
        dataset, single_sample_downstream=single_sample_downstream
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
            external=prepared_ext,
            compute_diagnostics=compute_diagnostics,
            sample_band_mat=sample_band_mat,
            part_band_mat=part_band_mat,
            part_spectra=part_spectra,
        )
        if prepared_ext is not None:
            ext_mse = fold_out["external_mse"]
            external_fold_deltas.append(ext_mse["covariate_ridge"] - ext_mse["paired_linear_ridge"])
            if external_part_delta_sum is not None:
                external_part_delta_sum += fold_out["external_participant_deltas"]
        fold_test_scores.append((te_idx, fold_out["part_paired_z"][te_idx]))
        for m in model_names:
            oof_preds[m][te_idx] = fold_out["predictions"][m]
        for m in oof_spec:
            oof_spec[m][te_idx] = fold_out["spectral_contributions"][m]

        settings_row = {"outer_fold": fold_idx, **fold_out["selected_settings"]}
        fold_settings.append(settings_row)
        if compute_diagnostics:
            rep_row = {"outer_fold": fold_idx, **fold_out["repeatability"]}
            fold_repeatability.append(rep_row)
            subspace_bases.append(fold_out["subspace_basis"])

    # Compute endpoint summaries for every model relative to covariate-only ridge.
    comparisons: dict[str, dict[str, float]] = {}
    for m in model_names:
        comparisons[m] = compute_endpoint_summary(
            y_all, oof_preds["covariate_ridge"], oof_preds[m], weights=mult
        )

    # Also compute direct paired_linear_ridge vs strongest_comparator difference.
    vs_strongest = compute_endpoint_summary(
        y_all, oof_preds["strongest_comparator"], oof_preds["paired_linear_ridge"], weights=mult
    )

    angles_df = compute_subspace_principal_angles(subspace_bases) if compute_diagnostics else pd.DataFrame()
    extra: dict[str, Any] = {}
    if prepared_ext is not None:
        n_folds_ext = max(len(external_fold_deltas), 1)
        extra["external_delta_mse"] = float(np.mean(external_fold_deltas))
        if external_part_delta_sum is not None:
            extra["external_participant_deltas"] = external_part_delta_sum / float(n_folds_ext)
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


def _supports_keyword(func: Callable[..., Any], name: str) -> bool:
    """Return True if ``func`` accepts keyword argument ``name`` directly or via **kwargs."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    return name in sig.parameters or any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
    )


def run_multiplicity_bootstrap(
    dataset: CohortDataset,
    *,
    cfg: PairedLipidConfig = PairedLipidConfig(),
    n_bootstraps: int = 2000,
    bootstrap_seed: int = 20261106,
    endpoint: str | None = None,
    use_gpu: bool | None = None,
    batch_size: int | None = None,
    allow_missing_outcome: bool = False,
) -> dict[str, Any]:
    """Participant-level multiplicity bootstrap keeping original fold memberships fixed."""
    target_col = endpoint or cfg.primary_endpoint
    if target_col not in dataset.outcomes:
        raise ValueError(f"Endpoint '{target_col}' not present in dataset outcomes.")
    y_raw = np.asarray(dataset.outcomes[target_col], dtype=float)
    observed = np.isfinite(y_raw)
    if not observed.all():
        if not allow_missing_outcome:
            raise ValueError(f"Endpoint '{target_col}' contains missing values among eligible participants.")
        y_all = np.where(observed, y_raw, 0.0)
    else:
        y_all = y_raw

    n_p = dataset.n_participants
    splits = make_nested_splits(dataset, cfg, cfg.primary_seed)
    outer_splits, inner_splits_per_fold = splits

    rng = np.random.default_rng(bootstrap_seed)
    M_all = np.zeros((n_bootstraps, n_p), dtype=np.float64)
    if n_bootstraps > 0 and n_p > 0:
        drawn_mat = rng.integers(0, n_p, size=(n_bootstraps, n_p))
        np.add.at(M_all, (np.arange(n_bootstraps)[:, None], drawn_mat), 1.0)
    if not observed.all():
        M_all *= observed[None, :]
    distinct_counts = np.count_nonzero(M_all > 0.0, axis=1)

    gcache = _get_dataset_gram_cache(dataset, cfg)
    base_weights = observed.astype(np.float64)[None, :]
    M_with_base = np.vstack([base_weights, M_all])
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
    exception_draws: set[int] = set()
    extra_fallback_draws: set[int] = set()
    execution: dict[str, Any] = {"device": "sequential_cpu", "gpu_requested": use_gpu}
    point_est_delta: float | None = None
    point_est_comp: float | None = None

    # No blanket fallback: a batched-engine error would otherwise rerun every draw sequentially (~40x slower).
    if fast_idx.size > 0 or base_elig:
        M_run = np.vstack([M_with_base[:1], M_all[fast_idx]]) if base_elig else M_all[fast_idx]
        batched_fallback_sub: set[int] = set()
        batched_kwargs: dict[str, Any] = {
            "use_gpu": use_gpu,
            "batch_size": batch_size,
            "execution_metadata": execution,
        }
        if _supports_keyword(_run_batched_multiplicity_draws, "fallback_draws_out"):
            batched_kwargs["fallback_draws_out"] = batched_fallback_sub
        dp_fast, dc_fast = _run_batched_multiplicity_draws(
            dataset,
            y_all,
            outer_splits,
            inner_splits_per_fold,
            M_run,
            cfg,
            gcache,
            **batched_kwargs,
        )
        if base_elig:
            if 0 not in batched_fallback_sub and np.isfinite(dp_fast[0]) and np.isfinite(dc_fast[0]):
                point_est_delta = float(dp_fast[0])
                point_est_comp = float(dc_fast[0])
            d_pair_arr[fast_idx] = dp_fast[1:]
            d_comp_arr[fast_idx] = dc_fast[1:]
            for sub_idx in batched_fallback_sub:
                if sub_idx > 0:
                    extra_fallback_draws.add(int(fast_idx[sub_idx - 1]))
        else:
            d_pair_arr[fast_idx] = dp_fast
            d_comp_arr[fast_idx] = dc_fast
            for sub_idx in batched_fallback_sub:
                extra_fallback_draws.add(int(fast_idx[sub_idx]))

    eval_extra_kwargs: dict[str, Any] = {}
    if allow_missing_outcome:
        eval_extra_kwargs["allow_missing_outcome"] = True

    if point_est_delta is None or point_est_comp is None:
        base_eval = evaluate_nested_cohort(
            dataset,
            cfg=cfg,
            endpoint=endpoint,
            precomputed_splits=splits,
            compute_diagnostics=False,
            **eval_extra_kwargs,
        )
        point_est_delta = float(
            base_eval["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"]
        )
        point_est_comp = float(base_eval["paired_vs_strongest_comparator"]["delta_mse"])

    all_fallback_idx = sorted(set(int(b) for b in fallback_idx) | extra_fallback_draws)
    for b in all_fallback_idx:
        mult = M_all[b]
        try:
            b_res = evaluate_nested_cohort(
                dataset,
                cfg=cfg,
                endpoint=endpoint,
                participant_multiplicities=mult,
                precomputed_splits=splits,
                compute_diagnostics=False,
                **eval_extra_kwargs,
            )
            d_pair_arr[b] = float(
                b_res["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]["delta_mse"]
            )
            d_comp_arr[b] = float(b_res["paired_vs_strongest_comparator"]["delta_mse"])
        except Exception as exc:
            d_pair_arr[b] = np.nan
            d_comp_arr[b] = np.nan
            exception_draws.add(int(b))
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
        elif b not in exception_draws:
            failed_draws += 1
            failure_reasons.append("nonfinite_bootstrap_estimate")

    # Per study design: if any bootstrap refit cannot produce the locked comparison,
    # mark that dataset's interval procedure as failed rather than discarding resamples.
    interval_valid = n_bootstraps > 0 and failed_draws == 0 and len(delta_paired) == n_bootstraps
    if interval_valid:
        arr_p = np.asarray(delta_paired, dtype=float)
        arr_c = np.asarray(delta_vs_comp, dtype=float)
        ci_paired = (float(np.percentile(arr_p, 2.5)), float(np.percentile(arr_p, 97.5)))
        ci_vs_comp = (float(np.percentile(arr_c, 2.5)), float(np.percentile(arr_c, 97.5)))
        mc_se = float(np.std(arr_p, ddof=1) / np.sqrt(n_bootstraps)) if n_bootstraps > 1 else float("nan")
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
        "n_fallback_draws": len(all_fallback_idx),
        "execution": execution,
        "failure_reasons": failure_reasons[:10],
        "mean_distinct_participants": float(np.mean(distinct_counts)) if distinct_counts.size else float("nan"),
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
    all_idx = np.arange(dataset.n_participants, dtype=int)
    cov_prep = CovariatePreprocessor.fit(
        dataset.covariate_frame,
        all_idx,
        np.ones(dataset.n_participants, dtype=float),
        preset=cfg.covariate_preset,
        raw_cache=dataset._cache,
    )
    X_cov = cov_prep.transform(dataset.covariate_frame, all_idx, raw_cache=dataset._cache)
    rng = np.random.default_rng(seed)

    # Reuse the rank-revealing SVD residualization basis across anchors sharing a valid mask.
    basis_cache: dict[bytes, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for anchor in anchors:
        if anchor not in dataset.outcomes:
            continue
        y_a = dataset.outcomes[anchor]
        valid = np.isfinite(spec_contrib) & np.isfinite(y_a)
        n_v = int(np.count_nonzero(valid))
        if n_v < 5:
            continue
        s_v = spec_contrib[valid]
        y_v = y_a[valid]

        mask_key = valid.tobytes()
        u_kept = basis_cache.get(mask_key)
        if u_kept is None:
            design = np.column_stack([np.ones(n_v, dtype=float), X_cov[valid]])
            u_mat, s_vals, _ = linalg.svd(design, full_matrices=False)
            if s_vals.size > 0 and s_vals[0] > 0.0:
                tol = max(design.shape) * np.finfo(float).eps * float(s_vals[0])
                u_kept = u_mat[:, s_vals > tol]
            else:
                u_kept = np.empty((n_v, 0), dtype=float)
            basis_cache[mask_key] = u_kept

        # Residualize both s_v and y_v on the rank-revealing orthonormal covariate basis.
        s_res = s_v - u_kept @ (u_kept.T @ s_v)
        y_res = y_v - u_kept @ (u_kept.T @ y_v)

        denom = float(linalg.norm(s_res) * linalg.norm(y_res))
        partial_r = float(np.dot(s_res, y_res) / denom) if denom > 1e-12 else 0.0
        spearman_rho = float(stats.spearmanr(s_res, y_res).statistic)

        if n_resamples > 0:
            idx_mat = rng.integers(0, n_v, size=(n_resamples, n_v))
            sr_mat = s_res[idx_mat]
            yr_mat = y_res[idx_mat]
            sr_c = sr_mat - np.mean(sr_mat, axis=1, keepdims=True)
            yr_c = yr_mat - np.mean(yr_mat, axis=1, keepdims=True)
            dn_vec = np.linalg.norm(sr_c, axis=1) * np.linalg.norm(yr_c, axis=1)
            valid_dn = dn_vec > 1e-12
            arr_r = np.sum(sr_c[valid_dn] * yr_c[valid_dn], axis=1) / dn_vec[valid_dn]
        else:
            arr_r = np.empty(0, dtype=float)
        sign_stability = float(np.mean(np.sign(arr_r) == np.sign(partial_r))) if arr_r.size else float("nan")

        rows.append(
            {
                "anchor": anchor,
                "n_participants": n_v,
                "partial_pearson_r": partial_r,
                "partial_spearman_rho": spearman_rho,
                "resample_ci_low": float(np.percentile(arr_r, 2.5)) if arr_r.size else float("nan"),
                "resample_ci_high": float(np.percentile(arr_r, 97.5)) if arr_r.size else float("nan"),
                "sign_stability": sign_stability,
                "attribution_status": "continuous_exploratory_summary_only",
            }
        )
    return pd.DataFrame(rows)
