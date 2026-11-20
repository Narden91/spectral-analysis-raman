"""Nested participant-level cross-validation, external scoring, and multiplicity bootstrap."""

from __future__ import annotations

import inspect
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple, get_args

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
    standardize_sample_matrix_on_train,
)
from src.analysis.paired_lipid.splits import (
    make_nested_splits as make_nested_splits,
)
from src.analysis.paired_lipid.splits import (
    make_participant_splits as make_participant_splits,
)
from src.analysis.raman_bands import RAMAN_BANDS

_outer_test_repeatability = outer_test_repeatability

IntervalMethod = Literal["studentized_t", "percentile"]
INTERVAL_METHODS: tuple[IntervalMethod, ...] = get_args(IntervalMethod)


def _effective_single_sample_downstream(
    dataset: CohortDataset,
    single_sample_downstream: bool | None,
) -> bool:
    """Resolve single-sample downstream flag, auto-defaulting to True for single_sample datasets."""
    if single_sample_downstream is not None:
        return bool(single_sample_downstream)
    summary = getattr(dataset, "manifest_summary", None)
    return bool(isinstance(summary, dict) and summary.get("subset_mode") == "single_sample")


def _maritz_jarrett_quantile_se(values: np.ndarray, q: float) -> float:
    """Deterministic Maritz-Jarrett (1978) order-statistic Monte Carlo SE for sample quantile q."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2 or not np.all(np.isfinite(arr)):
        return float("nan")
    x = np.sort(arr)
    n = int(x.size)
    grid = np.linspace(0.0, 1.0, n + 1, dtype=float)
    cdf = stats.beta.cdf(grid, q * float(n + 1), (1.0 - q) * float(n + 1))
    w = np.diff(cdf)
    c1 = float(w @ x)
    var = float(w @ ((x - c1) ** 2))
    return float(math.sqrt(max(var, 0.0)))


def _canonical_raman_band_labels(wn_1300: np.ndarray, wn_2850: np.ndarray) -> list[str]:
    """Tag each canonical spectral channel with its inclusive RAMAN_BANDS name, or empty string."""
    labels_1300 = [""] * int(wn_1300.size)
    labels_2850 = [""] * int(wn_2850.size)
    for name, band in RAMAN_BANDS.items():
        lo, hi = float(band.bounds[0]), float(band.bounds[1])
        if hi < 2000.0:
            for idx in np.flatnonzero((wn_1300 >= lo) & (wn_1300 <= hi)):
                labels_1300[int(idx)] = name
        else:
            for idx in np.flatnonzero((wn_2850 >= lo) & (wn_2850 <= hi)):
                labels_2850[int(idx)] = name
    return labels_1300 + labels_2850


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
    part_band_mat: np.ndarray | None = None
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
        """Precompute participant-averaged spectra, 21-band features, and optional raw covariates once."""
        if endpoint not in ext_ds.outcomes:
            raise ValueError(f"Endpoint '{endpoint}' not present in external test dataset outcomes.")
        part_1300, part_2850 = _get_participant_spectra(
            ext_ds, single_sample_downstream=single_sample_downstream
        )
        ext_sample_band, _ = extract_sample_band_matrix(ext_ds)
        ext_part_band = aggregate_samples_to_participants(
            ext_sample_band, ext_ds, single_sample_only=single_sample_downstream
        )
        ext_part_band.setflags(write=False)
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
            part_band_mat=ext_part_band,
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
        if external_test.part_band_mat is None:
            raise ValueError(
                "PreparedExternalTest.part_band_mat is required for external comparator scoring; "
                "construct via PreparedExternalTest.from_dataset(...)."
            )
        return external_test
    if target_col not in external_test.outcomes:
        raise ValueError(f"Endpoint '{target_col}' not present in external test dataset outcomes.")
    part_1300, part_2850 = _get_participant_spectra(
        external_test, single_sample_downstream=single_sample_downstream
    )
    ext_sample_band, _ = extract_sample_band_matrix(external_test)
    ext_part_band = aggregate_samples_to_participants(
        ext_sample_band, external_test, single_sample_only=single_sample_downstream
    )
    ext_part_band.setflags(write=False)
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
        part_band_mat=ext_part_band,
        raw_cache=external_test._cache,
    )


def _select_strongest_comparator_name(band_mse: float, pca_mse: float, pls_mse: float) -> str:
    """Select strongest established comparator using 12-decimal rounded inner MSE with fixed tie order."""
    comp_candidates = (
        ("fixed_band_ridge", round(float(band_mse), 12)),
        ("pca_ridge", round(float(pca_mse), 12)),
        ("pls_regression", round(float(pls_mse), 12)),
    )
    return min(enumerate(comp_candidates), key=lambda item: (item[1][1], item[0]))[1][0]


def _score_external_fold(
    prepared_ext: PreparedExternalTest,
    outer_cov_prep: CovariatePreprocessor,
    outer_spec_prep: WeightedSpectralPreprocessor,
    fit_cov_outer: MultiAlphaRidgeResult,
    fit_paired_outer: MultiAlphaRidgeResult,
    sel_k: int,
    V_outer_r: np.ndarray,
    *,
    X_cov_ext: np.ndarray | None = None,
    pred_comp_ext: np.ndarray | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray | None]:
    """Score already-fitted outer covariate, paired, and optional comparator models externally."""
    if X_cov_ext is None:
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
    ext_mse: dict[str, float] = {
        "covariate_ridge": float(np.mean(sq_cov_ext)),
        "paired_linear_ridge": float(np.mean(sq_pair_ext)),
    }
    ext_deltas_comp: np.ndarray | None = None
    if pred_comp_ext is not None:
        sq_comp_ext = (prepared_ext.y_ext - pred_comp_ext) ** 2
        ext_mse["strongest_comparator"] = float(np.mean(sq_comp_ext))
        ext_deltas_comp = sq_comp_ext - sq_pair_ext
    return ext_mse, sq_cov_ext - sq_pair_ext, ext_deltas_comp


@dataclass(slots=True)
class _InnerFoldCache:
    """Precomputed primal representations for one inner cross-validation division."""

    in_tr: np.ndarray
    in_val: np.ndarray
    w_tr: np.ndarray
    w_val: np.ndarray
    y_tr: np.ndarray
    y_val: np.ndarray
    cov_prep: CovariatePreprocessor
    X_cov_tr: np.ndarray
    X_cov_val: np.ndarray
    spec_prep: WeightedSpectralPreprocessor
    part_pca_max: np.ndarray
    paired_moments: PairedMoments
    part_scaled_tr: np.ndarray
    part_scaled_val: np.ndarray


class _InnerSelection(NamedTuple):
    """Inner-CV choices and diagnostics for one outer fold; alpha entries index cfg.ridge_alpha_grid."""

    cov_alpha_idx: int
    band_alpha_idx: int
    band_mse: float
    pca_setting: tuple[int, int]
    pca_mse: float
    pls_components: int
    pls_mse: float
    paired_setting: tuple[int, float, int, int]
    invalid_candidates: list[dict[str, Any]]
    cov_mse: float = float("nan")
    paired_mse: float = float("nan")
    inner_min_paired_distinct: int = 0
    inner_max_paired_distinct: int = 0
    inner_min_numerical_rank: int = 0
    inner_dropped_covariates: tuple[str, ...] = ()


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
        compute_inner_diagnostics=True,
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
        cov_alpha_idx=int(sel["best_cov_a"][0]),
        band_alpha_idx=int(sel["best_band_a"][0]),
        band_mse=float(sel["best_band_mse"][0]),
        pca_setting=(k_list[best_pca_k_idx], best_pca_a),
        pca_mse=float(sel["best_pca_mse"][0]),
        pls_components=int(sel["best_pls_c_idx"][0]) + 1,
        pls_mse=float(sel["best_pls_mse"][0]),
        paired_setting=(k_list[best_k_idx], float(lam_list[best_pair_l_idx]), int(r_list[best_r_idx]), best_pair_a),
        invalid_candidates=[],
        cov_mse=float(sel["best_cov_mse"][0]),
        paired_mse=float(sel["best_pair_mse"][0]),
        inner_min_paired_distinct=int(sel["inner_min_paired_distinct"][0]),
        inner_max_paired_distinct=int(sel["inner_max_paired_distinct"][0]),
        inner_min_numerical_rank=int(sel["inner_min_numerical_rank"][0]),
        inner_dropped_covariates=tuple(sel["inner_dropped_covariates"][0]),
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
                cov_prep=cov_prep,
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
    best_cov_mse = float(cov_mses[best_cov_alpha_idx])

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
    best_paired_mse = float("inf")

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
                except (ValueError, linalg.LinAlgError) as exc:
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
                        best_paired_mse = float(mse_val)

    if best_paired_setting is None:
        raise ValueError("No valid paired linear candidate found in inner selection.")

    inner_paired_counts = [int(ic.paired_moments.n_paired_distinct) for ic in inner_cache]
    inner_ranks = [int(ic.spec_prep.numerical_rank) for ic in inner_cache]
    inner_dropped_covs = tuple(
        sorted({col for ic in inner_cache for col in ic.cov_prep.dropped_columns})
    )

    return _InnerSelection(
        cov_alpha_idx=best_cov_alpha_idx,
        band_alpha_idx=best_band_alpha_idx,
        band_mse=best_band_mse,
        pca_setting=best_pca_setting,
        pca_mse=best_pca_mse,
        pls_components=best_pls_c,
        pls_mse=best_pls_mse,
        paired_setting=best_paired_setting,
        invalid_candidates=invalid_candidates,
        cov_mse=best_cov_mse,
        paired_mse=best_paired_mse,
        inner_min_paired_distinct=min(inner_paired_counts),
        inner_max_paired_distinct=max(inner_paired_counts),
        inner_min_numerical_rank=min(inner_ranks),
        inner_dropped_covariates=inner_dropped_covs,
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
    best_cov_alpha_idx = selection.cov_alpha_idx
    best_cov_mse = selection.cov_mse
    best_band_alpha_idx = selection.band_alpha_idx
    best_band_mse = selection.band_mse
    best_pca_setting = selection.pca_setting
    best_pca_mse = selection.pca_mse
    best_pls_c = selection.pls_components
    best_pls_mse = selection.pls_mse
    best_paired_setting = selection.paired_setting
    best_paired_mse = selection.paired_mse
    invalid_candidates = selection.invalid_candidates
    inner_min_paired_distinct = selection.inner_min_paired_distinct
    inner_max_paired_distinct = selection.inner_max_paired_distinct
    inner_min_numerical_rank = selection.inner_min_numerical_rank
    inner_dropped_covariates = selection.inner_dropped_covariates

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

    strongest_comp_name = _select_strongest_comparator_name(best_band_mse, best_pca_mse, best_pls_mse)
    comp_preds_map = {
        "fixed_band_ridge": pred_band_ote,
        "pca_ridge": pred_pca_ote,
        "pls_regression": pred_pls_ote,
    }

    sample_band_std = (
        standardize_sample_matrix_on_train(sample_band_mat, dataset, outer_train_idx, outer_w_tr)
        if (compute_diagnostics or include_band_paired)
        else None
    )

    p1 = dataset.wavenumbers_1300.size
    p2 = dataset.wavenumbers_2850.size
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
            sample_band_discrim=sample_band_std,
            fit_pca_outer=fit_pca_outer,
            fit_pls_outer=fit_pls_outer,
            outer_spec_prep=outer_spec_prep,
            strongest_comparator_name=strongest_comp_name,
        )
        subspace_basis = map_subspace_to_metric_orthonormal_basis(
            outer_spec_prep,
            sel_k,
            V_outer_r,
            p1,
            p2,
        )
        pca_subspace_basis = map_subspace_to_metric_orthonormal_basis(
            outer_spec_prep,
            sel_pca_k,
            np.eye(sel_pca_k, dtype=float),
            p1,
            p2,
        )
        # Map effective regression coefficients back to canonical normalized spectral channels:
        # s_is = (x_is - mean)^T w_spec + const, preserving exact MFA scaling and PCA/paired projections.
        beta_pair_eff = fit_paired_outer.coef_spec[0] / fit_paired_outer.z_scale
        paired_spec_w = outer_spec_prep.pca_components[:sel_k].T @ (V_outer_r @ beta_pair_eff)
        paired_spec_w[:p1] *= outer_spec_prep.scale_1300
        paired_spec_w[p1:] *= outer_spec_prep.scale_2850

        beta_pca_eff = fit_pca_outer.coef_spec[0] / fit_pca_outer.z_scale
        pca_spec_w = outer_spec_prep.pca_components[:sel_pca_k].T @ beta_pca_eff
        pca_spec_w[:p1] *= outer_spec_prep.scale_1300
        pca_spec_w[p1:] *= outer_spec_prep.scale_2850

        pls_idx_eff = min(max(int(best_pls_c), 1), fit_pls_outer.max_components) - 1
        pls_spec_w = fit_pls_outer.coefs_by_comp[pls_idx_eff, fit_pls_outer.p_cov :].copy()
        pls_spec_w[:p1] *= outer_spec_prep.scale_1300
        pls_spec_w[p1:] *= outer_spec_prep.scale_2850
    else:
        repeatability_info = {}
        subspace_basis = np.empty((0, 0), dtype=float)
        pca_subspace_basis = np.empty((0, 0), dtype=float)
        paired_spec_w = np.empty(0, dtype=float)
        pca_spec_w = np.empty(0, dtype=float)
        pls_spec_w = np.empty(0, dtype=float)

    tr_T_outer = float(np.trace(outer_moments.T_max[:sel_k, :sel_k]))
    tr_W_outer = float(np.trace(outer_moments.W_max[:sel_k, :sel_k]))
    mean_ab_shift_l2 = float(linalg.norm(outer_moments.mean_ab_shift[:sel_k]))
    tr_W_over_tr_T = float(tr_W_outer / tr_T_outer) if tr_T_outer > 1e-15 else float("nan")
    shift_share_of_tr_W = (
        float(0.5 * (mean_ab_shift_l2**2) / tr_W_outer) if tr_W_outer > 1e-15 else float("nan")
    )

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
            "inner_mse_covariate_ridge": float(best_cov_mse),
            "inner_mse_fixed_band_ridge": float(best_band_mse),
            "inner_mse_pca_ridge": float(best_pca_mse),
            "inner_mse_pls_regression": float(best_pls_mse),
            "inner_mse_paired_linear_ridge": float(best_paired_mse),
            "n_paired_distinct_train": int(outer_moments.n_paired_distinct),
            "n_paired_multiplicity_train": float(outer_moments.n_paired_multiplicity),
            "numerical_rank_train": int(outer_spec_prep.numerical_rank),
            "inner_min_paired_distinct": int(inner_min_paired_distinct),
            "inner_max_paired_distinct": int(inner_max_paired_distinct),
            "inner_min_numerical_rank": int(inner_min_numerical_rank),
            "inner_dropped_covariates": json.dumps(list(inner_dropped_covariates)),
            "tr_T": tr_T_outer,
            "tr_W": tr_W_outer,
            "tr_W_over_tr_T": tr_W_over_tr_T,
            "mean_ab_shift_l2": mean_ab_shift_l2,
            "mean_ab_shift_share_of_tr_W": shift_share_of_tr_W,
            "tau": float(tau_outer),
            "top_rho": float(eigvals_outer[0]),
            "top_rhos_json": json.dumps([round(float(v), 10) for v in eigvals_outer]),
            "dropped_covariates": outer_cov_prep.dropped_columns,
            "included_missing_indicators": json.dumps(list(outer_cov_prep.include_missing_indicators_for)),
            "n_invalid_candidates": len(invalid_candidates),
            "invalid_candidate_reasons": json.dumps(invalid_candidates),
        },
        "repeatability": repeatability_info,
        "subspace_basis": subspace_basis,
        "pca_subspace_basis": pca_subspace_basis,
        "paired_spectral_weights": paired_spec_w,
        "pca_spectral_weights": pca_spec_w,
        "pls_spectral_weights": pls_spec_w,
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
            ext_sb, _ = extract_sample_band_matrix(ext_ds)
            ext_pb = aggregate_samples_to_participants(
                ext_sb, ext_ds, single_sample_only=single_sample_downstream
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
                part_band_mat=ext_pb,
                raw_cache=ext_ds._cache,
            )
        X_cov_ext = outer_cov_prep.transform(
            prepared_ext.raw_cov_frame, prepared_ext.all_indices, raw_cache=prepared_ext.raw_cache
        )
        pred_comp_ext: np.ndarray | None = None
        if strongest_comp_name == "fixed_band_ridge":
            if prepared_ext.part_band_mat is None:
                raise ValueError(
                    "PreparedExternalTest.part_band_mat is required to score fixed_band_ridge externally; "
                    "construct via PreparedExternalTest.from_dataset(...)."
                )
            pred_comp_ext, _ = predict_ridge_grid(
                fit_band_outer, X_cov_ext, prepared_ext.part_band_mat, alpha_idx=0
            )
        elif strongest_comp_name == "pca_ridge":
            ext_part_pca = outer_spec_prep.transform_pca(
                prepared_ext.spec_1300, prepared_ext.spec_2850, k=sel_pca_k
            )
            pred_comp_ext, _ = predict_ridge_grid(fit_pca_outer, X_cov_ext, ext_part_pca, alpha_idx=0)
        elif strongest_comp_name == "pls_regression":
            pred_comp_ext, _ = fit_pls_outer.predict_from_raw_blocks(
                outer_spec_prep,
                X_cov_ext,
                prepared_ext.spec_1300,
                prepared_ext.spec_2850,
                n_components=best_pls_c,
            )
        ext_mse, ext_deltas, ext_deltas_comp = _score_external_fold(
            prepared_ext,
            outer_cov_prep,
            outer_spec_prep,
            fit_cov_outer,
            fit_paired_outer,
            sel_k,
            V_outer_r,
            X_cov_ext=X_cov_ext,
            pred_comp_ext=pred_comp_ext,
        )
        out["external_mse"] = ext_mse
        out["external_participant_deltas"] = ext_deltas
        if ext_deltas_comp is not None:
            out["external_participant_deltas_vs_comparator"] = ext_deltas_comp

    if include_band_paired:
        assert sample_band_std is not None
        out["predictions"]["band_paired_ridge"] = _band_paired_ablation_predictions(
            dataset,
            cfg,
            sample_band_std,
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
    try:
        band_moments = PairedMoments.from_sample_coordinates(
            sample_band_mat, dataset, outer_train_idx, outer_w_tr
        )
        k_band = sample_band_mat.shape[1]
        V_b, _, _ = band_moments.solve_directions(k_band, lam, eigenvalue_tolerance=cfg.eigenvalue_tolerance)
    except (ValueError, linalg.LinAlgError):
        return np.full(outer_test_idx.size, np.nan, dtype=float)
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
    single_sample_downstream: bool | None = None,
    participant_multiplicities: np.ndarray | None = None,
    allow_missing_outcome: bool = False,
) -> dict[str, Any]:
    """Fit outer covariate, paired, and inner-selected comparator models per fold and score externally."""
    eff_single_sample = _effective_single_sample_downstream(dataset, single_sample_downstream)
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
        single_sample_downstream=eff_single_sample,
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
    sample_band_mat: np.ndarray | None = None
    part_band_mat: np.ndarray | None = None
    part_spectra: tuple[np.ndarray, np.ndarray] | None = None

    alphas = list(cfg.ridge_alpha_grid)
    external_fold_deltas: list[float] = []
    external_fold_deltas_comp: list[float] = []
    external_part_delta_sum = np.zeros(prepared_ext.n_participants, dtype=float)
    external_part_delta_comp_sum = np.zeros(prepared_ext.n_participants, dtype=float)

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
            single_sample_downstream=eff_single_sample,
            include_band_paired=False,
        )
        outer_cov_prep = CovariatePreprocessor.fit(
            dataset.covariate_frame, tr_idx, outer_w_tr, preset=cfg.covariate_preset, raw_cache=dataset._cache
        )
        X_cov_otr = outer_cov_prep.transform(dataset.covariate_frame, tr_idx, raw_cache=dataset._cache)
        X_cov_ext = outer_cov_prep.transform(
            prepared_ext.raw_cov_frame, prepared_ext.all_indices, raw_cache=prepared_ext.raw_cache
        )
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
            sample_paired_z, dataset, single_sample_only=eff_single_sample
        )
        fit_paired_outer = fit_ridge_grid(
            X_cov_otr,
            part_paired_z[tr_idx],
            y_otr,
            outer_w_tr,
            [alphas[sel_a_idx]],
        )

        # Fit only the single established comparator selected in inner validation for this outer fold.
        strongest_comp_name = _select_strongest_comparator_name(
            selection.band_mse, selection.pca_mse, selection.pls_mse
        )
        pred_comp_ext: np.ndarray | None = None
        if strongest_comp_name == "fixed_band_ridge":
            if prepared_ext.part_band_mat is None:
                raise ValueError(
                    "PreparedExternalTest.part_band_mat is required to score fixed_band_ridge externally; "
                    "construct via PreparedExternalTest.from_dataset(...)."
                )
            if part_band_mat is None:
                sample_band_mat, _ = extract_sample_band_matrix(dataset)
                part_band_mat = aggregate_samples_to_participants(
                    sample_band_mat, dataset, single_sample_only=eff_single_sample
                )
            fit_band_outer = fit_ridge_grid(
                X_cov_otr,
                part_band_mat[tr_idx],
                y_otr,
                outer_w_tr,
                [alphas[selection.band_alpha_idx]],
            )
            pred_comp_ext, _ = predict_ridge_grid(
                fit_band_outer, X_cov_ext, prepared_ext.part_band_mat, alpha_idx=0
            )
        elif strongest_comp_name == "pca_ridge":
            sel_pca_k, sel_pca_a_idx = selection.pca_setting
            sample_pca_comp = (
                sample_pca_outer[:, :sel_pca_k]
                if sel_pca_k <= sel_k
                else outer_spec_prep.transform_pca(dataset.sample_1300, dataset.sample_2850, k=sel_pca_k)
            )
            part_pca_comp = aggregate_samples_to_participants(
                sample_pca_comp, dataset, single_sample_only=eff_single_sample
            )
            fit_pca_outer = fit_ridge_grid(
                X_cov_otr,
                part_pca_comp[tr_idx],
                y_otr,
                outer_w_tr,
                [alphas[sel_pca_a_idx]],
            )
            ext_part_pca = outer_spec_prep.transform_pca(
                prepared_ext.spec_1300, prepared_ext.spec_2850, k=sel_pca_k
            )
            pred_comp_ext, _ = predict_ridge_grid(fit_pca_outer, X_cov_ext, ext_part_pca, alpha_idx=0)
        elif strongest_comp_name == "pls_regression":
            if part_spectra is None:
                part_spectra = _get_participant_spectra(
                    dataset, single_sample_downstream=eff_single_sample
                )
            part_1300, part_2850 = part_spectra
            part_scaled_tr = outer_spec_prep.transform_scaled_blocks(part_1300[tr_idx], part_2850[tr_idx])
            fit_pls_outer = WeightedPLSResult.fit(
                X_cov_otr,
                part_scaled_tr,
                y_otr,
                outer_w_tr,
                selection.pls_components,
            )
            pred_comp_ext, _ = fit_pls_outer.predict_from_raw_blocks(
                outer_spec_prep,
                X_cov_ext,
                prepared_ext.spec_1300,
                prepared_ext.spec_2850,
                n_components=selection.pls_components,
            )

        ext_mse, ext_deltas, ext_deltas_comp = _score_external_fold(
            prepared_ext,
            outer_cov_prep,
            outer_spec_prep,
            fit_cov_outer,
            fit_paired_outer,
            sel_k,
            V_outer_r,
            X_cov_ext=X_cov_ext,
            pred_comp_ext=pred_comp_ext,
        )
        external_fold_deltas.append(ext_mse["covariate_ridge"] - ext_mse["paired_linear_ridge"])
        external_part_delta_sum += ext_deltas
        if ext_deltas_comp is not None:
            external_fold_deltas_comp.append(
                ext_mse["strongest_comparator"] - ext_mse["paired_linear_ridge"]
            )
            external_part_delta_comp_sum += ext_deltas_comp

    n_folds_ext = max(len(external_fold_deltas), 1)
    out: dict[str, Any] = {
        "endpoint": target_col,
        "external_delta_mse": float(np.mean(external_fold_deltas)),
        "external_participant_deltas": external_part_delta_sum / float(n_folds_ext),
    }
    if external_fold_deltas_comp:
        n_folds_comp = max(len(external_fold_deltas_comp), 1)
        out["external_delta_vs_comparator"] = float(np.mean(external_fold_deltas_comp))
        out["external_participant_deltas_vs_comparator"] = external_part_delta_comp_sum / float(n_folds_comp)
    return out


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
    single_sample_downstream: bool | None = None,
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
    eff_single_sample = _effective_single_sample_downstream(dataset, single_sample_downstream)
    target_col = endpoint or cfg.primary_endpoint
    if target_col not in dataset.outcomes:
        raise ValueError(f"Endpoint '{target_col}' not present in dataset outcomes.")
    y_all = dataset.outcomes[target_col]
    prepared_ext = _prepare_external_test(
        external_test,
        target_col,
        single_sample_downstream=eff_single_sample,
        preset=cfg.covariate_preset,
    )
    external_fold_deltas: list[float] = []
    external_fold_deltas_comp: list[float] = []
    external_part_delta_sum: np.ndarray | None = (
        np.zeros(prepared_ext.n_participants, dtype=float) if prepared_ext is not None else None
    )
    external_part_delta_comp_sum: np.ndarray | None = (
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
        sample_band_mat, dataset, single_sample_only=eff_single_sample
    )
    part_spectra = _get_participant_spectra(
        dataset, single_sample_downstream=eff_single_sample
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
    pca_subspace_bases: list[np.ndarray] = []
    fold_paired_weights: list[np.ndarray] = []
    fold_pca_weights: list[np.ndarray] = []
    fold_pls_weights: list[np.ndarray] = []

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
            single_sample_downstream=eff_single_sample,
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
            if "strongest_comparator" in ext_mse and "external_participant_deltas_vs_comparator" in fold_out:
                external_fold_deltas_comp.append(
                    ext_mse["strongest_comparator"] - ext_mse["paired_linear_ridge"]
                )
                if external_part_delta_comp_sum is not None:
                    external_part_delta_comp_sum += fold_out["external_participant_deltas_vs_comparator"]
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
            pca_subspace_bases.append(fold_out["pca_subspace_basis"])
            fold_paired_weights.append(fold_out["paired_spectral_weights"])
            fold_pca_weights.append(fold_out["pca_spectral_weights"])
            fold_pls_weights.append(fold_out["pls_spectral_weights"])

    # Compute endpoint summaries for every model relative to covariate-only ridge.
    # If any active test participant lacks a finite prediction (e.g., singular secondary ablation),
    # report the model non-estimable rather than silently dropping participants.
    comparisons: dict[str, dict[str, float]] = {}
    active_eval = observed & (mult > 0.0)
    empty_arr = np.empty(0, dtype=float)
    for m in model_names:
        if np.any(active_eval) and not np.all(np.isfinite(oof_preds[m][active_eval])):
            comparisons[m] = compute_endpoint_summary(empty_arr, empty_arr, empty_arr)
        else:
            comparisons[m] = compute_endpoint_summary(
                y_all, oof_preds["covariate_ridge"], oof_preds[m], weights=mult
            )

    # Also compute direct paired_linear_ridge vs strongest_comparator difference.
    vs_strongest = compute_endpoint_summary(
        y_all, oof_preds["strongest_comparator"], oof_preds["paired_linear_ridge"], weights=mult
    )

    if compute_diagnostics:
        angles_paired = compute_subspace_principal_angles(subspace_bases, model_name="paired_linear_ridge")
        angles_pca = compute_subspace_principal_angles(pca_subspace_bases, model_name="pca_ridge")
        non_empty_angles = [df for df in (angles_paired, angles_pca) if not df.empty]
        angles_df = pd.concat(non_empty_angles, ignore_index=True) if non_empty_angles else pd.DataFrame()

        wn_1300 = dataset.wavenumbers_1300
        wn_2850 = dataset.wavenumbers_2850
        blocks = ["1300"] * int(wn_1300.size) + ["2850"] * int(wn_2850.size)
        wns = np.concatenate([wn_1300, wn_2850])
        weights_dict: dict[str, Any] = {
            "block": blocks,
            "wavenumber_cm1": wns,
            "raman_band": _canonical_raman_band_labels(wn_1300, wn_2850),
        }
        if fold_paired_weights:
            paired_mat = np.column_stack(fold_paired_weights)
            for f_idx in range(paired_mat.shape[1]):
                weights_dict[f"paired_weight_fold_{f_idx}"] = paired_mat[:, f_idx]
            weights_dict["paired_weight_mean"] = np.mean(paired_mat, axis=1)
            weights_dict["paired_weight_sd"] = (
                np.std(paired_mat, axis=1, ddof=1) if paired_mat.shape[1] > 1 else np.zeros(wns.size)
            )
            weights_dict["paired_weight_abs_mean"] = np.mean(np.abs(paired_mat), axis=1)
        if fold_pca_weights:
            pca_mat = np.column_stack(fold_pca_weights)
            weights_dict["pca_weight_mean"] = np.mean(pca_mat, axis=1)
        if fold_pls_weights:
            pls_mat = np.column_stack(fold_pls_weights)
            weights_dict["pls_weight_mean"] = np.mean(pls_mat, axis=1)
        fold_spectral_weights_df = pd.DataFrame(weights_dict)
    else:
        angles_df = pd.DataFrame()
        fold_spectral_weights_df = pd.DataFrame()

    extra: dict[str, Any] = {}
    if prepared_ext is not None:
        n_folds_ext = max(len(external_fold_deltas), 1)
        extra["external_delta_mse"] = float(np.mean(external_fold_deltas))
        if external_part_delta_sum is not None:
            extra["external_participant_deltas"] = external_part_delta_sum / float(n_folds_ext)
        if external_fold_deltas_comp:
            n_folds_comp = max(len(external_fold_deltas_comp), 1)
            extra["external_delta_vs_comparator"] = float(np.mean(external_fold_deltas_comp))
            if external_part_delta_comp_sum is not None:
                extra["external_participant_deltas_vs_comparator"] = (
                    external_part_delta_comp_sum / float(n_folds_comp)
                )
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
        "fold_spectral_weights": fold_spectral_weights_df,
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


def _summarize_bootstrap_contrast(
    point_est: float,
    sd_0: float,
    draws: np.ndarray,
    draw_sds: np.ndarray,
    *,
    interval_method: IntervalMethod,
    valid: bool,
) -> dict[str, Any]:
    """Compute centered studentized-t or percentile 95% CI, bias, SD, and Maritz-Jarrett MC SEs."""
    nan_pair = (float("nan"), float("nan"))
    if not valid or draws.size == 0 or not np.isfinite(point_est):
        return {
            "ci": nan_pair,
            "ci_percentile": nan_pair,
            "boot_sd": float("nan"),
            "boot_bias": float("nan"),
            "mc_se_mean": float("nan"),
            "mc_se_ci_low": float("nan"),
            "mc_se_ci_high": float("nan"),
            "pivot_quantiles": nan_pair,
        }
    pct_lo = float(np.percentile(draws, 2.5))
    pct_hi = float(np.percentile(draws, 97.5))
    boot_mean = float(np.mean(draws))
    boot_sd = float(np.std(draws, ddof=1)) if draws.size > 1 else float("nan")
    boot_bias = float(boot_mean - point_est)
    mc_se_mean = float(boot_sd / math.sqrt(draws.size)) if draws.size > 1 else float("nan")
    if interval_method == "studentized_t":
        if (
            not np.isfinite(sd_0)
            or sd_0 <= 1e-12
            or draw_sds.size != draws.size
            or not np.all(np.isfinite(draw_sds))
            or np.any(draw_sds <= 1e-12)
        ):
            return {
                "ci": nan_pair,
                "ci_percentile": (pct_lo, pct_hi),
                "boot_sd": boot_sd,
                "boot_bias": boot_bias,
                "mc_se_mean": mc_se_mean,
                "mc_se_ci_low": float("nan"),
                "mc_se_ci_high": float("nan"),
                "pivot_quantiles": nan_pair,
            }
        t_star = (draws - boot_mean) / draw_sds
        q_lo = float(np.percentile(t_star, 2.5))
        q_hi = float(np.percentile(t_star, 97.5))
        ci_lo = float(point_est - q_hi * sd_0)
        ci_hi = float(point_est - q_lo * sd_0)
        mc_se_lo = float(sd_0 * _maritz_jarrett_quantile_se(t_star, 0.975))
        mc_se_hi = float(sd_0 * _maritz_jarrett_quantile_se(t_star, 0.025))
        pivot_q = (q_lo, q_hi)
    else:
        ci_lo, ci_hi = pct_lo, pct_hi
        mc_se_lo = _maritz_jarrett_quantile_se(draws, 0.025)
        mc_se_hi = _maritz_jarrett_quantile_se(draws, 0.975)
        pivot_q = nan_pair
    return {
        "ci": (ci_lo, ci_hi),
        "ci_percentile": (pct_lo, pct_hi),
        "boot_sd": boot_sd,
        "boot_bias": boot_bias,
        "mc_se_mean": mc_se_mean,
        "mc_se_ci_low": mc_se_lo,
        "mc_se_ci_high": mc_se_hi,
        "pivot_quantiles": pivot_q,
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
    allow_missing_outcome: bool = False,
    interval_method: IntervalMethod = "studentized_t",
    single_sample_downstream: bool | None = None,
) -> dict[str, Any]:
    """Participant-level multiplicity bootstrap keeping original fold memberships fixed."""
    if interval_method not in INTERVAL_METHODS:
        raise ValueError(
            f"Unknown interval_method: {interval_method!r}. Expected one of {INTERVAL_METHODS}."
        )
    eff_single_sample = _effective_single_sample_downstream(dataset, single_sample_downstream)
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
    paired_idx_attr = getattr(dataset, "paired_participant_indices", None)
    if paired_idx_attr is not None and np.asarray(paired_idx_attr).size > 0 and n_bootstraps > 0:
        paired_p_arr = np.asarray(paired_idx_attr, dtype=int)
        mean_distinct_paired = float(np.mean(np.count_nonzero(M_all[:, paired_p_arr] > 0.0, axis=1)))
    else:
        mean_distinct_paired = float("nan")

    gcache = _get_dataset_gram_cache(dataset, cfg)
    base_weights = observed.astype(np.float64)[None, :]
    M_with_base = np.vstack([base_weights, M_all])
    if eff_single_sample:
        # Batched Gram cache averages paired samples; single_sample downstream must use sequential refits.
        elig_with_base = np.zeros(n_bootstraps + 1, dtype=bool)
    else:
        elig_with_base = _check_fast_batch_eligibility(
            dataset, outer_splits, inner_splits_per_fold, M_with_base, cfg, gcache
        )
    base_elig = bool(elig_with_base[0])
    fast_mask = elig_with_base[1:]
    fast_idx = np.flatnonzero(fast_mask)
    fallback_idx = np.flatnonzero(~fast_mask)

    d_pair_arr = np.full(n_bootstraps, np.nan, dtype=np.float64)
    d_comp_arr = np.full(n_bootstraps, np.nan, dtype=np.float64)
    d_strong_arr = np.full(n_bootstraps, np.nan, dtype=np.float64)
    sd_pair_arr = np.full(n_bootstraps, np.nan, dtype=np.float64)
    sd_comp_arr = np.full(n_bootstraps, np.nan, dtype=np.float64)
    sd_strong_arr = np.full(n_bootstraps, np.nan, dtype=np.float64)

    failed_draws = 0
    failure_reasons: list[str] = []
    exception_draws: set[int] = set()
    extra_fallback_draws: set[int] = set()
    execution: dict[str, Any] = {"device": "sequential_cpu", "gpu_requested": use_gpu}
    point_est_delta: float | None = None
    point_est_comp: float | None = None
    point_est_strong: float | None = None
    sd_0_delta = float("nan")
    sd_0_comp = float("nan")
    sd_0_strong = float("nan")

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
        if _supports_keyword(_run_batched_multiplicity_draws, "return_oof_sd"):
            batched_kwargs["return_oof_sd"] = True
        batched_out = _run_batched_multiplicity_draws(
            dataset,
            y_all,
            outer_splits,
            inner_splits_per_fold,
            M_run,
            cfg,
            gcache,
            **batched_kwargs,
        )
        if isinstance(batched_out, dict):
            dp_fast = batched_out["delta_pair"]
            dc_fast = batched_out["delta_comp"]
            ds_fast = batched_out["delta_strong"]
            sdp_fast = batched_out["sd_pair"]
            sdc_fast = batched_out["sd_comp"]
            sds_fast = batched_out["sd_strong"]
        else:
            dp_fast, dc_fast = batched_out
            ds_fast = np.full_like(dp_fast, np.nan)
            sdp_fast = np.full_like(dp_fast, np.nan)
            sdc_fast = np.full_like(dc_fast, np.nan)
            sds_fast = np.full_like(dp_fast, np.nan)

        if base_elig:
            base_sds_ok = interval_method != "studentized_t" or (
                np.isfinite(sdp_fast[0]) and np.isfinite(sdc_fast[0])
            )
            if 0 not in batched_fallback_sub and np.isfinite(dp_fast[0]) and np.isfinite(dc_fast[0]) and base_sds_ok:
                point_est_delta = float(dp_fast[0])
                point_est_comp = float(dc_fast[0])
                point_est_strong = float(ds_fast[0])
                sd_0_delta = float(sdp_fast[0])
                sd_0_comp = float(sdc_fast[0])
                sd_0_strong = float(sds_fast[0])
            d_pair_arr[fast_idx] = dp_fast[1:]
            d_comp_arr[fast_idx] = dc_fast[1:]
            d_strong_arr[fast_idx] = ds_fast[1:]
            sd_pair_arr[fast_idx] = sdp_fast[1:]
            sd_comp_arr[fast_idx] = sdc_fast[1:]
            sd_strong_arr[fast_idx] = sds_fast[1:]
            for sub_idx in batched_fallback_sub:
                if sub_idx > 0:
                    extra_fallback_draws.add(int(fast_idx[sub_idx - 1]))
        else:
            d_pair_arr[fast_idx] = dp_fast
            d_comp_arr[fast_idx] = dc_fast
            d_strong_arr[fast_idx] = ds_fast
            sd_pair_arr[fast_idx] = sdp_fast
            sd_comp_arr[fast_idx] = sdc_fast
            sd_strong_arr[fast_idx] = sds_fast
            for sub_idx in batched_fallback_sub:
                extra_fallback_draws.add(int(fast_idx[sub_idx]))

    eval_extra_kwargs: dict[str, Any] = {}
    if allow_missing_outcome:
        eval_extra_kwargs["allow_missing_outcome"] = True
    if eff_single_sample:
        eval_extra_kwargs["single_sample_downstream"] = True

    if point_est_delta is None or point_est_comp is None:
        base_eval = evaluate_nested_cohort(
            dataset,
            cfg=cfg,
            endpoint=endpoint,
            precomputed_splits=splits,
            compute_diagnostics=False,
            **eval_extra_kwargs,
        )
        m_pair_0 = base_eval["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]
        m_comp_0 = base_eval["paired_vs_strongest_comparator"]
        m_strong_0 = base_eval["comparisons_vs_covariate_ridge"].get("strongest_comparator", {})
        point_est_delta = float(m_pair_0["delta_mse"])
        point_est_comp = float(m_comp_0["delta_mse"])
        point_est_strong = float(m_strong_0.get("delta_mse", float("nan")))
        sd_0_delta = float(m_pair_0.get("oof_loss_diff_sd", float("nan")))
        sd_0_comp = float(m_comp_0.get("oof_loss_diff_sd", float("nan")))
        sd_0_strong = float(m_strong_0.get("oof_loss_diff_sd", float("nan")))

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
            m_pair_b = b_res["comparisons_vs_covariate_ridge"]["paired_linear_ridge"]
            m_comp_b = b_res["paired_vs_strongest_comparator"]
            m_strong_b = b_res["comparisons_vs_covariate_ridge"].get("strongest_comparator", {})
            d_pair_arr[b] = float(m_pair_b["delta_mse"])
            d_comp_arr[b] = float(m_comp_b["delta_mse"])
            d_strong_arr[b] = float(m_strong_b.get("delta_mse", float("nan")))
            sd_pair_arr[b] = float(m_pair_b.get("oof_loss_diff_sd", float("nan")))
            sd_comp_arr[b] = float(m_comp_b.get("oof_loss_diff_sd", float("nan")))
            sd_strong_arr[b] = float(m_strong_b.get("oof_loss_diff_sd", float("nan")))
        except Exception as exc:
            d_pair_arr[b] = np.nan
            d_comp_arr[b] = np.nan
            d_strong_arr[b] = np.nan
            sd_pair_arr[b] = np.nan
            sd_comp_arr[b] = np.nan
            sd_strong_arr[b] = np.nan
            exception_draws.add(int(b))
            failed_draws += 1
            failure_reasons.append(str(exc))

    delta_paired: list[float] = []
    delta_vs_comp: list[float] = []
    delta_strong: list[float] = []
    oof_sds_paired: list[float] = []
    oof_sds_vs_comp: list[float] = []
    oof_sds_strong: list[float] = []
    for b in range(n_bootstraps):
        dp = float(d_pair_arr[b])
        dc = float(d_comp_arr[b])
        ds_val = float(d_strong_arr[b])
        sdp = float(sd_pair_arr[b])
        sdc = float(sd_comp_arr[b])
        sds_val = float(sd_strong_arr[b])
        if not (np.isfinite(dp) and np.isfinite(dc)):
            if b not in exception_draws:
                failed_draws += 1
                failure_reasons.append("nonfinite_bootstrap_estimate")
            continue
        if interval_method == "studentized_t" and not (
            np.isfinite(sdp) and sdp > 1e-12 and np.isfinite(sdc) and sdc > 1e-12
        ):
            failed_draws += 1
            failure_reasons.append("degenerate_oof_loss_diff_sd")
            continue
        delta_paired.append(dp)
        delta_vs_comp.append(dc)
        delta_strong.append(ds_val)
        oof_sds_paired.append(sdp)
        oof_sds_vs_comp.append(sdc)
        oof_sds_strong.append(sds_val)

    base_sd_ok = (
        interval_method != "studentized_t"
        or n_bootstraps == 0
        or (np.isfinite(sd_0_delta) and sd_0_delta > 1e-12 and np.isfinite(sd_0_comp) and sd_0_comp > 1e-12)
    )
    if not base_sd_ok and failed_draws == 0:
        failure_reasons.append("degenerate_base_oof_loss_diff_sd")

    # Per study design: if any bootstrap refit cannot produce the locked comparison (or studentized SD),
    # mark that dataset's interval procedure as failed rather than discarding resamples.
    interval_valid = (
        n_bootstraps > 0
        and failed_draws == 0
        and len(delta_paired) == n_bootstraps
        and base_sd_ok
    )

    arr_p = np.asarray(delta_paired, dtype=float)
    arr_c = np.asarray(delta_vs_comp, dtype=float)
    arr_s = np.asarray(delta_strong, dtype=float)
    arr_sdp = np.asarray(oof_sds_paired, dtype=float)
    arr_sdc = np.asarray(oof_sds_vs_comp, dtype=float)
    arr_sds = np.asarray(oof_sds_strong, dtype=float)

    stat_p = _summarize_bootstrap_contrast(
        point_est_delta, sd_0_delta, arr_p, arr_sdp, interval_method=interval_method, valid=interval_valid
    )
    stat_c = _summarize_bootstrap_contrast(
        point_est_comp, sd_0_comp, arr_c, arr_sdc, interval_method=interval_method, valid=interval_valid
    )
    strong_valid = bool(
        interval_valid
        and point_est_strong is not None
        and np.isfinite(point_est_strong)
        and arr_s.size == n_bootstraps
        and np.all(np.isfinite(arr_s))
    )
    stat_s = _summarize_bootstrap_contrast(
        float(point_est_strong) if point_est_strong is not None else float("nan"),
        sd_0_strong,
        arr_s,
        arr_sds,
        interval_method=interval_method,
        valid=strong_valid,
    )

    return {
        "interval_method": str(interval_method),
        "point_estimate_delta_mse": point_est_delta,
        "point_estimate_vs_comparator": point_est_comp,
        "point_estimate_strongest_vs_covariate": (
            float(point_est_strong) if point_est_strong is not None else float("nan")
        ),
        "ci_95_delta_mse": stat_p["ci"],
        "ci_95_vs_comparator": stat_c["ci"],
        "ci_95_strongest_vs_covariate": stat_s["ci"],
        "ci_95_percentile_delta_mse": stat_p["ci_percentile"],
        "ci_95_percentile_vs_comparator": stat_c["ci_percentile"],
        "ci_95_percentile_strongest_vs_covariate": stat_s["ci_percentile"],
        "interval_procedure_succeeded": interval_valid,
        "n_bootstraps_requested": n_bootstraps,
        "n_bootstraps_succeeded": len(delta_paired),
        "n_failed_refits": failed_draws,
        "n_fallback_draws": len(all_fallback_idx),
        "execution": execution,
        "failure_reasons": failure_reasons[:10],
        "mean_distinct_participants": float(np.mean(distinct_counts)) if distinct_counts.size else float("nan"),
        "min_distinct_participants": int(np.min(distinct_counts)) if distinct_counts.size else 0,
        "max_distinct_participants": int(np.max(distinct_counts)) if distinct_counts.size else 0,
        "mean_distinct_paired_participants": mean_distinct_paired,
        "oof_loss_diff_sd_delta_mse": sd_0_delta,
        "oof_loss_diff_sd_vs_comparator": sd_0_comp,
        "oof_loss_diff_sd_strongest_vs_covariate": sd_0_strong,
        "bootstrap_sd_delta_mse": stat_p["boot_sd"],
        "bootstrap_sd_vs_comparator": stat_c["boot_sd"],
        "bootstrap_sd_strongest_vs_covariate": stat_s["boot_sd"],
        "bootstrap_bias_delta_mse": stat_p["boot_bias"],
        "bootstrap_bias_vs_comparator": stat_c["boot_bias"],
        "bootstrap_bias_strongest_vs_covariate": stat_s["boot_bias"],
        "pivot_quantiles_delta_mse": stat_p["pivot_quantiles"],
        "pivot_quantiles_vs_comparator": stat_c["pivot_quantiles"],
        "pivot_quantiles_strongest_vs_covariate": stat_s["pivot_quantiles"],
        # SE of the bootstrap mean, retained alongside Maritz-Jarrett CI-endpoint MC SEs.
        "monte_carlo_se_of_mean": stat_p["mc_se_mean"],
        "monte_carlo_se_of_mean_vs_comparator": stat_c["mc_se_mean"],
        "mc_se_ci_95_low_delta_mse": stat_p["mc_se_ci_low"],
        "mc_se_ci_95_high_delta_mse": stat_p["mc_se_ci_high"],
        "mc_se_ci_95_low_vs_comparator": stat_c["mc_se_ci_low"],
        "mc_se_ci_95_high_vs_comparator": stat_c["mc_se_ci_high"],
        "mc_se_ci_95_low_strongest_vs_covariate": stat_s["mc_se_ci_low"],
        "mc_se_ci_95_high_strongest_vs_covariate": stat_s["mc_se_ci_high"],
        "bootstrap_draws_delta_mse": arr_p,
        "bootstrap_draws_vs_comparator": arr_c,
        "bootstrap_draws_strongest_vs_covariate": arr_s,
        "bootstrap_oof_sds_delta_mse": arr_sdp,
        "bootstrap_oof_sds_vs_comparator": arr_sdc,
        "bootstrap_oof_sds_strongest_vs_covariate": arr_sds,
    }


def _residualize_and_partial_r(
    design: np.ndarray,
    s_vals_vec: np.ndarray,
    y_vals_vec: np.ndarray,
    eps_f: float,
) -> tuple[float, np.ndarray, np.ndarray, float, int]:
    """Residualize s and y on the rank-revealing orthonormal column space of design."""
    u_mat, sv, _ = linalg.svd(design, full_matrices=False, check_finite=False)
    if sv.size > 0 and sv[0] > 0.0:
        tol = max(design.shape) * eps_f * float(sv[0])
        u_kept = u_mat[:, sv > tol]
        rank_design = int(u_kept.shape[1])
        s_res = s_vals_vec - u_kept @ (u_kept.T @ s_vals_vec)
        y_res = y_vals_vec - u_kept @ (u_kept.T @ y_vals_vec)
    else:
        rank_design = 1
        s_res = s_vals_vec - float(np.mean(s_vals_vec))
        y_res = y_vals_vec - float(np.mean(y_vals_vec))
    denom = float(linalg.norm(s_res) * linalg.norm(y_res))
    partial_r = float(np.dot(s_res, y_res) / denom) if denom > 1e-12 else 0.0
    return partial_r, s_res, y_res, denom, rank_design


def summarize_anchor_associations(
    dataset: CohortDataset,
    eval_result: dict[str, Any],
    *,
    anchors: Sequence[str] = ("ldl_mmol_l", "hdl_mmol_l", "triglycerides_mmol_l", "plt_10e9_l"),
    n_resamples: int = 2000,
    seed: int = 42,
    cfg: PairedLipidConfig = PairedLipidConfig(),
) -> pd.DataFrame:
    """Compute continuous anchor associations and resample stability without categorical source claims."""
    spec_contrib = eval_result["oof_spectral_contributions"]["paired_linear_ridge"]
    raw_cov_mat, _ = CovariatePreprocessor.raw_design_matrix(
        dataset.covariate_frame, preset=cfg.covariate_preset, raw_cache=dataset._cache
    )
    outer_splits = eval_result["splits"][0] if "splits" in eval_result else None
    rng = np.random.default_rng(seed)

    # Reuse the valid-subset covariate design and SVD basis across anchors sharing the same valid mask.
    basis_cache: dict[bytes, tuple[np.ndarray, np.ndarray, bool]] = {}
    rows: list[dict[str, Any]] = []
    eps_f = float(np.finfo(float).eps)
    for anchor in anchors:
        if anchor not in dataset.outcomes:
            continue
        y_a = dataset.outcomes[anchor]
        valid = np.isfinite(spec_contrib) & np.isfinite(y_a)
        valid_idx = np.flatnonzero(valid)
        n_v = int(valid_idx.size)
        if n_v < 5:
            continue
        s_v = spec_contrib[valid_idx]
        y_v = y_a[valid_idx]

        mask_key = valid.tobytes()
        cached_basis = basis_cache.get(mask_key)
        if cached_basis is None:
            cov_prep = CovariatePreprocessor.fit(
                dataset.covariate_frame,
                valid_idx,
                np.ones(n_v, dtype=float),
                preset=cfg.covariate_preset,
                raw_cache=dataset._cache,
            )
            X_cov_valid = cov_prep.transform(
                dataset.covariate_frame, valid_idx, raw_cache=dataset._cache
            )
            design = np.column_stack([np.ones(n_v, dtype=float), X_cov_valid])
            u_mat, s_vals, _ = linalg.svd(design, full_matrices=False)
            if s_vals.size > 0 and s_vals[0] > 0.0:
                tol = max(design.shape) * eps_f * float(s_vals[0])
                u_kept = u_mat[:, s_vals > tol]
            else:
                u_kept = np.empty((n_v, 0), dtype=float)
            has_cov_missing = bool(not np.all(np.isfinite(raw_cov_mat[valid_idx])))
            cached_basis = (design, u_kept, has_cov_missing)
            basis_cache[mask_key] = cached_basis
        else:
            design, u_kept, has_cov_missing = cached_basis

        # Residualize both s_v and y_v on the rank-revealing orthonormal covariate basis.
        s_res = s_v - u_kept @ (u_kept.T @ s_v)
        y_res = y_v - u_kept @ (u_kept.T @ y_v)

        denom = float(linalg.norm(s_res) * linalg.norm(y_res))
        if denom > 1e-12 and float(np.ptp(s_res)) > 1e-12 and float(np.ptp(y_res)) > 1e-12:
            partial_r = float(np.dot(s_res, y_res) / denom)
            spearman_rho = float(stats.spearmanr(s_res, y_res).statistic)
        else:
            partial_r = 0.0
            spearman_rho = 0.0

        # Per-fold partial correlations avoid pooling heterogeneous outer-fold models.
        fold_rs: list[float] = []
        if outer_splits is not None:
            for _, te_idx in outer_splits:
                fold_valid = te_idx[valid[te_idx]]
                if fold_valid.size < 5:
                    continue
                f_cov_prep = CovariatePreprocessor.fit(
                    dataset.covariate_frame,
                    fold_valid,
                    np.ones(fold_valid.size, dtype=float),
                    preset=cfg.covariate_preset,
                    raw_cache=dataset._cache,
                )
                X_cov_f = f_cov_prep.transform(
                    dataset.covariate_frame, fold_valid, raw_cache=dataset._cache
                )
                d_f = np.column_stack([np.ones(fold_valid.size, dtype=float), X_cov_f])
                r_f, _, _, dn_f, rank_f = _residualize_and_partial_r(
                    d_f, spec_contrib[fold_valid], y_a[fold_valid], eps_f
                )
                # Require >= 2 residual degrees of freedom so saturated folds do not inject bogus zeros.
                if int(fold_valid.size) - rank_f >= 2 and dn_f > 1e-12:
                    fold_rs.append(r_f)

        if n_resamples > 0:
            # Re-residualize on resampled covariates per participant draw so covariate-adjustment
            # uncertainty is preserved; refit CovariatePreprocessor per draw when missingness exists.
            idx_mat = rng.integers(0, n_v, size=(n_resamples, n_v))
            boot_rs: list[float] = []
            for b_idx in idx_mat:
                if has_cov_missing:
                    boot_mult = np.bincount(b_idx, minlength=n_v).astype(float)
                    cp_b = CovariatePreprocessor.fit(
                        dataset.covariate_frame,
                        valid_idx,
                        boot_mult,
                        preset=cfg.covariate_preset,
                        raw_cache=dataset._cache,
                    )
                    X_cov_b = cp_b.transform(
                        dataset.covariate_frame, valid_idx[b_idx], raw_cache=dataset._cache
                    )
                    d_b = np.column_stack([np.ones(n_v, dtype=float), X_cov_b])
                else:
                    d_b = design[b_idx]
                r_b, _, _, dn_b, rank_b = _residualize_and_partial_r(d_b, s_v[b_idx], y_v[b_idx], eps_f)
                if n_v - rank_b >= 2 and dn_b > 1e-12:
                    boot_rs.append(r_b)
            arr_r = np.asarray(boot_rs, dtype=float)
        else:
            arr_r = np.empty(0, dtype=float)
        sign_stability = float(np.mean(np.sign(arr_r) == np.sign(partial_r))) if arr_r.size else float("nan")
        target_role = (
            "primary_supervised_target" if anchor == cfg.primary_endpoint else "unseen_secondary_anchor"
        )

        rows.append(
            {
                "anchor": anchor,
                "target_role": target_role,
                "n_participants": n_v,
                "partial_pearson_r": partial_r,
                "partial_spearman_rho": spearman_rho,
                "resample_ci_low": float(np.percentile(arr_r, 2.5)) if arr_r.size else float("nan"),
                "resample_ci_high": float(np.percentile(arr_r, 97.5)) if arr_r.size else float("nan"),
                "sign_stability": sign_stability,
                "fold_median_partial_r": float(np.median(fold_rs)) if fold_rs else float("nan"),
                "fold_min_partial_r": float(np.min(fold_rs)) if fold_rs else float("nan"),
                "fold_max_partial_r": float(np.max(fold_rs)) if fold_rs else float("nan"),
                "attribution_status": "continuous_exploratory_summary_only",
            }
        )
    return pd.DataFrame(rows)
