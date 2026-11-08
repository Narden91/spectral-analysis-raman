"""Phase 13: Minimal candidate Raman panel.

Selects a sparse panel of 5–10 spectral peaks/ratios associated with clinical
endpoints using nested cross-validated ElasticNet / L1-logistic regression,
full-pipeline permutation testing, and bootstrap selection-stability summaries.

All results are exploratory and subject-level; no diagnostic, predictive,
causal, or biomarker claims are made.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import ElasticNet
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src import get_classification_phase_dir, get_confirmatory_panel_dir, get_cross_region_phase_dir
from src import resolve_cfg as _cfg
from src.analysis.clinical_feature_associations import (
    feature_columns,
    load_disease_feature_frame,
)
from src.analysis.io_utils import write_csv_atomic
from src.analysis.phase_status import STATUS_MISSING_REQUIRED_DATA, STATUS_NOT_TESTED_GATE, PhaseStatus
from src.analysis.sklearn_utils import fit_with_convergence_capture, l1_logistic_model
from src.statistics.sequential_permutation import (
    resolve_sequential_stopping_config,
    run_curtailed_permutation_draws,
    validate_sequential_stopping_config,
)

logger = logging.getLogger(__name__)

_ELASTICNET_MAX_ITER = 5_000
_ELASTICNET_TOL = 1e-3

_DEFAULT_BINARY_TARGETS = (
    {"group_variable": "moca_severity_group", "ordered_groups": ("low_moca", "high_moca")},
    {"group_variable": "delta_moca_group", "ordered_groups": ("decline", "stable_or_improved")},
)


def _panel_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("analysis", {}).get("minimal_panel", {})


def _load_integrated_candidates(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        logger.warning("Integrated candidate table not found: %s", path)
        return None
    df = pd.read_csv(path)
    if df.empty:
        logger.info("Integrated candidate table is empty")
    return df


def _load_preregistered_panel_pool(
    disease_only: pd.DataFrame,
    panel_matrix_path: str | Path | None,
) -> pd.DataFrame:
    path = Path(panel_matrix_path) if panel_matrix_path is not None else get_confirmatory_panel_dir() / "panel_feature_matrix.csv"
    if not path.exists():
        logger.warning("Fixed-panel feature matrix not found: %s", path)
        return pd.DataFrame(index=disease_only.index)
    panel = pd.read_csv(path)
    if "subject_id" not in panel.columns:
        raise ValueError("Fixed-panel feature matrix is missing subject_id")

    panel = panel.copy()
    panel["subject_id"] = panel["subject_id"].astype(str)
    aligned = panel.set_index("subject_id").reindex(disease_only["subject_id"].astype(str))
    pool = aligned.drop(columns=["subject_id"], errors="ignore")
    pool = pool.apply(pd.to_numeric, errors="coerce")
    pool.index = disease_only.index
    return pool.dropna(axis=1, how="all")


def _region_feature_names(feature_columns: list[str], region: pd.Series) -> list[str]:
    block = str(region["spectral_block"])
    lower = float(region["start_cm-1"])
    upper = float(region["end_cm-1"])
    out: list[str] = []
    for column in feature_columns:
        if not column.startswith(f"{block}__"):
            continue
        _, raw = column.split("__", 1)
        try:
            wavenumber = float(raw)
        except ValueError:
            continue
        if lower <= wavenumber <= upper:
            out.append(column)
    return out


def _band_feature(frame: pd.DataFrame, feature_names: list[str], block: str, lower: float, upper: float) -> pd.Series:
    columns: list[str] = []
    for feature_name in feature_names:
        if not feature_name.startswith(f"{block}__"):
            continue
        wavenumber = float(feature_name.split("__", 1)[1])
        if lower <= wavenumber <= upper:
            columns.append(feature_name)
    if not columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return frame[columns].mean(axis=1)


def _build_ratio_features(frame: pd.DataFrame, feature_names: list[str], cfg: dict[str, Any]) -> pd.DataFrame:
    bands_cfg = cfg.get("whole_spectra", {}).get("bands", {})
    ratio_pairs = cfg.get("whole_spectra", {}).get("ratio_pairs", [])
    band_series: dict[str, pd.Series] = {}
    for band_name, limits in bands_cfg.items():
        lower, upper = float(limits[0]), float(limits[1])
        block = "1300" if upper < 2000 else "2850"
        band_series[str(band_name)] = _band_feature(frame, feature_names, block, lower, upper)

    ratios: dict[str, pd.Series] = {}
    for pair in ratio_pairs:
        left, right = str(pair[0]), str(pair[1])
        if left not in band_series or right not in band_series:
            continue
        denominator = band_series[right].replace(0.0, np.nan)
        ratios[f"ratio__{left}__over__{right}"] = band_series[left] / denominator
    return pd.DataFrame(ratios, index=frame.index)


def _build_feature_pool(
    frame: pd.DataFrame,
    all_feature_columns: list[str],
    integrated: pd.DataFrame | None,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    pool: dict[str, pd.Series] = {}

    # Candidate-derived models require cluster-supported regions. FDR-only and
    # fallback regions remain descriptive and never enter model selection.
    if integrated is not None and not integrated.empty and "evidence_status" in integrated.columns:
        priority = integrated.loc[
            integrated["evidence_status"].eq("cluster_supported_exploratory")
        ].copy()
        for _, region in priority.iterrows():
            columns = _region_feature_names(all_feature_columns, region)
            if not columns:
                continue
            key = f"region__{region['candidate_region_id']}"
            pool[key] = frame[columns].mean(axis=1)

    # Biochemical band ratios.
    ratios = _build_ratio_features(frame, all_feature_columns, cfg)
    for column in ratios.columns:
        pool[column] = ratios[column]

    # Individual band integrals from configured biochemical bands.
    bands_cfg = cfg.get("whole_spectra", {}).get("bands", {})
    for band_name, limits in bands_cfg.items():
        lower, upper = float(limits[0]), float(limits[1])
        block = "1300" if upper < 2000 else "2850"
        series = _band_feature(frame, all_feature_columns, block, lower, upper)
        if series.notna().any():
            pool[f"band__{band_name}"] = series

    return pd.DataFrame(pool, index=frame.index)


def _cv_strategy(n_samples: int, n_splits: int, random_state: int, stratified: bool = False, y: np.ndarray | None = None):
    if n_samples < 15:
        return LeaveOneOut()
    if stratified and y is not None:
        return StratifiedKFold(n_splits=min(n_splits, min(np.bincount(y))), shuffle=True, random_state=random_state)
    return KFold(n_splits=min(n_splits, n_samples), shuffle=True, random_state=random_state)


def _fit_elasticnet(model: ElasticNet, X: np.ndarray, y: np.ndarray, context: str) -> None:
    fit_with_convergence_capture(model, X, y, context=context)


def _fit_tuned_elasticnet(
    X: np.ndarray,
    y: np.ndarray,
    cv: KFold,
    random_state: int,
    context: str,
) -> tuple[StandardScaler, ElasticNet]:
    folds = list(cv.split(X))
    best: tuple[float, float] | None = None
    best_mse = float("inf")
    for l1_ratio in (0.1, 0.5, 0.9, 1.0):
        for alpha in np.logspace(-4, 1, 8):
            fold_mse: list[float] = []
            for train_idx, test_idx in folds:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X[train_idx])
                X_test = scaler.transform(X[test_idx])
                model = ElasticNet(
                    alpha=float(alpha),
                    l1_ratio=l1_ratio,
                    random_state=random_state,
                    max_iter=_ELASTICNET_MAX_ITER,
                    tol=_ELASTICNET_TOL,
                )
                _fit_elasticnet(model, X_train, y[train_idx], context)
                fold_mse.append(float(mean_squared_error(y[test_idx], model.predict(X_test))))
            mean_mse = float(np.mean(fold_mse))
            if mean_mse < best_mse:
                best_mse = mean_mse
                best = (float(alpha), l1_ratio)
    assert best is not None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = ElasticNet(
        alpha=best[0],
        l1_ratio=best[1],
        random_state=random_state,
        max_iter=_ELASTICNET_MAX_ITER,
        tol=_ELASTICNET_TOL,
    )
    _fit_elasticnet(model, X_scaled, y, context)
    return scaler, model


def _select_and_fit_elasticnet(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    cv: KFold,
    random_state: int,
    context: str,
    max_features: int,
    min_features: int,
) -> tuple[list[str], pd.DataFrame, StandardScaler, ElasticNet]:
    """Tune and stability-select from one inner-CV pass on an outer training set."""
    folds = list(cv.split(X))
    best: tuple[float, float, list[np.ndarray]] | None = None
    best_mse = float("inf")
    for l1_ratio in (0.1, 0.5, 0.9, 1.0):
        for alpha in np.logspace(-4, 1, 8):
            fold_mse: list[float] = []
            coefficients: list[np.ndarray] = []
            for train_idx, test_idx in folds:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X[train_idx])
                X_test = scaler.transform(X[test_idx])
                model = ElasticNet(
                    alpha=float(alpha), l1_ratio=l1_ratio, random_state=random_state,
                    max_iter=_ELASTICNET_MAX_ITER, tol=_ELASTICNET_TOL,
                )
                _fit_elasticnet(model, X_train, y[train_idx], context)
                fold_mse.append(float(mean_squared_error(y[test_idx], model.predict(X_test))))
                coefficients.append(np.asarray(model.coef_, dtype=float))
            mean_mse = float(np.mean(fold_mse))
            if mean_mse < best_mse:
                best_mse = mean_mse
                best = (float(alpha), float(l1_ratio), coefficients)
    assert best is not None
    coefficient_matrix = np.vstack(best[2])
    selection_df = pd.DataFrame(
        {
            "feature_name": feature_names,
            "selection_frequency": np.mean(np.abs(coefficient_matrix) > 1e-10, axis=0),
            "mean_absolute_coefficient": np.mean(np.abs(coefficient_matrix), axis=0),
        }
    ).sort_values(["selection_frequency", "mean_absolute_coefficient"], ascending=False).reset_index(drop=True)
    selected_frame = selection_df.loc[selection_df["selection_frequency"] > 0]
    if len(selected_frame) < min_features:
        selected_frame = selection_df.head(min_features)
    selected = selected_frame.head(max_features)["feature_name"].tolist()
    selected_idx = [feature_names.index(name) for name in selected]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[:, selected_idx])
    model = ElasticNet(
        alpha=best[0], l1_ratio=best[1], random_state=random_state,
        max_iter=_ELASTICNET_MAX_ITER, tol=_ELASTICNET_TOL,
    )
    _fit_elasticnet(model, X_scaled, y, context)
    return selected, selection_df, scaler, model


def _fit_tuned_logistic(
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    c_values: list[float],
    random_state: int,
    context: str,
) -> tuple[StandardScaler, Any, float]:
    folds = list(cv.split(X, y))
    best_c = c_values[0]
    best_score = -np.inf
    for c in c_values:
        scores: list[float] = []
        for train_idx, test_idx in folds:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            model = l1_logistic_model(c=c, random_state=random_state)
            fit_with_convergence_capture(model, X_train, y[train_idx], context=context)
            scores.append(float(balanced_accuracy_score(y[test_idx], model.predict(X_test))))
        score = float(np.mean(scores))
        if score > best_score:
            best_score = score
            best_c = c
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = l1_logistic_model(c=best_c, random_state=random_state)
    fit_with_convergence_capture(model, X_scaled, y, context=context)
    return scaler, model, best_c


def _normalize_n_jobs(n_jobs: int) -> int:
    return 1 if n_jobs == 0 else n_jobs


def _parallel_values(n_jobs: int, calls: list[Any]) -> list[Any]:
    if n_jobs == 1 or len(calls) <= 1:
        return [func(*args) for func, args in calls]
    return Parallel(n_jobs=n_jobs)(delayed(func)(*args) for func, args in calls)


def _nested_cv_elasticnet_with_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    n_outer_splits: int,
    n_inner_splits: int,
    random_state: int,
    split_random_state: int | None = None,
    max_features: int,
    min_features: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, float]]:
    """Nested CV with feature selection *inside* each outer training fold.

    Selection on the full dataset before CV is classic data leakage: the held-out
    test subjects' labels influenced which features were chosen, biasing CV metrics
    and the permutation reference distribution downward.  Here selection runs only on X_train so
    X_test subjects are never seen during feature selection.
    """
    n_samples = X.shape[0]
    split_seed = random_state if split_random_state is None else split_random_state
    outer_cv = _cv_strategy(n_samples, n_outer_splits, split_seed)
    y_pred = np.zeros_like(y, dtype=float)
    selected_alphas: list[float] = []
    selected_l1_ratios: list[float] = []

    for train_idx, test_idx in outer_cv.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test = X[test_idx]

        # Select features on training fold only — test fold is never seen here.
        inner_cv = KFold(n_splits=min(n_inner_splits, len(X_train)), shuffle=True, random_state=split_seed)
        selected_fold, _, scaler, model = _select_and_fit_elasticnet(
            X_train, y_train, feature_names, inner_cv, random_state,
            "minimal panel inner CV", max_features, min_features,
        )
        sel_idx = [feature_names.index(s) for s in selected_fold]
        X_test_sel = X_test[:, sel_idx]
        y_pred[test_idx] = model.predict(scaler.transform(X_test_sel))
        selected_alphas.append(float(model.alpha))
        selected_l1_ratios.append(float(model.l1_ratio))

    perf = {
        "r2": float(r2_score(y, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "mae": float(mean_absolute_error(y, y_pred)),
    }
    cv_info = {
        "selected_alphas": selected_alphas,
        "selected_l1_ratios": selected_l1_ratios,
        "cv_strategy": type(outer_cv).__name__,
        "selection": "within_fold",
    }
    return y_pred, cv_info, perf


def _nested_cv_logistic_l1_with_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    n_outer_splits: int,
    n_inner_splits: int,
    random_state: int,
    split_random_state: int | None = None,
    max_features: int,
    min_features: int,
    c_values: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, float]]:
    """Nested CV (binary) with feature selection inside each outer training fold."""
    if c_values is None:
        c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    split_seed = random_state if split_random_state is None else split_random_state
    class_counts = np.bincount(y)
    n_splits = min(n_outer_splits, max(2, class_counts.min()))
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)

    y_pred = np.zeros_like(y, dtype=int)
    y_prob = np.zeros_like(y, dtype=float)
    selected_cs: list[float] = []

    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test = X[test_idx]

        selected_fold, _, _ = _select_panel(
            X_train, y_train, feature_names,
            n_outer_splits=max(2, min(n_inner_splits, np.bincount(y_train).min())),
            n_inner_splits=max(2, min(n_inner_splits - 1, np.bincount(y_train).min())),
            random_state=split_seed,
            max_features=max_features,
            min_features=min_features,
            task="binary",
        )
        sel_idx = [feature_names.index(s) for s in selected_fold]
        X_train_sel = X_train[:, sel_idx]
        X_test_sel = X_test[:, sel_idx]

        inner_splits = min(n_inner_splits, np.bincount(y_train).min())
        inner_cv = StratifiedKFold(n_splits=max(2, inner_splits), shuffle=True, random_state=split_seed)
        scaler, final_clf, best_c = _fit_tuned_logistic(
            X_train_sel, y_train, inner_cv, c_values, random_state, "minimal panel inner CV"
        )
        X_test_scaled = scaler.transform(X_test_sel)
        y_pred[test_idx] = final_clf.predict(X_test_scaled)
        if len(final_clf.classes_) == 2:
            y_prob[test_idx] = final_clf.predict_proba(X_test_scaled)[:, 1]
        else:
            y_prob[test_idx] = y_pred[test_idx]
        selected_cs.append(best_c)

    perf = {
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
    }
    try:
        perf["roc_auc"] = float(roc_auc_score(y, y_prob))
    except ValueError:
        perf["roc_auc"] = np.nan

    cv_info = {
        "selected_C": selected_cs,
        "cv_strategy": type(outer_cv).__name__,
        "selection": "within_fold",
    }
    return y_pred, y_prob, cv_info, perf


def _permuted_continuous_r2(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_outer_splits: int,
    n_inner_splits: int,
    random_state: int,
    split_random_state: int,
    max_features: int,
    min_features: int,
) -> float:
    rng = np.random.default_rng(random_state)
    y_perm = y[rng.permutation(len(y))]
    # Feature selection must be re-applied under permuted labels so the reference
    # distribution is not conditional on selection performed on observed labels.
    _, _, perf = _nested_cv_elasticnet_with_selection(
        X,
        y_perm,
        feature_names,
        n_outer_splits=n_outer_splits,
        n_inner_splits=n_inner_splits,
        random_state=random_state,
        split_random_state=split_random_state,
        max_features=max_features,
        min_features=min_features,
    )
    return float(perf["r2"])


def _permuted_binary_balanced_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_outer_splits: int,
    n_inner_splits: int,
    random_state: int,
    split_random_state: int,
    c_values: list[float] | None,
    max_features: int,
    min_features: int,
) -> float:
    rng = np.random.default_rng(random_state)
    y_perm = y[rng.permutation(len(y))]
    _, _, _, perf = _nested_cv_logistic_l1_with_selection(
        X,
        y_perm,
        feature_names,
        n_outer_splits=n_outer_splits,
        n_inner_splits=n_inner_splits,
        random_state=random_state,
        split_random_state=split_random_state,
        max_features=max_features,
        min_features=min_features,
        c_values=c_values,
    )
    return float(perf["balanced_accuracy"])


def _bootstrap_stability_sample(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    task: str,
    random_state: int,
    n_outer_splits: int,
    n_inner_splits: int,
    max_features: int,
    min_features: int,
) -> np.ndarray | None:
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, X.shape[0], size=X.shape[0])
    X_boot, y_boot = X[idx], y[idx]
    if len(np.unique(y_boot)) < 2 and task == "binary":
        return None
    selected, _selection, coefficient_folds = _select_panel(
        X_boot,
        y_boot,
        feature_names,
        n_outer_splits=n_outer_splits,
        n_inner_splits=n_inner_splits,
        random_state=random_state,
        max_features=max_features,
        min_features=min_features,
        task=task,
    )
    mean_coefficients = coefficient_folds.mean(axis=0).to_numpy(dtype=float)
    selected_set = set(selected)
    return np.array(
        [coef if name in selected_set else 0.0 for name, coef in zip(feature_names, mean_coefficients)],
        dtype=float,
    )


def _optional_gpu_status() -> str:
    try:
        import cupy as cp  # type: ignore[import-not-found]

        count = int(cp.cuda.runtime.getDeviceCount())
        return f"cupy_available:{count}_cuda_device(s)" if count else "cupy_available:no_cuda_device"
    except Exception:
        return "not_available"


def _permutation_test_continuous(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    n_outer_splits: int,
    n_inner_splits: int,
    n_permutations: int,
    random_state: int,
    observed_r2: float,
    n_jobs: int,
    max_features: int,
    min_features: int,
    panel_cfg: dict | None = None,
) -> tuple[float, float, float, int]:
    """Draws stop early once the Monte Carlo p-value's CI clears alpha.

    Besag & Clifford (1991) curtailment; see
    `src.statistics.sequential_permutation`. Returns the draws actually used
    as the fourth element alongside the usual (p_value, null_mean, null_std).
    """
    stopping = resolve_sequential_stopping_config(panel_cfg or {}, default_max=n_permutations)
    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=stopping.permutation_max, dtype=np.int64)
    seed_args = [
        (X, y, feature_names, n_outer_splits, n_inner_splits, int(seed), random_state, max_features, min_features)
        for seed in seeds
    ]
    null_r2 = run_curtailed_permutation_draws(
        _permuted_continuous_r2, seed_args, observed=observed_r2, config=stopping, n_jobs=n_jobs
    )
    null_arr = np.array(null_r2)
    finite = null_arr[np.isfinite(null_arr)]
    if len(finite) == 0:
        return np.nan, np.nan, np.nan, len(null_r2)
    p_value = float((1 + np.sum(finite >= observed_r2)) / (len(finite) + 1))
    return p_value, float(np.mean(finite)), float(np.std(finite)), len(null_r2)


def _permutation_test_binary(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    n_outer_splits: int,
    n_inner_splits: int,
    n_permutations: int,
    random_state: int,
    observed_balanced_accuracy: float,
    n_jobs: int,
    c_values: list[float] | None = None,
    max_features: int = 10,
    min_features: int = 5,
    panel_cfg: dict | None = None,
) -> tuple[float, float, float, int]:
    """Draws stop early once the Monte Carlo p-value's CI clears alpha.

    Besag & Clifford (1991) curtailment; see
    `src.statistics.sequential_permutation`. Returns the draws actually used
    as the fourth element alongside the usual (p_value, null_mean, null_std).
    """
    stopping = resolve_sequential_stopping_config(panel_cfg or {}, default_max=n_permutations)
    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=stopping.permutation_max, dtype=np.int64)
    seed_args = [
        (
            X, y, feature_names, n_outer_splits, n_inner_splits, int(seed), random_state,
            c_values, max_features, min_features,
        )
        for seed in seeds
    ]
    null_scores = run_curtailed_permutation_draws(
        _permuted_binary_balanced_accuracy,
        seed_args,
        observed=observed_balanced_accuracy,
        config=stopping,
        n_jobs=n_jobs,
    )
    null_arr = np.array(null_scores)
    finite = null_arr[np.isfinite(null_arr)]
    if len(finite) == 0:
        return np.nan, np.nan, np.nan, len(null_scores)
    p_value = float((1 + np.sum(finite >= observed_balanced_accuracy)) / (len(finite) + 1))
    return p_value, float(np.mean(finite)), float(np.std(finite)), len(null_scores)


def _bootstrap_stability_summary(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    n_bootstrap: int,
    random_state: int,
    task: str,
    n_jobs: int,
    n_outer_splits: int,
    n_inner_splits: int,
    max_features: int,
    min_features: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    coef_samples: dict[str, list[float]] = {name: [] for name in feature_names}
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_bootstrap, dtype=np.int64)
    calls = [
        (
            _bootstrap_stability_sample,
            (
                X,
                y,
                feature_names,
                task,
                int(seed),
                n_outer_splits,
                n_inner_splits,
                max_features,
                min_features,
            ),
        )
        for seed in seeds
    ]
    for coefs in _parallel_values(n_jobs, calls):
        if coefs is None:
            continue
        for name, coef in zip(feature_names, coefs):
            coef_samples[name].append(float(coef))

    rows: list[dict[str, object]] = []
    for name in feature_names:
        samples = np.array(coef_samples[name], dtype=float)
        rows.append({
            "feature_name": name,
            "stability_interval_low": float(np.percentile(samples, 2.5)) if len(samples) else np.nan,
            "stability_interval_high": float(np.percentile(samples, 97.5)) if len(samples) else np.nan,
            "mean_coefficient": float(np.mean(samples)) if len(samples) else np.nan,
            "bootstrap_selection_frequency": float(np.mean(np.abs(samples) > 1e-12)) if len(samples) else np.nan,
            "n_bootstrap_used": int(len(samples)),
            "estimation_scope": "bootstrap_selection_stability",
            "validated": False,
            "status": "ok" if len(samples) else "missing_required_data",
        })
    return pd.DataFrame(rows)


def _select_panel(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    n_outer_splits: int,
    n_inner_splits: int,
    random_state: int,
    max_features: int,
    min_features: int,
    task: str,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """Stability-select a sparse panel inside nested CV."""
    n_samples = X.shape[0]
    if task == "continuous":
        outer_cv = _cv_strategy(n_samples, n_outer_splits, random_state)
    else:
        class_counts = np.bincount(y)
        n_splits = min(n_outer_splits, max(2, class_counts.min()))
        outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    coef_matrix = np.zeros((len(feature_names), 0))
    for train_idx, _test_idx in outer_cv.split(X, y) if isinstance(outer_cv, StratifiedKFold) else outer_cv.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        if task == "continuous":
            inner_cv = KFold(n_splits=min(n_inner_splits, len(X_train)), shuffle=True, random_state=random_state)
            _scaler, model = _fit_tuned_elasticnet(
                X_train, y_train, inner_cv, random_state, "panel-selection inner CV"
            )
            coefs = model.coef_.reshape(-1, 1)
        else:
            inner_splits = min(n_inner_splits, np.bincount(y_train).min())
            inner_cv = StratifiedKFold(n_splits=max(2, inner_splits), shuffle=True, random_state=random_state)
            _scaler, model, _best_c = _fit_tuned_logistic(
                X_train,
                y_train,
                inner_cv,
                [0.01, 0.1, 1.0, 10.0, 100.0],
                random_state,
                "panel-selection inner CV",
            )
            coefs = model.coef_.reshape(-1, 1)
        coef_matrix = np.concatenate([coef_matrix, coefs], axis=1)

    selection_freq = np.mean(np.abs(coef_matrix) > 1e-10, axis=1)
    mean_coef = np.mean(np.abs(coef_matrix), axis=1)
    selection_df = pd.DataFrame({
        "feature_name": feature_names,
        "selection_frequency": selection_freq,
        "mean_absolute_coefficient": mean_coef,
    }).sort_values(["selection_frequency", "mean_absolute_coefficient"], ascending=False).reset_index(drop=True)

    # Keep features selected in at least one fold, then rank by frequency + magnitude.
    ever_selected = selection_df.loc[selection_df["selection_frequency"] > 0].copy()
    if len(ever_selected) < min_features:
        ever_selected = selection_df.head(min_features).copy()
    selected = ever_selected.head(max_features)["feature_name"].tolist()

    return selected, selection_df, pd.DataFrame(coef_matrix.T, columns=feature_names)


def _feature_source_and_description(feature_name: str, cfg: dict[str, Any]) -> tuple[str, str]:
    if feature_name in cfg.get("analysis", {}).get("preregistered_panel", {}).get("indices", {}):
        return "preregistered_index", feature_name
    if feature_name in cfg.get("analysis", {}).get("preregistered_panel", {}).get("integrals", {}):
        return "preregistered_integral", feature_name
    if feature_name.startswith("region__"):
        return "candidate_region", feature_name.replace("region__", "")
    if feature_name.startswith("ratio__"):
        parts = feature_name.split("__")
        return "ratio", f"{parts[1]} / {parts[2]}"
    if feature_name.startswith("band__"):
        return "band_integral", feature_name.replace("band__", "")
    return "unknown", feature_name


def run_minimal_panel(
    *,
    feature_matrix_path: str | Path | None = None,
    integrated_candidates_path: str | Path | None = None,
    panel_matrix_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Run the minimal candidate Raman panel analysis."""
    cfg = _cfg(cfg)
    panel_cfg = _panel_cfg(cfg)
    if not bool(panel_cfg.get("enabled", True)):
        return {}
    validate_sequential_stopping_config(panel_cfg, section="analysis.minimal_panel")

    feature_source = str(panel_cfg.get("feature_source", "candidate_regions"))
    is_preregistered_source = feature_source == "preregistered_panel"
    target_dir = Path(output_dir) if output_dir is not None else (
        get_confirmatory_panel_dir() if is_preregistered_source else get_classification_phase_dir("minimal_panel")
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    disease_only, _feature_meta = load_disease_feature_frame(feature_matrix_path=feature_matrix_path)
    all_feature_columns = feature_columns(disease_only)

    if is_preregistered_source:
        pool = _load_preregistered_panel_pool(disease_only, panel_matrix_path)
    else:
        if integrated_candidates_path is None:
            integrated_candidates_path = get_cross_region_phase_dir("integrated_candidates") / "integrated_candidate_regions.csv"
        integrated = _load_integrated_candidates(integrated_candidates_path)
        has_cluster_support = (
            integrated is not None
            and not integrated.empty
            and "evidence_status" in integrated.columns
            and integrated["evidence_status"].eq("cluster_supported_exploratory").any()
        )
        pool = (
            _build_feature_pool(disease_only, all_feature_columns, integrated, cfg)
            if has_cluster_support
            else pd.DataFrame(index=disease_only.index)
        )

    output_stem = "panel_minimal_model" if is_preregistered_source else "minimal_panel"
    selected_filename = "panel_minimal_model_features.csv" if is_preregistered_source else "minimal_candidate_panel.csv"
    performance_filename = f"{output_stem}_performance.csv" if is_preregistered_source else "minimal_panel_cv_performance.csv"
    coefficients_filename = f"{output_stem}_coefficients.csv" if is_preregistered_source else "minimal_panel_coefficients.csv"

    if pool.empty:
        status = STATUS_MISSING_REQUIRED_DATA if is_preregistered_source else STATUS_NOT_TESTED_GATE
        reason = (
            "preregistered panel matrix has no usable features"
            if is_preregistered_source
            else "no cluster-supported candidate region available; candidate-derived panel not tested"
        )
        logger.warning("No eligible panel features available; skipping CSV writes (status=%s)", status)
        return {
            "_status": PhaseStatus(
                phase_key="panel_minimal_model" if is_preregistered_source else "minimal_panel",
                status=status,
                reason=reason,
                n_subjects=None,
                n_artifacts=0,
            ).as_row()
        }

    max_features = int(panel_cfg.get("max_features", 10))
    min_features = int(panel_cfg.get("min_features", 5))
    n_outer_splits = int(panel_cfg.get("n_outer_splits", 5))
    n_inner_splits = int(panel_cfg.get("n_inner_splits", 3))
    n_permutations = int(panel_cfg.get("n_permutations", 1000))
    n_bootstrap = int(panel_cfg.get("n_bootstrap", 200))
    random_state = int(panel_cfg.get("random_state", 42))
    n_jobs = _normalize_n_jobs(int(panel_cfg.get("n_jobs", -1)))
    gpu_status = _optional_gpu_status() if bool(panel_cfg.get("detect_gpu", True)) else "not_checked"
    logger.debug("Minimal panel compute backend: sklearn_cpu; optional GPU status: %s", gpu_status)

    continuous_targets = [str(t) for t in panel_cfg.get("continuous_targets", ["moca_baseline", "moca_3m", "delta_moca"])]
    binary_targets = panel_cfg.get("binary_targets", _DEFAULT_BINARY_TARGETS)

    panel_rows: list[dict[str, object]] = []
    perf_rows: list[dict[str, object]] = []
    coef_rows: list[dict[str, object]] = []

    feature_names = pool.columns.tolist()
    X_full = pool.to_numpy(dtype=float)

    for target in continuous_targets:
        if target not in disease_only.columns:
            logger.info("Continuous target '%s' not found; skipping", target)
            continue
        y = pd.to_numeric(disease_only[target], errors="coerce")
        valid = y.notna() & np.isfinite(X_full).all(axis=1)
        if valid.sum() < 10:
            logger.info("Target '%s' has only %d valid subjects; skipping", target, valid.sum())
            continue
        X = X_full[valid]
        y_arr = y.loc[valid].to_numpy(dtype=float)

        # _select_panel on full data is kept ONLY for reporting selection frequency.
        # Performance metrics and p-values use _with_selection variants that fold
        # feature selection inside each outer training fold (no leakage).
        selected, selection_df, _ = _select_panel(
            X, y_arr, feature_names,
            n_outer_splits=n_outer_splits,
            n_inner_splits=n_inner_splits,
            random_state=random_state,
            max_features=max_features,
            min_features=min_features,
            task="continuous",
        )

        _, cv_info, perf = _nested_cv_elasticnet_with_selection(
            X,
            y_arr,
            feature_names,
            n_outer_splits=n_outer_splits,
            n_inner_splits=n_inner_splits,
            random_state=random_state,
            max_features=max_features,
            min_features=min_features,
        )
        p_value, _, _, n_permutations_used = _permutation_test_continuous(
            X,
            y_arr,
            feature_names,
            n_outer_splits=n_outer_splits,
            n_inner_splits=n_inner_splits,
            n_permutations=n_permutations,
            random_state=random_state,
            observed_r2=perf["r2"],
            n_jobs=n_jobs,
            max_features=max_features,
            min_features=min_features,
            panel_cfg=panel_cfg,
        )
        stability_df = _bootstrap_stability_summary(
            X,
            y_arr,
            feature_names,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            task="continuous",
            n_jobs=n_jobs,
            n_outer_splits=n_outer_splits,
            n_inner_splits=n_inner_splits,
            max_features=max_features,
            min_features=min_features,
        )

        for _, row in selection_df.iterrows():
            source, desc = _feature_source_and_description(str(row["feature_name"]), cfg)
            panel_rows.append({
                "endpoint": target,
                "endpoint_type": "continuous",
                "feature_name": row["feature_name"],
                "feature_source": source,
                "feature_description": desc,
                "selection_frequency": float(row["selection_frequency"]),
                "mean_absolute_coefficient": float(row["mean_absolute_coefficient"]),
                "panel_selected": bool(row["feature_name"] in selected),
                "estimation_scope": "full_data_descriptive",
                "validated": False,
                "status": "ok",
            })
        perf_rows.append({
            "endpoint": target,
            "endpoint_type": "continuous",
            "metric": "r2",
            "value": perf["r2"],
            "n_subjects": int(valid.sum()),
            "permutation_p_value": p_value,
            "n_permutations": n_permutations_used,
            "cv_strategy": cv_info["cv_strategy"],
            "compute_backend": "sklearn_cpu",
            "gpu_status": gpu_status,
            "inference_role": "exploratory",
            "status": "ok",
        })
        perf_rows.append({
            "endpoint": target,
            "endpoint_type": "continuous",
            "metric": "rmse",
            "value": perf["rmse"],
            "n_subjects": int(valid.sum()),
            "permutation_p_value": pd.NA,
            "n_permutations": pd.NA,
            "cv_strategy": cv_info["cv_strategy"],
            "compute_backend": "sklearn_cpu",
            "gpu_status": gpu_status,
            "inference_role": "exploratory",
            "status": "ok",
        })
        for _, row in stability_df.iterrows():
            coef_rows.append({
                "endpoint": target,
                "feature_name": row["feature_name"],
                "mean_coefficient": row["mean_coefficient"],
                "stability_interval_low": row["stability_interval_low"],
                "stability_interval_high": row["stability_interval_high"],
                "bootstrap_selection_frequency": row["bootstrap_selection_frequency"],
                "n_bootstrap_used": row["n_bootstrap_used"],
                "estimation_scope": row["estimation_scope"],
                "validated": row["validated"],
                "status": row["status"],
            })

    for target_def in binary_targets:
        group_var = str(target_def["group_variable"])
        ordered_groups = tuple(target_def["ordered_groups"])
        if group_var not in disease_only.columns:
            logger.info("Binary target '%s' not found; skipping", group_var)
            continue
        subset = disease_only[disease_only[group_var].isin(ordered_groups)].copy()
        subset[group_var] = subset[group_var].astype(str)
        y = (subset[group_var] == ordered_groups[1]).astype(int)
        pool_subset = pool.loc[subset.index]
        X = pool_subset.to_numpy(dtype=float)
        valid = y.notna() & np.isfinite(X).all(axis=1)
        if valid.sum() < 10 or y.loc[valid].nunique() < 2:
            logger.info("Binary target '%s' has insufficient valid subjects; skipping", group_var)
            continue
        X = X[valid]
        y_arr = y.loc[valid].to_numpy(dtype=int)

        pool_feature_names = pool_subset.columns.tolist()
        # _select_panel on full data is kept ONLY for reporting selection frequency.
        selected, selection_df, _ = _select_panel(
            X, y_arr, pool_feature_names,
            n_outer_splits=n_outer_splits,
            n_inner_splits=n_inner_splits,
            random_state=random_state,
            max_features=max_features,
            min_features=min_features,
            task="binary",
        )

        _, _, cv_info, perf = _nested_cv_logistic_l1_with_selection(
            X,
            y_arr,
            pool_feature_names,
            n_outer_splits=n_outer_splits,
            n_inner_splits=n_inner_splits,
            random_state=random_state,
            max_features=max_features,
            min_features=min_features,
        )
        p_value, _, _, n_permutations_used = _permutation_test_binary(
            X,
            y_arr,
            pool_feature_names,
            n_outer_splits=n_outer_splits,
            n_inner_splits=n_inner_splits,
            n_permutations=n_permutations,
            random_state=random_state,
            observed_balanced_accuracy=perf["balanced_accuracy"],
            n_jobs=n_jobs,
            max_features=max_features,
            min_features=min_features,
            panel_cfg=panel_cfg,
        )
        stability_df = _bootstrap_stability_summary(
            X,
            y_arr,
            pool_feature_names,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            task="binary",
            n_jobs=n_jobs,
            n_outer_splits=n_outer_splits,
            n_inner_splits=n_inner_splits,
            max_features=max_features,
            min_features=min_features,
        )

        for _, row in selection_df.iterrows():
            source, desc = _feature_source_and_description(str(row["feature_name"]), cfg)
            panel_rows.append({
                "endpoint": group_var,
                "endpoint_type": "binary",
                "feature_name": row["feature_name"],
                "feature_source": source,
                "feature_description": desc,
                "selection_frequency": float(row["selection_frequency"]),
                "mean_absolute_coefficient": float(row["mean_absolute_coefficient"]),
                "panel_selected": bool(row["feature_name"] in selected),
                "estimation_scope": "full_data_descriptive",
                "validated": False,
                "status": "ok",
            })
        perf_rows.append({
            "endpoint": group_var,
            "endpoint_type": "binary",
            "metric": "balanced_accuracy",
            "value": perf["balanced_accuracy"],
            "n_subjects": int(valid.sum()),
            "permutation_p_value": p_value,
            "n_permutations": n_permutations_used,
            "cv_strategy": cv_info["cv_strategy"],
            "compute_backend": "sklearn_cpu",
            "gpu_status": gpu_status,
            "inference_role": "exploratory",
            "status": "ok",
        })
        perf_rows.append({
            "endpoint": group_var,
            "endpoint_type": "binary",
            "metric": "roc_auc",
            "value": perf.get("roc_auc", np.nan),
            "n_subjects": int(valid.sum()),
            "permutation_p_value": pd.NA,
            "n_permutations": pd.NA,
            "cv_strategy": cv_info["cv_strategy"],
            "compute_backend": "sklearn_cpu",
            "gpu_status": gpu_status,
            "inference_role": "exploratory",
            "status": "ok",
        })
        for _, row in stability_df.iterrows():
            coef_rows.append({
                "endpoint": group_var,
                "feature_name": row["feature_name"],
                "mean_coefficient": row["mean_coefficient"],
                "stability_interval_low": row["stability_interval_low"],
                "stability_interval_high": row["stability_interval_high"],
                "bootstrap_selection_frequency": row["bootstrap_selection_frequency"],
                "n_bootstrap_used": row["n_bootstrap_used"],
                "estimation_scope": row["estimation_scope"],
                "validated": row["validated"],
                "status": row["status"],
            })

    panel_df = pd.DataFrame(
        panel_rows,
        columns=[
            "endpoint", "endpoint_type", "feature_name", "feature_source", "feature_description",
            "selection_frequency", "mean_absolute_coefficient", "panel_selected",
            "estimation_scope", "validated", "status",
        ],
    )
    perf_df = pd.DataFrame(
        perf_rows,
        columns=[
            "endpoint", "endpoint_type", "metric", "value", "n_subjects", "permutation_p_value",
            "n_permutations", "cv_strategy", "compute_backend", "gpu_status", "inference_role", "status",
        ],
    )
    coef_df = pd.DataFrame(
        coef_rows,
        columns=[
            "endpoint", "feature_name", "mean_coefficient", "stability_interval_low",
            "stability_interval_high", "bootstrap_selection_frequency", "n_bootstrap_used",
            "estimation_scope", "validated", "status",
        ],
    )

    panel_path = target_dir / selected_filename
    perf_path = target_dir / performance_filename
    coef_path = target_dir / coefficients_filename

    write_csv_atomic(panel_df, panel_path, index=False)
    write_csv_atomic(perf_df, perf_path, index=False)
    write_csv_atomic(coef_df, coef_path, index=False)

    logger.info("Minimal candidate panel outputs written to %s", target_dir)
    return {
        ("panel_minimal_model_features" if is_preregistered_source else "minimal_candidate_panel"): panel_path,
        ("panel_minimal_model_performance" if is_preregistered_source else "minimal_panel_cv_performance"): perf_path,
        ("panel_minimal_model_coefficients" if is_preregistered_source else "minimal_panel_coefficients"): coef_path,
    }
