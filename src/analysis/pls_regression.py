"""Phase 8: PLS1 Regression for continuous clinical endpoints."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut, LeaveOneOut
from sklearn.preprocessing import StandardScaler

from src import get_classification_phase_dir
from src import resolve_cfg as _cfg
from src.analysis.clinical_feature_associations import (
    feature_columns,
    load_disease_feature_frame,
)
from src.analysis.io_utils import write_csv_atomic, write_parquet_atomic
from src.analysis.phase_status import (
    EVIDENCE_EXPLORATORY_SIGNAL,
    EVIDENCE_NO_SUPPORTED_SIGNAL,
    STATUS_INSUFFICIENT_N,
    STATUS_OK,
    PhaseStatus,
)
from src.statistics.sequential_permutation import (
    resolve_sequential_stopping_config,
    should_stop,
    validate_sequential_stopping_config,
)

logger = logging.getLogger(__name__)


def _pls_cfg(cfg: dict) -> dict:
    return cfg.get("analysis", {}).get("pls_regression", {})


def _holm_adjust(p_values: pd.Series) -> pd.Series:
    """Return Holm-adjusted p-values while preserving missing values."""
    numeric = pd.to_numeric(p_values, errors="coerce")
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = numeric.dropna()
    if valid.empty:
        return adjusted
    order = np.argsort(valid.to_numpy(dtype=float))
    ordered = valid.to_numpy(dtype=float)[order]
    values = np.maximum.accumulate(ordered * (len(valid) - np.arange(len(valid))))
    restored = np.empty(len(valid), dtype=float)
    restored[order] = np.minimum(values, 1.0)
    adjusted.loc[valid.index] = restored
    return adjusted


def _calculate_vip(model: PLSRegression) -> np.ndarray:
    """Calculate Variable Importance in Projection (VIP) scores for PLS model."""
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, n_comp = w.shape

    s = np.sum(t**2, axis=0) * np.sum(q**2, axis=0)
    total_s = np.sum(s)

    if total_s == 0:
        return np.zeros(p)

    w_norm = w / np.linalg.norm(w, axis=0)
    vip = np.sqrt(p * np.sum(s * (w_norm**2), axis=1) / total_s)
    return vip


def _get_cv_strategy(
    frame: pd.DataFrame,
    n_splits: int,
    random_state: int,
) -> tuple[object, str, bool]:
    """Determine the optimal CV strategy based on sample size and repeated rows."""
    n_subjects = frame["subject_id"].nunique()
    n_rows = len(frame)

    if n_rows > n_subjects:
        if n_subjects < 15:
            return LeaveOneGroupOut(), "LeaveOneGroupOut", True
        return GroupKFold(n_splits=min(n_splits, n_subjects)), "GroupKFold", False

    if n_subjects < 15:
        return LeaveOneOut(), "LeaveOneOut", True
    return KFold(n_splits=min(n_splits, n_subjects), shuffle=True, random_state=random_state), "KFold", False


def _tune_pls_components(
    X_train: np.ndarray,
    y_train: np.ndarray,
    component_grid: list[int],
    random_state: int,
) -> int:
    """Tune n_components using an inner cross-validation."""
    max_comps = min(max(component_grid), X_train.shape[1], X_train.shape[0] - 2)
    valid_grid = [c for c in component_grid if c <= max_comps and c > 0]
    if not valid_grid:
        return 1

    if len(valid_grid) == 1:
        return valid_grid[0]

    inner_cv = KFold(n_splits=min(3, X_train.shape[0]), shuffle=True, random_state=random_state)
    best_mse = float("inf")
    best_c = valid_grid[0]

    for c in valid_grid:
        mse_sum = 0.0
        for in_tr, in_te in inner_cv.split(X_train):
            scaler = StandardScaler()
            X_in_tr = scaler.fit_transform(X_train[in_tr])
            X_in_te = scaler.transform(X_train[in_te])
            y_in_tr = y_train[in_tr]
            y_in_te = y_train[in_te]

            inner_model = PLSRegression(n_components=c, scale=False)
            inner_model.fit(X_in_tr, y_in_tr)
            preds = inner_model.predict(X_in_te).ravel()
            mse_sum += np.mean((y_in_te - preds) ** 2)

        if mse_sum < best_mse:
            best_mse = mse_sum
            best_c = c

    return best_c


def _permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    cv_strategy: object,
    groups: np.ndarray | None,
    component_grid: list[int],
    tune_inside_cv: bool,
    observed_r2: float,
    n_permutations: int,
    random_state: int,
    n_jobs: int = 1,
    pls_cfg: dict | None = None,
) -> tuple[float, float, float, int]:
    """Evaluate the entire nested-CV PLS procedure against permuted outcomes.

    Draws stop early once the Monte Carlo p-value's Clopper-Pearson interval
    clears `permutation_alpha` at `permutation_confidence` (Besag & Clifford
    1991 curtailment; see `src.statistics.sequential_permutation`).
    `n_permutations` is the ceiling on draws, not necessarily the count
    actually run — returns the draws actually used as the fourth element.
    """
    n_jobs = 1 if n_jobs == 0 else n_jobs
    stopping = resolve_sequential_stopping_config(pls_cfg or {}, default_max=n_permutations)
    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=stopping.permutation_max, dtype=np.int64)

    null_r2: list[float] = []
    successes = 0
    offset = 0
    while offset < len(seeds) and not should_stop(successes=successes, draws=len(null_r2), config=stopping):
        chunk = seeds[offset : offset + stopping.check_interval]
        offset += stopping.check_interval
        if n_jobs != 1 and len(chunk) > 1:
            chunk_r2 = Parallel(n_jobs=n_jobs)(
                delayed(_permuted_pls_r2)(
                    X, y, cv_strategy, groups, component_grid, tune_inside_cv, random_state, int(seed)
                )
                for seed in chunk
            )
        else:
            chunk_r2 = [
                _permuted_pls_r2(
                    X, y, cv_strategy, groups, component_grid, tune_inside_cv, random_state, int(seed)
                )
                for seed in chunk
            ]
        null_r2.extend(chunk_r2)
        successes += sum(1 for value in chunk_r2 if np.isfinite(value) and value >= observed_r2)

    null_r2_arr = np.array(null_r2)
    finite_null = null_r2_arr[np.isfinite(null_r2_arr)]
    if len(finite_null) == 0:
        return np.nan, np.nan, np.nan, len(null_r2)

    p_value = float((1 + np.sum(finite_null >= observed_r2)) / (len(finite_null) + 1))
    return p_value, float(np.mean(finite_null)), float(np.std(finite_null)), len(null_r2)


def _permuted_pls_r2(
    X: np.ndarray,
    y: np.ndarray,
    cv_strategy: object,
    groups: np.ndarray | None,
    component_grid: list[int],
    tune_inside_cv: bool,
    random_state: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    return _permuted_pls_r2_with_rng(
        X, y, cv_strategy, groups, component_grid, tune_inside_cv, random_state, rng
    )


def _permuted_pls_r2_with_rng(
    X: np.ndarray,
    y: np.ndarray,
    cv_strategy: object,
    groups: np.ndarray | None,
    component_grid: list[int],
    tune_inside_cv: bool,
    random_state: int,
    rng: np.random.Generator,
) -> float:
    y_perm = rng.permutation(y)
    y_pred = np.zeros_like(y, dtype=float)

    cv_kwargs = {"groups": groups} if groups is not None and not isinstance(cv_strategy, (KFold, LeaveOneOut)) else {}
    for train_idx, test_idx in cv_strategy.split(X, y_perm, **cv_kwargs):
        X_train, y_train = X[train_idx], y_perm[train_idx]
        X_test = X[test_idx]

        if tune_inside_cv:
            best_c = _tune_pls_components(X_train, y_train, component_grid, random_state)
        else:
            best_c = min(component_grid[-1], X_train.shape[1], X_train.shape[0] - 2)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = PLSRegression(n_components=min(best_c, X_train.shape[0] - 2), scale=False)
        model.fit(X_train_scaled, y_train)
        y_pred[test_idx] = model.predict(X_test_scaled).ravel()

    ss_res = np.sum((y_perm - y_pred) ** 2)
    ss_tot = np.sum((y_perm - np.mean(y_perm)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def run_pls_regression(
    *,
    feature_matrix_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    cfg: dict | None = None,
) -> dict[str, Path]:
    cfg = _cfg(cfg)
    pls_cfg = _pls_cfg(cfg)
    if not bool(pls_cfg.get("enabled", True)):
        return {}
    validate_sequential_stopping_config(pls_cfg, section="analysis.pls_regression")

    disease_only, feature_meta = load_disease_feature_frame(feature_matrix_path=feature_matrix_path)
    all_feature_columns = feature_columns(disease_only)

    target_dir = Path(output_dir) if output_dir is not None else get_classification_phase_dir("pls_regression")
    target_dir.mkdir(parents=True, exist_ok=True)

    targets = pls_cfg.get("targets", ["moca_baseline", "moca_3m", "delta_moca"])
    component_grid = pls_cfg.get("component_grid", [1, 2, 3, 4, 5])
    tune_inside_cv = bool(pls_cfg.get("tune_components_inside_cv", True))
    n_cv_splits = int(pls_cfg.get("n_cv_splits", 5))
    min_subjects = int(pls_cfg.get("min_subjects", 10))
    vip_threshold = float(pls_cfg.get("vip_threshold", 1.0))
    n_permutations = int(pls_cfg.get("n_permutations", 1000))
    n_jobs = int(pls_cfg.get("n_jobs", 1))
    random_state = int(pls_cfg.get("random_state", 42))

    perf_rows: list[dict[str, object]] = []
    vip_rows: list[dict[str, object]] = []
    coef_rows: list[dict[str, object]] = []
    perm_rows: list[dict[str, object]] = []

    for target in targets:
        if target not in disease_only.columns:
            logger.info("Target '%s' not found, skipping PLS regression", target)
            continue

        target_values = pd.to_numeric(disease_only[target], errors="coerce")
        valid_mask = target_values.notna()
        if valid_mask.sum() < min_subjects:
            logger.info("Target '%s' has only %d valid subjects, skipping", target, valid_mask.sum())
            continue

        frame = disease_only.loc[valid_mask].copy()
        X = frame[all_feature_columns].to_numpy(dtype=float)
        y = pd.to_numeric(frame[target]).to_numpy(dtype=float)
        groups = frame["subject_id"].to_numpy()

        cv_strategy, cv_name, high_variance = _get_cv_strategy(frame, n_cv_splits, random_state)

        y_pred = np.zeros_like(y, dtype=float)
        fold_selected_components = []
        fold_vip_selected: list[np.ndarray] = []
        fold_coefficient_signs: list[np.ndarray] = []

        cv_kwargs = {"groups": groups} if not isinstance(cv_strategy, (KFold, LeaveOneOut)) else {}
        for train_idx, test_idx in cv_strategy.split(X, y, **cv_kwargs):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test = X[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            if tune_inside_cv:
                best_c = _tune_pls_components(X_train, y_train, component_grid, random_state)
            else:
                best_c = min(component_grid[-1], X_train_scaled.shape[1], X_train_scaled.shape[0] - 2)

            fold_selected_components.append(best_c)

            model = PLSRegression(n_components=best_c, scale=False)
            model.fit(X_train_scaled, y_train)
            y_pred[test_idx] = model.predict(X_test_scaled).ravel()
            fold_vip_selected.append(_calculate_vip(model) >= vip_threshold)
            fold_coefficient_signs.append(np.sign(model.coef_.ravel() / scaler.scale_))

        # Final model tuning and fitting
        final_scaler = StandardScaler()
        X_scaled = final_scaler.fit_transform(X)
        if tune_inside_cv:
            final_c = _tune_pls_components(X, y, component_grid, random_state)
        else:
            final_c = min(component_grid[-1], X_scaled.shape[1], X_scaled.shape[0] - 2)

        final_model = PLSRegression(n_components=final_c, scale=False)
        final_model.fit(X_scaled, y)

        vip_scores = _calculate_vip(final_model)

        # Calculate raw coefficients (mapping scaled variables back to original space)
        # y is not scaled by StandardScaler in our code, PLSRegression might scale it internally,
        # but coef_ relates X_scaled to y.
        raw_coefs = final_model.coef_.ravel() / final_scaler.scale_
        selection_frequency = np.mean(np.vstack(fold_vip_selected), axis=0)
        sign_stability = np.abs(np.mean(np.vstack(fold_coefficient_signs), axis=0))

        for i, col in enumerate(all_feature_columns):
            meta = feature_meta[feature_meta["feature_name"] == col].iloc[0]
            vip_rows.append({
                "target": target,
                "feature_name": col,
                "spectral_block": meta["spectral_block"],
                "wavenumber": meta["wavenumber"],
                "vip_score": float(vip_scores[i]),
                "above_threshold": bool(vip_scores[i] >= vip_threshold),
                "estimation_scope": "full_data_descriptive",
                "validated": False,
                "selection_frequency": float(selection_frequency[i]),
                "sign_stability": float(sign_stability[i]),
            })
            coef_rows.append({
                "target": target,
                "feature_name": col,
                "spectral_block": meta["spectral_block"],
                "wavenumber": meta["wavenumber"],
                "coefficient": float(raw_coefs[i]),
                "estimation_scope": "full_data_descriptive",
                "validated": False,
                "selection_frequency": float(selection_frequency[i]),
                "sign_stability": float(sign_stability[i]),
            })

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        observed_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        p_value, null_mean, null_std, n_permutations_used = _permutation_test(
            X,
            y,
            cv_strategy,
            groups,
            component_grid,
            tune_inside_cv,
            observed_r2,
            n_permutations,
            random_state,
            n_jobs=n_jobs,
            pls_cfg=pls_cfg,
        )

        perf_rows.append({
            "target": target,
            "r2": observed_r2,
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "mae": float(mean_absolute_error(y, y_pred)),
            "n_subjects": frame["subject_id"].nunique(),
            "selected_n_components": final_c,
            "cv_strategy": cv_name,
            "permutation_p_value": p_value,
            "high_variance_flag": high_variance,
        })

        perm_rows.append({
            "target": target,
            "observed_r2": observed_r2,
            "null_distribution_mean": null_mean,
            "null_distribution_std": null_std,
            "permutation_p_value": p_value,
            "n_permutations": n_permutations_used,
        })

    perf_df = pd.DataFrame(
        perf_rows,
        columns=[
            "target", "r2", "rmse", "mae", "n_subjects", "selected_n_components",
            "cv_strategy", "permutation_p_value", "high_variance_flag",
        ],
    )
    vip_df = pd.DataFrame(
        vip_rows,
        columns=[
            "target", "feature_name", "spectral_block", "wavenumber", "vip_score", "above_threshold",
            "estimation_scope", "validated", "selection_frequency", "sign_stability",
        ],
    )
    coef_df = pd.DataFrame(
        coef_rows,
        columns=[
            "target", "feature_name", "spectral_block", "wavenumber", "coefficient",
            "estimation_scope", "validated", "selection_frequency", "sign_stability",
        ],
    )
    perm_df = pd.DataFrame(
        perm_rows,
        columns=["target", "observed_r2", "null_distribution_mean", "null_distribution_std", "permutation_p_value", "n_permutations"],
    )

    if perf_df.empty and vip_df.empty and coef_df.empty and perm_df.empty:
        logger.info("No PLS regression target reached min_subjects=%d; skipping CSV writes", min_subjects)
        return {
            "_status": PhaseStatus(
                phase_key="pls_regression",
                status=STATUS_INSUFFICIENT_N,
                reason=f"no target reached min_subjects={min_subjects}",
                n_subjects=int(disease_only["subject_id"].nunique()) if "subject_id" in disease_only.columns else None,
                n_artifacts=0,
            ).as_row()
        }

    gate_alpha = float(pls_cfg.get("permutation_family_alpha", 0.05))
    perf_df["permutation_p_value_holm"] = _holm_adjust(perf_df["permutation_p_value"])
    perf_df["model_supported"] = perf_df["permutation_p_value_holm"].lt(gate_alpha).fillna(False)
    perm_df = perm_df.merge(
        perf_df[["target", "permutation_p_value_holm", "model_supported"]],
        on="target",
        how="left",
        validate="one_to_one",
    )
    feature_df = vip_df.merge(
        coef_df[["target", "feature_name", "coefficient"]],
        on=["target", "feature_name"],
        how="inner",
        validate="one_to_one",
    ).merge(
        perf_df[["target", "permutation_p_value_holm", "model_supported"]],
        on="target",
        how="left",
        validate="many_to_one",
    )
    feature_df["inference_role"] = "descriptive_model_audit"

    perf_path = target_dir / "pls_regression_performance.csv"
    effects_path = target_dir / "pls_feature_effects.parquet"
    perm_path = target_dir / "pls_permutation_results.csv"

    write_csv_atomic(perf_df, perf_path, index=False)
    write_parquet_atomic(feature_df, effects_path, index=False)
    write_csv_atomic(perm_df, perm_path, index=False)
    for obsolete_path in (target_dir / "pls_vip_bands.csv", target_dir / "pls_coefficients.csv"):
        obsolete_path.unlink(missing_ok=True)

    logger.info("PLS1 regression outputs written to %s", target_dir)
    evidence_status = (
        EVIDENCE_EXPLORATORY_SIGNAL
        if bool(perf_df["model_supported"].any())
        else EVIDENCE_NO_SUPPORTED_SIGNAL
    )
    return {
        "pls_regression_performance": perf_path,
        "pls_feature_effects": effects_path,
        "pls_permutation_results": perm_path,
        "_status": PhaseStatus(
            phase_key="pls_regression",
            status=STATUS_OK,
            evidence_status=evidence_status,
            reason=(
                "at least one endpoint survived the Holm-adjusted permutation gate"
                if evidence_status == EVIDENCE_EXPLORATORY_SIGNAL
                else "no endpoint survived the Holm-adjusted permutation gate"
            ),
            n_subjects=int(disease_only["subject_id"].nunique()),
            n_artifacts=3,
        ).as_row(),
    }
