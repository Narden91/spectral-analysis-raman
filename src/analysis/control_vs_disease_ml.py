"""Control-vs-disease ML classification layer (disease_vs_control mode only).

Subject-level, nested-CV supervised classification of Control vs. Disease from
whole Raman spectra. This module is a separate, read-only downstream consumer
of the Biological PCA subject feature matrix — see plan_ml.md for the full
scientific-rigor rationale (confound handling, claims discipline, data facts).

Positive class = Disease (is_control == False -> 1). Exploratory only: no
diagnostic, biomarker, causal, or clinical-utility claims. Age is not modeled
anywhere in this pipeline because no age data exists for control subjects in
this cohort (49/50 missing) and no proxy exists in the repository; inventing
one would inject bias rather than remove it.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src import config, get_ml_classification_dir, get_run_metadata_dir
from src.analysis.clinical_feature_associations import (
    feature_columns,
    load_subject_feature_matrix,
)
from src.analysis.io_utils import write_csv_atomic
from src.analysis.pls_regression import _calculate_vip
from src.analysis.sklearn_utils import fit_with_convergence_capture, l1_logistic_model
from src.data_preparation import resolve_data_source, write_analysis_provenance

logger = logging.getLogger(__name__)

POSITIVE_CLASS_LABEL = "Disease"
NEGATIVE_CLASS_LABEL = "Control"

_MIN_SUBJECTS_PER_CLASS = 10


# ---------------------------------------------------------------------------
# Config access
# ---------------------------------------------------------------------------


def _cfg(cfg_in: dict[str, Any] | None = None) -> dict[str, Any]:
    return config if cfg_in is None else cfg_in


def _ml_cfg(cfg_in: dict[str, Any]) -> dict[str, Any]:
    return cfg_in.get("analysis", {}).get("control_vs_disease_ml", {})


# ---------------------------------------------------------------------------
# Phase timing (local — mirrors disease_only_pipeline._phase_timer without
# importing that module's private symbols across a module boundary)
# ---------------------------------------------------------------------------


@contextmanager
def _phase_timer(timings: list[dict[str, object]], phase_key: str, phase_label: str):
    start = time.perf_counter()
    status = "ok"
    try:
        yield
    except Exception:
        status = "failed"
        raise
    finally:
        timings.append(
            {
                "phase": phase_key,
                "label": phase_label,
                "elapsed_seconds": time.perf_counter() - start,
                "status": status,
            }
        )


def _write_phase_timings(timings: list[dict[str, object]], metadata_dir: Path) -> Path | None:
    if not timings:
        return None
    path = metadata_dir / "control_vs_disease_ml_phase_timing.csv"
    write_csv_atomic(pd.DataFrame(timings), path, index=False)
    return path


# ---------------------------------------------------------------------------
# Load + guard
# ---------------------------------------------------------------------------


def _load_cohort(feature_matrix_path: str | Path | None) -> pd.DataFrame:
    frame = load_subject_feature_matrix(feature_matrix_path=feature_matrix_path)
    if "is_control" not in frame.columns:
        raise ValueError("Subject feature matrix is missing is_control; cannot build Control vs Disease labels")
    if frame["is_control"].isna().any():
        n_missing = int(frame["is_control"].isna().sum())
        raise ValueError(f"Subject feature matrix has {n_missing} subject(s) with missing is_control")

    n_control = int(frame["is_control"].astype(bool).sum())
    n_disease = int((~frame["is_control"].astype(bool)).sum())
    if n_control < _MIN_SUBJECTS_PER_CLASS or n_disease < _MIN_SUBJECTS_PER_CLASS:
        raise ValueError(
            "Control vs Disease ML requires at least "
            f"{_MIN_SUBJECTS_PER_CLASS} subjects per class; got control={n_control}, disease={n_disease}"
        )

    if "diagnosis_family" in frame.columns:
        mislabeled = frame.loc[
            frame["is_control"].astype(bool) & frame["diagnosis_family"].notna()
        ]
        if not mislabeled.empty:
            logger.warning(
                "Label anomaly: %d subject(s) marked is_control=True but carry a non-null "
                "diagnosis_family (%s). Verify labels before treating results as final. Subject_id(s): %s",
                len(mislabeled),
                sorted(mislabeled["diagnosis_family"].dropna().unique().tolist()),
                mislabeled["subject_id"].astype(str).tolist(),
            )

    return frame


def _labels(frame: pd.DataFrame) -> np.ndarray:
    return (~frame["is_control"].astype(bool)).astype(int).to_numpy()


# ---------------------------------------------------------------------------
# Feature representations
# ---------------------------------------------------------------------------


def _whole_spectrum_features(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    columns = feature_columns(frame)
    if not columns:
        raise ValueError("Subject feature matrix has no whole-spectrum feature columns")
    return frame[columns].to_numpy(dtype=float), columns


def _band_feature(frame: pd.DataFrame, all_columns: list[str], block: str, lower: float, upper: float) -> pd.Series:
    cols: list[str] = []
    for column in all_columns:
        if not column.startswith(f"{block}__"):
            continue
        wavenumber = float(column.split("__", 1)[1])
        if lower <= wavenumber <= upper:
            cols.append(column)
    if not cols:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return frame[cols].mean(axis=1)


def _band_panel_features(
    frame: pd.DataFrame, all_columns: list[str], cfg_in: dict[str, Any]
) -> tuple[np.ndarray, list[str]]:
    bands_cfg = cfg_in.get("whole_spectra", {}).get("bands", {})
    if not bands_cfg:
        raise ValueError("config whole_spectra.bands is empty; band_panel representation requires the 21-band panel")
    band_series: dict[str, pd.Series] = {}
    for band_name, limits in bands_cfg.items():
        lower, upper = float(limits[0]), float(limits[1])
        block = "1300" if upper < 2000 else "2850"
        band_series[str(band_name)] = _band_feature(frame, all_columns, block, lower, upper)
    band_frame = pd.DataFrame(band_series, index=frame.index)
    return band_frame.to_numpy(dtype=float), list(band_frame.columns)


def _representation_matrix(
    frame: pd.DataFrame, representation: str, cfg_in: dict[str, Any]
) -> tuple[np.ndarray, list[str]]:
    if representation == "whole":
        return _whole_spectrum_features(frame)
    if representation == "band_panel":
        all_columns = feature_columns(frame)
        return _band_panel_features(frame, all_columns, cfg_in)
    raise ValueError(f"Unknown feature representation '{representation}'")


# ---------------------------------------------------------------------------
# Phase A: cohort description
# ---------------------------------------------------------------------------


def _cohort_summary(frame: pd.DataFrame) -> pd.DataFrame:
    is_control = frame["is_control"].astype(bool)
    n_control = int(is_control.sum())
    n_disease = int((~is_control).sum())
    n_total = n_control + n_disease

    rows: list[dict[str, object]] = [
        {"metric": "n_subjects_total", "value": n_total},
        {"metric": "n_control", "value": n_control},
        {"metric": "n_disease", "value": n_disease},
        {"metric": "imbalance_ratio_disease_to_control", "value": n_disease / n_control if n_control else np.nan},
        {"metric": "majority_class_baseline_accuracy", "value": max(n_control, n_disease) / n_total if n_total else np.nan},
        {"metric": "majority_class_baseline_balanced_accuracy", "value": 0.5},
        {"metric": "positive_class", "value": POSITIVE_CLASS_LABEL},
        {"metric": "negative_class", "value": NEGATIVE_CLASS_LABEL},
    ]

    sex_column = next((c for c in ("gender", "sex") if c in frame.columns), None)
    if sex_column is not None:
        crosstab = pd.crosstab(is_control, frame[sex_column].astype(str))
        for control_flag, row in crosstab.iterrows():
            group_label = NEGATIVE_CLASS_LABEL if control_flag else POSITIVE_CLASS_LABEL
            for sex_value, count in row.items():
                rows.append({"metric": f"n_{group_label.lower()}_{sex_value}", "value": int(count)})
        if crosstab.shape[0] == 2 and crosstab.shape[1] == 2:
            chi2, chi2_p, _, _ = stats.chi2_contingency(crosstab.to_numpy())
            rows.append({"metric": "sex_vs_class_chi2", "value": float(chi2)})
            rows.append({"metric": "sex_vs_class_chi2_p_value", "value": float(chi2_p)})
    else:
        logger.warning("No sex/gender column found in feature matrix; sex-confound analysis will be unavailable")

    if "age_years" in frame.columns:
        age_non_null = frame.groupby(is_control)["age_years"].apply(lambda s: int(s.notna().sum()))
        for control_flag, count in age_non_null.items():
            group_label = NEGATIVE_CLASS_LABEL if control_flag else POSITIVE_CLASS_LABEL
            rows.append({"metric": f"n_{group_label.lower()}_with_age", "value": int(count)})
    rows.append(
        {
            "metric": "age_adjustment_status",
            "value": "not_attempted_no_control_age_data",
        }
    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# In-fold-safe model fit/evaluate helpers (mirror minimal_panel/splsda pattern)
# ---------------------------------------------------------------------------


def _normalize_n_jobs(n_jobs: int) -> int:
    return 1 if n_jobs == 0 else n_jobs


def _tune_threshold(y_true: np.ndarray, scores: np.ndarray, metric: str) -> float:
    """Pick the decision threshold maximizing `metric` on given (in-fold) scores."""
    candidates = np.unique(scores)
    if candidates.size == 0:
        return 0.5
    thresholds = np.concatenate([[candidates[0] - 1e-6], (candidates[:-1] + candidates[1:]) / 2.0, [candidates[-1] + 1e-6]])
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)
        if metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, preds)
        else:
            raise ValueError(f"Unsupported threshold_metric '{metric}'")
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _fit_tuned_pls_da(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    component_grid: list[int],
    n_inner_splits: int,
    threshold_metric: str,
    random_state: int,
) -> tuple[StandardScaler, PLSRegression, float]:
    """Tune n_components + decision threshold on inner CV; fit final model on all of X_train."""
    class_counts = np.bincount(y_train)
    n_splits = min(n_inner_splits, int(class_counts.min()))
    best_n_comp = component_grid[0]
    best_threshold = 0.5
    best_score = -np.inf

    if n_splits >= 2:
        inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = list(inner_cv.split(X_train, y_train))
        max_valid_comp = min(X_train.shape[0] - 2, X_train.shape[1])
        valid_grid = [c for c in component_grid if 1 <= c <= max_valid_comp]
        if not valid_grid:
            valid_grid = [1]

        for n_comp in valid_grid:
            oof_scores = np.zeros(len(y_train), dtype=float)
            for tr_idx, te_idx in folds:
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_train[tr_idx])
                X_te = scaler.transform(X_train[te_idx])
                pls = PLSRegression(n_components=n_comp, scale=False)
                pls.fit(X_tr, y_train[tr_idx])
                oof_scores[te_idx] = pls.predict(X_te).ravel()
            threshold = _tune_threshold(y_train, oof_scores, threshold_metric)
            preds = (oof_scores >= threshold).astype(int)
            score = balanced_accuracy_score(y_train, preds)
            if score > best_score:
                best_score = score
                best_n_comp = n_comp
                best_threshold = threshold

    max_final_comp = min(X_train.shape[0] - 2, X_train.shape[1])
    final_n_comp = max(1, min(best_n_comp, max_final_comp))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = PLSRegression(n_components=final_n_comp, scale=False)
    model.fit(X_scaled, y_train)
    return scaler, model, best_threshold


def _fit_tuned_logistic_threshold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    c_grid: list[float],
    n_inner_splits: int,
    threshold_metric: str,
    random_state: int,
) -> tuple[StandardScaler, Any, float]:
    class_counts = np.bincount(y_train)
    n_splits = min(n_inner_splits, int(class_counts.min()))
    best_c = c_grid[0]
    best_threshold = 0.5
    best_score = -np.inf

    if n_splits >= 2:
        inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = list(inner_cv.split(X_train, y_train))
        for c in c_grid:
            oof_scores = np.zeros(len(y_train), dtype=float)
            for tr_idx, te_idx in folds:
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_train[tr_idx])
                X_te = scaler.transform(X_train[te_idx])
                model = l1_logistic_model(c=c, random_state=random_state)
                fit_with_convergence_capture(model, X_tr, y_train[tr_idx], context="cvd_ml logistic inner CV")
                oof_scores[te_idx] = model.predict_proba(X_te)[:, 1]
            threshold = _tune_threshold(y_train, oof_scores, threshold_metric)
            preds = (oof_scores >= threshold).astype(int)
            score = balanced_accuracy_score(y_train, preds)
            if score > best_score:
                best_score = score
                best_c = c
                best_threshold = threshold

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = l1_logistic_model(c=best_c, random_state=random_state)
    fit_with_convergence_capture(model, X_scaled, y_train, context="cvd_ml logistic final")
    return scaler, model, best_threshold


def _fit_tuned_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_estimators_grid: list[int],
    max_depth_grid: list[int | None],
    n_inner_splits: int,
    threshold_metric: str,
    random_state: int,
    n_jobs: int,
) -> tuple[RandomForestClassifier, float]:
    """Random Forest needs no external scaler; tune hyperparameters + threshold on inner CV."""
    class_counts = np.bincount(y_train)
    n_splits = min(n_inner_splits, int(class_counts.min()))
    best_params = (n_estimators_grid[0], max_depth_grid[0])
    best_threshold = 0.5
    best_score = -np.inf

    if n_splits >= 2:
        inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = list(inner_cv.split(X_train, y_train))
        for n_estimators in n_estimators_grid:
            for max_depth in max_depth_grid:
                oof_scores = np.zeros(len(y_train), dtype=float)
                for tr_idx, te_idx in folds:
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=n_jobs,
                    )
                    model.fit(X_train[tr_idx], y_train[tr_idx])
                    oof_scores[te_idx] = model.predict_proba(X_train[te_idx])[:, 1]
                threshold = _tune_threshold(y_train, oof_scores, threshold_metric)
                preds = (oof_scores >= threshold).astype(int)
                score = balanced_accuracy_score(y_train, preds)
                if score > best_score:
                    best_score = score
                    best_params = (n_estimators, max_depth)
                    best_threshold = threshold

    final_model = RandomForestClassifier(
        n_estimators=best_params[0],
        max_depth=best_params[1],
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=n_jobs,
    )
    final_model.fit(X_train, y_train)
    return final_model, best_threshold


# ---------------------------------------------------------------------------
# Nested CV evaluators — one per model family
# ---------------------------------------------------------------------------


def _nested_cv_pls_da(
    X: np.ndarray,
    y: np.ndarray,
    *,
    component_grid: list[int],
    outer_cv,
    n_inner_splits: int,
    threshold_metric: str,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_pred, y_score) out-of-fold arrays via repeated nested CV.

    With RepeatedStratifiedKFold each subject is scored once per repeat; the
    caller averages repeats. Returns stacked per-repeat predictions instead —
    see `_run_nested_cv_repeated` for the repeat-averaging wrapper.
    """
    y_pred = np.zeros_like(y, dtype=int)
    y_score = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in outer_cv:
        X_train, y_train = X[train_idx], y[train_idx]
        X_test = X[test_idx]
        scaler, model, threshold = _fit_tuned_pls_da(
            X_train,
            y_train,
            component_grid=component_grid,
            n_inner_splits=n_inner_splits,
            threshold_metric=threshold_metric,
            random_state=random_state,
        )
        scores = model.predict(scaler.transform(X_test)).ravel()
        y_score[test_idx] = scores
        y_pred[test_idx] = (scores >= threshold).astype(int)
    return y_pred, y_score


def _nested_cv_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    c_grid: list[float],
    outer_cv,
    n_inner_splits: int,
    threshold_metric: str,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    y_pred = np.zeros_like(y, dtype=int)
    y_score = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in outer_cv:
        X_train, y_train = X[train_idx], y[train_idx]
        X_test = X[test_idx]
        scaler, model, threshold = _fit_tuned_logistic_threshold(
            X_train,
            y_train,
            c_grid=c_grid,
            n_inner_splits=n_inner_splits,
            threshold_metric=threshold_metric,
            random_state=random_state,
        )
        scores = model.predict_proba(scaler.transform(X_test))[:, 1]
        y_score[test_idx] = scores
        y_pred[test_idx] = (scores >= threshold).astype(int)
    return y_pred, y_score


def _nested_cv_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_estimators_grid: list[int],
    max_depth_grid: list[int | None],
    outer_cv,
    n_inner_splits: int,
    threshold_metric: str,
    random_state: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    y_pred = np.zeros_like(y, dtype=int)
    y_score = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in outer_cv:
        X_train, y_train = X[train_idx], y[train_idx]
        X_test = X[test_idx]
        model, threshold = _fit_tuned_random_forest(
            X_train,
            y_train,
            n_estimators_grid=n_estimators_grid,
            max_depth_grid=max_depth_grid,
            n_inner_splits=n_inner_splits,
            threshold_metric=threshold_metric,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        scores = model.predict_proba(X_test)[:, 1]
        y_score[test_idx] = scores
        y_pred[test_idx] = (scores >= threshold).astype(int)
    return y_pred, y_score


_MODEL_EVALUATORS = {
    "pls_da": _nested_cv_pls_da,
    "l1_logistic": _nested_cv_logistic,
    "random_forest": _nested_cv_random_forest,
}


def _run_repeated_nested_cv(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    ml_cfg: dict[str, Any],
    n_jobs: int,
) -> pd.DataFrame:
    """Run RepeatedStratifiedKFold nested CV; return one row per (repeat, subject_index)."""
    n_outer_splits = int(ml_cfg.get("n_outer_splits", 5))
    n_outer_repeats = int(ml_cfg.get("n_outer_repeats", 10))
    n_inner_splits = int(ml_cfg.get("n_inner_splits", 4))
    threshold_metric = str(ml_cfg.get("threshold_metric", "balanced_accuracy"))
    random_state = int(ml_cfg.get("random_state", 42))

    n_splits = min(n_outer_splits, int(np.bincount(y).min()))
    repeated_cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_outer_repeats, random_state=random_state)

    rows: list[dict[str, object]] = []
    splits = list(repeated_cv.split(X, y))
    n_splits_per_repeat = n_splits
    for repeat_index in range(n_outer_repeats):
        repeat_splits = splits[repeat_index * n_splits_per_repeat : (repeat_index + 1) * n_splits_per_repeat]

        if model_name == "pls_da":
            y_pred, y_score = _nested_cv_pls_da(
                X, y,
                component_grid=list(ml_cfg.get("pls_components_grid", [1, 2, 3, 4, 5, 6, 8, 10])),
                outer_cv=repeat_splits,
                n_inner_splits=n_inner_splits,
                threshold_metric=threshold_metric,
                random_state=random_state + repeat_index,
            )
        elif model_name == "l1_logistic":
            y_pred, y_score = _nested_cv_logistic(
                X, y,
                c_grid=list(ml_cfg.get("logistic_c_grid", [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])),
                outer_cv=repeat_splits,
                n_inner_splits=n_inner_splits,
                threshold_metric=threshold_metric,
                random_state=random_state + repeat_index,
            )
        elif model_name == "random_forest":
            y_pred, y_score = _nested_cv_random_forest(
                X, y,
                n_estimators_grid=list(ml_cfg.get("rf_n_estimators_grid", [200, 500])),
                max_depth_grid=list(ml_cfg.get("rf_max_depth_grid", [None, 4, 8])),
                outer_cv=repeat_splits,
                n_inner_splits=n_inner_splits,
                threshold_metric=threshold_metric,
                random_state=random_state + repeat_index,
                n_jobs=n_jobs,
            )
        else:
            raise ValueError(f"Unknown model '{model_name}'")

        for subject_index in range(len(y)):
            rows.append(
                {
                    "repeat": repeat_index,
                    "subject_index": subject_index,
                    "y_true": int(y[subject_index]),
                    "y_pred": int(y_pred[subject_index]),
                    "y_score": float(y_score[subject_index]),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    except ValueError:
        metrics["roc_auc"] = np.nan
    try:
        metrics["pr_auc"] = average_precision_score(y_true, y_score)
    except ValueError:
        metrics["pr_auc"] = np.nan
    return metrics


def _aggregate_repeat_metrics(oof_predictions: pd.DataFrame) -> dict[str, float]:
    """Average each metric across repeats (each repeat covers every subject once)."""
    per_repeat: list[dict[str, float]] = []
    for _repeat, group in oof_predictions.groupby("repeat"):
        per_repeat.append(
            _classification_metrics(
                group["y_true"].to_numpy(),
                group["y_pred"].to_numpy(),
                group["y_score"].to_numpy(),
            )
        )
    repeat_df = pd.DataFrame(per_repeat)
    return {f"{metric}_mean": float(repeat_df[metric].mean()) for metric in repeat_df.columns} | {
        f"{metric}_std": float(repeat_df[metric].std(ddof=1)) if len(repeat_df) > 1 else 0.0
        for metric in repeat_df.columns
    }


# ---------------------------------------------------------------------------
# Feature importance (final full-data descriptive fit — not a CV estimate)
# ---------------------------------------------------------------------------


def _feature_importance(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    ml_cfg: dict[str, Any],
    n_jobs: int,
) -> pd.DataFrame:
    random_state = int(ml_cfg.get("random_state", 42))
    n_inner_splits = int(ml_cfg.get("n_inner_splits", 4))
    threshold_metric = str(ml_cfg.get("threshold_metric", "balanced_accuracy"))

    if model_name == "pls_da":
        scaler, model, _threshold = _fit_tuned_pls_da(
            X, y,
            component_grid=list(ml_cfg.get("pls_components_grid", [1, 2, 3, 4, 5, 6, 8, 10])),
            n_inner_splits=n_inner_splits,
            threshold_metric=threshold_metric,
            random_state=random_state,
        )
        vip = _calculate_vip(model)
        return pd.DataFrame(
            {
                "feature_name": feature_names,
                "importance": vip,
                "importance_type": "vip",
                "estimation_scope": "full_data_descriptive",
                "validated": False,
            }
        )
    if model_name == "l1_logistic":
        scaler, model, _threshold = _fit_tuned_logistic_threshold(
            X, y,
            c_grid=list(ml_cfg.get("logistic_c_grid", [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])),
            n_inner_splits=n_inner_splits,
            threshold_metric=threshold_metric,
            random_state=random_state,
        )
        coefs = model.coef_.ravel()
        return pd.DataFrame(
            {
                "feature_name": feature_names,
                "importance": coefs,
                "importance_type": "l1_logistic_coefficient",
                "estimation_scope": "full_data_descriptive",
                "validated": False,
            }
        )
    if model_name == "random_forest":
        model, _threshold = _fit_tuned_random_forest(
            X, y,
            n_estimators_grid=list(ml_cfg.get("rf_n_estimators_grid", [200, 500])),
            max_depth_grid=list(ml_cfg.get("rf_max_depth_grid", [None, 4, 8])),
            n_inner_splits=n_inner_splits,
            threshold_metric=threshold_metric,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        result = permutation_importance(
            model, X, y, n_repeats=30, random_state=random_state, n_jobs=n_jobs, scoring="balanced_accuracy"
        )
        return pd.DataFrame(
            {
                "feature_name": feature_names,
                "importance": result.importances_mean,
                "importance_std": result.importances_std,
                "importance_type": "permutation_importance",
                "estimation_scope": "full_data_descriptive",
                "validated": False,
            }
        )
    raise ValueError(f"Unknown model '{model_name}'")


# ---------------------------------------------------------------------------
# Public entry point (Phase A + Phase B only — permutation/bootstrap/report
# land in a follow-up phase per plan_ml.md)
# ---------------------------------------------------------------------------


def run_control_vs_disease_ml_pipeline(
    *,
    feature_matrix_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Path]:
    cfg_in = _cfg(cfg)
    ml_cfg = _ml_cfg(cfg_in)
    if not bool(ml_cfg.get("enabled", True)):
        return {}

    target_dir = Path(output_dir) if output_dir is not None else get_ml_classification_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    phase_timings: list[dict[str, object]] = []
    outputs: dict[str, Path] = {}

    with _phase_timer(phase_timings, "load_cohort", "Load cohort and labels"):
        frame = _load_cohort(feature_matrix_path)
        y = _labels(frame)

    with _phase_timer(phase_timings, "cohort_summary", "Cohort summary"):
        cohort_df = _cohort_summary(frame)
        cohort_path = write_csv_atomic(cohort_df, target_dir / "cohort_summary.csv", index=False)
        outputs["cohort_summary"] = cohort_path

    representations = list(ml_cfg.get("representations", ["whole", "band_panel"]))
    n_jobs = _normalize_n_jobs(int(ml_cfg.get("n_jobs", -1)))

    comparison_rows: list[dict[str, object]] = []
    for representation in representations:
        with _phase_timer(
            phase_timings, f"features_{representation}", f"Build {representation} feature matrix"
        ):
            X, feature_names = _representation_matrix(frame, representation, cfg_in)
            valid_rows = np.isfinite(X).all(axis=1)
            if not valid_rows.all():
                logger.warning(
                    "%d subject(s) dropped from representation '%s' due to non-finite features",
                    int((~valid_rows).sum()),
                    representation,
                )
            X_valid = X[valid_rows]
            y_valid = y[valid_rows]

        for model_name in _MODEL_EVALUATORS:
            with _phase_timer(
                phase_timings, f"{model_name}_{representation}", f"{model_name} ({representation}) nested CV"
            ):
                oof = _run_repeated_nested_cv(model_name, X_valid, y_valid, ml_cfg=ml_cfg, n_jobs=n_jobs)
                oof_path = write_csv_atomic(
                    oof, target_dir / f"{model_name}_{representation}_cv_predictions.csv", index=False
                )
                outputs[f"{model_name}_{representation}_cv_predictions"] = oof_path

                metrics = _aggregate_repeat_metrics(oof)
                comparison_rows.append(
                    {
                        "model": model_name,
                        "representation": representation,
                        "n_subjects": int(len(y_valid)),
                        **metrics,
                    }
                )

                importance_df = _feature_importance(
                    model_name, X_valid, y_valid, feature_names, ml_cfg=ml_cfg, n_jobs=n_jobs
                )
                importance_path = write_csv_atomic(
                    importance_df,
                    target_dir / f"{model_name}_{representation}_feature_importance.csv",
                    index=False,
                )
                outputs[f"{model_name}_{representation}_feature_importance"] = importance_path

    with _phase_timer(phase_timings, "model_comparison", "Write model comparison table"):
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df.insert(0, "positive_class", POSITIVE_CLASS_LABEL)
        comparison_path = write_csv_atomic(comparison_df, target_dir / "model_comparison.csv", index=False)
        outputs["model_comparison"] = comparison_path

    with _phase_timer(phase_timings, "provenance", "Write provenance artifacts"):
        write_analysis_provenance(
            target_dir,
            resolve_data_source(cfg_in),
            cfg_in,
            command="run_control_vs_disease_ml_pipeline",
        )

    metadata_dir = get_run_metadata_dir() if output_dir is None else target_dir
    timing_path = _write_phase_timings(phase_timings, metadata_dir)
    if timing_path is not None:
        outputs["phase_timing"] = timing_path

    logger.info("Control-vs-disease ML outputs written to %s", target_dir)
    return outputs
