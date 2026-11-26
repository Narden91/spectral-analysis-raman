"""Control-vs-disease ML classification layer (disease_vs_control mode only).

Subject-level, nested-CV supervised classification of Control vs. Disease from
whole Raman spectra. This module is a separate, read-only downstream consumer
of the Biological PCA subject feature matrix. Config validation and phase
docstrings define the confound handling, claims discipline, and data limits.

Positive class = Disease (is_control == False -> 1). Exploratory only: no
diagnostic, biomarker, causal, or clinical-utility claims. These spectral
classifiers do not include age as a predictor; availability is summarized from
the supplied cohort rather than inferred from historical data.

Output map:
    01_cohort                  class balance and confound counts
    02_model_evaluation        nested CV, importance, metrics, bootstrap CIs
    03_confounder_checks       sex-only baseline and female-only refits
    04_permutation_tests       unrestricted and sex-stratified reference distributions
    05_univariate_band_screen  descriptive Mann-Whitney tests with FDR
    06_report                  figures and final performance summary
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import time
import warnings
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from scipy import stats
from scipy.linalg import pinv
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
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
from sklearn.utils import check_random_state
from sklearn.utils.parallel import Parallel, delayed

from src import (
    config,
    get_ml_classification_dir,
    get_ml_classification_phase_dir,
    get_run_metadata_dir,
)
from src.analysis.clinical_feature_associations import (
    feature_columns,
    load_subject_feature_matrix,
)
from src.analysis.control_vs_disease_ml_report import run_control_vs_disease_ml_report
from src.analysis.io_utils import write_csv_atomic, write_json_atomic, write_parquet_atomic
from src.analysis.paired_lipid._backend import _single_thread_blas
from src.analysis.pls_regression import _calculate_vip
from src.analysis.sklearn_utils import fit_with_convergence_capture, l1_logistic_model
from src.analysis.statistics import fdr_correction
from src.data_preparation import resolve_data_source, write_analysis_provenance
from src.data_preparation.manifest import config_hash, hash_file
from src.statistics.cluster_permutation import normalize_n_jobs
from src.statistics.sequential_permutation import (
    resolve_sequential_stopping_config,
    should_stop,
)

logger = logging.getLogger(__name__)

POSITIVE_CLASS_LABEL = "Disease"
NEGATIVE_CLASS_LABEL = "Control"

_MIN_SUBJECTS_PER_CLASS = 10
_MIN_SUBJECTS_PER_CLASS_STRATIFIED = 10
_MODEL_NAMES = ("pls_da", "l1_logistic", "random_forest")
_CHECKPOINT_SCHEMA_VERSION = 2
_CHECKPOINT_FLUSH_EVERY = 25

PredictFn = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# Config access
# ---------------------------------------------------------------------------


def _cfg(cfg_in: dict[str, Any] | None = None) -> dict[str, Any]:
    return config if cfg_in is None else cfg_in


def _ml_cfg(cfg_in: dict[str, Any]) -> dict[str, Any]:
    return cfg_in.get("analysis", {}).get("control_vs_disease_ml", {})


def _pls_component_grid(ml_cfg: dict[str, Any]) -> list[int]:
    return list(ml_cfg.get("pls_components_grid", [1, 2, 3, 4, 5, 6, 8, 10]))


def _logistic_c_grid(ml_cfg: dict[str, Any]) -> list[float]:
    return list(ml_cfg.get("logistic_c_grid", [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]))


def _rf_n_estimators_grid(ml_cfg: dict[str, Any]) -> list[int]:
    return list(ml_cfg.get("rf_n_estimators_grid", [200, 500]))


def _rf_max_depth_grid(ml_cfg: dict[str, Any]) -> list[int | None]:
    return list(ml_cfg.get("rf_max_depth_grid", [None, 4, 8]))


def _n_outer_splits(ml_cfg: dict[str, Any]) -> int:
    return int(ml_cfg.get("n_outer_splits", 5))


def _n_inner_splits(ml_cfg: dict[str, Any]) -> int:
    return int(ml_cfg.get("n_inner_splits", 4))


def _threshold_metric(ml_cfg: dict[str, Any]) -> str:
    return str(ml_cfg.get("threshold_metric", "balanced_accuracy"))


def _random_state(ml_cfg: dict[str, Any]) -> int:
    return int(ml_cfg.get("random_state", 42))


def _n_permutations(ml_cfg: dict[str, Any]) -> int:
    return int(ml_cfg.get("n_permutations", 1000))


def _file_sha256(path: str | Path | None) -> str:
    """Return a content fingerprint for a feature matrix used by resumable work."""
    if path is None:
        return "default_feature_matrix"
    return hash_file(path)


@dataclass(frozen=True)
class _CheckpointStore:
    """Small, fingerprinted store for expensive ML resampling outputs."""

    directory: Path
    identity: dict[str, object]
    resume: bool

    @classmethod
    def create(
        cls,
        *,
        metadata_dir: Path,
        cfg_in: dict[str, Any],
        feature_matrix_path: str | Path | None,
        data_manifest_hash: str | None,
        resume: bool,
    ) -> "_CheckpointStore":
        if resume and feature_matrix_path is None:
            raise ValueError("Resuming ML requires an explicit subject feature matrix path")
        identity = {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "implementation_sha256": config_hash({
                name: hash_file(Path(__file__).with_name(name))
                for name in ("control_vs_disease_ml.py", "pls_regression.py", "sklearn_utils.py")
            }),
            "analysis_config_hash": config_hash(cfg_in),
            "ml_config_hash": config_hash(_ml_cfg(cfg_in)),
            "data_manifest_hash": data_manifest_hash or "not_supplied",
            "feature_matrix_sha256": _file_sha256(feature_matrix_path),
        }
        fingerprint = config_hash(identity)
        checkpoint_root = metadata_dir / "control_vs_disease_ml_checkpoints"
        directory = checkpoint_root / fingerprint
        metadata_path = directory / "checkpoint_manifest.json"
        preexisting = metadata_path.exists()

        if resume and not preexisting and any(checkpoint_root.glob("*/checkpoint_manifest.json")):
            raise ValueError(
                "No ML resume checkpoint matches current config, data manifest, or feature matrix. "
                "Run without --resume-ml to start a new ML resampling run."
            )
        if resume and preexisting:
            stored = json.loads(metadata_path.read_text(encoding="utf-8"))
            if stored != identity:
                raise ValueError(
                    "ML resume checkpoint does not match current config, data manifest, or feature matrix. "
                    "Run without --resume-ml to start a new ML resampling run."
                )
        directory.mkdir(parents=True, exist_ok=True)
        write_json_atomic(metadata_path, identity)
        return cls(directory=directory, identity=identity, resume=resume)

    def resampling_path(self, kind: str, model_name: str, representation: str, label: str) -> Path:
        return self.directory / f"{kind}_{model_name}_{representation}_{label}.parquet"

    def stage_path(self, model_name: str, representation: str, label: str) -> Path:
        return self.directory / f"stage_{model_name}_{representation}_{label}.json"

    def stage_complete(
        self, model_name: str, representation: str, label: str, *, artifacts: Sequence[Path] = (),
    ) -> bool:
        path = self.stage_path(model_name, representation, label)
        if not self.resume or not path.is_file():
            return False
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("identity") != self.identity or stored.get("status") != "ok":
            return False
        hashes = stored.get("artifacts", {})
        return all(
            artifact.is_file() and hashes.get(str(artifact.resolve())) == hash_file(artifact)
            for artifact in artifacts
        )

    def mark_stage_complete(
        self, model_name: str, representation: str, label: str, *, artifacts: Sequence[Path] = (),
    ) -> None:
        write_json_atomic(
            self.stage_path(model_name, representation, label),
            {"identity": self.identity, "status": "ok",
             "artifacts": {str(path.resolve()): hash_file(path) for path in artifacts}},
        )


# ---------------------------------------------------------------------------
# Phase timing (local — mirrors disease_only_pipeline._phase_timer without
# importing that module's private symbols across a module boundary)
# ---------------------------------------------------------------------------


@contextmanager
def _phase_timer(
    timings: list[dict[str, object]],
    phase_key: str,
    phase_label: str,
    *,
    metadata_dir: Path,
    progress: Progress | None = None,
    task_id: int | None = None,
):
    if progress is not None and task_id is not None:
        progress.update(task_id, description=f"[bold blue]{phase_label}")
    logger.info("Starting control-vs-disease ML phase: %s", phase_label)
    start = time.perf_counter()
    status = "ok"
    try:
        yield
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except Exception:
        status = "failed"
        raise
    finally:
        elapsed = time.perf_counter() - start
        timings.append(
            {
                "phase": phase_key,
                "label": phase_label,
                "elapsed_seconds": elapsed,
                "status": status,
            }
        )
        _write_phase_timings(timings, metadata_dir)
        logger.info("Finished control-vs-disease ML phase: %s status=%s elapsed=%.2fs", phase_label, status, elapsed)
        if progress is not None and task_id is not None:
            progress.advance(task_id)


def _write_phase_timings(timings: list[dict[str, object]], metadata_dir: Path) -> Path | None:
    if not timings:
        return None
    path = metadata_dir / "control_vs_disease_ml_phase_timing.csv"
    write_csv_atomic(pd.DataFrame(timings), path, index=False)
    return path


@contextmanager
def _progress_task(progress: Progress, description: str, total: int):
    """Create a temporary detailed progress task and always remove it."""
    task_id = progress.add_task(description, total=total)
    try:
        yield task_id
    finally:
        progress.remove_task(task_id)


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
        mislabeled = frame.loc[frame["is_control"].astype(bool) & frame["diagnosis_family"].notna()]
        if not mislabeled.empty:
            logger.warning(
                "Label anomaly: %d subject(s) marked is_control=True but carry a nonmissing "
                "diagnosis_family (%s). Verify labels before treating results as final. Subject_id(s): %s",
                len(mislabeled),
                sorted(mislabeled["diagnosis_family"].dropna().unique().tolist()),
                mislabeled["subject_id"].astype(str).tolist(),
            )

    return frame


def _labels(frame: pd.DataFrame) -> np.ndarray:
    return (~frame["is_control"].astype(bool)).astype(int).to_numpy()


def _sex_column(frame: pd.DataFrame) -> str | None:
    """Prefer clinical sex; legacy gender metadata is never a per-row replacement."""
    return next((c for c in ("sex", "gender") if c in frame.columns), None)


# ---------------------------------------------------------------------------
# Feature representations
# ---------------------------------------------------------------------------


def _whole_spectrum_features(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    columns = feature_columns(frame)
    if not columns:
        raise ValueError("Subject feature matrix has no whole-spectrum feature columns")
    return frame[columns].to_numpy(dtype=float), columns


def _band_feature(frame: pd.DataFrame, all_columns: list[str], block: str, lower: float, upper: float) -> pd.Series:
    cols = [
        column
        for column in all_columns
        if column.startswith(f"{block}__") and lower <= float(column.split("__", 1)[1]) <= upper
    ]
    if not cols:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return frame[cols].mean(axis=1)


def _band_panel(frame: pd.DataFrame, all_columns: list[str], cfg_in: dict[str, Any]) -> pd.DataFrame:
    """Shared by the band-panel representation and univariate band screen."""
    bands_cfg = cfg_in.get("whole_spectra", {}).get("bands", {})
    if not bands_cfg:
        raise ValueError("config whole_spectra.bands is empty; band_panel representation requires the 21-band panel")
    band_series = {}
    for band_name, limits in bands_cfg.items():
        lower, upper = float(limits[0]), float(limits[1])
        block = "1300" if upper < 2000 else "2850"
        band_series[str(band_name)] = _band_feature(frame, all_columns, block, lower, upper)
    return pd.DataFrame(band_series, index=frame.index)


def _band_panel_features(
    frame: pd.DataFrame, all_columns: list[str], cfg_in: dict[str, Any]
) -> tuple[np.ndarray, list[str]]:
    band_frame = _band_panel(frame, all_columns, cfg_in)
    return band_frame.to_numpy(dtype=float), list(band_frame.columns)


def _representation_matrix(
    frame: pd.DataFrame, representation: str, cfg_in: dict[str, Any]
) -> tuple[np.ndarray, list[str]]:
    if representation == "whole":
        return _whole_spectrum_features(frame)
    if representation == "band_panel":
        return _band_panel_features(frame, feature_columns(frame), cfg_in)
    raise ValueError(f"Unknown feature representation '{representation}'")


# ---------------------------------------------------------------------------
# Cohort description
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
        {
            "metric": "majority_class_baseline_accuracy",
            "value": max(n_control, n_disease) / n_total if n_total else np.nan,
        },
        {"metric": "majority_class_baseline_balanced_accuracy", "value": 0.5},
        {"metric": "positive_class", "value": POSITIVE_CLASS_LABEL},
        {"metric": "negative_class", "value": NEGATIVE_CLASS_LABEL},
    ]

    sex_column = _sex_column(frame)
    rows.append({"metric": "sex_covariate_column", "value": sex_column or "not_available"})
    if sex_column is not None:
        crosstab = pd.crosstab(is_control, frame[sex_column].astype(str))
        for control_flag, row in crosstab.iterrows():
            group_label = NEGATIVE_CLASS_LABEL if control_flag else POSITIVE_CLASS_LABEL
            for sex_value, count in row.items():
                rows.append({"metric": f"n_{group_label.lower()}_{sex_value}", "value": int(count)})
        if crosstab.shape == (2, 2):
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
    rows.append({"metric": "age_adjustment_status", "value": "not_modeled"})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model fitters — one function per model family, each doing its own in-fold
# hyperparameter + threshold tuning and returning a uniform _ModelFit. Every
# other function in this module (nested CV, permutation, bootstrap, feature
# importance) dispatches through `_fit_tuned_model` instead of branching on
# model_name itself.
# ---------------------------------------------------------------------------


def _tune_threshold_and_score(y_true: np.ndarray, scores: np.ndarray, metric: str) -> tuple[float, float]:
    """Pick threshold maximizing balanced accuracy without repeated sklearn calls."""
    if metric != "balanced_accuracy":
        raise ValueError(f"Unsupported threshold_metric '{metric}'")
    candidates = np.unique(scores)
    if candidates.size == 0:
        return 0.5, np.nan
    thresholds = np.concatenate(
        [[candidates[0] - 1e-6], (candidates[:-1] + candidates[1:]) / 2.0, [candidates[-1] + 1e-6]]
    )
    predictions = scores[None, :] >= thresholds[:, None]
    positive = y_true == 1
    negative = ~positive
    sensitivity = predictions[:, positive].mean(axis=1) if positive.any() else np.full(len(thresholds), np.nan)
    specificity = (~predictions[:, negative]).mean(axis=1) if negative.any() else np.full(len(thresholds), np.nan)
    balanced_accuracy = (sensitivity + specificity) / 2.0
    best_index = int(np.argmax(balanced_accuracy))
    return float(thresholds[best_index]), float(balanced_accuracy[best_index])


def _tune_threshold(y_true: np.ndarray, scores: np.ndarray, metric: str) -> float:
    """Pick the decision threshold maximizing `metric` on given (in-fold) scores."""
    return _tune_threshold_and_score(y_true, scores, metric)[0]


def _inner_cv_folds(
    X_train: np.ndarray, y_train: np.ndarray, n_inner_splits: int, random_state: int,
    *, groups: np.ndarray | None = None,
) -> list | None:
    if groups is None:
        n_splits = min(n_inner_splits, int(np.bincount(y_train).min()))
        if n_splits < 2:
            return None
        inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(inner_cv.split(X_train, y_train))

    groups = np.asarray(groups)
    if groups.shape != y_train.shape:
        raise ValueError("Inner CV participant groups must match training rows")
    _, first, inverse = np.unique(groups, return_index=True, return_inverse=True)
    group_y = y_train[first]
    if not np.array_equal(y_train, group_y[inverse]):
        raise ValueError("Copies of a participant have inconsistent labels")
    n_splits = min(n_inner_splits, int(np.bincount(group_y, minlength=2).min()))
    if n_splits < 2:
        return None
    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return [
        (np.flatnonzero(np.isin(inverse, train_groups)), np.flatnonzero(np.isin(inverse, test_groups)))
        for train_groups, test_groups in inner_cv.split(X_train[first], group_y)
    ]


def _scaled_inner_folds(
    X_train: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Scale each inner training/test pair once and share it across a model grid."""
    scaled: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for tr_idx, te_idx in folds:
        scaler = StandardScaler(copy=False)
        scaled.append((tr_idx, te_idx, scaler.fit_transform(X_train[tr_idx]), scaler.transform(X_train[te_idx])))
    return scaled


def _scale_outer_folds(
    X: np.ndarray,
    outer_cv: Sequence[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pre-scale fixed outer train/test splits once for permutation runs."""
    scaled: list[tuple[np.ndarray, np.ndarray]] = []
    for train_idx, test_idx in outer_cv:
        scaler = StandardScaler(copy=False)
        scaled.append((scaler.fit_transform(X[train_idx]), scaler.transform(X[test_idx])))
    return scaled


@dataclass(frozen=True)
class _ModelFit:
    """A model fitted on one training fold, ready to score a test fold.

    `estimator` is the raw fitted sklearn object (PLSRegression, LogisticRegression,
    or RandomForestClassifier) — nested CV / permutation callers only need
    `predict_proba`/`threshold`, but feature-importance extraction needs the
    estimator itself (VIP for PLS, coefficients for logistic, permutation
    importance for the forest).
    """

    predict_proba: PredictFn
    threshold: float
    estimator: object
    scaler: StandardScaler | None


def _fit_pls_da(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ml_cfg: dict[str, Any],
    n_inner_splits: int,
    threshold_metric: str,
    random_state: int,
    n_jobs: int,
    groups: np.ndarray | None = None,
    inner_folds: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    outer_scaled_train: np.ndarray | None = None,
) -> _ModelFit:
    component_grid = _pls_component_grid(ml_cfg)
    folds = inner_folds if inner_folds is not None else _inner_cv_folds(
        X_train, y_train, n_inner_splits, random_state, groups=groups
    )
    best_n_comp, best_threshold, best_score = component_grid[0], 0.5, -np.inf

    if folds is not None:
        scaled_folds = _scaled_inner_folds(X_train, folds)
        max_valid_comp = min(min(len(tr_idx) for tr_idx, _ in folds), X_train.shape[1])
        valid_grid = [c for c in component_grid if 1 <= c <= max_valid_comp] or [1]
        unique_valid = list(dict.fromkeys(valid_grid))
        if len(unique_valid) == 1:
            n_comp = unique_valid[0]
            oof_scores = np.zeros(len(y_train), dtype=float)
            for tr_idx, te_idx, X_tr, X_te in scaled_folds:
                pls = PLSRegression(n_components=n_comp, scale=False, copy=False)
                pls.fit(X_tr, y_train[tr_idx])
                oof_scores[te_idx] = pls.predict(X_te).ravel()
            best_threshold, best_score = _tune_threshold_and_score(y_train, oof_scores, threshold_metric)
            best_n_comp = n_comp
        else:
            max_comp = max(unique_valid)
            oof_by_comp = {c: np.zeros(len(y_train), dtype=float) for c in unique_valid}
            for tr_idx, te_idx, X_tr, X_te in scaled_folds:
                pls = PLSRegression(n_components=max_comp, scale=False, copy=False)
                pls.fit(X_tr, y_train[tr_idx])
                W = pls.x_weights_
                P = pls.x_loadings_
                Q = pls.y_loadings_
                X_te_c = X_te - pls._x_mean
                y_mean = pls._y_mean
                for n_comp in unique_valid:
                    rot_c = np.dot(
                        W[:, :n_comp],
                        pinv(np.dot(P[:, :n_comp].T, W[:, :n_comp]), check_finite=False),
                    )
                    coef_c = np.dot(rot_c, Q[:, :n_comp].T)
                    oof_by_comp[n_comp][te_idx] = (X_te_c @ coef_c + y_mean).ravel()
            for n_comp in valid_grid:
                threshold, score = _tune_threshold_and_score(y_train, oof_by_comp[n_comp], threshold_metric)
                if score > best_score:
                    best_score, best_n_comp, best_threshold = score, n_comp, threshold

    final_n_comp = max(1, min(best_n_comp, min(X_train.shape[0] - 2, X_train.shape[1])))
    if outer_scaled_train is None:
        scaler: StandardScaler | None = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        pls_copy = False
    else:
        scaler = None
        X_scaled = outer_scaled_train
        pls_copy = True
    model = PLSRegression(n_components=final_n_comp, scale=False, copy=pls_copy)
    model.fit(X_scaled, y_train)
    predict_fn = (
        (lambda X: model.predict(scaler.transform(X)).ravel())
        if scaler is not None
        else (lambda X: model.predict(X).ravel())
    )
    return _ModelFit(
        predict_proba=predict_fn,
        threshold=best_threshold,
        estimator=model,
        scaler=scaler,
    )


def _fit_l1_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ml_cfg: dict[str, Any],
    n_inner_splits: int,
    threshold_metric: str,
    random_state: int,
    n_jobs: int,
    groups: np.ndarray | None = None,
    inner_folds: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    outer_scaled_train: np.ndarray | None = None,
) -> _ModelFit:
    c_grid = _logistic_c_grid(ml_cfg)
    folds = inner_folds if inner_folds is not None else _inner_cv_folds(
        X_train, y_train, n_inner_splits, random_state, groups=groups
    )
    best_c, best_threshold, best_score = c_grid[0], 0.5, -np.inf

    if folds is not None:
        scaled_folds = _scaled_inner_folds(X_train, folds)
        for c in c_grid:
            oof_scores = np.zeros(len(y_train), dtype=float)
            for tr_idx, te_idx, X_tr, X_te in scaled_folds:
                model = l1_logistic_model(c=c, random_state=random_state)
                fit_with_convergence_capture(model, X_tr, y_train[tr_idx], context="cvd_ml logistic inner CV")
                oof_scores[te_idx] = model.predict_proba(X_te)[:, 1]
            threshold, score = _tune_threshold_and_score(y_train, oof_scores, threshold_metric)
            if score > best_score:
                best_score, best_c, best_threshold = score, c, threshold

    if outer_scaled_train is None:
        scaler: StandardScaler | None = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
    else:
        scaler = None
        X_scaled = outer_scaled_train
    model = l1_logistic_model(c=best_c, random_state=random_state)
    fit_with_convergence_capture(model, X_scaled, y_train, context="cvd_ml logistic final")
    predict_fn = (
        (lambda X: model.predict_proba(scaler.transform(X))[:, 1])
        if scaler is not None
        else (lambda X: model.predict_proba(X)[:, 1])
    )
    return _ModelFit(
        predict_proba=predict_fn,
        threshold=best_threshold,
        estimator=model,
        scaler=scaler,
    )


def _fit_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ml_cfg: dict[str, Any],
    n_inner_splits: int,
    threshold_metric: str,
    random_state: int,
    n_jobs: int,
    groups: np.ndarray | None = None,
    inner_folds: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
) -> _ModelFit:
    n_estimators_grid = _rf_n_estimators_grid(ml_cfg)
    max_depth_grid = _rf_max_depth_grid(ml_cfg)
    folds = inner_folds if inner_folds is not None else _inner_cv_folds(
        X_train, y_train, n_inner_splits, random_state, groups=groups
    )
    best_params, best_threshold, best_score = (n_estimators_grid[0], max_depth_grid[0]), 0.5, -np.inf

    if folds is not None:
        unique_estimators = sorted(set(n_estimators_grid))
        oof_by_params: dict[tuple[int, int | None], np.ndarray] = {
            (n_estimators, max_depth): np.zeros(len(y_train), dtype=float)
            for n_estimators in unique_estimators
            for max_depth in max_depth_grid
        }
        for max_depth in max_depth_grid:
            for tr_idx, te_idx in folds:
                X_tr_fold = X_train[tr_idx]
                y_tr_fold = y_train[tr_idx]
                X_te_fold = X_train[te_idx]
                model: RandomForestClassifier | None = None
                for n_estimators in unique_estimators:
                    if model is None:
                        model = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            class_weight="balanced_subsample",
                            random_state=random_state,
                            n_jobs=n_jobs,
                            warm_start=True,
                        )
                    else:
                        model.set_params(n_estimators=n_estimators)
                    # Training data never changes while this forest grows; sklearn's
                    # generic warm-start warning only applies when it does.
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message='class_weight presets "balanced" or "balanced_subsample" are not recommended for warm_start.*',
                            category=UserWarning,
                        )
                        model.fit(X_tr_fold, y_tr_fold)
                    oof_by_params[(n_estimators, max_depth)][te_idx] = model.predict_proba(X_te_fold)[:, 1]

        # Preserve configured grid order for deterministic tie breaking.
        for n_estimators in n_estimators_grid:
            for max_depth in max_depth_grid:
                threshold, score = _tune_threshold_and_score(
                    y_train, oof_by_params[(n_estimators, max_depth)], threshold_metric
                )
                if score > best_score:
                    best_score, best_params, best_threshold = score, (n_estimators, max_depth), threshold

    final_model = RandomForestClassifier(
        n_estimators=best_params[0],
        max_depth=best_params[1],
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=n_jobs,
    )
    final_model.fit(X_train, y_train)
    return _ModelFit(
        predict_proba=lambda X: final_model.predict_proba(X)[:, 1],
        threshold=best_threshold,
        estimator=final_model,
        scaler=None,
    )


_MODEL_FITTERS: dict[str, Callable[..., _ModelFit]] = {
    "pls_da": _fit_pls_da,
    "l1_logistic": _fit_l1_logistic,
    "random_forest": _fit_random_forest,
}


def _fit_tuned_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    ml_cfg: dict[str, Any],
    n_inner_splits: int,
    threshold_metric: str,
    random_state: int,
    n_jobs: int,
    groups: np.ndarray | None = None,
    inner_folds: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    outer_scaled_train: np.ndarray | None = None,
) -> _ModelFit:
    """Tune hyperparameters + decision threshold on inner CV; fit final model on all of X_train.

    All in-fold work (scaling, hyperparameter search, threshold selection)
    happens strictly within X_train/y_train — callers never pass test-fold
    rows here, which is what keeps every nested-CV and permutation-test path
    leakage-safe by construction rather than by convention.
    """
    fitter = _MODEL_FITTERS.get(model_name)
    if fitter is None:
        raise ValueError(f"Unknown model '{model_name}'")
    kwargs: dict[str, Any] = {}
    if groups is not None:
        kwargs["groups"] = groups
    if inner_folds is not None:
        kwargs["inner_folds"] = inner_folds
    if outer_scaled_train is not None and model_name in ("pls_da", "l1_logistic"):
        kwargs["outer_scaled_train"] = outer_scaled_train
    return fitter(X_train, y_train, ml_cfg, n_inner_splits, threshold_metric, random_state, n_jobs, **kwargs)


# ---------------------------------------------------------------------------
# Nested CV — one implementation shared by repeated-CV point estimates and the
# single-split permutation workhorse.
# ---------------------------------------------------------------------------


def _resolve_parallel_workers(n_tasks: int, n_jobs: int | None) -> int:
    """Resolve effective worker count while preventing nested-pool oversubscription."""
    if n_tasks <= 1 or n_jobs == 0 or n_jobs == 1:
        return 1
    if os.environ.get("_PAIRED_LIPID_IN_BOOTSTRAP_WORKER") == "1":
        return 1
    if multiprocessing.parent_process() is not None:
        return 1
    try:
        import joblib.parallel

        active_backend, _ = joblib.parallel.get_active_backend()
        if getattr(active_backend, "nesting_level", 0) > 0:
            return 1
    except Exception:
        pass
    cpu_cnt = os.cpu_count() or 1
    if n_jobs is None or int(n_jobs) == -1:
        raw = cpu_cnt
    elif int(n_jobs) < -1:
        raw = max(1, cpu_cnt + 1 + int(n_jobs))
    else:
        raw = int(n_jobs)
    if raw <= 1:
        return 1
    return min(raw, n_tasks)


def _rf_inner_chain_task(
    outer_key: int,
    max_depth: int | None,
    X: np.ndarray,
    y: np.ndarray,
    global_tr_idx: np.ndarray,
    global_te_idx: np.ndarray,
    te_idx: np.ndarray,
    unique_estimators: list[int],
    fold_random_state: int,
    single_thread_blas: bool = True,
) -> tuple[int, int | None, np.ndarray, dict[int, np.ndarray]]:
    """Run one warm-start RandomForest chain across n_estimators for one inner fold."""
    with _single_thread_blas() if single_thread_blas else nullcontext():
        X_tr_fold = X[global_tr_idx]
        y_tr_fold = y[global_tr_idx]
        X_te_fold = X[global_te_idx]
        model: RandomForestClassifier | None = None
        scores_by_est: dict[int, np.ndarray] = {}
        for n_estimators in unique_estimators:
            if model is None:
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    class_weight="balanced_subsample",
                    random_state=fold_random_state,
                    n_jobs=1,
                    warm_start=True,
                )
            else:
                model.set_params(n_estimators=n_estimators)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message='class_weight presets "balanced" or "balanced_subsample" are not recommended for warm_start.*',
                    category=UserWarning,
                )
                model.fit(X_tr_fold, y_tr_fold)
            scores_by_est[n_estimators] = model.predict_proba(X_te_fold)[:, 1]
        return outer_key, max_depth, te_idx, scores_by_est


def _rf_outer_final_task(
    outer_key: int,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    best_params: tuple[int, int | None],
    best_threshold: float,
    fold_random_state: int,
    ml_cfg: dict[str, Any],
    repeat_index: int,
    fold_index: int,
    feature_names: list[str] | None,
    include_predictions: bool,
    include_importance: bool,
    single_thread_blas: bool = True,
) -> tuple[int, tuple[np.ndarray, np.ndarray, np.ndarray] | None, pd.DataFrame | None]:
    """Fit one outer RandomForest model and return predictions and/or importance."""
    with _single_thread_blas() if single_thread_blas else nullcontext():
        final_model = RandomForestClassifier(
            n_estimators=best_params[0],
            max_depth=best_params[1],
            class_weight="balanced_subsample",
            random_state=fold_random_state,
            n_jobs=1,
        )
        final_model.fit(X[train_idx], y[train_idx])
        pred_tuple: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        if include_predictions:
            scores = final_model.predict_proba(X[test_idx])[:, 1]
            pred_tuple = (test_idx, (scores >= best_threshold).astype(int), scores)
        imp_frame: pd.DataFrame | None = None
        if include_importance:
            assert feature_names is not None
            fit = _ModelFit(
                predict_proba=lambda X_in: final_model.predict_proba(X_in)[:, 1],
                threshold=best_threshold,
                estimator=final_model,
                scaler=None,
            )
            imp_frame = _outer_fold_importance_frame_from_fit(
                "random_forest",
                fit,
                X[test_idx],
                y[test_idx],
                feature_names,
                ml_cfg=ml_cfg,
                repeat_index=repeat_index,
                fold_index=fold_index,
                n_train=len(train_idx),
            )
        return outer_key, pred_tuple, imp_frame


def _run_rf_parallel_folds(
    X: np.ndarray,
    y: np.ndarray,
    fold_specs: Sequence[tuple[int, int, np.ndarray, np.ndarray, int]],
    *,
    ml_cfg: dict[str, Any],
    n_inner_splits: int,
    threshold_metric: str,
    n_jobs: int,
    feature_names: list[str] | None = None,
    include_predictions: bool = True,
    include_importance: bool = False,
    fold_callback: Callable[[], None] | None = None,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]], dict[int, pd.DataFrame]]:
    """Flatten RF inner warm-start chains and outer refits across all available workers."""
    if not fold_specs:
        return {}, {}

    n_estimators_grid = _rf_n_estimators_grid(ml_cfg)
    max_depth_grid = _rf_max_depth_grid(ml_cfg)
    unique_estimators = sorted(set(n_estimators_grid))
    unique_depths = list(dict.fromkeys(max_depth_grid))
    default_params = (n_estimators_grid[0], max_depth_grid[0])

    y_train_by_fold: dict[int, np.ndarray] = {}
    oof_by_params_by_fold: dict[int, dict[tuple[int, int | None], np.ndarray]] = {}
    best_params_by_fold: dict[int, tuple[int, int | None]] = {}
    best_threshold_by_fold: dict[int, float] = {}
    phase1_tasks: list[tuple[int, int | None, np.ndarray, np.ndarray, np.ndarray, int]] = []

    for outer_key, (_repeat_index, _fold_index, train_idx, _test_idx, fold_random_state) in enumerate(fold_specs):
        y_tr_outer = y[train_idx]
        inner_folds = _inner_cv_folds(X[train_idx], y_tr_outer, n_inner_splits, fold_random_state)
        if inner_folds is None:
            best_params_by_fold[outer_key] = default_params
            best_threshold_by_fold[outer_key] = 0.5
            continue
        y_train_by_fold[outer_key] = y_tr_outer
        oof_by_params_by_fold[outer_key] = {
            (n_estimators, max_depth): np.zeros(len(y_tr_outer), dtype=float)
            for n_estimators in unique_estimators
            for max_depth in unique_depths
        }
        for max_depth in unique_depths:
            for tr_idx, te_idx in inner_folds:
                phase1_tasks.append(
                    (
                        outer_key,
                        max_depth,
                        train_idx[tr_idx],
                        train_idx[te_idx],
                        te_idx,
                        fold_random_state,
                    )
                )

    workers = _resolve_parallel_workers(max(len(phase1_tasks), len(fold_specs)), n_jobs)
    preds_by_fold: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    imp_by_fold: dict[int, pd.DataFrame] = {}

    with Parallel(n_jobs=workers, return_as="list") as parallel:
        if phase1_tasks:
            phase1_results = parallel(
                delayed(_rf_inner_chain_task)(
                    outer_key,
                    max_depth,
                    X,
                    y,
                    global_tr_idx,
                    global_te_idx,
                    te_idx,
                    unique_estimators,
                    fold_random_state,
                    workers > 1,
                )
                for outer_key, max_depth, global_tr_idx, global_te_idx, te_idx, fold_random_state in phase1_tasks
            )
            for outer_key, max_depth, te_idx, scores_by_est in phase1_results:
                fold_oof = oof_by_params_by_fold[outer_key]
                for n_estimators, scores in scores_by_est.items():
                    fold_oof[(n_estimators, max_depth)][te_idx] = scores

            for outer_key, y_tr_outer in y_train_by_fold.items():
                fold_oof = oof_by_params_by_fold[outer_key]
                best_params, best_threshold, best_score = default_params, 0.5, -np.inf
                for n_estimators in n_estimators_grid:
                    for max_depth in max_depth_grid:
                        threshold, score = _tune_threshold_and_score(
                            y_tr_outer, fold_oof[(n_estimators, max_depth)], threshold_metric
                        )
                        if score > best_score:
                            best_score, best_params, best_threshold = score, (n_estimators, max_depth), threshold
                best_params_by_fold[outer_key] = best_params
                best_threshold_by_fold[outer_key] = best_threshold

        phase2_results = parallel(
            delayed(_rf_outer_final_task)(
                outer_key,
                X,
                y,
                train_idx,
                test_idx,
                best_params_by_fold[outer_key],
                best_threshold_by_fold[outer_key],
                fold_random_state,
                ml_cfg,
                repeat_index,
                fold_index,
                feature_names,
                include_predictions,
                include_importance,
                workers > 1,
            )
            for outer_key, (repeat_index, fold_index, train_idx, test_idx, fold_random_state) in enumerate(fold_specs)
        )
        for outer_key, pred_tuple, imp_frame in phase2_results:
            if pred_tuple is not None:
                preds_by_fold[outer_key] = pred_tuple
            if imp_frame is not None:
                imp_by_fold[outer_key] = imp_frame
            if fold_callback is not None:
                fold_callback()

    return preds_by_fold, imp_by_fold


def _fit_outer_fold_prediction(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    ml_cfg: dict[str, Any],
    n_inner_splits: int,
    threshold_metric: str,
    random_state: int,
    estimator_n_jobs: int,
    outer_scaled_fold: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one outer fold and return its held-out predictions."""
    if outer_scaled_fold is not None and model_name in ("pls_da", "l1_logistic"):
        X_outer_tr_scaled, X_outer_te_scaled = outer_scaled_fold
        fit = _fit_tuned_model(
            model_name,
            X[train_idx],
            y[train_idx],
            ml_cfg=ml_cfg,
            n_inner_splits=n_inner_splits,
            threshold_metric=threshold_metric,
            random_state=random_state,
            n_jobs=estimator_n_jobs,
            outer_scaled_train=X_outer_tr_scaled,
        )
        scores = fit.predict_proba(X_outer_te_scaled)
        return test_idx, (scores >= fit.threshold).astype(int), scores

    fit = _fit_tuned_model(
        model_name,
        X[train_idx],
        y[train_idx],
        ml_cfg=ml_cfg,
        n_inner_splits=n_inner_splits,
        threshold_metric=threshold_metric,
        random_state=random_state,
        n_jobs=estimator_n_jobs,
    )
    scores = fit.predict_proba(X[test_idx])
    return test_idx, (scores >= fit.threshold).astype(int), scores


def _cv_predict(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    outer_cv: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    ml_cfg: dict[str, Any],
    n_inner_splits: int,
    threshold_metric: str,
    random_state: int,
    n_jobs: int,
    fold_callback: Callable[[], None] | None = None,
    outer_scaled_folds: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit+predict across one set of outer folds. Returns (y_pred, y_score) out-of-fold arrays."""
    y_pred = np.zeros_like(y, dtype=int)
    y_score = np.zeros_like(y, dtype=float)
    if model_name == "random_forest" and len(outer_cv) > 0:
        n_rf_tasks = len(outer_cv) * max(1, len(_rf_max_depth_grid(ml_cfg)) * n_inner_splits)
        if _resolve_parallel_workers(n_rf_tasks, n_jobs) > 1:
            fold_specs = [
                (0, fold_index, train_idx, test_idx, random_state)
                for fold_index, (train_idx, test_idx) in enumerate(outer_cv)
            ]
            preds_by_fold, _ = _run_rf_parallel_folds(
                X,
                y,
                fold_specs,
                ml_cfg=ml_cfg,
                n_inner_splits=n_inner_splits,
                threshold_metric=threshold_metric,
                n_jobs=n_jobs,
                include_predictions=True,
                include_importance=False,
                fold_callback=fold_callback,
            )
            for fold_index in range(len(fold_specs)):
                test_idx, fold_pred, fold_score = preds_by_fold[fold_index]
                y_pred[test_idx] = fold_pred
                y_score[test_idx] = fold_score
            return y_pred, y_score

    for fold_index, (train_idx, test_idx) in enumerate(outer_cv):
        scaled_fold = outer_scaled_folds[fold_index] if outer_scaled_folds is not None else None
        test_idx, fold_pred, fold_score = _fit_outer_fold_prediction(
            model_name,
            X,
            y,
            train_idx,
            test_idx,
            ml_cfg,
            n_inner_splits,
            threshold_metric,
            random_state,
            1 if model_name == "random_forest" else n_jobs,
            outer_scaled_fold=scaled_fold,
        )
        y_score[test_idx] = fold_score
        y_pred[test_idx] = fold_pred
        if fold_callback is not None:
            fold_callback()
    return y_pred, y_score


def _locked_joint_stratified_split_plan(
    y: np.ndarray,
    strata: np.ndarray,
    *,
    n_outer_splits: int,
    n_outer_repeats: int,
    random_state: int,
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Create one deterministic outer-fold plan preserving every class-sex stratum."""
    sex = np.asarray(strata, dtype=object)
    if sex.shape != y.shape or pd.isna(sex).any() or np.any(np.char.strip(sex.astype(str)) == ""):
        raise ValueError("Sex-stratified evaluation requires one non-missing sex value per subject")
    joint_labels = np.char.add(np.char.add(y.astype(str), "__"), sex.astype(str))
    counts = pd.Series(joint_labels).value_counts()
    if len(counts) < 2 or int(counts.min()) < n_outer_splits:
        raise ValueError(
            "Cannot create the locked joint class-sex outer folds: each observed class-sex stratum "
            f"must contain at least {n_outer_splits} subjects; observed counts={counts.to_dict()}"
        )
    indices = np.arange(len(y))
    return [
        list(
            StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=random_state + repeat).split(
                indices, joint_labels
            )
        )
        for repeat in range(n_outer_repeats)
    ]


def _run_repeated_nested_cv(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    ml_cfg: dict[str, Any],
    n_jobs: int,
    outer_split_plan: Sequence[Sequence[tuple[np.ndarray, np.ndarray]]] | None = None,
    fold_callback: Callable[[], None] | None = None,
) -> pd.DataFrame:
    """Run RepeatedStratifiedKFold nested CV; return one row per (repeat, subject_index)."""
    n_outer_splits = _n_outer_splits(ml_cfg)
    n_outer_repeats = int(ml_cfg.get("n_outer_repeats", 10))
    n_inner_splits = _n_inner_splits(ml_cfg)
    threshold_metric = _threshold_metric(ml_cfg)
    random_state = _random_state(ml_cfg)

    if outer_split_plan is None:
        n_splits = min(n_outer_splits, int(np.bincount(y).min()))
        repeated_cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_outer_repeats, random_state=random_state)
        all_splits = list(repeated_cv.split(X, y))
        split_plan = [all_splits[repeat * n_splits : (repeat + 1) * n_splits] for repeat in range(n_outer_repeats)]
    else:
        split_plan = [list(repeat_splits) for repeat_splits in outer_split_plan]
        if len(split_plan) != n_outer_repeats:
            raise ValueError("Locked outer split plan repeat count does not match n_outer_repeats")

    fold_specs = [
        (repeat_index, fold_index, train_idx, test_idx, random_state + repeat_index)
        for repeat_index, repeat_splits in enumerate(split_plan)
        for fold_index, (train_idx, test_idx) in enumerate(repeat_splits)
    ]

    if model_name == "random_forest" and fold_specs:
        n_rf_tasks = len(fold_specs) * max(1, len(_rf_max_depth_grid(ml_cfg)) * n_inner_splits)
        if _resolve_parallel_workers(n_rf_tasks, n_jobs) > 1:
            preds_by_fold, _ = _run_rf_parallel_folds(
                X,
                y,
                fold_specs,
                ml_cfg=ml_cfg,
                n_inner_splits=n_inner_splits,
                threshold_metric=threshold_metric,
                n_jobs=n_jobs,
                include_predictions=True,
                include_importance=False,
                fold_callback=fold_callback,
            )
            pred_by_repeat = {repeat_index: np.zeros_like(y, dtype=int) for repeat_index in range(len(split_plan))}
            score_by_repeat = {repeat_index: np.zeros_like(y, dtype=float) for repeat_index in range(len(split_plan))}
            for outer_key, (repeat_index, _fold_index, _train_idx, _test_idx, _rs) in enumerate(fold_specs):
                test_idx, fold_pred, fold_score = preds_by_fold[outer_key]
                pred_by_repeat[repeat_index][test_idx] = fold_pred
                score_by_repeat[repeat_index][test_idx] = fold_score
            rows: list[dict[str, object]] = []
            for repeat_index in range(len(split_plan)):
                y_pred = pred_by_repeat[repeat_index]
                y_score = score_by_repeat[repeat_index]
                rows.extend(
                    {
                        "repeat": repeat_index,
                        "subject_index": subject_index,
                        "y_true": int(y[subject_index]),
                        "y_pred": int(y_pred[subject_index]),
                        "y_score": float(y_score[subject_index]),
                    }
                    for subject_index in range(len(y))
                )
            return pd.DataFrame(rows)

    rows = []
    if _resolve_parallel_workers(len(fold_specs), n_jobs) > 1:
        calls = [
            (
                outer_key,
                (
                    model_name,
                    X,
                    y,
                    train_idx,
                    test_idx,
                    ml_cfg,
                    n_inner_splits,
                    threshold_metric,
                    fold_random_state,
                    1,
                ),
            )
            for outer_key, (_repeat_index, _fold_index, train_idx, test_idx, fold_random_state) in enumerate(fold_specs)
        ]
        pred_by_repeat = {repeat_index: np.zeros_like(y, dtype=int) for repeat_index in range(len(split_plan))}
        score_by_repeat = {repeat_index: np.zeros_like(y, dtype=float) for repeat_index in range(len(split_plan))}

        def on_result(outer_key: int, result: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
            repeat_index = fold_specs[outer_key][0]
            test_idx, fold_pred, fold_score = result
            pred_by_repeat[repeat_index][test_idx] = fold_pred
            score_by_repeat[repeat_index][test_idx] = fold_score
            if fold_callback is not None:
                fold_callback()

        _stream_parallel(n_jobs, _fit_outer_fold_prediction, calls, on_result)
        for repeat_index in range(len(split_plan)):
            y_pred = pred_by_repeat[repeat_index]
            y_score = score_by_repeat[repeat_index]
            rows.extend(
                {
                    "repeat": repeat_index,
                    "subject_index": subject_index,
                    "y_true": int(y[subject_index]),
                    "y_pred": int(y_pred[subject_index]),
                    "y_score": float(y_score[subject_index]),
                }
                for subject_index in range(len(y))
            )
        return pd.DataFrame(rows)

    for repeat_index, repeat_splits in enumerate(split_plan):
        y_pred, y_score = _cv_predict(
            model_name,
            X,
            y,
            repeat_splits,
            ml_cfg=ml_cfg,
            n_inner_splits=n_inner_splits,
            threshold_metric=threshold_metric,
            random_state=random_state + repeat_index,
            n_jobs=n_jobs,
            fold_callback=fold_callback,
        )
        rows.extend(
            {
                "repeat": repeat_index,
                "subject_index": subject_index,
                "y_true": int(y[subject_index]),
                "y_pred": int(y_pred[subject_index]),
                "y_score": float(y_score[subject_index]),
            }
            for subject_index in range(len(y))
        )

    return pd.DataFrame(rows)


def _fixed_plan_balanced_accuracy(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    ml_cfg: dict[str, Any],
    outer_cv: Sequence[tuple[np.ndarray, np.ndarray]],
    outer_scaled_folds: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
) -> float:
    """Score one fixed locked outer plan for observed and permuted labels alike."""
    y_pred, _y_score = _cv_predict(
        model_name,
        X,
        y,
        outer_cv,
        ml_cfg=ml_cfg,
        n_inner_splits=_n_inner_splits(ml_cfg),
        threshold_metric=_threshold_metric(ml_cfg),
        random_state=_random_state(ml_cfg),
        n_jobs=1,
        outer_scaled_folds=outer_scaled_folds,
    )
    return float(balanced_accuracy_score(y, y_pred))


def _repeat_zero_balanced_accuracy(oof_predictions: pd.DataFrame) -> float:
    """Read the permutation observed score off repeat 0 of the already-fitted nested-CV pass.

    `_run_repeated_nested_cv`'s repeat 0 uses `random_state=_random_state(ml_cfg)` on the
    same locked `outer_split_plan[0]` fold split that `_fixed_plan_balanced_accuracy` would
    otherwise refit from scratch — identical seed, identical folds, identical predictions.
    """
    repeat_zero = oof_predictions.loc[oof_predictions["repeat"] == 0]
    return float(balanced_accuracy_score(repeat_zero["y_true"], repeat_zero["y_pred"]))


# ---------------------------------------------------------------------------
# Permutation tests
# ---------------------------------------------------------------------------


def _permute_labels(y: np.ndarray, rng: np.random.Generator, strata: np.ndarray | None) -> np.ndarray:
    """Permute y unrestricted, or within each stratum (preserving stratum-label association)."""
    if strata is None:
        return rng.permutation(y)
    y_perm = y.copy()
    for stratum_value in np.unique(strata):
        mask = strata == stratum_value
        y_perm[mask] = rng.permutation(y[mask])
    return y_perm


def _permuted_score(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    strata: np.ndarray | None,
    ml_cfg: dict[str, Any],
    seed: int,
    outer_split_plan: Sequence[Sequence[tuple[np.ndarray, np.ndarray]]],
    outer_scaled_folds: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
) -> float:
    rng = np.random.default_rng(seed)
    y_perm = _permute_labels(y, rng, strata)
    if len(np.unique(y_perm)) < 2:
        return np.nan
    if outer_scaled_folds is not None:
        return _fixed_plan_balanced_accuracy(
            model_name,
            X,
            y_perm,
            ml_cfg=ml_cfg,
            outer_cv=outer_split_plan[0],
            outer_scaled_folds=outer_scaled_folds,
        )
    return _fixed_plan_balanced_accuracy(model_name, X, y_perm, ml_cfg=ml_cfg, outer_cv=outer_split_plan[0])


def _resampling_seeds(random_state: int, count: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    return rng.integers(0, np.iinfo(np.int32).max, size=count, dtype=np.int64)


def _load_checkpoint_rows(path: Path | None, *, resume: bool) -> pd.DataFrame:
    if not resume or path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _run_seeded_call(
    seed: int,
    func: Callable[..., Any],
    args: tuple[Any, ...],
    single_thread_blas: bool = False,
) -> tuple[int, Any]:
    """Return a draw's seed with its result so unordered workers remain traceable."""
    if single_thread_blas:
        with _single_thread_blas():
            return seed, func(*args)
    return seed, func(*args)


def _stream_parallel(
    n_jobs: int,
    func: Callable[..., Any],
    calls: Sequence[tuple[int, tuple[Any, ...]]],
    on_result: Callable[[int, Any], None],
) -> None:
    """Yield completed deterministic jobs to parent process for progress/checkpointing."""
    workers = _resolve_parallel_workers(len(calls), n_jobs)
    if workers <= 1:
        for seed, args in calls:
            on_result(seed, func(*args))
        return

    parallel = Parallel(n_jobs=workers, return_as="generator_unordered")
    results = parallel(delayed(_run_seeded_call)(seed, func, args, True) for seed, args in calls)
    for seed, result in results:
        on_result(seed, result)


def _stream_parallel_chunked(
    n_jobs: int,
    func: Callable[..., Any],
    calls: Sequence[tuple[int, tuple[Any, ...]]],
    on_result: Callable[[int, Any], None],
    *,
    chunk_size: int,
    should_continue: Callable[[], bool],
) -> None:
    """Run `calls` in fixed-size batches, checking `should_continue` between batches.

    A single `Parallel(return_as="generator_unordered")` cannot be interrupted
    cleanly mid-stream: breaking out of the generator early leaves in-flight
    futures whose callbacks fire after the executor has been torn down,
    raising `ShutdownExecutorError` on every cancelled task. Submitting in
    bounded chunks and only starting the next chunk after a stopping check
    avoids that — no futures are ever cancelled, each chunk always completes.
    """
    workers = _resolve_parallel_workers(len(calls), n_jobs)
    offset = 0
    if workers <= 1:
        while offset < len(calls) and should_continue():
            chunk = calls[offset : offset + chunk_size]
            offset += chunk_size
            for seed, args in chunk:
                on_result(seed, func(*args))
        return

    with Parallel(n_jobs=workers, return_as="list") as parallel:
        while offset < len(calls) and should_continue():
            chunk = calls[offset : offset + chunk_size]
            offset += chunk_size
            if len(chunk) <= 1:
                for seed, args in chunk:
                    on_result(seed, func(*args))
                continue
            results = parallel(delayed(_run_seeded_call)(seed, func, args, True) for seed, args in chunk)
            for seed, result in results:
                on_result(seed, result)


def run_permutation_test(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    observed_balanced_accuracy: float,
    strata: np.ndarray | None,
    ml_cfg: dict[str, Any],
    n_jobs: int,
    random_state: int,
    outer_split_plan: Sequence[Sequence[tuple[np.ndarray, np.ndarray]]],
    checkpoint_path: Path | None = None,
    resume: bool = False,
    progress: Progress | None = None,
    progress_task_id: int | None = None,
) -> tuple[float, np.ndarray]:
    """Full-pipeline label-permutation test. Returns (p_value, null_distribution).

    p = (1 + #{reference_statistic >= observed}) / (n_drawn + 1). `strata` is None for the
    unrestricted reference distribution, or an array of per-subject sex labels for the
    sex-stratified reference distribution (§0.1): permuting within sex strata preserves the
    real sex-class association, so a model that only reads sex will score no
    better under this reference distribution than under the unpermuted data.

    Draws stop early once the Monte Carlo p-value's Clopper-Pearson interval
    clears `permutation_alpha` at `permutation_confidence` (Besag & Clifford
    1991 curtailment) — see `src.statistics.sequential_permutation`. Seeds
    are still pre-generated up to `permutation_max` (defaults to the legacy
    `n_permutations` value) so a resumed or fully-ambiguous run always has a
    deterministic seed to draw next; only the number of seeds actually
    *evaluated* is reduced. `permutation_min` draws always run first,
    preserving Monte Carlo p-value validity (the observed statistic must
    always be counted alongside a real, pre-committed sample of the reference distribution).
    """
    if not np.isfinite(observed_balanced_accuracy):
        raise ValueError("Observed permutation statistic must be finite")
    stopping = resolve_sequential_stopping_config(ml_cfg, default_max=_n_permutations(ml_cfg))
    seeds = _resampling_seeds(random_state, stopping.permutation_max)
    existing = _load_checkpoint_rows(checkpoint_path, resume=resume)
    score_by_seed = {
        int(row.seed): float(row.null_balanced_accuracy)
        for row in existing.itertuples(index=False)
        if hasattr(row, "seed") and hasattr(row, "null_balanced_accuracy")
    }
    if not all(np.isfinite(score) for score in score_by_seed.values()):
        raise ValueError("Permutation checkpoint contains a non-finite score")

    outer_cv = outer_split_plan[0] if outer_split_plan else ()
    perm_plan = [outer_cv] if outer_split_plan else [[]]
    outer_scaled_folds: list[tuple[np.ndarray, np.ndarray]] | None = None
    if model_name in ("pls_da", "l1_logistic") and outer_cv:
        outer_scaled_folds = _scale_outer_folds(X, outer_cv)

    pending = [
        (
            int(seed),
            (model_name, X, y, strata, ml_cfg, int(seed), perm_plan, outer_scaled_folds)
            if outer_scaled_folds is not None
            else (model_name, X, y, strata, ml_cfg, int(seed), perm_plan),
        )
        for seed in seeds
        if int(seed) not in score_by_seed
    ]
    if progress is not None and progress_task_id is not None:
        progress.update(progress_task_id, completed=len(score_by_seed), total=stopping.permutation_max)

    def on_result(seed: int, score: float) -> None:
        if not np.isfinite(score):
            raise ValueError(f"Permutation score must be finite (seed={seed})")
        score_by_seed[seed] = float(score)
        if checkpoint_path is not None and len(score_by_seed) % _CHECKPOINT_FLUSH_EVERY == 0:
            write_parquet_atomic(
                pd.DataFrame({"seed": list(score_by_seed), "null_balanced_accuracy": list(score_by_seed.values())}),
                checkpoint_path,
                index=False,
            )
        if progress is not None and progress_task_id is not None:
            progress.advance(progress_task_id)

    def should_continue() -> bool:
        drawn = len(score_by_seed)
        successes = sum(1 for score in score_by_seed.values() if score >= observed_balanced_accuracy)
        return not should_stop(successes, drawn, stopping)

    try:
        _stream_parallel_chunked(
            n_jobs,
            _permuted_score,
            pending,
            on_result,
            chunk_size=stopping.check_interval,
            should_continue=should_continue,
        )
    finally:
        if checkpoint_path is not None:
            write_parquet_atomic(
                pd.DataFrame({"seed": list(score_by_seed), "null_balanced_accuracy": list(score_by_seed.values())}),
                checkpoint_path,
                index=False,
            )

    null_arr = np.asarray([score_by_seed[int(seed)] for seed in seeds if int(seed) in score_by_seed], dtype=float)
    if len(null_arr) == 0:
        return np.nan, null_arr
    p_value = float((1 + np.sum(null_arr >= observed_balanced_accuracy)) / (len(null_arr) + 1))
    return p_value, null_arr


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
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
    per_repeat = [
        _classification_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy(), group["y_score"].to_numpy())
        for _repeat, group in oof_predictions.groupby("repeat")
    ]
    repeat_df = pd.DataFrame(per_repeat)
    means = {f"{metric}_mean": float(repeat_df[metric].mean()) for metric in repeat_df.columns}
    stds = {
        f"{metric}_std": float(repeat_df[metric].std(ddof=1)) if len(repeat_df) > 1 else 0.0
        for metric in repeat_df.columns
    }
    return means | stds


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def _bootstrap_metric_sample(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    ml_cfg: dict[str, Any],
    seed: int,
) -> dict[str, float] | None:
    """Fit an in-bag draw with grouped inner tuning, then score unseen participants."""
    rng = np.random.default_rng(seed)
    n_samples = X.shape[0]
    in_bag_idx = rng.integers(0, n_samples, size=n_samples)
    out_of_bag_idx = np.setdiff1d(np.arange(n_samples), in_bag_idx, assume_unique=False)

    X_in, y_in = X[in_bag_idx], y[in_bag_idx]
    if len(np.unique(y_in)) < 2 or len(out_of_bag_idx) == 0:
        return None
    X_out, y_out = X[out_of_bag_idx], y[out_of_bag_idx]
    if len(np.unique(y_out)) < 2:
        return None

    n_inner_splits = _n_inner_splits(ml_cfg)
    threshold_metric = _threshold_metric(ml_cfg)
    inner_folds = _inner_cv_folds(X_in, y_in, n_inner_splits, seed, groups=in_bag_idx)
    if inner_folds is None:
        return None
    fit = _fit_tuned_model(
        model_name,
        X_in,
        y_in,
        ml_cfg=ml_cfg,
        n_inner_splits=n_inner_splits,
        threshold_metric=threshold_metric,
        random_state=seed,
        n_jobs=1,
        groups=in_bag_idx,
        inner_folds=inner_folds,
    )
    scores = fit.predict_proba(X_out)
    y_pred = (scores >= fit.threshold).astype(int)
    return _classification_metrics(y_out, y_pred, scores)


def run_bootstrap_ci(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    ml_cfg: dict[str, Any],
    n_jobs: int,
    random_state: int,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    progress: Progress | None = None,
    progress_task_id: int | None = None,
) -> dict[str, float]:
    """Bootstrap 95% percentile CIs on headline metrics, via out-of-bag scoring.

    Each draw fits on an in-bag resample and scores only on subjects never
    selected into it (`_bootstrap_metric_sample`) — see that function's
    docstring for why naive resample-then-CV leaks. Reports the 2.5th/97.5th
    percentile of each metric across draws.
    """
    n_bootstrap = int(ml_cfg.get("n_bootstrap", 1000))
    seeds = _resampling_seeds(random_state, n_bootstrap)
    existing = _load_checkpoint_rows(checkpoint_path, resume=resume)
    sample_by_seed: dict[int, dict[str, float] | None] = {}
    if not existing.empty and "seed" in existing:
        for row in existing.to_dict("records"):
            seed = int(row.pop("seed"))
            valid = bool(row.pop("valid", False))
            sample_by_seed[seed] = (
                {key: float(value) for key, value in row.items() if pd.notna(value)} if valid else None
            )
    pending = [(int(seed), (model_name, X, y, ml_cfg, int(seed))) for seed in seeds if int(seed) not in sample_by_seed]
    if progress is not None and progress_task_id is not None:
        progress.update(progress_task_id, completed=len(sample_by_seed), total=n_bootstrap)

    def checkpoint_frame() -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for seed, sample in sample_by_seed.items():
            rows.append({"seed": seed, "valid": sample is not None, **(sample or {})})
        return pd.DataFrame(rows)

    def on_result(seed: int, sample: dict[str, float] | None) -> None:
        sample_by_seed[seed] = sample
        if checkpoint_path is not None and len(sample_by_seed) % _CHECKPOINT_FLUSH_EVERY == 0:
            write_parquet_atomic(checkpoint_frame(), checkpoint_path, index=False)
        if progress is not None and progress_task_id is not None:
            progress.advance(progress_task_id)

    try:
        _stream_parallel(n_jobs, _bootstrap_metric_sample, pending, on_result)
    finally:
        if checkpoint_path is not None:
            write_parquet_atomic(checkpoint_frame(), checkpoint_path, index=False)

    valid_samples = [sample_by_seed[int(seed)] for seed in seeds if sample_by_seed[int(seed)] is not None]
    if not valid_samples:
        return {}

    sample_df = pd.DataFrame(valid_samples)
    ci: dict[str, float] = {"n_bootstrap_used": len(valid_samples)}
    for metric in sample_df.columns:
        finite = sample_df[metric].dropna().to_numpy()
        if len(finite) == 0:
            ci[f"{metric}_ci_low"] = np.nan
            ci[f"{metric}_ci_high"] = np.nan
            continue
        ci[f"{metric}_ci_low"] = float(np.percentile(finite, 2.5))
        ci[f"{metric}_ci_high"] = float(np.percentile(finite, 97.5))
    return ci


# ---------------------------------------------------------------------------
# Univariate band screen (descriptive context, not a model)
# ---------------------------------------------------------------------------


def run_univariate_band_screen(
    frame: pd.DataFrame, y: np.ndarray, valid_rows: np.ndarray, cfg_in: dict[str, Any]
) -> pd.DataFrame:
    """Mann-Whitney U per biochemical band (Control vs. Disease) + BH-FDR.

    Purely descriptive context for the multivariate models above — this is
    not itself a claim and is not gated on any model's significance.
    """
    all_columns = feature_columns(frame)
    band_frame = _band_panel(frame.loc[valid_rows], all_columns, cfg_in)
    y_valid = y[valid_rows]
    control_mask = y_valid == 0
    disease_mask = y_valid == 1

    rows: list[dict[str, object]] = []
    p_values: list[float] = []
    for band_name in band_frame.columns:
        values = band_frame[band_name].to_numpy(dtype=float)
        control_values = values[control_mask]
        disease_values = values[disease_mask]
        finite = np.isfinite(control_values).all() and np.isfinite(disease_values).all()
        if not finite or len(control_values) < 2 or len(disease_values) < 2:
            rows.append({"band": band_name, "status": "insufficient_data"})
            p_values.append(np.nan)
            continue
        statistic, p_value = stats.mannwhitneyu(disease_values, control_values, alternative="two-sided")
        n1, n2 = len(disease_values), len(control_values)
        rank_biserial = (2.0 * statistic / (n1 * n2)) - 1.0
        rows.append(
            {
                "band": band_name,
                "n_control": n2,
                "n_disease": n1,
                "median_control": float(np.median(control_values)),
                "median_disease": float(np.median(disease_values)),
                "u_statistic": float(statistic),
                "effect_size_rank_biserial": float(rank_biserial),
                "p_value": float(p_value),
                "status": "ok",
            }
        )
        p_values.append(float(p_value))

    result = pd.DataFrame(rows)
    q_values, significant = fdr_correction(np.asarray(p_values, dtype=float), alpha=0.05)
    result["fdr_q"] = q_values
    result["significant_fdr"] = significant
    return result


# ---------------------------------------------------------------------------


def _rf_permutation_importance(
    estimator: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_repeats: int,
    random_state: int,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute balanced-accuracy permutation importance matching sklearn RNG order.

    For fitted binary RandomForestClassifier estimators, exploits tree split
    sparsity by evaluating only trees whose split features include the permuted
    column across batched float32 test copies, falling back to exact forest
    accumulation on near-tie probability margins.
    """
    if (
        isinstance(estimator, RandomForestClassifier)
        and estimator.n_outputs_ == 1
        and estimator.n_classes_ == 2
        and np.array_equal(estimator.classes_, np.array([0, 1]))
        and len(X_test) > 0
        and np.isfinite(X_test).all()
    ):
        pos_mask = y_test == 1
        neg_mask = y_test == 0
        n_pos = int(pos_mask.sum())
        n_neg = int(neg_mask.sum())
        if n_pos > 0 and n_neg > 0:
            n_test, n_features = X_test.shape
            rs = check_random_state(random_state)
            seed_inner = int(rs.randint(np.iinfo(np.int32).max + 1))
            rs_col = check_random_state(seed_inner)
            shuffling_idx = np.arange(n_test)
            cur_perm = np.arange(n_test)
            perm_orders = np.empty((n_repeats, n_test), dtype=np.intp)
            for r in range(n_repeats):
                rs_col.shuffle(shuffling_idx)
                cur_perm = cur_perm[shuffling_idx]
                perm_orders[r] = cur_perm

            X_t32 = np.ascontiguousarray(X_test, dtype=np.float32)
            trees = estimator.estimators_
            n_trees = len(trees)
            base_tree_proba = np.array([est.predict_proba(X_t32, check_input=False) for est in trees])
            base_sum = base_tree_proba.sum(axis=0)
            base_pred = estimator.predict(X_t32)
            base_ba = 0.5 * (
                float((base_pred[pos_mask] == 1).sum()) / n_pos
                + float((base_pred[neg_mask] == 0).sum()) / n_neg
            )

            trees_by_feat: list[list[int]] = [[] for _ in range(n_features)]
            for t_idx, est in enumerate(trees):
                feats = est.tree_.feature
                for feat_idx in np.unique(feats[feats >= 0]):
                    trees_by_feat[int(feat_idx)].append(t_idx)

            X_perm_buf = np.tile(X_t32, (n_repeats, 1))
            scores = np.full((n_features, n_repeats), base_ba, dtype=np.float64)
            tie_tol = 1e-11 * n_trees
            for col_idx in range(n_features):
                t_list = trees_by_feat[col_idx]
                if not t_list:
                    continue
                col_vals = X_t32[:, col_idx]
                for r in range(n_repeats):
                    X_perm_buf[r * n_test : (r + 1) * n_test, col_idx] = col_vals[perm_orders[r]]
                p_sum = np.broadcast_to(base_sum, (n_repeats, n_test, 2)).copy()
                for t_idx in t_list:
                    p_new = trees[t_idx].predict_proba(X_perm_buf, check_input=False).reshape(n_repeats, n_test, 2)
                    p_sum += p_new - base_tree_proba[t_idx][None, :, :]
                if np.any(np.abs(p_sum[:, :, 0] - p_sum[:, :, 1]) <= tie_tol):
                    p_exact = estimator.predict_proba(X_perm_buf).reshape(n_repeats, n_test, 2)
                    preds = estimator.classes_.take(np.argmax(p_exact, axis=2), axis=0)
                else:
                    preds = estimator.classes_.take(np.argmax(p_sum, axis=2), axis=0)
                for r in range(n_repeats):
                    X_perm_buf[r * n_test : (r + 1) * n_test, col_idx] = col_vals
                scores[col_idx, :] = 0.5 * (
                    (preds[:, pos_mask] == 1).sum(axis=1) / n_pos
                    + (preds[:, neg_mask] == 0).sum(axis=1) / n_neg
                )

            importances = base_ba - scores
            return np.mean(importances, axis=1), np.std(importances, axis=1)

    result = permutation_importance(
        estimator,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
        scoring="balanced_accuracy",
    )
    return result.importances_mean, result.importances_std


def _outer_fold_importance_frame_from_fit(
    model_name: str,
    fit: _ModelFit,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    *,
    ml_cfg: dict[str, Any],
    repeat_index: int,
    fold_index: int,
    n_train: int,
    estimator_n_jobs: int = 1,
) -> pd.DataFrame:
    """Build one outer fold's importance DataFrame from an already-fitted model."""
    base = {
        "feature_name": feature_names,
        "repeat": repeat_index,
        "fold": fold_index,
        "n_train": n_train,
        "n_test": len(X_test),
        "estimation_scope": "outer_fold",
    }
    if model_name == "pls_da":
        return pd.DataFrame({**base, "importance": _calculate_vip(fit.estimator), "importance_type": "vip_outer_train"})
    if model_name == "l1_logistic":
        return pd.DataFrame(
            {
                **base,
                "importance": fit.estimator.coef_.ravel(),
                "importance_type": "l1_logistic_coefficient_outer_train",
            }
        )

    random_state = _random_state(ml_cfg)
    importances_mean, importances_std = _rf_permutation_importance(
        fit.estimator,
        X_test,
        y_test,
        n_repeats=int(ml_cfg.get("importance_permutation_repeats", 5)),
        random_state=random_state + repeat_index * 101 + fold_index,
        n_jobs=estimator_n_jobs,
    )
    return pd.DataFrame(
        {
            **base,
            "importance": importances_mean,
            "importance_std": importances_std,
            "importance_type": "permutation_importance_outer_test",
        }
    )


def _summarize_importance_folds(rows: Sequence[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate per-fold importance frames and compute outer-fold summary."""
    long = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if long.empty:
        return long, long
    summary = (
        long.groupby(["feature_name", "importance_type"], as_index=False)
        .agg(
            importance_mean=("importance", "mean"),
            importance_median=("importance", "median"),
            importance_std_across_folds=("importance", "std"),
            n_outer_folds=("importance", "count"),
        )
        .sort_values("importance_mean", key=np.abs, ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    summary["estimation_scope"] = "outer_fold_summary"
    return long, summary


def _outer_fold_importance_frame(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    ml_cfg: dict[str, Any],
    repeat_index: int,
    fold_index: int,
    estimator_n_jobs: int,
) -> pd.DataFrame:
    """Fit one outer fold and calculate leakage-free feature importance."""
    random_state = _random_state(ml_cfg)
    fit = _fit_tuned_model(
        model_name,
        X[train_idx],
        y[train_idx],
        ml_cfg=ml_cfg,
        n_inner_splits=_n_inner_splits(ml_cfg),
        threshold_metric=_threshold_metric(ml_cfg),
        random_state=random_state + repeat_index,
        n_jobs=estimator_n_jobs,
    )
    return _outer_fold_importance_frame_from_fit(
        model_name,
        fit,
        X[test_idx],
        y[test_idx],
        feature_names,
        ml_cfg=ml_cfg,
        repeat_index=repeat_index,
        fold_index=fold_index,
        n_train=len(train_idx),
        estimator_n_jobs=estimator_n_jobs,
    )


def _outer_fold_importance_task(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    ml_cfg: dict[str, Any],
    repeat_index: int,
    fold_index: int,
) -> pd.DataFrame:
    """Process-safe wrapper for one single-threaded outer-fold importance task."""
    return _outer_fold_importance_frame(
        model_name,
        X,
        y,
        feature_names,
        train_idx,
        test_idx,
        ml_cfg=ml_cfg,
        repeat_index=repeat_index,
        fold_index=fold_index,
        estimator_n_jobs=1,
    )


def _outer_fold_feature_importance(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    ml_cfg: dict[str, Any],
    n_jobs: int,
    outer_split_plan: Sequence[Sequence[tuple[np.ndarray, np.ndarray]]],
    fold_callback: Callable[[], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate importances on outer folds, never from a full-data refit.

    PLS VIP and L1 coefficients are calculated from each outer-training fit.
    Forest permutation importance is scored only on the corresponding held-out
    outer test fold.  The long output exposes stability across all folds.
    """
    rows: list[pd.DataFrame] = []
    fold_specs = [
        (repeat_index, fold_index, train_idx, test_idx)
        for repeat_index, repeat_splits in enumerate(outer_split_plan)
        for fold_index, (train_idx, test_idx) in enumerate(repeat_splits)
    ]

    if model_name == "random_forest" and fold_specs:
        n_inner_splits = _n_inner_splits(ml_cfg)
        n_rf_tasks = len(fold_specs) * max(1, len(_rf_max_depth_grid(ml_cfg)) * n_inner_splits)
        if _resolve_parallel_workers(n_rf_tasks, n_jobs) > 1:
            random_state = _random_state(ml_cfg)
            rf_specs = [
                (repeat_index, fold_index, train_idx, test_idx, random_state + repeat_index)
                for repeat_index, fold_index, train_idx, test_idx in fold_specs
            ]
            _, imp_by_fold = _run_rf_parallel_folds(
                X,
                y,
                rf_specs,
                ml_cfg=ml_cfg,
                n_inner_splits=n_inner_splits,
                threshold_metric=_threshold_metric(ml_cfg),
                n_jobs=n_jobs,
                feature_names=feature_names,
                include_predictions=False,
                include_importance=True,
                fold_callback=fold_callback,
            )
            rows = [imp_by_fold[outer_key] for outer_key in range(len(rf_specs))]
            return _summarize_importance_folds(rows)

    if _resolve_parallel_workers(len(fold_specs), n_jobs) > 1:
        calls = [
            (
                fold_key,
                (
                    model_name,
                    X,
                    y,
                    feature_names,
                    train_idx,
                    test_idx,
                    ml_cfg,
                    repeat_index,
                    fold_index,
                ),
            )
            for fold_key, (repeat_index, fold_index, train_idx, test_idx) in enumerate(fold_specs)
        ]
        frames_by_fold: dict[int, pd.DataFrame] = {}

        def on_result(fold_key: int, frame: pd.DataFrame) -> None:
            frames_by_fold[fold_key] = frame
            if fold_callback is not None:
                fold_callback()

        _stream_parallel(n_jobs, _outer_fold_importance_task, calls, on_result)
        rows = [frames_by_fold[fold_key] for fold_key, _args in calls]
    else:
        for repeat_index, fold_index, train_idx, test_idx in fold_specs:
            rows.append(
                _outer_fold_importance_frame(
                    model_name,
                    X,
                    y,
                    feature_names,
                    train_idx,
                    test_idx,
                    ml_cfg=ml_cfg,
                    repeat_index=repeat_index,
                    fold_index=fold_index,
                    estimator_n_jobs=1 if model_name == "random_forest" else n_jobs,
                )
            )
            if fold_callback is not None:
                fold_callback()
    return _summarize_importance_folds(rows)


# Sex-confound battery
# ---------------------------------------------------------------------------


def _sex_only_baseline(
    frame: pd.DataFrame, y: np.ndarray, valid_rows: np.ndarray,
    *, outer_split_plan: Sequence[Sequence[tuple[np.ndarray, np.ndarray]]] | None = None,
    random_state: int = 42,
) -> dict[str, object]:
    """Evaluate a sex-only model on the spectral models' held-out folds."""
    sex_column = _sex_column(frame)
    if sex_column is None:
        return {"status": "missing_sex_column"}
    recorded_sex = frame.loc[valid_rows, sex_column].astype("string").str.strip().str.lower()
    if not recorded_sex.isin(["male", "female"]).all():
        return {"status": "missing_sex_values", "sex_covariate_column": sex_column}
    sex_values = recorded_sex.to_numpy()
    y_valid = y[valid_rows]
    X_sex = (sex_values == "male").astype(float).reshape(-1, 1)
    if outer_split_plan is None:
        n_splits = min(5, int(np.bincount(y_valid).min()))
        if n_splits < 2:
            return {"status": "insufficient_class_size_for_cv"}
        outer_split_plan = [list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(X_sex, y_valid))]

    repeat_metrics = []
    for repeat_index, folds in enumerate(outer_split_plan):
        y_pred = np.zeros_like(y_valid, dtype=int)
        y_score = np.zeros_like(y_valid, dtype=float)
        for train_idx, test_idx in folds:
            model = LogisticRegression(class_weight="balanced", random_state=random_state + repeat_index, max_iter=1000)
            fit_with_convergence_capture(model, X_sex[train_idx], y_valid[train_idx], context="sex-only baseline")
            train_scores = model.predict_proba(X_sex[train_idx])[:, 1]
            y_score[test_idx] = model.predict_proba(X_sex[test_idx])[:, 1]
            threshold = _tune_threshold(y_valid[train_idx], train_scores, "balanced_accuracy")
            y_pred[test_idx] = (y_score[test_idx] >= threshold).astype(int)
        repeat_metrics.append(_classification_metrics(y_valid, y_pred, y_score))
    metrics = pd.DataFrame(repeat_metrics).mean().to_dict()
    return {"status": "ok", "sex_only_cv_repeats": len(repeat_metrics), "sex_covariate_column": sex_column,
            **{f"sex_only_{key}": value for key, value in metrics.items()}}


def _sex_stratified_refit(
    model_name: str,
    frame: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    valid_rows: np.ndarray,
    *,
    ml_cfg: dict[str, Any],
    n_jobs: int,
) -> dict[str, object]:
    """Refit + evaluate the full nested-CV pipeline within the female-only subset.

    Female-only is used because it is the sex stratum with adequate, roughly
    balanced n in this cohort (45 disease / 40 control); the male stratum
    (52 disease / 10 control) is too imbalanced for inference and is reported
    descriptively only via n counts in cohort_summary.csv, not refit here.
    """
    sex_column = _sex_column(frame)
    if sex_column is None:
        return {"status": "missing_sex_column"}

    sex_values = frame.loc[valid_rows, sex_column].astype(str).to_numpy()
    female_mask = sex_values == "female"
    X_female, y_female = X[female_mask], y[female_mask]

    n_control = int((y_female == 0).sum())
    n_disease = int((y_female == 1).sum())
    if n_control < _MIN_SUBJECTS_PER_CLASS_STRATIFIED or n_disease < _MIN_SUBJECTS_PER_CLASS_STRATIFIED:
        return {"status": "insufficient_n", "n_female_control": n_control, "n_female_disease": n_disease}

    oof = _run_repeated_nested_cv(model_name, X_female, y_female, ml_cfg=ml_cfg, n_jobs=n_jobs)
    metrics = _aggregate_repeat_metrics(oof)
    return {
        "status": "ok",
        "n_female_control": n_control,
        "n_female_disease": n_disease,
        **{f"female_only_{k}": v for k, v in metrics.items()},
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _append_null_distribution_rows(
    rows: list[dict[str, object]],
    *,
    model_name: str,
    representation: str,
    null_type: str,
    random_state: int,
    null_scores: np.ndarray,
    n_permutations: int,
    n_splits: int,
) -> None:
    """Append deterministic permutation rows for a permutation run that may have stopped early.

    `n_permutations` is the configured ceiling, not necessarily the draw
    count: adaptive stopping (`src.statistics.sequential_permutation`) can
    resolve the Monte Carlo p-value in fewer draws. Seeds are generated
    deterministically from `random_state`, so the first `null_scores.size`
    seeds of that sequence are exactly the ones actually drawn.
    """
    if null_scores.size == 0:
        return
    if null_scores.size > n_permutations:
        raise RuntimeError(
            f"{null_type} permutation test returned {null_scores.size} scores; "
            f"exceeds configured ceiling {n_permutations}"
        )
    seeds = _resampling_seeds(random_state, n_permutations)[: null_scores.size]
    rows.extend(
        {
            "model": model_name,
            "representation": representation,
            "null_type": null_type,
            "seed": int(seed),
            "permutation_cv_splits": n_splits,
            "permutation_cv_repeats": 1,
            "null_balanced_accuracy": float(score),
        }
        for seed, score in zip(seeds, null_scores, strict=True)
    )


def _observed_outer_fold_task(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    ml_cfg: dict[str, Any],
    repeat_index: int,
    fold_index: int,
    n_inner_splits: int,
    threshold_metric: str,
    fold_random_state: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], pd.DataFrame]:
    """Fit one outer fold once and return both predictions and importance."""
    fit = _fit_tuned_model(
        model_name,
        X[train_idx],
        y[train_idx],
        ml_cfg=ml_cfg,
        n_inner_splits=n_inner_splits,
        threshold_metric=threshold_metric,
        random_state=fold_random_state,
        n_jobs=1,
    )
    scores = fit.predict_proba(X[test_idx])
    pred_tuple = (test_idx, (scores >= fit.threshold).astype(int), scores)
    imp_frame = _outer_fold_importance_frame_from_fit(
        model_name,
        fit,
        X[test_idx],
        y[test_idx],
        feature_names,
        ml_cfg=ml_cfg,
        repeat_index=repeat_index,
        fold_index=fold_index,
        n_train=len(train_idx),
        estimator_n_jobs=1,
    )
    return pred_tuple, imp_frame


def _run_repeated_nested_cv_with_importance(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    ml_cfg: dict[str, Any],
    n_jobs: int,
    outer_split_plan: Sequence[Sequence[tuple[np.ndarray, np.ndarray]]],
    fold_callback: Callable[[], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit each outer fold once to produce OOF predictions and feature importances."""
    n_outer_repeats = int(ml_cfg.get("n_outer_repeats", 10))
    n_inner_splits = _n_inner_splits(ml_cfg)
    threshold_metric = _threshold_metric(ml_cfg)
    random_state = _random_state(ml_cfg)
    split_plan = [list(repeat_splits) for repeat_splits in outer_split_plan]
    if len(split_plan) != n_outer_repeats:
        raise ValueError("Locked outer split plan repeat count does not match n_outer_repeats")

    fold_specs = [
        (repeat_index, fold_index, train_idx, test_idx, random_state + repeat_index)
        for repeat_index, repeat_splits in enumerate(split_plan)
        for fold_index, (train_idx, test_idx) in enumerate(repeat_splits)
    ]

    preds_by_fold: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    imp_by_fold: dict[int, pd.DataFrame] = {}

    if (
        model_name == "random_forest"
        and fold_specs
        and _resolve_parallel_workers(
            len(fold_specs) * max(1, len(_rf_max_depth_grid(ml_cfg)) * n_inner_splits), n_jobs
        )
        > 1
    ):
        preds_by_fold, imp_by_fold = _run_rf_parallel_folds(
            X,
            y,
            fold_specs,
            ml_cfg=ml_cfg,
            n_inner_splits=n_inner_splits,
            threshold_metric=threshold_metric,
            n_jobs=n_jobs,
            feature_names=feature_names,
            include_predictions=True,
            include_importance=True,
            fold_callback=fold_callback,
        )
    elif _resolve_parallel_workers(len(fold_specs), n_jobs) > 1:
        calls = [
            (
                outer_key,
                (
                    model_name,
                    X,
                    y,
                    feature_names,
                    train_idx,
                    test_idx,
                    ml_cfg,
                    repeat_index,
                    fold_index,
                    n_inner_splits,
                    threshold_metric,
                    fold_random_state,
                ),
            )
            for outer_key, (repeat_index, fold_index, train_idx, test_idx, fold_random_state) in enumerate(fold_specs)
        ]

        def on_result(
            outer_key: int,
            result: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], pd.DataFrame],
        ) -> None:
            pred_tuple, imp_frame = result
            preds_by_fold[outer_key] = pred_tuple
            imp_by_fold[outer_key] = imp_frame
            if fold_callback is not None:
                fold_callback()

        _stream_parallel(n_jobs, _observed_outer_fold_task, calls, on_result)
    else:
        for outer_key, (repeat_index, fold_index, train_idx, test_idx, fold_random_state) in enumerate(fold_specs):
            pred_tuple, imp_frame = _observed_outer_fold_task(
                model_name,
                X,
                y,
                feature_names,
                train_idx,
                test_idx,
                ml_cfg,
                repeat_index,
                fold_index,
                n_inner_splits,
                threshold_metric,
                fold_random_state,
            )
            preds_by_fold[outer_key] = pred_tuple
            imp_by_fold[outer_key] = imp_frame
            if fold_callback is not None:
                fold_callback()

    pred_by_repeat = {repeat_index: np.zeros_like(y, dtype=int) for repeat_index in range(len(split_plan))}
    score_by_repeat = {repeat_index: np.zeros_like(y, dtype=float) for repeat_index in range(len(split_plan))}
    for outer_key, (repeat_index, _fold_index, _train_idx, _test_idx, _rs) in enumerate(fold_specs):
        test_idx, fold_pred, fold_score = preds_by_fold[outer_key]
        pred_by_repeat[repeat_index][test_idx] = fold_pred
        score_by_repeat[repeat_index][test_idx] = fold_score

    oof_rows: list[dict[str, object]] = []
    for repeat_index in range(len(split_plan)):
        y_pred = pred_by_repeat[repeat_index]
        y_score = score_by_repeat[repeat_index]
        oof_rows.extend(
            {
                "repeat": repeat_index,
                "subject_index": subject_index,
                "y_true": int(y[subject_index]),
                "y_pred": int(y_pred[subject_index]),
                "y_score": float(y_score[subject_index]),
            }
            for subject_index in range(len(y))
        )
    oof = pd.DataFrame(oof_rows)
    imp_rows = [imp_by_fold[outer_key] for outer_key in range(len(fold_specs))]
    importance_folds, importance_df = _summarize_importance_folds(imp_rows)
    return oof, importance_folds, importance_df


def _run_observed_model_phase(
    *,
    model_name: str,
    representation: str,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    feature_names: list[str],
    ml_cfg: dict[str, Any],
    n_jobs: int,
    outer_split_plan: Sequence[Sequence[tuple[np.ndarray, np.ndarray]]],
    checkpoints: _CheckpointStore,
    model_dir: Path,
    outputs: dict[str, Path],
    phase: Callable[..., AbstractContextManager[None]],
    progress: Progress,
) -> tuple[pd.DataFrame, dict[str, float]]:
    cv_path = model_dir / f"{model_name}_{representation}_cv_predictions.csv"
    importance_folds_path = model_dir / f"{model_name}_{representation}_feature_importance_outer_folds.csv"
    importance_path = model_dir / f"{model_name}_{representation}_feature_importance.csv"
    observed_artifacts = (cv_path, importance_folds_path, importance_path)
    observed_complete = checkpoints.stage_complete(model_name, representation, "observed", artifacts=observed_artifacts)
    with phase(f"{model_name}_{representation}", f"{model_name} ({representation}) nested CV"):
        if (
            observed_complete
            and cv_path.exists()
            and importance_folds_path.exists()
            and importance_path.exists()
        ):
            oof = pd.read_csv(cv_path)
            metrics = _aggregate_repeat_metrics(oof)
            logger.info("Reused checkpointed observed ML outputs for %s (%s)", model_name, representation)
        else:
            with _progress_task(
                progress,
                f"{model_name} ({representation}) nested & importance folds",
                sum(len(splits) for splits in outer_split_plan),
            ) as fold_task:
                oof, importance_folds, importance_df = _run_repeated_nested_cv_with_importance(
                    model_name,
                    X_valid,
                    y_valid,
                    feature_names,
                    ml_cfg=ml_cfg,
                    n_jobs=n_jobs,
                    outer_split_plan=outer_split_plan,
                    fold_callback=lambda: progress.advance(fold_task),
                )
            outputs[f"{model_name}_{representation}_cv_predictions"] = write_csv_atomic(
                oof, cv_path, index=False
            )
            metrics = _aggregate_repeat_metrics(oof)
            outputs[f"{model_name}_{representation}_feature_importance_folds"] = write_csv_atomic(
                importance_folds, importance_folds_path, index=False
            )
            outputs[f"{model_name}_{representation}_feature_importance"] = write_csv_atomic(
                importance_df, importance_path, index=False
            )
            checkpoints.mark_stage_complete(model_name, representation, "observed", artifacts=observed_artifacts)
    return oof, metrics


def _run_model_permutation_phase(
    *,
    model_name: str,
    representation: str,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    strata: np.ndarray | None,
    permutation_observed_score: float,
    ml_cfg: dict[str, Any],
    n_jobs: int,
    random_state: int,
    outer_split_plan: Sequence[Sequence[tuple[np.ndarray, np.ndarray]]],
    checkpoints: _CheckpointStore,
    resume: bool,
    null_distribution_rows: list[dict[str, object]],
    phase: Callable[..., AbstractContextManager[None]],
    progress: Progress,
) -> tuple[float, float, int, int]:
    with _progress_task(
        progress,
        f"{model_name} ({representation}) unrestricted permutations",
        _n_permutations(ml_cfg),
    ) as unrestricted_task:
        with phase(f"{model_name}_{representation}_permutation", f"{model_name} ({representation}) permutation test (unrestricted)"):
            p_unrestricted, null_unrestricted = run_permutation_test(
                model_name,
                X_valid,
                y_valid,
                observed_balanced_accuracy=permutation_observed_score,
                strata=None,
                ml_cfg=ml_cfg,
                n_jobs=n_jobs,
                random_state=random_state,
                outer_split_plan=outer_split_plan,
                checkpoint_path=checkpoints.resampling_path(
                    "permutation", model_name, representation, "unrestricted"
                ),
                resume=resume,
                progress=progress,
                progress_task_id=unrestricted_task,
            )

    p_sex_stratified, null_sex_stratified = np.nan, np.array([])
    if strata is not None:
        with _progress_task(
            progress,
            f"{model_name} ({representation}) sex-stratified permutations",
            _n_permutations(ml_cfg),
        ) as sex_task:
            with phase(f"{model_name}_{representation}_permutation_sex", f"{model_name} ({representation}) permutation test (sex-stratified)"):
                p_sex_stratified, null_sex_stratified = run_permutation_test(
                    model_name,
                    X_valid,
                    y_valid,
                    observed_balanced_accuracy=permutation_observed_score,
                    strata=strata,
                    ml_cfg=ml_cfg,
                    n_jobs=n_jobs,
                    random_state=random_state + 1,
                    outer_split_plan=outer_split_plan,
                    checkpoint_path=checkpoints.resampling_path(
                        "permutation", model_name, representation, "sex_stratified"
                    ),
                    resume=resume,
                    progress=progress,
                    progress_task_id=sex_task,
                )

    for null_scores, null_type, null_seed in (
        (null_unrestricted, "unrestricted", random_state),
        (null_sex_stratified, "sex_stratified", random_state + 1),
    ):
        _append_null_distribution_rows(
            null_distribution_rows,
            model_name=model_name,
            representation=representation,
            null_type=null_type,
            random_state=null_seed,
            null_scores=null_scores,
            n_permutations=_n_permutations(ml_cfg),
            n_splits=len(outer_split_plan[0]),
        )
    return p_unrestricted, p_sex_stratified, len(null_unrestricted), len(null_sex_stratified)


def _write_model_comparison(
    comparison_rows: list[dict[str, object]],
    model_dir: Path,
) -> Path:
    comparison_df = pd.DataFrame(comparison_rows)
    primary_mask = comparison_df["is_primary_model"].astype(bool)
    family_p = comparison_df["permutation_p_value_sex_stratified"].to_numpy(dtype=float)
    if not np.isfinite(family_p).all():
        raise RuntimeError(
            "Sex-stratified permutation inference failed; refusing to substitute an unrestricted reference distribution"
        )
    fdr_q, fdr_significant = fdr_correction(family_p, alpha=0.05)
    comparison_df["permutation_p_fdr_q"] = fdr_q
    comparison_df["permutation_fdr_significant"] = fdr_significant
    comparison_df.insert(0, "positive_class", POSITIVE_CLASS_LABEL)
    path = write_csv_atomic(
        comparison_df, model_dir / "model_comparison.csv", index=False
    )
    if primary_mask.any():
        primary_summary_columns = [
            "model",
            "representation",
            "balanced_accuracy_mean",
            "permutation_p_value_sex_stratified",
        ]
        logger.info(
            "Primary result: %s", comparison_df.loc[primary_mask, primary_summary_columns].to_dict("records")
        )
    return path


def run_control_vs_disease_ml_pipeline(
    *,
    feature_matrix_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    cfg: dict[str, Any] | None = None,
    resume: bool = False,
    data_manifest_hash: str | None = None,
) -> dict[str, Path]:
    cfg_in = _cfg(cfg)
    ml_cfg = _ml_cfg(cfg_in)
    if not bool(ml_cfg.get("enabled", True)):
        return {}

    target_dir = Path(output_dir) if output_dir is not None else get_ml_classification_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    cohort_dir = get_ml_classification_phase_dir("cohort", root=target_dir, create=False)
    model_dir = get_ml_classification_phase_dir("model_evaluation", root=target_dir, create=False)
    confounder_dir = get_ml_classification_phase_dir("confounder_checks", root=target_dir, create=False)
    permutation_dir = get_ml_classification_phase_dir("permutation_tests", root=target_dir, create=False)
    band_screen_dir = get_ml_classification_phase_dir("band_screen", root=target_dir, create=False)
    metadata_dir = get_run_metadata_dir() if output_dir is None else target_dir / "00_run_metadata"
    checkpoints = _CheckpointStore.create(
        metadata_dir=metadata_dir,
        cfg_in=cfg_in,
        feature_matrix_path=feature_matrix_path,
        data_manifest_hash=data_manifest_hash,
        resume=resume,
    )
    phase_timings: list[dict[str, object]] = []
    outputs: dict[str, Path] = {}

    representations = list(ml_cfg.get("representations", ["whole", "band_panel"]))
    n_jobs = normalize_n_jobs(int(ml_cfg.get("n_jobs", -1)))
    random_state = _random_state(ml_cfg)
    comparison_rows: list[dict[str, object]] = []
    confounder_rows: list[dict[str, object]] = []
    null_distribution_rows: list[dict[str, object]] = []
    band_screen_frames: list[pd.DataFrame] = []
    phase_count = 2 + len(representations) * (2 + len(_MODEL_NAMES) * 5) + 5 + int(output_dir is not None)
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=False,
    )
    overall_task = progress.add_task("Control-vs-disease ML", total=phase_count)

    phase = partial(
        _phase_timer, phase_timings, metadata_dir=metadata_dir,
        progress=progress, task_id=overall_task,
    )
    with progress:
        with phase("load_cohort", "Load cohort and labels"):
            frame = _load_cohort(feature_matrix_path)
            y = _labels(frame)

        with phase("cohort_summary", "Cohort summary"):
            cohort_df = _cohort_summary(frame)
            outputs["cohort_summary"] = write_csv_atomic(cohort_df, cohort_dir / "cohort_summary.csv", index=False)

        stratify_by = ml_cfg.get("stratify_permutation_by")
        if stratify_by not in (None, "sex"):
            raise ValueError(f"Unsupported stratify_permutation_by: {stratify_by!r}; expected 'sex' or None")
        sex_column = _sex_column(frame)
        for representation in representations:
            with phase(f"features_{representation}", f"Build {representation} feature matrix"):
                X, feature_names = _representation_matrix(frame, representation, cfg_in)
                valid_rows = np.isfinite(X).all(axis=1)
                if not valid_rows.all():
                    logger.warning(
                        "%d subject(s) dropped from representation '%s' due to non-finite features",
                        int((~valid_rows).sum()),
                        representation,
                    )
                X_valid, y_valid = X[valid_rows], y[valid_rows]

            strata = None
            if sex_column is not None and stratify_by == "sex":
                recorded_sex = frame.loc[valid_rows, sex_column].astype("string").str.strip().str.lower()
                if not recorded_sex.isin(["male", "female"]).all():
                    raise ValueError(
                        "Sex-stratified inference requires complete recorded sex values; "
                        "folder labels cannot replace missing clinical sex"
                    )
                strata = recorded_sex.to_numpy()
            if stratify_by == "sex" and strata is None:
                raise ValueError(
                    "Sex-stratified classification requires a non-missing sex column; unrestricted inference is not a fallback"
                )
            outer_split_plan = _locked_joint_stratified_split_plan(
                y_valid,
                strata,
                n_outer_splits=_n_outer_splits(ml_cfg),
                n_outer_repeats=int(ml_cfg.get("n_outer_repeats", 10)),
                random_state=random_state,
            )
            permutation_outer_fits = 2 * _n_permutations(ml_cfg) * len(outer_split_plan[0])
            workload_message = (
                f"{representation}: subjects={len(y_valid)}, features={X_valid.shape[1]}, models={len(_MODEL_NAMES)}, "
                f"workers={n_jobs}, permutation outer fits/model={permutation_outer_fits} "
                f"({len(outer_split_plan[0])} locked folds × 2 reference distributions × {_n_permutations(ml_cfg)} draws)"
            )
            logger.info("Control-vs-disease ML workload: %s", workload_message)
            progress.console.print(f"[cyan]ML workload:[/] {workload_message}")
            sex_only = _sex_only_baseline(frame, y, valid_rows, outer_split_plan=outer_split_plan, random_state=random_state)

            with phase(f"band_screen_{representation}", f"Univariate band screen ({representation})"):
                if representation == "band_panel":
                    band_screen_frames.append(run_univariate_band_screen(frame, y, valid_rows, cfg_in))

            for model_name in _MODEL_NAMES:
                oof, metrics = _run_observed_model_phase(
                    model_name=model_name, representation=representation,
                    X_valid=X_valid, y_valid=y_valid, feature_names=feature_names,
                    ml_cfg=ml_cfg, n_jobs=n_jobs, outer_split_plan=outer_split_plan,
                    checkpoints=checkpoints, model_dir=model_dir, outputs=outputs,
                    phase=phase, progress=progress,
                )

                permutation_observed_score = _repeat_zero_balanced_accuracy(oof)
                logger.info(
                    "Permutation observed score %s (%s): %.4f on locked 5-fold plan",
                    model_name,
                    representation,
                    permutation_observed_score,
                )

                p_unrestricted, p_sex_stratified, n_unrestricted, n_sex_stratified = _run_model_permutation_phase(
                    model_name=model_name, representation=representation,
                    X_valid=X_valid, y_valid=y_valid, strata=strata,
                    permutation_observed_score=permutation_observed_score,
                    ml_cfg=ml_cfg, n_jobs=n_jobs, random_state=random_state,
                    outer_split_plan=outer_split_plan, checkpoints=checkpoints, resume=resume,
                    null_distribution_rows=null_distribution_rows, phase=phase, progress=progress,
                )

                with _progress_task(
                    progress,
                    f"{model_name} ({representation}) bootstrap draws",
                    int(ml_cfg.get("n_bootstrap", 1000)),
                ) as bootstrap_task:
                    with phase(f"{model_name}_{representation}_bootstrap", f"{model_name} ({representation}) bootstrap CI"):
                        bootstrap_ci = run_bootstrap_ci(
                            model_name,
                            X_valid,
                            y_valid,
                            ml_cfg=ml_cfg,
                            n_jobs=n_jobs,
                            random_state=random_state,
                            checkpoint_path=checkpoints.resampling_path(
                                "bootstrap", model_name, representation, "metrics"
                            ),
                            resume=resume,
                            progress=progress,
                            progress_task_id=bootstrap_task,
                        )

                comparison_rows.append(
                    {
                        "model": model_name,
                        "representation": representation,
                        "n_subjects": int(len(y_valid)),
                        **metrics,
                        **{f"bootstrap_{k}": v for k, v in bootstrap_ci.items()},
                        "permutation_observed_balanced_accuracy": permutation_observed_score,
                        "permutation_cv_splits": len(outer_split_plan[0]),
                        "permutation_cv_repeats": 1,
                        "permutation_p_value_unrestricted": p_unrestricted,
                        "permutation_p_value_sex_stratified": p_sex_stratified,
                        "n_permutations": n_unrestricted,
                        "n_permutations_sex_stratified": n_sex_stratified,
                        "is_primary_model": model_name == ml_cfg.get("primary_model")
                        and representation == ml_cfg.get("primary_representation"),
                    }
                )

                with phase(f"{model_name}_{representation}_sex_refit", f"{model_name} ({representation}) female-only sensitivity refit"):
                    female_refit = _sex_stratified_refit(
                        model_name, frame, X_valid, y_valid, valid_rows, ml_cfg=ml_cfg, n_jobs=n_jobs
                    )
                confounder_rows.append(
                    {"model": model_name, "representation": representation, **female_refit, **sex_only}
                )

        with phase("model_comparison", "Write model comparison table"):
            outputs["model_comparison"] = _write_model_comparison(comparison_rows, model_dir)

        with phase("confounder_analysis", "Write confounder analysis table"):
            outputs["confounder_analysis"] = write_csv_atomic(
                pd.DataFrame(confounder_rows), confounder_dir / "confounder_analysis.csv", index=False
            )

        with phase("null_distributions", "Write permutation reference distributions"):
            outputs["permutation_null_distributions"] = write_csv_atomic(
                pd.DataFrame(null_distribution_rows),
                permutation_dir / "permutation_null_distributions.csv",
                index=False,
            )

        with phase("univariate_band_screen", "Write univariate band screen"):
            band_screen_df = pd.concat(band_screen_frames, ignore_index=True) if band_screen_frames else pd.DataFrame()
            outputs["univariate_band_screen"] = write_csv_atomic(
                band_screen_df, band_screen_dir / "univariate_band_screen.csv", index=False
            )

        with phase("report_and_figures", "Write figures and narrative report"):
            outputs.update(run_control_vs_disease_ml_report(target_dir, cfg_in))

        if output_dir is not None:
            with phase("provenance", "Write provenance artifacts"):
                write_analysis_provenance(
                    target_dir / "00_run_metadata",
                    resolve_data_source(cfg_in),
                    cfg_in,
                    command="run_control_vs_disease_ml_pipeline",
                )

    timing_path = _write_phase_timings(phase_timings, metadata_dir)
    if timing_path is not None:
        outputs["phase_timing"] = timing_path

    logger.info("Control-vs-disease ML outputs written to %s", target_dir)
    return outputs
