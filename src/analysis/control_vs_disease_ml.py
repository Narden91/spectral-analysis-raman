"""Control-vs-disease ML classification layer (disease_vs_control mode only).

Subject-level, nested-CV supervised classification of Control vs. Disease from
whole Raman spectra. This module is a separate, read-only downstream consumer
of the Biological PCA subject feature matrix. Config validation and phase
docstrings define the confound handling, claims discipline, and data limits.

Positive class = Disease (is_control == False -> 1). Exploratory only: no
diagnostic, biomarker, causal, or clinical-utility claims. Age is not modeled
anywhere in this pipeline because no age data exists for control subjects in
this cohort (49/50 missing) and no proxy exists in the repository; inventing
one would inject bias rather than remove it.

Output map:
    01_cohort                  class balance and confound counts
    02_model_evaluation        nested CV, importance, metrics, bootstrap CIs
    03_confounder_checks       sex-only baseline and female-only refits
    04_permutation_tests       unrestricted and sex-stratified reference distributions
    05_univariate_band_screen  descriptive Mann-Whitney tests with FDR
    06_report                  figures and final performance summary
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import warnings
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from scipy import stats
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
from src.analysis.pls_regression import _calculate_vip
from src.analysis.sklearn_utils import fit_with_convergence_capture, l1_logistic_model
from src.analysis.statistics import fdr_correction
from src.data_preparation import resolve_data_source, write_analysis_provenance
from src.data_preparation.manifest import config_hash
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
_CHECKPOINT_SCHEMA_VERSION = 1
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


def _normalize_n_jobs(n_jobs: int) -> int:
    return 1 if n_jobs == 0 else n_jobs


def _file_sha256(path: str | Path | None) -> str:
    """Return a content fingerprint for a feature matrix used by resumable work."""
    if path is None:
        return "default_feature_matrix"
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class _CheckpointStore:
    """Small, fingerprinted store for expensive ML resampling outputs."""

    directory: Path
    identity: dict[str, object]
    resume: bool
    legacy_compatible: bool

    @classmethod
    def create(
        cls,
        *,
        metadata_dir: Path,
        cfg_in: dict[str, Any],
        feature_matrix_path: str | Path | None,
        data_manifest_hash: str | None,
        resume: bool,
        legacy_run_manifest: dict[str, object] | None,
    ) -> "_CheckpointStore":
        if resume and feature_matrix_path is None:
            raise ValueError("Resuming ML requires an explicit subject feature matrix path")
        identity = {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "analysis_config_hash": config_hash(cfg_in),
            "ml_config_hash": config_hash(_ml_cfg(cfg_in)),
            "data_manifest_hash": data_manifest_hash or "not_supplied",
            "feature_matrix_sha256": _file_sha256(feature_matrix_path),
        }
        fingerprint = config_hash(identity)
        directory = metadata_dir / "control_vs_disease_ml_checkpoints" / fingerprint
        metadata_path = directory / "checkpoint_manifest.json"
        preexisting = metadata_path.exists()
        legacy_compatible = bool(
            legacy_run_manifest
            and legacy_run_manifest.get("analysis_config_hash") == identity["analysis_config_hash"]
            and legacy_run_manifest.get("data_manifest_hash") == identity["data_manifest_hash"]
            and legacy_run_manifest.get("analysis_mode") == "disease_vs_control"
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
        return cls(directory=directory, identity=identity, resume=resume, legacy_compatible=legacy_compatible)

    def resampling_path(self, kind: str, model_name: str, representation: str, label: str) -> Path:
        return self.directory / f"{kind}_{model_name}_{representation}_{label}.parquet"

    def stage_path(self, model_name: str, representation: str, label: str) -> Path:
        return self.directory / f"stage_{model_name}_{representation}_{label}.json"

    def stage_complete(self, model_name: str, representation: str, label: str) -> bool:
        return self.resume and self.stage_path(model_name, representation, label).exists()

    def mark_stage_complete(self, model_name: str, representation: str, label: str) -> None:
        write_json_atomic(
            self.stage_path(model_name, representation, label), {"identity": self.identity, "status": "ok"}
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
    return next((c for c in ("gender", "sex") if c in frame.columns), None)


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
    rows.append({"metric": "age_adjustment_status", "value": "not_attempted_no_control_age_data"})

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


def _inner_cv_folds(X_train: np.ndarray, y_train: np.ndarray, n_inner_splits: int, random_state: int) -> list | None:
    n_splits = min(n_inner_splits, int(np.bincount(y_train).min()))
    if n_splits < 2:
        return None
    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(inner_cv.split(X_train, y_train))


def _scaled_inner_folds(
    X_train: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Scale each inner training/test pair once and share it across a model grid."""
    scaled: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for tr_idx, te_idx in folds:
        scaler = StandardScaler()
        scaled.append((tr_idx, te_idx, scaler.fit_transform(X_train[tr_idx]), scaler.transform(X_train[te_idx])))
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
) -> _ModelFit:
    component_grid = _pls_component_grid(ml_cfg)
    folds = _inner_cv_folds(X_train, y_train, n_inner_splits, random_state)
    best_n_comp, best_threshold, best_score = component_grid[0], 0.5, -np.inf

    if folds is not None:
        scaled_folds = _scaled_inner_folds(X_train, folds)
        max_valid_comp = min(min(len(tr_idx) for tr_idx, _ in folds), X_train.shape[1])
        valid_grid = [c for c in component_grid if 1 <= c <= max_valid_comp] or [1]
        for n_comp in valid_grid:
            oof_scores = np.zeros(len(y_train), dtype=float)
            for tr_idx, te_idx, X_tr, X_te in scaled_folds:
                pls = PLSRegression(n_components=n_comp, scale=False)
                pls.fit(X_tr, y_train[tr_idx])
                oof_scores[te_idx] = pls.predict(X_te).ravel()
            threshold, score = _tune_threshold_and_score(y_train, oof_scores, threshold_metric)
            if score > best_score:
                best_score, best_n_comp, best_threshold = score, n_comp, threshold

    final_n_comp = max(1, min(best_n_comp, min(X_train.shape[0] - 2, X_train.shape[1])))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = PLSRegression(n_components=final_n_comp, scale=False)
    model.fit(X_scaled, y_train)
    return _ModelFit(
        predict_proba=lambda X: model.predict(scaler.transform(X)).ravel(),
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
) -> _ModelFit:
    c_grid = _logistic_c_grid(ml_cfg)
    folds = _inner_cv_folds(X_train, y_train, n_inner_splits, random_state)
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

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = l1_logistic_model(c=best_c, random_state=random_state)
    fit_with_convergence_capture(model, X_scaled, y_train, context="cvd_ml logistic final")
    return _ModelFit(
        predict_proba=lambda X: model.predict_proba(scaler.transform(X))[:, 1],
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
) -> _ModelFit:
    n_estimators_grid = _rf_n_estimators_grid(ml_cfg)
    max_depth_grid = _rf_max_depth_grid(ml_cfg)
    folds = _inner_cv_folds(X_train, y_train, n_inner_splits, random_state)
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
                        model.fit(X_train[tr_idx], y_train[tr_idx])
                    oof_by_params[(n_estimators, max_depth)][te_idx] = model.predict_proba(X_train[te_idx])[:, 1]

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
    return fitter(X_train, y_train, ml_cfg, n_inner_splits, threshold_metric, random_state, n_jobs)


# ---------------------------------------------------------------------------
# Nested CV — one implementation shared by repeated-CV point estimates and the
# single-split permutation workhorse.
# ---------------------------------------------------------------------------


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one outer fold and return its held-out predictions."""
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
) -> tuple[np.ndarray, np.ndarray]:
    """Fit+predict across one set of outer folds. Returns (y_pred, y_score) out-of-fold arrays."""
    y_pred = np.zeros_like(y, dtype=int)
    y_score = np.zeros_like(y, dtype=float)
    if model_name == "random_forest" and n_jobs != 1 and len(outer_cv) > 1:
        # Parallelize independent outer folds rather than nesting joblib pools.
        # The data splits, random seeds, and candidate selection are unchanged.
        calls = [
            (
                fold_index,
                (
                    model_name,
                    X,
                    y,
                    train_idx,
                    test_idx,
                    ml_cfg,
                    n_inner_splits,
                    threshold_metric,
                    random_state,
                    1,
                ),
            )
            for fold_index, (train_idx, test_idx) in enumerate(outer_cv)
        ]

        def on_result(_fold_index: int, result: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
            test_idx, fold_pred, fold_score = result
            y_pred[test_idx] = fold_pred
            y_score[test_idx] = fold_score
            if fold_callback is not None:
                fold_callback()

        _stream_parallel(n_jobs, _fit_outer_fold_prediction, calls, on_result)
        return y_pred, y_score

    for train_idx, test_idx in outer_cv:
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
            n_jobs,
        )
        y_score[test_idx] = fold_score
        y_pred[test_idx] = fold_pred
        if fold_callback is not None:
            fold_callback()
    return y_pred, y_score


def _outer_splits(y: np.ndarray, n_outer_splits: int, random_state: int) -> StratifiedKFold:
    n_splits = min(n_outer_splits, int(np.bincount(y).min()))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


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

    rows: list[dict[str, object]] = []
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


def _single_split_balanced_accuracy(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    ml_cfg: dict[str, Any],
    n_jobs: int,
    random_state: int,
) -> float:
    """Run one un-repeated outer StratifiedKFold pass; return balanced accuracy.

    This is the permutation-test workhorse. Permutation only needs a point
    estimate per permutation draw — the reference distribution's spread across draws
    already captures Monte Carlo noise, so repeating the outer CV per
    permutation (as the observed estimate does) would multiply
    cost by n_outer_repeats for no statistical benefit. This mirrors the
    single-split-per-permutation convention already used by
    `splsda._permuted_splsda_score` and `pls_regression._permuted_pls_r2`
    elsewhere in this repo.
    """
    n_outer_splits = _n_outer_splits(ml_cfg)
    n_inner_splits = _n_inner_splits(ml_cfg)
    threshold_metric = _threshold_metric(ml_cfg)
    outer_cv = list(_outer_splits(y, n_outer_splits, random_state).split(X, y))
    # Parallelism is applied across permutations by the caller, not within one.
    y_pred, _y_score = _cv_predict(
        model_name,
        X,
        y,
        outer_cv,
        ml_cfg=ml_cfg,
        n_inner_splits=n_inner_splits,
        threshold_metric=threshold_metric,
        random_state=random_state,
        n_jobs=1,
    )
    return float(balanced_accuracy_score(y, y_pred))


def _fixed_plan_balanced_accuracy(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    ml_cfg: dict[str, Any],
    outer_cv: Sequence[tuple[np.ndarray, np.ndarray]],
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
    outer_split_plan: Sequence[Sequence[tuple[np.ndarray, np.ndarray]]] | None = None,
) -> float:
    rng = np.random.default_rng(seed)
    y_perm = _permute_labels(y, rng, strata)
    if len(np.unique(y_perm)) < 2:
        return np.nan
    if outer_split_plan is None:
        return _single_split_balanced_accuracy(model_name, X, y_perm, ml_cfg=ml_cfg, n_jobs=1, random_state=seed)
    return _fixed_plan_balanced_accuracy(model_name, X, y_perm, ml_cfg=ml_cfg, outer_cv=outer_split_plan[0])


def _resampling_seeds(random_state: int, count: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    return rng.integers(0, np.iinfo(np.int32).max, size=count, dtype=np.int64)


def _load_checkpoint_rows(path: Path | None, *, resume: bool) -> pd.DataFrame:
    if not resume or path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _run_seeded_call(seed: int, func: Callable[..., Any], args: tuple[Any, ...]) -> tuple[int, Any]:
    """Return a draw's seed with its result so unordered workers remain traceable."""
    return seed, func(*args)


def _stream_parallel(
    n_jobs: int,
    func: Callable[..., Any],
    calls: Sequence[tuple[int, tuple[Any, ...]]],
    on_result: Callable[[int, Any], None],
) -> None:
    """Yield completed deterministic jobs to parent process for progress/checkpointing."""
    if n_jobs == 1 or len(calls) <= 1:
        for seed, args in calls:
            on_result(seed, func(*args))
        return

    parallel = Parallel(n_jobs=n_jobs, return_as="generator_unordered")
    results = parallel(delayed(_run_seeded_call)(seed, func, args) for seed, args in calls)
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
    offset = 0
    while offset < len(calls) and should_continue():
        chunk = calls[offset : offset + chunk_size]
        offset += chunk_size
        if n_jobs == 1 or len(chunk) <= 1:
            for seed, args in chunk:
                on_result(seed, func(*args))
            continue
        parallel = Parallel(n_jobs=n_jobs, return_as="list")
        results = parallel(delayed(_run_seeded_call)(seed, func, args) for seed, args in chunk)
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
    outer_split_plan: Sequence[Sequence[tuple[np.ndarray, np.ndarray]]] | None = None,
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
    pending = [
        (int(seed), (model_name, X, y, strata, ml_cfg, int(seed), outer_split_plan))
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
    """One out-of-bag bootstrap draw: fit on the in-bag resample, score on subjects never drawn.

    Resampling subjects with replacement and then running cross-validation on
    the resampled array (as a naive implementation might) leaks: a duplicated
    subject's two copies can land in different CV folds, so the model can see
    a near-identical copy of a "held-out" subject during training, inflating
    performance metrics. The classic fix — used here — is out-of-bag scoring:
    fit once on the in-bag resample (duplicates and all; that's fine, nothing
    is held out from *fitting*), then evaluate only on the subjects whose
    original row was never selected into this resample. Those subjects are
    guaranteed absent from every duplicate too, so there is no leakage path.
    """
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
    fit = _fit_tuned_model(
        model_name,
        X_in,
        y_in,
        ml_cfg=ml_cfg,
        n_inner_splits=n_inner_splits,
        threshold_metric=threshold_metric,
        random_state=seed,
        n_jobs=1,
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
        random_state=random_state + repeat_index * 101 + fold_index,
        n_jobs=estimator_n_jobs,
    )
    base = {
        "feature_name": feature_names,
        "repeat": repeat_index,
        "fold": fold_index,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
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

    result = permutation_importance(
        fit.estimator,
        X[test_idx],
        y[test_idx],
        n_repeats=int(ml_cfg.get("importance_permutation_repeats", 5)),
        random_state=random_state + repeat_index * 101 + fold_index,
        n_jobs=estimator_n_jobs,
        scoring="balanced_accuracy",
    )
    return pd.DataFrame(
        {
            **base,
            "importance": result.importances_mean,
            "importance_std": result.importances_std,
            "importance_type": "permutation_importance_outer_test",
        }
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
    """Process-safe wrapper for one single-threaded RF importance fold."""
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

    if model_name == "random_forest" and n_jobs != 1:
        fold_specs = [
            (repeat_index, fold_index, train_idx, test_idx)
            for repeat_index, repeat_splits in enumerate(outer_split_plan)
            for fold_index, (train_idx, test_idx) in enumerate(repeat_splits)
        ]
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
        for repeat_index, repeat_splits in enumerate(outer_split_plan):
            for fold_index, (train_idx, test_idx) in enumerate(repeat_splits):
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
                        estimator_n_jobs=n_jobs,
                    )
                )
                if fold_callback is not None:
                    fold_callback()
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


# Sex-confound battery
# ---------------------------------------------------------------------------


def _sex_only_baseline(frame: pd.DataFrame, y: np.ndarray, valid_rows: np.ndarray) -> dict[str, object]:
    """Logistic model using sex alone. The spectral model must beat this to
    support any spectral claim (§0.1) — it is not itself a claim of anything.
    """
    sex_column = _sex_column(frame)
    if sex_column is None:
        return {"status": "missing_sex_column"}

    sex_values = frame.loc[valid_rows, sex_column].astype(str).to_numpy()
    y_valid = y[valid_rows]
    X_sex = (sex_values == "male").astype(float).reshape(-1, 1)

    n_splits = min(5, int(np.bincount(y_valid).min()))
    if n_splits < 2:
        return {"status": "insufficient_class_size_for_cv"}

    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred = np.zeros_like(y_valid, dtype=int)
    y_score = np.zeros_like(y_valid, dtype=float)
    for train_idx, test_idx in outer_cv.split(X_sex, y_valid):
        model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
        fit_with_convergence_capture(model, X_sex[train_idx], y_valid[train_idx], context="sex-only baseline")
        train_scores = model.predict_proba(X_sex[train_idx])[:, 1]
        y_score[test_idx] = model.predict_proba(X_sex[test_idx])[:, 1]
        threshold = _tune_threshold(y_valid[train_idx], train_scores, "balanced_accuracy")
        y_pred[test_idx] = (y_score[test_idx] >= threshold).astype(int)

    metrics = _classification_metrics(y_valid, y_pred, y_score)
    return {"status": "ok", **{f"sex_only_{k}": v for k, v in metrics.items()}}


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


def _legacy_observed_outputs_are_compatible(
    model_dir: Path,
    model_name: str,
    representation: str,
    *,
    n_subjects: int,
    n_outer_repeats: int,
) -> bool:
    """Allow one-time reuse of pre-checkpoint observed outputs from this run root."""
    paths = (
        model_dir / f"{model_name}_{representation}_cv_predictions.csv",
        model_dir / f"{model_name}_{representation}_feature_importance_outer_folds.csv",
        model_dir / f"{model_name}_{representation}_feature_importance.csv",
    )
    if not all(path.exists() for path in paths):
        return False
    try:
        oof = pd.read_csv(paths[0])
        importance = pd.read_csv(paths[2])
    except (OSError, pd.errors.ParserError):
        return False
    return len(oof) == n_subjects * n_outer_repeats and not importance.empty


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
    observed_complete = checkpoints.stage_complete(model_name, representation, "observed") or (
        checkpoints.legacy_compatible
        and _legacy_observed_outputs_are_compatible(
            model_dir,
            model_name,
            representation,
            n_subjects=len(y_valid),
            n_outer_repeats=len(outer_split_plan),
        )
    )
    with phase(f"{model_name}_{representation}", f"{model_name} ({representation}) nested CV"):
        if (
            observed_complete
            and cv_path.exists()
            and importance_folds_path.exists()
            and importance_path.exists()
        ):
            oof = pd.read_csv(cv_path)
            metrics = _aggregate_repeat_metrics(oof)
            logger.info("Reused compatible observed ML outputs for %s (%s)", model_name, representation)
        else:
            with _progress_task(
                progress,
                f"{model_name} ({representation}) nested folds",
                sum(len(splits) for splits in outer_split_plan),
            ) as fold_task:
                oof = _run_repeated_nested_cv(
                    model_name,
                    X_valid,
                    y_valid,
                    ml_cfg=ml_cfg,
                    n_jobs=n_jobs,
                    outer_split_plan=outer_split_plan,
                    fold_callback=lambda: progress.advance(fold_task),
                )
            outputs[f"{model_name}_{representation}_cv_predictions"] = write_csv_atomic(
                oof, cv_path, index=False
            )
            metrics = _aggregate_repeat_metrics(oof)
            with _progress_task(
                progress,
                f"{model_name} ({representation}) importance folds",
                sum(len(splits) for splits in outer_split_plan),
            ) as importance_task:
                importance_folds, importance_df = _outer_fold_feature_importance(
                    model_name,
                    X_valid,
                    y_valid,
                    feature_names,
                    ml_cfg=ml_cfg,
                    n_jobs=n_jobs,
                    outer_split_plan=outer_split_plan,
                    fold_callback=lambda: progress.advance(importance_task),
                )
            outputs[f"{model_name}_{representation}_feature_importance_folds"] = write_csv_atomic(
                importance_folds, importance_folds_path, index=False
            )
            outputs[f"{model_name}_{representation}_feature_importance"] = write_csv_atomic(
                importance_df, importance_path, index=False
            )
            checkpoints.mark_stage_complete(model_name, representation, "observed")
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
) -> tuple[float, float]:
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
    return p_unrestricted, p_sex_stratified


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
    legacy_run_manifest: dict[str, object] | None = None,
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
        legacy_run_manifest=legacy_run_manifest,
    )
    phase_timings: list[dict[str, object]] = []
    outputs: dict[str, Path] = {}

    representations = list(ml_cfg.get("representations", ["whole", "band_panel"]))
    n_jobs = _normalize_n_jobs(int(ml_cfg.get("n_jobs", -1)))
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

            strata = (
                frame.loc[valid_rows, sex_column].astype(str).to_numpy()
                if (sex_column is not None and stratify_by == "sex")
                else None
            )
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
            sex_only = _sex_only_baseline(frame, y, valid_rows)

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

                p_unrestricted, p_sex_stratified = _run_model_permutation_phase(
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
                        "n_permutations": _n_permutations(ml_cfg),
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
