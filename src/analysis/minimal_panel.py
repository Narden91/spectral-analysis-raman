"""Phase 13: Minimal candidate Raman panel.

Selects a sparse panel of 5–10 spectral peaks/ratios associated with clinical
endpoints using nested cross-validated ElasticNet / L1-logistic regression,
permutation testing, and bootstrap confidence intervals.

All results are exploratory and subject-level; no diagnostic, predictive,
causal, or biomarker claims are made.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, ElasticNetCV, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src import config, get_cross_region_phase_dir
from src.analysis.clinical_feature_associations import (
    feature_columns,
    load_disease_feature_frame,
)

logger = logging.getLogger(__name__)

_DEFAULT_BINARY_TARGETS = (
    {"group_variable": "moca_severity_group", "ordered_groups": ("low_moca", "high_moca")},
    {"group_variable": "delta_moca_group", "ordered_groups": ("decline", "stable_or_improved")},
)


def _cfg(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    return config if cfg is None else cfg


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
        logger.warning("Integrated candidate table is empty")
    return df


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

    # Region-mean features from integrated candidate regions (High/Medium priority).
    if integrated is not None and not integrated.empty and "evidence_level" in integrated.columns:
        priority = integrated.loc[integrated["evidence_level"].isin({"High", "Medium"})].copy()
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


def _nested_cv_elasticnet(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_outer_splits: int,
    n_inner_splits: int,
    random_state: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, float]]:
    n_samples = X.shape[0]
    outer_cv = _cv_strategy(n_samples, n_outer_splits, random_state)
    y_pred = np.zeros_like(y, dtype=float)
    selected_alphas: list[float] = []
    selected_l1_ratios: list[float] = []

    for train_idx, test_idx in outer_cv.split(X, y) if isinstance(outer_cv, StratifiedKFold) else outer_cv.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test = X[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        inner_cv = KFold(n_splits=min(n_inner_splits, len(X_train)), shuffle=True, random_state=random_state)
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
            cv=inner_cv,
            random_state=random_state,
            max_iter=5000,
        )
        model.fit(X_train_scaled, y_train)
        y_pred[test_idx] = model.predict(X_test_scaled)
        selected_alphas.append(float(model.alpha_))
        selected_l1_ratios.append(float(model.l1_ratio_))

    perf = {
        "r2": float(r2_score(y, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "mae": float(mean_absolute_error(y, y_pred)),
    }
    cv_info = {
        "selected_alphas": selected_alphas,
        "selected_l1_ratios": selected_l1_ratios,
        "cv_strategy": type(outer_cv).__name__,
    }
    return y_pred, cv_info, perf


def _nested_cv_logistic_l1(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_outer_splits: int,
    n_inner_splits: int,
    random_state: int,
    c_values: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, float]]:
    if c_values is None:
        c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    n_samples = X.shape[0]
    class_counts = np.bincount(y)
    n_splits = min(n_outer_splits, max(2, class_counts.min()))
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    y_pred = np.zeros_like(y, dtype=int)
    y_prob = np.zeros_like(y, dtype=float)
    selected_cs: list[float] = []

    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test = X[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        inner_splits = min(n_inner_splits, np.bincount(y_train).min())
        inner_cv = StratifiedKFold(n_splits=max(2, inner_splits), shuffle=True, random_state=random_state)
        best_c = c_values[0]
        best_score = -1.0
        for c in c_values:
            scores = []
            for in_train, in_test in inner_cv.split(X_train_scaled, y_train):
                clf = LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    C=c,
                    class_weight="balanced",
                    random_state=random_state,
                )
                clf.fit(X_train_scaled[in_train], y_train[in_train])
                scores.append(balanced_accuracy_score(y_train[in_test], clf.predict(X_train_scaled[in_test])))
            mean_score = float(np.mean(scores))
            if mean_score > best_score:
                best_score = mean_score
                best_c = c

        final_clf = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=best_c,
            class_weight="balanced",
            random_state=random_state,
        )
        final_clf.fit(X_train_scaled, y_train)
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
    }
    return y_pred, y_prob, cv_info, perf


def _permutation_test_continuous(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_outer_splits: int,
    n_inner_splits: int,
    n_permutations: int,
    random_state: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(random_state)
    observed_r2 = _nested_cv_elasticnet(
        X, y, n_outer_splits=n_outer_splits, n_inner_splits=n_inner_splits, random_state=random_state
    )[2]["r2"]
    null_r2: list[float] = []
    for _ in range(n_permutations):
        y_perm = y[rng.permutation(len(y))]
        _, _, perf = _nested_cv_elasticnet(
            X, y_perm, n_outer_splits=n_outer_splits, n_inner_splits=n_inner_splits, random_state=random_state
        )
        null_r2.append(perf["r2"])
    null_arr = np.array(null_r2)
    finite = null_arr[np.isfinite(null_arr)]
    if len(finite) == 0:
        return np.nan, np.nan, np.nan
    p_value = float(np.mean(finite >= observed_r2))
    return p_value, float(np.mean(finite)), float(np.std(finite))


def _permutation_test_binary(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_outer_splits: int,
    n_inner_splits: int,
    n_permutations: int,
    random_state: int,
    c_values: list[float] | None = None,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(random_state)
    observed_bal_acc = _nested_cv_logistic_l1(
        X, y, n_outer_splits=n_outer_splits, n_inner_splits=n_inner_splits, random_state=random_state, c_values=c_values
    )[3]["balanced_accuracy"]
    null_scores: list[float] = []
    for _ in range(n_permutations):
        y_perm = y[rng.permutation(len(y))]
        _, _, _, perf = _nested_cv_logistic_l1(
            X, y_perm, n_outer_splits=n_outer_splits, n_inner_splits=n_inner_splits, random_state=random_state, c_values=c_values
        )
        null_scores.append(perf["balanced_accuracy"])
    null_arr = np.array(null_scores)
    finite = null_arr[np.isfinite(null_arr)]
    if len(finite) == 0:
        return np.nan, np.nan, np.nan
    p_value = float(np.mean(finite >= observed_bal_acc))
    return p_value, float(np.mean(finite)), float(np.std(finite))


def _bootstrap_coefficient_ci(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    n_bootstrap: int,
    random_state: int,
    task: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]
    coef_samples: dict[str, list[float]] = {name: [] for name in feature_names}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_samples, size=n_samples)
        X_boot, y_boot = X[idx], y[idx]
        if len(np.unique(y_boot)) < 2 and task == "binary":
            continue
        scaler = StandardScaler()
        X_boot_scaled = scaler.fit_transform(X_boot)
        try:
            if task == "continuous":
                model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state, max_iter=5000)
                model.fit(X_boot_scaled, y_boot)
                coefs = model.coef_
            else:
                model = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, class_weight="balanced", random_state=random_state)
                model.fit(X_boot_scaled, y_boot)
                coefs = model.coef_.ravel()
            for name, coef in zip(feature_names, coefs):
                coef_samples[name].append(float(coef))
        except Exception:
            continue

    rows: list[dict[str, object]] = []
    for name in feature_names:
        samples = np.array(coef_samples[name], dtype=float)
        rows.append({
            "feature_name": name,
            "ci_low": float(np.percentile(samples, 2.5)) if len(samples) else np.nan,
            "ci_high": float(np.percentile(samples, 97.5)) if len(samples) else np.nan,
            "mean_coefficient": float(np.mean(samples)) if len(samples) else np.nan,
            "n_bootstrap_used": int(len(samples)),
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
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        if task == "continuous":
            inner_cv = KFold(n_splits=min(n_inner_splits, len(X_train)), shuffle=True, random_state=random_state)
            model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0], cv=inner_cv, random_state=random_state, max_iter=5000)
            model.fit(X_train_scaled, y_train)
            coefs = model.coef_.reshape(-1, 1)
        else:
            inner_splits = min(n_inner_splits, np.bincount(y_train).min())
            inner_cv = StratifiedKFold(n_splits=max(2, inner_splits), shuffle=True, random_state=random_state)
            best_c = 1.0
            best_score = -1.0
            for c in [0.01, 0.1, 1.0, 10.0, 100.0]:
                scores = []
                for in_train, in_test in inner_cv.split(X_train_scaled, y_train):
                    clf = LogisticRegression(penalty="l1", solver="liblinear", C=c, class_weight="balanced", random_state=random_state)
                    clf.fit(X_train_scaled[in_train], y_train[in_train])
                    scores.append(balanced_accuracy_score(y_train[in_test], clf.predict(X_train_scaled[in_test])))
                if np.mean(scores) > best_score:
                    best_score = np.mean(scores)
                    best_c = c
            model = LogisticRegression(penalty="l1", solver="liblinear", C=best_c, class_weight="balanced", random_state=random_state)
            model.fit(X_train_scaled, y_train)
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
    output_dir: str | Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Run the minimal candidate Raman panel analysis."""
    cfg = _cfg(cfg)
    panel_cfg = _panel_cfg(cfg)
    if not bool(panel_cfg.get("enabled", True)):
        return {}

    target_dir = (
        Path(output_dir)
        if output_dir is not None
        else get_cross_region_phase_dir("minimal_panel")
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    disease_only, _feature_meta = load_disease_feature_frame(feature_matrix_path=feature_matrix_path)
    all_feature_columns = feature_columns(disease_only)

    if integrated_candidates_path is None:
        integrated_candidates_path = get_cross_region_phase_dir("integrated_candidates") / "integrated_candidate_regions.csv"
    integrated = _load_integrated_candidates(integrated_candidates_path)

    pool = _build_feature_pool(disease_only, all_feature_columns, integrated, cfg)
    if pool.empty:
        logger.warning("No candidate panel features available; writing empty outputs")
        pd.DataFrame(columns=[
            "endpoint", "endpoint_type", "feature_name", "feature_source", "feature_description",
            "selection_frequency", "mean_absolute_coefficient", "panel_selected",
        ]).to_csv(target_dir / "minimal_candidate_panel.csv", index=False)
        pd.DataFrame(columns=[
            "endpoint", "endpoint_type", "metric", "value", "n_subjects", "permutation_p_value",
            "n_permutations", "cv_strategy",
        ]).to_csv(target_dir / "minimal_panel_cv_performance.csv", index=False)
        pd.DataFrame(columns=[
            "endpoint", "feature_name", "mean_coefficient", "ci_low", "ci_high", "n_bootstrap_used",
        ]).to_csv(target_dir / "minimal_panel_coefficients.csv", index=False)
        return {
            "minimal_candidate_panel": target_dir / "minimal_candidate_panel.csv",
            "minimal_panel_cv_performance": target_dir / "minimal_panel_cv_performance.csv",
            "minimal_panel_coefficients": target_dir / "minimal_panel_coefficients.csv",
        }

    max_features = int(panel_cfg.get("max_features", 10))
    min_features = int(panel_cfg.get("min_features", 5))
    n_outer_splits = int(panel_cfg.get("n_outer_splits", 5))
    n_inner_splits = int(panel_cfg.get("n_inner_splits", 3))
    n_permutations = int(panel_cfg.get("n_permutations", 1000))
    n_bootstrap = int(panel_cfg.get("n_bootstrap", 200))
    random_state = int(panel_cfg.get("random_state", 42))

    continuous_targets = [str(t) for t in panel_cfg.get("continuous_targets", ["moca_baseline", "moca_3m", "delta_moca"])]
    binary_targets = panel_cfg.get("binary_targets", _DEFAULT_BINARY_TARGETS)

    panel_rows: list[dict[str, object]] = []
    perf_rows: list[dict[str, object]] = []
    coef_rows: list[dict[str, object]] = []

    feature_names = pool.columns.tolist()
    X_full = pool.to_numpy(dtype=float)

    for target in continuous_targets:
        if target not in disease_only.columns:
            logger.warning("Continuous target '%s' not found; skipping", target)
            continue
        y = pd.to_numeric(disease_only[target], errors="coerce")
        valid = y.notna() & np.isfinite(X_full).all(axis=1)
        if valid.sum() < 10:
            logger.warning("Target '%s' has only %d valid subjects; skipping", target, valid.sum())
            continue
        X = X_full[valid]
        y_arr = y.loc[valid].to_numpy(dtype=float)

        selected, selection_df, _ = _select_panel(
            X, y_arr, feature_names,
            n_outer_splits=n_outer_splits,
            n_inner_splits=n_inner_splits,
            random_state=random_state,
            max_features=max_features,
            min_features=min_features,
            task="continuous",
        )

        _, cv_info, perf = _nested_cv_elasticnet(
            X[:, [feature_names.index(s) for s in selected]],
            y_arr,
            n_outer_splits=n_outer_splits,
            n_inner_splits=n_inner_splits,
            random_state=random_state,
        )
        p_value, null_mean, null_std = _permutation_test_continuous(
            X[:, [feature_names.index(s) for s in selected]],
            y_arr,
            n_outer_splits=n_outer_splits,
            n_inner_splits=n_inner_splits,
            n_permutations=n_permutations,
            random_state=random_state,
        )
        ci_df = _bootstrap_coefficient_ci(
            X[:, [feature_names.index(s) for s in selected]],
            y_arr,
            selected,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            task="continuous",
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
            })
        perf_rows.append({
            "endpoint": target,
            "endpoint_type": "continuous",
            "metric": "r2",
            "value": perf["r2"],
            "n_subjects": int(valid.sum()),
            "permutation_p_value": p_value,
            "n_permutations": n_permutations,
            "cv_strategy": cv_info["cv_strategy"],
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
        })
        for _, row in ci_df.iterrows():
            coef_rows.append({
                "endpoint": target,
                "feature_name": row["feature_name"],
                "mean_coefficient": row["mean_coefficient"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "n_bootstrap_used": row["n_bootstrap_used"],
            })

    for target_def in binary_targets:
        group_var = str(target_def["group_variable"])
        ordered_groups = tuple(target_def["ordered_groups"])
        if group_var not in disease_only.columns:
            logger.warning("Binary target '%s' not found; skipping", group_var)
            continue
        subset = disease_only[disease_only[group_var].isin(ordered_groups)].copy()
        subset[group_var] = subset[group_var].astype(str)
        y = (subset[group_var] == ordered_groups[1]).astype(int)
        pool_subset = pool.loc[subset.index]
        X = pool_subset.to_numpy(dtype=float)
        valid = y.notna() & np.isfinite(X).all(axis=1)
        if valid.sum() < 10 or y.loc[valid].nunique() < 2:
            logger.warning("Binary target '%s' has insufficient valid subjects; skipping", group_var)
            continue
        X = X[valid]
        y_arr = y.loc[valid].to_numpy(dtype=int)

        selected, selection_df, _ = _select_panel(
            X, y_arr, pool_subset.columns.tolist(),
            n_outer_splits=n_outer_splits,
            n_inner_splits=n_inner_splits,
            random_state=random_state,
            max_features=max_features,
            min_features=min_features,
            task="binary",
        )

        _, _, cv_info, perf = _nested_cv_logistic_l1(
            X[:, [pool_subset.columns.get_loc(s) for s in selected]],
            y_arr,
            n_outer_splits=n_outer_splits,
            n_inner_splits=n_inner_splits,
            random_state=random_state,
        )
        p_value, null_mean, null_std = _permutation_test_binary(
            X[:, [pool_subset.columns.get_loc(s) for s in selected]],
            y_arr,
            n_outer_splits=n_outer_splits,
            n_inner_splits=n_inner_splits,
            n_permutations=n_permutations,
            random_state=random_state,
        )
        ci_df = _bootstrap_coefficient_ci(
            X[:, [pool_subset.columns.get_loc(s) for s in selected]],
            y_arr,
            selected,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            task="binary",
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
            })
        perf_rows.append({
            "endpoint": group_var,
            "endpoint_type": "binary",
            "metric": "balanced_accuracy",
            "value": perf["balanced_accuracy"],
            "n_subjects": int(valid.sum()),
            "permutation_p_value": p_value,
            "n_permutations": n_permutations,
            "cv_strategy": cv_info["cv_strategy"],
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
        })
        for _, row in ci_df.iterrows():
            coef_rows.append({
                "endpoint": group_var,
                "feature_name": row["feature_name"],
                "mean_coefficient": row["mean_coefficient"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "n_bootstrap_used": row["n_bootstrap_used"],
            })

    panel_df = pd.DataFrame(
        panel_rows,
        columns=[
            "endpoint", "endpoint_type", "feature_name", "feature_source", "feature_description",
            "selection_frequency", "mean_absolute_coefficient", "panel_selected",
        ],
    )
    perf_df = pd.DataFrame(
        perf_rows,
        columns=[
            "endpoint", "endpoint_type", "metric", "value", "n_subjects", "permutation_p_value",
            "n_permutations", "cv_strategy",
        ],
    )
    coef_df = pd.DataFrame(
        coef_rows,
        columns=[
            "endpoint", "feature_name", "mean_coefficient", "ci_low", "ci_high", "n_bootstrap_used",
        ],
    )

    panel_path = target_dir / "minimal_candidate_panel.csv"
    perf_path = target_dir / "minimal_panel_cv_performance.csv"
    coef_path = target_dir / "minimal_panel_coefficients.csv"

    panel_df.to_csv(panel_path, index=False)
    perf_df.to_csv(perf_path, index=False)
    coef_df.to_csv(coef_path, index=False)

    logger.info("Minimal candidate panel outputs written to %s", target_dir)
    return {
        "minimal_candidate_panel": panel_path,
        "minimal_panel_cv_performance": perf_path,
        "minimal_panel_coefficients": coef_path,
    }
