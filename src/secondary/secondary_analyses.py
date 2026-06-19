from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.analysis.statistics import fdr_correction
from src.features.compile_features import compile_feature_matrices


_SECONDARY_COLUMN_ALIASES = {
    "MOCA baseline Total [TEST]": ["moca_baseline_total"],
    "ASPECTS scale": ["aspects_scale_value"],
    "Diagnosis": ["diagnosis_raw", "diagnosis_normalized", "diagnosis_family"],
}


def run_secondary_analyses(
    feature_df: pd.DataFrame,
    area_df: pd.DataFrame | None = None,
    clinical_df: pd.DataFrame | None = None,
    split_index: int | None = None,
    feature_cols: list[str] | None = None,
    label_col: str = "is_control",
    disease_value: int | None = None,
    moca_col: str = "MOCA baseline Total [TEST]",
    aspects_col: str = "ASPECTS scale",
    subtype_col: str = "Diagnosis",
    random_state: int = 42,
) -> dict:
    """Run all Phase 8 secondary analyses."""
    secondary_cfg = config.get("secondary", {})
    moca_col = _resolve_secondary_column(feature_df, moca_col)
    aspects_col = _resolve_secondary_column(feature_df, aspects_col)
    subtype_col = _resolve_secondary_column(feature_df, subtype_col)

    use_cv_safe_features = area_df is not None and not area_df.empty

    if use_cv_safe_features:
        moca = predict_moca_disease_cv_safe(
            feature_df,
            area_df=area_df,
            clinical_df=clinical_df,
            split_index=split_index,
            feature_cols=feature_cols,
            moca_col=moca_col,
            label_col=label_col,
            disease_value=disease_value,
            method=str(secondary_cfg.get("moca_method", "ridge")),
            cv_mode=str(secondary_cfg.get("moca_cv", "loo")),
            random_state=random_state,
        )
    else:
        moca = predict_moca_disease(
            feature_df,
            feature_cols=feature_cols,
            moca_col=moca_col,
            label_col=label_col,
            disease_value=disease_value,
            method=str(secondary_cfg.get("moca_method", "ridge")),
            cv_mode=str(secondary_cfg.get("moca_cv", "loo")),
            random_state=random_state,
        )

    aspects = analyze_aspects_correlations(
        feature_df,
        feature_cols=feature_cols,
        aspects_col=aspects_col,
        label_col=label_col,
        disease_value=disease_value,
    )

    if use_cv_safe_features:
        subtype = explore_subtypes_multiclass_cv_safe(
            feature_df,
            area_df=area_df,
            clinical_df=clinical_df,
            split_index=split_index,
            feature_cols=feature_cols,
            subtype_col=subtype_col,
            label_col=label_col,
            disease_value=disease_value,
            random_state=random_state,
        )
    else:
        subtype = explore_subtypes_multiclass(
            feature_df,
            feature_cols=feature_cols,
            subtype_col=subtype_col,
            label_col=label_col,
            disease_value=disease_value,
            random_state=random_state,
        )

    source_label = "cv_train_only" if use_cv_safe_features else "descriptive_non_cv"
    if isinstance(moca, dict):
        moca["feature_source"] = source_label
    if isinstance(subtype, dict):
        subtype["feature_source"] = source_label

    return {
        "moca": moca,
        "aspects": aspects,
        "subtype": subtype,
    }


def predict_moca_disease_cv_safe(
    feature_df: pd.DataFrame,
    area_df: pd.DataFrame,
    clinical_df: pd.DataFrame | None = None,
    split_index: int | None = None,
    feature_cols: list[str] | None = None,
    moca_col: str = "MOCA baseline Total [TEST]",
    label_col: str = "is_control",
    disease_value: int | None = None,
    method: str = "ridge",
    cv_mode: str = "loo",
    random_state: int = 42,
) -> dict:
    """Leak-safe MoCA prediction with per-fold train-only feature fitting."""
    if cv_mode.lower() != "loo":
        raise ValueError("Only cv_mode='loo' is currently supported")

    if "subject_id" not in feature_df.columns:
        return {
            "status": "skipped",
            "reason": "subject_id is required for CV-safe secondary analyses",
            "predictions": pd.DataFrame(),
            "metrics": {},
        }

    df = _disease_subset(feature_df, label_col=label_col, disease_value=disease_value)
    if moca_col not in df.columns:
        raise ValueError(f"moca_col '{moca_col}' not found")

    work = df[["subject_id", moca_col]].copy()
    work["subject_id"] = work["subject_id"].astype(str)
    work[moca_col] = pd.to_numeric(work[moca_col], errors="coerce")
    work = work.dropna(subset=[moca_col]).drop_duplicates(subset=["subject_id"], keep="first").reset_index(drop=True)

    if len(work) < 3:
        return {
            "status": "skipped",
            "reason": "Not enough disease subjects with MOCA for LOO",
            "predictions": pd.DataFrame(),
            "metrics": {},
        }

    subject_ids = work["subject_id"].to_numpy(dtype=object)
    target_map = work.set_index("subject_id")[moca_col].to_dict()
    pred_map: dict[str, float] = {}

    loo = LeaveOneOut()
    for fold_idx, (train_idx, test_idx) in enumerate(loo.split(subject_ids), start=1):
        train_subjects = set(subject_ids[train_idx].tolist())
        test_subjects = set(subject_ids[test_idx].tolist())

        compiled = _compile_secondary_fold_features(
            area_df=area_df,
            clinical_df=clinical_df,
            train_subjects=train_subjects,
            test_subjects=test_subjects,
            split_index=split_index,
            random_state=random_state + fold_idx,
        )

        x_train_df = compiled["train_features_df"]
        x_test_df = compiled["test_features_df"]
        cols = _fold_feature_columns(x_train_df, feature_cols)

        x_train = x_train_df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        x_test = x_test_df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

        if x_train.shape[1] == 0:
            continue

        train_ids = x_train_df["subject_id"].astype(str).tolist()
        test_ids = x_test_df["subject_id"].astype(str).tolist()
        y_train = _targets_from_map(train_ids, target_map=target_map, target_name=moca_col).astype(float)

        model = _build_regressor(
            method=method,
            n_features=x_train.shape[1],
            n_train=x_train.shape[0],
            random_state=random_state + fold_idx,
        )
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )
        pipe.fit(x_train, y_train)
        y_pred = pipe.predict(x_test)

        for sid, pred in zip(test_ids, y_pred.tolist()):
            pred_map[str(sid)] = float(pred)

    pred_rows: list[dict] = []
    for sid in subject_ids.tolist():
        if sid in pred_map:
            pred_rows.append(
                {
                    "subject_id": sid,
                    "y_true": float(target_map[sid]),
                    "y_pred": float(pred_map[sid]),
                }
            )

    pred_df = pd.DataFrame(pred_rows)
    if pred_df.empty:
        return {
            "status": "skipped",
            "reason": "No CV predictions produced for MOCA",
            "predictions": pd.DataFrame(),
            "metrics": {},
        }

    y_true = pred_df["y_true"].to_numpy(dtype=float)
    y_pred = pred_df["y_pred"].to_numpy(dtype=float)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    sp = tuple(spearmanr(y_true, y_pred))
    metrics["spearman_rho"] = float(sp[0])
    metrics["spearman_p"] = float(sp[1])

    return {
        "status": "ok",
        "method": method,
        "predictions": pred_df,
        "metrics": metrics,
        "n_subjects": int(len(pred_df)),
    }


def explore_subtypes_multiclass_cv_safe(
    feature_df: pd.DataFrame,
    area_df: pd.DataFrame,
    clinical_df: pd.DataFrame | None = None,
    split_index: int | None = None,
    feature_cols: list[str] | None = None,
    subtype_col: str = "Diagnosis",
    label_col: str = "is_control",
    disease_value: int | None = None,
    min_count_per_class: int = 5,
    random_state: int = 42,
) -> dict:
    """Leak-safe subtype exploration with per-fold train-only feature fitting."""
    if "subject_id" not in feature_df.columns:
        return {"status": "skipped", "reason": "subject_id is required", "results": pd.DataFrame()}

    df = _disease_subset(feature_df, label_col=label_col, disease_value=disease_value)
    if subtype_col not in df.columns:
        return {"status": "skipped", "reason": f"Column '{subtype_col}' not found", "results": pd.DataFrame()}

    work = df[["subject_id", subtype_col]].copy()
    work["subject_id"] = work["subject_id"].astype(str)
    work[subtype_col] = work[subtype_col].astype(str).str.strip()
    work = work[work[subtype_col].str.len() > 0]
    work = work.drop_duplicates(subset=["subject_id"], keep="first")

    counts = work[subtype_col].value_counts()
    keep_classes = counts[counts >= int(min_count_per_class)].index.tolist()
    work = work[work[subtype_col].isin(keep_classes)].reset_index(drop=True)

    if len(keep_classes) < 2:
        return {
            "status": "skipped",
            "reason": "Insufficient subtype sample size after minimum-count filter",
            "class_counts": counts.rename_axis("subtype").reset_index(name="count"),
            "results": pd.DataFrame(),
        }

    subject_ids = work["subject_id"].to_numpy(dtype=object)
    y = work[subtype_col].to_numpy(dtype=object)
    subtype_map = work.set_index("subject_id")[subtype_col].to_dict()

    class_counts = pd.Series(y).value_counts()
    n_splits = int(min(5, class_counts.min()))
    if n_splits < 2:
        return {
            "status": "skipped",
            "reason": "Not enough observations per subtype for stratified CV",
            "class_counts": class_counts.rename_axis("subtype").reset_index(name="count"),
            "results": pd.DataFrame(),
        }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores: list[float] = []
    rows: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(subject_ids, y), start=1):
        train_subjects = set(subject_ids[train_idx].tolist())
        test_subjects = set(subject_ids[test_idx].tolist())

        compiled = _compile_secondary_fold_features(
            area_df=area_df,
            clinical_df=clinical_df,
            train_subjects=train_subjects,
            test_subjects=test_subjects,
            split_index=split_index,
            random_state=random_state + fold_idx,
        )

        x_train_df = compiled["train_features_df"]
        x_test_df = compiled["test_features_df"]
        cols = _fold_feature_columns(x_train_df, feature_cols)

        x_train = x_train_df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        x_test = x_test_df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

        if x_train.shape[1] == 0:
            continue

        train_ids = x_train_df["subject_id"].astype(str).tolist()
        test_ids = x_test_df["subject_id"].astype(str).tolist()
        y_train = _targets_from_map(train_ids, target_map=subtype_map, target_name=subtype_col)
        y_test = _targets_from_map(test_ids, target_map=subtype_map, target_name=subtype_col)

        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=random_state + fold_idx,
                    ),
                ),
            ]
        )
        pipe.fit(x_train, y_train)
        y_pred = pipe.predict(x_test)
        score = float(balanced_accuracy_score(y_test, y_pred))
        fold_scores.append(score)
        rows.append({"fold": fold_idx, "balanced_accuracy": score})

    if not fold_scores:
        return {
            "status": "skipped",
            "reason": "No usable features available for subtype CV folds",
            "class_counts": class_counts.rename_axis("subtype").reset_index(name="count"),
            "results": pd.DataFrame(),
        }

    return {
        "status": "ok",
        "n_classes": int(len(np.unique(y))),
        "n_subjects": int(len(y)),
        "class_counts": class_counts.rename_axis("subtype").reset_index(name="count"),
        "results": pd.DataFrame(rows),
        "mean_balanced_accuracy": float(np.mean(fold_scores)),
        "std_balanced_accuracy": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
    }


def _compile_secondary_fold_features(
    area_df: pd.DataFrame,
    clinical_df: pd.DataFrame | None,
    train_subjects: set[str],
    test_subjects: set[str],
    split_index: int | None,
    random_state: int,
) -> dict:
    train_area = area_df[area_df["subject_id"].astype(str).isin(train_subjects)].copy()
    test_area = area_df[area_df["subject_id"].astype(str).isin(test_subjects)].copy()

    if train_area.empty or test_area.empty:
        raise ValueError("Secondary fold has empty train/test area subset")

    relative_enabled_default = bool(config.get("relative_features", {}).get("enabled", False))

    return compile_feature_matrices(
        train_area_df=train_area,
        test_area_df=test_area,
        clinical_df=clinical_df,
        split_index=split_index,
        random_state=random_state,
        include_relative_features=bool(
            config.get("classification_flow", {}).get("include_relative_features", relative_enabled_default)
        ),
    )


def _fold_feature_columns(df: pd.DataFrame, feature_cols: list[str] | None) -> list[str]:
    if feature_cols is not None:
        cols = [c for c in feature_cols if c in df.columns and c != "subject_id"]
        if cols:
            return cols
    return _resolve_feature_cols(df, feature_cols=None, exclude={"subject_id"})


def _targets_from_map(subject_ids: list[str], target_map: dict, target_name: str) -> np.ndarray:
    missing = [sid for sid in subject_ids if sid not in target_map]
    if missing:
        raise ValueError(
            f"Missing target '{target_name}' for {len(missing)} subject(s): {missing[:10]}"
        )
    return np.asarray([target_map[sid] for sid in subject_ids], dtype=object)


def _resolve_secondary_column(feature_df: pd.DataFrame, requested: str) -> str:
    if requested in feature_df.columns:
        return requested
    for alias in _SECONDARY_COLUMN_ALIASES.get(requested, []):
        if alias in feature_df.columns:
            return alias
    return requested


def predict_moca_disease(
    feature_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    moca_col: str = "MOCA baseline Total [TEST]",
    label_col: str = "is_control",
    disease_value: int | None = None,
    method: str = "ridge",
    cv_mode: str = "loo",
    random_state: int = 42,
) -> dict:
    """Predict MOCA in disease subjects using leave-one-out style validation."""
    if cv_mode.lower() != "loo":
        raise ValueError("Only cv_mode='loo' is currently supported")

    df = _disease_subset(feature_df, label_col=label_col, disease_value=disease_value)
    if moca_col not in df.columns:
        raise ValueError(f"moca_col '{moca_col}' not found")

    cols = _resolve_feature_cols(df, feature_cols, exclude={moca_col, label_col, "subject_id"})
    work = df[[*cols, moca_col] + (["subject_id"] if "subject_id" in df.columns else [])].copy()
    work[moca_col] = pd.to_numeric(work[moca_col], errors="coerce")
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=[moca_col]).reset_index(drop=True)

    if len(work) < 3:
        return {
            "status": "skipped",
            "reason": "Not enough disease subjects with MOCA for LOO",
            "predictions": pd.DataFrame(),
            "metrics": {},
        }

    x_all = work[cols].to_numpy(dtype=float)
    y_all = work[moca_col].to_numpy(dtype=float)

    preds = np.zeros_like(y_all, dtype=float)
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(x_all):
        i = int(test_idx[0])
        x_train = x_all[train_idx]
        y_train = y_all[train_idx]
        x_test = x_all[test_idx]

        model = _build_regressor(method=method, n_features=x_train.shape[1], n_train=x_train.shape[0], random_state=random_state + i)

        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )
        pipe.fit(x_train, y_train)
        preds[i] = float(pipe.predict(x_test)[0])

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_all, preds))),
        "mae": float(mean_absolute_error(y_all, preds)),
        "r2": float(r2_score(y_all, preds)),
    }
    sp = tuple(spearmanr(y_all, preds))
    metrics["spearman_rho"] = float(sp[0])
    metrics["spearman_p"] = float(sp[1])

    pred_df = pd.DataFrame(
        {
            "y_true": y_all,
            "y_pred": preds,
        }
    )
    if "subject_id" in work.columns:
        pred_df.insert(0, "subject_id", work["subject_id"].astype(str).to_numpy())

    return {
        "status": "ok",
        "method": method,
        "predictions": pred_df,
        "metrics": metrics,
        "n_subjects": int(len(work)),
    }


def analyze_aspects_correlations(
    feature_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    aspects_col: str = "ASPECTS scale",
    label_col: str = "is_control",
    disease_value: int | None = None,
) -> pd.DataFrame:
    """Compute Spearman correlation of each feature with ASPECTS in disease subjects."""
    df = _disease_subset(feature_df, label_col=label_col, disease_value=disease_value)
    if aspects_col not in df.columns:
        raise ValueError(f"aspects_col '{aspects_col}' not found")

    cols = _resolve_feature_cols(df, feature_cols, exclude={aspects_col, label_col, "subject_id"})
    target = pd.to_numeric(df[aspects_col], errors="coerce")

    rows: list[dict] = []
    for col in cols:
        x = pd.to_numeric(df[col], errors="coerce")
        mask = target.notna() & x.notna()
        n = int(mask.sum())
        if n < 3:
            rho, p = np.nan, np.nan
        else:
            sp = tuple(spearmanr(x[mask].to_numpy(dtype=float), target[mask].to_numpy(dtype=float)))
            rho, p = float(sp[0]), float(sp[1])
        rows.append(
            {
                "feature": col,
                "n": n,
                "rho": float(rho) if np.isfinite(float(rho)) else np.nan,
                "p_value": float(p) if np.isfinite(float(p)) else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    out["p_fdr_bh"] = fdr_correction(out["p_value"].to_numpy(dtype=float))[0]
    return out.sort_values(["p_fdr_bh", "p_value"], ascending=[True, True]).reset_index(drop=True)


def explore_subtypes_multiclass(
    feature_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    subtype_col: str = "Diagnosis",
    label_col: str = "is_control",
    disease_value: int | None = None,
    min_count_per_class: int = 5,
    random_state: int = 42,
) -> dict:
    """Run conditional multi-class subtype exploration in disease subjects."""
    df = _disease_subset(feature_df, label_col=label_col, disease_value=disease_value)
    if subtype_col not in df.columns:
        return {"status": "skipped", "reason": f"Column '{subtype_col}' not found", "results": pd.DataFrame()}

    cols = _resolve_feature_cols(df, feature_cols, exclude={subtype_col, label_col, "subject_id"})
    work = df[[*cols, subtype_col]].copy()
    work[subtype_col] = work[subtype_col].astype(str).str.strip()
    work = work[work[subtype_col].str.len() > 0]

    counts = work[subtype_col].value_counts()
    keep_classes = counts[counts >= int(min_count_per_class)].index.tolist()
    work = work[work[subtype_col].isin(keep_classes)].reset_index(drop=True)

    if len(keep_classes) < 2:
        return {
            "status": "skipped",
            "reason": "Insufficient subtype sample size after minimum-count filter",
            "class_counts": counts.reset_index().rename(columns={"index": "subtype", subtype_col: "count"}),
            "results": pd.DataFrame(),
        }

    x = work[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y = work[subtype_col].to_numpy(dtype=object)

    class_counts = pd.Series(y).value_counts()
    n_splits = int(min(5, class_counts.min()))
    if n_splits < 2:
        return {
            "status": "skipped",
            "reason": "Not enough observations per subtype for stratified CV",
            "class_counts": class_counts.reset_index().rename(columns={"index": "subtype", 0: "count"}),
            "results": pd.DataFrame(),
        }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores: list[float] = []
    rows: list[dict] = []

    for fold_idx, (tr, te) in enumerate(cv.split(x, y), start=1):
        x_train, y_train = x[tr], y[tr]
        x_test, y_test = x[te], y[te]

        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=random_state + fold_idx,
                    ),
                ),
            ]
        )
        pipe.fit(x_train, y_train)
        y_pred = pipe.predict(x_test)
        score = float(balanced_accuracy_score(y_test, y_pred))
        fold_scores.append(score)
        rows.append({"fold": fold_idx, "balanced_accuracy": score})

    return {
        "status": "ok",
        "n_classes": int(len(np.unique(y))),
        "n_subjects": int(len(y)),
        "class_counts": class_counts.rename_axis("subtype").reset_index(name="count"),
        "results": pd.DataFrame(rows),
        "mean_balanced_accuracy": float(np.mean(fold_scores)),
        "std_balanced_accuracy": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
    }


def _build_regressor(method: str, n_features: int, n_train: int, random_state: int):
    m = method.lower().strip()
    if m == "ridge":
        return Ridge(alpha=1.0, random_state=random_state)
    if m == "pls":
        n_comp = int(max(1, min(5, n_features, n_train - 1)))
        return PLSRegression(n_components=n_comp)
    raise ValueError("method must be 'ridge' or 'pls'")


def _disease_subset(feature_df: pd.DataFrame, label_col: str, disease_value: int | None) -> pd.DataFrame:
    if label_col not in feature_df.columns:
        raise ValueError(f"label_col '{label_col}' not found")
    if disease_value is None:
        disease_value = _infer_disease_value(feature_df, label_col=label_col)
    return feature_df[feature_df[label_col] == disease_value].copy().reset_index(drop=True)


def _infer_disease_value(feature_df: pd.DataFrame, label_col: str) -> int:
    values = pd.Series(feature_df[label_col]).dropna().unique().tolist()
    values_set = set(values)

    if "control" in label_col.lower() and values_set.issubset({0, 1}):
        return 0
    if values_set.issubset({0, 1}):
        return 1

    if len(values) == 0:
        raise ValueError(f"Cannot infer disease label from empty column '{label_col}'")
    return int(values[0])


def _resolve_feature_cols(df: pd.DataFrame, feature_cols: list[str] | None, exclude: set[str]) -> list[str]:
    if feature_cols is not None:
        cols = [c for c in feature_cols if c in df.columns and c not in exclude]
        if not cols:
            raise ValueError("feature_cols has no usable columns")
        return cols

    inferred = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not inferred:
        raise ValueError("No numeric feature columns found")
    return inferred
