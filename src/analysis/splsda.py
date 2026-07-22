"""Phase 9: PLS-L1DA Exploratory Classifier (sPLS-DA-like model)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src import get_classification_phase_dir
from src import resolve_cfg as _cfg
from src.analysis.clinical_feature_associations import (
    feature_columns,
    load_disease_feature_frame,
)
from src.analysis.phase_status import STATUS_NOT_TESTED_GATE, PhaseStatus
from src.analysis.sklearn_utils import fit_with_convergence_capture, l1_logistic_model
from src.statistics.sequential_permutation import (
    resolve_sequential_stopping_config,
    should_stop,
    validate_sequential_stopping_config,
)

logger = logging.getLogger(__name__)


def _splsda_cfg(cfg: dict) -> dict:
    return cfg.get("analysis", {}).get("splsda", {})


def _remove_orthogonal_component(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Exploratory Orthogonal Projections to Latent Structures (OPLS)."""
    y_c = y - np.mean(y)

    w = np.dot(X.T, y_c) / (np.dot(y_c.T, y_c) + 1e-10)
    w_norm = np.linalg.norm(w)
    if w_norm < 1e-10:
        return X.copy()
    w /= w_norm

    t = np.dot(X, w)
    t_norm_sq = np.dot(t.T, t)
    if t_norm_sq < 1e-10:
        return X.copy()

    p = np.dot(X.T, t) / t_norm_sq

    w_ortho = p - np.dot(w.T, p) * w
    w_ortho_norm = np.linalg.norm(w_ortho)
    if w_ortho_norm < 1e-10:
        return X.copy()
    w_ortho /= w_ortho_norm

    t_ortho = np.dot(X, w_ortho)
    t_ortho_norm_sq = np.dot(t_ortho.T, t_ortho)
    if t_ortho_norm_sq < 1e-10:
        return X.copy()

    p_ortho = np.dot(X.T, t_ortho) / t_ortho_norm_sq

    X_ortho_removed = X - np.outer(t_ortho, p_ortho)
    return X_ortho_removed


def _tune_pls_l1da(
    X_train: np.ndarray,
    y_train: np.ndarray,
    component_grid: list[int],
    c_range: list[float],
    n_inner_splits: int,
    random_state: int,
    groups: np.ndarray | None = None
) -> tuple[int, float]:
    best_n_comp = component_grid[0]
    best_c = c_range[0]
    best_score = -1.0

    min_class_count = np.min(np.bincount(y_train))
    n_splits = min(n_inner_splits, min_class_count)
    if n_splits < 2:
        return best_n_comp, best_c

    if groups is not None:
        inner_cv = StratifiedGroupKFold(n_splits=n_splits)
        cv_splitter = inner_cv.split(X_train, y_train, groups=groups)
    else:
        inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_splitter = inner_cv.split(X_train, y_train)

    folds = list(cv_splitter)

    for n_comp in component_grid:
        if n_comp > X_train.shape[0] - 2 or n_comp > X_train.shape[1] or n_comp < 1:
            continue

        for c in c_range:
            scores = []
            for tr_idx, te_idx in folds:
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_train[tr_idx])
                X_te = scaler.transform(X_train[te_idx])
                y_tr = y_train[tr_idx]
                y_te = y_train[te_idx]

                pls = PLSRegression(n_components=n_comp, scale=False)
                pls.fit(X_tr, y_tr)

                X_tr_pls = pls.transform(X_tr)
                X_te_pls = pls.transform(X_te)

                clf = l1_logistic_model(c=c, random_state=random_state)
                fit_with_convergence_capture(clf, X_tr_pls, y_tr, context="sPLS-DA inner CV")

                preds = clf.predict(X_te_pls)
                scores.append(balanced_accuracy_score(y_te, preds))

            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_n_comp = n_comp
                best_c = c

    return best_n_comp, best_c


def _permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    cv_strategy: object,
    groups: np.ndarray | None,
    component_grid: list[int],
    c_range: list[float],
    n_inner_splits: int,
    observed_bal_acc: float,
    n_permutations: int,
    random_state: int,
    n_jobs: int = 1,
    splsda_cfg: dict | None = None,
) -> tuple[float, int]:
    """Draws stop early once the Monte Carlo p-value's CI clears alpha.

    Besag & Clifford (1991) curtailment; see
    `src.statistics.sequential_permutation`. `n_permutations` is the ceiling
    on draws, not necessarily the count actually run — returns the draws
    actually used alongside the p-value.
    """
    n_jobs = 1 if n_jobs == 0 else n_jobs
    stopping = resolve_sequential_stopping_config(splsda_cfg or {}, default_max=n_permutations)
    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=stopping.permutation_max, dtype=np.int64)

    null_scores: list[float] = []
    successes = 0
    offset = 0
    while offset < len(seeds) and not should_stop(successes=successes, draws=len(null_scores), config=stopping):
        chunk = seeds[offset : offset + stopping.check_interval]
        offset += stopping.check_interval
        if n_jobs != 1 and len(chunk) > 1:
            chunk_scores = Parallel(n_jobs=n_jobs)(
                delayed(_permuted_splsda_score)(
                    X,
                    y,
                    cv_strategy,
                    groups,
                    component_grid,
                    c_range,
                    n_inner_splits,
                    random_state,
                    int(seed),
                )
                for seed in chunk
            )
        else:
            chunk_scores = [
                _permuted_splsda_score(
                    X,
                    y,
                    cv_strategy,
                    groups,
                    component_grid,
                    c_range,
                    n_inner_splits,
                    random_state,
                    int(seed),
                )
                for seed in chunk
            ]
        null_scores.extend(chunk_scores)
        successes += sum(1 for value in chunk_scores if value >= observed_bal_acc)

    p_value = float((1 + successes) / (len(null_scores) + 1)) if null_scores else float("nan")
    return p_value, len(null_scores)


def _permuted_splsda_score(
    X: np.ndarray,
    y: np.ndarray,
    cv_strategy: object,
    groups: np.ndarray | None,
    component_grid: list[int],
    c_range: list[float],
    n_inner_splits: int,
    random_state: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    return _permuted_splsda_score_with_rng(
        X,
        y,
        cv_strategy,
        groups,
        component_grid,
        c_range,
        n_inner_splits,
        random_state,
        rng,
    )


def _permuted_splsda_score_with_rng(
    X: np.ndarray,
    y: np.ndarray,
    cv_strategy: object,
    groups: np.ndarray | None,
    component_grid: list[int],
    c_range: list[float],
    n_inner_splits: int,
    random_state: int,
    rng: np.random.Generator,
) -> float:
    y_perm = rng.permutation(y)
    y_pred = np.zeros_like(y, dtype=int)

    cv_kwargs = {"groups": groups} if groups is not None else {}
    for train_idx, test_idx in cv_strategy.split(X, y_perm, **cv_kwargs):
        X_train, y_train = X[train_idx], y_perm[train_idx]
        X_test = X[test_idx]

        best_n_comp, best_c = _tune_pls_l1da(
            X_train,
            y_train,
            component_grid,
            c_range,
            n_inner_splits,
            random_state,
            groups[train_idx] if groups is not None else None,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        n_c = min(best_n_comp, X_train.shape[0] - 2)
        pls = PLSRegression(n_components=max(1, n_c), scale=False)
        pls.fit(X_train_scaled, y_train)

        X_train_pls = pls.transform(X_train_scaled)
        X_test_pls = pls.transform(X_test_scaled)

        clf = l1_logistic_model(c=best_c, random_state=random_state)
        fit_with_convergence_capture(clf, X_train_pls, y_train, context="sPLS-DA permutation CV")

        y_pred[test_idx] = clf.predict(X_test_pls)

    return float(balanced_accuracy_score(y_perm, y_pred))


def run_splsda_classifier(
    *,
    feature_matrix_path: str | Path | None = None,
    pls_outputs: dict[str, Path] | None = None,
    output_dir: str | Path | None = None,
    cfg: dict | None = None,
) -> dict[str, object]:
    cfg = _cfg(cfg)
    splsda_cfg = _splsda_cfg(cfg)

    if not bool(splsda_cfg.get("enabled", True)):
        return {}
    validate_sequential_stopping_config(splsda_cfg, section="analysis.splsda")

    target_dir = Path(output_dir) if output_dir is not None else get_classification_phase_dir("splsda")
    target_dir.mkdir(parents=True, exist_ok=True)

    gate_p = float(splsda_cfg.get("gate_on_pls_p_value", 0.05))
    if pls_outputs is not None and "pls_permutation_results" in pls_outputs:
        try:
            pls_perm = pd.read_csv(pls_outputs["pls_permutation_results"])
            if pls_perm["permutation_p_value"].min() >= gate_p:
                logger.info("Skipping PLS-L1DA: No PLS model passed permutation gate (p < %s)", gate_p)
                gated_targets = splsda_cfg.get("targets", [])
                if not gated_targets:
                    return {
                        "_status": PhaseStatus(
                            phase_key="splsda",
                            status=STATUS_NOT_TESTED_GATE,
                            reason="no PLS model passed permutation gate and no targets configured",
                            n_subjects=None,
                            n_artifacts=0,
                        ).as_row()
                    }
                perf_path = target_dir / "pls_l1da_cv_performance.csv"
                perm_path = target_dir / "pls_l1da_permutation_results.csv"
                pd.DataFrame(
                    [
                        {
                            "group_variable": target.get("group_variable"),
                            "method_label": "PLS-L1DA",
                            "inference_role": "conditional_exploratory",
                            "status": "not_tested_gate",
                        }
                        for target in gated_targets
                    ],
                    columns=[
                        "group_variable",
                        "balanced_accuracy",
                        "roc_auc",
                        "n_subjects",
                        "n_per_class",
                        "selected_n_components",
                        "selected_C",
                        "cv_strategy",
                        "method_label",
                        "inference_role",
                        "status",
                    ]
                ).to_csv(perf_path, index=False)
                # No per-feature rows exist here (the gate fires before any band is
                # ever scored), unlike perf_path/perm_path which carry one legitimate
                # status row per gated target. Skip the header-only write.
                pd.DataFrame(
                    [
                        {
                            "group_variable": target.get("group_variable"),
                            "n_permutations": 0,
                            "inference_role": "conditional_exploratory",
                            "status": "not_tested_gate",
                        }
                        for target in gated_targets
                    ],
                    columns=[
                        "group_variable",
                        "observed_balanced_accuracy",
                        "observed_roc_auc",
                        "permutation_p_value",
                        "n_permutations",
                        "inference_role",
                        "status",
                    ]
                ).to_csv(perm_path, index=False)
                return {
                    "pls_l1da_cv_performance": perf_path,
                    "pls_l1da_permutation_results": perm_path,
                    "_status": PhaseStatus(
                        phase_key="splsda",
                        status=STATUS_NOT_TESTED_GATE,
                        reason=f"no PLS model passed permutation gate (p < {gate_p:g})",
                        n_subjects=None,
                        n_artifacts=2,
                    ).as_row(),
                }
        except Exception as e:
            logger.warning("Could not read PLS permutation results to check gate: %s", e)

    disease_only, feature_meta = load_disease_feature_frame(feature_matrix_path=feature_matrix_path)
    all_feature_columns = feature_columns(disease_only)

    targets = splsda_cfg.get("targets", [
        {"group_variable": "moca_severity_group", "ordered_groups": ["low_moca", "high_moca"]},
        {"group_variable": "delta_moca_group", "ordered_groups": ["decline", "stable_or_improved"]}
    ])
    component_grid = splsda_cfg.get("component_grid", [1, 2, 3, 4, 5])
    c_range = splsda_cfg.get("c_range", [0.01, 0.1, 1.0, 10.0])
    n_outer_splits = int(splsda_cfg.get("n_outer_splits", 5))
    n_inner_splits = int(splsda_cfg.get("n_inner_splits", 3))
    min_subjects_per_group = int(splsda_cfg.get("min_subjects_per_group", 3))
    n_permutations = int(splsda_cfg.get("n_permutations", 500))
    n_jobs = int(splsda_cfg.get("n_jobs", 1))
    random_state = int(splsda_cfg.get("random_state", 42))
    emit_oplsda = bool(splsda_cfg.get("emit_oplsda", False))

    perf_rows: list[dict[str, object]] = []
    bands_rows: list[dict[str, object]] = []
    perm_rows: list[dict[str, object]] = []
    oplsda_rows: list[dict[str, object]] = []

    for target_def in targets:
        group_var = target_def["group_variable"]
        ordered_groups = target_def["ordered_groups"]

        if group_var not in disease_only.columns:
            logger.info("Target '%s' not found, skipping PLS-L1DA", group_var)
            perf_rows.append(
                {
                    "group_variable": group_var,
                    "method_label": "PLS-L1DA",
                    "inference_role": "conditional_exploratory",
                    "status": "missing_required_data",
                }
            )
            continue

        frame = disease_only[disease_only[group_var].isin(ordered_groups)].copy()
        class_counts = frame[group_var].value_counts()
        if len(class_counts) < 2 or class_counts.min() < min_subjects_per_group:
            logger.info(
                "Target '%s' has insufficient class sizes (%s), skipping. Minimum required is %d per class.",
                group_var,
                dict(class_counts),
                min_subjects_per_group,
            )
            perf_rows.append(
                {
                    "group_variable": group_var,
                    "n_subjects": int(len(frame)),
                    "n_per_class": str(dict(class_counts)),
                    "method_label": "PLS-L1DA",
                    "inference_role": "conditional_exploratory",
                    "status": "insufficient_n",
                }
            )
            continue

        y = (frame[group_var] == ordered_groups[1]).astype(int).to_numpy()
        X = frame[all_feature_columns].to_numpy(dtype=float)

        groups = None
        if frame["subject_id"].nunique() < len(frame):
            groups = frame["subject_id"].to_numpy()

        n_splits = min(n_outer_splits, class_counts.min())
        if groups is not None:
            cv_strategy = StratifiedGroupKFold(n_splits=n_splits)
            cv_name = "StratifiedGroupKFold"
        else:
            cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            cv_name = "StratifiedKFold"

        y_pred = np.zeros_like(y, dtype=int)
        y_prob = np.zeros_like(y, dtype=float)
        fold_selected: list[np.ndarray] = []
        fold_signs: list[np.ndarray] = []

        cv_kwargs = {"groups": groups} if groups is not None else {}

        for train_idx, test_idx in cv_strategy.split(X, y, **cv_kwargs):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test = X[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            best_n, best_c = _tune_pls_l1da(X_train, y_train, component_grid, c_range, n_inner_splits, random_state,
                                            groups[train_idx] if groups is not None else None)

            pls = PLSRegression(n_components=best_n, scale=False)
            pls.fit(X_train_scaled, y_train)

            X_train_pls = pls.transform(X_train_scaled)
            X_test_pls = pls.transform(X_test_scaled)

            clf = l1_logistic_model(c=best_c, random_state=random_state)
            fit_with_convergence_capture(clf, X_train_pls, y_train, context="sPLS-DA outer CV")
            fold_weights = np.dot(pls.x_loadings_, clf.coef_.ravel())
            fold_selected.append(np.abs(fold_weights) > 1e-12)
            fold_signs.append(np.sign(fold_weights))

            y_pred[test_idx] = clf.predict(X_test_pls)
            if len(clf.classes_) == 2:
                y_prob[test_idx] = clf.predict_proba(X_test_pls)[:, 1]
            else:
                y_prob[test_idx] = y_pred[test_idx]

        bal_acc = balanced_accuracy_score(y, y_pred)
        try:
            roc_auc = roc_auc_score(y, y_prob)
        except ValueError:
            roc_auc = np.nan

        # Final model fit for feature extraction
        final_scaler = StandardScaler()
        X_scaled = final_scaler.fit_transform(X)
        final_n, final_c = _tune_pls_l1da(X, y, component_grid, c_range, n_inner_splits, random_state, groups)

        final_pls = PLSRegression(n_components=final_n, scale=False)
        final_pls.fit(X_scaled, y)
        X_scaled_pls = final_pls.transform(X_scaled)

        final_clf = l1_logistic_model(c=final_c, random_state=random_state)
        fit_with_convergence_capture(final_clf, X_scaled_pls, y, context="sPLS-DA final fit")

        log_reg_coefs = final_clf.coef_.ravel()
        pls_loadings = final_pls.x_loadings_
        overall_weights = np.dot(pls_loadings, log_reg_coefs)
        selection_frequency = np.mean(np.vstack(fold_selected), axis=0)
        sign_stability = np.abs(np.mean(np.vstack(fold_signs), axis=0))

        for i, col in enumerate(all_feature_columns):
            meta = feature_meta[feature_meta["feature_name"] == col].iloc[0]
            bands_rows.append({
                "group_variable": group_var,
                "feature_name": col,
                "spectral_block": meta["spectral_block"],
                "wavenumber": meta["wavenumber"],
                "loading_weight": float(overall_weights[i]),
                "selected": bool(overall_weights[i] != 0.0),
                "estimation_scope": "full_data_descriptive",
                "validated": False,
                "selection_frequency": float(selection_frequency[i]),
                "sign_stability": float(sign_stability[i]),
            })

        p_value, n_permutations_used = _permutation_test(
            X,
            y,
            cv_strategy,
            groups,
            component_grid,
            c_range,
            n_inner_splits,
            bal_acc,
            n_permutations,
            random_state,
            n_jobs=n_jobs,
            splsda_cfg=splsda_cfg,
        )

        perf_rows.append({
            "group_variable": group_var,
            "balanced_accuracy": bal_acc,
            "roc_auc": roc_auc,
            "n_subjects": frame["subject_id"].nunique(),
            "n_per_class": str(dict(class_counts)),
            "selected_n_components": final_n,
            "selected_C": final_c,
            "cv_strategy": cv_name,
            "method_label": "PLS-L1DA",
            "inference_role": "conditional_exploratory",
            "status": "ok",
        })

        perm_rows.append({
            "group_variable": group_var,
            "observed_balanced_accuracy": bal_acc,
            "observed_roc_auc": roc_auc,
            "permutation_p_value": p_value,
            "n_permutations": n_permutations_used,
            "inference_role": "conditional_exploratory",
            "status": "ok",
        })

        if emit_oplsda:
            logger.info("Running optional OPLS-DA for target %s", group_var)
            X_ortho = _remove_orthogonal_component(X_scaled, y)
            opls_model = PLSRegression(n_components=final_n, scale=False)
            opls_model.fit(X_ortho, y)
            y_pred_opls = np.round(np.clip(opls_model.predict(X_ortho).ravel(), 0, 1))
            opls_acc = np.mean(y == y_pred_opls)

            ss_res = np.sum((y - opls_model.predict(X_ortho).ravel())**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2y = 1 - ss_res/ss_tot if ss_tot > 0 else 0

            oplsda_rows.append({
                "group_variable": group_var,
                "accuracy": opls_acc,
                "Q2": np.nan,  # Too expensive to do nested CV for OPLS here
                "R2X": np.nan,
                "R2Y": r2y,
                "n_orthogonal_components": 1
            })

    res: dict[str, Path] = {}
    if perf_rows:
        perf_df = pd.DataFrame(perf_rows)
        perf_path = target_dir / "pls_l1da_cv_performance.csv"
        perf_df.to_csv(perf_path, index=False)
        res["pls_l1da_cv_performance"] = perf_path

        # bands_rows/perm_rows are only populated when at least one target
        # actually reached model fitting (i.e. cleared insufficient_n/
        # missing_data); a fully gated run has zero rows in both and should
        # not produce header-only files. perf_rows always has one row per
        # target (including gated ones, carrying their status), so it is
        # written unconditionally above.
        if bands_rows:
            bands_df = pd.DataFrame(bands_rows)
            bands_path = target_dir / "pls_l1da_selected_bands.csv"
            bands_df.to_csv(bands_path, index=False)
            res["pls_l1da_selected_bands"] = bands_path

        if perm_rows:
            perm_df = pd.DataFrame(perm_rows)
            perm_path = target_dir / "pls_l1da_permutation_results.csv"
            perm_df.to_csv(perm_path, index=False)
            res["pls_l1da_permutation_results"] = perm_path

        if emit_oplsda and oplsda_rows:
            oplsda_df = pd.DataFrame(oplsda_rows)
            oplsda_path = target_dir / "oplsda_results.csv"
            oplsda_df.to_csv(oplsda_path, index=False)
            res["oplsda_results"] = oplsda_path

        logger.info("PLS-L1DA outputs written to %s", target_dir)

    return res
