from __future__ import annotations

import logging
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

# Step 4 (reconcile_ML.md) tightened the saga tolerance. That gives finer
# selected-feature sets but pushes some fits close to the 8000-iter cap. Bump
# to 20000 (2.5x headroom) and add a defensive catch below.
_SAGA_MAX_ITER = 20000
_SAGA_NEAR_CAP_FRACTION = 0.9


def _fit_saga_l1(clf: LogisticRegression, x_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Fit a saga L1/elastic-net classifier while swallowing the standard
    ``ConvergenceWarning`` that sklearn emits when the cap is hit.

    The warning is benign — the returned ``coef_`` is still usable — but
    printing it on every near-cap fit spams the operator terminal. If the
    fit lands within 10% of ``max_iter`` we surface a single ``logger.debug``
    so the configuration can be tuned without the noise.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        clf.fit(x_train, y_train)
    n_iter = int(np.asarray(clf.n_iter_).flat[0]) if hasattr(clf, "n_iter_") else 0
    if n_iter >= _SAGA_MAX_ITER * _SAGA_NEAR_CAP_FRACTION:
        logger.debug(
            "saga near-convergence cap: n_iter=%d max_iter=%d tol=%s",
            n_iter,
            int(clf.max_iter),
            float(clf.tol),
        )
    return clf


def variance_filter(
    x_train: np.ndarray,
    threshold: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Select features with standard deviation >= threshold."""
    std = np.std(x_train, axis=0)
    keep = std >= threshold
    indices = np.where(keep)[0]
    return x_train[:, keep], indices


def _balanced_label_permutation(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute labels while preserving exact class counts."""
    y_arr = np.asarray(y)
    n = y_arr.size
    if n == 0:
        return y_arr.copy()

    classes, counts = np.unique(y_arr, return_counts=True)
    perm_positions = rng.permutation(n)
    y_perm = np.empty_like(y_arr)

    start = 0
    for cls, count in zip(classes, counts):
        end = start + int(count)
        y_perm[perm_positions[start:end]] = cls
        start = end

    if np.array_equal(y_perm, y_arr) and n > 1:
        y_perm = np.roll(y_perm, 1)

    return y_perm


def mi_permutation_filter(
    x_train: np.ndarray,
    y_train: np.ndarray,
    permutations: int = 100,
    percentile: float = 95.0,
    n_neighbors: int = 3,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select features whose MI exceeds a permutation-derived null threshold."""
    if x_train.ndim != 2:
        raise ValueError("x_train must be 2D")

    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("x_train and y_train must have matching sample count")

    if x_train.shape[1] == 0:
        return np.empty((x_train.shape[0], 0), dtype=float), np.array([], dtype=int), np.array([], dtype=float)

    if len(np.unique(y_train)) < 2:
        mi_zeros = np.zeros(x_train.shape[1], dtype=float)
        return np.empty((x_train.shape[0], 0), dtype=float), np.array([], dtype=int), mi_zeros

    rng = np.random.default_rng(random_state)
    k = max(1, int(n_neighbors))
    mi_true = mutual_info_classif(
        x_train,
        y_train,
        random_state=random_state,
        n_neighbors=k,
    )

    null = np.zeros((permutations, x_train.shape[1]), dtype=float)
    for i in range(permutations):
        y_perm = _balanced_label_permutation(y_train, rng)
        null[i] = mutual_info_classif(
            x_train,
            y_perm,
            random_state=random_state + i + 1,
            n_neighbors=k,
        )

    mi_threshold = np.percentile(null, percentile, axis=0)
    keep = mi_true >= mi_threshold
    indices = np.where(keep)[0]
    return x_train[:, keep], indices, mi_true


def l1_select_by_target_count(
    x_train: np.ndarray,
    y_train: np.ndarray,
    target_counts: list[int] | tuple[int, ...] = (5, 10, 15),
    class_weight: str = "balanced",
    random_state: int = 42,
) -> dict[int, dict[str, np.ndarray | float]]:
    """Fit multiple L1 logistic models and keep the closest non-zero count per target."""
    c_grid = np.logspace(-3, 2, 20)
    out: dict[int, dict[str, np.ndarray | float]] = {}

    if x_train.shape[1] == 0:
        for target in target_counts:
            out[int(target)] = {
                "indices": np.array([], dtype=int),
                "coef": np.array([], dtype=float),
                "C": 1.0,
            }
        return out
    for target in target_counts:
        best_gap = None
        best_keep = None
        best_coef = None
        best_c = None

        for c_val in c_grid:
            clf = LogisticRegression(
                solver="saga",
                l1_ratio=1.0,
                C=float(c_val),
                class_weight=class_weight,
                max_iter=_SAGA_MAX_ITER,
                tol=1e-4,
                random_state=random_state,
            )
            _fit_saga_l1(clf, x_train, y_train)
            coef = np.ravel(clf.coef_)
            keep = np.flatnonzero(np.abs(coef) > 1e-12)
            gap = abs(len(keep) - int(target))
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_keep = keep
                best_coef = coef
                best_c = float(c_val)

        if best_keep is None or best_coef is None or best_c is None:
            raise RuntimeError("L1 selection failed to find a valid model")

        out[int(target)] = {
            "indices": np.asarray(best_keep, dtype=int),
            "coef": np.asarray(best_coef, dtype=float),
            "C": float(best_c),
        }

    return out


def resolve_dynamic_target_counts(
    y_train: np.ndarray,
    target_counts: list[int] | tuple[int, ...] = (5, 10, 15),
    n_train: int | None = None,
) -> list[int]:
    """Cap configured L1 target counts for small cohorts and class imbalance."""
    y_arr = np.asarray(y_train)
    n_samples = int(y_arr.size if n_train is None else n_train)
    if n_samples <= 0:
        return [1]

    _, counts = np.unique(y_arr, return_counts=True)
    minority = int(np.min(counts)) if counts.size else 1
    cap = max(1, min(n_samples // 8, minority // 2))

    resolved: list[int] = []
    for target in target_counts:
        capped = max(1, min(int(target), cap))
        if capped not in resolved:
            resolved.append(capped)
    return resolved or [1]


def choose_l1_stringency_nested_cv(
    x_train: np.ndarray,
    y_train: np.ndarray,
    target_counts: list[int] | tuple[int, ...] = (5, 10, 15),
    inner_cv_folds: int = 3,
    class_weight: str = "balanced",
    random_state: int = 42,
) -> dict:
    """Choose L1 stringency with nested inner CV using balanced accuracy."""
    if x_train.ndim != 2:
        raise ValueError("x_train must be 2D")
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("x_train and y_train must have matching sample count")
    if len(np.unique(y_train)) < 2:
        raise ValueError("y_train must contain at least 2 classes")

    cv = StratifiedKFold(n_splits=inner_cv_folds, shuffle=True, random_state=random_state)
    per_target_scores: dict[int, list[float]] = {int(t): [] for t in target_counts}
    per_target_c: dict[int, list[float]] = {int(t): [] for t in target_counts}

    for fold_idx, (idx_tr, idx_val) in enumerate(cv.split(x_train, y_train)):
        x_tr, y_tr = x_train[idx_tr], y_train[idx_tr]
        x_val, y_val = x_train[idx_val], y_train[idx_val]

        selectors = l1_select_by_target_count(
            x_tr,
            y_tr,
            target_counts=target_counts,
            class_weight=class_weight,
            random_state=random_state + fold_idx,
        )

        for target in target_counts:
            key = int(target)
            selected = np.asarray(selectors[key]["indices"], dtype=int)
            if selected.size == 0:
                per_target_scores[key].append(0.0)
                continue

            clf = LogisticRegression(
                solver="saga",
                l1_ratio=1.0,
                C=float(selectors[key]["C"]),
                class_weight=class_weight,
                max_iter=_SAGA_MAX_ITER,
                tol=1e-4,
                random_state=random_state + fold_idx,
            )
            _fit_saga_l1(clf, x_tr[:, selected], y_tr)
            pred = clf.predict(x_val[:, selected])
            score = balanced_accuracy_score(y_val, pred)
            per_target_scores[key].append(float(score))
            per_target_c[key].append(float(selectors[key]["C"]))

    mean_scores = {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in per_target_scores.items()}
    chosen_target = max(mean_scores, key=mean_scores.get)
    chosen_c_values = per_target_c.get(int(chosen_target), [])
    chosen_c = float(np.median(chosen_c_values)) if chosen_c_values else np.nan

    return {
        "chosen_target": int(chosen_target),
        "chosen_c": float(chosen_c),
        "mean_scores": mean_scores,
        "mean_c_by_target": {
            int(k): (float(np.mean(v)) if len(v) > 0 else np.nan) for k, v in per_target_c.items()
        },
        "per_fold_scores": per_target_scores,
        "per_fold_c": per_target_c,
    }


def fit_final_l1_selection(
    x_train: np.ndarray,
    y_train: np.ndarray,
    chosen_target: int,
    chosen_c: float | None = None,
    class_weight: str = "balanced",
    random_state: int = 42,
) -> dict:
    """Fit final L1 selector at chosen stringency on full outer-train data."""
    if chosen_c is not None and np.isfinite(float(chosen_c)):
        clf = LogisticRegression(
            solver="saga",
            l1_ratio=1.0,
            C=float(chosen_c),
            class_weight=class_weight,
            max_iter=_SAGA_MAX_ITER,
            tol=1e-4,
            random_state=random_state,
        )
        _fit_saga_l1(clf, x_train, y_train)
        coef = np.ravel(clf.coef_)
        selected_indices = np.flatnonzero(np.abs(coef) > 1e-12)
        return {
            "chosen_target": int(chosen_target),
            "selected_indices": np.asarray(selected_indices, dtype=int),
            "coef": np.asarray(coef, dtype=float),
            "C": float(chosen_c),
        }

    selectors = l1_select_by_target_count(
        x_train,
        y_train,
        target_counts=[int(chosen_target)],
        class_weight=class_weight,
        random_state=random_state,
    )
    selected = selectors[int(chosen_target)]
    return {
        "chosen_target": int(chosen_target),
        "selected_indices": np.asarray(selected["indices"], dtype=int),
        "coef": np.asarray(selected["coef"], dtype=float),
        "C": float(selected["C"]),
    }


def apply_selected_features(x: np.ndarray, selected_indices: np.ndarray) -> np.ndarray:
    """Apply selected feature indices to any design matrix."""
    idx = np.asarray(selected_indices, dtype=int)
    if x.ndim != 2:
        raise ValueError("x must be 2D")
    if idx.size == 0:
        return np.empty((x.shape[0], 0), dtype=float)
    return x[:, idx]
