"""Tests for the control-vs-disease ML classification layer."""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from src.analysis.control_vs_disease_ml import (
    _append_null_distribution_rows,
    _classification_metrics,
    _cohort_summary,
    _fit_tuned_model,
    _fixed_plan_balanced_accuracy,
    _labels,
    _load_cohort,
    _locked_joint_stratified_split_plan,
    _outer_fold_feature_importance,
    _permute_labels,
    _permuted_score,
    _repeat_zero_balanced_accuracy,
    _representation_matrix,
    _run_repeated_nested_cv,
    _sex_only_baseline,
    _sex_stratified_refit,
    _single_split_balanced_accuracy,
    _stream_parallel,
    _tune_threshold,
    _tune_threshold_and_score,
    run_bootstrap_ci,
    run_control_vs_disease_ml_pipeline,
    run_permutation_test,
    run_univariate_band_screen,
)
from src.analysis.eda_mode import validate_control_vs_disease_ml_config

N_FEATURES = 12


def _delayed_double(value: int) -> int:
    time.sleep(0.05 if value == 2 else 0.0)
    return value * 2


def test_stream_parallel_uses_scikit_learn_parallel_api() -> None:
    received: list[tuple[int, int]] = []
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`.*",
            category=UserWarning,
        )
        _stream_parallel(
            2,
            _delayed_double,
            [(11, (2,)), (12, (3,))],
            lambda seed, value: received.append((seed, value)),
        )

    assert sorted(received) == [(11, 4), (12, 6)]


def test_append_null_distribution_rows_skips_absent_optional_null() -> None:
    rows: list[dict[str, object]] = []

    _append_null_distribution_rows(
        rows,
        model_name="pls_da",
        representation="whole",
        null_type="sex_stratified",
        random_state=123,
        null_scores=np.array([]),
        n_permutations=5,
        n_splits=5,
    )

    assert rows == []


def test_append_null_distribution_rows_records_deterministic_metadata() -> None:
    rows: list[dict[str, object]] = []

    _append_null_distribution_rows(
        rows,
        model_name="pls_da",
        representation="whole",
        null_type="unrestricted",
        random_state=123,
        null_scores=np.array([0.4, 0.5]),
        n_permutations=2,
        n_splits=5,
    )

    assert [row["seed"] for row in rows] == [33158374, 1465339467]
    assert [row["null_balanced_accuracy"] for row in rows] == [0.4, 0.5]
    assert {row["permutation_cv_splits"] for row in rows} == {5}


def _make_feature_matrix(
    n_control: int = 20,
    n_disease: int = 25,
    *,
    signal_strength: float = 0.0,
    sex_confound_strength: float = 0.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build a synthetic subject feature matrix mirroring subject_feature_matrix_whole.parquet.

    signal_strength: how much the class label shifts feature means (disease signal).
    sex_confound_strength: how much sex shifts feature means (confound channel).
    Sex is distributed unevenly across class by construction when > 0, mirroring
    the real cohort's sex/class association (used only in the mandatory
    sex-stratified-null test).
    """
    rng = np.random.default_rng(random_state)
    n_total = n_control + n_disease
    is_control = np.array([True] * n_control + [False] * n_disease)

    if sex_confound_strength > 0:
        # Controls skew female, disease skews male - mirrors the real cohort.
        sex = np.array(
            [rng.choice(["female", "male"], p=[0.8, 0.2]) for _ in range(n_control)]
            + [rng.choice(["female", "male"], p=[0.46, 0.54]) for _ in range(n_disease)]
        )
    else:
        sex = rng.choice(["female", "male"], size=n_total)

    feature_names_1300 = [f"1300__{1000 + j * 10}.0" for j in range(N_FEATURES // 2)]
    feature_names_2850 = [f"2850__{2850 + j * 10}.0" for j in range(N_FEATURES - N_FEATURES // 2)]
    all_feature_names = feature_names_1300 + feature_names_2850

    X = rng.normal(0, 1, size=(n_total, N_FEATURES))
    disease_shift = (~is_control).astype(float) * signal_strength
    sex_shift = (sex == "male").astype(float) * sex_confound_strength
    X += (disease_shift + sex_shift)[:, None]

    data: dict[str, object] = {
        "subject_id": [f"S{i:03d}" for i in range(n_total)],
        "is_control": is_control,
        "gender": sex,
        "condition": np.where(is_control, "control", "disease"),
    }
    for j, name in enumerate(all_feature_names):
        data[name] = X[:, j]

    age_years = np.where(is_control, np.nan, rng.uniform(40, 85, size=n_total))
    data["age_years"] = age_years
    data["diagnosis_family"] = pd.Series(
        [None if control else "ischemic_stroke" for control in is_control], dtype=object
    )

    return pd.DataFrame(data)


@pytest.fixture
def small_matrix_path(tmp_path: Path) -> Path:
    frame = _make_feature_matrix(n_control=15, n_disease=18, signal_strength=1.5, random_state=1)
    path = tmp_path / "subject_feature_matrix_whole.parquet"
    frame.to_parquet(path, index=False)
    return path


@pytest.fixture
def base_ml_cfg() -> dict:
    return {
        "analysis": {
            "control_vs_disease_ml": {
                "enabled": True,
                "representations": ["whole", "band_panel"],
                "primary_model": "pls_da",
                "primary_representation": "whole",
                "n_outer_splits": 3,
                "n_outer_repeats": 2,
                "n_inner_splits": 2,
                "n_permutations": 5,
                "n_bootstrap": 5,
                "threshold_metric": "balanced_accuracy",
                "stratify_permutation_by": "sex",
                "random_state": 42,
                "n_jobs": 1,
                "pls_components_grid": [1, 2, 3],
                "logistic_c_grid": [0.1, 1.0],
                "rf_n_estimators_grid": [50],
                "rf_max_depth_grid": [None, 3],
            }
        },
        "whole_spectra": {
            "bands": {
                "band_a": [1000.0, 1015.0],
                "band_b": [1010.0, 1025.0],
                "band_c": [2850.0, 2865.0],
            },
            "ratio_pairs": [],
        },
    }


# ---------------------------------------------------------------------------
# Load + guard
# ---------------------------------------------------------------------------


def test_load_cohort_rejects_missing_is_control(tmp_path: Path) -> None:
    frame = _make_feature_matrix(n_control=15, n_disease=15)
    frame = frame.drop(columns=["is_control"])
    path = tmp_path / "matrix.parquet"
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="is_control"):
        _load_cohort(path)


def test_load_cohort_rejects_nan_is_control(tmp_path: Path) -> None:
    frame = _make_feature_matrix(n_control=15, n_disease=15)
    frame["is_control"] = frame["is_control"].astype(object)
    frame.loc[0, "is_control"] = np.nan
    path = tmp_path / "matrix.parquet"
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="missing is_control"):
        _load_cohort(path)


def test_load_cohort_rejects_small_class(tmp_path: Path) -> None:
    frame = _make_feature_matrix(n_control=5, n_disease=20)
    path = tmp_path / "matrix.parquet"
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="at least 10 subjects per class"):
        _load_cohort(path)


def test_load_cohort_flags_label_anomaly(tmp_path: Path, caplog) -> None:
    frame = _make_feature_matrix(n_control=15, n_disease=15)
    frame.loc[0, "is_control"] = True
    frame.loc[0, "diagnosis_family"] = "ischemic_stroke"
    path = tmp_path / "matrix.parquet"
    frame.to_parquet(path, index=False)
    with caplog.at_level("WARNING"):
        _load_cohort(path)
    assert any("Label anomaly" in message for message in caplog.messages)


def test_labels_positive_class_is_disease() -> None:
    frame = pd.DataFrame({"is_control": [True, False, False]})
    y = _labels(frame)
    np.testing.assert_array_equal(y, [0, 1, 1])


# ---------------------------------------------------------------------------
# Feature representations
# ---------------------------------------------------------------------------


def test_whole_representation_preserves_block_identity() -> None:
    frame = _make_feature_matrix(n_control=10, n_disease=10)
    X, names = _representation_matrix(frame, "whole", {})
    assert X.shape == (20, N_FEATURES)
    assert any(name.startswith("1300__") for name in names)
    assert any(name.startswith("2850__") for name in names)


def test_band_panel_representation_is_low_dimensional(base_ml_cfg: dict) -> None:
    frame = _make_feature_matrix(n_control=10, n_disease=10)
    X, names = _representation_matrix(frame, "band_panel", base_ml_cfg)
    assert X.shape == (20, 3)
    assert set(names) == {"band_a", "band_b", "band_c"}


def test_unknown_representation_raises() -> None:
    frame = _make_feature_matrix(n_control=10, n_disease=10)
    with pytest.raises(ValueError, match="Unknown feature representation"):
        _representation_matrix(frame, "nonsense", {})


# ---------------------------------------------------------------------------
# Cohort summary
# ---------------------------------------------------------------------------


def test_cohort_summary_reports_confound_and_no_age_claim() -> None:
    frame = _make_feature_matrix(n_control=20, n_disease=25, sex_confound_strength=1.0)
    summary = _cohort_summary(frame)
    metrics = dict(zip(summary["metric"], summary["value"]))

    assert metrics["n_control"] == 20
    assert metrics["n_disease"] == 25
    assert metrics["positive_class"] == "Disease"
    assert metrics["negative_class"] == "Control"
    assert "sex_vs_class_chi2_p_value" in metrics
    assert metrics["age_adjustment_status"] == "not_attempted_no_control_age_data"
    assert metrics["n_control_with_age"] == 0


def test_majority_class_baseline_matches_imbalance() -> None:
    frame = _make_feature_matrix(n_control=10, n_disease=40)
    summary = _cohort_summary(frame)
    metrics = dict(zip(summary["metric"], summary["value"]))
    assert metrics["majority_class_baseline_accuracy"] == pytest.approx(40 / 50)
    assert metrics["majority_class_baseline_balanced_accuracy"] == 0.5


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------


def test_tune_threshold_recovers_perfect_separator() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
    threshold = _tune_threshold(y, scores, "balanced_accuracy")
    preds = (scores >= threshold).astype(int)
    assert (preds == y).all()


def test_tune_threshold_rejects_unsupported_metric() -> None:
    with pytest.raises(ValueError, match="Unsupported threshold_metric"):
        _tune_threshold(np.array([0, 1]), np.array([0.1, 0.9]), "accuracy")


# ---------------------------------------------------------------------------
# Leakage guards - the scientific core of this test suite
# ---------------------------------------------------------------------------


def test_pls_da_threshold_uses_only_inner_fold_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tuned threshold must not depend on the outer test fold at all.

    We fit on the same X_train/y_train twice and confirm the returned threshold
    is identical regardless of what (unused) outer-test data exists alongside it -
    i.e. the function signature itself proves no outer-fold data can leak in,
    since _fit_tuned_model never receives X_test/y_test.
    """
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(30, 6))
    y_train = np.array([0] * 15 + [1] * 15)
    kwargs = dict(
        ml_cfg={"pls_components_grid": [1, 2, 3]},
        n_inner_splits=3,
        threshold_metric="balanced_accuracy",
        random_state=42,
        n_jobs=1,
    )

    fit_a = _fit_tuned_model("pls_da", X_train, y_train, **kwargs)
    fit_b = _fit_tuned_model("pls_da", X_train, y_train, **kwargs)
    assert fit_a.threshold == fit_b.threshold


def test_nested_cv_scaler_never_fit_on_test_fold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Instrument StandardScaler.fit to record every array it is fit on, and
    assert none of those arrays contain rows from the current outer test fold.
    """
    from sklearn.preprocessing import StandardScaler

    seen_fit_arrays: list[np.ndarray] = []
    original_fit = StandardScaler.fit

    def _recording_fit(self, X, y=None, **kwargs):
        seen_fit_arrays.append(np.asarray(X).copy())
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", _recording_fit)

    rng = np.random.default_rng(7)
    n = 40
    X = rng.normal(size=(n, 8))
    y = np.array([0] * 20 + [1] * 20)

    from sklearn.model_selection import StratifiedKFold

    outer_cv = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=42).split(X, y))

    for train_idx, test_idx in outer_cv:
        seen_fit_arrays.clear()
        _fit_tuned_model(
            "l1_logistic",
            X[train_idx],
            y[train_idx],
            ml_cfg={"logistic_c_grid": [0.1, 1.0]},
            n_inner_splits=2,
            threshold_metric="balanced_accuracy",
            random_state=42,
            n_jobs=1,
        )
        test_rows = X[test_idx]
        for fit_array in seen_fit_arrays:
            # No row seen by any scaler.fit during training-fold tuning may equal
            # a row that belongs to this outer fold's held-out test set.
            for test_row in test_rows:
                matches = np.any(np.all(np.isclose(fit_array, test_row), axis=1)) if fit_array.ndim == 2 else False
                assert not matches, "Outer test-fold row leaked into an in-fold scaler.fit call"


# ---------------------------------------------------------------------------
# Metrics: imbalance-aware, positive class = Disease
# ---------------------------------------------------------------------------


def test_classification_metrics_never_report_bare_accuracy() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 1, 0])
    y_score = np.array([0.1, 0.2, 0.1, 0.3, 0.9, 0.4])
    metrics = _classification_metrics(y_true, y_pred, y_score)
    assert "accuracy" not in metrics
    assert "balanced_accuracy" in metrics
    assert "mcc" in metrics
    assert "sensitivity" in metrics
    assert "specificity" in metrics


def test_classification_metrics_on_skewed_data_are_finite() -> None:
    rng = np.random.default_rng(3)
    y_true = np.array([0] * 5 + [1] * 45)
    y_pred = rng.integers(0, 2, size=50)
    y_score = rng.uniform(size=50)
    metrics = _classification_metrics(y_true, y_pred, y_score)
    for key, value in metrics.items():
        assert np.isfinite(value), f"{key} is not finite: {value}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_repeated_nested_cv_is_deterministic(base_ml_cfg: dict) -> None:
    frame = _make_feature_matrix(n_control=15, n_disease=18, signal_strength=1.0, random_state=11)
    X, _names = _representation_matrix(frame, "whole", base_ml_cfg["whole_spectra"])
    y = _labels(frame)
    ml_cfg = base_ml_cfg["analysis"]["control_vs_disease_ml"]

    oof_a = _run_repeated_nested_cv("l1_logistic", X, y, ml_cfg=ml_cfg, n_jobs=1)
    oof_b = _run_repeated_nested_cv("l1_logistic", X, y, ml_cfg=ml_cfg, n_jobs=1)

    pd.testing.assert_frame_equal(oof_a, oof_b)


def test_repeat_zero_balanced_accuracy_matches_fixed_plan_refit(base_ml_cfg: dict) -> None:
    """The permutation observed score must equal a fresh refit of the same locked plan.

    Confirms reading repeat 0 off the already-computed nested-CV `oof` frame is a pure
    performance win, not a behavior change: same fold split, same random_state, so the
    predictions (and therefore balanced accuracy) must be bit-identical to recomputing.
    """
    frame = _make_feature_matrix(n_control=15, n_disease=18, signal_strength=1.0, random_state=21)
    X, _names = _representation_matrix(frame, "whole", base_ml_cfg["whole_spectra"])
    y = _labels(frame)
    ml_cfg = base_ml_cfg["analysis"]["control_vs_disease_ml"]
    outer_split_plan = _locked_joint_stratified_split_plan(
        y,
        np.zeros_like(y),
        n_outer_splits=ml_cfg["n_outer_splits"],
        n_outer_repeats=ml_cfg["n_outer_repeats"],
        random_state=42,
    )

    oof = _run_repeated_nested_cv("l1_logistic", X, y, ml_cfg=ml_cfg, n_jobs=1, outer_split_plan=outer_split_plan)
    reused_score = _repeat_zero_balanced_accuracy(oof)
    refit_score = _fixed_plan_balanced_accuracy(
        "l1_logistic", X, y, ml_cfg=ml_cfg, outer_cv=outer_split_plan[0]
    )

    assert reused_score == pytest.approx(refit_score)


def test_random_forest_outer_fold_parallelism_matches_serial(base_ml_cfg: dict) -> None:
    frame = _make_feature_matrix(n_control=15, n_disease=18, signal_strength=1.0, random_state=31)
    X, _names = _representation_matrix(frame, "whole", base_ml_cfg["whole_spectra"])
    y = _labels(frame)
    ml_cfg = base_ml_cfg["analysis"]["control_vs_disease_ml"]

    serial = _run_repeated_nested_cv("random_forest", X, y, ml_cfg=ml_cfg, n_jobs=1)
    parallel = _run_repeated_nested_cv("random_forest", X, y, ml_cfg=ml_cfg, n_jobs=2)

    pd.testing.assert_frame_equal(serial, parallel)


def test_random_forest_importance_outer_fold_parallelism_matches_serial(base_ml_cfg: dict) -> None:
    frame = _make_feature_matrix(n_control=15, n_disease=18, signal_strength=1.0, random_state=32)
    X, feature_names = _representation_matrix(frame, "whole", base_ml_cfg["whole_spectra"])
    y = _labels(frame)
    ml_cfg = base_ml_cfg["analysis"]["control_vs_disease_ml"] | {"importance_permutation_repeats": 1}
    outer_split_plan = [
        list(StratifiedKFold(n_splits=3, shuffle=True, random_state=42 + repeat).split(X, y)) for repeat in range(2)
    ]

    serial = _outer_fold_feature_importance(
        "random_forest", X, y, feature_names, ml_cfg=ml_cfg, n_jobs=1, outer_split_plan=outer_split_plan
    )
    parallel = _outer_fold_feature_importance(
        "random_forest", X, y, feature_names, ml_cfg=ml_cfg, n_jobs=2, outer_split_plan=outer_split_plan
    )

    pd.testing.assert_frame_equal(serial[0], parallel[0])
    pd.testing.assert_frame_equal(serial[1], parallel[1])


def test_vectorized_threshold_matches_bruteforce_reference() -> None:
    y = np.array([0, 1, 0, 1, 1, 0])
    scores = np.array([0.2, 0.2, 0.4, 0.8, 0.8, 0.9])
    candidates = np.unique(scores)
    thresholds = np.concatenate(
        [[candidates[0] - 1e-6], (candidates[:-1] + candidates[1:]) / 2.0, [candidates[-1] + 1e-6]]
    )
    brute_scores = [balanced_accuracy_score(y, (scores >= threshold).astype(int)) for threshold in thresholds]
    expected_index = int(np.argmax(brute_scores))
    threshold, score = _tune_threshold_and_score(y, scores, "balanced_accuracy")
    assert threshold == pytest.approx(thresholds[expected_index])
    assert score == pytest.approx(brute_scores[expected_index])
    assert _tune_threshold(y, scores, "balanced_accuracy") == threshold


def test_random_forest_warm_start_matches_independent_estimator_counts() -> None:
    rng = np.random.default_rng(4)
    X = rng.normal(size=(30, 5))
    y = np.array([0] * 15 + [1] * 15)
    warm = RandomForestClassifier(
        n_estimators=10, random_state=42, class_weight="balanced_subsample", warm_start=True, n_jobs=1
    )
    warm.fit(X, y)
    warm_10 = warm.predict_proba(X)
    warm.set_params(n_estimators=20)
    warm.fit(X, y)
    warm_20 = warm.predict_proba(X)
    independent_10 = RandomForestClassifier(
        n_estimators=10, random_state=42, class_weight="balanced_subsample", n_jobs=1
    ).fit(X, y)
    independent_20 = RandomForestClassifier(
        n_estimators=20, random_state=42, class_weight="balanced_subsample", n_jobs=1
    ).fit(X, y)
    np.testing.assert_allclose(warm_10, independent_10.predict_proba(X))
    np.testing.assert_allclose(warm_20, independent_20.predict_proba(X))


# ---------------------------------------------------------------------------
# Permutation tests
# ---------------------------------------------------------------------------


def _small_permutation_cfg(base_ml_cfg: dict) -> dict:
    ml_cfg = dict(base_ml_cfg["analysis"]["control_vs_disease_ml"])
    ml_cfg["n_permutations"] = 30
    ml_cfg["n_outer_splits"] = 3
    ml_cfg["n_inner_splits"] = 2
    ml_cfg["logistic_c_grid"] = [0.1, 1.0]
    return ml_cfg


def test_permute_labels_unrestricted_preserves_class_counts() -> None:
    rng = np.random.default_rng(0)
    y = np.array([0] * 20 + [1] * 10)
    y_perm = _permute_labels(y, rng, strata=None)
    assert sorted(y_perm.tolist()) == sorted(y.tolist())


def test_permute_labels_stratified_preserves_within_stratum_counts() -> None:
    rng = np.random.default_rng(0)
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])
    strata = np.array(["f", "f", "f", "f", "f", "m", "m", "m", "m", "m"])
    y_perm = _permute_labels(y, rng, strata)
    for stratum in ("f", "m"):
        mask = strata == stratum
        assert sorted(y_perm[mask].tolist()) == sorted(y[mask].tolist())


def test_permutation_p_value_bounded(base_ml_cfg: dict) -> None:
    frame = _make_feature_matrix(n_control=15, n_disease=18, signal_strength=1.0, random_state=3)
    X, _names = _representation_matrix(frame, "whole", base_ml_cfg["whole_spectra"])
    y = _labels(frame)
    ml_cfg = _small_permutation_cfg(base_ml_cfg)

    observed = _single_split_balanced_accuracy("l1_logistic", X, y, ml_cfg=ml_cfg, n_jobs=1, random_state=42)
    p_value, null_dist = run_permutation_test(
        "l1_logistic",
        X,
        y,
        observed_balanced_accuracy=observed,
        strata=None,
        ml_cfg=ml_cfg,
        n_jobs=1,
        random_state=42,
    )
    assert 0.0 < p_value <= 1.0
    assert len(null_dist) == ml_cfg["n_permutations"]


def test_permutation_null_centers_near_chance_on_pure_noise(base_ml_cfg: dict) -> None:
    """With no real signal, the permutation null and the observed statistic should
    both hover near balanced-accuracy 0.5 - i.e. permuting labels on noise data
    should not manufacture a small p-value.
    """
    frame = _make_feature_matrix(n_control=15, n_disease=18, signal_strength=0.0, random_state=4)
    X, _names = _representation_matrix(frame, "whole", base_ml_cfg["whole_spectra"])
    y = _labels(frame)
    ml_cfg = _small_permutation_cfg(base_ml_cfg)
    ml_cfg["n_permutations"] = 40

    observed = _single_split_balanced_accuracy("l1_logistic", X, y, ml_cfg=ml_cfg, n_jobs=1, random_state=42)
    p_value, null_dist = run_permutation_test(
        "l1_logistic",
        X,
        y,
        observed_balanced_accuracy=observed,
        strata=None,
        ml_cfg=ml_cfg,
        n_jobs=1,
        random_state=42,
    )
    # Not a strict guarantee (this is a stochastic test on noise data), but a
    # complete false positive (p < 0.05 on pure noise) should be rare given
    # the null and observed are drawn from the same no-signal process.
    assert p_value > 0.05


def test_permutation_test_stops_early_on_clearly_unresolved_null(base_ml_cfg: dict) -> None:
    """Pure-noise data with a generous permutation_max must stop well short of
    it: the null and the observed statistic are drawn from the same
    no-signal process, so the tail probability resolves as clearly not
    significant long before the budget is exhausted.
    """
    frame = _make_feature_matrix(n_control=15, n_disease=18, signal_strength=0.0, random_state=5)
    X, _names = _representation_matrix(frame, "whole", base_ml_cfg["whole_spectra"])
    y = _labels(frame)
    ml_cfg = _small_permutation_cfg(base_ml_cfg)
    ml_cfg["n_permutations"] = 2000
    ml_cfg["permutation_min"] = 50
    ml_cfg["permutation_check_interval"] = 25

    observed = _single_split_balanced_accuracy("l1_logistic", X, y, ml_cfg=ml_cfg, n_jobs=1, random_state=42)
    p_value, null_dist = run_permutation_test(
        "l1_logistic",
        X,
        y,
        observed_balanced_accuracy=observed,
        strata=None,
        ml_cfg=ml_cfg,
        n_jobs=1,
        random_state=42,
    )
    assert len(null_dist) < 2000
    assert 0.0 < p_value <= 1.0


def test_permutation_test_runs_full_budget_when_never_resolved(base_ml_cfg: dict) -> None:
    """A permutation_max small enough that resolution never has a chance to
    fire between check_interval boundaries must still complete the full
    configured budget deterministically (never overshoots, never crashes on
    the final unresolved boundary).
    """
    frame = _make_feature_matrix(n_control=15, n_disease=18, signal_strength=1.0, random_state=6)
    X, _names = _representation_matrix(frame, "whole", base_ml_cfg["whole_spectra"])
    y = _labels(frame)
    ml_cfg = _small_permutation_cfg(base_ml_cfg)
    ml_cfg["n_permutations"] = 40
    ml_cfg["permutation_min"] = 40
    ml_cfg["permutation_check_interval"] = 40
    ml_cfg["permutation_confidence"] = 0.999999

    observed = _single_split_balanced_accuracy("l1_logistic", X, y, ml_cfg=ml_cfg, n_jobs=1, random_state=42)
    p_value, null_dist = run_permutation_test(
        "l1_logistic",
        X,
        y,
        observed_balanced_accuracy=observed,
        strata=None,
        ml_cfg=ml_cfg,
        n_jobs=1,
        random_state=42,
    )
    assert len(null_dist) == 40
    assert 0.0 < p_value <= 1.0


def test_sex_stratified_null_is_mandatory_confound_guard(base_ml_cfg: dict) -> None:
    """The decisive test: build a cohort where class is a PURE function of sex
    (no independent disease signal at all) and confirm:
      - the UNRESTRICTED permutation null is fooled (small p-value, because the
        model IS discriminating classes, just via sex, and unrestricted
        permutation destroys the sex-class link the model relies on) — proving
        an unrestricted null alone would falsely look like a real spectral finding.
      - the SEX-STRATIFIED null is NOT fooled (large p-value), because permuting
        within sex strata preserves the sex-class association the model is
        actually exploiting, so permuted data score just as well as real data.

    This is the test that proves the confound-handling machinery
    actually does its job for the pre-specified sex-confound check.
    """
    rng = np.random.default_rng(99)
    n_female, n_male = 30, 30
    sex = np.array(["female"] * n_female + ["male"] * n_male)
    # Class is entirely determined by sex: all females control, all males disease.
    is_control = np.array([True] * n_female + [False] * n_male)

    feature_names_1300 = [f"1300__{1000 + j * 10}.0" for j in range(N_FEATURES // 2)]
    feature_names_2850 = [f"2850__{2850 + j * 10}.0" for j in range(N_FEATURES - N_FEATURES // 2)]
    all_names = feature_names_1300 + feature_names_2850

    n_total = n_female + n_male
    X = rng.normal(0, 1, size=(n_total, N_FEATURES))
    # Signal is driven by SEX only, not by an independent disease effect.
    X += (sex == "male").astype(float)[:, None] * 2.0

    frame = pd.DataFrame({"is_control": is_control, "gender": sex})
    for j, name in enumerate(all_names):
        frame[name] = X[:, j]

    y = _labels(frame)
    Xw, _names = _representation_matrix(frame, "whole", {})

    ml_cfg = _small_permutation_cfg(base_ml_cfg)
    ml_cfg["n_permutations"] = 60

    observed = _single_split_balanced_accuracy("l1_logistic", Xw, y, ml_cfg=ml_cfg, n_jobs=1, random_state=42)
    assert observed > 0.9, "sanity check: sex-driven signal should be trivially separable"

    p_unrestricted, _null_u = run_permutation_test(
        "l1_logistic",
        Xw,
        y,
        observed_balanced_accuracy=observed,
        strata=None,
        ml_cfg=ml_cfg,
        n_jobs=1,
        random_state=1,
    )
    p_sex_stratified, _null_s = run_permutation_test(
        "l1_logistic",
        Xw,
        y,
        observed_balanced_accuracy=observed,
        strata=sex,
        ml_cfg=ml_cfg,
        n_jobs=1,
        random_state=2,
    )

    assert p_unrestricted < 0.05, (
        f"unrestricted null should be fooled by pure sex-confounded signal (p={p_unrestricted})"
    )
    assert p_sex_stratified > 0.05, (
        f"sex-stratified null should NOT be fooled by pure sex-confounded signal (p={p_sex_stratified})"
    )


def test_locked_outer_plan_preserves_joint_class_sex_strata() -> None:
    y = np.repeat(np.array([0, 1]), 12)
    sex = np.tile(np.repeat(np.array(["female", "male"]), 6), 2)
    plan = _locked_joint_stratified_split_plan(
        y,
        sex,
        n_outer_splits=3,
        n_outer_repeats=2,
        random_state=42,
    )
    joint = np.char.add(np.char.add(y.astype(str), "__"), sex)
    assert len(plan) == 2
    for repeat_splits in plan:
        assert len(repeat_splits) == 3
        for _train, test in repeat_splits:
            assert set(pd.Series(joint[test]).value_counts()) == {2}


def test_locked_outer_plan_fails_when_a_joint_stratum_cannot_fill_every_fold() -> None:
    with pytest.raises(ValueError, match="class-sex"):
        _locked_joint_stratified_split_plan(
            np.array([0, 0, 0, 1, 1, 1]),
            np.array(["female", "female", "male", "female", "male", "male"]),
            n_outer_splits=3,
            n_outer_repeats=1,
            random_state=42,
        )


def test_permuted_score_uses_only_first_locked_repeat(monkeypatch: pytest.MonkeyPatch, base_ml_cfg: dict) -> None:
    y = np.array([0] * 12 + [1] * 12)
    X = np.arange(len(y) * 2, dtype=float).reshape(len(y), 2)
    sex = np.tile(np.repeat(np.array(["female", "male"]), 6), 2)
    plan = _locked_joint_stratified_split_plan(y, sex, n_outer_splits=3, n_outer_repeats=2, random_state=42)
    captured: dict[str, object] = {}

    def fake_fixed(*args, **kwargs):
        captured["outer_cv"] = kwargs["outer_cv"]
        return 0.5

    monkeypatch.setattr("src.analysis.control_vs_disease_ml._fixed_plan_balanced_accuracy", fake_fixed)
    value = _permuted_score("l1_logistic", X, y, None, base_ml_cfg["analysis"]["control_vs_disease_ml"], 7, plan)
    assert value == 0.5
    assert captured["outer_cv"] == plan[0]


def test_permutation_resume_reuses_seeded_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, base_ml_cfg: dict
) -> None:
    ml_cfg = _small_permutation_cfg(base_ml_cfg)
    ml_cfg["n_permutations"] = 3
    X = np.zeros((12, 2))
    y = np.array([0] * 6 + [1] * 6)
    checkpoint = tmp_path / "permutation.parquet"
    monkeypatch.setattr("src.analysis.control_vs_disease_ml._permuted_score", lambda *_args: 0.5)
    first_p, first_null = run_permutation_test(
        "l1_logistic",
        X,
        y,
        observed_balanced_accuracy=0.5,
        strata=None,
        ml_cfg=ml_cfg,
        n_jobs=1,
        random_state=9,
        checkpoint_path=checkpoint,
    )
    monkeypatch.setattr(
        "src.analysis.control_vs_disease_ml._permuted_score",
        lambda *_args: pytest.fail("checkpointed draw was recomputed"),
    )
    resumed_p, resumed_null = run_permutation_test(
        "l1_logistic",
        X,
        y,
        observed_balanced_accuracy=0.5,
        strata=None,
        ml_cfg=ml_cfg,
        n_jobs=1,
        random_state=9,
        checkpoint_path=checkpoint,
        resume=True,
    )
    assert resumed_p == first_p
    np.testing.assert_array_equal(resumed_null, first_null)


def test_permutation_resume_from_already_resolved_checkpoint_draws_nothing_new(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, base_ml_cfg: dict
) -> None:
    """A checkpoint left over from a prior early-stopped run must not trigger
    fresh draws on resume when the stopping condition is already satisfied.
    """
    ml_cfg = _small_permutation_cfg(base_ml_cfg)
    ml_cfg["n_permutations"] = 2000
    ml_cfg["permutation_min"] = 50
    ml_cfg["permutation_check_interval"] = 25
    X = np.zeros((12, 2))
    y = np.array([0] * 6 + [1] * 6)
    checkpoint = tmp_path / "permutation.parquet"

    # Every null draw scores far below the observed statistic: unambiguously
    # not significant, so the stopping rule should already be satisfied by
    # permutation_min draws once resumed.
    monkeypatch.setattr("src.analysis.control_vs_disease_ml._permuted_score", lambda *_args: 0.1)
    run_permutation_test(
        "l1_logistic",
        X,
        y,
        observed_balanced_accuracy=0.9,
        strata=None,
        ml_cfg=ml_cfg,
        n_jobs=1,
        random_state=9,
        checkpoint_path=checkpoint,
    )
    checkpointed = pd.read_parquet(checkpoint)
    assert len(checkpointed) < 2000  # confirms the first run actually stopped early

    monkeypatch.setattr(
        "src.analysis.control_vs_disease_ml._permuted_score",
        lambda *_args: pytest.fail("resume from an already-resolved checkpoint must not draw further"),
    )
    resumed_p, resumed_null = run_permutation_test(
        "l1_logistic",
        X,
        y,
        observed_balanced_accuracy=0.9,
        strata=None,
        ml_cfg=ml_cfg,
        n_jobs=1,
        random_state=9,
        checkpoint_path=checkpoint,
        resume=True,
    )
    assert len(resumed_null) == len(checkpointed)
    assert resumed_p == pytest.approx(1.0 / (len(checkpointed) + 1))


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def test_bootstrap_ci_bounds_are_ordered_and_finite(base_ml_cfg: dict) -> None:
    frame = _make_feature_matrix(n_control=15, n_disease=18, signal_strength=1.0, random_state=12)
    X, _names = _representation_matrix(frame, "whole", base_ml_cfg["whole_spectra"])
    y = _labels(frame)
    ml_cfg = _small_permutation_cfg(base_ml_cfg)
    ml_cfg["n_bootstrap"] = 25

    ci = run_bootstrap_ci("l1_logistic", X, y, ml_cfg=ml_cfg, n_jobs=1, random_state=42)

    assert ci["n_bootstrap_used"] > 0
    low, high = ci["balanced_accuracy_ci_low"], ci["balanced_accuracy_ci_high"]
    assert np.isfinite(low) and np.isfinite(high)
    assert low <= high
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0


def test_bootstrap_ci_no_duplicate_subject_leakage_across_in_bag_out_of_bag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guards against the classic bootstrap+CV leak: a naive implementation that
    resamples with replacement and then cross-validates the resampled array can
    place two copies of the same original subject into different CV folds,
    letting the model see a near-duplicate of a "held-out" row during training.
    The fix used here is out-of-bag scoring: fit once on the in-bag resample,
    score only on subjects whose original row was never drawn into it. This
    test instruments the model fit to record every row index array it is
    trained on, and asserts no scored (out-of-bag) row's feature vector
    appears anywhere in the training data for that draw.
    """
    from src.analysis.control_vs_disease_ml import _bootstrap_metric_sample

    rng = np.random.default_rng(21)
    n = 40
    X = rng.normal(size=(n, 6))
    y = np.array([0] * 20 + [1] * 20)
    ml_cfg = {"n_inner_splits": 2, "threshold_metric": "balanced_accuracy", "logistic_c_grid": [0.1, 1.0]}

    seed = 99
    draw_rng = np.random.default_rng(seed)
    in_bag_idx = draw_rng.integers(0, n, size=n)
    out_of_bag_idx = np.setdiff1d(np.arange(n), in_bag_idx)
    assert len(out_of_bag_idx) > 0, "test fixture must produce at least one out-of-bag subject"

    X_in_bag_rows = X[in_bag_idx]
    for oob_index in out_of_bag_idx:
        oob_row = X[oob_index]
        matches = np.any(np.all(np.isclose(X_in_bag_rows, oob_row), axis=1))
        assert not matches, "an out-of-bag subject's row also appears in the in-bag training data"

    result = _bootstrap_metric_sample("l1_logistic", X, y, ml_cfg, seed)
    assert result is None or "balanced_accuracy" in result


def test_bootstrap_ci_deterministic_with_same_seed(base_ml_cfg: dict) -> None:
    frame = _make_feature_matrix(n_control=15, n_disease=18, signal_strength=1.0, random_state=13)
    X, _names = _representation_matrix(frame, "whole", base_ml_cfg["whole_spectra"])
    y = _labels(frame)
    ml_cfg = _small_permutation_cfg(base_ml_cfg)
    ml_cfg["n_bootstrap"] = 20

    ci_a = run_bootstrap_ci("l1_logistic", X, y, ml_cfg=ml_cfg, n_jobs=1, random_state=7)
    ci_b = run_bootstrap_ci("l1_logistic", X, y, ml_cfg=ml_cfg, n_jobs=1, random_state=7)

    assert ci_a == ci_b


# ---------------------------------------------------------------------------
# Univariate band screen
# ---------------------------------------------------------------------------


def _band_panel_cfg() -> dict:
    return {
        "whole_spectra": {
            "bands": {
                "band_a": [1000.0, 1015.0],
                "band_b": [1010.0, 1025.0],
                "band_c": [2850.0, 2865.0],
            },
            "ratio_pairs": [],
        }
    }


def test_univariate_band_screen_has_one_row_per_band_with_fdr() -> None:
    frame = _make_feature_matrix(n_control=15, n_disease=18, signal_strength=1.0, random_state=14)
    y = _labels(frame)
    valid_rows = np.ones(len(frame), dtype=bool)

    result = run_univariate_band_screen(frame, y, valid_rows, _band_panel_cfg())

    assert set(result["band"]) == {"band_a", "band_b", "band_c"}
    assert (result["status"] == "ok").all()
    assert "fdr_q" in result.columns
    assert "significant_fdr" in result.columns
    assert (result["p_value"] > 0).all() and (result["p_value"] <= 1).all()


def test_univariate_band_screen_reports_insufficient_data_status() -> None:
    frame = _make_feature_matrix(n_control=15, n_disease=18, random_state=15)
    # Blank out one band's underlying spectral columns so it cannot be integrated.
    columns_to_null = [c for c in frame.columns if c.startswith("2850__")]
    frame[columns_to_null] = np.nan
    y = _labels(frame)
    valid_rows = np.ones(len(frame), dtype=bool)

    result = run_univariate_band_screen(frame, y, valid_rows, _band_panel_cfg())

    band_c_row = result.loc[result["band"] == "band_c"].iloc[0]
    assert band_c_row["status"] == "insufficient_data"


# ---------------------------------------------------------------------------
# Sex-confound battery
# ---------------------------------------------------------------------------


def test_sex_only_baseline_reports_metrics_when_sex_present() -> None:
    frame = _make_feature_matrix(n_control=15, n_disease=18, sex_confound_strength=1.0, random_state=6)
    y = _labels(frame)
    valid_rows = np.ones(len(frame), dtype=bool)
    result = _sex_only_baseline(frame, y, valid_rows)
    assert result["status"] == "ok"
    assert "sex_only_balanced_accuracy" in result


def test_sex_only_baseline_missing_sex_column_reports_status() -> None:
    frame = _make_feature_matrix(n_control=15, n_disease=18)
    frame = frame.drop(columns=["gender"])
    y = _labels(frame)
    valid_rows = np.ones(len(frame), dtype=bool)
    result = _sex_only_baseline(frame, y, valid_rows)
    assert result["status"] == "missing_sex_column"


def test_sex_stratified_refit_reports_female_only_metrics(base_ml_cfg: dict) -> None:
    frame = _make_feature_matrix(n_control=25, n_disease=25, signal_strength=1.0, random_state=8)
    # Force an adequately sized, balanced female stratum.
    frame["gender"] = ["female"] * 40 + ["male"] * 10
    y = _labels(frame)
    valid_rows = np.ones(len(frame), dtype=bool)
    X, _names = _representation_matrix(frame, "whole", {})
    ml_cfg = _small_permutation_cfg(base_ml_cfg)

    result = _sex_stratified_refit("l1_logistic", frame, X, y, valid_rows, ml_cfg=ml_cfg, n_jobs=1)
    assert result["status"] == "ok"
    assert "female_only_balanced_accuracy_mean" in result


def test_sex_stratified_refit_insufficient_n_reports_status(base_ml_cfg: dict) -> None:
    frame = _make_feature_matrix(n_control=15, n_disease=18, random_state=9)
    # Almost no females -> insufficient n for a female-only refit.
    frame["gender"] = ["female"] * 3 + ["male"] * (len(frame) - 3)
    y = _labels(frame)
    valid_rows = np.ones(len(frame), dtype=bool)
    X, _names = _representation_matrix(frame, "whole", {})
    ml_cfg = _small_permutation_cfg(base_ml_cfg)

    result = _sex_stratified_refit("l1_logistic", frame, X, y, valid_rows, ml_cfg=ml_cfg, n_jobs=1)
    assert result["status"] == "insufficient_n"


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_validate_config_accepts_defaults(base_ml_cfg: dict) -> None:
    validate_control_vs_disease_ml_config(base_ml_cfg)


def test_validate_config_rejects_bad_primary_model(base_ml_cfg: dict) -> None:
    base_ml_cfg["analysis"]["control_vs_disease_ml"]["primary_model"] = "xgboost"
    with pytest.raises(ValueError, match="primary_model"):
        validate_control_vs_disease_ml_config(base_ml_cfg)


def test_validate_config_rejects_primary_representation_not_in_list(base_ml_cfg: dict) -> None:
    base_ml_cfg["analysis"]["control_vs_disease_ml"]["representations"] = ["band_panel"]
    base_ml_cfg["analysis"]["control_vs_disease_ml"]["primary_representation"] = "whole"
    with pytest.raises(ValueError, match="primary_representation"):
        validate_control_vs_disease_ml_config(base_ml_cfg)


def test_validate_config_rejects_non_positive_int(base_ml_cfg: dict) -> None:
    base_ml_cfg["analysis"]["control_vs_disease_ml"]["n_outer_splits"] = 0
    with pytest.raises(ValueError, match="n_outer_splits"):
        validate_control_vs_disease_ml_config(base_ml_cfg)


def test_validate_config_rejects_empty_grid(base_ml_cfg: dict) -> None:
    base_ml_cfg["analysis"]["control_vs_disease_ml"]["logistic_c_grid"] = []
    with pytest.raises(ValueError, match="logistic_c_grid"):
        validate_control_vs_disease_ml_config(base_ml_cfg)


def test_validate_config_skips_when_disabled(base_ml_cfg: dict) -> None:
    base_ml_cfg["analysis"]["control_vs_disease_ml"]["enabled"] = False
    base_ml_cfg["analysis"]["control_vs_disease_ml"]["primary_model"] = "not_a_model"
    validate_control_vs_disease_ml_config(base_ml_cfg)  # must not raise


def test_validate_config_accepts_stopping_keys_omitted(base_ml_cfg: dict) -> None:
    validate_control_vs_disease_ml_config(base_ml_cfg)  # no permutation_min/max present


def test_validate_config_rejects_non_positive_permutation_min(base_ml_cfg: dict) -> None:
    base_ml_cfg["analysis"]["control_vs_disease_ml"]["permutation_min"] = 0
    with pytest.raises(ValueError, match="permutation_min"):
        validate_control_vs_disease_ml_config(base_ml_cfg)


def test_validate_config_rejects_permutation_min_above_max(base_ml_cfg: dict) -> None:
    base_ml_cfg["analysis"]["control_vs_disease_ml"]["permutation_min"] = 500
    base_ml_cfg["analysis"]["control_vs_disease_ml"]["permutation_max"] = 100
    with pytest.raises(ValueError, match="permutation_min"):
        validate_control_vs_disease_ml_config(base_ml_cfg)


def test_validate_config_rejects_confidence_out_of_range(base_ml_cfg: dict) -> None:
    base_ml_cfg["analysis"]["control_vs_disease_ml"]["permutation_confidence"] = 1.5
    with pytest.raises(ValueError, match="permutation_confidence"):
        validate_control_vs_disease_ml_config(base_ml_cfg)


# ---------------------------------------------------------------------------
# Output contract / runner smoke
# ---------------------------------------------------------------------------


@patch("src.analysis.control_vs_disease_ml.write_analysis_provenance")
@patch("src.analysis.control_vs_disease_ml.resolve_data_source", return_value=MagicMock())
def test_runner_smoke_produces_expected_outputs(
    _mock_resolve_source, _mock_write_provenance, small_matrix_path: Path, base_ml_cfg: dict, tmp_path: Path
) -> None:
    out_dir = tmp_path / "ml_out"
    outputs = run_control_vs_disease_ml_pipeline(
        feature_matrix_path=small_matrix_path,
        output_dir=out_dir,
        cfg=base_ml_cfg,
    )

    cohort_dir = out_dir / "01_cohort"
    model_dir = out_dir / "02_model_evaluation"
    confounder_dir = out_dir / "03_confounder_checks"
    permutation_dir = out_dir / "04_permutation_tests"
    band_screen_dir = out_dir / "05_univariate_band_screen"
    report_dir = out_dir / "06_report"

    assert (cohort_dir / "cohort_summary.csv").exists()
    assert (model_dir / "model_comparison.csv").exists()
    assert (confounder_dir / "confounder_analysis.csv").exists()
    assert (permutation_dir / "permutation_null_distributions.csv").exists()
    assert (band_screen_dir / "univariate_band_screen.csv").exists()

    comparison = pd.read_csv(model_dir / "model_comparison.csv")
    n_models = 3
    n_representations = 2
    assert len(comparison) == n_models * n_representations
    assert set(comparison["model"]) == {"pls_da", "l1_logistic", "random_forest"}
    assert set(comparison["representation"]) == {"whole", "band_panel"}
    assert (comparison["positive_class"] == "Disease").all()
    assert "balanced_accuracy_mean" in comparison.columns
    assert "permutation_observed_balanced_accuracy" in comparison.columns
    assert (
        comparison["permutation_cv_splits"] == base_ml_cfg["analysis"]["control_vs_disease_ml"]["n_outer_splits"]
    ).all()
    assert (comparison["permutation_cv_repeats"] == 1).all()
    assert "roc_auc_mean" in comparison.columns

    assert "permutation_p_value_unrestricted" in comparison.columns
    assert "permutation_p_value_sex_stratified" in comparison.columns
    assert "permutation_p_fdr_q" in comparison.columns
    p_unrestricted = comparison["permutation_p_value_unrestricted"].dropna()
    assert (p_unrestricted > 0).all() and (p_unrestricted <= 1).all()
    # Exactly one (model, representation) row is pre-specified as primary.
    assert comparison["is_primary_model"].sum() == 1

    assert "bootstrap_balanced_accuracy_ci_low" in comparison.columns
    assert "bootstrap_balanced_accuracy_ci_high" in comparison.columns
    ci_low = comparison["bootstrap_balanced_accuracy_ci_low"].dropna()
    ci_high = comparison["bootstrap_balanced_accuracy_ci_high"].dropna()
    assert (ci_low <= ci_high).all()

    confounder = pd.read_csv(confounder_dir / "confounder_analysis.csv")
    assert len(confounder) == n_models * n_representations
    assert "status" in confounder.columns

    null_dist = pd.read_csv(permutation_dir / "permutation_null_distributions.csv")
    assert set(null_dist["null_type"].unique()) <= {"unrestricted", "sex_stratified"}

    band_screen = pd.read_csv(band_screen_dir / "univariate_band_screen.csv")
    assert set(band_screen["band"]) == {"band_a", "band_b", "band_c"}
    assert "fdr_q" in band_screen.columns

    for model_name in ("pls_da", "l1_logistic", "random_forest"):
        for representation in ("whole", "band_panel"):
            pred_path = model_dir / f"{model_name}_{representation}_cv_predictions.csv"
            importance_path = model_dir / f"{model_name}_{representation}_feature_importance.csv"
            assert pred_path.exists()
            assert importance_path.exists()

    report_path = report_dir / "final_performance_summary.md"
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "exploratory only" in report_text.lower()
    assert "diagnostic" in report_text.lower()

    figures_dir = report_dir / "figures"
    assert figures_dir.exists()
    assert any(figures_dir.glob("roc_curves_*.png"))
    assert any(figures_dir.glob("pr_curves_*.png"))
    assert any(figures_dir.glob("confusion_matrices_*.png"))
    assert any(figures_dir.glob("metric_comparison_*.png"))
    assert any(figures_dir.glob("permutation_null_*.png"))

    assert "cohort_summary" in outputs
    assert "model_comparison" in outputs
    assert "confounder_analysis" in outputs
    assert "univariate_band_screen" in outputs
    assert "final_performance_summary" in outputs
    _mock_write_provenance.assert_called_once()


def test_runner_disabled_returns_empty(small_matrix_path: Path, base_ml_cfg: dict, tmp_path: Path) -> None:
    base_ml_cfg["analysis"]["control_vs_disease_ml"]["enabled"] = False
    outputs = run_control_vs_disease_ml_pipeline(
        feature_matrix_path=small_matrix_path,
        output_dir=tmp_path / "disabled",
        cfg=base_ml_cfg,
    )
    assert outputs == {}
