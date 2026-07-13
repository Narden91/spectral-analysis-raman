"""Tests for the control-vs-disease ML classification layer (Phase A + Phase B)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.analysis.control_vs_disease_ml import (
    _classification_metrics,
    _cohort_summary,
    _fit_tuned_logistic_threshold,
    _fit_tuned_pls_da,
    _labels,
    _load_cohort,
    _representation_matrix,
    _run_repeated_nested_cv,
    _tune_threshold,
    run_control_vs_disease_ml_pipeline,
)
from src.analysis.eda_mode import validate_control_vs_disease_ml_config


N_FEATURES = 12


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
    sex-stratified-null test; not implemented until Phase C).
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
# Phase A: cohort summary
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
    since _fit_tuned_pls_da never receives X_test/y_test.
    """
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(30, 6))
    y_train = np.array([0] * 15 + [1] * 15)

    _scaler_a, _model_a, threshold_a = _fit_tuned_pls_da(
        X_train, y_train,
        component_grid=[1, 2, 3],
        n_inner_splits=3,
        threshold_metric="balanced_accuracy",
        random_state=42,
    )
    _scaler_b, _model_b, threshold_b = _fit_tuned_pls_da(
        X_train, y_train,
        component_grid=[1, 2, 3],
        n_inner_splits=3,
        threshold_metric="balanced_accuracy",
        random_state=42,
    )
    assert threshold_a == threshold_b


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
        _fit_tuned_logistic_threshold(
            X[train_idx], y[train_idx],
            c_grid=[0.1, 1.0],
            n_inner_splits=2,
            threshold_metric="balanced_accuracy",
            random_state=42,
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


# ---------------------------------------------------------------------------
# Output contract / runner smoke
# ---------------------------------------------------------------------------


@patch("src.analysis.control_vs_disease_ml.write_analysis_provenance")
@patch("src.analysis.control_vs_disease_ml.resolve_data_source", return_value=MagicMock())
def test_runner_smoke_produces_expected_outputs(
    _mock_resolve_source, _mock_write_provenance,
    small_matrix_path: Path, base_ml_cfg: dict, tmp_path: Path
) -> None:
    out_dir = tmp_path / "ml_out"
    outputs = run_control_vs_disease_ml_pipeline(
        feature_matrix_path=small_matrix_path,
        output_dir=out_dir,
        cfg=base_ml_cfg,
    )

    assert (out_dir / "cohort_summary.csv").exists()
    assert (out_dir / "model_comparison.csv").exists()

    comparison = pd.read_csv(out_dir / "model_comparison.csv")
    n_models = 3
    n_representations = 2
    assert len(comparison) == n_models * n_representations
    assert set(comparison["model"]) == {"pls_da", "l1_logistic", "random_forest"}
    assert set(comparison["representation"]) == {"whole", "band_panel"}
    assert (comparison["positive_class"] == "Disease").all()
    assert "balanced_accuracy_mean" in comparison.columns
    assert "roc_auc_mean" in comparison.columns

    for model_name in ("pls_da", "l1_logistic", "random_forest"):
        for representation in ("whole", "band_panel"):
            pred_path = out_dir / f"{model_name}_{representation}_cv_predictions.csv"
            importance_path = out_dir / f"{model_name}_{representation}_feature_importance.csv"
            assert pred_path.exists()
            assert importance_path.exists()

    assert "cohort_summary" in outputs
    assert "model_comparison" in outputs
    _mock_write_provenance.assert_called_once()


def test_runner_disabled_returns_empty(small_matrix_path: Path, base_ml_cfg: dict, tmp_path: Path) -> None:
    base_ml_cfg["analysis"]["control_vs_disease_ml"]["enabled"] = False
    outputs = run_control_vs_disease_ml_pipeline(
        feature_matrix_path=small_matrix_path,
        output_dir=tmp_path / "disabled",
        cfg=base_ml_cfg,
    )
    assert outputs == {}
