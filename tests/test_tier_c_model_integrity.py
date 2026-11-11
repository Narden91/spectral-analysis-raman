"""Engineering regressions for participant isolation and result provenance."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

import src.analysis.control_vs_disease_ml as ml
import src.analysis.pls_regression as pls
from src.analysis.control_vs_disease_ml_report import _build_report_text
from src.analysis.robust_contrasts import _bootstrap_ci, _bootstrap_partial_spearman_ci


def test_inner_folds_keep_all_copies_of_a_participant_together() -> None:
    groups = np.repeat(np.arange(12), [1, 2, 3, 1, 4, 2, 3, 1, 2, 4, 1, 3])
    y = (groups >= 6).astype(int)
    X = np.column_stack([groups, groups**2])
    folds = ml._inner_cv_folds(X, y, 3, 42, groups=groups)
    seen = []
    for train, test in folds:
        assert not set(groups[train]) & set(groups[test])
        assert set(y[train]) == {0, 1}
        assert set(y[test]) == {0, 1}
        seen.extend(test)
    np.testing.assert_array_equal(np.sort(seen), np.arange(len(y)))


def test_inner_folds_reject_conflicting_participant_labels() -> None:
    with pytest.raises(ValueError, match="inconsistent labels"):
        ml._inner_cv_folds(np.ones((4, 1)), np.array([0, 1, 0, 1]), 2, 1, groups=np.array([1, 1, 2, 3]))


def test_bootstrap_passes_original_participant_groups_to_tuning(monkeypatch) -> None:
    X = np.column_stack([np.arange(40), np.sin(np.arange(40))])
    y = (np.arange(40) >= 20).astype(int)
    recorded = []
    original = ml._fit_tuned_model

    def capture(name, X_train, y_train, **kwargs):
        recorded.append(kwargs["groups"])
        np.testing.assert_array_equal(X_train[:, 0], kwargs["groups"])
        return original(name, X_train, y_train, **kwargs)

    monkeypatch.setattr(ml, "_fit_tuned_model", capture)
    result = ml._bootstrap_metric_sample(
        "l1_logistic", X, y, {"n_inner_splits": 2, "logistic_c_grid": [0.1]}, 99,
    )
    assert result is not None
    assert len(recorded) == 1
    assert len(np.unique(recorded[0])) < len(recorded[0])


def test_importance_refit_uses_prediction_fit_seed(monkeypatch) -> None:
    calls = []

    def capture(name, X_train, y_train, **kwargs):
        calls.append((kwargs["random_state"], X_train.copy()))
        return ml._ModelFit(lambda X: np.full(len(X), 0.5), 0.5,
                            SimpleNamespace(coef_=np.ones((1, X_train.shape[1]))), None)

    monkeypatch.setattr(ml, "_fit_tuned_model", capture)
    X = np.arange(24).reshape(12, 2)
    y = np.tile([0, 1], 6)
    train, test = np.arange(8), np.arange(8, 12)
    ml._fit_outer_fold_prediction("l1_logistic", X, y, train, test, {"random_state": 7}, 2, "balanced_accuracy", 9, 1)
    ml._outer_fold_importance_frame(
        "l1_logistic", X, y, ["a", "b"], train, test, ml_cfg={"random_state": 7},
        repeat_index=2, fold_index=4, estimator_n_jobs=1,
    )
    assert [seed for seed, _ in calls] == [9, 9]
    np.testing.assert_array_equal(calls[0][1], calls[1][1])


def test_observed_checkpoint_rejects_modified_or_unrecorded_artifacts(tmp_path) -> None:
    checkpoint = ml._CheckpointStore(tmp_path, {"fixture": "identity"}, True)
    artifact = tmp_path / "observed.csv"
    artifact.write_text("value\n1\n", encoding="utf-8")
    checkpoint.mark_stage_complete("pls_da", "whole", "observed", artifacts=[artifact])
    assert checkpoint.stage_complete("pls_da", "whole", "observed", artifacts=[artifact])
    artifact.write_text("value\n2\n", encoding="utf-8")
    assert not checkpoint.stage_complete("pls_da", "whole", "observed", artifacts=[artifact])
    checkpoint.mark_stage_complete("pls_da", "whole", "observed")
    assert not checkpoint.stage_complete("pls_da", "whole", "observed", artifacts=[artifact])


def test_sex_baseline_reuses_every_supplied_outer_fold(monkeypatch) -> None:
    y = np.tile([0, 1], 8)
    frame = pd.DataFrame({"gender": np.tile(["female", "male", "male", "female"], 4)})
    folds = list(StratifiedKFold(2, shuffle=True, random_state=3).split(np.zeros((16, 1)), y))
    expected = [(frame.iloc[train]["gender"].eq("male").to_numpy(dtype=float), y[train]) for train, _ in folds] * 2
    recorded = []

    def capture(model, X, target, **kwargs):
        recorded.append((X.ravel().copy(), target.copy()))
        model.fit(X, target)

    monkeypatch.setattr(ml, "fit_with_convergence_capture", capture)
    result = ml._sex_only_baseline(frame, y, np.ones(16, dtype=bool), outer_split_plan=[folds, folds])
    assert result["sex_only_cv_repeats"] == 2
    assert len(recorded) == len(expected)
    for actual, wanted in zip(recorded, expected):
        np.testing.assert_array_equal(actual[0], wanted[0])
        np.testing.assert_array_equal(actual[1], wanted[1])


def test_bootstrap_intervals_are_independent_of_worker_count() -> None:
    left, right = np.arange(8, dtype=float), np.arange(10, dtype=float) ** 1.1
    def statistic(a, b):
        return float(np.mean(a) - np.mean(b))
    serial = _bootstrap_ci(left, right, statistic, n_bootstrap=29, random_state=17, n_jobs=1)
    parallel = _bootstrap_ci(left, right, statistic, n_bootstrap=29, random_state=17, n_jobs=2)
    assert serial == parallel
    frame = pd.DataFrame({"x": np.sin(np.arange(20)), "y": np.cos(np.arange(20) * 1.3), "age": np.arange(20)})
    kwargs = {"feature": "x", "endpoint": "y", "covariates": ["age"], "n_bootstrap": 29, "random_state": 18}
    assert _bootstrap_partial_spearman_ci(frame, n_jobs=1, **kwargs) == _bootstrap_partial_spearman_ci(frame, n_jobs=2, **kwargs)


def test_pls_candidates_fit_every_inner_training_fold(monkeypatch) -> None:
    fitted_counts = []

    class RecordingPLS:
        def __init__(self, n_components, **kwargs):
            self.n_components = n_components

        def fit(self, X, y):
            assert self.n_components <= min(len(X) - 1, X.shape[1])
            fitted_counts.append(self.n_components)
            self.mean = float(np.mean(y))
            return self

        def predict(self, X):
            return np.full(len(X), self.mean)

    monkeypatch.setattr(pls, "PLSRegression", RecordingPLS)
    pls._tune_pls_components(np.arange(120).reshape(12, 10), np.arange(12), list(range(1, 10)), 42)
    assert fitted_counts and max(fitted_counts) == 7


def test_report_does_not_invent_confound_or_missing_age_findings() -> None:
    cohort = pd.DataFrame({"metric": ["n_subjects_total", "n_control", "n_disease", "n_control_with_age", "n_disease_with_age", "sex_vs_class_chi2_p_value"],
                           "value": [12, 5, 7, 4, 7, 0.9]})
    row = {"model": "pls_da", "representation": "whole", "n_subjects": 12, "is_primary_model": True,
           "balanced_accuracy_mean": 0.45, "roc_auc_mean": 0.4, "mcc_mean": -0.1, "sensitivity_mean": 0.4,
           "specificity_mean": 0.5, "permutation_observed_balanced_accuracy": 0.45,
           "permutation_p_value_unrestricted": 0.8, "permutation_p_value_sex_stratified": 0.9, "permutation_p_fdr_q": 0.9}
    confounds = pd.DataFrame([{**row, "status": "ok", "sex_only_roc_auc": 0.95}])
    report = _build_report_text(cohort, pd.DataFrame([row]), confounds, pd.DataFrame())
    assert "Recorded ages: control=4, disease=7" in report
    assert "0.950" in report
    for unsupported in ["80%", "46%", "49 of 50", "exceeds this baseline", "does not collapse", "significantly associated"]:
        assert unsupported not in report


def test_confound_geometry_selects_and_residualizes_within_outer_training(monkeypatch) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_sex_disease_confound_geometry.py"
    spec = importlib.util.spec_from_file_location("confound_geometry_integrity", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    X = np.column_stack([np.arange(32), np.sin(np.arange(32)), np.cos(np.arange(32))])
    disease = np.repeat([0, 1], 16)
    sex = np.tile(np.repeat([0, 1], 8), 2)
    training_ids = []

    def select(X_train, y_train, seed):
        assert len(X_train) < len(X)
        training_ids.append(set(X_train[:, 0]))
        return 1

    monkeypatch.setattr(module, "select_n_components", select)
    scores = module.nested_confound_predictions(X, disease, sex)
    assert len(training_ids) == 10
    assert all(training_ids[i] == training_ids[i + 1] for i in range(0, 10, 2))
    assert all(np.isfinite(values).all() for values in scores[:3])
    assert len(scores[3]) == 5


@pytest.mark.parametrize("flags", [[True, True], ["False", "True"]])
def test_report_rejects_ambiguous_primary_model_flags(flags) -> None:
    from src.analysis.control_vs_disease_ml_report import _primary_row

    with pytest.raises(ValueError):
        _primary_row(pd.DataFrame({"is_primary_model": flags}))


def test_report_rejects_duplicate_confounder_results() -> None:
    from src.analysis.control_vs_disease_ml_report import _confounder_row

    rows = pd.DataFrame({"model": ["pls_da", "pls_da"], "representation": ["whole", "whole"]})
    with pytest.raises(ValueError, match="duplicate"):
        _confounder_row(rows, "pls_da", "whole")


def test_clinical_sex_is_preferred_without_filling_gaps_from_folder_labels() -> None:
    frame = pd.DataFrame({"sex": ["female", pd.NA, "male", "female"], "gender": ["male"] * 4})
    assert ml._sex_column(frame) == "sex"
    result = ml._sex_only_baseline(frame, np.array([0, 1, 0, 1]), np.ones(4, dtype=bool))
    assert result["status"] == "missing_sex_values"
    assert result["sex_covariate_column"] == "sex"


def test_mandatory_sex_stratification_stops_before_fitting_missing_clinical_sex(tmp_path, monkeypatch) -> None:
    count = 40
    frame = pd.DataFrame({
        "subject_id": [f"fixture_{i}" for i in range(count)],
        "is_control": np.repeat([True, False], count // 2),
        "gender": np.tile(["female", "male"], count // 2),
        "sex": [pd.NA] + ["female"] * (count - 1),
        "1300__1000.0": np.sin(np.arange(count)),
        "2850__2800.0": np.cos(np.arange(count)),
    })
    monkeypatch.setattr(ml, "_load_cohort", lambda path: frame)

    def fit_forbidden(**kwargs):
        pytest.fail("A model was fitted before missing clinical sex was rejected")

    monkeypatch.setattr(ml, "_run_observed_model_phase", fit_forbidden)
    cfg = {"analysis": {"control_vs_disease_ml": {
        "enabled": True, "representations": ["whole"], "stratify_permutation_by": "sex",
    }}}
    with pytest.raises(ValueError, match="complete recorded sex"):
        ml.run_control_vs_disease_ml_pipeline(output_dir=tmp_path / "outputs", cfg=cfg)
