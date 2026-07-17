"""Small, fast tests for the control-vs-disease figures/report module.

Deliberately does not run the ML pipeline (nested CV, permutation, bootstrap
are all seconds-to-minutes; this module should be verifiable in milliseconds)
- it writes tiny hand-built CSVs matching the real output schema and checks
that figures/report render from them. The pipeline-level integration is
already covered by test_control_vs_disease_ml.py's runner smoke test.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import get_ml_classification_phase_dir
from src.analysis.control_vs_disease_ml_report import (
    NEGATIVE_CLASS_LABEL,
    POSITIVE_CLASS_LABEL,
    run_control_vs_disease_ml_report,
    write_all_figures,
    write_final_performance_summary,
)


@pytest.fixture
def ml_output_dir(tmp_path: Path) -> Path:
    """A minimal but schema-correct set of control_vs_disease_ml.py output CSVs."""
    target_dir = tmp_path / "05_ml_classification"
    target_dir.mkdir()
    cohort_dir = get_ml_classification_phase_dir("cohort", root=target_dir)
    model_dir = get_ml_classification_phase_dir("model_evaluation", root=target_dir)
    confounder_dir = get_ml_classification_phase_dir("confounder_checks", root=target_dir)
    permutation_dir = get_ml_classification_phase_dir("permutation_tests", root=target_dir)
    band_screen_dir = get_ml_classification_phase_dir("band_screen", root=target_dir)

    pd.DataFrame(
        {
            "metric": [
                "n_subjects_total", "n_control", "n_disease", "imbalance_ratio_disease_to_control",
                "majority_class_baseline_accuracy", "majority_class_baseline_balanced_accuracy",
                "sex_vs_class_chi2", "sex_vs_class_chi2_p_value", "age_adjustment_status",
            ],
            "value": [10, 4, 6, 1.5, 0.6, 0.5, 3.2, 0.07, "not_attempted_no_control_age_data"],
        }
    ).to_csv(cohort_dir / "cohort_summary.csv", index=False)

    comparison_rows = []
    for model_name in ("pls_da", "l1_logistic", "random_forest"):
        for representation in ("whole", "band_panel"):
            comparison_rows.append(
                {
                    "positive_class": "Disease",
                    "model": model_name,
                    "representation": representation,
                    "n_subjects": 10,
                    "balanced_accuracy_mean": 0.65,
                    "permutation_observed_balanced_accuracy": 0.64,
                    "permutation_cv_splits": 5,
                    "permutation_cv_repeats": 1,
                    "roc_auc_mean": 0.7,
                    "mcc_mean": 0.3,
                    "sensitivity_mean": 0.6,
                    "specificity_mean": 0.7,
                    "bootstrap_balanced_accuracy_ci_low": 0.5,
                    "bootstrap_balanced_accuracy_ci_high": 0.8,
                    "bootstrap_roc_auc_ci_low": 0.55,
                    "bootstrap_roc_auc_ci_high": 0.85,
                    "permutation_p_value_unrestricted": 0.02,
                    "permutation_p_value_sex_stratified": 0.04,
                    "permutation_p_fdr_q": 0.05,
                    "is_primary_model": model_name == "pls_da" and representation == "whole",
                }
            )
    pd.DataFrame(comparison_rows).to_csv(model_dir / "model_comparison.csv", index=False)

    confounder_rows = [
        {
            "model": model_name,
            "representation": representation,
            "status": "ok",
            "n_female_control": 3,
            "n_female_disease": 4,
            "female_only_balanced_accuracy_mean": 0.62,
            "sex_only_balanced_accuracy": 0.55,
            "sex_only_roc_auc": 0.58,
        }
        for model_name in ("pls_da", "l1_logistic", "random_forest")
        for representation in ("whole", "band_panel")
    ]
    pd.DataFrame(confounder_rows).to_csv(confounder_dir / "confounder_analysis.csv", index=False)

    null_rows = []
    for model_name in ("pls_da", "l1_logistic", "random_forest"):
        for representation in ("whole", "band_panel"):
            for null_type in ("unrestricted", "sex_stratified"):
                for value in (0.48, 0.5, 0.52, 0.49, 0.51):
                    null_rows.append(
                        {"model": model_name, "representation": representation, "null_type": null_type, "null_balanced_accuracy": value}
                    )
    pd.DataFrame(null_rows).to_csv(permutation_dir / "permutation_null_distributions.csv", index=False)

    pd.DataFrame(
        {
            "band": ["band_a", "band_b"],
            "n_control": [4, 4],
            "n_disease": [6, 6],
            "p_value": [0.01, 0.5],
            "fdr_q": [0.02, 0.5],
            "significant_fdr": [True, False],
        }
    ).to_csv(band_screen_dir / "univariate_band_screen.csv", index=False)

    for model_name in ("pls_da", "l1_logistic", "random_forest"):
        for representation in ("whole", "band_panel"):
            oof_rows = []
            for repeat in range(2):
                for subject_index in range(10):
                    y_true = 1 if subject_index < 6 else 0
                    oof_rows.append(
                        {
                            "repeat": repeat,
                            "subject_index": subject_index,
                            "y_true": y_true,
                            "y_pred": y_true,
                            "y_score": 0.8 if y_true == 1 else 0.2,
                        }
                    )
            pd.DataFrame(oof_rows).to_csv(
                model_dir / f"{model_name}_{representation}_cv_predictions.csv", index=False
            )

    return target_dir


def test_class_label_constants_match_core_module() -> None:
    """These are intentionally duplicated (not imported) to avoid a circular
    import between the core module and this report module - see the comment
    in control_vs_disease_ml_report.py. This test is the guard that keeps
    them in sync if either module's constant ever changes.
    """
    from src.analysis.control_vs_disease_ml import NEGATIVE_CLASS_LABEL as core_negative
    from src.analysis.control_vs_disease_ml import POSITIVE_CLASS_LABEL as core_positive

    assert POSITIVE_CLASS_LABEL == core_positive
    assert NEGATIVE_CLASS_LABEL == core_negative


def test_ml_output_phase_numbering_is_contiguous(tmp_path: Path) -> None:
    root = tmp_path / "05_ml_classification"
    phases = [
        "cohort",
        "model_evaluation",
        "confounder_checks",
        "permutation_tests",
        "band_screen",
        "report",
    ]

    assert [get_ml_classification_phase_dir(phase, root=root, create=False).name for phase in phases] == [
        "01_cohort",
        "02_model_evaluation",
        "03_confounder_checks",
        "04_permutation_tests",
        "05_univariate_band_screen",
        "06_report",
    ]


def test_final_performance_summary_contains_claims_discipline_language(ml_output_dir: Path) -> None:
    report_path = write_final_performance_summary(ml_output_dir)
    assert report_path is not None
    text = report_path.read_text(encoding="utf-8")
    assert "exploratory only" in text.lower()
    assert "diagnostic" in text.lower()
    assert "no external" in text.lower() or "external validation" in text.lower()


def test_final_performance_summary_includes_primary_result(ml_output_dir: Path) -> None:
    text = write_final_performance_summary(ml_output_dir).read_text(encoding="utf-8")
    assert "PLS-DA" in text
    assert "0.650" in text or "0.65" in text


def test_final_performance_summary_reports_sex_confound(ml_output_dir: Path) -> None:
    text = write_final_performance_summary(ml_output_dir).read_text(encoding="utf-8")
    assert "sex" in text.lower()
    assert "female-only" in text.lower() or "sensitivity refit" in text.lower()


def test_final_performance_summary_missing_comparison_returns_none(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert write_final_performance_summary(empty_dir) is None


def test_write_all_figures_produces_expected_files(ml_output_dir: Path) -> None:
    paths = write_all_figures(ml_output_dir)
    figures_dir = get_ml_classification_phase_dir("report", root=ml_output_dir) / "figures"

    assert any(figures_dir.glob("roc_curves_*.png"))
    assert any(figures_dir.glob("pr_curves_*.png"))
    assert any(figures_dir.glob("confusion_matrices_*.png"))
    assert any(figures_dir.glob("metric_comparison_*.png"))
    assert any(figures_dir.glob("permutation_null_*.png"))
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_write_all_figures_missing_comparison_returns_empty(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert write_all_figures(empty_dir) == {}


def test_run_control_vs_disease_ml_report_writes_both(ml_output_dir: Path) -> None:
    outputs = run_control_vs_disease_ml_report(ml_output_dir, cfg=None)
    assert "final_performance_summary" in outputs
    assert any(key.startswith("figure_") for key in outputs)
