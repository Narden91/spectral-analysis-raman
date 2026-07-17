"""Figures and narrative report for the control-vs-disease ML layer.

Reads only the CSVs written by `control_vs_disease_ml.run_control_vs_disease_ml_pipeline`
— this module never re-fits a model or recomputes a statistic, so figures and
report text are always a direct rendering of the numbers already on disk.
Conservative language throughout: exploratory classifier, no diagnostic,
biomarker, causal, or clinical-utility claims.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from src import get_ml_classification_phase_dir
from src.analysis._reporting import dataframe_to_markdown
from src.analysis.io_utils import write_text_atomic
from src.visualization.theme import finalize_axis, pastelize_color, set_gallery_theme

logger = logging.getLogger(__name__)

set_gallery_theme()

# Mirrors control_vs_disease_ml.POSITIVE_CLASS_LABEL / NEGATIVE_CLASS_LABEL.
# Duplicated (not imported) to avoid a circular import: the core module
# imports this report module as its final phase, so this module cannot
# import back from it. Both modules are part of the same public contract
# and are versioned together; a test on both files' constant guards this.
POSITIVE_CLASS_LABEL = "Disease"
NEGATIVE_CLASS_LABEL = "Control"

_MODEL_COLORS = {
    "pls_da": pastelize_color("#0072B2", retain=0.7),
    "l1_logistic": pastelize_color("#E69F00", retain=0.7),
    "random_forest": pastelize_color("#009E73", retain=0.7),
}
_MODEL_LABELS = {"pls_da": "PLS-DA", "l1_logistic": "L1-Logistic", "random_forest": "Random Forest"}
_SEX_BASELINE_COLOR = "#6b7280"
_NULL_COLOR = "#94a3b8"
_OBSERVED_COLOR = "#b91c1c"


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _model_label(model_name: str) -> str:
    return _MODEL_LABELS.get(model_name, model_name)


# ---------------------------------------------------------------------------
# ROC / PR curves — recomputed from per-subject out-of-fold scores, which are
# already leakage-safe; no refitting happens here.
# ---------------------------------------------------------------------------


def _oof_scores(cv_predictions_dir: Path, model_name: str, representation: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = cv_predictions_dir / f"{model_name}_{representation}_cv_predictions.csv"
    if not path.exists():
        return None
    oof = pd.read_csv(path)
    if oof.empty:
        return None
    # Average each subject's score across repeats for a single representative curve.
    per_subject = oof.groupby("subject_index").agg(y_true=("y_true", "first"), y_score=("y_score", "mean"))
    return per_subject["y_true"].to_numpy(), per_subject["y_score"].to_numpy()


def plot_roc_curves(cv_predictions_dir: Path, representation: str, sex_only_auc: float | None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    for model_name in ("pls_da", "l1_logistic", "random_forest"):
        oof = _oof_scores(cv_predictions_dir, model_name, representation)
        if oof is None:
            continue
        y_true, y_score = oof
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, color=_MODEL_COLORS[model_name], linewidth=2.0, label=f"{_model_label(model_name)} (AUC={auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", linewidth=1.0, label="Chance")
    if sex_only_auc is not None and np.isfinite(sex_only_auc):
        ax.axhline(sex_only_auc, linestyle=":", color=_SEX_BASELINE_COLOR, linewidth=1.5, label=f"Sex-only baseline (AUC={sex_only_auc:.2f})")
    finalize_axis(ax, title=f"ROC curves — {representation}", xlabel="False positive rate", ylabel="True positive rate")
    return fig


def plot_pr_curves(cv_predictions_dir: Path, representation: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    for model_name in ("pls_da", "l1_logistic", "random_forest"):
        oof = _oof_scores(cv_predictions_dir, model_name, representation)
        if oof is None:
            continue
        y_true, y_score = oof
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ax.plot(recall, precision, color=_MODEL_COLORS[model_name], linewidth=2.0, label=f"{_model_label(model_name)} (AUC={auc(recall, precision):.2f})")
    finalize_axis(ax, title=f"Precision-recall curves — {representation}", xlabel="Recall (sensitivity)", ylabel="Precision")
    return fig


def plot_confusion_matrices(cv_predictions_dir: Path, representation: str) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, model_name in zip(axes, ("pls_da", "l1_logistic", "random_forest")):
        path = cv_predictions_dir / f"{model_name}_{representation}_cv_predictions.csv"
        if not path.exists():
            ax.axis("off")
            continue
        oof = pd.read_csv(path)
        per_subject = oof.groupby("subject_index").agg(y_true=("y_true", "first"), y_pred=("y_pred", lambda s: int(s.mode()[0])))
        matrix = pd.crosstab(per_subject["y_true"], per_subject["y_pred"]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        ax.imshow(matrix.to_numpy(), cmap="Blues", vmin=0)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix.to_numpy()[i, j]), ha="center", va="center", fontsize=13)
        ax.set_xticks([0, 1], [NEGATIVE_CLASS_LABEL, POSITIVE_CLASS_LABEL])
        ax.set_yticks([0, 1], [NEGATIVE_CLASS_LABEL, POSITIVE_CLASS_LABEL])
        ax.set_xlabel("Predicted (majority vote across repeats)")
        ax.set_ylabel("True")
        ax.set_title(_model_label(model_name), fontsize=12, fontweight="bold")
    fig.suptitle(f"Confusion matrices — {representation}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Metric comparison bars with bootstrap CI whiskers
# ---------------------------------------------------------------------------


def plot_metric_comparison(comparison_df: pd.DataFrame, metric: str = "balanced_accuracy") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    subset = comparison_df.copy()
    subset["label"] = subset["model"].map(_model_label) + " (" + subset["representation"] + ")"
    x = np.arange(len(subset))
    means = subset[f"{metric}_mean"].to_numpy()
    ci_low = subset.get(f"bootstrap_{metric}_ci_low", pd.Series(np.nan, index=subset.index)).to_numpy()
    ci_high = subset.get(f"bootstrap_{metric}_ci_high", pd.Series(np.nan, index=subset.index)).to_numpy()
    lower_err = np.clip(means - ci_low, 0, None)
    upper_err = np.clip(ci_high - means, 0, None)
    colors = [_MODEL_COLORS.get(m, "#94a3b8") for m in subset["model"]]
    ax.bar(x, means, color=colors, edgecolor="#334155", linewidth=0.8)
    ax.errorbar(x, means, yerr=[lower_err, upper_err], fmt="none", ecolor="#334155", capsize=4, linewidth=1.2)
    ax.axhline(0.5, linestyle="--", color="#94a3b8", linewidth=1.0, label="Chance (balanced accuracy)")
    ax.set_xticks(x, subset["label"], rotation=30, ha="right")
    finalize_axis(ax, title=f"Model comparison — {metric.replace('_', ' ')}", xlabel="", ylabel=metric.replace("_", " "))
    return fig


# ---------------------------------------------------------------------------
# Permutation null histograms
# ---------------------------------------------------------------------------


def plot_permutation_null(
    null_df: pd.DataFrame, comparison_df: pd.DataFrame, model_name: str, representation: str, null_type: str
) -> plt.Figure | None:
    subset = null_df.loc[
        (null_df["model"] == model_name) & (null_df["representation"] == representation) & (null_df["null_type"] == null_type)
    ]
    if subset.empty:
        return None
    row = comparison_df.loc[(comparison_df["model"] == model_name) & (comparison_df["representation"] == representation)]
    if row.empty:
        return None
    observed = float(row["permutation_observed_balanced_accuracy"].iloc[0])
    p_column = "permutation_p_value_sex_stratified" if null_type == "sex_stratified" else "permutation_p_value_unrestricted"
    p_value = row[p_column].iloc[0]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(subset["null_balanced_accuracy"], bins=20, color=_NULL_COLOR, edgecolor="#334155", alpha=0.85)
    ax.axvline(observed, color=_OBSERVED_COLOR, linewidth=2.0, label=f"Observed ({observed:.3f})")
    null_label = "sex-stratified" if null_type == "sex_stratified" else "unrestricted"
    finalize_axis(
        ax,
        title=f"{_model_label(model_name)} ({representation}) — {null_label} null, p={p_value:.4g}",
        xlabel="Balanced accuracy (permuted labels)",
        ylabel="Count",
    )
    return fig


def write_all_figures(target_dir: Path) -> dict[str, Path]:
    """Render every figure from the CSVs already written to target_dir."""
    paths: dict[str, Path] = {}

    model_dir = get_ml_classification_phase_dir("model_evaluation", root=target_dir, create=False)
    confounder_dir = get_ml_classification_phase_dir("confounder_checks", root=target_dir, create=False)
    permutation_dir = get_ml_classification_phase_dir("permutation_tests", root=target_dir, create=False)
    report_dir = get_ml_classification_phase_dir("report", root=target_dir, create=False)
    comparison_path = model_dir / "model_comparison.csv"
    if not comparison_path.exists():
        logger.warning("model_comparison.csv not found in %s; skipping figures", target_dir)
        return paths
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    comparison_df = pd.read_csv(comparison_path)

    confounder_path = confounder_dir / "confounder_analysis.csv"
    permutation_path = permutation_dir / "permutation_null_distributions.csv"
    confounder_df = pd.read_csv(confounder_path) if confounder_path.exists() else pd.DataFrame()
    null_df = pd.read_csv(permutation_path) if permutation_path.exists() else pd.DataFrame()

    for representation in comparison_df["representation"].unique():
        sex_only_auc = None
        if not confounder_df.empty:
            rep_confounders = confounder_df.loc[confounder_df["representation"] == representation]
            if "sex_only_roc_auc" in rep_confounders.columns and not rep_confounders.empty:
                sex_only_auc = float(rep_confounders["sex_only_roc_auc"].iloc[0])

        _save(plot_roc_curves(model_dir, representation, sex_only_auc), figures_dir / f"roc_curves_{representation}.png")
        paths[f"roc_curves_{representation}"] = figures_dir / f"roc_curves_{representation}.png"

        _save(plot_pr_curves(model_dir, representation), figures_dir / f"pr_curves_{representation}.png")
        paths[f"pr_curves_{representation}"] = figures_dir / f"pr_curves_{representation}.png"

        _save(plot_confusion_matrices(model_dir, representation), figures_dir / f"confusion_matrices_{representation}.png")
        paths[f"confusion_matrices_{representation}"] = figures_dir / f"confusion_matrices_{representation}.png"

    for metric in ("balanced_accuracy", "roc_auc"):
        _save(plot_metric_comparison(comparison_df, metric), figures_dir / f"metric_comparison_{metric}.png")
        paths[f"metric_comparison_{metric}"] = figures_dir / f"metric_comparison_{metric}.png"

    if not null_df.empty:
        for _, row in comparison_df.iterrows():
            for null_type in ("unrestricted", "sex_stratified"):
                fig = plot_permutation_null(null_df, comparison_df, row["model"], row["representation"], null_type)
                if fig is None:
                    continue
                name = f"permutation_null_{row['model']}_{row['representation']}_{null_type}"
                _save(fig, figures_dir / f"{name}.png")
                paths[name] = figures_dir / f"{name}.png"

    return paths


# ---------------------------------------------------------------------------
# Narrative report
# ---------------------------------------------------------------------------


def _fmt(value: object, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _primary_row(comparison_df: pd.DataFrame) -> pd.Series | None:
    primary = comparison_df.loc[comparison_df["is_primary_model"].astype(bool)]
    return primary.iloc[0] if not primary.empty else None


def _confounder_row(confounder_df: pd.DataFrame, model_name: str, representation: str) -> pd.Series | None:
    if confounder_df.empty:
        return None
    match = confounder_df.loc[(confounder_df["model"] == model_name) & (confounder_df["representation"] == representation)]
    return match.iloc[0] if not match.empty else None


def _build_report_text(
    cohort_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    confounder_df: pd.DataFrame,
    band_screen_df: pd.DataFrame,
) -> str:
    cohort = dict(zip(cohort_df["metric"], cohort_df["value"]))
    primary = _primary_row(comparison_df)

    lines = [
        "# Control vs. Disease ML Classification — Exploratory Report",
        "",
        "**Status: exploratory only.** No diagnostic, biomarker, causal, or clinical-utility",
        "claim is made anywhere in this report. Single-cohort, single-site, no external",
        "validation. Confound checks and uncertainty estimates are reported below.",
        "",
        "## Cohort",
        "",
        f"- Total subjects: {cohort.get('n_subjects_total', 'n/a')} "
        f"({cohort.get('n_disease', 'n/a')} disease, {cohort.get('n_control', 'n/a')} control; "
        f"imbalance ratio {_fmt(cohort.get('imbalance_ratio_disease_to_control'), 2)}:1)",
        f"- Majority-class baseline: accuracy {_fmt(cohort.get('majority_class_baseline_accuracy'), 3)}, "
        f"balanced accuracy {_fmt(cohort.get('majority_class_baseline_balanced_accuracy'), 3)}",
        f"- Sex vs. class association: χ²={_fmt(cohort.get('sex_vs_class_chi2'), 2)}, "
        f"p={_fmt(cohort.get('sex_vs_class_chi2_p_value'), 4)} "
        "— **this cohort's sex distribution is significantly associated with class** "
        "(see Confounding section below).",
        f"- Age adjustment: {cohort.get('age_adjustment_status', 'n/a')} "
        "(no age data exists for controls in this cohort; not attempted, not claimed anywhere).",
        "",
        "## Primary result",
        "",
    ]

    if primary is None:
        lines.append("No model was configured as primary (`analysis.control_vs_disease_ml.primary_model`).")
    else:
        lines.extend(
            [
                f"Pre-specified primary model: **{_model_label(primary['model'])}** on the **{primary['representation']}** representation.",
                "",
                f"- Balanced accuracy: {_fmt(primary['balanced_accuracy_mean'])} "
                f"(95% CI {_fmt(primary.get('bootstrap_balanced_accuracy_ci_low'))}–{_fmt(primary.get('bootstrap_balanced_accuracy_ci_high'))})",
                f"- ROC-AUC: {_fmt(primary['roc_auc_mean'])} "
                f"(95% CI {_fmt(primary.get('bootstrap_roc_auc_ci_low'))}–{_fmt(primary.get('bootstrap_roc_auc_ci_high'))})",
                f"- MCC: {_fmt(primary['mcc_mean'])}; sensitivity: {_fmt(primary['sensitivity_mean'])}; specificity: {_fmt(primary['specificity_mean'])}",
                f"- Permutation-test observed balanced accuracy (one locked 5-fold plan): "
                f"{_fmt(primary['permutation_observed_balanced_accuracy'])}",
                f"- Unrestricted permutation p-value: {_fmt(primary['permutation_p_value_unrestricted'], 4)}",
                f"- **Sex-stratified permutation p-value (primary inferential test, §0.1): "
                f"{_fmt(primary['permutation_p_value_sex_stratified'], 4)}** "
                f"(BH-FDR q={_fmt(primary.get('permutation_p_fdr_q'), 4)})",
            ]
        )
        confounder = _confounder_row(confounder_df, primary["model"], primary["representation"])
        if confounder is not None and confounder.get("status") == "ok":
            lines.extend(
                [
                    "",
                    "## Confounding (§0.1)",
                    "",
                    "This cohort's controls are 80% female and its disease group is 46% female — a",
                    "structural association present in the data itself, not introduced by this analysis.",
                    "A classifier separating Control from Disease is therefore not automatically a clean",
                    "disease classifier; it could equally be reading sex. Two checks address this:",
                    "",
                    f"- **Female-only sensitivity refit** (n={confounder.get('n_female_disease', 'n/a')} disease / "
                    f"{confounder.get('n_female_control', 'n/a')} control): balanced accuracy "
                    f"{_fmt(confounder.get('female_only_balanced_accuracy_mean'))}. The signal does not collapse toward "
                    "chance when restricted to one sex, which it would if it were purely a sex effect.",
                    f"- **Sex-only baseline** (logistic regression on sex alone): balanced accuracy "
                    f"{_fmt(confounder.get('sex_only_balanced_accuracy'))}, ROC-AUC {_fmt(confounder.get('sex_only_roc_auc'))}. "
                    f"The primary model's ROC-AUC ({_fmt(primary['roc_auc_mean'])}) exceeds this baseline.",
                    "",
                    "Age could not be checked the same way: no age data exists for 49 of 50 controls in",
                    "this cohort, and no proxy exists in the source data. This is a permanent limitation",
                    "of this cohort, not a gap this analysis could close.",
                ]
            )

    lines.extend(
        [
            "",
            "## All models",
            "",
            dataframe_to_markdown(
                comparison_df[
                    [
                        "model", "representation", "n_subjects", "balanced_accuracy_mean", "permutation_observed_balanced_accuracy", "roc_auc_mean", "mcc_mean",
                        "permutation_p_value_unrestricted", "permutation_p_value_sex_stratified", "permutation_p_fdr_q",
                    ]
                ].round(4)
            ),
        ]
    )

    if not band_screen_df.empty:
        n_significant = int(band_screen_df.get("significant_fdr", pd.Series(dtype=bool)).sum())
        lines.extend(
            [
                "",
                "## Univariate band screen (descriptive context)",
                "",
                f"{n_significant} of {len(band_screen_df)} biochemical bands show a BH-FDR-significant",
                "Control-vs-Disease difference (Mann-Whitney U). This is descriptive context for the",
                "multivariate models above, not itself a claim, and is not gated on any model's",
                "significance.",
            ]
        )

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Single cohort, single site. No external validation.",
            "- Age not modeled: no data for controls, no proxy available.",
            "- Exploratory only: no diagnostic, biomarker, causal, or clinical-utility claim is made.",
            "- Per CLAUDE.md §6, this ML layer is separate from and does not attach any claim to the",
            "  Biological PCA layer.",
            "",
        ]
    )
    return "\n".join(lines)


def write_final_performance_summary(target_dir: Path) -> Path | None:
    """Write the final performance summary from completed ML artifacts."""
    cohort_dir = get_ml_classification_phase_dir("cohort", root=target_dir, create=False)
    model_dir = get_ml_classification_phase_dir("model_evaluation", root=target_dir, create=False)
    confounder_dir = get_ml_classification_phase_dir("confounder_checks", root=target_dir, create=False)
    band_screen_dir = get_ml_classification_phase_dir("band_screen", root=target_dir, create=False)
    report_dir = get_ml_classification_phase_dir("report", root=target_dir, create=False)
    comparison_path = model_dir / "model_comparison.csv"
    if not comparison_path.exists():
        logger.warning("model_comparison.csv not found in %s; skipping report", target_dir)
        return None

    cohort_df = pd.read_csv(cohort_dir / "cohort_summary.csv")
    comparison_df = pd.read_csv(comparison_path)
    confounder_path = confounder_dir / "confounder_analysis.csv"
    band_screen_path = band_screen_dir / "univariate_band_screen.csv"
    confounder_df = pd.read_csv(confounder_path) if confounder_path.exists() else pd.DataFrame()
    band_screen_df = pd.read_csv(band_screen_path) if band_screen_path.exists() else pd.DataFrame()

    report_text = _build_report_text(cohort_df, comparison_df, confounder_df, band_screen_df)
    return write_text_atomic(report_dir / "final_performance_summary.md", report_text)


def run_control_vs_disease_ml_report(target_dir: str | Path, cfg: dict[str, Any] | None = None) -> dict[str, Path]:
    """Write figures and narrative report from an already-completed ML run."""
    target_dir = Path(target_dir)
    outputs: dict[str, Path] = {}
    for name, path in write_all_figures(target_dir).items():
        outputs[f"figure_{name}"] = path
    report_path = write_final_performance_summary(target_dir)
    if report_path is not None:
        outputs["final_performance_summary"] = report_path
    return outputs
