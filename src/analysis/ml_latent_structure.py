"""Latent-structure interpretive reporting for the ML track.

Generates a manuscript-oriented report from saved ML artifacts without
rerunning cross-validation.  Optionally supports a region-masked
feature-extraction comparison branch.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis._reporting import dataframe_to_markdown

logger = logging.getLogger(__name__)


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None


def _summarize_feature_families(feature_importance_df: pd.DataFrame | None) -> dict:
    if feature_importance_df is None or feature_importance_df.empty or "feature" not in feature_importance_df.columns:
        return {}

    def _family(feature_name: str) -> str:
        if "pca" in feature_name.lower():
            return "pca"
        if "ch_lipid" in feature_name.lower() or "ch_" in feature_name.lower():
            return "ch_lipid_state"
        if "band" in feature_name.lower() or "ratio" in feature_name.lower():
            return "eda_bands"
        if "peak" in feature_name.lower():
            return "peak"
        if "hetero" in feature_name.lower():
            return "heterogeneity"
        if "sex" in feature_name.lower():
            return "sex_covariate"
        return "other"

    df = feature_importance_df.copy()
    df["family"] = df["feature"].apply(_family)
    score_col = next(
        (c for c in ("importance", "mean_importance", "score") if c in df.columns),
        df.columns[-1],
    )
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    family_scores = df.groupby("family")[score_col].sum().sort_values(ascending=False)
    family_counts = df["family"].value_counts()
    top_features = df.sort_values(score_col, ascending=False).head(10)

    return {
        "family_scores": family_scores,
        "family_counts": family_counts,
        "top_features": top_features,
        "score_column": score_col,
    }


def _summarize_ablation(ablation_df: pd.DataFrame | None) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["track", "delta_metric", "direction"])
    if ablation_df is None or ablation_df.empty:
        return empty
    metric_col = next(
        (c for c in ("balanced_accuracy", "mean_test_score", "score") if c in ablation_df.columns),
        ablation_df.columns[-1],
    )
    if "track" not in ablation_df.columns:
        return empty
    baseline = ablation_df[ablation_df["track"].str.lower() == "all"]
    if baseline.empty:
        return empty
    baseline_score = float(baseline[metric_col].iloc[0])
    rows = []
    for _, row in ablation_df.iterrows():
        delta = float(row[metric_col]) - baseline_score
        rows.append(
            {
                "track": str(row["track"]),
                "delta_metric": delta,
                "direction": "increases" if delta > 0 else "decreases" if delta < 0 else "neutral",
            }
        )
    return pd.DataFrame(rows)


def _region_in_top_features(
    feature_importance_df: pd.DataFrame | None,
    windows: list[list[float]],
) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["feature", "importance", "in_suspect_window"])
    if feature_importance_df is None or feature_importance_df.empty:
        return empty
    score_col = next(
        (c for c in ("importance", "mean_importance", "score") if c in feature_importance_df.columns),
        feature_importance_df.columns[-1],
    )
    df = feature_importance_df.copy()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    def _in_window(name: str) -> bool:
        numbers = [float(token) for token in str(name).replace("_", " ").split() if _is_number(token)]
        if not numbers:
            return False
        return any(float(lo) <= n <= float(hi) for n in numbers for lo, hi in windows)

    df["in_suspect_window"] = df["feature"].apply(_in_window)
    return df[["feature", score_col, "in_suspect_window"]].rename(columns={score_col: "importance"})


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def generate_latent_structure_ml_report(
    ml_output_dir: Path | str,
    out_path: Path | str | None = None,
    suspect_windows: list[list[float]] | None = None,
) -> str:
    """Write ``latent_structure_ml_report.md`` from saved ML artifacts.

    Parameters
    ----------
    ml_output_dir : Path or str
        Directory containing ML pipeline outputs.
    out_path : Path or str or None
        Output markdown path. Defaults to ``<ml_output_dir>/latent_structure_ml_report.md``.
    suspect_windows : list of [lo, hi] or None
        Windows to check among top features.

    Returns
    -------
    str
        The written report text.
    """
    ml_dir = Path(ml_output_dir)
    out_path = Path(out_path) if out_path is not None else ml_dir / "latent_structure_ml_report.md"
    suspect_windows = suspect_windows or []

    feature_importance = _safe_read_csv(ml_dir / "feature_importance.csv")
    ablation = _safe_read_csv(ml_dir / "feature_track_ablation.csv")
    sex_stratified = _safe_read_csv(ml_dir / "sex_stratified_metrics.csv")
    ch_audit = _safe_read_csv(ml_dir / "archive" / "ch_lipid_state_audit.csv")

    family_summary = _summarize_feature_families(feature_importance)
    ablation_summary = _summarize_ablation(ablation)
    region_features = _region_in_top_features(feature_importance, suspect_windows)

    lines = [
        "# ML latent-structure interpretive report",
        "",
        "## Feature-family contributions",
        "",
    ]
    if family_summary:
        lines.append(f"Score column used: **{family_summary['score_column']}**")
        lines.append("")
        lines.append(dataframe_to_markdown(family_summary["family_scores"].to_frame("score")))
        lines.append("")
        lines.append("### Top features")
        lines.append("")
        top_cols = ["feature", family_summary["score_column"]]
        lines.append(dataframe_to_markdown(family_summary["top_features"][top_cols]))
        lines.append("")
    else:
        lines.append("Feature-importance artifact not available.")
        lines.append("")

    lines.extend(
        [
            "## Feature-track ablation",
            "",
            dataframe_to_markdown(ablation_summary) if not ablation_summary.empty else "Ablation artifact not available.",
            "",
            "## CH-stretch lipid-state audit",
            "",
            dataframe_to_markdown(ch_audit.head(20)) if ch_audit is not None else "CH lipid-state audit not available.",
            "",
        ]
    )

    if suspect_windows:
        lines.extend(
            [
                "## Suspect-region presence in top features",
                f"Windows: {suspect_windows}",
                "",
                dataframe_to_markdown(region_features) if not region_features.empty else "No feature-importance data available.",
                "",
            ]
        )

    lines.extend(
        [
            "## Sex-stratified performance",
            "",
            dataframe_to_markdown(sex_stratified) if sex_stratified is not None else "Sex-stratified metrics not available.",
            "",
            "## Interpretation checklist",
            "",
            "- Which feature families dominate? " + (", ".join(family_summary["family_scores"].index.tolist()) if family_summary else "N/A"),
            "- Do PCA-derived features matter? " + ("Yes" if family_summary and "pca" in family_summary["family_scores"].index else "No / not available"),
            "- Do CH-stretch lipid-state features contribute? " + ("Yes" if family_summary and "ch_lipid_state" in family_summary["family_scores"].index else "No / not available"),
            "- Does the suspect region appear among top features? " + ("Yes" if region_features.get("in_suspect_window", pd.Series([], dtype=bool)).any() else "No / not available"),
            "- Is performance modulated by sex? " + ("Check table above" if sex_stratified is not None else "Not available"),
            "",
        ]
    )

    report_text = "\n".join(lines)
    out_path.write_text(report_text, encoding="utf-8")
    logger.info("ML latent-structure report written to %s", out_path)
    return report_text


