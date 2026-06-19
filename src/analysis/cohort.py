from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_cohort_summary(
    df_merged: pd.DataFrame,
    expected_files_per_subject: int = 12,
) -> dict[str, pd.DataFrame]:
    """Build cohort-level summary tables from merged index + clinical metadata."""
    if df_merged.empty:
        empty = pd.DataFrame()
        return {
            "group_counts": empty,
            "subject_summary": empty,
            "clinical_missingness": empty,
            "incomplete_subjects": empty,
        }


    base = df_merged.copy()
    base["subject_id"] = base["subject_id"].astype(str)
    if "gender" in base.columns:
        base["gender"] = base["gender"].astype(str).str.lower()

    group_counts = (
        base.groupby(["is_control", "gender"], dropna=False)["subject_id"]
        .nunique()
        .reset_index(name="n_subjects")
    )
    group_counts["condition"] = group_counts["is_control"].map(
        {True: "Control", False: "Disease"}
    )
    group_counts = group_counts[["condition", "gender", "n_subjects"]]

    subject_summary = (
        base.groupby("subject_id", dropna=False)
        .agg(
            n_files=("filepath", "nunique"),
            n_slides=("slide_number", "nunique"),
            condition=("is_control", "first"),
            gender=("gender", "first"),
        )
        .reset_index()
    )
    subject_summary["condition"] = subject_summary["condition"].map(
        {True: "Control", False: "Disease"}
    )
    subject_summary["is_incomplete"] = (
        subject_summary["n_files"] < expected_files_per_subject
    )

    clinical_columns = [
        c
        for c in base.columns
        if c
        not in {
            "subject_id",
            "filename",
            "filepath",
            "extension",
            "is_redone",
            "is_control",
            "gender",
            "slide_number",
            "spectral_range_cm-1",
            "measurement_number",
            "area_id",
        }
    ]

    clinical_missingness = pd.DataFrame(columns=["column", "missing_rate"])
    if clinical_columns:
        missing_rows = []
        unique_subjects = base["subject_id"].nunique()
        for col in clinical_columns:
            per_subject = (
                base[["subject_id", col]].drop_duplicates(subset=["subject_id"]).copy()
            )
            missing_rate = per_subject[col].isna().sum() / max(unique_subjects, 1)
            missing_rows.append({"column": col, "missing_rate": float(missing_rate)})
        clinical_missingness = pd.DataFrame(missing_rows).sort_values(
            "missing_rate", ascending=False
        )

    incomplete_subjects = subject_summary[subject_summary["is_incomplete"]].copy()

    return {
        "group_counts": group_counts,
        "subject_summary": subject_summary,
        "clinical_missingness": clinical_missingness,
        "incomplete_subjects": incomplete_subjects,
    }



def save_cohort_summary(
    summary: dict[str, pd.DataFrame],
    output_dir: Path,
    prefix: str = "cohort",
) -> dict[str, Path]:
    """Save cohort summary tables to CSV files and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, table in summary.items():
        out_path = output_dir / f"{prefix}_{key}.csv"
        table.to_csv(out_path, index=False)
        paths[key] = out_path
    return paths
