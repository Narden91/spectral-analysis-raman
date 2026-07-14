from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.cohort import build_cohort_summary

COHORT_TABLE_KEYS = (
    "group_counts",
    "subject_summary",
    "clinical_missingness",
    "incomplete_subjects",
)


def build_materialized_cohorts(df_merged: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    control_vs_disease = build_cohort_summary(df_merged)
    disease_only_input = df_merged.loc[~df_merged.get("is_control", pd.Series(False, index=df_merged.index)).fillna(False)].copy()
    disease_only = build_cohort_summary(disease_only_input)
    return {
        "control_vs_disease": control_vs_disease,
        "disease_only": disease_only,
    }


def write_cohort_tables(
    cohort_root: str | Path,
    cohorts: dict[str, dict[str, pd.DataFrame]],
) -> list[Path]:
    output_root = Path(cohort_root)
    written: list[Path] = []
    for cohort_name, tables in cohorts.items():
        cohort_dir = output_root / cohort_name
        cohort_dir.mkdir(parents=True, exist_ok=True)
        for table_name in COHORT_TABLE_KEYS:
            frame = tables.get(table_name, pd.DataFrame()).copy()
            if "subject_id" in frame.columns:
                frame["subject_id"] = frame["subject_id"].astype(str)
            else:
                frame["subject_id"] = pd.Series(pd.NA, index=frame.index, dtype="object")
            if "condition" in frame.columns and "group_label" not in frame.columns:
                frame["group_label"] = frame["condition"].astype(str)
            if "group_label" not in frame.columns:
                frame["group_label"] = pd.NA
            frame["cohort_name"] = cohort_name
            parquet_path = cohort_dir / f"{table_name}.parquet"
            frame.to_parquet(parquet_path, index=False)
            written.append(parquet_path)
    return written
