"""Phase I: summarize fixed post hoc panel features by exploratory subgroup."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import get_classification_phase_dir, get_confirmatory_panel_dir
from src import resolve_cfg as _cfg
from src.analysis.clinical_feature_associations import filter_disease_subjects, load_subject_feature_matrix
from src.analysis.io_utils import load_csv_or_empty, write_csv_atomic

logger = logging.getLogger(__name__)




def _panel_cluster_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("analysis", {}).get("panel_cluster_summary", {})


def _summary_rows(frame: pd.DataFrame, variables: list[str], variable_type: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for subgroup, group in frame.groupby("subgroup_label", dropna=False):
        for variable in variables:
            values = pd.to_numeric(group[variable], errors="coerce").dropna()
            rows.append(
                {
                    "subgroup_label": subgroup,
                    "variable_name": variable,
                    "variable_type": variable_type,
                    "n": int(values.size),
                    "mean": float(values.mean()) if values.size else np.nan,
                    "median": float(values.median()) if values.size else np.nan,
                    "sd": float(values.std(ddof=1)) if values.size > 1 else np.nan,
                    "q25": float(values.quantile(0.25)) if values.size else np.nan,
                    "q75": float(values.quantile(0.75)) if values.size else np.nan,
                }
            )
    return rows


def summarize_panel_by_subgroup(
    *,
    clinical: pd.DataFrame,
    panel: pd.DataFrame,
    assignments: pd.DataFrame,
    endpoints: list[str],
) -> pd.DataFrame:
    """Return cluster-level summaries for panel features and endpoint values."""
    required = {"subject_id", "subgroup_label"}
    if not required.issubset(assignments.columns) or "subject_id" not in panel.columns:
        return pd.DataFrame(
            columns=["subgroup_label", "variable_name", "variable_type", "n", "mean", "median", "sd", "q25", "q75"]
        )

    clinical = clinical.copy()
    panel = panel.copy()
    assignments = assignments.copy()
    for frame in (clinical, panel, assignments):
        frame["subject_id"] = frame["subject_id"].astype(str)

    merged = assignments[["subject_id", "subgroup_label"]].merge(panel, on="subject_id", how="inner")
    endpoint_columns = [endpoint for endpoint in endpoints if endpoint in clinical.columns]
    if endpoint_columns:
        merged = merged.merge(clinical[["subject_id", *endpoint_columns]], on="subject_id", how="left")

    panel_features = [column for column in panel.columns if column != "subject_id"]
    rows = _summary_rows(merged, panel_features, "panel_feature")
    rows.extend(_summary_rows(merged, endpoint_columns, "endpoint"))
    return pd.DataFrame(
        rows,
        columns=["subgroup_label", "variable_name", "variable_type", "n", "mean", "median", "sd", "q25", "q75"],
    )


def run_panel_cluster_summary(
    *,
    feature_matrix_path: str | Path | None = None,
    panel_matrix_path: str | Path | None = None,
    subgroup_assignments_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write the Phase I cluster x panel-feature summary table."""
    cfg = _cfg(cfg)
    phase_cfg = _panel_cluster_cfg(cfg)
    if not bool(phase_cfg.get("enabled", True)):
        return {}

    target_dir = Path(output_dir) if output_dir is not None else get_confirmatory_panel_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    panel_path = Path(panel_matrix_path) if panel_matrix_path is not None else target_dir / "panel_feature_matrix.csv"
    assignments_path = (
        Path(subgroup_assignments_path)
        if subgroup_assignments_path is not None
        else get_classification_phase_dir("disease_subgroups") / "disease_subgroup_assignments.csv"
    )
    output_path = target_dir / "panel_cluster_summary.csv"

    clinical = filter_disease_subjects(load_subject_feature_matrix(feature_matrix_path=feature_matrix_path))
    panel = load_csv_or_empty(panel_path)
    assignments = load_csv_or_empty(assignments_path)
    endpoints = [str(item) for item in phase_cfg.get("endpoints", ["moca_baseline", "moca_3m", "delta_moca"])]

    summary = summarize_panel_by_subgroup(
        clinical=clinical,
        panel=panel,
        assignments=assignments,
        endpoints=endpoints,
    )
    write_csv_atomic(summary, output_path, index=False)
    logger.info("Panel cluster summary written to %s", output_path)
    return {"panel_cluster_summary": output_path}
