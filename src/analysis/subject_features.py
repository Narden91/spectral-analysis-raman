"""Subject-level spectral feature matrix exports."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src import config, get_cross_region_phase_dir
from src.analysis.eda_mode import resolve_eda_analysis_mode
from src.clinical.endpoints import build_clinical_endpoints
from src.data_preparation.loaders import load_clinical_subjects
from src.data_preparation.store import DataSource, get_active_data_source

if TYPE_CHECKING:
    from src.analysis.biological_pca import SubjectInputs

logger = logging.getLogger(__name__)

SPECTRAL_PREFIXES = ("1300__", "2850__")
# Baseline-corrected, un-normalized companion features (positive-valued basis for
# physical band-integral intensities; see src/analysis/biochemical_ratios.py).
# Kept out of the SNV/area SPECTRAL_PREFIXES matrices and exported separately so the
# "whole" matrix's {1300, 2850} block-identity invariant stays exact.
SPECTRAL_BC_PREFIXES = ("1300_bc__", "2850_bc__")
RUNTIME_ENDPOINT_COLUMNS = (
    "moca_baseline",
    "moca_3m",
    "rbans_total",
    "age",
    "aspects_score",
    "sex",
    "diagnosis",
    "lesion_location",
    "delta_moca",
    "moca_severity_group",
    "delta_moca_group",
    "lesion_location_group",
)


def _runtime_endpoint_table(cfg: dict) -> pd.DataFrame:
    runtime_path = (
        cfg.get("analysis", {})
        .get("clinical_endpoints_runtime", {})
        .get("subject_table_path")
    )
    if runtime_path:
        path = Path(runtime_path)
        if path.exists():
            return pd.read_csv(path)

    try:
        return build_clinical_endpoints(cfg=cfg).clinical_subject_table
    # A missing clinical table is expected in case-control runs (controls carry no
    # endpoints); keep enrichment optional there instead of aborting the export.
    except (ValueError, FileNotFoundError) as exc:
        logger.warning("Subject feature export skipped endpoint enrichment: %s", exc)
        return pd.DataFrame(columns=["subject_id", *RUNTIME_ENDPOINT_COLUMNS])


def _clinical_table(cfg: dict, data_source: DataSource | None = None) -> pd.DataFrame:
    clinical = load_clinical_subjects(data_source).copy()
    if clinical.empty:
        return pd.DataFrame(columns=["subject_id"])
    clinical["subject_id"] = clinical["subject_id"].astype(str)

    runtime = _runtime_endpoint_table(cfg).copy()
    if runtime.empty:
        return clinical
    runtime["subject_id"] = runtime["subject_id"].astype(str)

    overlapping = [
        column
        for column in runtime.columns
        if column != "subject_id" and column in clinical.columns
    ]
    runtime = runtime.drop(columns=overlapping, errors="ignore")
    return clinical.merge(runtime, on="subject_id", how="left")


def _region_qc_wide(subject_region_counts: pd.DataFrame) -> pd.DataFrame:
    if subject_region_counts.empty:
        return pd.DataFrame(columns=["subject_id"])

    counts = subject_region_counts.copy()
    counts["subject_id"] = counts["subject_id"].astype(str)
    wide = counts.pivot(index="subject_id", columns="spectral_range")
    wide.columns = [
        f"{metric}_{str(block)}"
        for metric, block in wide.columns.to_flat_index()
    ]
    return wide.reset_index()


def _subject_source_provenance(
    inputs: "SubjectInputs", cfg: dict, data_source: DataSource | None = None
) -> pd.DataFrame:
    base = inputs.subject_spectra[["subject_id", "n_measurements", "n_areas", "n_total_pixels", "n_retained_pixels"]].copy()
    base["subject_id"] = base["subject_id"].astype(str)
    if "condition" in inputs.subject_spectra.columns:
        base["condition"] = inputs.subject_spectra["condition"]
    if "gender" in inputs.subject_spectra.columns:
        base["gender"] = inputs.subject_spectra["gender"]
    if "group" in inputs.subject_spectra.columns:
        base["group"] = inputs.subject_spectra["group"]

    qc_wide = _region_qc_wide(inputs.subject_region_counts)
    provenance = inputs.input_provenance.copy()
    provenance["subject_id"] = provenance["subject_id"].astype(str)
    provenance["filepath"] = provenance["filepath"].astype(str)

    area_counts = (
        provenance.groupby(["subject_id", "spectral_range"], dropna=False)
        .agg(
            n_source_rows=("filepath", "size"),
            n_unique_files=("filepath", "nunique"),
            n_unique_samples=("sample_id", "nunique"),
            n_unique_spots=("spot_id", "nunique"),
        )
        .reset_index()
    )
    area_wide = area_counts.pivot(index="subject_id", columns="spectral_range")
    area_wide.columns = [
        f"{metric}_{str(block)}"
        for metric, block in area_wide.columns.to_flat_index()
    ]
    area_wide = area_wide.reset_index()

    source = data_source or get_active_data_source()
    source_name = source.source_name if source is not None else "preprocessed"
    base["analysis_mode"] = resolve_eda_analysis_mode(cfg)
    base["feature_aggregation"] = str(cfg.get("biological_pca", {}).get("aggregation", "median"))
    base["data_source"] = source_name
    base["has_block_1300"] = True
    base["has_block_2850"] = True

    return (
        base.merge(qc_wide, on="subject_id", how="left")
        .merge(area_wide, on="subject_id", how="left")
        .sort_values("subject_id")
        .reset_index(drop=True)
    )


def _ordered_columns(frame: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    preferred = [
        "subject_id",
        "condition",
        "gender",
        "group",
        "analysis_mode",
        "feature_aggregation",
        "data_source",
    ]
    qc_prefixes = (
        "n_measurements",
        "n_areas",
        "n_total_pixels",
        "n_retained_pixels",
        "n_source_rows",
        "n_unique_files",
        "n_unique_samples",
        "n_unique_spots",
        "has_block_",
    )
    ordered: list[str] = []
    for column in preferred:
        if column in frame.columns and column not in ordered:
            ordered.append(column)
    for column in frame.columns:
        if column in ordered or column in feature_columns:
            continue
        if any(column.startswith(prefix) for prefix in qc_prefixes):
            ordered.append(column)
    for column in frame.columns:
        if column in ordered or column in feature_columns:
            continue
        ordered.append(column)
    ordered.extend(feature_columns)
    return ordered


def _validate_subject_feature_matrix(frame: pd.DataFrame, filename: str, feature_columns: list[str]) -> None:
    if "subject_id" not in frame.columns:
        raise ValueError(f"{filename} is missing subject_id")
    if frame["subject_id"].astype(str).duplicated().any():
        raise ValueError(f"{filename} contains duplicated subject_id values")
    if not feature_columns:
        raise ValueError(f"{filename} is missing spectral feature columns")
    if filename == "subject_feature_matrix_whole.parquet":
        required = {"1300", "2850"}
        found = {column.split("__", 1)[0] for column in feature_columns}
        if found != required:
            raise ValueError(
                "Whole-spectrum subject feature matrix must preserve both 1300__ and 2850__ feature identities"
            )


def build_subject_feature_matrices(
    inputs: "SubjectInputs",
    *,
    cfg: dict | None = None,
    data_source: DataSource | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    cfg = config if cfg is None else cfg
    subject_spectra = inputs.subject_spectra.copy()
    subject_spectra["subject_id"] = subject_spectra["subject_id"].astype(str)

    clinical = _clinical_table(cfg, data_source)
    provenance = _subject_source_provenance(inputs, cfg, data_source)
    base = (
        subject_spectra.drop(columns=["condition", "gender", "group"], errors="ignore")
        .merge(clinical, on="subject_id", how="left")
        .merge(provenance, on="subject_id", how="left", suffixes=("", "_provenance"))
    )

    spectral_columns = [
        column for column in subject_spectra.columns if column.startswith(SPECTRAL_PREFIXES)
    ]
    bc_columns = [
        column for column in subject_spectra.columns if column.startswith(SPECTRAL_BC_PREFIXES)
    ]
    non_spectral = [
        column
        for column in base.columns
        if column not in spectral_columns
        and column not in bc_columns
        and not column.endswith("_provenance")
    ]

    block_features = {
        "1300": [column for column in spectral_columns if column.startswith("1300__")],
        "2850": [column for column in spectral_columns if column.startswith("2850__")],
        "whole": spectral_columns,
    }

    matrices: dict[str, pd.DataFrame] = {}
    for block, features in block_features.items():
        frame = base[non_spectral + features].copy()
        frame = frame.loc[:, _ordered_columns(frame, features)]
        filename = f"subject_feature_matrix_{block}.parquet"
        _validate_subject_feature_matrix(frame, filename, features)
        matrices[block] = frame

    if bc_columns:
        bc_frame = base[["subject_id"] + bc_columns].copy()
        _validate_subject_feature_matrix(bc_frame, "subject_feature_matrix_whole_bc.parquet", bc_columns)
        matrices["whole_bc"] = bc_frame

    return matrices, provenance


def write_subject_feature_exports(
    inputs: "SubjectInputs",
    *,
    output_dir: str | Path | None = None,
    cfg: dict | None = None,
    data_source: DataSource | None = None,
) -> dict[str, Path]:
    cfg = config if cfg is None else cfg
    target_dir = get_cross_region_phase_dir("subject_features") if output_dir is None else Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    matrices, provenance = build_subject_feature_matrices(inputs, cfg=cfg, data_source=data_source)
    paths = {
        "1300": target_dir / "subject_feature_matrix_1300.parquet",
        "2850": target_dir / "subject_feature_matrix_2850.parquet",
        "whole": target_dir / "subject_feature_matrix_whole.parquet",
    }
    if "whole_bc" in matrices:
        paths["whole_bc"] = target_dir / "subject_feature_matrix_whole_bc.parquet"
    for block, path in paths.items():
        matrices[block].to_parquet(path, index=False)

    provenance_path = target_dir / "subject_feature_provenance.csv"
    provenance.to_csv(provenance_path, index=False)

    outputs = {
        "subject_feature_matrix_1300": paths["1300"],
        "subject_feature_matrix_2850": paths["2850"],
        "subject_feature_matrix_whole": paths["whole"],
        "subject_feature_provenance": provenance_path,
    }
    if "whole_bc" in paths:
        outputs["subject_feature_matrix_whole_bc"] = paths["whole_bc"]
    return outputs
