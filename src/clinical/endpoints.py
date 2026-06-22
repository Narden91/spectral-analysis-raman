from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src import config as global_config
from src import get_run_metadata_dir, resolve_project_path
from src.metadata.clinical import resolve_clinical_csv_path

_ENDPOINT_ALIASES: dict[str, tuple[str, ...]] = {
    "subject_id": ("subject_id", "participant_id"),
    "moca_baseline": (
        "moca_baseline_total",
        "moca_total_baseline",
        "moca baseline total [test]",
    ),
    "moca_3m": (
        "moca_3m_total",
        "moca_total_3m",
        "moca +3 months total [test]",
        "moca_3_months_total",
    ),
    "rbans_total": (
        "rbans_total",
        "rbans_total_3m",
        "rbans +3 months total scale",
        "rbans_total_3_months",
    ),
    "age": ("age", "age_years"),
    "aspects_score": ("aspects_score", "aspects scale", "aspects_scale_value"),
    "sex": ("sex", "gender"),
    "diagnosis": ("diagnosis_category", "diagnosis_text", "diagnosis", "diagnosis_raw"),
    "lesion_location": ("lesion_location", "diagnosis_category", "diagnosis_text"),
}

_REQUIRED_ENDPOINT_COLUMNS = (
    "subject_id",
    "moca_baseline",
    "moca_3m",
    "rbans_total",
    "age",
    "aspects_score",
    "sex",
    "diagnosis",
    "lesion_location",
)


@dataclass(frozen=True)
class ClinicalEndpointResult:
    source_path: Path
    output_dir: Path
    clinical_subject_table: pd.DataFrame
    clinical_endpoint_summary: pd.DataFrame
    clinical_missingness_report: pd.DataFrame
    clinical_group_counts: pd.DataFrame
    clinical_column_map: dict[str, str]


def _cfg(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    return global_config if cfg is None else cfg


def _endpoint_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("analysis", {}).get("clinical_endpoints", {})


def _normalize_columns(columns: pd.Index) -> dict[str, str]:
    return {str(column).strip().lower(): str(column) for column in columns}


def _candidate_names(endpoint_name: str, endpoint_cfg: dict[str, Any]) -> list[str]:
    config_key_map = {
        "subject_id": "subject_id_col",
        "moca_baseline": "moca_baseline_col",
        "moca_3m": "moca_3m_col",
        "rbans_total": "rbans_col",
        "age": "age_col",
        "aspects_score": "aspects_col",
        "sex": "sex_col",
        "diagnosis": "diagnosis_col",
        "lesion_location": "lesion_location_col",
    }
    candidates: list[str] = []
    configured = endpoint_cfg.get(config_key_map[endpoint_name])
    if configured:
        candidates.append(str(configured))
    candidates.extend(_ENDPOINT_ALIASES.get(endpoint_name, ()))

    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(str(candidate))
    return ordered


def _resolve_column_name(data: pd.DataFrame, endpoint_name: str, endpoint_cfg: dict[str, Any]) -> str:
    normalized_columns = _normalize_columns(data.columns)
    candidates = _candidate_names(endpoint_name, endpoint_cfg)
    for candidate in candidates:
        match = normalized_columns.get(candidate.strip().lower())
        if match is not None:
            return match
    raise ValueError(
        f"Could not map required clinical column '{endpoint_name}'. "
        f"Tried: {', '.join(candidates)}"
    )


def _normalize_subject_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _parse_numeric(series: pd.Series, canonical_name: str) -> pd.Series:
    parsed = pd.to_numeric(series, errors="coerce")
    if parsed.notna().sum() == 0:
        raise ValueError(f"Clinical endpoint column '{canonical_name}' could not be parsed as numeric.")
    return parsed


def _normalize_sex(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    mapped = normalized.replace(
        {
            "f": "female",
            "female": "female",
            "m": "male",
            "male": "male",
        }
    )
    return mapped.where(mapped.isin(["female", "male"]))


def _derive_lesion_group(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    posterior_mask = text.str.contains("posterior|cerebell|brainstem", na=False)
    left_mask = text.str.contains("left", na=False)
    right_mask = text.str.contains("right", na=False)

    grouped = pd.Series(pd.NA, index=series.index, dtype="string")
    grouped.loc[posterior_mask] = "posterior"
    grouped.loc[~posterior_mask & left_mask] = "left"
    grouped.loc[~posterior_mask & ~left_mask & right_mask] = "right"
    grouped.loc[grouped.isna() & text.notna()] = "other"
    return grouped


def _missingness_report(subject_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in subject_table.columns:
        missing = int(subject_table[column].isna().sum())
        rows.append(
            {
                "column": column,
                "n_missing": missing,
                "missing_rate": float(missing / max(len(subject_table), 1)),
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_rate", "column"], ascending=[False, True]).reset_index(drop=True)


def _group_counts(subject_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in ("moca_severity_group", "delta_moca_group", "lesion_location_group"):
        counts = subject_table[column].fillna("missing").value_counts(dropna=False).sort_index()
        for group_name, count in counts.items():
            rows.append(
                {
                    "group_variable": column,
                    "group_name": str(group_name),
                    "n_subjects": int(count),
                }
            )
    return pd.DataFrame(rows)


def _summary_table(subject_table: pd.DataFrame, source_path: Path, endpoint_cfg: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_path": str(source_path),
                "n_subjects": int(len(subject_table)),
                "n_unique_subject_ids": int(subject_table["subject_id"].nunique()),
                "severity_threshold": float(endpoint_cfg.get("severity_threshold", 20)),
                "decline_threshold": float(endpoint_cfg.get("decline_threshold", 0)),
            }
        ]
    )


def _default_output_dir(output_dir: str | Path | None = None) -> Path:
    if output_dir is not None:
        resolved = resolve_project_path(output_dir)
        assert resolved is not None
        return resolved
    return get_run_metadata_dir() / "clinical_endpoints"


def build_clinical_endpoints(
    *,
    cfg: dict[str, Any] | None = None,
    clinical_csv: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> ClinicalEndpointResult:
    cfg = _cfg(cfg)
    endpoint_cfg = _endpoint_cfg(cfg)
    source_path = resolve_clinical_csv_path(clinical_csv or endpoint_cfg.get("clinical_table"))
    if source_path is None:
        raise ValueError("Clinical endpoint input CSV could not be resolved.")

    data = pd.read_csv(source_path, encoding="utf-8-sig")
    resolved_columns = {
        endpoint_name: _resolve_column_name(data, endpoint_name, endpoint_cfg)
        for endpoint_name in _REQUIRED_ENDPOINT_COLUMNS
    }

    subject_table = pd.DataFrame(
        {
            "subject_id": _normalize_subject_id(data[resolved_columns["subject_id"]]),
            "moca_baseline": _parse_numeric(data[resolved_columns["moca_baseline"]], "moca_baseline"),
            "moca_3m": _parse_numeric(data[resolved_columns["moca_3m"]], "moca_3m"),
            "rbans_total": _parse_numeric(data[resolved_columns["rbans_total"]], "rbans_total"),
            "age": _parse_numeric(data[resolved_columns["age"]], "age"),
            "aspects_score": _parse_numeric(data[resolved_columns["aspects_score"]], "aspects_score"),
            "sex": _normalize_sex(data[resolved_columns["sex"]]),
            "diagnosis": data[resolved_columns["diagnosis"]].astype("string").str.strip(),
            "lesion_location": data[resolved_columns["lesion_location"]].astype("string").str.strip(),
        }
    )
    subject_table = subject_table.dropna(subset=["subject_id"]).copy()
    subject_table = subject_table[subject_table["subject_id"].str.len() > 0].copy()
    if subject_table["subject_id"].duplicated().any():
        duplicates = subject_table.loc[subject_table["subject_id"].duplicated(), "subject_id"].tolist()
        raise ValueError(f"Clinical endpoint table has duplicate subject_id values: {duplicates[:5]}")

    severity_threshold = float(endpoint_cfg.get("severity_threshold", 20))
    decline_threshold = float(endpoint_cfg.get("decline_threshold", 0))
    subject_table["delta_moca"] = subject_table["moca_3m"] - subject_table["moca_baseline"]
    subject_table["moca_severity_group"] = pd.Series(pd.NA, index=subject_table.index, dtype="string")
    subject_table.loc[subject_table["moca_baseline"] < severity_threshold, "moca_severity_group"] = "low_moca"
    subject_table.loc[subject_table["moca_baseline"] >= severity_threshold, "moca_severity_group"] = "high_moca"
    subject_table["delta_moca_group"] = pd.Series(pd.NA, index=subject_table.index, dtype="string")
    subject_table.loc[subject_table["delta_moca"] < decline_threshold, "delta_moca_group"] = "decline"
    subject_table.loc[subject_table["delta_moca"] >= decline_threshold, "delta_moca_group"] = "stable_or_improved"
    subject_table["lesion_location_group"] = _derive_lesion_group(subject_table["lesion_location"])

    return ClinicalEndpointResult(
        source_path=source_path,
        output_dir=_default_output_dir(output_dir),
        clinical_subject_table=subject_table,
        clinical_endpoint_summary=_summary_table(subject_table, source_path, endpoint_cfg),
        clinical_missingness_report=_missingness_report(subject_table),
        clinical_group_counts=_group_counts(subject_table),
        clinical_column_map=resolved_columns,
    )


def write_clinical_endpoint_outputs(result: ClinicalEndpointResult) -> dict[str, Path]:
    result.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "clinical_endpoint_summary.csv": result.clinical_endpoint_summary,
        "clinical_subject_table.csv": result.clinical_subject_table,
        "clinical_missingness_report.csv": result.clinical_missingness_report,
        "clinical_group_counts.csv": result.clinical_group_counts,
    }
    written: dict[str, Path] = {}
    for filename, table in outputs.items():
        path = result.output_dir / filename
        table.to_csv(path, index=False)
        written[filename] = path

    column_map_path = result.output_dir / "clinical_column_map.json"
    column_map_path.write_text(json.dumps(result.clinical_column_map, indent=2), encoding="utf-8")
    written["clinical_column_map.json"] = column_map_path
    return written


def build_and_write_clinical_endpoints(
    *,
    cfg: dict[str, Any] | None = None,
    clinical_csv: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> ClinicalEndpointResult:
    result = build_clinical_endpoints(cfg=cfg, clinical_csv=clinical_csv, output_dir=output_dir)
    write_clinical_endpoint_outputs(result)
    return result
