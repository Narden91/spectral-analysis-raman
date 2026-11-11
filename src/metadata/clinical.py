from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import resolve_cfg, resolve_project_path
from src.io.file_crawler import disambiguate_subject_ids
from src.metadata.clinical_master import is_clinical_master, load_master_for_linkage

logger = logging.getLogger(__name__)

_KNOWN_CLINICAL_COLUMNS = {
    "subject_id",
    "participant_id",
    "gender",
    "gender_binary",
    "is_female",
    "age",
    "age_years",
    "diagnosis",
    "diagnosis_text",
    "diagnosis_category",
    "rbans +3 months total scale",
    "rbans_total_3m",
    "moca baseline total [test]",
    "moca_total_baseline",
    "moca +3 months total [test]",
    "moca_total_3m",
    "aspects scale",
    "aspects_score",
    "death",
}

_CLINICAL_ALIAS_SPECS = {
    "age_years": ["age", "age_years"],
    "rbans_total_3_months": ["RBANS +3 months TOTAL SCALE", "rbans_total_3m"],
    "moca_baseline_total": ["MOCA baseline Total [TEST]", "moca_total_baseline"],
    "moca_3_months_total": ["MOCA +3 months Total [TEST]", "moca_total_3m"],
    "aspects_scale_value": ["ASPECTS scale", "aspects_score"],
}

_CURATED_COLUMNS = [
    "subject_id",
    "subject_id_raw",
    "condition",
    "is_control",
    "gender",
    "gender_binary",
    "clinical_present",
    "gender_matches_data",
    "group_matches_data",
    "clinical_condition",
    "diagnosis_raw",
    "diagnosis_normalized",
    "diagnosis_family",
    "age_years",
    "moca_baseline_total",
    "moca_3_months_total",
    "delta_moca",
    "aspects_scale_value",
    "rbans_total_3_months",
    "death_binary",
    "n_files",
    "n_slides",
    "n_measurements",
    "n_areas",
    "n_paired_areas",
    "available_ranges",
    "has_range_1300",
    "has_range_2850",
    "has_both_ranges",
    "gender_conflict_in_data",
    "group_conflict_in_data",
]

_MISSING_TOKENS = {
    "",
    "-",
    "\u2212",  # U+2212 minus sign used as placeholder in some exports
    "#n/a",    # Excel's not-available token
    "nd",      # not detected
    "x",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "missing",
    "unknown",
    "not available",
}


def resolve_clinical_csv_path(
    explicit_path: str | Path | None = None,
    cfg: dict | None = None,
) -> Path:
    """Return the explicit clinical CSV, else ``data.clinical_data_path``.

    A missing file raises instead of returning None: a silent fallback once
    let analyses run against the wrong clinical table or without one.
    """
    raw_path = explicit_path or resolve_cfg(cfg).get("data", {}).get("clinical_data_path")
    if not raw_path:
        raise FileNotFoundError("No clinical CSV configured: set data.clinical_data_path.")
    path = resolve_project_path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"Clinical CSV not found: {path}")
    return path


def load_clinical_data(
    clinical_csv: str | Path | None = None,
    *,
    data_index: pd.DataFrame,
) -> pd.DataFrame:
    """Load clinical metadata and reconcile it with the file index into a subject-level table."""
    path = resolve_clinical_csv_path(clinical_csv)
    clinical = _load_raw_clinical_data(path)

    subject_inventory = build_subject_inventory(data_index=data_index)
    if subject_inventory.empty:
        return clinical

    curated = _build_curated_subject_table(subject_inventory, clinical)

    logger.info(
        "Loaded canonical clinical metadata: %d subjects, %d columns",
        len(curated),
        curated.shape[1],
    )
    return curated


def build_subject_inventory(data_index: pd.DataFrame) -> pd.DataFrame:
    """Summarize the reorganized data tree into one row per subject."""
    if data_index.empty:
        return pd.DataFrame(columns=["subject_id"])

    base = disambiguate_subject_ids(data_index)
    base["subject_id"] = base["subject_id"].astype(str).str.strip()
    base["subject_id_raw"] = base.get("subject_id_raw", base["subject_id"]).astype(str).str.strip()
    if "gender" in base.columns:
        base["gender"] = base["gender"].map(_normalize_gender_value)

    rows: list[dict] = []
    for subject_id, subject_df in base.groupby("subject_id", sort=False, dropna=False):
        gender_values = _normalized_string_values(subject_df.get("gender"))
        group_values = _normalized_bool_values(subject_df.get("is_control"))
        range_values = _normalized_range_values(subject_df.get("spectral_range_cm-1"))

        n_paired_areas = 0
        if {"area_id", "spectral_range_cm-1"}.issubset(subject_df.columns):
            range_per_area = subject_df.groupby("area_id", sort=False)["spectral_range_cm-1"].apply(
                _normalized_range_values
            )
            n_paired_areas = int(
                range_per_area.map(lambda values: {"1300", "2850"}.issubset(set(values))).sum()
            )

        gender_value = gender_values[0] if gender_values else pd.NA
        is_control = group_values[0] if group_values else pd.NA
        rows.append(
            {
                "subject_id": str(subject_id),
                "subject_id_raw": str(subject_df["subject_id_raw"].iloc[0]),
                "gender": gender_value,
                "gender_binary": _gender_to_binary(gender_value),
                "is_control": is_control,
                "condition": _condition_from_bool(is_control),
                "n_files": int(subject_df["filepath"].nunique()) if "filepath" in subject_df.columns else len(subject_df),
                "n_slides": int(subject_df["slide_number"].nunique()) if "slide_number" in subject_df.columns else pd.NA,
                "n_measurements": int(subject_df["measurement_number"].nunique()) if "measurement_number" in subject_df.columns else pd.NA,
                "n_areas": int(subject_df["area_id"].nunique()) if "area_id" in subject_df.columns else pd.NA,
                "n_paired_areas": n_paired_areas,
                "available_ranges": "|".join(range_values),
                "has_range_1300": "1300" in range_values,
                "has_range_2850": "2850" in range_values,
                "has_both_ranges": {"1300", "2850"}.issubset(set(range_values)),
                "gender_conflict_in_data": len(gender_values) > 1,
                "group_conflict_in_data": len(group_values) > 1,
            }
        )

    inventory = pd.DataFrame(rows)
    if not inventory.empty:
        inventory["gender_binary"] = inventory["gender_binary"].astype("Int64")
    return inventory


def merge_index_with_clinical(
    df_index: pd.DataFrame,
    clinical_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join file index and clinical metadata on subject_id."""
    if df_index.empty:
        return df_index.copy()

    out = disambiguate_subject_ids(df_index)
    out["subject_id"] = out["subject_id"].astype(str).str.strip()

    if clinical_df.empty or "subject_id" not in clinical_df.columns:
        return out

    merged = out.merge(clinical_df, on="subject_id", how="left", suffixes=("", "_clinical"))
    return merged


def _load_raw_clinical_data(path: Path) -> pd.DataFrame:
    clinical = pd.read_csv(path, encoding="utf-8-sig", dtype={"subject_id": "string", "participant_id": "string"})
    if is_clinical_master(clinical.columns):
        return load_master_for_linkage(path)
    if clinical.empty:
        logger.warning("Clinical CSV is empty: %s", path)
        return pd.DataFrame(columns=["subject_id"])

    clinical = _standardize_columns(clinical)
    clinical = _normalize_missing_values(clinical)
    clinical["subject_id"] = clinical["subject_id"].astype(str).str.strip()
    clinical = clinical.dropna(subset=["subject_id"])
    clinical = clinical[clinical["subject_id"].str.len() > 0]
    clinical["gender_binary"] = _encode_gender_binary(clinical)
    clinical = _drop_zero_variance_columns(clinical, keep=_protected_clinical_columns(clinical.columns))
    clinical = clinical.drop_duplicates(subset=["subject_id"], keep="first")

    logger.info(
        "Loaded raw clinical metadata: %d rows, %d columns from %s",
        len(clinical),
        clinical.shape[1],
        path,
    )
    return clinical


def _build_curated_subject_table(
    subject_inventory: pd.DataFrame,
    clinical_df: pd.DataFrame,
) -> pd.DataFrame:
    if is_clinical_master(clinical_df.columns):
        return _build_master_subject_table(subject_inventory, clinical_df)
    curated = subject_inventory.copy()
    clinical = clinical_df.copy()
    clinical_cols: list[str] = []

    if not clinical.empty:
        clinical["subject_id"] = clinical["subject_id"].astype(str).str.strip()
        clinical["subject_id_raw"] = clinical["subject_id"]
        rename_map = {}
        if "gender" in clinical.columns:
            rename_map["gender"] = "gender_clinical_raw"
        if "gender_binary" in clinical.columns:
            rename_map["gender_binary"] = "gender_binary_clinical_raw"
        clinical = clinical.rename(columns=rename_map)
        clinical_cols = [col for col in clinical.columns if col not in {"subject_id", "subject_id_raw"}]
        curated = curated.merge(clinical[["subject_id_raw", *clinical_cols]], on="subject_id_raw", how="left")
        curated["clinical_present"] = curated[clinical_cols].notna().any(axis=1)
        curated["clinical_condition"] = _infer_clinical_condition(curated)
        curated = _prune_duplicate_clinical_matches(curated, clinical_cols)
    else:
        curated["clinical_present"] = False

    curated["clinical_present"] = curated["clinical_present"].fillna(False)
    curated["diagnosis_raw"] = _first_available_column(
        curated, ["Diagnosis", "diagnosis", "diagnosis_text", "diagnosis_raw"]
    )
    curated["diagnosis_normalized"] = _normalize_diagnosis_series(curated["diagnosis_raw"])
    curated["diagnosis_family"] = curated["diagnosis_normalized"].map(_diagnosis_family)
    curated["clinical_condition"] = _infer_clinical_condition(curated)
    curated["gender_matches_data"] = _compare_optional_text(
        _first_available_column(curated, ["gender_clinical_raw", "gender_clinical", "gender"]),
        curated["gender"],
    )
    curated["group_matches_data"] = _compare_optional_text(
        curated["clinical_condition"],
        curated["condition"],
    )
    curated = _quarantine_group_mismatched_clinical_rows(curated, clinical_cols)
    curated["gender_binary"] = curated["gender"].map(_gender_to_binary).astype("Int64")

    for alias, candidates in _CLINICAL_ALIAS_SPECS.items():
        curated[alias] = _extract_numeric_alias(curated, candidates)

    curated["delta_moca"] = curated["moca_3_months_total"] - curated["moca_baseline_total"]

    death_source = _first_available_column(curated, ["Death", "death"])
    curated["death_binary"] = _encode_death_binary(death_source)

    ordered = [col for col in _CURATED_COLUMNS if col in curated.columns]
    tail = [col for col in curated.columns if col not in ordered]
    return curated[ordered + tail]



def _build_master_subject_table(subject_inventory: pd.DataFrame, clinical: pd.DataFrame) -> pd.DataFrame:
    """Link explicit cohort/ID pairs and keep unverified clinical observations missing."""
    inventory = subject_inventory.copy()
    inventory["subject_id_raw"] = inventory["subject_id_raw"].astype("string").str.strip()
    master = clinical.copy()
    master["_master_is_control"] = master["cohort"].eq("control")
    master = master.loc[pd.to_numeric(master["has_spectra"], errors="raise").eq(1)].copy()
    overlap = set(inventory.columns).intersection(master.columns)
    master = master.rename(columns={column: f"{column}_clinical_raw" for column in overlap})
    curated = inventory.merge(
        master,
        left_on=["subject_id_raw", "is_control"],
        right_on=["spectral_subject_id", "_master_is_control"],
        how="left",
        validate="many_to_one",
        sort=False,
    ).drop(columns="_master_is_control")
    verified = curated["identity_verified"].eq(True).fillna(False)
    if curated.loc[verified].duplicated(["cohort", "spectral_subject_id"]).any():
        raise ValueError("Verified clinical identity maps to multiple spectral subjects")
    provenance = {
        "participant_key", "cohort", "spectral_subject_id", "has_spectra", "has_clinical_record",
        "spectral_link_status", "qc_flags", "identity_verified", "sex_source", "sex_spectral_folder",
    }
    clinical_columns = [column for column in master if column not in provenance and column in curated]
    for column in clinical_columns:
        curated[column] = curated[column].where(verified)
    curated["identity_verified"] = verified
    curated["clinical_present"] = verified
    curated["gender_clinical_raw"] = curated.get("sex", pd.Series(pd.NA, index=curated.index))
    curated["gender_matches_data"] = _compare_optional_text(curated["gender_clinical_raw"], curated["gender"])
    curated["clinical_condition"] = curated["cohort"].map({"stroke": "Disease", "control": "Control"}).where(verified)
    curated["group_matches_data"] = _compare_optional_text(curated["clinical_condition"], curated["condition"])
    curated["diagnosis_raw"] = curated.get("diagnosis_text", pd.Series(pd.NA, index=curated.index))
    curated["diagnosis_normalized"] = _normalize_diagnosis_series(curated["diagnosis_raw"])
    curated["diagnosis_family"] = curated["diagnosis_normalized"].map(_diagnosis_family)
    aliases = {
        "age_years": "age_years", "rbans_total_3_months": "rbans_total_3m",
        "moca_baseline_total": "moca_baseline", "moca_3_months_total": "moca_3m",
        "aspects_scale_value": "aspects",
    }
    for alias, source in aliases.items():
        source = f"{source}_clinical_raw" if source in overlap else source
        curated[alias] = pd.to_numeric(curated.get(source, pd.Series(pd.NA, index=curated.index)), errors="raise")
    curated["delta_moca"] = curated["moca_3_months_total"] - curated["moca_baseline_total"]
    death = pd.to_numeric(curated.get("death", pd.Series(pd.NA, index=curated.index)), errors="raise")
    if not death.dropna().isin([0, 1]).all():
        raise ValueError("Clinical-master death values must be 0, 1, or missing")
    curated["death_binary"] = death.astype("Int64")
    ordered = [column for column in _CURATED_COLUMNS if column in curated]
    return curated[ordered + [column for column in curated if column not in ordered]]

def _prune_duplicate_clinical_matches(curated: pd.DataFrame, clinical_cols: list[str]) -> pd.DataFrame:
    if "subject_id_raw" not in curated.columns:
        return curated

    out = curated.copy()
    dup_mask = out["clinical_present"].astype(bool) & out["subject_id_raw"].astype(str).duplicated(keep=False)
    if not dup_mask.any():
        return out

    for _, dup_df in out.loc[dup_mask].groupby("subject_id_raw", sort=False):
        keep_idx = _resolve_duplicate_clinical_match(dup_df)
        if keep_idx is None:
            clear_idx = dup_df.index
        else:
            clear_idx = dup_df.index.difference(pd.Index([keep_idx]))
        out.loc[clear_idx, clinical_cols] = pd.NA
        out.loc[clear_idx, "clinical_present"] = False

    return out


def _quarantine_group_mismatched_clinical_rows(curated: pd.DataFrame, clinical_cols: list[str]) -> pd.DataFrame:
    """Drop clinical fields for subjects whose folder-derived condition disagrees with the
    clinical-CSV-derived condition. A raw-data folder path (Control/Disease) is ground truth
    for group membership; a mismatch means the clinical row was joined by a colliding
    subject_id_raw and belongs to a different person, so it must not contribute any clinical
    value (diagnosis, age, MoCA, etc.) to this subject.
    """
    if "group_matches_data" not in curated.columns:
        return curated

    mismatch_mask = (curated["group_matches_data"] == False).fillna(False).astype(bool)  # noqa: E712 (nullable boolean; explicit False only, NA/True left alone)
    if not mismatch_mask.any():
        return curated

    out = curated.copy()
    derived_cols = [c for c in ("diagnosis_raw", "diagnosis_normalized", "diagnosis_family", "clinical_condition") if c in out.columns]
    quarantine_cols = [c for c in {*clinical_cols, *derived_cols} if c in out.columns]
    out.loc[mismatch_mask, quarantine_cols] = pd.NA
    out.loc[mismatch_mask, "clinical_present"] = False
    out.loc[mismatch_mask, "group_matches_data"] = pd.NA
    if "gender_matches_data" in out.columns:
        out.loc[mismatch_mask, "gender_matches_data"] = pd.NA

    logger.warning(
        "Quarantined clinical data for %d subject(s) whose folder-derived condition "
        "disagrees with the clinical CSV (colliding subject_id_raw, different person): %s",
        int(mismatch_mask.sum()),
        sorted(out.loc[mismatch_mask, "subject_id"].astype(str).tolist()),
    )
    return out


def _resolve_duplicate_clinical_match(dup_df: pd.DataFrame) -> object | None:
    candidates = pd.Index(dup_df.index)

    clinical_conditions = dup_df["clinical_condition"].dropna().astype(str).str.strip().str.lower().unique().tolist()
    if len(clinical_conditions) == 1:
        condition_matches = dup_df.index[
            dup_df["condition"].astype("string").str.strip().str.lower() == clinical_conditions[0]
        ]
        if len(condition_matches) == 1:
            return condition_matches[0]
        if len(condition_matches) > 0:
            candidates = pd.Index(condition_matches)

    clinical_gender = _clinical_gender_series(dup_df).dropna().astype(str).str.strip().str.lower().unique().tolist()
    if len(clinical_gender) == 1:
        gender_col_name = "gender"
        gender_matches = dup_df.index[
            dup_df[gender_col_name].astype("string").str.strip().str.lower() == clinical_gender[0]
        ]
        gender_matches = pd.Index([idx for idx in gender_matches if idx in candidates])
        if len(gender_matches) == 1:
            return gender_matches[0]
        if len(gender_matches) > 0:
            candidates = gender_matches

    if len(candidates) == 1:
        return candidates[0]
    return None


def _clinical_gender_series(df: pd.DataFrame) -> pd.Series:
    if "gender_clinical_raw" in df.columns:
        return df["gender_clinical_raw"].map(_normalize_gender_value)
    if "Gender Female=1" in df.columns:
        raw = df["Gender Female=1"]
        numeric = pd.to_numeric(raw, errors="coerce")
        if numeric.notna().any() and set(numeric.dropna().unique().tolist()).issubset({0.0, 1.0}):
            return numeric.map({1.0: "female", 0.0: "male"})
        return raw.map(_normalize_gender_value)
    if "gender_binary_clinical_raw" in df.columns:
        numeric = pd.to_numeric(df["gender_binary_clinical_raw"], errors="coerce")
        return numeric.map({0.0: "female", 1.0: "male"})
    return pd.Series(pd.NA, index=df.index, dtype="object")


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    col_map = {c: c.strip() for c in data.columns}
    data = data.rename(columns=col_map)

    if "subject_id" not in data.columns:
        if "participant_id" not in data.columns:
            raise ValueError("Clinical table requires an explicit subject_id or participant_id column")
        data = data.rename(columns={"participant_id": "subject_id"})

    # Normalize "gender" header from CSV (already canonical)
    for column in data.columns:
        lower = column.lower()
        if lower == "gender" and column != "gender":
            data = data.rename(columns={column: "gender"})

    return data


def _normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    for column in data.columns:
        col = data[column]
        # pandas 3.0+ uses StringDtype ("str") instead of object for string columns;
        # check both to stay compatible with older pandas as well.
        is_str_like = pd.api.types.is_object_dtype(col) or pd.api.types.is_string_dtype(col)
        if is_str_like:
            normalized = col.astype(str).str.strip()
            lower = normalized.str.lower()
            data[column] = normalized.mask(lower.isin(_MISSING_TOKENS))
    return data


def _encode_gender_binary(df: pd.DataFrame) -> pd.Series:
    if "gender" not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")

    gender = df["gender"].astype(str).str.strip().str.lower()
    mapped = gender.map(
        {
            "female": 0,
            "f": 0,
            "male": 1,
            "m": 1,
        }
    )
    return mapped.astype("Int64")


def _drop_zero_variance_columns(df: pd.DataFrame, keep: set[str]) -> pd.DataFrame:
    data = df.copy()
    drop_cols: list[str] = []
    for column in data.columns:
        if column in keep:
            continue
        if data[column].dropna().nunique() <= 1:
            drop_cols.append(column)

    if drop_cols:
        logger.info(
            "Dropping %d zero-variance clinical column(s): %s",
            len(drop_cols),
            drop_cols,
        )
        data = data.drop(columns=drop_cols)

    return data


def _protected_clinical_columns(columns: pd.Index | list[str]) -> set[str]:
    protected = {"subject_id"}
    for column in columns:
        normalized = str(column).strip().lower()
        if normalized in _KNOWN_CLINICAL_COLUMNS:
            protected.add(str(column))
    return protected


def _normalize_gender_value(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    normalized = str(value).strip().lower()
    if not normalized or normalized in {"unknown", "nan", "none"}:
        return pd.NA
    if normalized in {"f", "female"}:
        return "female"
    if normalized in {"m", "male"}:
        return "male"
    return normalized


def _normalized_string_values(series: pd.Series | None) -> list[str]:
    if series is None:
        return []
    values = []
    for value in series:
        normalized = _normalize_gender_value(value)
        if pd.notna(normalized):
            values.append(str(normalized))
    return sorted(set(values))


def _normalized_bool_values(series: pd.Series | None) -> list[bool]:
    if series is None:
        return []
    values: list[bool] = []
    for value in series.dropna().tolist():
        values.append(bool(value))
    return sorted(set(values))


def _normalized_range_values(series: pd.Series | None) -> list[str]:
    if series is None:
        return []
    values: list[str] = []
    for value in series.dropna().tolist():
        try:
            values.append(str(int(float(value))))
        except (TypeError, ValueError):
            text = str(value).strip()
            if text:
                values.append(text)
    return sorted(set(values))


def _gender_to_binary(value: object) -> object:
    normalized = _normalize_gender_value(value)
    if normalized == "female":
        return 0
    if normalized == "male":
        return 1
    return pd.NA


def _condition_from_bool(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    return "Control" if bool(value) else "Disease"


def _first_available_column(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for column in candidates:
        if column in df.columns:
            return df[column]
    return pd.Series(pd.NA, index=df.index, dtype="object")


def _extract_numeric_alias(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for column in candidates:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _normalize_diagnosis_series(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    normalized = normalized.replace({"<NA>": pd.NA})
    normalized = normalized.str.replace(r"\s+", " ", regex=True)
    return normalized


def _diagnosis_family(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value)
    if "control" in text or "healthy" in text:
        return "control"
    if "stroke" in text or "ischemic" in text or "ischaemic" in text:
        return "ischemic_stroke"
    return "other"


def _infer_clinical_condition(df: pd.DataFrame) -> pd.Series:
    diagnosis = _first_available_column(df, ["diagnosis_normalized", "Diagnosis", "diagnosis", "diagnosis_raw"])
    normalized = _normalize_diagnosis_series(diagnosis)
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    control_mask = normalized.str.contains("control|healthy", case=False, na=False)
    disease_mask = normalized.str.contains(
        "stroke|ischemic|ischaemic|brainstem|cerebell|vascular",
        case=False,
        na=False,
    )
    out.loc[control_mask] = "Control"
    out.loc[disease_mask] = "Disease"
    return out


def _compare_optional_text(left: pd.Series, right: pd.Series) -> pd.Series:
    matches = pd.Series(pd.NA, index=left.index, dtype="boolean")
    left_norm = left.astype("string").str.strip().str.lower()
    right_norm = right.astype("string").str.strip().str.lower()
    mask = left_norm.notna() & right_norm.notna()
    matches.loc[mask] = left_norm.loc[mask] == right_norm.loc[mask]
    return matches


def _encode_death_binary(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    mapped = normalized.map({"yes": 1, "y": 1, "true": 1, "dead": 1, "no": 0, "n": 0, "false": 0, "alive": 0})
    return mapped.astype("Int64")
