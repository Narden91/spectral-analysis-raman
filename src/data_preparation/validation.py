from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_preparation.manifest import (
    MANIFEST_VERSION,
    SCHEMA_VERSION,
    hash_file,
    load_manifest,
)


PARQUET_FILES = ("file_index.parquet", "clinical_subjects.parquet", "area_metadata.parquet")
NPZ_FILES = (
    "spectra_1300.npz",
    "spectra_2850.npz",
    "area_summaries_1300.npz",
    "area_summaries_2850.npz",
    "whole_area_store.npz",
)
SPECTRA_KEYS = ("spectra", "wavenumbers", "measurement_ids", "subject_ids", "file_ids")
AREA_KEYS = ("area_summary", "area_ids", "measurement_ids", "subject_ids", "feature_names")
WHOLE_KEYS = ("values", "area_ids", "measurement_ids", "subject_ids", "feature_names")


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    manifest: dict[str, Any] | None = None
    manifest_hash: str | None = None
    is_stale: bool = False


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], label: str, errors: list[str]) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        errors.append(f"{label} missing required columns: {', '.join(missing)}")


def _validate_npz_keys(path: Path, required_keys: tuple[str, ...], errors: list[str]) -> None:
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in required_keys if key not in data.files]
        if missing:
            errors.append(f"{path.name} missing required keys: {', '.join(missing)}")


def _validate_npz_shapes(path: Path, label: str, errors: list[str]) -> None:
    with np.load(path, allow_pickle=False) as data:
        if label == "spectra":
            spectra = np.asarray(data["spectra"])
            n_rows = spectra.shape[0]
            if spectra.ndim != 2:
                errors.append(f"{path.name} spectra must be 2D")
            for key in ("measurement_ids", "subject_ids", "file_ids"):
                if np.asarray(data[key]).shape[0] != n_rows:
                    errors.append(f"{path.name} {key} length does not match spectra rows")
        elif label == "area":
            values = np.asarray(data["area_summary"])
            n_rows = values.shape[0]
            if values.ndim != 2:
                errors.append(f"{path.name} area_summary must be 2D")
            for key in ("area_ids", "measurement_ids", "subject_ids"):
                if np.asarray(data[key]).shape[0] != n_rows:
                    errors.append(f"{path.name} {key} length does not match area_summary rows")
        elif label == "whole":
            values = np.asarray(data["values"])
            n_rows = values.shape[0]
            if values.ndim != 2:
                errors.append(f"{path.name} values must be 2D")
            for key in ("area_ids", "measurement_ids", "subject_ids"):
                if np.asarray(data[key]).shape[0] != n_rows:
                    errors.append(f"{path.name} {key} length does not match values rows")


def _validate_generated_signatures(store_path: Path, manifest: dict[str, Any], errors: list[str]) -> None:
    generated = manifest.get("generated_file_signatures", {})
    for relative_name, signature in generated.items():
        candidate = store_path / relative_name
        if not candidate.is_file():
            errors.append(f"Manifest references missing generated file: {relative_name}")
            continue
        if signature.get("sha256") != hash_file(candidate):
            errors.append(f"Generated file hash mismatch for {relative_name}")


def _validate_manifest_counts(store_path: Path, manifest: dict[str, Any], errors: list[str]) -> None:
    try:
        file_index = pd.read_parquet(store_path / "file_index.parquet")
        clinical = pd.read_parquet(store_path / "clinical_subjects.parquet")
        area_metadata = pd.read_parquet(store_path / "area_metadata.parquet")
    except Exception as exc:
        errors.append(f"Failed to read parquet tables for count validation: {exc}")
        return

    counts = manifest.get("counts", {})
    expected = {
        "subjects": int(clinical["subject_id"].nunique()) if "subject_id" in clinical.columns else len(clinical),
        "files": int(file_index["file_id"].nunique()) if "file_id" in file_index.columns else len(file_index),
        "measurements": int(file_index["measurement_id"].nunique()) if "measurement_id" in file_index.columns else len(file_index),
        "areas": int(area_metadata["area_id"].nunique()) if "area_id" in area_metadata.columns else len(area_metadata),
    }
    for key, value in expected.items():
        if int(counts.get(key, -1)) != value:
            errors.append(f"Manifest count mismatch for {key}: manifest={counts.get(key)} actual={value}")


def _validate_cohorts(store_path: Path, clinical: pd.DataFrame, errors: list[str]) -> None:
    subject_ids = set(clinical["subject_id"].astype(str).tolist()) if "subject_id" in clinical.columns else set()
    cohort_root = store_path / "cohort"
    if not cohort_root.exists():
        errors.append("Missing cohort directory")
        return
    for cohort_file in cohort_root.rglob("*.parquet"):
        cohort = pd.read_parquet(cohort_file)
        _require_columns(cohort, ("subject_id", "cohort_name", "group_label"), str(cohort_file.relative_to(store_path)), errors)
        subject_series = cohort.get("subject_id", pd.Series(dtype=object))
        present_subject_ids = {
            str(value)
            for value in subject_series.dropna().tolist()
            if str(value).strip() != ""
        }
        unknown = sorted(present_subject_ids - subject_ids)
        if unknown:
            errors.append(f"{cohort_file.name} references unknown subject_id values: {', '.join(unknown[:10])}")


def validate_store(path: str | Path, expected_store_type: str, config: dict) -> ValidationResult:
    store_path = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    manifest_path = store_path / "manifest.json"
    if not manifest_path.is_file():
        return ValidationResult(False, errors=["Missing manifest.json"])

    manifest = load_manifest(manifest_path)
    if int(manifest.get("manifest_version", -1)) != MANIFEST_VERSION:
        errors.append(f"Unsupported manifest_version: {manifest.get('manifest_version')}")
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        errors.append(f"Unsupported schema_version: {manifest.get('schema_version')}")
    if manifest.get("store_type") != expected_store_type:
        errors.append(
            f"Store type mismatch: expected {expected_store_type}, found {manifest.get('store_type')}"
        )

    for filename in PARQUET_FILES + NPZ_FILES:
        if not (store_path / filename).is_file():
            errors.append(f"Missing required file: {filename}")

    if errors:
        return ValidationResult(False, errors=errors, warnings=warnings, manifest=manifest, manifest_hash=manifest.get("manifest_hash"))

    file_index = pd.read_parquet(store_path / "file_index.parquet")
    clinical = pd.read_parquet(store_path / "clinical_subjects.parquet")
    area_metadata = pd.read_parquet(store_path / "area_metadata.parquet")

    _require_columns(
        file_index,
        ("file_id", "subject_id", "measurement_id", "source_file_path", "source_file_name", "spectral_range", "raw_file_signature"),
        "file_index.parquet",
        errors,
    )
    _require_columns(clinical, ("subject_id",), "clinical_subjects.parquet", errors)
    _require_columns(
        area_metadata,
        ("area_id", "subject_id", "file_id", "measurement_id", "spectral_range"),
        "area_metadata.parquet",
        errors,
    )

    for column in ("subject_id",):
        if column in clinical.columns and clinical[column].astype(str).duplicated().any():
            errors.append(f"Duplicate {column} in clinical_subjects.parquet")
    if "measurement_id" in file_index.columns and file_index["measurement_id"].astype(str).duplicated().any():
        errors.append("Duplicate measurement_id in file_index.parquet")

    _validate_npz_keys(store_path / "spectra_1300.npz", SPECTRA_KEYS, errors)
    _validate_npz_keys(store_path / "spectra_2850.npz", SPECTRA_KEYS, errors)
    _validate_npz_keys(store_path / "area_summaries_1300.npz", AREA_KEYS, errors)
    _validate_npz_keys(store_path / "area_summaries_2850.npz", AREA_KEYS, errors)
    _validate_npz_keys(store_path / "whole_area_store.npz", WHOLE_KEYS, errors)

    _validate_npz_shapes(store_path / "spectra_1300.npz", "spectra", errors)
    _validate_npz_shapes(store_path / "spectra_2850.npz", "spectra", errors)
    _validate_npz_shapes(store_path / "area_summaries_1300.npz", "area", errors)
    _validate_npz_shapes(store_path / "area_summaries_2850.npz", "area", errors)
    _validate_npz_shapes(store_path / "whole_area_store.npz", "whole", errors)

    _validate_generated_signatures(store_path, manifest, errors)
    _validate_manifest_counts(store_path, manifest, errors)
    _validate_cohorts(store_path, clinical, errors)

    is_stale = False
    if config.get("data", {}).get("require_valid_manifest", True):
        policy = str(config.get("data", {}).get("stale_manifest_policy", "error")).strip().lower()
        if policy not in {"error", "rebuild"}:
            warnings.append(f"Unknown stale_manifest_policy={policy!r}; treating as error")
            policy = "error"
        source = manifest.get("source", {})
        source_type = source.get("type")
        source_path = source.get("path")
        source_manifest_hash = source.get("manifest_hash")
        if source_type in {"parsed", "preprocessed"} and source_path and source_manifest_hash:
            candidate_manifest = Path(source_path) / "manifest.json"
            if candidate_manifest.is_file():
                current = load_manifest(candidate_manifest)
                if current.get("manifest_hash") != source_manifest_hash:
                    is_stale = True
                    if policy == "error":
                        errors.append("Store is stale relative to its source manifest")
        input_signatures = manifest.get("input_signatures", {})
        for relative_name, expected in input_signatures.items():
            input_path = Path(relative_name)
            if not input_path.is_file():
                input_path = Path(config.get("data", {}).get("raw_text_dir", "data/raw")).parent / relative_name
            if input_path.is_file() and expected.get("sha256") != hash_file(input_path):
                is_stale = True
                if policy == "error":
                    errors.append(f"Store is stale relative to input {relative_name}")

    return ValidationResult(
        is_valid=not errors,
        errors=errors,
        warnings=warnings,
        manifest=manifest,
        manifest_hash=manifest.get("manifest_hash"),
        is_stale=is_stale,
    )
