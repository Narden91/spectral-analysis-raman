"""Structural and provenance validation for canonical Raman stores."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.constants import CANONICAL_REGION_BOUNDS, SPECTRAL_RANGES
from src.data_preparation.manifest import (
    MANIFEST_VERSION,
    SCHEMA_VERSION,
    config_hash,
    hash_file,
    hash_manifest,
    load_manifest,
)


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    manifest: dict[str, Any] | None = None
    manifest_hash: str | None = None
    is_stale: bool = False


def _required_files(store_type: str) -> tuple[str, ...]:
    tables = ("measurements.parquet", "subjects.parquet", "areas.parquet", "cohort")
    axes = tuple(f"wavenumbers_{code}.npy" for code in SPECTRAL_RANGES)
    if store_type == "parsed":
        return tables + ("pixels.parquet",) + tuple(
            name for code in SPECTRAL_RANGES for name in (f"raw_spectra_{code}.npy", f"native_wavenumbers_{code}.npy")
        )
    if store_type == "preprocessed":
        return tables + ("qc.parquet",) + axes + tuple(
            f"{basis}_{code}.npy" for basis in ("normalized", "pre_normalization") for code in SPECTRAL_RANGES
        )
    raise ValueError(f"Unsupported store type: {store_type}")


def _require_columns(frame: pd.DataFrame, names: tuple[str, ...], label: str, errors: list[str]) -> None:
    missing = [name for name in names if name not in frame]
    if missing:
        errors.append(f"{label} missing columns: {', '.join(missing)}")


def _matches_signature(path: Path, signature: dict[str, Any], *, strict: bool) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return False
    expected_hash = signature.get("sha256")
    same_stat = int(signature.get("size", -1)) == int(stat.st_size) and int(signature.get("mtime_ns", -1)) == int(stat.st_mtime_ns)
    if same_stat and not strict:
        return True
    return expected_hash is not None and expected_hash == hash_file(path)


def _validate_signatures(store_path: Path, manifest: dict[str, Any], errors: list[str], *, strict: bool) -> None:
    for relative, signature in manifest.get("generated_file_signatures", {}).items():
        path = store_path / relative
        if not _matches_signature(path, signature, strict=strict):
            errors.append(f"Generated artifact signature mismatch: {relative}")


def _validate_axis(path: Path, matrix_path: Path, errors: list[str]) -> None:
    axis = np.load(path, mmap_mode="r")
    matrix = np.load(matrix_path, mmap_mode="r")
    if axis.ndim != 1 or axis.size < 2 or not np.all(np.isfinite(axis)) or not np.all(np.diff(axis) > 0):
        errors.append(f"Invalid axis: {path.name}")
    if matrix.ndim != 2 or matrix.shape[1] != axis.size:
        errors.append(f"Matrix/axis shape mismatch: {matrix_path.name}")
    elif matrix.size and not np.all(np.isfinite(matrix)):
        errors.append(f"Non-finite spectral values: {matrix_path.name}")


def _validate_preprocessed_invariants(store_path: Path, code: str, config: dict, errors: list[str]) -> None:
    axis = np.asarray(np.load(store_path / f"wavenumbers_{code}.npy", mmap_mode="r"), dtype=float)
    region = "fingerprint" if code == "1300" else "ch_stretch"
    interpolation = config.get("preprocessing", {}).get("interpolation", {})
    lower, upper = interpolation.get("frozen_grids", {}).get(region, CANONICAL_REGION_BOUNDS[region])
    spacing = float(interpolation.get("target_spacing", 1.0))
    expected = np.arange(float(lower), float(upper) + spacing, spacing, dtype=float)
    if axis.shape != expected.shape or not np.allclose(axis, expected, rtol=0.0, atol=1e-8):
        errors.append(f"{code} axis does not match the frozen canonical grid")

    spectra = np.asarray(np.load(store_path / f"normalized_{code}.npy", mmap_mode="r"), dtype=float)
    normalization = config.get("preprocessing", {}).get("ranges", {}).get(region, {}).get("normalization", {})
    method = str(normalization.get("method", "none")).lower()
    if spectra.size and method == "snv":
        means = spectra.mean(axis=1)
        scales = spectra.std(axis=1, ddof=0)
        if not np.allclose(means, 0.0, rtol=0.0, atol=1e-8):
            errors.append(f"{code} normalized spectra violate the SNV mean invariant")
        if not np.allclose(scales, 1.0, rtol=1e-6, atol=1e-6):
            errors.append(f"{code} normalized spectra violate the SNV scale invariant")

    if spectra.size and method == "area":
        areas = np.trapezoid(np.abs(spectra), axis, axis=1)
        if areas.size and not np.allclose(areas, 1.0, rtol=1e-6, atol=1e-6):
            errors.append(f"{code} normalized spectra violate absolute-area normalization")


def _validate_qc_reasons(qc: pd.DataFrame, errors: list[str]) -> None:
    for reason in qc["qc_reason"].dropna().astype(str):
        tokens = [token.strip() for token in reason.split(";") if token.strip() and token.strip() != "ok"]
        if len(tokens) != len(set(tokens)):
            errors.append("qc_reason contains duplicated reason tokens")
            return


def _validate_native_axes(store_path: Path, measurements: pd.DataFrame, code: str, errors: list[str]) -> None:
    spectra = np.load(store_path / f"raw_spectra_{code}.npy", mmap_mode="r")
    axes = np.load(store_path / f"native_wavenumbers_{code}.npy", mmap_mode="r")
    if axes.ndim != 2 or axes.shape[1] < 2 or not np.all(np.isfinite(axes)) or not np.all(np.diff(axes, axis=1) > 0):
        errors.append(f"Invalid native axes: native_wavenumbers_{code}.npy")
        return
    if spectra.ndim != 2 or spectra.shape[1] != axes.shape[1]:
        errors.append(f"Raw/native-axis shape mismatch: {code}")
    rows = measurements.loc[measurements["spectral_range"].astype(str) == code]
    native_rows = rows["native_axis_row"].to_numpy(dtype=int)
    if len(native_rows) != len(rows) or len(native_rows) and (native_rows.min() < 0 or native_rows.max() >= len(axes) or len(np.unique(native_rows)) != len(rows)):
        errors.append(f"{code} native_axis_row does not index native axes one-to-one")


def _validate_relations(store_type: str, store_path: Path, errors: list[str]) -> None:
    measurements = pd.read_parquet(store_path / "measurements.parquet")
    subjects = pd.read_parquet(store_path / "subjects.parquet")
    areas = pd.read_parquet(store_path / "areas.parquet")
    _require_columns(measurements, ("measurement_id", "file_id", "subject_id", "area_id", "spectral_range", "raw_start", "raw_stop", "native_axis_row", "sample_id", "spot_id"), "measurements", errors)
    _require_columns(subjects, ("subject_id",), "subjects", errors)
    _require_columns(areas, ("measurement_id", "subject_id", "area_id", "spectral_range"), "areas", errors)
    if "measurement_id" in measurements and measurements["measurement_id"].astype(str).duplicated().any():
        errors.append("Duplicate measurement_id in measurements")
    if "subject_id" in subjects and subjects["subject_id"].astype(str).duplicated().any():
        errors.append("Duplicate subject_id in subjects")
    unknown = set(measurements.get("subject_id", pd.Series(dtype=str)).astype(str)) - set(subjects.get("subject_id", pd.Series(dtype=str)).astype(str))
    if unknown:
        errors.append("Measurements reference unknown subjects")
    if len(areas) != len(measurements) or set(areas.get("measurement_id", pd.Series(dtype=str)).astype(str)) != set(measurements.get("measurement_id", pd.Series(dtype=str)).astype(str)):
        errors.append("areas is not a one-row-per-measurement table")

    if store_type == "parsed":
        pixels = pd.read_parquet(store_path / "pixels.parquet")
        _require_columns(pixels, ("measurement_id", "spectral_range", "pixel_index", "raw_row", "x", "y"), "pixels", errors)
        if pixels.duplicated(["measurement_id", "pixel_index"]).any():
            errors.append("Duplicate pixel identity in pixels")
        for code in SPECTRAL_RANGES:
            _validate_native_axes(store_path, measurements, code, errors)
    else:
        qc = pd.read_parquet(store_path / "qc.parquet")
        _require_columns(qc, ("measurement_id", "spectral_range", "pixel_index", "retained", "normalized_row", "pre_normalization_row", "qc_reason"), "qc", errors)
        if qc.duplicated(["measurement_id", "pixel_index"]).any():
            errors.append("Duplicate pixel identity in qc")
        retained = qc.loc[qc.get("retained", pd.Series(False, index=qc.index)).astype(bool)]
        _validate_qc_reasons(qc, errors)
        for code in SPECTRAL_RANGES:
            subset = retained.loc[retained["spectral_range"].astype(str) == code]
            for column in ("normalized_row", "pre_normalization_row"):
                rows = subset[column].to_numpy(dtype=int)
                if len(rows) and (rows.min() != 0 or rows.max() != len(rows) - 1 or len(np.unique(rows)) != len(rows)):
                    errors.append(f"{code} {column} is not a dense retained-pixel index")


def _validate_cohorts(store_path: Path, subjects: pd.DataFrame, errors: list[str]) -> None:
    root = store_path / "cohort"
    if not root.is_dir():
        errors.append("Missing cohort directory")
        return
    subject_ids = set(subjects["subject_id"].astype(str))
    for path in root.rglob("*.parquet"):
        frame = pd.read_parquet(path)
        if "subject_id" not in frame:
            continue
        unknown = set(frame["subject_id"].dropna().astype(str)) - subject_ids
        if unknown:
            errors.append(f"{path.name} references unknown subjects")


def validate_store(path: str | Path, expected_store_type: str, config: dict, *, strict: bool = False) -> ValidationResult:
    store_path = Path(path)
    errors: list[str] = []
    manifest_path = store_path / "manifest.json"
    if not manifest_path.is_file():
        return ValidationResult(False, errors=["Missing manifest.json"])
    manifest = load_manifest(manifest_path)
    if manifest.get("manifest_hash") != hash_manifest(manifest):
        errors.append("Manifest hash mismatch")
    if int(manifest.get("manifest_version", -1)) != MANIFEST_VERSION:
        errors.append("Unsupported manifest version")
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        errors.append("Unsupported schema version")
    if manifest.get("store_type") != expected_store_type:
        errors.append(f"Store type mismatch: expected {expected_store_type}")
    for name in _required_files(expected_store_type):
        if not (store_path / name).exists():
            errors.append(f"Missing required artifact: {name}")
    if errors:
        return ValidationResult(False, errors=errors, manifest=manifest, manifest_hash=manifest.get("manifest_hash"))

    _validate_relations(expected_store_type, store_path, errors)
    for code in SPECTRAL_RANGES:
        if expected_store_type == "preprocessed":
            _validate_axis(store_path / f"wavenumbers_{code}.npy", store_path / f"normalized_{code}.npy", errors)
            _validate_axis(store_path / f"wavenumbers_{code}.npy", store_path / f"pre_normalization_{code}.npy", errors)
            _validate_preprocessed_invariants(store_path, code, config, errors)
    _validate_signatures(store_path, manifest, errors, strict=strict)
    subjects = pd.read_parquet(store_path / "subjects.parquet")
    _validate_cohorts(store_path, subjects, errors)

    stale = False
    if expected_store_type == "preprocessed" and manifest.get("preprocessing_config_hash") != config_hash(config.get("preprocessing", {})):
        stale = True
        errors.append("Store is stale relative to resolved preprocessing configuration")
    source = manifest.get("source", {})
    if source.get("type") == "parsed":
        source_path = Path(str(source.get("path", "")))
        source_manifest = source_path / "manifest.json"
        if not source_manifest.is_file() or load_manifest(source_manifest).get("manifest_hash") != source.get("manifest_hash"):
            stale = True
            errors.append("Store is stale relative to its parsed source")
        elif strict:
            source_validation = validate_store(source_path, "parsed", config, strict=True)
            if not source_validation.is_valid:
                stale = True
                errors.append("Parsed source fails strict validation: " + "; ".join(source_validation.errors[:3]))
    for raw_path, signature in manifest.get("input_signatures", {}).items():
        source_path = Path(raw_path)
        if not source_path.is_file() or not _matches_signature(source_path, signature, strict=strict):
            stale = True
            errors.append(f"Store is stale relative to input {raw_path}")
    return ValidationResult(not errors, errors, manifest, manifest.get("manifest_hash"), stale)
