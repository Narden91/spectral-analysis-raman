from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src import config as global_config, resolve_project_path
from src.data_preparation.cohorts import build_materialized_cohorts, write_cohort_tables
from src.data_preparation.manifest import (
    MANIFEST_VERSION,
    SCHEMA_VERSION,
    config_hash,
    file_signature,
    hash_file,
    iso_utc_now,
    write_manifest,
)
from src.fusion.area_pipeline import process_subject_areas
from src.io.file_crawler import build_file_index, disambiguate_subject_ids
from src.io.txt_reader import read_txt_spectrum
from src.metadata.clinical import load_clinical_data, merge_index_with_clinical
from src.preprocessing.pipeline import run_preprocessing

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MaterializedStore:
    store_type: str
    path: Path
    manifest: dict[str, Any]


def _cfg(cfg: dict | None = None) -> dict:
    return global_config if cfg is None else cfg


def _data_cfg(cfg: dict) -> dict:
    return cfg.get("data", {})


def _store_dir(store_type: str, cfg: dict) -> Path:
    if store_type == "parsed":
        return resolve_project_path(_data_cfg(cfg).get("parsed_dir", "data/parsed"))
    if store_type == "preprocessed":
        return resolve_project_path(_data_cfg(cfg).get("preprocessed_dir", "data/preprocessed"))
    raise ValueError(f"Unsupported store_type: {store_type}")


def _raw_text_dir(cfg: dict) -> Path:
    return resolve_project_path(_data_cfg(cfg).get("raw_text_dir", "data/raw"))


def _clinical_data_path(cfg: dict) -> Path:
    return resolve_project_path(_data_cfg(cfg).get("clinical_data_path", "data/clinical_data.csv"))


def _code_version() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _measurement_id(row: pd.Series) -> str:
    range_value = str(int(float(row["spectral_range_cm-1"])))
    return "__".join(
        [
            str(row["subject_id"]),
            str(row["slide_number"]),
            range_value,
            f"meas{int(row['measurement_number'])}",
        ]
    )


def _file_id(row: pd.Series) -> str:
    relative = str(Path(row["filepath"]).as_posix()).lower()
    return hash_file_id(relative)


def hash_file_id(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _prepare_file_index(cfg: dict) -> pd.DataFrame:
    legacy_root = cfg.setdefault("data", {}).get("root_dir")
    if not legacy_root:
        cfg["data"]["root_dir"] = str(_raw_text_dir(cfg))
    frame = disambiguate_subject_ids(build_file_index(measurement="all")).copy()
    if frame.empty:
        return frame
    frame["filepath"] = frame["filepath"].map(lambda value: str(Path(value)))
    frame = frame.sort_values(
        ["subject_id", "slide_number", "spectral_range_cm-1", "measurement_number", "filepath"],
        kind="mergesort",
    ).reset_index(drop=True)
    frame["file_id"] = frame.apply(_file_id, axis=1)
    frame["measurement_id"] = frame.apply(_measurement_id, axis=1)
    frame["spectral_range"] = frame["spectral_range_cm-1"].map(lambda value: str(int(float(value))))
    frame["subject_id"] = frame["subject_id"].astype(str)
    frame["area_id"] = frame["area_id"].astype(str)
    frame["source_file_path"] = frame["filepath"].map(lambda value: str(Path(value).as_posix()))
    frame["source_file_name"] = frame["filename"].astype(str)
    frame["raw_file_signature"] = frame["filepath"].map(hash_file)
    return frame


def _range_key(range_code: str) -> str:
    return "fingerprint" if str(range_code) == "1300" else "ch_stretch"


def _common_axis(vectors: list[np.ndarray]) -> np.ndarray:
    if not vectors:
        return np.array([], dtype=float)
    reference = np.asarray(vectors[0], dtype=float)
    return reference


def _resample_vector(source_axis: np.ndarray, target_axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    if source_axis.shape == target_axis.shape and np.allclose(source_axis, target_axis):
        return np.asarray(values, dtype=float)
    return np.interp(target_axis, source_axis, np.asarray(values, dtype=float))


def _read_raw_measurement_vectors(file_index: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, dict[str, list[Any]]] = {
        "1300": {"axes": [], "rows": []},
        "2850": {"axes": [], "rows": []},
    }
    for _, row in file_index.iterrows():
        range_code = str(int(float(row["spectral_range_cm-1"])))
        wavenumbers, _, _, spectra, _ = read_txt_spectrum(row["filepath"])
        flat = np.asarray(spectra, dtype=float)
        if flat.ndim == 1:
            flat = flat.reshape(1, -1)
        elif flat.ndim > 2:
            flat = flat.reshape(-1, flat.shape[-1])
        grouped[range_code]["axes"].append(np.asarray(wavenumbers, dtype=float))
        grouped[range_code]["rows"].append(
            {
                "measurement_id": row["measurement_id"],
                "subject_id": str(row["subject_id"]),
                "file_id": row["file_id"],
                "area_id": str(row["area_id"]),
                "axis": np.asarray(wavenumbers, dtype=float),
                "vector": np.nanmedian(flat, axis=0),
            }
        )
    output: dict[str, dict[str, np.ndarray]] = {}
    for range_code, payload in grouped.items():
        axes = payload["axes"]
        target_axis = _common_axis(axes)
        rows = payload["rows"]
        if target_axis.size == 0:
            output[range_code] = {
                "spectra": np.empty((0, 0)),
                "wavenumbers": np.array([], dtype=float),
                "measurement_ids": np.array([], dtype=str),
                "subject_ids": np.array([], dtype=str),
                "file_ids": np.array([], dtype=str),
                "area_ids": np.array([], dtype=str),
            }
            continue
        vectors = np.vstack(
            [_resample_vector(row["axis"], target_axis, row["vector"]) for row in rows]
        )
        output[range_code] = {
            "spectra": vectors,
            "wavenumbers": target_axis,
            "measurement_ids": np.asarray([row["measurement_id"] for row in rows], dtype=str),
            "subject_ids": np.asarray([row["subject_id"] for row in rows], dtype=str),
            "file_ids": np.asarray([row["file_id"] for row in rows], dtype=str),
            "area_ids": np.asarray([row["area_id"] for row in rows], dtype=str),
        }
    return output


def _preprocess_measurement_vectors(file_index: pd.DataFrame) -> tuple[dict[str, dict[str, np.ndarray]], pd.DataFrame]:
    grouped: dict[str, dict[str, list[Any]]] = {
        "1300": {"axes": [], "rows": []},
        "2850": {"axes": [], "rows": []},
    }
    area_rows: list[dict[str, Any]] = []
    for _, row in file_index.iterrows():
        range_code = str(int(float(row["spectral_range_cm-1"])))
        wavenumbers, _, _, spectra, _ = read_txt_spectrum(row["filepath"])
        flat = np.asarray(spectra, dtype=float)
        if flat.ndim == 1:
            flat = flat.reshape(1, -1)
        elif flat.ndim > 2:
            flat = flat.reshape(-1, flat.shape[-1])
        result = run_preprocessing(
            flat,
            np.asarray(wavenumbers, dtype=float),
            region=_range_key(range_code),
            screen_dead=True,
            suppress_cosmic=None,
            correct_bl=True,
            smooth=None,
            interpolate=False,
            filter_quality=None,
            normalize=True,
            norm_for="eda",
            record_stages=False,
        )
        valid = np.asarray(result.spectra, dtype=float)
        valid_area = valid.size > 0
        retained_pixels = int(valid.shape[0]) if valid_area else 0
        area_rows.append(
            {
                "measurement_id": row["measurement_id"],
                "file_id": row["file_id"],
                "subject_id": str(row["subject_id"]),
                "area_id": str(row["area_id"]),
                "spectral_range": range_code,
                "valid_area": bool(valid_area),
                "n_retained_pixels": retained_pixels,
            }
        )
        if not valid_area:
            continue
        grouped[range_code]["axes"].append(np.asarray(result.wavenumbers, dtype=float))
        grouped[range_code]["rows"].append(
            {
                "measurement_id": row["measurement_id"],
                "subject_id": str(row["subject_id"]),
                "file_id": row["file_id"],
                "area_id": str(row["area_id"]),
                "axis": np.asarray(result.wavenumbers, dtype=float),
                "vector": np.nanmedian(valid, axis=0),
            }
        )
    output: dict[str, dict[str, np.ndarray]] = {}
    for range_code, payload in grouped.items():
        target_axis = _common_axis(payload["axes"])
        rows = payload["rows"]
        if target_axis.size == 0:
            output[range_code] = {
                "spectra": np.empty((0, 0)),
                "wavenumbers": np.array([], dtype=float),
                "measurement_ids": np.array([], dtype=str),
                "subject_ids": np.array([], dtype=str),
                "file_ids": np.array([], dtype=str),
                "area_ids": np.array([], dtype=str),
            }
            continue
        vectors = np.vstack(
            [_resample_vector(row["axis"], target_axis, row["vector"]) for row in rows]
        )
        output[range_code] = {
            "spectra": vectors,
            "wavenumbers": target_axis,
            "measurement_ids": np.asarray([row["measurement_id"] for row in rows], dtype=str),
            "subject_ids": np.asarray([row["subject_id"] for row in rows], dtype=str),
            "file_ids": np.asarray([row["file_id"] for row in rows], dtype=str),
            "area_ids": np.asarray([row["area_id"] for row in rows], dtype=str),
        }
    return output, pd.DataFrame(area_rows)


def _build_area_metadata(file_index: pd.DataFrame, area_flags: pd.DataFrame | None = None) -> pd.DataFrame:
    columns = [
        "area_id",
        "subject_id",
        "file_id",
        "measurement_id",
        "spectral_range_cm-1",
        "slide_number",
        "measurement_number",
        "gender",
        "is_control",
    ]
    metadata = file_index[[column for column in columns if column in file_index.columns]].copy()
    metadata = metadata.rename(columns={"spectral_range_cm-1": "spectral_range"})
    metadata["subject_id"] = metadata["subject_id"].astype(str)
    metadata["file_id"] = metadata["file_id"].astype(str)
    metadata["measurement_id"] = metadata["measurement_id"].astype(str)
    metadata["area_id"] = metadata["area_id"].astype(str)
    metadata["spectral_range"] = metadata["spectral_range"].map(lambda value: str(int(float(value))))
    if area_flags is not None and not area_flags.empty:
        area_flags = area_flags.copy()
        for column in ("subject_id", "file_id", "measurement_id", "area_id"):
            area_flags[column] = area_flags[column].astype(str)
        area_flags["spectral_range"] = area_flags["spectral_range"].astype(str)
        metadata = metadata.merge(
            area_flags,
            on=["measurement_id", "file_id", "subject_id", "area_id", "spectral_range"],
            how="left",
        )
    return metadata.reset_index(drop=True)


def _area_npz_payload(spectra_payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    feature_names = np.asarray([f"{shift:.1f}" for shift in spectra_payload["wavenumbers"]], dtype=str)
    return {
        "area_summary": spectra_payload["spectra"],
        "area_ids": spectra_payload["area_ids"],
        "measurement_ids": spectra_payload["measurement_ids"],
        "subject_ids": spectra_payload["subject_ids"],
        "feature_names": feature_names,
    }


def _whole_area_store_from_ranges(
    payload_1300: dict[str, np.ndarray],
    payload_2850: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    rows_1300 = {
        area_id: index
        for index, area_id in enumerate(payload_1300["area_ids"].astype(str))
    }
    rows_2850 = {
        area_id: index
        for index, area_id in enumerate(payload_2850["area_ids"].astype(str))
    }
    paired_area_ids = sorted(set(rows_1300) & set(rows_2850))
    if not paired_area_ids:
        return {
            "values": np.empty((0, 0), dtype=float),
            "area_ids": np.array([], dtype=str),
            "measurement_ids": np.array([], dtype=str),
            "subject_ids": np.array([], dtype=str),
            "feature_names": np.array([], dtype=str),
        }
    values = []
    measurement_ids = []
    subject_ids = []
    for area_id in paired_area_ids:
        left_idx = rows_1300[area_id]
        right_idx = rows_2850[area_id]
        values.append(
            np.concatenate(
                [
                    payload_1300["spectra"][left_idx],
                    payload_2850["spectra"][right_idx],
                ]
            )
        )
        measurement_ids.append(
            f"{payload_1300['measurement_ids'][left_idx]}|{payload_2850['measurement_ids'][right_idx]}"
        )
        subject_ids.append(str(payload_1300["subject_ids"][left_idx]))
    feature_names = np.asarray(
        [f"1300__{shift:.1f}" for shift in payload_1300["wavenumbers"]]
        + [f"2850__{shift:.1f}" for shift in payload_2850["wavenumbers"]],
        dtype=str,
    )
    return {
        "values": np.vstack(values),
        "area_ids": np.asarray(paired_area_ids, dtype=str),
        "measurement_ids": np.asarray(measurement_ids, dtype=str),
        "subject_ids": np.asarray(subject_ids, dtype=str),
        "feature_names": feature_names,
    }


def _whole_area_store_preprocessed(file_index: pd.DataFrame) -> dict[str, np.ndarray]:
    rows: list[dict[str, Any]] = []
    feature_names: np.ndarray | None = None
    for subject_id, subject_df in file_index.groupby("subject_id", sort=False):
        area_results = process_subject_areas(subject_df, persist_artifacts=False)
        for result in area_results:
            values = np.asarray(result["area_mean"], dtype=float)
            if feature_names is None:
                feature_names = np.asarray(
                    [f"1300__{float(shift):.1f}" for shift in np.asarray(result["wn_1300"], dtype=float)]
                    + [f"2850__{float(shift):.1f}" for shift in np.asarray(result["wn_2850"], dtype=float)],
                    dtype=str,
                )
            rows.append(
                {
                    "area_id": str(result["area_id"]),
                    "measurement_id": f"{result['subject_id']}__{result['area_id']}",
                    "subject_id": str(subject_id),
                    "values": values,
                }
            )
    if not rows:
        return {
            "values": np.empty((0, 0), dtype=float),
            "area_ids": np.array([], dtype=str),
            "measurement_ids": np.array([], dtype=str),
            "subject_ids": np.array([], dtype=str),
            "feature_names": np.array([], dtype=str),
        }
    return {
        "values": np.vstack([row["values"] for row in rows]),
        "area_ids": np.asarray([row["area_id"] for row in rows], dtype=str),
        "measurement_ids": np.asarray([row["measurement_id"] for row in rows], dtype=str),
        "subject_ids": np.asarray([row["subject_id"] for row in rows], dtype=str),
        "feature_names": feature_names if feature_names is not None else np.array([], dtype=str),
    }


def _write_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **payload)


def _write_tables(
    store_dir: Path,
    file_index: pd.DataFrame,
    clinical_subjects: pd.DataFrame,
    area_metadata: pd.DataFrame,
    *,
    write_csv_mirrors: bool,
) -> list[Path]:
    written: list[Path] = []
    tables = {
        "file_index.parquet": file_index,
        "clinical_subjects.parquet": clinical_subjects,
        "area_metadata.parquet": area_metadata,
    }
    for filename, frame in tables.items():
        path = store_dir / filename
        frame.to_parquet(path, index=False)
        written.append(path)
        if write_csv_mirrors:
            csv_path = path.with_suffix(".csv")
            frame.to_csv(csv_path, index=False)
            written.append(csv_path)
    return written


def _input_signatures(cfg: dict, file_index: pd.DataFrame) -> dict[str, dict[str, Any]]:
    inputs: dict[str, str | Path] = {
        str(_clinical_data_path(cfg)): _clinical_data_path(cfg),
    }
    for filepath in sorted(file_index["filepath"].astype(str).unique().tolist()):
        inputs[filepath] = Path(filepath)
    return {
        name: file_signature(path).to_dict()
        for name, path in inputs.items()
    }


def _generated_signatures(root: Path) -> dict[str, dict[str, Any]]:
    files = {
        str(path.relative_to(root)): path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    return {
        name: file_signature(path).to_dict()
        for name, path in files.items()
    }


def _build_manifest(
    *,
    store_type: str,
    store_dir: Path,
    source: dict[str, Any],
    cfg: dict,
    file_index: pd.DataFrame,
    area_metadata: pd.DataFrame,
    spectra_payloads: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    return {
        "manifest_version": MANIFEST_VERSION,
        "schema_version": SCHEMA_VERSION,
        "store_type": store_type,
        "created_at": iso_utc_now(),
        "source": source,
        "table_format": "parquet",
        "array_format": "npz",
        "input_signatures": _input_signatures(cfg, file_index),
        "generated_file_signatures": _generated_signatures(store_dir),
        "preprocessing_config_hash": config_hash(cfg.get("preprocessing", {})),
        "materialization_config_hash": config_hash(cfg.get("data", {})),
        "code_version": _code_version(),
        "counts": {
            "subjects": int(file_index["subject_id"].astype(str).nunique()),
            "files": int(file_index["file_id"].astype(str).nunique()),
            "measurements": int(file_index["measurement_id"].astype(str).nunique()),
            "areas": int(area_metadata["area_id"].astype(str).nunique()) if not area_metadata.empty else 0,
            "spectra_1300": int(spectra_payloads["1300"]["spectra"].shape[0]),
            "spectra_2850": int(spectra_payloads["2850"]["spectra"].shape[0]),
        },
    }


def _atomic_replace(store_dir: Path, writer: callable) -> MaterializedStore:
    store_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{store_dir.name}_", dir=store_dir.parent))
    try:
        materialized = writer(staging_dir)
        backup_dir = None
        if store_dir.exists():
            backup_dir = Path(tempfile.mkdtemp(prefix=f".{store_dir.name}_backup_", dir=store_dir.parent))
            shutil.rmtree(backup_dir, ignore_errors=True)
            store_dir.replace(backup_dir)
        staging_dir.replace(store_dir)
        if backup_dir is not None:
            shutil.rmtree(backup_dir, ignore_errors=True)
        return materialized
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def build_parsed_store(cfg: dict | None = None, *, force: bool = False) -> MaterializedStore:
    config = _cfg(cfg)
    store_dir = _store_dir("parsed", config)

    def _write(staging_dir: Path) -> MaterializedStore:
        file_index = _prepare_file_index(config)
        clinical = load_clinical_data(data_index=file_index, write_curated=False)
        merged = merge_index_with_clinical(file_index, clinical)
        spectra_payloads = _read_raw_measurement_vectors(file_index)
        area_metadata = _build_area_metadata(file_index)
        write_csv_mirrors = bool(_data_cfg(config).get("write_csv_mirrors", True))
        _write_tables(staging_dir, file_index, clinical, area_metadata, write_csv_mirrors=write_csv_mirrors)
        _write_npz(staging_dir / "spectra_1300.npz", {key: value for key, value in spectra_payloads["1300"].items() if key != "area_ids"})
        _write_npz(staging_dir / "spectra_2850.npz", {key: value for key, value in spectra_payloads["2850"].items() if key != "area_ids"})
        _write_npz(staging_dir / "area_summaries_1300.npz", _area_npz_payload(spectra_payloads["1300"]))
        _write_npz(staging_dir / "area_summaries_2850.npz", _area_npz_payload(spectra_payloads["2850"]))
        _write_npz(
            staging_dir / "whole_area_store.npz",
            _whole_area_store_from_ranges(spectra_payloads["1300"], spectra_payloads["2850"]),
        )
        cohorts = build_materialized_cohorts(merged)
        write_cohort_tables(staging_dir / "cohort", cohorts, write_csv_mirrors=write_csv_mirrors)
        manifest = _build_manifest(
            store_type="parsed",
            store_dir=staging_dir,
            source={
                "type": "raw_text",
                "path": str(_raw_text_dir(config)),
                "manifest_hash": None,
            },
            cfg=config,
            file_index=file_index,
            area_metadata=area_metadata,
            spectra_payloads=spectra_payloads,
        )
        manifest = write_manifest(staging_dir / "manifest.json", manifest)
        return MaterializedStore("parsed", store_dir, manifest)

    if force or store_dir.exists():
        return _atomic_replace(store_dir, _write)
    return _atomic_replace(store_dir, _write)


def build_preprocessed_store(cfg: dict | None = None, *, force: bool = False) -> MaterializedStore:
    config = _cfg(cfg)
    parsed_store = build_parsed_store(config, force=False)
    parsed_manifest_hash = parsed_store.manifest.get("manifest_hash")
    parsed_dir = _store_dir("parsed", config)
    store_dir = _store_dir("preprocessed", config)

    def _write(staging_dir: Path) -> MaterializedStore:
        file_index = pd.read_parquet(parsed_dir / "file_index.parquet")
        clinical = pd.read_parquet(parsed_dir / "clinical_subjects.parquet")
        spectra_payloads, area_flags = _preprocess_measurement_vectors(file_index)
        area_metadata = _build_area_metadata(file_index, area_flags)
        retained_ids = set(spectra_payloads["1300"]["measurement_ids"].astype(str)) | set(
            spectra_payloads["2850"]["measurement_ids"].astype(str)
        )
        retained_index = file_index[file_index["measurement_id"].astype(str).isin(retained_ids)].copy()
        retained_merged = merge_index_with_clinical(retained_index, clinical)
        write_csv_mirrors = bool(_data_cfg(config).get("write_csv_mirrors", True))
        _write_tables(staging_dir, retained_index, clinical, area_metadata, write_csv_mirrors=write_csv_mirrors)
        _write_npz(staging_dir / "spectra_1300.npz", {key: value for key, value in spectra_payloads["1300"].items() if key != "area_ids"})
        _write_npz(staging_dir / "spectra_2850.npz", {key: value for key, value in spectra_payloads["2850"].items() if key != "area_ids"})
        _write_npz(staging_dir / "area_summaries_1300.npz", _area_npz_payload(spectra_payloads["1300"]))
        _write_npz(staging_dir / "area_summaries_2850.npz", _area_npz_payload(spectra_payloads["2850"]))
        _write_npz(staging_dir / "whole_area_store.npz", _whole_area_store_preprocessed(retained_index))
        cohorts = build_materialized_cohorts(retained_merged)
        write_cohort_tables(staging_dir / "cohort", cohorts, write_csv_mirrors=write_csv_mirrors)
        manifest = _build_manifest(
            store_type="preprocessed",
            store_dir=staging_dir,
            source={
                "type": "parsed",
                "path": str(parsed_dir),
                "manifest_hash": parsed_manifest_hash,
            },
            cfg=config,
            file_index=retained_index,
            area_metadata=area_metadata,
            spectra_payloads=spectra_payloads,
        )
        manifest = write_manifest(staging_dir / "manifest.json", manifest)
        return MaterializedStore("preprocessed", store_dir, manifest)

    if force or store_dir.exists():
        return _atomic_replace(store_dir, _write)
    return _atomic_replace(store_dir, _write)
