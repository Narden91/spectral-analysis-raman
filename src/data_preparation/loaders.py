from __future__ import annotations

import numpy as np
import pandas as pd

from src import resolve_cfg as _cfg
from src.data_preparation.materialize import _prepare_file_index, _read_raw_measurement_vectors
from src.data_preparation.store import DataSource, get_active_data_source, resolve_data_source
from src.metadata.clinical import load_clinical_data


def _source(data_source: DataSource | None = None) -> DataSource:
    if data_source is not None:
        return data_source
    active = get_active_data_source()
    if active is not None:
        return active
    return resolve_data_source()


def load_file_index(data_source: DataSource | None = None) -> pd.DataFrame:
    source = _source(data_source)
    if source.source_name == "raw_text":
        return _prepare_file_index(_cfg())
    return pd.read_parquet(source.source_path / "file_index.parquet")


def load_clinical_subjects(data_source: DataSource | None = None) -> pd.DataFrame:
    source = _source(data_source)
    if source.source_name == "raw_text":
        return load_clinical_data(data_index=load_file_index(source), write_curated=False)
    return pd.read_parquet(source.source_path / "clinical_subjects.parquet")


def load_area_metadata(data_source: DataSource | None = None) -> pd.DataFrame:
    source = _source(data_source)
    if source.source_name == "raw_text":
        file_index = load_file_index(source)
        metadata = file_index.rename(columns={"spectral_range_cm-1": "spectral_range"}).copy()
        return metadata
    return pd.read_parquet(source.source_path / "area_metadata.parquet")


def load_spectra(data_source: DataSource | None, spectral_range: str) -> dict[str, np.ndarray]:
    source = _source(data_source)
    filename = f"spectra_{spectral_range}.npz"
    if source.source_name == "raw_text":
        payload = _read_raw_measurement_vectors(load_file_index(source))[spectral_range]
        return {
            "spectra": payload["spectra"],
            "wavenumbers": payload["wavenumbers"],
            "measurement_ids": payload["measurement_ids"],
            "subject_ids": payload["subject_ids"],
            "file_ids": payload["file_ids"],
        }
    with np.load(source.source_path / filename, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def load_area_summaries(data_source: DataSource | None, spectral_range: str) -> dict[str, np.ndarray]:
    source = _source(data_source)
    filename = f"area_summaries_{spectral_range}.npz"
    if source.source_name == "raw_text":
        payload = _read_raw_measurement_vectors(load_file_index(source))[spectral_range]
        return {
            "area_summary": payload["spectra"],
            "area_ids": payload["area_ids"],
            "measurement_ids": payload["measurement_ids"],
            "subject_ids": payload["subject_ids"],
            "feature_names": np.asarray([f"{shift:.1f}" for shift in payload["wavenumbers"]], dtype=str),
        }
    with np.load(source.source_path / filename, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def load_whole_area_store(data_source: DataSource | None = None) -> dict[str, np.ndarray]:
    source = _source(data_source)
    if source.source_name == "raw_text":
        area_1300 = _read_raw_measurement_vectors(load_file_index(source))["1300"]
        area_2850 = _read_raw_measurement_vectors(load_file_index(source))["2850"]
        from src.data_preparation.materialize import _whole_area_store_from_ranges

        return _whole_area_store_from_ranges(area_1300, area_2850)
    with np.load(source.source_path / "whole_area_store.npz", allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def load_cohort(
    data_source: DataSource | None,
    cohort_kind: str,
    cohort_name: str,
) -> pd.DataFrame:
    source = _source(data_source)
    if source.source_name == "raw_text":
        raise RuntimeError("Cohort parquet tables are available only for parsed/preprocessed sources")
    return pd.read_parquet(source.source_path / "cohort" / cohort_name / f"{cohort_kind}.parquet")

