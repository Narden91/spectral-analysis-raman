"""Read-only, grain-explicit views over the canonical stores."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_preparation.manifest import load_manifest
from src.data_preparation.store import DataSource, get_active_data_source, resolve_data_source

SPECTRAL_RANGES = ("1300", "2850")

# Per-process cache of derived reads, keyed by data-source identity so a
# rebuilt store (new manifest_hash) never serves stale entries. Cache scope is
# limited to table DataFrames, memmap handles, and small derived summaries
# (n_area x n_chan); never densified pixel-level matrices, which are the
# actual memory cost and must stay uncached.
_TABLE_CACHE: dict[tuple[str, str, str], pd.DataFrame] = {}
_ARRAY_CACHE: dict[tuple[str, str, str, str], tuple[np.ndarray, np.ndarray]] = {}
_SUMMARY_CACHE: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
_WHOLE_AREA_CACHE: dict[tuple[str, str], dict[str, np.ndarray]] = {}


def _source_key(source: DataSource) -> tuple[str, str]:
    return (str(source.source_path), str(source.manifest_hash))


def clear_loader_caches() -> None:
    """Drop all cached tables/arrays/summaries. Tests and store rebuilds only."""
    _TABLE_CACHE.clear()
    _ARRAY_CACHE.clear()
    _SUMMARY_CACHE.clear()
    _WHOLE_AREA_CACHE.clear()


def _source(data_source: DataSource | None = None) -> DataSource:
    return data_source or get_active_data_source() or resolve_data_source()


def _parsed_path(source: DataSource) -> Path:
    if source.source_name == "parsed":
        return source.source_path
    if source.source_name != "preprocessed":
        raise RuntimeError("Only parsed and preprocessed stores are supported")
    source_manifest = source.manifest or load_manifest(source.source_path / "manifest.json")
    return Path(source_manifest["source"]["path"])


def _cached_table(source: DataSource, filename: str) -> pd.DataFrame:
    key = (*_source_key(source), filename)
    cached = _TABLE_CACHE.get(key)
    if cached is None:
        cached = pd.read_parquet(source.source_path / filename)
        _TABLE_CACHE[key] = cached
    return cached


def load_file_index(data_source: DataSource | None = None) -> pd.DataFrame:
    return _cached_table(_source(data_source), "measurements.parquet")


def load_clinical_subjects(data_source: DataSource | None = None) -> pd.DataFrame:
    return _cached_table(_source(data_source), "subjects.parquet")


def load_area_metadata(data_source: DataSource | None = None) -> pd.DataFrame:
    return _cached_table(_source(data_source), "areas.parquet")


def _pixel_rows(source: DataSource, spectral_range: str, *, retained_only: bool) -> pd.DataFrame:
    code = str(spectral_range)
    if source.source_name == "parsed":
        pixels = pd.read_parquet(source.source_path / "pixels.parquet")
        return pixels.loc[pixels["spectral_range"].astype(str) == code].sort_values("raw_row")
    if source.source_name != "preprocessed":
        raise RuntimeError("Pixel rows require a materialized source")
    qc = pd.read_parquet(source.source_path / "qc.parquet")
    rows = qc.loc[qc["spectral_range"].astype(str) == code]
    if retained_only:
        rows = rows.loc[rows["retained"].astype(bool)].sort_values("normalized_row")
    return rows


def _spectra_filename(source: DataSource, spectral_range: str, basis: str) -> str:
    code = str(spectral_range)
    return f"raw_spectra_{code}.npy" if source.source_name == "parsed" else f"{basis}_{code}.npy"


def _row_column(source: DataSource, basis: str) -> str:
    return "raw_row" if source.source_name == "parsed" else f"{basis}_row"


def _load_array(source: DataSource, spectral_range: str, basis: str) -> tuple[np.ndarray, np.ndarray]:
    code = str(spectral_range)
    if source.source_name == "parsed":
        raise RuntimeError(
            "Parsed spectra retain measurement-specific native axes. "
            "Use load_parsed_measurement_spectra() or the canonical preprocessed store."
        )
    key = (*_source_key(source), code, basis)
    cached = _ARRAY_CACHE.get(key)
    if cached is None:
        spectra = np.load(source.source_path / _spectra_filename(source, code, basis), mmap_mode="r")
        axis = np.load(source.source_path / f"wavenumbers_{code}.npy", mmap_mode="r")
        cached = (spectra, axis)
        _ARRAY_CACHE[key] = cached
    return cached


def load_parsed_measurement_spectra(
    data_source: DataSource | None,
    measurement_id: str,
) -> dict[str, np.ndarray]:
    """Return raw pixels and their unmodified native axis for one acquisition."""
    source = _source(data_source)
    if source.source_name != "parsed":
        raise RuntimeError("load_parsed_measurement_spectra requires the parsed data source")
    measurements = load_file_index(source).set_index("measurement_id")
    row = measurements.loc[str(measurement_id)]
    if isinstance(row, pd.DataFrame):
        raise RuntimeError(f"Duplicate parsed measurement_id: {measurement_id}")
    code = str(row["spectral_range"])
    raw = np.load(source.source_path / f"raw_spectra_{code}.npy", mmap_mode="r")
    axes = np.load(source.source_path / f"native_wavenumbers_{code}.npy", mmap_mode="r")
    start, stop = int(row["raw_start"]), int(row["raw_stop"])
    return {
        "spectra": raw[start:stop],
        "wavenumbers": np.asarray(axes[int(row["native_axis_row"])], dtype=float),
        "measurement_id": np.asarray([str(measurement_id)]),
    }


def _feature_names(spectral_range: str, axis: np.ndarray) -> np.ndarray:
    prefix = f"{spectral_range}__" if spectral_range else ""
    return np.asarray([f"{prefix}{float(value):.1f}" for value in axis], dtype=str)


def load_spectra(data_source: DataSource | None, spectral_range: str) -> dict[str, np.ndarray]:
    source = _source(data_source)
    code = str(spectral_range)
    rows = _pixel_rows(source, code, retained_only=source.source_name == "preprocessed")
    spectra, axis = _load_array(source, code, basis="normalized")
    measurements = load_file_index(source).set_index("measurement_id")
    measurement_ids = rows["measurement_id"].astype(str)
    return {
        "spectra": spectra,
        "wavenumbers": axis,
        "measurement_ids": measurement_ids.to_numpy(),
        "subject_ids": rows["subject_id"].astype(str).to_numpy(),
        "file_ids": measurement_ids.map(measurements["file_id"]).astype(str).to_numpy(),
    }


def _summarize_rows(
    rows: pd.DataFrame,
    spectra: np.ndarray,
    row_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    usable_rows = rows.loc[rows[row_column] >= 0]
    groups = list(usable_rows.groupby("measurement_id", sort=False))
    measurement_ids = np.asarray([str(measurement_id) for measurement_id, _ in groups], dtype=str)
    summaries = [
        np.nanmedian(np.asarray(spectra[group[row_column].to_numpy(dtype=int)], dtype=float), axis=0)
        for _, group in groups
    ]
    if summaries:
        return measurement_ids, np.vstack(summaries)
    return measurement_ids, np.empty((0, spectra.shape[1]), dtype=float)


def _summary_payload(
    rows: pd.DataFrame,
    spectra: np.ndarray,
    axis: np.ndarray,
    row_column: str,
    measurements: pd.DataFrame,
) -> dict[str, np.ndarray]:
    measurement_ids, values = _summarize_rows(rows, spectra, row_column)
    metadata = measurements.reindex(measurement_ids)
    return {
        "area_summary": values,
        "area_ids": metadata["area_id"].astype(str).to_numpy(),
        "measurement_ids": measurement_ids,
        "subject_ids": metadata["subject_id"].astype(str).to_numpy(),
        "feature_names": _feature_names("", axis),
    }


def load_area_summaries(data_source: DataSource | None, spectral_range: str) -> dict[str, np.ndarray]:
    source = _source(data_source)
    code = str(spectral_range)
    key = (*_source_key(source), code)
    cached = _SUMMARY_CACHE.get(key)
    if cached is not None:
        return cached

    rows = _pixel_rows(source, code, retained_only=source.source_name == "preprocessed")
    measurements = load_file_index(source).set_index("measurement_id")
    spectra, axis = _load_array(source, code, basis="normalized")
    payload = _summary_payload(rows, spectra, axis, _row_column(source, "normalized"), measurements)
    if source.source_name == "preprocessed":
        baseline_corrected, _ = _load_array(source, code, basis="pre_normalization")
        payload["area_summary_bc"] = _summarize_rows(
            rows, baseline_corrected, _row_column(source, "pre_normalization")
        )[1]
    _SUMMARY_CACHE[key] = payload
    return payload


def _pair_key(frame: pd.DataFrame) -> pd.Series:
    mode = frame["acquisition_mode"].fillna("none").astype(str)
    return (
        frame["subject_id"].astype(str)
        + "__"
        + frame["slide_number"].astype(str)
        + "__"
        + frame["measurement_number"].astype(str)
        + "__"
        + mode
    )


def _paired_measurements(measurements: pd.DataFrame) -> Iterator[tuple[pd.Series, pd.Series]]:
    for _, group in measurements.groupby("pair_key", sort=False):
        by_range = {str(code): rows for code, rows in group.groupby("spectral_range", sort=False)}
        if set(by_range) != set(SPECTRAL_RANGES) or any(len(rows) != 1 for rows in by_range.values()):
            continue
        yield by_range["1300"].iloc[0], by_range["2850"].iloc[0]


def _coordinates_by_measurement(parsed: Path, source: DataSource) -> dict[str, pd.DataFrame]:
    pixels = pd.read_parquet(parsed / "pixels.parquet", columns=["measurement_id", "pixel_index", "x", "y"])
    qc = pd.read_parquet(
        source.source_path / "qc.parquet",
        columns=["measurement_id", "pixel_index", "normalized_row", "retained"],
    )
    retained = qc.loc[qc["retained"].astype(bool), ["measurement_id", "pixel_index", "normalized_row"]]
    resolved = retained.merge(pixels, on=["measurement_id", "pixel_index"], validate="one_to_one")
    return {
        str(measurement_id): group[["x", "y", "normalized_row"]]
        for measurement_id, group in resolved.groupby("measurement_id", sort=False)
    }


def _whole_area_features(axes: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([_feature_names(code, axes[code]) for code in SPECTRAL_RANGES])


def load_whole_area_store(data_source: DataSource | None = None) -> dict[str, np.ndarray]:
    source = _source(data_source)
    key = _source_key(source)
    cached = _WHOLE_AREA_CACHE.get(key)
    if cached is not None:
        return cached

    if source.source_name != "preprocessed":
        left, right = load_area_summaries(source, "1300"), load_area_summaries(source, "2850")
        result = _concatenate_area_summaries(left, right)
        _WHOLE_AREA_CACHE[key] = result
        return result

    measurements = load_file_index(source).copy()
    measurements["pair_key"] = _pair_key(measurements)
    spectral_data = {code: _load_array(source, code, basis="normalized") for code in SPECTRAL_RANGES}
    arrays = {code: values[0] for code, values in spectral_data.items()}
    axes = {code: values[1] for code, values in spectral_data.items()}
    coordinates = _coordinates_by_measurement(_parsed_path(source), source)
    values: list[np.ndarray] = []
    area_ids: list[str] = []
    measurement_ids: list[str] = []
    subject_ids: list[str] = []
    for left, right in _paired_measurements(measurements):
        left_id, right_id = str(left["measurement_id"]), str(right["measurement_id"])
        left_coordinates = coordinates.get(left_id)
        right_coordinates = coordinates.get(right_id)
        if left_coordinates is None or right_coordinates is None:
            continue
        paired = left_coordinates.merge(
            right_coordinates,
            on=["x", "y"],
            suffixes=("_1300", "_2850"),
            validate="one_to_one",
        )
        if paired.empty:
            continue
        values.append(
            np.concatenate(
                (
                    np.nanmedian(arrays["1300"][paired["normalized_row_1300"].to_numpy(dtype=int)], axis=0),
                    np.nanmedian(arrays["2850"][paired["normalized_row_2850"].to_numpy(dtype=int)], axis=0),
                )
            )
        )
        area_ids.append(str(left["area_id"]))
        measurement_ids.append(f"{left_id}|{right_id}")
        subject_ids.append(str(left["subject_id"]))
    features = _whole_area_features(axes)
    result = {
        "values": np.vstack(values) if values else np.empty((0, len(features))),
        "area_ids": np.asarray(area_ids, dtype=str),
        "measurement_ids": np.asarray(measurement_ids, dtype=str),
        "subject_ids": np.asarray(subject_ids, dtype=str),
        "feature_names": features,
    }
    _WHOLE_AREA_CACHE[key] = result
    return result


def _concatenate_area_summaries(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    left_rows = {(str(subject), str(area)): index for index, (subject, area) in enumerate(zip(left["subject_ids"], left["area_ids"]))}
    right_rows = {(str(subject), str(area)): index for index, (subject, area) in enumerate(zip(right["subject_ids"], right["area_ids"]))}
    keys = sorted(set(left_rows) & set(right_rows))
    values = [np.concatenate((left["area_summary"][left_rows[key]], right["area_summary"][right_rows[key]])) for key in keys]
    features = np.concatenate((_feature_names("1300", left["feature_names"].astype(float)), _feature_names("2850", right["feature_names"].astype(float))))
    return {
        "values": np.vstack(values) if values else np.empty((0, len(features))),
        "area_ids": np.asarray([key[1] for key in keys], dtype=str),
        "measurement_ids": np.asarray([f"{left['measurement_ids'][left_rows[key]]}|{right['measurement_ids'][right_rows[key]]}" for key in keys], dtype=str),
        "subject_ids": np.asarray([key[0] for key in keys], dtype=str),
        "feature_names": features,
    }


def load_cohort(data_source: DataSource | None, cohort_kind: str, cohort_name: str) -> pd.DataFrame:
    source = _source(data_source)
    return pd.read_parquet(source.source_path / "cohort" / cohort_name / f"{cohort_kind}.parquet")
