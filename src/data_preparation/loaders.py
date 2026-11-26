"""Read-only, grain-explicit views over the canonical stores."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import SPECTRAL_RANGES
from src.data_preparation.manifest import load_manifest
from src.data_preparation.store import DataSource, get_active_data_source, resolve_data_source

# Per-process cache of derived reads, keyed by data-source identity so a
# rebuilt store (new manifest_hash) never serves stale entries. Cache scope is
# limited to table DataFrames, memmap handles, and small derived summaries
# (n_area x n_chan); never densified pixel-level matrices, which are the
# actual memory cost and must stay uncached.
_TABLE_CACHE: dict[tuple[str, str, str], pd.DataFrame] = {}
_ARRAY_CACHE: dict[tuple[str, str, str, str], tuple[np.ndarray, np.ndarray]] = {}
_SUMMARY_CACHE: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
_WHOLE_AREA_CACHE: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}


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


def _pixel_rows(
    source: DataSource,
    spectral_range: str,
    *,
    retained_only: bool,
    columns: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    code = str(spectral_range)
    if source.source_name == "parsed":
        read_cols = list(dict.fromkeys((*columns, "spectral_range", "raw_row"))) if columns is not None else None
        pixels = pd.read_parquet(source.source_path / "pixels.parquet", columns=read_cols)
        return pixels.loc[pixels["spectral_range"].astype(str) == code].sort_values("raw_row")
    if source.source_name != "preprocessed":
        raise RuntimeError("Pixel rows require a materialized source")
    if columns is not None:
        required = ("spectral_range", "retained", "normalized_row") if retained_only else ("spectral_range",)
        read_cols = list(dict.fromkeys((*columns, *required)))
        qc = pd.read_parquet(source.source_path / "qc.parquet", columns=read_cols)
    else:
        qc = pd.read_parquet(source.source_path / "qc.parquet")
    rows = qc.loc[qc["spectral_range"].astype(str) == code]
    if retained_only:
        rows = rows.loc[rows["retained"].astype(bool)].sort_values("normalized_row")
    return rows


def _load_array(source: DataSource, spectral_range: str, basis: str) -> tuple[np.ndarray, np.ndarray]:
    code = str(spectral_range)
    if source.source_name == "parsed":
        raise RuntimeError(
            "Parsed spectra retain measurement-specific native axes. "
            "Use open_ingested_dataset() or the canonical preprocessed store."
        )
    key = (*_source_key(source), code, basis)
    cached = _ARRAY_CACHE.get(key)
    if cached is None:
        spectra = np.load(source.source_path / f"{basis}_{code}.npy", mmap_mode="r")
        axis = np.load(source.source_path / f"wavenumbers_{code}.npy", mmap_mode="r")
        cached = (spectra, axis)
        _ARRAY_CACHE[key] = cached
    return cached


def _feature_names(spectral_range: str, axis: np.ndarray) -> np.ndarray:
    prefix = f"{spectral_range}__" if spectral_range else ""
    return np.asarray([f"{prefix}{float(value):.1f}" for value in axis], dtype=str)


def load_spectra(data_source: DataSource | None, spectral_range: str) -> dict[str, np.ndarray]:
    source = _source(data_source)
    code = str(spectral_range)
    cols = (
        ("measurement_id", "subject_id", "spectral_range", "retained", "normalized_row")
        if source.source_name == "preprocessed"
        else None
    )
    rows = _pixel_rows(source, code, retained_only=source.source_name == "preprocessed", columns=cols)
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


def _median_block_into(
    raw_block: np.ndarray,
    out_row: np.ndarray,
    t_flat: np.ndarray | None = None,
) -> None:
    k, n_cols = raw_block.shape
    if k == 1:
        out_row[:] = raw_block[0]
        return
    if t_flat is not None and k * n_cols <= t_flat.size:
        t_buf = t_flat[: n_cols * k].reshape(n_cols, k)
        np.copyto(t_buf, raw_block.T)
        np.median(t_buf, axis=1, out=out_row, overwrite_input=True)
        if np.isnan(out_row).any():
            nan_cols = np.isnan(out_row)
            block_nan = (
                raw_block[:, nan_cols]
                if raw_block.dtype == np.float64
                else np.asarray(raw_block[:, nan_cols], dtype=float)
            )
            out_row[nan_cols] = np.nanmedian(block_nan, axis=0)
        return
    block = raw_block if raw_block.dtype == np.float64 else np.asarray(raw_block, dtype=float)
    if not np.isnan(block).any():
        np.median(block, axis=0, out=out_row)
        return
    nan_cols = np.isnan(block).any(axis=0)
    if nan_cols.all():
        out_row[:] = np.nanmedian(block, axis=0)
        return
    finite_cols = ~nan_cols
    out_row[finite_cols] = np.median(block[:, finite_cols], axis=0)
    out_row[nan_cols] = np.nanmedian(block[:, nan_cols], axis=0)


def _summarize_rows(
    rows: pd.DataFrame,
    spectra: np.ndarray,
    row_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    n_cols = int(spectra.shape[1])
    if rows.empty:
        return np.asarray([], dtype=str), np.empty((0, n_cols), dtype=float)

    row_series = rows[row_column]
    row_raw = row_series.to_numpy(copy=False)
    if row_raw.dtype.kind in "iu":
        valid_mask = row_raw >= 0
        valid_rows = row_raw[valid_mask].astype(np.intp, copy=False)
    else:
        numeric_rows = pd.to_numeric(row_series, errors="coerce")
        valid_mask = (numeric_rows >= 0).to_numpy(dtype=bool, na_value=False)
        valid_rows = numeric_rows.to_numpy()[valid_mask].astype(np.intp, copy=False)

    if not np.any(valid_mask):
        return np.asarray([], dtype=str), np.empty((0, n_cols), dtype=float)

    mids_valid = rows["measurement_id"].to_numpy()[valid_mask]
    codes, uniques = pd.factorize(mids_valid, sort=False)
    if np.any(codes < 0):
        keep = codes >= 0
        if not np.any(keep):
            return np.asarray([], dtype=str), np.empty((0, n_cols), dtype=float)
        codes = codes[keep]
        valid_rows = valid_rows[keep]

    n_groups = len(uniques)
    measurement_ids = np.asarray([str(u) for u in uniques], dtype=str)
    out = np.empty((n_groups, n_cols), dtype=float)
    if n_cols == 0:
        return measurement_ids, out

    if len(codes) > 1 and np.any(codes[1:] < codes[:-1]):
        order = np.argsort(codes, kind="mergesort")
        codes = codes[order]
        valid_rows = valid_rows[order]

    counts = np.bincount(codes, minlength=n_groups)
    ends = np.cumsum(counts)
    starts = ends - counts

    max_noncontig_k = 0
    max_t_k = 0
    group_slices: list[tuple[int, np.ndarray, bool]] = []
    for g in range(n_groups):
        g_rows = valid_rows[starts[g] : ends[g]]
        k = len(g_rows)
        if k == 1:
            group_slices.append((g, g_rows, True))
            continue
        is_contig = (
            int(g_rows[-1]) - int(g_rows[0]) == k - 1
            and bool(np.all(g_rows[1:] == g_rows[:-1] + 1))
        )
        if k <= 16384 and k > max_t_k:
            max_t_k = k
        if not is_contig and k <= 16384 and k > max_noncontig_k:
            max_noncontig_k = k
        group_slices.append((g, g_rows, is_contig))

    buf = np.empty((max_noncontig_k, n_cols), dtype=spectra.dtype) if max_noncontig_k > 0 else None
    t_flat = np.empty(max_t_k * n_cols, dtype=float) if max_t_k > 0 else None
    for g, g_rows, is_contig in group_slices:
        k = len(g_rows)
        if k == 1:
            out[g] = spectra[int(g_rows[0])]
            continue
        if is_contig:
            raw_block = spectra[int(g_rows[0]) : int(g_rows[0]) + k]
        else:
            g_rows_sorted = np.sort(g_rows)
            if buf is not None and k <= max_noncontig_k:
                np.take(spectra, g_rows_sorted, axis=0, out=buf[:k])
                raw_block = buf[:k]
            else:
                raw_block = np.take(spectra, g_rows_sorted, axis=0)
        _median_block_into(raw_block, out[g], t_flat)

    return measurement_ids, out


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

    cols = (
        ("measurement_id", "spectral_range", "retained", "normalized_row", "pre_normalization_row")
        if source.source_name == "preprocessed"
        else None
    )
    rows = _pixel_rows(source, code, retained_only=source.source_name == "preprocessed", columns=cols)
    measurements = load_file_index(source).set_index("measurement_id")
    spectra, axis = _load_array(source, code, basis="normalized")
    payload = _summary_payload(rows, spectra, axis, "normalized_row", measurements)
    if source.source_name == "preprocessed":
        baseline_corrected, _ = _load_array(source, code, basis="pre_normalization")
        payload["area_summary_bc"] = _summarize_rows(
            rows, baseline_corrected, "pre_normalization_row"
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
        columns=["measurement_id", "pixel_index", "normalized_row", "pre_normalization_row", "retained"],
    )
    retained = qc.loc[qc["retained"].astype(bool), ["measurement_id", "pixel_index", "normalized_row", "pre_normalization_row"]]
    resolved = retained.merge(pixels, on=["measurement_id", "pixel_index"], validate="one_to_one")
    return {
        str(measurement_id): group[["x", "y", "normalized_row", "pre_normalization_row"]]
        for measurement_id, group in resolved.groupby("measurement_id", sort=False)
    }


def _whole_area_features(axes: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([_feature_names(code, axes[code]) for code in SPECTRAL_RANGES])


def load_whole_area_store(
    data_source: DataSource | None = None, *, basis: str = "normalized",
) -> dict[str, np.ndarray]:
    source = _source(data_source)
    if basis not in {"normalized", "pre_normalization"}:
        raise ValueError(f"Unsupported spectral basis: {basis}")
    key = (*_source_key(source), basis)
    cached = _WHOLE_AREA_CACHE.get(key)
    if cached is not None:
        return cached

    measurements = load_file_index(source).copy()
    measurements["pair_key"] = _pair_key(measurements)
    spectral_data = {code: _load_array(source, code, basis=basis) for code in SPECTRAL_RANGES}
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
                    np.nanmedian(arrays["1300"][paired[f"{basis}_row_1300"].to_numpy(dtype=int)], axis=0),
                    np.nanmedian(arrays["2850"][paired[f"{basis}_row_2850"].to_numpy(dtype=int)], axis=0),
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


def load_cohort(data_source: DataSource | None, cohort_kind: str, cohort_name: str) -> pd.DataFrame:
    source = _source(data_source)
    return pd.read_parquet(source.source_path / "cohort" / cohort_name / f"{cohort_kind}.parquet")
