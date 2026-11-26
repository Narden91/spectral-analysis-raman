"""Build the two canonical Raman stores.

``parsed`` owns immutable pixel-level raw spectra. ``preprocessed`` owns the
two necessary derived bases and QC provenance.  Analysis code never crawls raw
files or reruns preprocessing.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import multiprocessing
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import resolve_cfg as _cfg
from src import resolve_project_path
from src.analysis.paired_lipid._backend import _single_thread_blas
from src.constants import SPECTRAL_RANGES
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
from src.data_preparation.validation import validate_store
from src.io.file_crawler import build_file_index, disambiguate_subject_ids
from src.io.txt_reader import read_txt_spectrum
from src.metadata.clinical import _load_raw_clinical_data, load_clinical_data, merge_index_with_clinical
from src.metadata.clinical_master import clinical_source_paths
from src.preprocessing.pipeline import flatten_spectra_rows, run_preprocessing

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class MaterializedStore:
    store_type: str
    path: Path
    manifest: dict[str, Any]


@dataclass
class _ProcessedMeasurement:
    row: pd.Series
    range_code: str
    result_spectra: np.ndarray
    pre_normalization: np.ndarray
    retained_mask: np.ndarray
    qc: pd.DataFrame
    axis: np.ndarray


def _data_cfg(cfg: dict) -> dict:
    return cfg.get("data", {})


def store_dir(source_name: str, cfg: dict) -> Path:
    """Return the configured directory of the ``parsed`` or ``preprocessed`` store."""
    key = {"parsed": "parsed_dir", "preprocessed": "preprocessed_dir"}.get(source_name)
    if key is None:
        raise ValueError(f"Unsupported data source: {source_name}")
    return resolve_project_path(_data_cfg(cfg).get(key, f"data/{source_name}"))


def _raw_text_dir(cfg: dict) -> Path:
    return resolve_project_path(_data_cfg(cfg).get("raw_text_dir", "data/raw"))


def _clinical_data_path(cfg: dict) -> Path:
    return resolve_project_path(_data_cfg(cfg).get("clinical_data_path", "data/clinical_master.csv"))


def _code_version() -> str:
    """Commit plus working-tree hash; a bare commit is insufficient when dirty."""
    repo_root = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(
            ["git", "log", "-1", "--format=%H"], cwd=repo_root, capture_output=True, check=True, text=True,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "--no-ext-diff", "HEAD"], cwd=repo_root, capture_output=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Cannot record the repository version for store provenance") from exc
    if not head:
        raise RuntimeError("Repository version is empty; cannot record store provenance")
    return f"{head}+dirty:{hashlib.sha256(diff).hexdigest()[:16]}" if diff else head


def hash_file_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _measurement_ids(frame: pd.DataFrame) -> pd.Series:
    ranges = pd.to_numeric(frame["spectral_range_cm-1"], errors="raise").astype(int).astype(str)
    modes = frame["acquisition_mode"].fillna("none").astype(str).str.lower()
    measurement_numbers = pd.to_numeric(frame["measurement_number"], errors="raise").astype(int).astype(str)
    return frame["subject_id"].astype(str).str.cat(
        [frame["slide_number"].astype(str), ranges, modes, "meas" + measurement_numbers], sep="__"
    )


def _prepare_file_index(cfg: dict) -> pd.DataFrame:
    frame = disambiguate_subject_ids(build_file_index(data_dir=_raw_text_dir(cfg))).copy()
    if frame.empty:
        raise FileNotFoundError(f"No canonical Raman measurement files found in {_raw_text_dir(cfg)}")
    frame["filepath"] = frame["filepath"].map(lambda value: str(Path(value)))
    frame["acquisition_mode"] = frame.get("acquisition_mode", pd.Series(pd.NA, index=frame.index)).astype("string")
    frame = frame.sort_values(
        ["subject_id", "slide_number", "spectral_range_cm-1", "acquisition_mode", "measurement_number", "filepath"],
        kind="mergesort",
    ).reset_index(drop=True)
    frame["file_id"] = frame["filepath"].map(lambda value: hash_file_id(Path(value).as_posix().lower()))
    frame["measurement_id"] = _measurement_ids(frame)
    if frame["measurement_id"].duplicated().any():
        raise ValueError("Duplicate measurement_id values after raw ingestion")
    frame["spectral_range"] = frame["spectral_range_cm-1"].map(lambda value: str(int(float(value))))
    frame["subject_id"] = frame["subject_id"].astype(str)
    frame["area_id"] = frame["area_id"].astype(str)
    # The source tree's sample folder is the experimental sample unit; area_id
    # is an acquisition label and must not be used as a sample surrogate.
    frame["sample_id"] = frame["filepath"].map(lambda value: Path(value).parent.name)
    frame["spot_id"] = frame["measurement_number"].map(lambda value: f"spot_{int(value)}")
    frame["source_file_path"] = frame["filepath"].map(lambda value: str(Path(value).as_posix()))
    frame["source_file_name"] = frame["filename"].astype(str)
    frame["raw_file_signature"] = frame["filepath"].map(hash_file)
    return frame


def _range_key(range_code: str) -> str:
    return "fingerprint" if range_code == "1300" else "ch_stretch"


def _flatten_with_coordinates(spectra: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = flatten_spectra_rows(spectra)
    if spectra.ndim == 3:
        coords = np.column_stack((np.repeat(x, len(y)), np.tile(y, len(x))))
    else:
        coords = np.column_stack((np.asarray(x, dtype=float), np.asarray(y, dtype=float)))
    if len(coords) != len(values):
        raise ValueError("Pixel coordinate count does not match flattened spectra")
    return values, coords


def _assert_same_axis(reference: np.ndarray | None, candidate: np.ndarray, range_code: str) -> np.ndarray:
    axis = np.asarray(candidate, dtype=float)
    if axis.ndim != 1 or axis.size < 2 or not np.all(np.diff(axis) > 0):
        raise ValueError(f"{range_code} has an invalid wavenumber axis")
    if reference is not None and (reference.shape != axis.shape or not np.allclose(reference, axis, rtol=0, atol=1e-8)):
        raise ValueError(f"{range_code} raw axes differ; calibrate/regrid explicitly before materialisation")
    return axis


def _frozen_target_axis(cfg: dict, range_code: str) -> np.ndarray:
    """Return the pre-registered analysis grid for one spectral region.

    These boundaries were verified against every native acquisition *after* the
    configured 15 cm-1 edge trim.  A missing or malformed grid is a hard error:
    deriving it from the current data would make future campaigns non-comparable.
    """
    interpolation = cfg.get("preprocessing", {}).get("interpolation", {})
    region = _range_key(range_code)
    bounds = interpolation.get("frozen_grids", {}).get(region)
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError(f"Missing preprocessing.interpolation.frozen_grids.{region}")
    start, stop = map(float, bounds)
    spacing = float(interpolation.get("target_spacing", 1.0))
    if not np.isfinite(spacing) or spacing <= 0 or stop <= start:
        raise ValueError(f"Invalid frozen interpolation grid for {region}")
    axis = np.round(np.arange(start, stop + spacing * 0.5, spacing), 6)
    if axis.size < 2 or axis[-1] > stop + 1e-8:
        raise ValueError(f"Invalid frozen interpolation axis for {region}")
    return axis


def _write_npy(path: Path, values: np.ndarray) -> None:
    np.save(path, np.asarray(values, dtype=float))


def _write_parsed_payload(
    store_dir: Path,
    measurements: pd.DataFrame,
    progress_callback: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    spectra_parts: dict[str, list[np.ndarray]] = {code: [] for code in SPECTRAL_RANGES}
    pixel_parts: list[pd.DataFrame] = []
    source_axes: dict[str, list[np.ndarray]] = {code: [] for code in SPECTRAL_RANGES}
    starts: dict[str, int] = {code: 0 for code in SPECTRAL_RANGES}
    indexed = measurements.copy()
    indexed["raw_start"] = 0
    indexed["raw_stop"] = 0
    indexed["source_axis_min"] = np.nan
    indexed["source_axis_max"] = np.nan
    indexed["source_axis_channels"] = 0
    indexed["native_axis_row"] = -1

    for position, (idx, row) in enumerate(indexed.iterrows(), start=1):
        code = str(row["spectral_range"])
        axis, x, y, spectra = read_txt_spectrum(row["filepath"])
        axis = _assert_same_axis(None, axis, code)
        values, coords = _flatten_with_coordinates(spectra, x, y)
        start = starts[code]
        stop = start + len(values)
        indexed.loc[idx, ["raw_start", "raw_stop"]] = (start, stop)
        indexed.loc[idx, ["source_axis_min", "source_axis_max", "source_axis_channels"]] = (axis[0], axis[-1], len(axis))
        indexed.loc[idx, "native_axis_row"] = len(source_axes[code])
        starts[code] = stop
        spectra_parts[code].append(values)
        source_axes[code].append(axis)
        pixel_parts.append(
            pd.DataFrame(
                {
                    "measurement_id": str(row["measurement_id"]),
                    "subject_id": str(row["subject_id"]),
                    "area_id": str(row["area_id"]),
                    "spectral_range": code,
                    "pixel_index": np.arange(len(values), dtype=int),
                    "x": coords[:, 0],
                    "y": coords[:, 1],
                    "raw_row": np.arange(start, stop, dtype=int),
                }
            )
        )
        if progress_callback is not None:
            progress_callback("Raw ingestion", position, len(indexed))

    for code in SPECTRAL_RANGES:
        axes = source_axes[code]
        if not axes:
            raise ValueError(f"Raw dataset does not contain required {code} range")
        channel_counts = {axis.size for axis in axes}
        if len(channel_counts) != 1:
            raise ValueError(f"{code} native channel counts differ; ragged raw storage is not supported")
        _write_npy(store_dir / f"raw_spectra_{code}.npy", np.vstack(spectra_parts[code]))
        _write_npy(store_dir / f"native_wavenumbers_{code}.npy", np.vstack(axes))
    pixels = pd.concat(pixel_parts, ignore_index=True)
    pixels.to_parquet(store_dir / "pixels.parquet", index=False)
    return indexed, {code: int(starts[code]) for code in SPECTRAL_RANGES}


def _append_reason(values: pd.Series, reason: str, mask: np.ndarray) -> pd.Series:
    out = values.fillna("").astype(str).copy()
    active = np.asarray(mask, dtype=bool)
    existing = out.copy()
    out.loc[active & (existing == "ok")] = reason
    append = active & (existing != "ok") & ~existing.str.split("; ").map(lambda parts: reason in parts)
    out.loc[append] = existing.loc[append] + "; " + reason
    return out


def _resolve_process_workers(n_maps: int, total_pixels: int, n_jobs: int | None) -> int:
    """Resolve safe worker count for map-level preprocessing parallelism."""
    if n_maps <= 1 or n_jobs == 0 or n_jobs == 1:
        return 1
    if os.environ.get("_PAIRED_LIPID_IN_BOOTSTRAP_WORKER") == "1":
        return 1
    if multiprocessing.parent_process() is not None:
        return 1
    try:
        from joblib.parallel import get_active_backend

        backend, _ = get_active_backend()
        if getattr(backend, "nesting_level", 0) > 0:
            return 1
    except Exception:
        pass

    auto_mode = n_jobs is None or int(n_jobs) < 0
    if auto_mode and (n_maps < 4 or total_pixels < 1000):
        return 1

    cpu_cnt = os.cpu_count() or 1
    if n_jobs is None or int(n_jobs) == -1:
        raw = cpu_cnt
    elif int(n_jobs) < -1:
        raw = max(1, cpu_cnt + 1 + int(n_jobs))
    else:
        raw = int(n_jobs)
    return max(1, min(raw, n_maps))


def _close_memmap_cache(arrays: dict[str, np.ndarray]) -> None:
    """Close underlying memmap handles after slices have been copied out."""
    for arr in arrays.values():
        mmap_obj = getattr(arr, "_mmap", None)
        if mmap_obj is not None:
            try:
                mmap_obj.close()
            except Exception:
                pass
    arrays.clear()


def _run_measurement_task(
    task: tuple[int, str, str, int, int, int, np.ndarray],
    raw_arrays: dict[str, np.ndarray],
    axis_arrays: dict[str, np.ndarray],
    parsed_dir: Path,
    cfg: dict,
    target_axes: dict[str, np.ndarray],
) -> tuple[int, str, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    pos, _mid, code, raw_start, raw_stop, axis_row, coords = task
    if code not in raw_arrays:
        raw_arrays[code] = np.load(parsed_dir / f"raw_spectra_{code}.npy", mmap_mode="r")
        axis_arrays[code] = np.load(parsed_dir / f"native_wavenumbers_{code}.npy", mmap_mode="r")
    raw_slice = np.array(raw_arrays[code][raw_start:raw_stop], dtype=float, copy=True)
    native_axis = np.array(axis_arrays[code][axis_row], dtype=float, copy=True)
    result = run_preprocessing(
        raw_slice,
        native_axis,
        region=_range_key(code),
        pixel_coords=coords,
        screen_dead=True,
        suppress_cosmic=None,
        correct_bl=True,
        smooth=None,
        interpolate=True,
        target_wavenumbers=target_axes[code],
        filter_quality=None,
        normalize=True,
        norm_for="eda",
        return_pre_normalization=True,
        cfg=cfg,
    )
    pre = np.asarray(result.stage_arrays.get("pre_normalization"), dtype=float)
    if pre.shape != result.spectra.shape:
        raise RuntimeError("Pre-normalization output is missing or misaligned")
    return (
        pos,
        code,
        np.asarray(result.spectra, dtype=float),
        pre,
        np.asarray(result.retained_mask, dtype=bool),
        result.qc_flags.merge(result.qc_metrics, on="row_index", how="left", suffixes=("", "_metric")),
        np.asarray(result.wavenumbers, dtype=float),
    )


def _process_measurement_chunk(
    parsed_dir: Path,
    tasks: list[tuple[int, str, str, int, int, int, np.ndarray]],
    cfg: dict,
    target_axes: dict[str, np.ndarray],
    *,
    single_thread_blas: bool = True,
) -> list[tuple[int, str, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]]:
    """Process one chunk of measurements inside a worker using local read-only memmaps."""
    raw_arrays: dict[str, np.ndarray] = {}
    axis_arrays: dict[str, np.ndarray] = {}
    outputs: list[tuple[int, str, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]] = []
    ctx = _single_thread_blas() if single_thread_blas else contextlib.nullcontext()
    try:
        with ctx:
            for task in tasks:
                outputs.append(_run_measurement_task(task, raw_arrays, axis_arrays, parsed_dir, cfg, target_axes))
    finally:
        _close_memmap_cache(raw_arrays)
        _close_memmap_cache(axis_arrays)
    return outputs


def _process_measurements(
    parsed_dir: Path,
    measurements: pd.DataFrame,
    cfg: dict,
    progress_callback: ProgressCallback | None = None,
    *,
    n_jobs: int | None = None,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    from src.data_preparation.dataset import IngestedRamanDataset
    from src.data_preparation.store import DataSource

    ingested = IngestedRamanDataset(
        DataSource(source_name="parsed", source_path=parsed_dir, manifest=None, manifest_hash=None, was_built=False)
    )
    rows_by_id = measurements.set_index("measurement_id", drop=False)
    pixel_groups = ingested._pixels()

    tasks: list[tuple[int, str, str, int, int, int, np.ndarray]] = []
    task_rows: list[pd.Series] = []
    present_codes: set[str] = set()
    total_pixels = 0
    for pos, (_, m_row) in enumerate(ingested._measurements.iterrows()):
        mid = str(m_row["measurement_id"])
        code = str(m_row["spectral_range"])
        raw_start = int(m_row["raw_start"])
        raw_stop = int(m_row["raw_stop"])
        native_axis_row = int(m_row["native_axis_row"])
        coords = np.array(pixel_groups[mid][["x", "y"]].to_numpy(dtype=float), dtype=float, copy=True)
        tasks.append((pos, mid, code, raw_start, raw_stop, native_axis_row, coords))
        task_rows.append(rows_by_id.loc[mid])
        present_codes.add(code)
        total_pixels += max(0, raw_stop - raw_start)

    target_axes = {code: _frozen_target_axis(cfg, code) for code in sorted(present_codes)}
    n_maps = len(tasks)
    workers = _resolve_process_workers(n_maps, total_pixels, n_jobs)
    processed: list[_ProcessedMeasurement | None] = [None] * n_maps

    if workers == 1:
        raw_arrays: dict[str, np.ndarray] = {}
        axis_arrays: dict[str, np.ndarray] = {}
        try:
            for pos, task in enumerate(tasks, start=1):
                item_pos, code, spec_arr, pre_arr, rmask, qc_df, axis_arr = _run_measurement_task(
                    task, raw_arrays, axis_arrays, parsed_dir, cfg, target_axes
                )
                processed[item_pos] = _ProcessedMeasurement(
                    row=task_rows[item_pos],
                    range_code=code,
                    result_spectra=spec_arr,
                    pre_normalization=pre_arr,
                    retained_mask=rmask,
                    qc=qc_df,
                    axis=axis_arr,
                )
                if progress_callback is not None:
                    progress_callback("Preprocessing", pos, n_maps)
        finally:
            _close_memmap_cache(raw_arrays)
            _close_memmap_cache(axis_arrays)
    else:
        chunk_size = max(1, n_maps // (workers * 4))
        chunks = [tasks[idx : idx + chunk_size] for idx in range(0, n_maps, chunk_size)]
        completed = 0
        for chunk_out in Parallel(n_jobs=workers, return_as="generator_unordered")(
            delayed(_process_measurement_chunk)(
                parsed_dir, chunk, cfg, target_axes, single_thread_blas=True
            )
            for chunk in chunks
        ):
            for item_pos, code, spec_arr, pre_arr, rmask, qc_df, axis_arr in chunk_out:
                processed[item_pos] = _ProcessedMeasurement(
                    row=task_rows[item_pos],
                    range_code=code,
                    result_spectra=spec_arr,
                    pre_normalization=pre_arr,
                    retained_mask=rmask,
                    qc=qc_df,
                    axis=axis_arr,
                )
            completed += len(chunk_out)
            if progress_callback is not None:
                progress_callback("Preprocessing", completed, n_maps)

    min_area = int(cfg.get("preprocessing", {}).get("retention", {}).get("min_pixels_per_area", 1))
    min_subject = int(cfg.get("preprocessing", {}).get("retention", {}).get("min_pixels_per_subject", 1))
    subject_counts: dict[tuple[str, str], int] = {}
    for item in processed:
        assert item is not None
        n_ret = int(item.retained_mask.sum())
        if n_ret >= min_area:
            key = (str(item.row["subject_id"]), item.range_code)
            subject_counts[key] = subject_counts.get(key, 0) + n_ret

    expected_axis: dict[str, np.ndarray | None] = {code: None for code in SPECTRAL_RANGES}
    total_kept = {code: 0 for code in SPECTRAL_RANGES}
    area_flags: list[tuple[int, bool, bool]] = []
    for item in processed:
        assert item is not None
        code = item.range_code
        expected_axis[code] = _assert_same_axis(expected_axis[code], item.axis, code)
        n_retained = int(item.retained_mask.sum())
        area_ok = n_retained >= min_area
        subject_ok = subject_counts.get((str(item.row["subject_id"]), code), 0) >= min_subject
        area_flags.append((n_retained, area_ok, subject_ok))
        if area_ok and subject_ok and n_retained > 0:
            total_kept[code] += n_retained

    result_arrays: dict[str, np.ndarray] = {}
    for code in SPECTRAL_RANGES:
        axis_arr = expected_axis[code]
        width = int(axis_arr.size) if axis_arr is not None else 0
        result_arrays[f"normalized_{code}"] = np.empty((total_kept[code], width), dtype=float)
        result_arrays[f"pre_normalization_{code}"] = np.empty((total_kept[code], width), dtype=float)
        result_arrays[f"wavenumbers_{code}"] = np.asarray(axis_arr, dtype=float)

    qc_parts: list[pd.DataFrame] = []
    area_rows: list[dict[str, Any]] = []
    offsets = {code: 0 for code in SPECTRAL_RANGES}

    for idx, (n_retained, area_ok, subject_ok) in enumerate(area_flags):
        item = processed[idx]
        assert item is not None
        processed[idx] = None
        row = item.row
        code = item.range_code
        final_mask = item.retained_mask & area_ok & subject_ok
        qc = item.qc.sort_values("row_index").reset_index(drop=True)
        if len(qc) != len(final_mask):
            raise RuntimeError("QC rows are not aligned to input pixels")
        qc["qc_reason"] = _append_reason(qc["qc_reason"], "area_min_pixels", item.retained_mask & (not area_ok))
        qc["qc_reason"] = _append_reason(qc["qc_reason"], "subject_min_pixels", item.retained_mask & area_ok & (not subject_ok))
        qc["retained"] = final_mask
        qc["measurement_id"] = str(row["measurement_id"])
        qc["subject_id"] = str(row["subject_id"])
        qc["area_id"] = str(row["area_id"])
        qc["spectral_range"] = code
        qc["pixel_index"] = np.arange(len(qc), dtype=int)
        qc["normalized_row"] = -1
        qc["pre_normalization_row"] = -1
        kept = np.flatnonzero(final_mask)
        if len(kept):
            output_start = offsets[code]
            output_stop = output_start + len(kept)
            qc.loc[kept, "normalized_row"] = np.arange(output_start, output_stop, dtype=int)
            qc.loc[kept, "pre_normalization_row"] = np.arange(output_start, output_stop, dtype=int)
            result_arrays[f"normalized_{code}"][output_start:output_stop] = item.result_spectra
            result_arrays[f"pre_normalization_{code}"][output_start:output_stop] = item.pre_normalization
            offsets[code] = output_stop
        qc_parts.append(qc)
        area_rows.append(
            {
                "measurement_id": str(row["measurement_id"]), "file_id": str(row["file_id"]),
                "subject_id": str(row["subject_id"]), "area_id": str(row["area_id"]),
                "spectral_range": code, "slide_number": str(row["slide_number"]),
                "measurement_number": int(row["measurement_number"]), "acquisition_mode": row.get("acquisition_mode"),
                "n_input_pixels": len(item.retained_mask), "n_qc_retained_pixels": n_retained,
                "n_retained_pixels": int(final_mask.sum()), "valid_area": bool(final_mask.any()),
                "retention_reason": "ok" if final_mask.any() else ("area_min_pixels" if not area_ok else "subject_min_pixels"),
            }
        )

    return result_arrays, pd.concat(qc_parts, ignore_index=True), pd.DataFrame(area_rows)


def _area_entity_count(areas: pd.DataFrame) -> int:
    return int(areas[["subject_id", "area_id"]].drop_duplicates().shape[0]) if not areas.empty else 0


def _write_cohorts(store_dir: Path, measurements: pd.DataFrame, subjects: pd.DataFrame) -> None:
    merged = merge_index_with_clinical(measurements, subjects)
    write_cohort_tables(store_dir / "cohort", build_materialized_cohorts(merged))


def _generated_signatures(root: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(root)): file_signature(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def _input_signatures(cfg: dict, measurements: pd.DataFrame) -> dict[str, dict[str, Any]]:
    signatures = {str(path): file_signature(path) for path in clinical_source_paths(cfg).values()}
    for row in measurements[["filepath", "raw_file_signature"]].itertuples(index=False):
        path = Path(str(row.filepath))
        stat = path.stat()
        signatures[str(path)] = {
            "sha256": str(row.raw_file_signature),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    return signatures


def _build_manifest(
    store_type: str, store_dir: Path, source: dict[str, Any], cfg: dict, measurements: pd.DataFrame,
    subjects: pd.DataFrame, areas: pd.DataFrame, counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "manifest_version": MANIFEST_VERSION, "schema_version": SCHEMA_VERSION, "store_type": store_type,
        "clinical_sources": {key: str(path) for key, path in clinical_source_paths(cfg).items()},
        "created_at": iso_utc_now(), "source": source, "table_format": "parquet", "array_format": "npy",
        "input_signatures": _input_signatures(cfg, measurements) if store_type == "parsed" else {}, "generated_file_signatures": _generated_signatures(store_dir),
        "preprocessing_config_hash": config_hash(cfg.get("preprocessing", {})),
        "materialization_config_hash": config_hash(cfg.get("data", {})), "code_version": _code_version(),
        "counts": {"subjects": int(subjects["subject_id"].nunique()), "files": int(measurements["file_id"].nunique()),
                   "measurements": int(measurements["measurement_id"].nunique()), "areas": _area_entity_count(areas), **counts},
    }


def _atomic_replace(store_dir: Path, writer: Callable[[Path], MaterializedStore]) -> MaterializedStore:
    """Build outside ``data/`` and atomically replace a completed store.

    Temporary materialization directories live in the operating system temp
    area, never alongside user-visible data stores.  Replacement is allowed
    only on the same filesystem volume so it remains an atomic rename.
    """
    store_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"raman_{store_dir.name}_staging_"))
    if staging.resolve().drive.casefold() != store_dir.resolve().drive.casefold():
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(
            "Atomic store replacement requires the system temporary directory and data store to share a volume"
        )
    backup_parent = Path(tempfile.mkdtemp(prefix=f"raman_{store_dir.name}_backup_"))
    backup = backup_parent / store_dir.name
    previous_store_moved = False
    try:
        output = writer(staging)
        if store_dir.exists():
            store_dir.replace(backup)
            previous_store_moved = True
        staging.replace(store_dir)
        return output
    except Exception:
        if previous_store_moved and not store_dir.exists() and backup.exists():
            backup.replace(store_dir)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(backup_parent, ignore_errors=True)


def build_parsed_store(
    cfg: dict | None = None,
    *,
    force: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> MaterializedStore:
    config = _cfg(cfg)
    clinical_paths = clinical_source_paths(config)
    _load_raw_clinical_data(clinical_paths["clinical_data_path"])
    store_path = store_dir("parsed", config)
    if store_path.exists() and not force and validate_store(store_path, "parsed", config).is_valid:
        from src.data_preparation.manifest import load_manifest
        if progress_callback is not None:
            progress_callback("Raw ingestion", 1, 1)
        return MaterializedStore("parsed", store_path, load_manifest(store_path / "manifest.json"))

    def write(staging: Path) -> MaterializedStore:
        measurements = _prepare_file_index(config)
        subjects = load_clinical_data(
            clinical_csv=_clinical_data_path(config), data_index=measurements,
        )
        measurements, pixel_counts = _write_parsed_payload(staging, measurements, progress_callback)
        measurements.to_parquet(staging / "measurements.parquet", index=False)
        subjects.to_parquet(staging / "subjects.parquet", index=False)
        areas = measurements[["measurement_id", "file_id", "subject_id", "sample_id", "spot_id", "area_id", "spectral_range", "slide_number", "measurement_number", "acquisition_mode"]].copy()
        areas.to_parquet(staging / "areas.parquet", index=False)
        _write_cohorts(staging, measurements, subjects)
        manifest = _build_manifest("parsed", staging, {"type": "raw_text", "path": str(_raw_text_dir(config)), "manifest_hash": None}, config, measurements, subjects, areas, {f"pixels_{code}": pixel_counts[code] for code in SPECTRAL_RANGES})
        return MaterializedStore("parsed", store_path, write_manifest(staging / "manifest.json", manifest))

    return _atomic_replace(store_path, write)


def build_preprocessed_store(
    cfg: dict | None = None,
    *,
    force: bool = False,
    progress_callback: ProgressCallback | None = None,
    n_jobs: int | None = None,
) -> MaterializedStore:
    config = _cfg(cfg)
    parsed_dir = store_dir("parsed", config)
    parsed = build_parsed_store(config, force=False, progress_callback=progress_callback)
    store_path = store_dir("preprocessed", config)
    if store_path.exists() and not force and validate_store(store_path, "preprocessed", config).is_valid:
        from src.data_preparation.manifest import load_manifest
        if progress_callback is not None:
            progress_callback("Preprocessing", 1, 1)
        return MaterializedStore("preprocessed", store_path, load_manifest(store_path / "manifest.json"))

    def write(staging: Path) -> MaterializedStore:
        measurements = pd.read_parquet(parsed_dir / "measurements.parquet")
        subjects = pd.read_parquet(parsed_dir / "subjects.parquet")
        arrays, qc, areas = _process_measurements(
            parsed_dir, measurements, config, progress_callback, n_jobs=n_jobs
        )
        for key, values in arrays.items():
            _write_npy(staging / f"{key}.npy", values)
        measurements.to_parquet(staging / "measurements.parquet", index=False)
        subjects.to_parquet(staging / "subjects.parquet", index=False)
        areas.to_parquet(staging / "areas.parquet", index=False)
        qc.to_parquet(staging / "qc.parquet", index=False)
        _write_cohorts(staging, measurements, subjects)
        counts = {f"pixels_{code}": int((qc["spectral_range"] == code).sum()) for code in SPECTRAL_RANGES}
        counts.update({f"retained_pixels_{code}": int((qc[(qc["spectral_range"] == code) & qc["retained"]]).shape[0]) for code in SPECTRAL_RANGES})
        manifest = _build_manifest("preprocessed", staging, {"type": "parsed", "path": str(parsed_dir), "manifest_hash": parsed.manifest.get("manifest_hash")}, config, measurements, subjects, areas, counts)
        return MaterializedStore("preprocessed", store_path, write_manifest(staging / "manifest.json", manifest))

    return _atomic_replace(store_path, write)


def prepare_data_stores(
    cfg: dict | None = None,
    *,
    force: bool = False,
    progress_callback: ProgressCallback | None = None,
    n_jobs: int | None = None,
) -> tuple[MaterializedStore, MaterializedStore]:
    """Build the complete, analysis-ready store pair from raw data."""
    config = _cfg(cfg)
    parsed = build_parsed_store(config, force=force, progress_callback=progress_callback)
    preprocessed = build_preprocessed_store(
        config, force=force, progress_callback=progress_callback, n_jobs=n_jobs
    )
    return parsed, preprocessed
