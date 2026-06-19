"""
Data loading utilities for batch-reading and aggregating Raman spectra.

Provides functions to load all spectra for a given group, aggregate
repeated measurements to subject-level mean spectra, and build
analysis-ready matrices.

Uses thread-parallel I/O for faster loading from disk.
"""

import logging
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


from src.io.file_crawler import disambiguate_subject_ids
from src.io.txt_reader import read_txt_spectrum
from src.preprocessing.cache import load_subject_mean
from src.preprocessing.pipeline import resolve_region_key
from src import config as _cfg, _output_dir

logger = logging.getLogger(__name__)

# Conservative thread count: read from config lazily on first call.
_MAX_IO_WORKERS: int | None = None


def _get_io_workers() -> int:
    global _MAX_IO_WORKERS
    if _MAX_IO_WORKERS is None:
        _MAX_IO_WORKERS = int(_cfg.get("analysis", {}).get("io_workers", 8))
    return _MAX_IO_WORKERS


def _normalized_index(df_index: pd.DataFrame) -> pd.DataFrame:
    return disambiguate_subject_ids(df_index)


def wave_axes_compatible(reference: np.ndarray, candidate: np.ndarray) -> bool:
    ref = np.asarray(reference, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    if ref.shape != cand.shape:
        return False
    if ref.size < 2:
        return np.allclose(ref, cand)

    ref_spacing = float(np.nanmedian(np.abs(np.diff(ref))))
    cand_spacing = float(np.nanmedian(np.abs(np.diff(cand))))
    spacing = min(ref_spacing, cand_spacing)
    if not np.isfinite(spacing) or spacing <= 0:
        return np.allclose(ref, cand)

    max_abs_diff = float(np.nanmax(np.abs(ref - cand)))
    return max_abs_diff <= 0.5 * spacing


def _read_one_file(fpath: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Read a single measurement file, returning (wavenumbers, flat_spectra) or None."""
    try:
        wave, _, _, spec, _ = read_txt_spectrum(fpath)
        # Ensure the spectrum array is a 2D array (n_records, n_wavenumbers).
        # Some readers may return a 1D vector for single-measurement files.
        spec = np.asarray(spec)
        if spec.ndim == 1:
            flat = spec.reshape(1, -1)
        elif spec.ndim == 2:
            flat = spec.reshape(-1, spec.shape[-1])
        else:
            # Collapse any extra leading dimensions while keeping the last axis
            flat = spec.reshape(-1, spec.shape[-1])
        return wave, flat
    except Exception as e:
        logger.warning("Error reading %s: %s", fpath, e)
        return None


def _average_area_vectors(
    wavenumbers: np.ndarray,
    area_vectors: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the whole-spectrum subject mean from paired-area vectors.

    EDA whole-spectrum mode currently treats each surviving paired area as one
    unit and computes the subject mean as a direct n_valid_pixels-weighted
    average across all paired areas for that subject.
    """
    if area_vectors.size == 0:
        return wavenumbers, np.array([])
    if weights.size == 0 or np.nansum(weights) <= 0:
        return wavenumbers, np.nanmean(area_vectors, axis=0)
    return wavenumbers, np.average(area_vectors, axis=0, weights=weights)


def _select_subject_ids(
    df_index: pd.DataFrame,
    gender: Optional[str] = None,
    is_control: Optional[bool] = None,
    spectral_range: str = "1300",
    max_subjects: Optional[int] = None,
) -> list[str]:
    if spectral_range == "both":
        mask = df_index["spectral_range_cm-1"].isin([1300.0, 2850.0])
    else:
        mask = df_index["spectral_range_cm-1"] == float(spectral_range)
    if gender is not None:
        mask &= df_index["gender"] == gender
    if is_control is not None:
        mask &= df_index["is_control"] == is_control

    subjects = sorted(df_index.loc[mask, "subject_id"].astype(str).unique().tolist())
    if max_subjects is not None:
        subjects = subjects[:max_subjects]
    return subjects


def load_subject_spectra(
    df_index: pd.DataFrame,
    subject_id: str,
    spectral_range: str = "1300",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and flatten all spectra for a single subject in a given spectral range.

    Returns
    -------
    wavenumbers : np.ndarray
        1D wavenumber axis.
    spectra : np.ndarray
        2D array (n_total_pixels, n_wavenumbers) — all pixels from all
        slides / measurements concatenated.
    """
    return _load_subject_spectra_from_index(_normalized_index(df_index), subject_id, spectral_range)


def _load_subject_spectra_from_index(
    df_index: pd.DataFrame,
    subject_id: str,
    spectral_range: str,
) -> tuple[np.ndarray, np.ndarray]:
    range_val = float(spectral_range)
    mask = (df_index["subject_id"] == subject_id) & (
        df_index["spectral_range_cm-1"] == range_val
    )
    rows = df_index[mask]

    fpaths = [Path(p) for p in rows["filepath"] if Path(p).exists()]

    if not fpaths:
        return np.array([]), np.array([])

    # Single file → skip thread overhead
    if len(fpaths) == 1:
        result = _read_one_file(fpaths[0])
        if result is None:
            return np.array([]), np.array([])
        return result

    # Multiple files → parallel I/O
    all_spectra: list[np.ndarray] = []
    wavenumbers = None

    with ThreadPoolExecutor(max_workers=min(_get_io_workers(), len(fpaths))) as pool:
        for result in pool.map(_read_one_file, fpaths):
            if result is not None:
                wave, flat = result
                if wavenumbers is None:
                    wavenumbers = wave
                elif not wave_axes_compatible(wavenumbers, wave):
                    raise ValueError(f"Wavenumber axis mismatch for subject {subject_id}")
                all_spectra.append(flat)

    if not all_spectra:
        return np.array([]), np.array([])

    return wavenumbers, np.vstack(all_spectra)




def _compute_subject_mean_spectrum_from_index(
    df_index: pd.DataFrame,
    subject_id: str,
    spectral_range: str,
) -> tuple[np.ndarray, np.ndarray]:
    if spectral_range == "both":
        wavenumbers, area_vectors, weights = _load_subject_paired_area_vectors_from_index(df_index, subject_id)
        return _average_area_vectors(wavenumbers, area_vectors, weights)

    region_key = resolve_region_key(spectral_range)
    range_val = float(spectral_range)
    mask = (df_index["subject_id"] == subject_id) & (
        df_index["spectral_range_cm-1"] == range_val
    )
    subject_rows = df_index[mask]
    
    if subject_rows.empty:
        return np.array([]), np.array([])
        
    from src.preprocessing.cache import build_area_cache
    from src.preprocessing.pipeline import retention_guard_config
    
    all_spectra = []
    wavenumbers = None
    
    for _, row in subject_rows.iterrows():
        cached = build_area_cache(
            filepath=row["filepath"],
            subject_id=str(subject_id),
            slide_number=str(row["slide_number"]),
            area_id=str(row["area_id"]),
            region=region_key,
        )
        if cached.spectra.size > 0:
            if wavenumbers is None:
                wavenumbers = cached.wavenumbers
            elif not wave_axes_compatible(wavenumbers, cached.wavenumbers):
                raise ValueError(f"Wavenumber axis mismatch in area {row['area_id']} for subject {subject_id}")
            all_spectra.append(cached.spectra)
            
    if not all_spectra or wavenumbers is None:
        return np.array([]), np.array([])
        
    pixels = np.vstack(all_spectra)
    guards = retention_guard_config()
    if pixels.shape[0] < guards["min_pixels_per_subject"]:
        return wavenumbers, np.array([])
        
    return wavenumbers, np.nanmean(pixels, axis=0)


def _concatenate_region_wavenumbers(area_result: dict[str, np.ndarray]) -> np.ndarray:
    wn_1300 = np.asarray(area_result["wn_1300"], dtype=float)
    wn_2850 = np.asarray(area_result["wn_2850"], dtype=float)
    if float(np.nanmax(wn_1300)) >= float(np.nanmin(wn_2850)):
        raise ValueError(
            "Whole-spectrum wavenumber axes must satisfy wn_1300.max() < wn_2850.min(); "
            f"got {float(np.nanmax(wn_1300)):.1f} and {float(np.nanmin(wn_2850)):.1f}"
        )

    range_pairs = [
        ("wn_1300", wn_1300),
        ("wn_2850", wn_2850),
    ]
    ordered_pairs = sorted(range_pairs, key=lambda pair: float(np.nanmin(pair[1])))

    for index, (label, axis) in enumerate(ordered_pairs):
        if axis.ndim != 1 or axis.size == 0:
            raise ValueError(f"Whole-spectrum axis '{label}' must be a non-empty 1D array")
        if index == 0:
            continue
        previous_label, previous_axis = ordered_pairs[index - 1]
        if float(np.nanmax(previous_axis)) >= float(np.nanmin(axis)):
            raise ValueError(
                "Whole-spectrum wavenumber axes must be strictly ordered and non-overlapping; "
                f"got {previous_label} max={float(np.nanmax(previous_axis)):.1f} and "
                f"{label} min={float(np.nanmin(axis)):.1f}"
            )

    return np.concatenate([axis for _, axis in ordered_pairs])


def _load_subject_paired_area_vectors_from_index(
    df_index: pd.DataFrame,
    subject_id: str,
    *,
    persist_artifacts: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subject_rows = df_index[df_index["subject_id"] == subject_id]
    if subject_rows.empty:
        return np.array([]), np.empty((0, 0)), np.array([])

    from src.fusion.area_pipeline import process_subject_areas

    area_results = process_subject_areas(subject_rows, persist_artifacts=persist_artifacts)
    if not area_results:
        return np.array([]), np.empty((0, 0)), np.array([])

    wavenumbers = _concatenate_region_wavenumbers(area_results[0])
    area_vectors = np.vstack([np.asarray(result["area_mean"]) for result in area_results])
    weights = np.asarray(
        [max(int(result["n_valid_pixels"]), 1) for result in area_results],
        dtype=float,
    )
    return wavenumbers, area_vectors, weights


def _whole_spectra_cache_dir() -> Path:
    cache_dir = _output_dir("processed_subdir") / "whole_spectra"
    return cache_dir


def _whole_spectra_persistence_defaults() -> tuple[bool, bool]:
    output_cfg = _cfg.get("output", {})
    return (
        bool(output_cfg.get("keep_whole_spectra_cache", True)),
        bool(output_cfg.get("keep_preprocessing_artifacts", True)),
    )


def _whole_spectra_cache_key(
    df_index: pd.DataFrame,
    subject_ids: list[str],
) -> str:
    relevant_cols = [
        "subject_id",
        "slide_number",
        "area_id",
        "spectral_range_cm-1",
        "measurement_number",
        "filepath",
    ]

    available_cols = [col for col in relevant_cols if col in df_index.columns]
    relevant = df_index[df_index["subject_id"].isin(subject_ids)].copy()
    if "spectral_range_cm-1" in relevant.columns:
        relevant = relevant[relevant["spectral_range_cm-1"].isin([1300.0, 2850.0])]
    relevant = relevant[available_cols].sort_values(available_cols, kind="stable")

    row_hash = hashlib.sha256(
        pd.util.hash_pandas_object(relevant.astype(str), index=False).to_numpy().tobytes()
    ).hexdigest()

    file_signatures = []
    for filepath in relevant["filepath"].dropna().astype(str).unique().tolist():
        path = Path(filepath)
        try:
            stat = path.stat()
            file_signatures.append((str(path.resolve()), stat.st_mtime_ns, stat.st_size))
        except OSError:
            file_signatures.append((str(path), None, None))

    payload = {
        "subject_ids": subject_ids,
        "row_hash": row_hash,
        "file_signatures": file_signatures,
        "preprocessing": _cfg.get("preprocessing", {}),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def _build_whole_spectra_area_store(
    df_index: pd.DataFrame,
    subject_ids: list[str],
    *,
    persist_artifacts: bool | None = None,
):
    from src.fusion.area_pipeline import process_subject_areas
    from src.fusion.area_summary import AreaStore

    store = AreaStore()
    if not subject_ids:
        return store

    selected_rows = df_index[df_index["subject_id"].isin(subject_ids)]
    subject_groups = {
        str(subject_id): group
        for subject_id, group in selected_rows.groupby("subject_id", sort=False)
    }

    for subject_id in subject_ids:
        subject_rows = subject_groups.get(subject_id)
        if subject_rows is None or subject_rows.empty:
            continue

        area_results = process_subject_areas(subject_rows, persist_artifacts=persist_artifacts)
        if not area_results:
            continue

        subject_wave = _concatenate_region_wavenumbers(area_results[0])
        if store.wavenumbers is None:
            store.set_wavenumbers(subject_wave)
        elif not wave_axes_compatible(store.wavenumbers, subject_wave):
            raise ValueError(
                "Wavenumber axis mismatch while building whole-spectrum area store "
                f"for subject '{subject_id}'"
            )

        for result in area_results:
            store.add(
                subject_id=result["subject_id"],
                slide_number=result["slide_number"],
                area_id=result["area_id"],
                area_mean=result["area_mean"],
                area_std=result["area_std"],
                n_valid_pixels=result["n_valid_pixels"],
                area_mean_bc=result.get("area_mean_bc"),
            )


    return store


def _load_or_build_whole_spectra_area_store(
    df_index: pd.DataFrame,
    subject_ids: list[str],
    *,
    persist_cache: bool | None = None,
    persist_artifacts: bool | None = None,
):
    from src.fusion.area_summary import AreaStore

    if not subject_ids:
        return AreaStore()

    default_persist_cache, default_persist_artifacts = _whole_spectra_persistence_defaults()
    should_persist_cache = default_persist_cache if persist_cache is None else bool(persist_cache)
    should_persist_artifacts = (
        default_persist_artifacts if persist_artifacts is None else bool(persist_artifacts)
    )

    cache_path = _whole_spectra_cache_dir() / (
        f"whole_spectra_{_whole_spectra_cache_key(df_index, subject_ids)}.npz"
    )
    if cache_path.exists():
        cached_store = AreaStore.load_npz(cache_path)
        if cached_store.wavenumbers is not None:
            logger.info("Loading cached whole-spectrum area store: %s", cache_path)
            return cached_store
        logger.warning("Ignoring incomplete whole-spectrum cache (missing axis): %s", cache_path)

    store = _build_whole_spectra_area_store(
        df_index,
        subject_ids,
        persist_artifacts=should_persist_artifacts,
    )
    if should_persist_cache and len(store) > 0 and store.wavenumbers is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        store.save_npz(cache_path)
        logger.info("Saved cached whole-spectrum area store: %s", cache_path)
    return store


def _whole_spectra_subject_matrix_from_store(
    store,
    subject_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    area_df = store.to_dataframe()
    if area_df.empty or store.wavenumbers is None:
        return np.array([]), np.empty((0, 0)), []

    subject_groups = {
        str(subject_id): group
        for subject_id, group in area_df.groupby("subject_id", sort=False)
    }
    mean_spectra: list[np.ndarray] = []
    valid_subjects: list[str] = []

    for subject_id in subject_ids:
        subject_rows = subject_groups.get(subject_id)
        if subject_rows is None or subject_rows.empty:
            logger.warning("Skipping subject %s (no valid paired areas)", subject_id)
            continue

        area_vectors = np.vstack(
            [np.asarray(vector, dtype=float) for vector in subject_rows["area_mean"].to_numpy()]
        )
        weights = subject_rows["n_valid_pixels"].to_numpy(dtype=float)
        _, mean_spectrum = _average_area_vectors(store.wavenumbers, area_vectors, weights)
        mean_spectra.append(mean_spectrum)
        valid_subjects.append(subject_id)

    if not mean_spectra:
        return np.array([]), np.empty((0, 0)), []

    return np.asarray(store.wavenumbers, dtype=float), np.vstack(mean_spectra), valid_subjects


def _build_group_matrix_for_both_range(
    df_index: pd.DataFrame,
    subject_ids: list[str],
    store=None,
    *,
    persist_cache: bool | None = None,
    persist_artifacts: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build whole-spectrum subject means for a group in one batch pass.

    This preserves the existing EDA aggregation contract for `range="both"`:
    paired areas are processed per subject, then averaged directly with
    n_valid_pixels weights across all surviving areas for that subject.
    """
    if not subject_ids:
        return np.array([]), np.empty((0, 0)), []

    if store is None:
        store = _load_or_build_whole_spectra_area_store(
            df_index,
            subject_ids,
            persist_cache=persist_cache,
            persist_artifacts=persist_artifacts,
        )
    return _whole_spectra_subject_matrix_from_store(store, subject_ids)


def build_whole_spectra_group_matrices(
    df_index: pd.DataFrame,
    group_filters: dict[str, dict[str, object]],
    max_subjects: Optional[int] = None,
    *,
    persist_cache: bool | None = None,
    persist_artifacts: bool | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, list[str]]]:
    """Build all whole-spectrum EDA groups from one shared area store.

    `range="both"` is the only mode where EDA needs paired-area fusion before
    subject aggregation. When multiple groups are loaded for one run, we can
    build the paired-area store once for the union of selected subjects and then
    slice subject means per group from that shared store.
    """
    df_index = _normalized_index(df_index)
    group_subjects: dict[str, list[str]] = {}
    ordered_subjects: list[str] = []
    seen_subjects: set[str] = set()

    for group_key, filters in group_filters.items():
        subjects = _select_subject_ids(
            df_index,
            gender=filters.get("gender"),
            is_control=filters.get("is_control"),
            spectral_range="both",
            max_subjects=max_subjects,
        )
        group_subjects[group_key] = subjects
        for subject_id in subjects:
            if subject_id in seen_subjects:
                continue
            seen_subjects.add(subject_id)
            ordered_subjects.append(subject_id)

    if not ordered_subjects:
        return np.array([]), {}, {group_key: [] for group_key in group_filters}

    logger.info(
        "Building shared whole-spectrum area store for %d subjects across %d groups",
        len(ordered_subjects),
        len(group_filters),
    )
    store = _load_or_build_whole_spectra_area_store(
        df_index,
        ordered_subjects,
        persist_cache=persist_cache,
        persist_artifacts=persist_artifacts,
    )

    wavenumbers: np.ndarray | None = None
    group_matrices: dict[str, np.ndarray] = {}
    group_ids = {group_key: [] for group_key in group_filters}

    for group_key, subjects in group_subjects.items():
        wave, matrix, valid_subjects = _build_group_matrix_for_both_range(
            df_index,
            subjects,
            store=store,
            persist_cache=persist_cache,
            persist_artifacts=persist_artifacts,
        )
        group_ids[group_key] = valid_subjects
        if matrix.size == 0:
            continue
        group_matrices[group_key] = matrix
        if wavenumbers is None:
            wavenumbers = wave
        elif not wave_axes_compatible(wavenumbers, wave):
            raise ValueError(
                "Wavenumber axis mismatch while slicing shared whole-spectrum group matrices"
            )

    if wavenumbers is None:
        return np.array([]), group_matrices, group_ids
    return wavenumbers, group_matrices, group_ids


def build_group_matrix(
    df_index: pd.DataFrame,
    gender: Optional[str] = None,
    is_control: Optional[bool] = None,
    spectral_range: str = "1300",
    max_subjects: Optional[int] = None,
    *,
    persist_cache: bool | None = None,
    persist_artifacts: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build a subject-level mean spectra matrix for a group.

    Uses thread-parallel loading for faster I/O.

    Parameters
    ----------
    df_index : pd.DataFrame
        The dataset index.
    gender : str or None
        Filter by gender ('male', 'female', or None for all).
    is_control : bool or None
        Filter by control status.
    spectral_range : str
        '1300', '2850', or 'both'. Single-range EDA flattens all pixels per
        subject before averaging; whole-spectrum EDA builds paired-area vectors
        first, then computes a direct `n_valid_pixels`-weighted subject mean.
    max_subjects : int or None
        Limit number of subjects (for quick testing).

    Returns
    -------
    wavenumbers : np.ndarray
        1D wavenumber axis.
    spectra_matrix : np.ndarray
        2D array (n_subjects, n_wavenumbers).
    subject_ids : list[str]
        Ordered list of subject IDs.
    """
    df_index = _normalized_index(df_index)
    subjects = _select_subject_ids(
        df_index,
        gender=gender,
        is_control=is_control,
        spectral_range=spectral_range,
        max_subjects=max_subjects,
    )

    logger.debug(
        "Building group matrix: gender=%s, control=%s, range=%s, n_subjects=%d",
        gender,
        is_control,
        spectral_range,
        len(subjects),
    )

    if spectral_range == "both":
        return _build_group_matrix_for_both_range(
            df_index,
            subjects,
            persist_cache=persist_cache,
            persist_artifacts=persist_artifacts,
        )

    def _load_one(sid: str) -> tuple[str, np.ndarray, np.ndarray]:
        w, ms = _compute_subject_mean_spectrum_from_index(df_index, sid, spectral_range)
        return sid, w, ms

    mean_spectra: list[np.ndarray] = []
    valid_subjects: list[str] = []
    wavenumbers = None

    # Parallel subject loading (order preserved by pool.map)
    with ThreadPoolExecutor(max_workers=_get_io_workers()) as pool:
        for sid, wave, mean_spec in pool.map(_load_one, subjects):
            if mean_spec.size == 0:
                logger.warning("Skipping subject %s (no valid spectra)", sid)
                continue
            if wavenumbers is not None and not wave_axes_compatible(wavenumbers, wave):
                raise ValueError(
                    f"Wavenumber axis mismatch while building group matrix for subject '{sid}'"
                )
            mean_spectra.append(mean_spec)
            valid_subjects.append(sid)
            if wavenumbers is None:
                wavenumbers = wave

    if not mean_spectra:
        return np.array([]), np.empty((0, 0)), []

    return wavenumbers, np.vstack(mean_spectra), valid_subjects
