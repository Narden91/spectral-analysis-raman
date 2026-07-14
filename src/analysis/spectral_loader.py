"""Subject-level analysis views of the materialized Raman store."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_preparation.loaders import load_area_summaries, load_spectra, load_whole_area_store
from src.io.file_crawler import disambiguate_subject_ids


def _normalized_index(frame: pd.DataFrame) -> pd.DataFrame:
    return disambiguate_subject_ids(frame)


def _select_subject_ids(
    frame: pd.DataFrame,
    gender: str | None = None,
    is_control: bool | None = None,
    spectral_range: str = "1300",
    max_subjects: int | None = None,
) -> list[str]:
    values = frame.copy()
    if spectral_range == "both":
        mask = values["spectral_range_cm-1"].isin((1300.0, 2850.0))
    else:
        mask = values["spectral_range_cm-1"] == float(spectral_range)
    if gender is not None:
        mask &= values["gender"].astype(str).str.lower() == gender.lower()
    if is_control is not None:
        mask &= values["is_control"].astype(bool) == is_control
    subjects = sorted(values.loc[mask, "subject_id"].astype(str).unique())
    return subjects[:max_subjects] if max_subjects is not None else subjects


def _axis(feature_names: np.ndarray) -> np.ndarray:
    names = np.asarray(feature_names, dtype=str)
    return np.asarray([float(name.split("__", 1)[-1]) for name in names], dtype=float)


def wave_axes_compatible(reference: np.ndarray, candidate: np.ndarray) -> bool:
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    return reference.shape == candidate.shape and np.allclose(reference, candidate, rtol=0, atol=1e-8)


def load_subject_spectra(
    df_index: pd.DataFrame,
    subject_id: str,
    spectral_range: str = "1300",
) -> tuple[np.ndarray, np.ndarray]:
    """Return retained pixel spectra for one subject from the active store."""
    payload = load_spectra(None, spectral_range)
    mask = payload["subject_ids"].astype(str) == str(subject_id)
    return np.asarray(payload["wavenumbers"], dtype=float), np.asarray(payload["spectra"][mask], dtype=float)


def _subject_matrix(payload: dict[str, np.ndarray], subjects: list[str], value_key: str) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(payload[value_key], dtype=float)
    row_subjects = payload["subject_ids"].astype(str)
    kept = [subject for subject in subjects if (row_subjects == subject).any()]
    matrix = np.vstack([np.nanmedian(values[row_subjects == subject], axis=0) for subject in kept]) if kept else np.empty((0, values.shape[1]))
    return matrix, kept


def build_group_matrix(
    df_index: pd.DataFrame,
    gender: str | None = None,
    is_control: bool | None = None,
    spectral_range: str = "1300",
    max_subjects: int | None = None,
    **_: object,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build one median spectrum per subject; areas are never independent rows."""
    frame = _normalized_index(df_index)
    subjects = _select_subject_ids(frame, gender, is_control, spectral_range, max_subjects)
    if spectral_range == "both":
        payload = load_whole_area_store()
        matrix, ids = _subject_matrix(payload, subjects, "values")
        return _axis(payload["feature_names"]), matrix, ids
    payload = load_area_summaries(None, spectral_range)
    matrix, ids = _subject_matrix(payload, subjects, "area_summary")
    return _axis(payload["feature_names"]), matrix, ids


def build_whole_spectra_group_matrices(
    df_index: pd.DataFrame,
    group_filters: dict[str, dict[str, object]],
    max_subjects: int | None = None,
    **_: object,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, list[str]]]:
    """Build all whole-spectrum groups from the same paired-pixel store."""
    frame = _normalized_index(df_index)
    payload = load_whole_area_store()
    axis = _axis(payload["feature_names"])
    matrices: dict[str, np.ndarray] = {}
    ids: dict[str, list[str]] = {}
    for name, filters in group_filters.items():
        subjects = _select_subject_ids(
            frame,
            gender=filters.get("gender") if isinstance(filters.get("gender"), str) else None,
            is_control=filters.get("is_control") if isinstance(filters.get("is_control"), bool) else None,
            spectral_range="both",
            max_subjects=max_subjects,
        )
        matrix, selected = _subject_matrix(payload, subjects, "values")
        matrices[name], ids[name] = matrix, selected
    return axis, matrices, ids
