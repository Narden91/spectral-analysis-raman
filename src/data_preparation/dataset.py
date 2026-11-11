"""Typed access layer over the canonical parsed/preprocessed stores.

``IngestedRamanDataset`` exposes immutable per-measurement raw views
(``MeasurementView``) with zero interpolation. ``AnalysisRamanDataset``
exposes the preprocessed, analysis-ready spectra and derived summaries.
Both wrap the existing Parquet/NPY stores and the caching in
``src.data_preparation.loaders`` — they add no new physical schema.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from src import resolve_cfg as _cfg
from src.data_preparation import loaders
from src.data_preparation.materialize import ProgressCallback, prepare_data_stores, store_dir
from src.data_preparation.store import DataSource, resolve_data_source
from src.data_preparation.validation import validate_store

SpectralRange = Literal["1300", "2850"]
SpectralBasis = Literal["normalized", "pre_normalization"]


def _readonly(array: np.ndarray) -> np.ndarray:
    view = np.asarray(array).view()
    view.flags.writeable = False
    return view


@dataclass(frozen=True)
class MeasurementView:
    """One raw acquisition: unmodified native axis, raw pixels, coordinates."""

    measurement_id: str
    file_id: str
    subject_id: str
    area_id: str
    spectral_range: SpectralRange
    native_wavenumbers: np.ndarray
    raw_spectra: np.ndarray
    pixel_indices: np.ndarray
    coordinates: np.ndarray


@dataclass(frozen=True)
class SpectraView:
    values: np.ndarray
    wavenumbers: np.ndarray
    basis: SpectralBasis
    spectral_range: SpectralRange
    measurement_ids: np.ndarray
    subject_ids: np.ndarray
    file_ids: np.ndarray


@dataclass(frozen=True)
class AreaSpectraView:
    values: np.ndarray
    feature_names: np.ndarray
    basis: SpectralBasis
    spectral_range: SpectralRange | Literal["both"]
    area_ids: np.ndarray
    measurement_ids: np.ndarray
    subject_ids: np.ndarray


@dataclass
class IngestedRamanDataset:
    """Read-only view over the ``parsed`` store. No interpolation, ever."""

    _source: DataSource
    _measurements: pd.DataFrame = field(init=False, repr=False)
    _raw: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _native_axes: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _pixel_groups: dict[str, pd.DataFrame] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self._source.source_name != "parsed":
            raise RuntimeError("IngestedRamanDataset requires the parsed data source")
        self._measurements = loaders.load_file_index(self._source).set_index("measurement_id", drop=False)

    def _pixels(self) -> dict[str, pd.DataFrame]:
        if self._pixel_groups is None:
            pixels = pd.read_parquet(self._source.source_path / "pixels.parquet")
            self._pixel_groups = {
                str(key): group.sort_values("pixel_index")
                for key, group in pixels.groupby("measurement_id", sort=False)
            }
        return self._pixel_groups

    def _raw_array(self, code: str) -> np.ndarray:
        if code not in self._raw:
            self._raw[code] = np.load(self._source.source_path / f"raw_spectra_{code}.npy", mmap_mode="r")
        return self._raw[code]

    def _native_axis_array(self, code: str) -> np.ndarray:
        if code not in self._native_axes:
            self._native_axes[code] = np.load(
                self._source.source_path / f"native_wavenumbers_{code}.npy", mmap_mode="r"
            )
        return self._native_axes[code]

    def measurement(self, measurement_id: str) -> MeasurementView:
        row = self._measurements.loc[str(measurement_id)]
        if isinstance(row, pd.DataFrame):
            raise RuntimeError(f"Duplicate measurement_id: {measurement_id}")
        return self._view_from_row(row)

    def _view_from_row(self, row: pd.Series) -> MeasurementView:
        code = str(row["spectral_range"])
        start, stop = int(row["raw_start"]), int(row["raw_stop"])
        measurement_id = str(row["measurement_id"])
        pixels = self._pixels()[measurement_id]
        return MeasurementView(
            measurement_id=measurement_id,
            file_id=str(row["file_id"]),
            subject_id=str(row["subject_id"]),
            area_id=str(row["area_id"]),
            spectral_range=code,
            native_wavenumbers=_readonly(
                np.asarray(self._native_axis_array(code)[int(row["native_axis_row"])], dtype=float)
            ),
            raw_spectra=self._raw_array(code)[start:stop],
            pixel_indices=_readonly(pixels["pixel_index"].to_numpy(dtype=int)),
            coordinates=_readonly(pixels[["x", "y"]].to_numpy(dtype=float)),
        )

    def iter_measurements(self, spectral_range: SpectralRange | None = None) -> Iterator[MeasurementView]:
        """Yield views in the same order as ``measurements.parquet`` rows."""
        frame = self._measurements
        if spectral_range is not None:
            frame = frame.loc[frame["spectral_range"].astype(str) == str(spectral_range)]
        for _, row in frame.iterrows():
            yield self._view_from_row(row)

    def measurements(self) -> pd.DataFrame:
        return self._measurements.reset_index(drop=True).copy()

    def subjects(self) -> pd.DataFrame:
        return loaders.load_clinical_subjects(self._source).copy()

    def areas(self) -> pd.DataFrame:
        return loaders.load_area_metadata(self._source).copy()


@dataclass
class AnalysisRamanDataset:
    """Read-only view over the ``preprocessed`` store."""

    _source: DataSource

    def spectra(self, spectral_range: SpectralRange, basis: SpectralBasis) -> SpectraView:
        code = str(spectral_range)
        payload = loaders.load_spectra(self._source, code) if basis == "normalized" else self._pre_normalization_spectra(code)
        return SpectraView(
            values=_readonly(payload["spectra"]),
            wavenumbers=_readonly(payload["wavenumbers"]),
            basis=basis,
            spectral_range=code,
            measurement_ids=payload["measurement_ids"],
            subject_ids=payload["subject_ids"],
            file_ids=payload["file_ids"],
        )

    def _pre_normalization_spectra(self, code: str) -> dict[str, np.ndarray]:
        spectra, axis = loaders._load_array(self._source, code, basis="pre_normalization")
        rows = loaders._pixel_rows(self._source, code, retained_only=True)
        measurements = loaders.load_file_index(self._source).set_index("measurement_id")
        measurement_ids = rows["measurement_id"].astype(str)
        return {
            "spectra": spectra,
            "wavenumbers": axis,
            "measurement_ids": measurement_ids.to_numpy(),
            "subject_ids": rows["subject_id"].astype(str).to_numpy(),
            "file_ids": measurement_ids.map(measurements["file_id"]).astype(str).to_numpy(),
        }

    def area_summaries(self, spectral_range: SpectralRange, basis: SpectralBasis) -> AreaSpectraView:
        code = str(spectral_range)
        payload = loaders.load_area_summaries(self._source, code)
        key = "area_summary" if basis == "normalized" else "area_summary_bc"
        return AreaSpectraView(
            values=_readonly(payload[key]),
            feature_names=payload["feature_names"],
            basis=basis,
            spectral_range=code,
            area_ids=payload["area_ids"],
            measurement_ids=payload["measurement_ids"],
            subject_ids=payload["subject_ids"],
        )

    def whole_area_summaries(self, basis: SpectralBasis = "normalized") -> AreaSpectraView:
        payload = loaders.load_whole_area_store(self._source, basis=basis)
        return AreaSpectraView(
            values=_readonly(payload["values"]),
            feature_names=payload["feature_names"],
            basis=basis,
            spectral_range="both",
            area_ids=payload["area_ids"],
            measurement_ids=payload["measurement_ids"],
            subject_ids=payload["subject_ids"],
        )

    def measurements(self) -> pd.DataFrame:
        return loaders.load_file_index(self._source).copy()

    def subjects(self) -> pd.DataFrame:
        return loaders.load_clinical_subjects(self._source).copy()

    def areas(self) -> pd.DataFrame:
        return loaders.load_area_metadata(self._source).copy()

    def qc(self) -> pd.DataFrame:
        return pd.read_parquet(self._source.source_path / "qc.parquet")

    def cohort(self, cohort_kind: str, cohort_name: str) -> pd.DataFrame:
        return loaders.load_cohort(self._source, cohort_kind, cohort_name).copy()

    @property
    def data_source(self) -> DataSource:
        return self._source


def _open(source_name: str, cfg: dict | None, *, strict: bool) -> DataSource:
    config = _cfg(cfg)
    source_path = store_dir(source_name, config)

    if not source_path.exists():
        raise FileNotFoundError(f"{source_name} store missing at {source_path}; run prepare_dataset() first")
    validation = validate_store(source_path, source_name, config, strict=strict)
    if not validation.is_valid:
        raise RuntimeError("; ".join(validation.errors))
    return DataSource(
        source_name=source_name,
        source_path=source_path,
        manifest=validation.manifest,
        manifest_hash=validation.manifest_hash,
        was_built=False,
    )


def open_ingested_dataset(cfg: dict | None = None, *, strict: bool = False) -> IngestedRamanDataset:
    """Open the parsed store read-only. Never builds, rebuilds, or repairs."""
    return IngestedRamanDataset(_open("parsed", cfg, strict=strict))


def open_analysis_dataset(cfg: dict | None = None, *, strict: bool = False) -> AnalysisRamanDataset:
    """Open the preprocessed store read-only. Never builds, rebuilds, or repairs."""
    return AnalysisRamanDataset(_open("preprocessed", cfg, strict=strict))


def prepare_dataset(
    cfg: dict | None = None,
    *,
    force: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> AnalysisRamanDataset:
    """Build/reuse both stores, strict-validate both, return the opened analysis dataset."""
    config = _cfg(cfg)
    parsed, preprocessed = prepare_data_stores(config, force=force, progress_callback=progress_callback)

    parsed_validation = validate_store(parsed.path, "parsed", config, strict=True)
    if not parsed_validation.is_valid:
        raise RuntimeError("Strict validation failed for parsed store: " + "; ".join(parsed_validation.errors))
    preprocessed_validation = validate_store(preprocessed.path, "preprocessed", config, strict=True)
    if not preprocessed_validation.is_valid:
        raise RuntimeError(
            "Strict validation failed for preprocessed store: " + "; ".join(preprocessed_validation.errors)
        )

    resolved = resolve_data_source(config, force=False)
    return AnalysisRamanDataset(resolved)
