from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src
from src.data_preparation.dataset import (
    AnalysisRamanDataset,
    open_analysis_dataset,
    open_ingested_dataset,
    prepare_dataset,
)
from src.data_preparation.materialize import build_parsed_store, build_preprocessed_store


def _write_measurement_file(path: Path, waves: list[float], base: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#X\t#Y\t#Wave\t#Intensity"]
    for x, y, offset in ((0.0, 0.0, 0.0), (1.0, 0.0, 5.0)):
        for idx, wave in enumerate(waves):
            intensity = base + offset + idx
            lines.append(f"{x:.1f}\t{y:.1f}\t{wave:.3f}\t{intensity:.3f}")
    path.write_text("\n".join(lines), encoding="utf-8")


@pytest.fixture()
def configured_store_paths(tmp_path: Path) -> dict[str, Path]:
    original = deepcopy(src.config)
    raw_dir = tmp_path / "data" / "raw"
    parsed_dir = tmp_path / "data" / "parsed"
    preprocessed_dir = tmp_path / "data" / "preprocessed"
    clinical_path = tmp_path / "data" / "clinical_data.csv"

    src.config.clear()
    src.config.update(original)
    src.config["data"].update(
        {
            "source": "preprocessed",
            "auto_build": True,
            "raw_text_dir": str(raw_dir),
            "root_dir": str(raw_dir),
            "clinical_data_path": str(clinical_path),
            "clinical_csv": str(clinical_path),
            "parsed_dir": str(parsed_dir),
            "preprocessed_dir": str(preprocessed_dir),
            "require_valid_manifest": True,
            "stale_manifest_policy": "error",
        }
    )
    src.config["preprocessing"]["retention"]["min_pixels_per_area"] = 1
    src.config["preprocessing"]["retention"]["min_pixels_per_subject"] = 1
    src.config["preprocessing"]["quality_filter"]["spectral_quality"]["snr_threshold"] = 0.0
    src.config["preprocessing"]["quality_filter"]["spectral_quality"]["min_area_percentile"] = 0
    src.config["preprocessing"]["interpolation"]["frozen_grids"] = {
        "fingerprint": [1240.0, 1249.0],
        "ch_stretch": [2840.0, 2849.0],
    }
    src.config["output"]["keep_preprocessing_artifacts"] = False
    src.config["output"]["keep_whole_spectra_cache"] = False

    waves_1300 = [1240.0 + idx for idx in range(10)]
    waves_2850 = [2840.0 + idx for idx in range(10)]
    _write_measurement_file(
        raw_dir / "Control" / "Female" / "1" / "sample_1" / "1_1_1300_range_meas1.txt", waves_1300, 10.0
    )
    _write_measurement_file(
        raw_dir / "Control" / "Female" / "1" / "sample_1" / "1_1_2850_range_meas1.txt", waves_2850, 15.0
    )
    _write_measurement_file(
        raw_dir / "Disease" / "Male" / "2" / "sample_1" / "2_1_1300_range_meas1.txt", waves_1300, 20.0
    )
    _write_measurement_file(
        raw_dir / "Disease" / "Male" / "2" / "sample_1" / "2_1_2850_range_meas1.txt", waves_2850, 25.0
    )
    clinical = pd.DataFrame(
        [
            {"subject_id": "1", "age_years": 50, "gender": "female", "diagnosis_text": "healthy control"},
            {"subject_id": "2", "age_years": 60, "gender": "male", "diagnosis_text": "ischemic stroke"},
        ]
    )
    clinical_path.parent.mkdir(parents=True, exist_ok=True)
    clinical.to_csv(clinical_path, index=False)

    try:
        yield {
            "raw_dir": raw_dir,
            "parsed_dir": parsed_dir,
            "preprocessed_dir": preprocessed_dir,
            "clinical_path": clinical_path,
        }
    finally:
        src.config.clear()
        src.config.update(original)


def _legacy_view(parsed_dir: Path, measurement_id: str) -> dict[str, np.ndarray]:
    """Reproduce materialize._process_measurements's exact current indexing."""
    measurements = pd.read_parquet(parsed_dir / "measurements.parquet")
    row = measurements.set_index("measurement_id").loc[measurement_id]
    code = str(row["spectral_range"])
    raw = np.load(parsed_dir / f"raw_spectra_{code}.npy", mmap_mode="r")
    native_axes = np.load(parsed_dir / f"native_wavenumbers_{code}.npy", mmap_mode="r")
    pixels = pd.read_parquet(parsed_dir / "pixels.parquet")
    pixel_groups = {key: group.sort_values("pixel_index") for key, group in pixels.groupby("measurement_id", sort=False)}
    start, stop = int(row["raw_start"]), int(row["raw_stop"])
    return {
        "raw_spectra": np.asarray(raw[start:stop]),
        "native_wavenumbers": np.asarray(native_axes[int(row["native_axis_row"])], dtype=float),
        "coordinates": pixel_groups[measurement_id][["x", "y"]].to_numpy(dtype=float),
        "dtype": raw.dtype,
    }


def test_measurement_view_matches_legacy_raw_slice_exactly(configured_store_paths: dict[str, Path]) -> None:
    build_parsed_store(src.config, force=True)
    dataset = open_ingested_dataset(src.config)
    measurements = dataset.measurements()

    for measurement_id in measurements["measurement_id"].astype(str):
        legacy = _legacy_view(configured_store_paths["parsed_dir"], measurement_id)
        view = dataset.measurement(measurement_id)

        assert np.array_equal(view.raw_spectra, legacy["raw_spectra"]), measurement_id
        assert np.array_equal(view.native_wavenumbers, legacy["native_wavenumbers"]), measurement_id
        assert np.array_equal(view.coordinates, legacy["coordinates"]), measurement_id


def test_measurement_view_preserves_dtype_and_sign(configured_store_paths: dict[str, Path]) -> None:
    build_parsed_store(src.config, force=True)
    dataset = open_ingested_dataset(src.config)
    measurements = dataset.measurements()
    measurement_id = str(measurements["measurement_id"].iloc[0])
    legacy = _legacy_view(configured_store_paths["parsed_dir"], measurement_id)
    view = dataset.measurement(measurement_id)

    assert view.raw_spectra.dtype == legacy["dtype"]


def test_measurement_view_native_axis_is_measurement_specific(configured_store_paths: dict[str, Path]) -> None:
    shifted = configured_store_paths["raw_dir"] / "Disease" / "Male" / "2" / "sample_1" / "2_1_1300_range_meas1.txt"
    _write_measurement_file(shifted, [1240.2 + idx for idx in range(10)], 20.0)
    build_parsed_store(src.config, force=True)
    dataset = open_ingested_dataset(src.config)
    measurements = dataset.measurements()
    range_1300 = measurements.loc[measurements["spectral_range"].astype(str) == "1300", "measurement_id"].astype(str)
    assert len(range_1300) == 2

    axes = [dataset.measurement(mid).native_wavenumbers for mid in range_1300]
    assert not np.array_equal(axes[0], axes[1]), "the two 1300 measurements must keep distinct native axes"


def test_measurement_view_arrays_are_read_only(configured_store_paths: dict[str, Path]) -> None:
    build_parsed_store(src.config, force=True)
    dataset = open_ingested_dataset(src.config)
    measurement_id = str(dataset.measurements()["measurement_id"].iloc[0])
    view = dataset.measurement(measurement_id)

    with pytest.raises(ValueError):
        view.native_wavenumbers[0] = 0.0
    with pytest.raises(ValueError):
        view.coordinates[0, 0] = 0.0


def test_iter_measurements_matches_stored_order(configured_store_paths: dict[str, Path]) -> None:
    build_parsed_store(src.config, force=True)
    dataset = open_ingested_dataset(src.config)
    expected_order = dataset.measurements()["measurement_id"].astype(str).tolist()
    actual_order = [view.measurement_id for view in dataset.iter_measurements()]
    assert actual_order == expected_order


def test_iter_measurements_filters_by_range(configured_store_paths: dict[str, Path]) -> None:
    build_parsed_store(src.config, force=True)
    dataset = open_ingested_dataset(src.config)
    views = list(dataset.iter_measurements(spectral_range="2850"))
    assert views and all(view.spectral_range == "2850" for view in views)


def test_unknown_measurement_id_rejected(configured_store_paths: dict[str, Path]) -> None:
    build_parsed_store(src.config, force=True)
    dataset = open_ingested_dataset(src.config)
    with pytest.raises(KeyError):
        dataset.measurement("does-not-exist")


def test_duplicate_measurement_id_rejected(configured_store_paths: dict[str, Path]) -> None:
    build_parsed_store(src.config, force=True)
    dataset = open_ingested_dataset(src.config)
    duplicated_id = str(dataset.measurements()["measurement_id"].iloc[0])
    corrupted = pd.concat([dataset._measurements, dataset._measurements.loc[[duplicated_id]]])
    dataset._measurements = corrupted

    with pytest.raises(RuntimeError, match="Duplicate"):
        dataset.measurement(duplicated_id)


def test_open_ingested_dataset_never_builds(configured_store_paths: dict[str, Path]) -> None:
    with pytest.raises(FileNotFoundError):
        open_ingested_dataset(src.config)


def test_open_analysis_dataset_never_builds(configured_store_paths: dict[str, Path]) -> None:
    with pytest.raises(FileNotFoundError):
        open_analysis_dataset(src.config)


def test_analysis_dataset_spectra_matches_legacy_loader(configured_store_paths: dict[str, Path]) -> None:
    build_preprocessed_store(src.config, force=True)
    dataset = open_analysis_dataset(src.config)

    from src.data_preparation.loaders import load_spectra

    legacy = load_spectra(dataset.data_source, "1300")
    view = dataset.spectra("1300", "normalized")

    assert np.array_equal(view.values, legacy["spectra"])
    assert np.array_equal(view.wavenumbers, legacy["wavenumbers"])
    assert np.array_equal(view.measurement_ids, legacy["measurement_ids"])
    assert np.array_equal(view.subject_ids, legacy["subject_ids"])
    assert np.array_equal(view.file_ids, legacy["file_ids"])


def test_analysis_dataset_area_summaries_matches_legacy_loader(configured_store_paths: dict[str, Path]) -> None:
    build_preprocessed_store(src.config, force=True)
    dataset = open_analysis_dataset(src.config)

    from src.data_preparation.loaders import load_area_summaries

    legacy = load_area_summaries(dataset.data_source, "1300")
    normalized = dataset.area_summaries("1300", "normalized")
    pre_norm = dataset.area_summaries("1300", "pre_normalization")

    assert np.array_equal(normalized.values, legacy["area_summary"])
    assert np.array_equal(pre_norm.values, legacy["area_summary_bc"])
    assert np.array_equal(normalized.area_ids, legacy["area_ids"])


@pytest.mark.parametrize("basis", ["normalized", "pre_normalization"])
def test_analysis_dataset_whole_area_summaries_matches_legacy_loader(configured_store_paths: dict[str, Path], basis: str) -> None:
    build_preprocessed_store(src.config, force=True)
    dataset = open_analysis_dataset(src.config)

    from src.data_preparation.loaders import load_whole_area_store

    legacy = load_whole_area_store(dataset.data_source, basis=basis)
    view = dataset.whole_area_summaries(basis)
    assert view.basis == basis
    assert not view.values.flags.writeable

    assert np.array_equal(view.values, legacy["values"])
    assert np.array_equal(view.area_ids, legacy["area_ids"])
    assert np.array_equal(view.measurement_ids, legacy["measurement_ids"])


def test_analysis_dataset_spectra_are_read_only(configured_store_paths: dict[str, Path]) -> None:
    build_preprocessed_store(src.config, force=True)
    dataset = open_analysis_dataset(src.config)
    view = dataset.spectra("1300", "normalized")
    with pytest.raises(ValueError):
        view.values[0, 0] = 0.0


def test_prepare_dataset_builds_and_strict_validates(configured_store_paths: dict[str, Path]) -> None:
    dataset = prepare_dataset(src.config, force=True)
    assert isinstance(dataset, AnalysisRamanDataset)
    assert dataset.spectra("1300", "normalized").values.shape[0] > 0


def test_prepare_dataset_rejects_invalid_range_or_basis(configured_store_paths: dict[str, Path]) -> None:
    dataset = prepare_dataset(src.config, force=True)
    with pytest.raises(Exception):
        dataset.spectra("9999", "normalized")  # type: ignore[arg-type]
