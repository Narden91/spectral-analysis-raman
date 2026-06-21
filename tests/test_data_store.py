from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src
from src.analysis.spectral_loader import build_group_matrix
from src.data_preparation.materialize import build_parsed_store, build_preprocessed_store
from src.data_preparation.store import DataSource, get_active_data_source, resolve_data_source, write_analysis_provenance
from src.data_preparation import validation as validation_module
from src.data_preparation.validation import validate_store


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
            "write_csv_mirrors": True,
            "require_valid_manifest": True,
            "stale_manifest_policy": "error",
        }
    )
    src.config["preprocessing"]["retention"]["min_pixels_per_area"] = 1
    src.config["preprocessing"]["retention"]["min_pixels_per_subject"] = 1
    src.config["preprocessing"]["quality_filter"]["spectral_quality"]["snr_threshold"] = 0.0
    src.config["preprocessing"]["quality_filter"]["spectral_quality"]["min_area_percentile"] = 0
    src.config["output"]["keep_preprocessing_artifacts"] = False
    src.config["output"]["keep_whole_spectra_cache"] = False

    waves_1300 = [1240.0 + idx for idx in range(10)]
    waves_2850 = [2840.0 + idx for idx in range(10)]
    _write_measurement_file(
        raw_dir / "Control" / "Female" / "1" / "sample_1" / "1_1_1300_range_meas1.txt",
        waves_1300,
        10.0,
    )
    _write_measurement_file(
        raw_dir / "Control" / "Female" / "1" / "sample_1" / "1_1_2850_range_meas1.txt",
        waves_2850,
        15.0,
    )
    _write_measurement_file(
        raw_dir / "Disease" / "Male" / "2" / "sample_1" / "2_1_1300_range_meas1.txt",
        waves_1300,
        20.0,
    )
    _write_measurement_file(
        raw_dir / "Disease" / "Male" / "2" / "sample_1" / "2_1_2850_range_meas1.txt",
        waves_2850,
        25.0,
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


def test_build_parsed_store_and_validate(configured_store_paths: dict[str, Path]) -> None:
    store = build_parsed_store(src.config, force=True)
    assert (store.path / "file_index.parquet").exists()
    assert (store.path / "spectra_1300.npz").exists()
    assert (store.path / "cohort" / "control_vs_disease" / "subject_summary.parquet").exists()

    validation = validate_store(store.path, "parsed", src.config)
    assert validation.is_valid

    manifest = json.loads((store.path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["areas"] == 2

    with np.load(store.path / "whole_area_store.npz", allow_pickle=False) as data:
        assert np.asarray(data["values"]).shape[0] == 2
        assert set(np.asarray(data["subject_ids"]).astype(str).tolist()) == {"1", "2"}


def test_resolve_preprocessed_source_and_group_matrix(configured_store_paths: dict[str, Path]) -> None:
    build_preprocessed_store(src.config, force=True)
    resolved = resolve_data_source(src.config, source_override="preprocessed")
    assert resolved.source_name == "preprocessed"
    assert get_active_data_source() is not None

    file_index = pd.read_parquet(configured_store_paths["preprocessed_dir"] / "file_index.parquet")
    wave, matrix, subject_ids = build_group_matrix(
        file_index,
        gender="male",
        is_control=False,
        spectral_range="1300",
    )
    assert wave.size > 0
    assert matrix.shape[0] == 1
    assert subject_ids == ["2"]


def test_stale_manifest_fails_after_raw_input_change(configured_store_paths: dict[str, Path]) -> None:
    build_parsed_store(src.config, force=True)
    target_file = configured_store_paths["raw_dir"] / "Disease" / "Male" / "2" / "sample_1" / "2_1_1300_range_meas1.txt"
    target_file.write_text(target_file.read_text(encoding="utf-8") + "\n0.0\t0.0\t1250.000\t99.000", encoding="utf-8")

    with pytest.raises(RuntimeError, match="stale"):
        resolve_data_source(src.config, source_override="parsed")


def test_validation_fails_when_required_file_missing(configured_store_paths: dict[str, Path]) -> None:
    store = build_parsed_store(src.config, force=True)
    (store.path / "area_metadata.parquet").unlink()
    validation = validate_store(store.path, "parsed", src.config)
    assert not validation.is_valid
    assert any("area_metadata.parquet" in error for error in validation.errors)


def test_validate_store_uses_stat_fast_path_for_unchanged_inputs(
    configured_store_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = build_parsed_store(src.config, force=True)

    def _unexpected_hash(_: str | Path) -> str:
        raise AssertionError("hash_file should not be called for unchanged input signatures")

    monkeypatch.setattr(validation_module, "hash_file", _unexpected_hash)
    validation = validate_store(store.path, "parsed", src.config)
    assert validation.is_valid


def test_write_analysis_provenance_records_analysis_mode(tmp_path: Path) -> None:
    data_source = DataSource(
        source_name="preprocessed",
        source_path=tmp_path / "preprocessed",
        manifest={"counts": {"subjects": 2}},
        manifest_hash="manifest-hash",
        was_built=False,
    )
    cfg = {"analysis": {"mode": "disease_only"}}

    write_analysis_provenance(tmp_path / "provenance", data_source, cfg, command="uv run scripts/run_eda.py")

    run_manifest = json.loads((tmp_path / "provenance" / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["analysis_mode"] == "disease_only"
