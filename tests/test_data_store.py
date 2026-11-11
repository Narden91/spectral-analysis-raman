from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src
from src.analysis.spectral_loader import build_group_matrix
from src.data_preparation import loaders as loaders_module
from src.data_preparation import validation as validation_module
from src.data_preparation.loaders import (
    clear_loader_caches,
    load_area_summaries,
    load_file_index,
    load_spectra,
    load_whole_area_store,
)
from src.data_preparation.materialize import (
    MaterializedStore,
    _append_reason,
    _atomic_replace,
    build_parsed_store,
    build_preprocessed_store,
)
from src.data_preparation.store import (
    DataSource,
    get_active_data_source,
    resolve_data_source,
    write_analysis_provenance,
)
from src.data_preparation.validation import validate_store


def test_append_reason_never_duplicates_existing_token() -> None:
    values = pd.Series(["ok", "area_min_pixels", "area_min_pixels; quality_filter"])
    mask = np.array([True, True, True])

    result = _append_reason(values, "area_min_pixels", mask)

    assert result.tolist() == ["area_min_pixels", "area_min_pixels", "area_min_pixels; quality_filter"]


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
    assert (store.path / "measurements.parquet").exists()
    assert (store.path / "raw_spectra_1300.npy").exists()
    assert (store.path / "pixels.parquet").exists()
    assert (store.path / "cohort" / "control_vs_disease" / "subject_summary.parquet").exists()

    validation = validate_store(store.path, "parsed", src.config)
    assert validation.is_valid

    manifest = json.loads((store.path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["areas"] == 2

    assert not (store.path / "spectra_1300.npz").exists()
    assert not (store.path / "area_summaries_1300.npz").exists()


def test_preprocessed_store_uses_supplied_normalization_config(configured_store_paths: dict[str, Path]) -> None:
    original = deepcopy(src.config)
    override = deepcopy(original)
    for region in ("fingerprint", "ch_stretch"):
        override["preprocessing"]["ranges"][region]["normalization"] = {"method": "none"}
    store = build_preprocessed_store(override, force=True)
    for code in ("1300", "2850"):
        normalized = np.load(store.path / f"normalized_{code}.npy")
        pre = np.load(store.path / f"pre_normalization_{code}.npy")
        assert len(normalized) > 0
        np.testing.assert_array_equal(normalized, pre)
    assert validate_store(store.path, "preprocessed", override, strict=True).is_valid
    assert src.config == original


def test_parsed_store_uses_supplied_clinical_path(configured_store_paths: dict[str, Path]) -> None:
    original = deepcopy(src.config)
    override = deepcopy(original)
    path = configured_store_paths["clinical_path"].with_name("other_clinical.csv")
    clinical = pd.read_csv(configured_store_paths["clinical_path"])
    clinical["age_years"] = 99
    clinical.to_csv(path, index=False)
    override["data"]["clinical_data_path"] = str(path)
    store = build_parsed_store(override, force=True)
    subjects = pd.read_parquet(store.path / "subjects.parquet")
    assert subjects["age_years"].tolist() == [99, 99]
    assert validate_store(store.path, "parsed", override, strict=True).is_valid
    assert src.config == original


def test_atomic_store_build_never_creates_staging_directories_in_data_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    target = data_root / "parsed"
    target.mkdir(parents=True)
    (target / "old.txt").write_text("old", encoding="utf-8")
    system_temp = tmp_path / "system_temp"
    system_temp.mkdir()
    monkeypatch.setattr("src.data_preparation.materialize.tempfile.tempdir", str(system_temp))

    def writer(staging: Path) -> MaterializedStore:
        (staging / "new.txt").write_text("new", encoding="utf-8")
        return MaterializedStore("parsed", target, {})

    _atomic_replace(target, writer)

    assert (target / "new.txt").read_text(encoding="utf-8") == "new"
    assert not list(data_root.glob(".parsed_*"))
    assert not list(system_temp.glob("raman_parsed_*"))


def test_parsed_store_preserves_measurement_specific_native_axes(configured_store_paths: dict[str, Path]) -> None:
    shifted = configured_store_paths["raw_dir"] / "Disease" / "Male" / "2" / "sample_1" / "2_1_1300_range_meas1.txt"
    _write_measurement_file(shifted, [1240.2 + idx for idx in range(10)], 20.0)
    store = build_parsed_store(src.config, force=True)
    axes = np.load(store.path / "native_wavenumbers_1300.npy")
    values = np.load(store.path / "raw_spectra_1300.npy")
    assert axes.shape == (2, 10)
    assert np.isclose(axes[1, 0], 1240.2)
    assert values.shape[1] == axes.shape[1]


def test_prepare_store_callbacks_report_raw_and_preprocessing_progress(
    configured_store_paths: dict[str, Path],
) -> None:
    events: list[tuple[str, int, int]] = []

    build_preprocessed_store(src.config, force=True, progress_callback=lambda stage, completed, total: events.append((stage, completed, total)))

    assert {stage for stage, _, _ in events} == {"Raw ingestion", "Preprocessing"}
    for stage in ("Raw ingestion", "Preprocessing"):
        stage_events = [event for event in events if event[0] == stage]
        assert stage_events[-1][1] == stage_events[-1][2] == 4


def test_resolve_preprocessed_source_and_group_matrix(configured_store_paths: dict[str, Path]) -> None:
    build_preprocessed_store(src.config, force=True)
    resolved = resolve_data_source(src.config)
    assert resolved.source_name == "preprocessed"
    assert get_active_data_source() is not None

    file_index = load_file_index(resolved)
    wave, matrix, subject_ids = build_group_matrix(
        file_index,
        gender="male",
        is_control=False,
        spectral_range="1300",
    )
    assert wave.size > 0
    assert matrix.shape[0] == 1
    assert subject_ids == ["2"]
    whole = load_whole_area_store(resolved)
    assert whole["values"].shape[0] == 2


def test_store_readers_share_canonical_arrays(configured_store_paths: dict[str, Path]) -> None:
    build_preprocessed_store(src.config, force=True)
    source = resolve_data_source(src.config)

    spectra = load_spectra(source, "1300")
    summaries = load_area_summaries(source, "1300")

    assert isinstance(spectra["spectra"], np.memmap)
    assert spectra["spectra"].shape[0] == spectra["measurement_ids"].size
    assert summaries["area_summary"].shape == summaries["area_summary_bc"].shape
    assert summaries["area_summary"].shape[0] == summaries["measurement_ids"].size
    assert summaries["feature_names"].shape[0] == spectra["wavenumbers"].size


def test_stale_manifest_fails_after_raw_input_change(configured_store_paths: dict[str, Path]) -> None:
    build_parsed_store(src.config, force=True)
    target_file = configured_store_paths["raw_dir"] / "Disease" / "Male" / "2" / "sample_1" / "2_1_1300_range_meas1.txt"
    target_file.write_text(target_file.read_text(encoding="utf-8") + "\n0.0\t0.0\t1250.000\t99.000", encoding="utf-8")

    parsed_cfg = deepcopy(src.config)
    parsed_cfg["data"]["source"] = "parsed"
    with pytest.raises(RuntimeError, match="stale"):
        resolve_data_source(parsed_cfg)


def test_validation_fails_when_required_file_missing(configured_store_paths: dict[str, Path]) -> None:
    store = build_parsed_store(src.config, force=True)
    (store.path / "areas.parquet").unlink()
    validation = validate_store(store.path, "parsed", src.config)
    assert not validation.is_valid
    assert any("areas.parquet" in error for error in validation.errors)


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


def test_preprocessed_store_validation_enforces_resolved_preprocessing_hash(
    configured_store_paths: dict[str, Path],
) -> None:
    store = build_preprocessed_store(src.config, force=True)
    reordered = deepcopy(src.config)
    reordered["preprocessing"] = dict(reversed(list(reordered["preprocessing"].items())))
    assert validate_store(store.path, "preprocessed", reordered).is_valid

    changed = deepcopy(src.config)
    changed["preprocessing"]["quality_filter"]["spectral_quality"]["snr_threshold"] += 0.5
    validation = validate_store(store.path, "preprocessed", changed)
    assert validation.is_stale
    assert not validation.is_valid
    assert any("preprocessing configuration" in error for error in validation.errors)


def test_parsed_store_ignores_preprocessing_only_changes(configured_store_paths: dict[str, Path]) -> None:
    store = build_parsed_store(src.config, force=True)
    changed = deepcopy(src.config)
    changed["preprocessing"]["quality_filter"]["spectral_quality"]["snr_threshold"] += 0.5
    assert validate_store(store.path, "parsed", changed).is_valid


def test_load_area_summaries_hits_cache_on_repeated_calls(
    configured_store_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_preprocessed_store(src.config, force=True)
    source = resolve_data_source(src.config)
    clear_loader_caches()

    calls = {"count": 0}
    original = pd.read_parquet

    def counting_read_parquet(*args: object, **kwargs: object) -> pd.DataFrame:
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(loaders_module.pd, "read_parquet", counting_read_parquet)

    first = load_area_summaries(source, "1300")
    calls_after_first = calls["count"]
    second = load_area_summaries(source, "1300")

    assert calls["count"] == calls_after_first, "second call must not re-read parquet"
    assert np.array_equal(first["area_summary"], second["area_summary"])
    assert first["area_summary"] is second["area_summary"]


def test_load_whole_area_store_hits_cache_on_repeated_calls(
    configured_store_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_preprocessed_store(src.config, force=True)
    source = resolve_data_source(src.config)
    clear_loader_caches()

    calls = {"count": 0}
    original = np.nanmedian

    def counting_nanmedian(*args: object, **kwargs: object) -> np.ndarray:
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(loaders_module.np, "nanmedian", counting_nanmedian)

    first = load_whole_area_store(source)
    calls_after_first = calls["count"]
    second = load_whole_area_store(source)

    assert calls["count"] == calls_after_first, "second call must not recompute medians"
    assert first["values"] is second["values"]


def test_forced_rebuild_invalidates_loader_cache(configured_store_paths: dict[str, Path]) -> None:
    build_preprocessed_store(src.config, force=True)
    source_before = resolve_data_source(src.config, force=True)
    summary_before = load_area_summaries(source_before, "1300")

    source_after = resolve_data_source(src.config, force=True)
    summary_after = load_area_summaries(source_after, "1300")

    assert source_before.manifest_hash != source_after.manifest_hash
    assert summary_before["area_summary"] is not summary_after["area_summary"]
    assert np.array_equal(summary_before["area_summary"], summary_after["area_summary"])


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


def test_code_version_fails_when_git_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data_preparation import materialize

    def fail(*args, **kwargs):
        raise FileNotFoundError("git is unavailable")

    monkeypatch.setattr(materialize.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="Cannot record the repository version"):
        materialize._code_version()


def test_code_version_uses_repository_directory_and_hashes_dirty_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib
    from types import SimpleNamespace

    from src.data_preparation import materialize

    calls = []
    diff = b"local changes"

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="a" * 40 + "\n" if command[1] == "log" else diff)

    monkeypatch.setattr(materialize.subprocess, "run", run)

    assert materialize._code_version() == "a" * 40 + "+dirty:" + hashlib.sha256(diff).hexdigest()[:16]
    assert len(calls) == 2
    assert all(kwargs["cwd"] == Path(materialize.__file__).resolve().parents[2] for _, kwargs in calls)


def test_code_version_rejects_empty_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from src.data_preparation import materialize

    monkeypatch.setattr(materialize.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=""))
    with pytest.raises(RuntimeError, match="Repository version is empty"):
        materialize._code_version()
