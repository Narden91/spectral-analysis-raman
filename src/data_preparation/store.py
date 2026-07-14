from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src import resolve_cfg as _cfg
from src import resolve_project_path
from src.data_preparation.manifest import config_hash, iso_utc_now
from src.data_preparation.materialize import build_parsed_store, build_preprocessed_store
from src.data_preparation.validation import validate_store


@dataclass(frozen=True)
class DataSource:
    source_name: str
    source_path: Path
    manifest: dict[str, Any] | None
    manifest_hash: str | None
    was_built: bool


_ACTIVE_DATA_SOURCE: DataSource | None = None


def set_active_data_source(data_source: DataSource) -> None:
    global _ACTIVE_DATA_SOURCE
    _ACTIVE_DATA_SOURCE = data_source


def get_active_data_source() -> DataSource | None:
    return _ACTIVE_DATA_SOURCE


def _store_dir(source_name: str, cfg: dict) -> Path:
    data_cfg = cfg.get("data", {})
    if source_name == "parsed":
        return resolve_project_path(data_cfg.get("parsed_dir", "data/parsed"))
    if source_name == "preprocessed":
        return resolve_project_path(data_cfg.get("preprocessed_dir", "data/preprocessed"))
    raise ValueError(f"Unsupported data source: {source_name}")


def _validated_source(source_name: str, cfg: dict, *, was_built: bool) -> DataSource:
    source_path = _store_dir(source_name, cfg)
    validation = validate_store(source_path, source_name, cfg)
    if not validation.is_valid:
        raise RuntimeError("; ".join(validation.errors))
    return DataSource(
        source_name=source_name,
        source_path=source_path,
        manifest=validation.manifest,
        manifest_hash=validation.manifest_hash,
        was_built=was_built,
    )


def resolve_data_source(
    cfg: dict | None = None,
    *,
    source_override: str | None = None,
    force: bool = False,
) -> DataSource:
    config = _cfg(cfg)
    source_name = str(source_override or config.get("data", {}).get("source", "preprocessed")).strip()
    auto_build = bool(config.get("data", {}).get("auto_build", True))

    if source_name == "raw_text":
        raise ValueError("raw_text is ingestion-only; build the parsed store before analysis")

    source_path = _store_dir(source_name, config)
    if force:
        if source_name == "parsed":
            build_parsed_store(config, force=True)
        elif source_name == "preprocessed":
            build_preprocessed_store(config, force=True)
        data_source = _validated_source(source_name, config, was_built=True)
        set_active_data_source(data_source)
        return data_source

    if source_path.exists():
        validation = validate_store(source_path, source_name, config)
        if validation.is_valid:
            data_source = DataSource(
                source_name=source_name,
                source_path=source_path,
                manifest=validation.manifest,
                manifest_hash=validation.manifest_hash,
                was_built=False,
            )
            set_active_data_source(data_source)
            return data_source
        raise RuntimeError("; ".join(validation.errors))

    if not auto_build:
        raise FileNotFoundError(
            f"Selected data source '{source_name}' is missing at {source_path} and auto_build is disabled"
        )

    if source_name == "parsed":
        build_parsed_store(config, force=False)
    elif source_name == "preprocessed":
        build_preprocessed_store(config, force=False)
    else:
        raise ValueError(f"Unsupported data source: {source_name}")
    data_source = _validated_source(source_name, config, was_built=True)
    set_active_data_source(data_source)
    return data_source


def data_source_summary(data_source: DataSource) -> dict[str, object]:
    manifest = data_source.manifest or {}
    counts = manifest.get("counts", {})
    return {
        "Data source": data_source.source_name,
        "Data path": str(data_source.source_path),
        "Table format": manifest.get("table_format", "raw_text"),
        "Array format": manifest.get("array_format", "raw_text"),
        "Manifest hash": data_source.manifest_hash or "n/a",
        "Manifest schema version": manifest.get("schema_version", "n/a"),
        "Subjects": counts.get("subjects", "n/a"),
        "Files": counts.get("files", "n/a"),
        "Measurements": counts.get("measurements", "n/a"),
        "Areas": counts.get("areas", "n/a"),
        "Pixels 1300": counts.get("retained_pixels_1300", counts.get("pixels_1300", "n/a")),
        "Pixels 2850": counts.get("retained_pixels_2850", counts.get("pixels_2850", "n/a")),
        "Source state": "built" if data_source.was_built else "loaded",
    }


def write_analysis_provenance(
    output_dir: str | Path,
    data_source: DataSource,
    cfg: dict,
    *,
    command: str,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest = data_source.manifest or {}
    run_manifest = {
        "active_data_source": data_source.source_name,
        "data_source_path": str(data_source.source_path),
        "data_manifest_hash": data_source.manifest_hash,
        "analysis_config_hash": config_hash(cfg),
        "analysis_mode": cfg.get("analysis", {}).get("mode"),
        "started_at": iso_utc_now(),
        "command": command,
    }
    (output_path / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_path / "data_manifest_snapshot.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_path / "config_snapshot.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=True),
        encoding="utf-8",
    )
