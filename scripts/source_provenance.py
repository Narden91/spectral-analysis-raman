"""Record saved-result inputs separately from the settings used to render them."""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from src.analysis.io_utils import write_json_atomic
from src.data_preparation.manifest import hash_file

_SNAPSHOT_NAMES = ("config_snapshot.yaml", "data_manifest_snapshot.json", "run_manifest.json")


def write_source_provenance(
    output_dir: str | Path,
    *,
    source_files: Iterable[str | Path],
    rendering_settings: Mapping[str, Any],
    source_metadata_dirs: Iterable[str | Path] = (),
) -> Path:
    """Hash source artifacts and copy genuine archived run snapshots when available.

    Nearby metadata is recorded as found, without asserting that a run is ready
    or that a saved result has received independent validation. Rendering settings
    are kept separate and never substituted for absent source-run settings.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sources = sorted({Path(path).resolve() for path in source_files})
    metadata_dirs = {Path(path).resolve() for path in source_metadata_dirs}
    source_rows = []
    for path in sources:
        row: dict[str, Any] = {"path": str(path)}
        try:
            row.update(sha256=hash_file(path), status="recorded")
        except OSError as exc:
            row.update(status="unavailable", error_type=type(exc).__name__)
        for ancestor in path.parents:
            candidate = ancestor / "00_run_metadata"
            if candidate.is_dir():
                metadata_dirs.add(candidate)
                row["nearby_source_metadata_dir"] = str(candidate)
                break
        source_rows.append(row)

    snapshot_rows = []
    for index, metadata_dir in enumerate(sorted(metadata_dirs)):
        available = []
        for name in _SNAPSHOT_NAMES:
            source = metadata_dir / name
            if not source.is_file():
                continue
            destination = output / "source_run_snapshots" / f"{index:02d}" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            available.append({"name": name, "path": str(destination.relative_to(output)), "sha256": hash_file(destination)})
        snapshot_rows.append({
            "source_metadata_dir": str(metadata_dir), "snapshots": available,
            "status": "found" if len(available) == len(_SNAPSHOT_NAMES) else "incomplete",
        })
    status = "found" if snapshot_rows and all(row["status"] == "found" for row in snapshot_rows) else "unavailable"
    if snapshot_rows and status != "found":
        status = "incomplete"
    return write_json_atomic(output / "source_provenance.json", {
        "source_files": source_rows,
        "source_run_provenance_status": status,
        "source_run_snapshots": snapshot_rows,
        "rendering_settings": dict(rendering_settings),
        "verification_limit": "Archived snapshots are copied as found; no dataset readiness or independent-validation claim is inferred.",
    })
