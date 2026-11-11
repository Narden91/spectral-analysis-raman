from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1
SCHEMA_VERSION = 5


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hash_file(path: str | Path) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_signature(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    stat = file_path.stat()
    return {
        "sha256": hash_file(file_path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def hash_json(payload: Any) -> str:
    return hash_bytes(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))


def hash_manifest(manifest: dict[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_hash", None)
    return hash_json(payload)


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> dict[str, Any]:
    output_path = Path(path)
    payload = dict(manifest)
    payload["manifest_hash"] = hash_manifest(payload)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def config_hash(payload: dict[str, Any]) -> str:
    return hash_json(payload)
