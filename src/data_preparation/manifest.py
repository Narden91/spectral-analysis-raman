from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


MANIFEST_VERSION = 1
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FileSignature:
    sha256: str
    size: int
    mtime_ns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
        }


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


def file_signature(path: str | Path) -> FileSignature:
    file_path = Path(path)
    stat = file_path.stat()
    return FileSignature(
        sha256=hash_file(file_path),
        size=int(stat.st_size),
        mtime_ns=int(stat.st_mtime_ns),
    )


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


def signatures_for_paths(paths: dict[str, str | Path]) -> dict[str, dict[str, Any]]:
    return {
        name: file_signature(path).to_dict()
        for name, path in sorted(paths.items())
    }

