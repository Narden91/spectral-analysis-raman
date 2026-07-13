"""Deterministic per-area preprocessing cache.

Writes one artifact per (subject_id, slide, measurement, range) area to
``data_preprocessed/<range_label>/<subject_id>/<area_id>.npz`` so every EDA
analysis (PCA, univariate, intra-disease correlation) loads the
same corrected, retained pixels instead of recomputing or mixing raw/SNV
paths.

Freshness follows the existing whole-spectra cache pattern
(``_whole_spectra_cache_key`` in ``src/analysis/spectral_loader.py``): the
key is derived from source file mtime/size plus a hash of the active
preprocessing config, so a stale or missing artifact is rebuilt and a
config change invalidates every cached area.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import config as _cfg
from src import resolve_project_path
from src.io.txt_reader import read_txt_spectrum
from src.preprocessing.pipeline import flatten_spectra_rows, retention_guard_config, run_preprocessing

logger = logging.getLogger(__name__)

_REGION_LABELS = {"fingerprint": "1300", "ch_stretch": "2850"}


@dataclass
class CachedArea:
    wavenumbers: np.ndarray
    spectra: np.ndarray
    manifest: dict


def _preprocessed_root() -> Path:
    root = _cfg.get("data", {}).get("preprocessed_dir", "data_preprocessed")
    return resolve_project_path(root)


def _cache_enabled() -> bool:
    return bool(_cfg.get("preprocessing", {}).get("cache", {}).get("enabled", True))


def _cache_version() -> int:
    return int(_cfg.get("preprocessing", {}).get("cache", {}).get("version", 1))


def _config_hash() -> str:
    payload = {
        "preprocessing": _cfg.get("preprocessing", {}),
        "cache_version": _cache_version(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _file_signature(filepath: str | Path) -> tuple[float, int] | None:
    try:
        stat = Path(filepath).stat()
        return (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return None


def _area_paths(range_label: str, subject_id: str, area_id: str) -> tuple[Path, Path]:
    area_dir = _preprocessed_root() / range_label / str(subject_id)
    npz_path = area_dir / f"{area_id}.npz"
    manifest_path = area_dir / f"{area_id}.manifest.json"
    return npz_path, manifest_path


def _is_fresh(manifest_path: Path, npz_path: Path, filepath: str, config_hash: str) -> bool:
    if not manifest_path.exists() or not npz_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("config_hash") != config_hash:
        return False
    signature = _file_signature(filepath)
    if signature is None:
        return False
    return (
        manifest.get("source_mtime_ns") == signature[0]
        and manifest.get("source_size") == signature[1]
    )


def build_area_cache(
    filepath: str | Path,
    *,
    subject_id: str,
    slide_number: str,
    area_id: str,
    region: str,
    force: bool = False,
) -> CachedArea:
    """Build (or load) the deterministic preprocessing cache for one area.

    `region` is the canonical region key ("fingerprint" or "ch_stretch").
    """
    range_label = _REGION_LABELS[region]
    npz_path, manifest_path = _area_paths(range_label, subject_id, area_id)
    config_hash = _config_hash()
    filepath = str(filepath)

    if _cache_enabled() and not force and _is_fresh(manifest_path, npz_path, filepath, config_hash):
        with np.load(npz_path) as npz:
            wavenumbers = npz["wavenumbers"]
            spectra = npz["spectra"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return CachedArea(wavenumbers=wavenumbers, spectra=spectra, manifest=manifest)

    wavenumbers, _, _, cube, _ = read_txt_spectrum(filepath)
    flat = flatten_spectra_rows(cube)

    n_input = int(flat.shape[0])
    result = run_preprocessing(
        flat,
        wavenumbers,
        region=region,
        screen_dead=True,
        suppress_cosmic=None,
        correct_bl=True,
        smooth=None,
        interpolate=False,
        filter_quality=None,
        normalize=True,
        norm_for="eda",
    )

    guards = retention_guard_config()
    n_retained = int(result.spectra.shape[0])
    dropped_reason = None
    if n_retained < guards["min_pixels_per_area"]:
        dropped_reason = (
            f"n_retained ({n_retained}) below min_pixels_per_area "
            f"({guards['min_pixels_per_area']})"
        )
        spectra_out = np.empty((0, result.spectra.shape[1] if result.spectra.size else 0), dtype=float)
    else:
        spectra_out = result.spectra

    signature = _file_signature(filepath)
    manifest = {
        "subject_id": subject_id,
        "slide_number": slide_number,
        "area_id": area_id,
        "region": region,
        "n_input": n_input,
        "n_retained": int(spectra_out.shape[0]),
        "dropped_reason": dropped_reason,
        "config_hash": config_hash,
        "source_filepath": filepath,
        "source_mtime_ns": signature[0] if signature else None,
        "source_size": signature[1] if signature else None,
    }

    if _cache_enabled():
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, wavenumbers=result.wavenumbers, spectra=spectra_out)
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    return CachedArea(wavenumbers=result.wavenumbers, spectra=spectra_out, manifest=manifest)


def load_subject_pixels(
    subject_id: str,
    region: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate all cached, retained pixels for a subject in one region."""
    range_label = _REGION_LABELS[region]
    subject_dir = _preprocessed_root() / range_label / str(subject_id)
    if not subject_dir.exists():
        return np.array([]), np.empty((0, 0))

    all_spectra: list[np.ndarray] = []
    wavenumbers: np.ndarray | None = None
    for npz_path in sorted(subject_dir.glob("*.npz")):
        with np.load(npz_path) as npz:
            wave = npz["wavenumbers"]
            spectra = npz["spectra"]
        if spectra.size == 0:
            continue
        if wavenumbers is None:
            wavenumbers = wave
        all_spectra.append(spectra)

    if not all_spectra or wavenumbers is None:
        return np.array([]), np.empty((0, 0))
    return wavenumbers, np.vstack(all_spectra)


def load_subject_mean(
    subject_id: str,
    region: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Subject-mean spectrum derived from all cached, retained pixels."""
    guards = retention_guard_config()
    wavenumbers, pixels = load_subject_pixels(subject_id, region)
    if pixels.size == 0 or pixels.shape[0] < guards["min_pixels_per_subject"]:
        return wavenumbers, np.array([])
    return wavenumbers, np.nanmean(pixels, axis=0)
