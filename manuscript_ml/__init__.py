"""Experiment runners for the reproducible Raman lipid representation study.

All scripts delegate mathematical, statistical, and spectral operations to
``src.analysis.paired_lipid`` and ``src.analysis.io_utils``
to preserve DRY and KISS principles.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.io_utils import write_json_atomic  # noqa: E402
from src.analysis.paired_lipid.cohort import (  # noqa: E402
    COVARIATE_PRESETS,
    PRE_EVENT_RISK_FACTORS,
    SUBSET_MODES,
    CohortDataset,
    PairedLipidConfig,
    SubsetMode,
    load_paired_cohort_from_store,
)
from src.analysis.paired_lipid.nested_cv import INTERVAL_METHODS  # noqa: E402
from src.analysis.paired_lipid.synthetic import generate_synthetic_cohort  # noqa: E402
from src.analysis.raman_bands import RAMAN_BANDS  # noqa: E402

FROZEN_SPECTRAL_GRID_BOUNDS: dict[str, Any] = {
    "block_1300_cm1": [734, 1788],
    "block_2850_cm1": [2451, 3197],
    "step_cm1": 1.0,
    "n_channels_1300": 1055,
    "n_channels_2850": 747,
}

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Results_manuscript_ml"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "manuscript_ml.yaml"

_TRACKED_PACKAGES: tuple[str, ...] = (
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "joblib",
    "pyyaml",
    "pyarrow",
    "statsmodels",
    "cupy",
    "cupy-cuda12x",
    "cupy-cuda13x",
)

__all__ = [
    "CohortDataset",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_OUTPUT_DIR",
    "PRE_EVENT_RISK_FACTORS",
    "PROJECT_ROOT",
    "PairedLipidConfig",
    "SubsetMode",
    "add_cohort_stage_args",
    "add_common_stage_args",
    "clear_dataset_cache",
    "ensure_output_dir",
    "generate_synthetic_cohort",
    "load_manuscript_config",
    "load_paired_cohort_from_store",
    "print_stage_outputs",
    "require_cohort_approval",
    "resolve_experiment_dataset",
    "write_json_atomic",
    "write_run_provenance",
]


def _validate_manuscript_sections(raw: Mapping[str, Any]) -> None:
    """Validate optional top-level experiment sections and scenario keyword arguments upfront."""
    if "readiness" in raw:
        from manuscript_ml.readiness import validate_readiness_expectations

        validate_readiness_expectations(raw["readiness"])

    if "seeds" in raw:
        seeds = raw["seeds"]
        if not isinstance(seeds, Mapping):
            raise ValueError("'seeds' section in manuscript config must be a mapping")
        for key, val in seeds.items():
            if isinstance(val, bool) or not isinstance(val, int):
                raise ValueError(f"Seed {key!r} must be an integer, got {val!r}")

    if "anchors" in raw:
        anchors = raw["anchors"]
        if not isinstance(anchors, Mapping):
            raise ValueError("'anchors' section in manuscript config must be a mapping")
        if "n_resamples" in anchors:
            n_res = anchors["n_resamples"]
            if isinstance(n_res, bool) or not isinstance(n_res, int) or n_res < 0:
                raise ValueError("anchors.n_resamples must be a non-negative integer")
        if "seed" in anchors and (isinstance(anchors["seed"], bool) or not isinstance(anchors["seed"], int)):
            raise ValueError("anchors.seed must be an integer")

    if "calibration" in raw:
        cal = raw["calibration"]
        if not isinstance(cal, Mapping):
            raise ValueError("'calibration' section in manuscript config must be a mapping")
        if "n_test_participants" in cal:
            n_test = cal["n_test_participants"]
            if isinstance(n_test, bool) or not isinstance(n_test, int) or n_test <= 0:
                raise ValueError("calibration.n_test_participants must be a positive integer")
        if "interval_method" in cal:
            im = cal["interval_method"]
            if not isinstance(im, str) or im not in INTERVAL_METHODS:
                raise ValueError(
                    f"calibration.interval_method must be one of {INTERVAL_METHODS}, got {im!r}"
                )
        if "signal_strength" in cal:
            sig_val = cal["signal_strength"]
            if isinstance(sig_val, bool) or not isinstance(sig_val, (int, float)) or not math.isfinite(float(sig_val)):
                raise ValueError("calibration.signal_strength must be a finite number")
        if "gate_wilson_lower_min" in cal:
            gw = cal["gate_wilson_lower_min"]
            if isinstance(gw, bool) or not isinstance(gw, (int, float)) or not (0.0 < float(gw) < 1.0):
                raise ValueError("calibration.gate_wilson_lower_min must lie strictly in (0, 1)")
        if "gate_max_failure_rate" in cal:
            gf = cal["gate_max_failure_rate"]
            if isinstance(gf, bool) or not isinstance(gf, (int, float)) or not (0.0 <= float(gf) <= 1.0):
                raise ValueError("calibration.gate_max_failure_rate must lie in [0, 1]")

    if "simulation" in raw:
        sim = raw["simulation"]
        if not isinstance(sim, Mapping):
            raise ValueError("'simulation' section in manuscript config must be a mapping")
        if "compact_grid" in sim and not isinstance(sim["compact_grid"], bool):
            raise ValueError("simulation.compact_grid must be a boolean")
        if "n_participants_grid" in sim:
            grid = sim["n_participants_grid"]
            if (
                not isinstance(grid, (list, tuple))
                or not grid
                or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in grid)
            ):
                raise ValueError("simulation.n_participants_grid must be a non-empty list of positive integers")
        if "mde_signal_grid" in sim:
            mde_grid = sim["mde_signal_grid"]
            if (
                not isinstance(mde_grid, (list, tuple))
                or not mde_grid
                or any(
                    isinstance(v, bool)
                    or not isinstance(v, (int, float))
                    or not math.isfinite(float(v))
                    or float(v) < 0.0
                    for v in mde_grid
                )
            ):
                raise ValueError("simulation.mde_signal_grid must be a non-empty list of non-negative finite numbers")
        if "scenarios" in sim:
            scenarios = sim["scenarios"]
            if not isinstance(scenarios, (list, tuple)) or not scenarios:
                raise ValueError("simulation.scenarios must be a non-empty sequence of scenario mappings")
            reserved = {"n_participants", "seed", "compact_grid", "cfg"}
            sig_params = inspect.signature(generate_synthetic_cohort).parameters
            allowed_gen_keys = {
                name
                for name, param in sig_params.items()
                if name not in reserved
                and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            }
            for idx, spec in enumerate(scenarios):
                if not isinstance(spec, Mapping):
                    raise ValueError(f"simulation.scenarios[{idx}] must be a mapping")
                name = spec.get("scenario")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(f"simulation.scenarios[{idx}] must define a non-empty 'scenario' string")
                preset = spec.get("covariate_preset")
                if not isinstance(preset, str) or preset not in COVARIATE_PRESETS:
                    raise ValueError(f"Unknown covariate_preset {preset!r} in scenario {name!r}")
                extra_keys = set(spec.keys()) - {"scenario", "covariate_preset"}
                unknown_keys = sorted(extra_keys - allowed_gen_keys)
                if unknown_keys:
                    raise ValueError(
                        f"Unknown or reserved generator argument(s) in scenario {name!r}: {unknown_keys}"
                    )


def load_manuscript_config(path: str | Path | None = None) -> tuple[PairedLipidConfig, dict[str, Any]]:
    """Read and validate the locked experiment settings; return the model config and raw mapping."""
    raw = yaml.safe_load(Path(path or DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or "paired_lipid" not in raw or not isinstance(raw["paired_lipid"], Mapping):
        raise ValueError("Manuscript configuration YAML must contain a 'paired_lipid' mapping")
    fields = {k: tuple(v) if isinstance(v, list) else v for k, v in raw["paired_lipid"].items()}
    cfg = PairedLipidConfig(**fields)
    _validate_manuscript_sections(raw)
    return cfg, dict(raw)


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@lru_cache(maxsize=1)
def _git_metadata() -> tuple[str, bool | None, str | None]:
    commit = _git_output("log", "-1", "--format=%H")
    status = _git_output("status", "--porcelain")
    dirty = (status != "") if status is not None else None
    diff_sha: str | None = None
    if dirty:
        diff_text = _git_output("diff", "HEAD") or ""
        combined = f"{status}\n---\n{diff_text}".encode("utf-8", errors="replace")
        diff_sha = hashlib.sha256(combined).hexdigest()
    return (
        commit if commit is not None else "unavailable",
        dirty,
        diff_sha,
    )


@lru_cache(maxsize=1)
def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
    return versions


def write_run_provenance(
    target_dir: Path,
    stage: str,
    *,
    config_path: str | Path | None,
    cfg: PairedLipidConfig,
    settings: Mapping[str, Any],
    source_meta: Mapping[str, Any] | None = None,
) -> Path:
    """Save the resolved configuration, code version and data identity next to a stage's outputs."""
    cfg_file = Path(config_path or DEFAULT_CONFIG_PATH)
    config_bytes = cfg_file.read_bytes()
    source = dict(source_meta) if source_meta is not None else {}
    store_path = source.get("store_path") or source.get("preprocessed_dir")
    manifest_path = Path(store_path) / "manifest.json" if store_path is not None else None
    data_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path is not None and manifest_path.is_file()
        else None
    )
    git_commit, git_worktree_dirty, git_diff_sha256 = _git_metadata()
    raman_bands_snapshot = {
        name: {"peak": int(band.peak), "bounds": [int(band.bounds[0]), int(band.bounds[1])], "category": band.category}
        for name, band in RAMAN_BANDS.items()
    }
    payload = {
        "stage": stage,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": list(sys.argv),
        "config_path": str(cfg_file),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "config_text": config_bytes.decode("utf-8"),
        "data_manifest": data_manifest,
        "data_manifest_status": "saved" if data_manifest is not None else "not_available",
        "resolved_paired_lipid_config": asdict(cfg),
        "frozen_spectral_grid_bounds": dict(FROZEN_SPECTRAL_GRID_BOUNDS),
        "raman_bands_registry": raman_bands_snapshot,
        "pre_event_risk_factors": list(PRE_EVENT_RISK_FACTORS),
        "stage_settings": dict(settings),
        "git_commit": git_commit,
        "git_worktree_dirty": git_worktree_dirty,
        "git_diff_sha256": git_diff_sha256,
        "python": sys.version,
        "packages": dict(_package_versions()),
        "data": dict(source_meta) if source_meta is not None else None,
    }
    return write_json_atomic(target_dir / f"run_provenance_{stage}.json", payload)


def add_common_stage_args(parser: argparse.ArgumentParser) -> None:
    """Add output-directory and YAML configuration arguments shared across all CLI runners."""
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None, help="Experiment settings YAML.")


def add_cohort_stage_args(parser: argparse.ArgumentParser, *, include_compact_grid: bool = True) -> None:
    """Add the flags shared by the cohort-evaluation stages (primary evaluation, sensitivity, run_all)."""
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cohort", action="store_true", help="Evaluate on the verified stroke cohort (default).")
    mode.add_argument(
        "--synthetic",
        action="store_true",
        help="Explicit development simulation; outputs are not cohort evidence.",
    )
    parser.add_argument(
        "--approve-cohort-evaluation",
        action="store_true",
        help="Confirm that all readiness gates in runbook.md are satisfied.",
    )
    parser.add_argument("--clinical-master", type=str, default=str(PROJECT_ROOT / "data" / "clinical_master.csv"))
    if include_compact_grid:
        parser.add_argument("--compact-grid", action="store_true")
    add_common_stage_args(parser)


def require_cohort_approval(*, use_synthetic: bool, approve_cohort_evaluation: bool) -> None:
    """Refuse unapproved evaluations on the real clinical cohort."""
    if not use_synthetic and not approve_cohort_evaluation:
        raise PermissionError(
            "Cohort evaluation requires explicit '--approve-cohort-evaluation' after "
            "completing all readiness checks in manuscript_ml/runbook.md."
        )


def print_stage_outputs(stage_name: str, outputs: Mapping[str, Path]) -> None:
    """Print the output artifact key-to-path mapping for a completed CLI stage."""
    for key, path in outputs.items():
        print(f"[{stage_name}] wrote {key}: {path}")


def ensure_output_dir(output_dir: str | Path | None = None) -> Path:
    """Resolve and create the target output directory."""
    target = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target


def resolve_experiment_dataset(
    *,
    use_synthetic: bool,
    approve_cohort_evaluation: bool = False,
    clinical_master_path: str | Path = PROJECT_ROOT / "data" / "clinical_master.csv",
    subset_mode: SubsetMode = "all_eligible",
    synthetic_participants: int = 92,
    synthetic_seed: int = 42,
    compact_grid: bool = False,
    cfg: PairedLipidConfig = PairedLipidConfig(),
    readiness: Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
) -> tuple[CohortDataset, dict[str, Any]]:
    """Resolve either the synthetic benchmark cohort or the verified preprocessed cohort.

    Results are memoized per dataset-affecting argument set within a process, so
    repeated stages share one dataset object and its Gram cache. Treat the returned
    dataset as read-only; call ``clear_dataset_cache()`` to force a reload.
    """
    if subset_mode not in SUBSET_MODES:
        raise ValueError(f"Unknown subset_mode: {subset_mode!r}. Expected one of {SUBSET_MODES}.")
    require_cohort_approval(
        use_synthetic=use_synthetic,
        approve_cohort_evaluation=approve_cohort_evaluation,
    )
    source_identity = ""
    readiness_json = "{}"
    if not use_synthetic:
        from src.data_preparation.manifest import hash_file, hash_manifest, load_manifest
        from src.data_preparation.store import get_active_data_source, resolve_data_source

        source = get_active_data_source() or resolve_data_source()
        disk_manifest_path = source.source_path / "manifest.json"
        disk_manifest = load_manifest(disk_manifest_path)
        disk_manifest_hash = disk_manifest.get("manifest_hash")
        if disk_manifest_hash is not None:
            computed_disk_hash = hash_manifest(disk_manifest)
            if str(disk_manifest_hash) != str(computed_disk_hash) or (
                source.manifest_hash is not None and str(source.manifest_hash) != str(disk_manifest_hash)
            ):
                raise RuntimeError(
                    "Active preprocessed store manifest_hash does not match manifest.json on disk; "
                    "re-resolve or rebuild the store before cohort evaluation."
                )
        source_identity = hash_file(disk_manifest_path) + hash_file(clinical_master_path)
        expectations = (
            readiness
            if readiness is not None
            else load_manuscript_config(config_path)[1]["readiness"]
        )
        readiness_json = json.dumps(dict(expectations), sort_keys=True)

    dataset_cfg_key = (
        cfg.primary_endpoint,
        tuple(cfg.secondary_endpoints),
        int(cfg.min_maps_per_block),
    )
    dataset, cached_meta = _resolve_experiment_dataset_cached(
        use_synthetic,
        str(Path(clinical_master_path).resolve()),
        subset_mode,
        synthetic_participants,
        synthetic_seed,
        compact_grid,
        dataset_cfg_key,
        source_identity,
        readiness_json,
    )
    return dataset, copy.deepcopy(cached_meta)


def clear_dataset_cache() -> None:
    """Drop all memoized experiment datasets."""
    _resolve_experiment_dataset_cached.cache_clear()


@lru_cache(maxsize=4)
def _resolve_experiment_dataset_cached(
    use_synthetic: bool,
    clinical_master_path: str,
    subset_mode: SubsetMode,
    synthetic_participants: int,
    synthetic_seed: int,
    compact_grid: bool,
    dataset_cfg_key: tuple[str, tuple[str, ...], int],
    source_identity: str,
    readiness_json: str,
) -> tuple[CohortDataset, dict[str, Any]]:
    primary_endpoint, secondary_endpoints, min_maps_per_block = dataset_cfg_key
    load_cfg = PairedLipidConfig(
        primary_endpoint=primary_endpoint,
        secondary_endpoints=secondary_endpoints,
        min_maps_per_block=min_maps_per_block,
    )
    if use_synthetic:
        dataset, truth = generate_synthetic_cohort(
            n_participants=synthetic_participants,
            seed=synthetic_seed,
            compact_grid=compact_grid,
            subset_mode=subset_mode,
            cfg=load_cfg,
        )
        meta = {
            "data_mode": "synthetic",
            "synthetic_seed": synthetic_seed,
            "compact_grid": compact_grid,
            "signal_strength": truth["signal_strength"],
        }
        return dataset, meta

    from src.data_preparation.store import get_active_data_source, resolve_data_source

    if subset_mode != "all_eligible":
        # Validate and cache readiness once on the full eligible cohort rather than
        # loading all_eligible again on every non-primary subset mode.
        _, base_meta = _resolve_experiment_dataset_cached(
            False,
            clinical_master_path,
            "all_eligible",
            synthetic_participants,
            synthetic_seed,
            compact_grid,
            dataset_cfg_key,
            source_identity,
            readiness_json,
        )
        source = get_active_data_source() or resolve_data_source()
        dataset, _ = load_paired_cohort_from_store(
            clinical_master_path=clinical_master_path,
            data_source=source,
            cfg=load_cfg,
            subset_mode=subset_mode,
        )
        return dataset, copy.deepcopy(base_meta)

    source = get_active_data_source() or resolve_data_source()
    dataset, clin_audit = load_paired_cohort_from_store(
        clinical_master_path=clinical_master_path,
        data_source=source,
        cfg=load_cfg,
        subset_mode="all_eligible",
    )
    from manuscript_ml.readiness import evaluate_cohort_readiness

    measured_readiness = evaluate_cohort_readiness(clin_audit, dataset.manifest_summary, json.loads(readiness_json))
    if not measured_readiness["ready_for_cohort_evaluation"]:
        raise RuntimeError("Current clinical identity or cohort-join counts do not match readiness expectations")
    return dataset, {
        "readiness": measured_readiness,
        "source_identity": source_identity,
        "data_mode": "cohort",
        "clinical_audit": clin_audit,
        "store_path": str(source.source_path),
        "store_manifest_hash": str(source.manifest_hash),
    }
