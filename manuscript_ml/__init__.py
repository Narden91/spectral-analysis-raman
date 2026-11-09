"""Experiment runners for the reproducible Raman lipid representation study.

All scripts delegate mathematical, statistical, and spectral operations to
``src.analysis.paired_lipid`` and ``src.analysis.io_utils``
to preserve DRY and KISS principles.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.paired_lipid.cohort import (
    PRE_EVENT_RISK_FACTORS,
    CohortDataset,
    PairedLipidConfig,
    SubsetMode,
    load_paired_cohort_from_store,
)
from src.analysis.paired_lipid.simulation import generate_synthetic_cohort

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Results_manuscript_ml"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "manuscript_ml.yaml"


def load_manuscript_config(path: str | Path | None = None) -> tuple[PairedLipidConfig, dict[str, Any]]:
    """Read the locked experiment settings; return the model config and the raw mapping."""
    raw = yaml.safe_load(Path(path or DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
    fields = {k: tuple(v) if isinstance(v, list) else v for k, v in raw["paired_lipid"].items()}
    return PairedLipidConfig(**fields), raw


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _package_versions() -> dict[str, str]:
    versions = {}
    for pkg in ("numpy", "scipy", "pandas", "scikit-learn", "joblib", "cupy-cuda12x", "cupy-cuda13x"):
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
    payload = {
        "stage": stage,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv,
        "config_path": str(cfg_file),
        "config_sha256": hashlib.sha256(cfg_file.read_bytes()).hexdigest(),
        "resolved_paired_lipid_config": asdict(cfg),
        "pre_event_risk_factors": list(PRE_EVENT_RISK_FACTORS),
        "stage_settings": dict(settings),
        "git_commit": _git("log", "-1", "--format=%H"),
        "git_worktree_dirty": _git("status", "--porcelain") not in ("", "unavailable"),
        "python": sys.version,
        "packages": _package_versions(),
        "data": dict(source_meta) if source_meta is not None else None,
    }
    return write_json_atomic(payload, target_dir / f"run_provenance_{stage}.json")


def add_cohort_stage_args(parser: argparse.ArgumentParser) -> None:
    """Add the flags shared by the cohort-evaluation stages (primary evaluation, sensitivity)."""
    parser.add_argument("--cohort", action="store_true", help="Evaluate on the verified stroke cohort.")
    parser.add_argument(
        "--approve-cohort-evaluation",
        action="store_true",
        help="Confirm that all readiness gates in runbook.md are satisfied.",
    )
    parser.add_argument("--clinical-master", type=str, default=str(PROJECT_ROOT / "data" / "clinical_master.csv"))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--compact-grid", action="store_true")
    parser.add_argument("--config", type=str, default=None, help="Experiment settings YAML.")


def ensure_output_dir(output_dir: str | Path | None = None) -> Path:
    """Resolve and create the target output directory."""
    target = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json_atomic(data: Mapping | dict[str, Any], path: str | Path) -> Path:
    """Write JSON atomically via a temporary sibling file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(target)
    return target


def resolve_experiment_dataset(
    *,
    use_synthetic: bool,
    approve_cohort_evaluation: bool = False,
    clinical_master_path: str | Path = PROJECT_ROOT / "data" / "clinical_master.csv",
    subset_mode: SubsetMode = "all_eligible",
    endpoint: str | None = None,
    synthetic_participants: int = 92,
    synthetic_seed: int = 42,
    compact_grid: bool = False,
    cfg: PairedLipidConfig = PairedLipidConfig(),
) -> tuple[CohortDataset, dict[str, Any]]:
    """Resolve either the synthetic benchmark cohort or the verified preprocessed cohort.

    Results are memoized per argument set within a process, so repeated stages
    share one dataset object and its Gram cache. Treat the returned dataset as
    read-only; call ``clear_dataset_cache()`` to force a reload.
    """
    if not use_synthetic and not approve_cohort_evaluation:
        raise PermissionError(
            "Cohort evaluation requires explicit '--approve-cohort-evaluation' after "
            "completing all readiness checks in manuscript_ml/runbook.md."
        )
    return _resolve_experiment_dataset_cached(
        use_synthetic,
        str(Path(clinical_master_path).resolve()),
        subset_mode,
        endpoint,
        synthetic_participants,
        synthetic_seed,
        compact_grid,
        cfg,
    )


def clear_dataset_cache() -> None:
    """Drop all memoized experiment datasets."""
    _resolve_experiment_dataset_cached.cache_clear()


@lru_cache(maxsize=16)
def _resolve_experiment_dataset_cached(
    use_synthetic: bool,
    clinical_master_path: str,
    subset_mode: str,
    endpoint: str | None,
    synthetic_participants: int,
    synthetic_seed: int,
    compact_grid: bool,
    cfg: PairedLipidConfig,
) -> tuple[CohortDataset, dict[str, Any]]:
    if use_synthetic:
        if subset_mode == "all_eligible":
            dataset, truth = generate_synthetic_cohort(
                n_participants=synthetic_participants,
                seed=synthetic_seed,
                compact_grid=compact_grid,
            )
        else:
            dataset, truth = _rebuild_synthetic_with_mode(
                n_participants=synthetic_participants,
                seed=synthetic_seed,
                compact_grid=compact_grid,
                subset_mode=subset_mode,
                endpoint=endpoint,
                cfg=cfg,
            )
        meta = {
            "data_mode": "synthetic",
            "synthetic_seed": synthetic_seed,
            "compact_grid": compact_grid,
            "signal_strength": truth["signal_strength"],
        }
        return dataset, meta

    from src.data_preparation.store import get_active_data_source, resolve_data_source

    source = get_active_data_source() or resolve_data_source()
    dataset, clin_audit = load_paired_cohort_from_store(
        clinical_master_path=clinical_master_path,
        data_source=source,
        cfg=cfg,
        subset_mode=subset_mode,
        endpoint=endpoint,
    )
    return dataset, {
        "data_mode": "cohort",
        "clinical_audit": clin_audit,
        "store_path": str(source.source_path),
        "store_manifest_hash": str(source.manifest_hash),
    }


def _rebuild_synthetic_with_mode(
    *,
    n_participants: int,
    seed: int,
    compact_grid: bool,
    subset_mode: SubsetMode,
    endpoint: str | None,
    cfg: PairedLipidConfig,
) -> tuple[CohortDataset, dict[str, Any]]:
    """Helper to generate a synthetic cohort and apply a specific subset_mode.

    Known limitation (synthetic smoke runs only): the rebuilt maps add map noise on top of samples
    that are already medians of noisy maps, so subset-mode spectra are noisier than the base cohort.
    """
    import numpy as np
    from src.analysis.paired_lipid.cohort import build_paired_cohort_dataset

    base_ds, truth = generate_synthetic_cohort(
        n_participants=n_participants,
        seed=seed,
        compact_grid=compact_grid,
    )
    # Reconstruct map records from base_ds samples: 3 noisy maps per sample (generator map noise SD),
    # and 2 maps in 20% of samples so that the three-map-complete subset actually excludes samples.
    map_rng = np.random.default_rng(seed + 7)
    clin_df = base_ds.covariate_frame.copy()
    clin_df["participant_id"] = base_ds.participant_ids
    clin_df["cohort"] = "stroke"
    clin_df["identity_verified"] = True
    for ep_name, vals in base_ds.outcomes.items():
        clin_df[ep_name] = vals

    map_rows: list[dict[str, Any]] = []
    spec_1300_rows: list[np.ndarray] = []
    spec_2850_rows: list[np.ndarray] = []
    for s_idx in range(base_ds.n_samples):
        pid = str(base_ds.participant_ids[base_ds.sample_participant_idx[s_idx]])
        sid = str(base_ds.sample_ids[s_idx])
        n_maps = 2 if map_rng.random() < 0.2 else 3
        for m_idx in range(n_maps):
            r_idx = len(spec_1300_rows)
            spec_1300_rows.append(base_ds.sample_1300[s_idx] + map_rng.normal(0.0, 0.04, base_ds.sample_1300.shape[1]))
            spec_2850_rows.append(base_ds.sample_2850[s_idx] + map_rng.normal(0.0, 0.04, base_ds.sample_2850.shape[1]))
            for rng_code in ("1300", "2850"):
                map_rows.append(
                    {
                        "participant_id": pid,
                        "sample_id": sid,
                        "spectral_range": rng_code,
                        "row_index": r_idx,
                        "retained": True,
                    }
                )
    ds = build_paired_cohort_dataset(
        clin_df,
        pd.DataFrame(map_rows),
        np.vstack(spec_1300_rows),
        np.vstack(spec_2850_rows),
        base_ds.wavenumbers_1300,
        base_ds.wavenumbers_2850,
        cfg=cfg,
        subset_mode=subset_mode,
        endpoint=endpoint,
    )
    return ds, truth
