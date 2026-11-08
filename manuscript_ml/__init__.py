"""Experiment runners for the reproducible Raman lipid representation study.

All scripts delegate mathematical, statistical, and spectral operations to
``src.analysis.paired_lipid_representation`` and ``src.analysis.io_utils``
to preserve DRY and KISS principles.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Literal, Mapping

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.io_utils import write_csv_atomic
from src.analysis.paired_lipid_representation import (
    CohortDataset,
    PairedLipidConfig,
    generate_synthetic_cohort,
    load_paired_cohort_from_store,
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Results_manuscript_ml"


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
    subset_mode: Literal["all_eligible", "paired_only", "single_sample", "three_map_complete"] = "all_eligible",
    endpoint: str | None = None,
    synthetic_participants: int = 92,
    synthetic_seed: int = 42,
    compact_grid: bool = False,
    cfg: PairedLipidConfig = PairedLipidConfig(),
) -> tuple[CohortDataset, dict[str, Any]]:
    """Resolve either the synthetic benchmark cohort or the verified preprocessed cohort."""
    if use_synthetic:
        dataset, truth = generate_synthetic_cohort(
            n_participants=synthetic_participants,
            seed=synthetic_seed,
            compact_grid=compact_grid,
        )
        if subset_mode != "all_eligible":
            # Rebuild synthetic dataset under requested subset_mode
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

    if not approve_cohort_evaluation:
        raise PermissionError(
            "Cohort evaluation requires explicit '--approve-cohort-evaluation' after "
            "completing all readiness checks in manuscript_ml/runbook.md."
        )
    dataset, clin_audit = load_paired_cohort_from_store(
        clinical_master_path=clinical_master_path,
        cfg=cfg,
        subset_mode=subset_mode,
        endpoint=endpoint,
    )
    meta = {
        "data_mode": "cohort",
        "clinical_audit": clin_audit,
    }
    return dataset, meta


def _rebuild_synthetic_with_mode(
    *,
    n_participants: int,
    seed: int,
    compact_grid: bool,
    subset_mode: Literal["all_eligible", "paired_only", "single_sample", "three_map_complete"],
    endpoint: str | None,
    cfg: PairedLipidConfig,
) -> tuple[CohortDataset, dict[str, Any]]:
    """Helper to generate a synthetic cohort and apply a specific subset_mode."""
    import numpy as np
    from src.analysis.paired_lipid_representation import build_paired_cohort_dataset

    base_ds, truth = generate_synthetic_cohort(
        n_participants=n_participants,
        seed=seed,
        compact_grid=compact_grid,
    )
    # Reconstruct map records from base_ds samples (3 maps per sample)
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
        for m_idx in range(3):
            r_idx = len(spec_1300_rows)
            spec_1300_rows.append(base_ds.sample_1300[s_idx])
            spec_2850_rows.append(base_ds.sample_2850[s_idx])
            for rng_code in ("1300", "2850"):
                map_rows.append(
                    {
                        "participant_id": pid,
                        "sample_id": sid,
                        "spectral_range": rng_code,
                        "row_index": r_idx,
                        "acquisition_mode": "SS",
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
