"""Step 1: Measurement, linkage, input lock, and computational readiness audit.

Verifies the clinical master table, checks preprocessed store availability,
benchmarks single-refit runtime and memory, and outputs ``readiness_manifest.json``
and ``limitations_table.csv``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manuscript_ml import ensure_output_dir, write_json_atomic
from src.analysis.io_utils import write_csv_atomic
from src.analysis.paired_lipid_representation import (
    PairedLipidConfig,
    build_limitations_table,
    evaluate_nested_cohort,
    generate_synthetic_cohort,
    load_clinical_master_verified,
)


def run_readiness_audit(
    *,
    clinical_master_path: str | Path = PROJECT_ROOT / "data" / "clinical_master.csv",
    preprocessed_dir: str | Path = PROJECT_ROOT / "data" / "preprocessed",
    output_dir: str | Path | None = None,
    n_planned_bootstraps: int = 2000,
    n_planned_sim_reps: int = 1000,
) -> dict[str, Path]:
    """Execute the pre-evaluation readiness audit and computational benchmark."""
    target_dir = ensure_output_dir(output_dir)
    cfg = PairedLipidConfig()

    master_path = Path(clinical_master_path)
    if master_path.is_file():
        _, clin_audit = load_clinical_master_verified(master_path)
        clin_ready = clin_audit["stroke_identity_verified"] >= 50
    else:
        clin_audit = {"status": "clinical_master_not_found", "path": str(master_path)}
        clin_ready = False

    store_path = Path(preprocessed_dir)
    store_manifest_exists = (store_path / "manifest.json").is_file()

    # Benchmark a single 5x3 nested evaluation on a 92-participant full-grid synthetic cohort
    t0 = time.perf_counter()
    bench_ds, _ = generate_synthetic_cohort(n_participants=92, n_single_sample=2, seed=42, compact_grid=False)
    _ = evaluate_nested_cohort(bench_ds, cfg=cfg)
    single_nested_sec = float(time.perf_counter() - t0)

    est_bootstrap_sec = single_nested_sec * float(n_planned_bootstraps)
    spectral_bytes = int(bench_ds.sample_1300.nbytes + bench_ds.sample_2850.nbytes)

    n_pca = len(cfg.pca_dimensions)
    n_lam = len(cfg.lambda_grid)
    n_dir = len(cfg.retained_directions)
    n_alpha = len(cfg.ridge_alpha_grid)
    paired_candidates_per_division = n_pca * n_lam * n_dir * n_alpha

    manifest: dict[str, Any] = {
        "study_design_reference": "docs/research_design.md",
        "locked_config": asdict(cfg),
        "measurement_readiness": {
            "clinical_master_present": master_path.is_file(),
            "clinical_audit": clin_audit,
            "preprocessed_store_present": store_manifest_exists,
            "ready_for_cohort_join": bool(clin_ready and store_manifest_exists),
        },
        "computational_readiness": {
            "single_nested_eval_seconds_full_grid": round(single_nested_sec, 4),
            "estimated_2000_bootstrap_minutes": round(est_bootstrap_sec / 60.0, 2),
            "requires_separate_campaign_approval": bool((est_bootstrap_sec / 60.0) > 5.0),
            "sample_spectral_memory_bytes": spectral_bytes,
            "paired_candidates_per_inner_fold": paired_candidates_per_division,
            "planned_bootstrap_refits": n_planned_bootstraps,
            "planned_simulation_repetitions_per_scenario": n_planned_sim_reps,
        },
        "attribution_readiness": {
            "status": "continuous_exploratory_summary_only",
            "categorical_assignment_enabled": False,
        },
    }

    limitations_df = build_limitations_table(clin_audit if master_path.is_file() else None)
    manifest_file = write_json_atomic(manifest, target_dir / "readiness_manifest.json")
    limitations_file = target_dir / "limitations_table.csv"
    write_csv_atomic(limitations_df, limitations_file)

    return {
        "readiness_manifest": manifest_file,
        "limitations_table": limitations_file,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit readiness gates and estimate computational budget.")
    parser.add_argument("--clinical-master", type=str, default=str(PROJECT_ROOT / "data" / "clinical_master.csv"))
    parser.add_argument("--preprocessed-dir", type=str, default=str(PROJECT_ROOT / "data" / "preprocessed"))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--planned-bootstraps", type=int, default=2000)
    parser.add_argument("--planned-sim-reps", type=int, default=1000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_readiness_audit(
        clinical_master_path=args.clinical_master,
        preprocessed_dir=args.preprocessed_dir,
        output_dir=args.output_dir,
        n_planned_bootstraps=args.planned_bootstraps,
        n_planned_sim_reps=args.planned_sim_reps,
    )
    for key, path in outputs.items():
        print(f"[audit_readiness] wrote {key}: {path}")


if __name__ == "__main__":
    main()

