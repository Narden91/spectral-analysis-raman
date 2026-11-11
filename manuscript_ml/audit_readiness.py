"""Step 1: Measurement, linkage, input lock, and computational readiness audit.

Verifies the clinical master table, checks preprocessed store availability,
records whether runtime has been measured on the real store, and outputs ``readiness_manifest.json``
and ``limitations_table.csv``.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manuscript_ml import ensure_output_dir, load_manuscript_config, write_json_atomic, write_run_provenance
from manuscript_ml.readiness import evaluate_cohort_readiness
from src.analysis.io_utils import write_csv_atomic
from src.analysis.paired_lipid.cohort import (
    build_limitations_table,
    load_clinical_master_verified,
    load_paired_cohort_from_store,
)


def run_readiness_audit(
    *,
    clinical_master_path: str | Path = PROJECT_ROOT / "data" / "clinical_master.csv",
    output_dir: str | Path | None = None,
    n_planned_bootstraps: int = 2000,
    n_planned_sim_reps: int = 1000,
    config_path: str | Path | None = None,
) -> dict[str, Path]:
    """Audit current measurement identities and joins without model fitting."""
    target_dir = ensure_output_dir(output_dir)
    cfg, raw_cfg = load_manuscript_config(config_path)
    expected_verified = int(raw_cfg["readiness"]["expected_identity_verified"])

    master_path = Path(clinical_master_path)
    if master_path.is_file():
        _, clin_audit = load_clinical_master_verified(master_path)
        clin_ready = clin_audit["stroke_identity_verified"] == expected_verified
    else:
        clin_audit = {"status": "clinical_master_not_found", "path": str(master_path)}
        clin_ready = False

    # The store the loaders read (data.preprocessed_dir in the project config). Check presence
    # first: resolve_data_source() would build a missing store, which an audit must never do.
    from src import resolve_cfg
    from src.data_preparation.materialize import store_dir
    from src.data_preparation.store import get_active_data_source, resolve_data_source

    store_path = store_dir("preprocessed", resolve_cfg())
    store_manifest_exists = (store_path / "manifest.json").is_file()

    # Dry-run the cohort join (counts only, no model fit): file presence alone once hid a join that
    # returned zero participants.
    expected_join = {k: int(v) for k, v in raw_cfg["readiness"]["expected_cohort_join"].items()}
    cohort_join: dict[str, Any] = {"status": "not_run", "expected": expected_join}
    if clin_ready and store_manifest_exists:
        try:
            source = get_active_data_source() or resolve_data_source()
            cohort_join.update(store_path=str(source.source_path), store_manifest_hash=source.manifest_hash)
            join_ds, _ = load_paired_cohort_from_store(master_path, data_source=source, cfg=cfg)
            observed = {k: int(join_ds.manifest_summary[k]) for k in expected_join}
            cohort_join.update(
                status="ok",
                observed=observed,
                exclusion_reasons=dict(Counter(e["reason"] for e in join_ds.exclusions)),
            )
        except (ValueError, RuntimeError) as exc:
            # RuntimeError: the store failed validation in resolve_data_source.
            cohort_join.update(status="failed", error=str(exc))

    measured_readiness = evaluate_cohort_readiness(
        clin_audit, cohort_join.get("observed", {}), raw_cfg["readiness"],
    )

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
            "expected_identity_verified": expected_verified,
            "identity_verified_matches_expected": bool(clin_ready),
            "ready_for_cohort_join": cohort_join.get("status") == "ok",
            "cohort_join": cohort_join,
            "ready_for_cohort_evaluation": measured_readiness["ready_for_cohort_evaluation"],
        },
        "computational_readiness": {
            "status": "real_data_runtime_not_measured",
            "requires_separate_campaign_approval": True,
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
    manifest_file = write_json_atomic(target_dir / "readiness_manifest.json", manifest)
    limitations_file = target_dir / "limitations_table.csv"
    write_csv_atomic(limitations_df, limitations_file, index=False)
    prov_path = write_run_provenance(
        target_dir,
        "readiness_audit",
        config_path=config_path,
        cfg=cfg,
        settings={"n_planned_bootstraps": n_planned_bootstraps, "n_planned_sim_reps": n_planned_sim_reps},
        source_meta={"clinical_audit": clin_audit, "preprocessed_dir": str(store_path)},
    )

    return {
        "run_provenance_readiness_audit": prov_path,
        "readiness_manifest": manifest_file,
        "limitations_table": limitations_file,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit readiness gates and estimate computational budget.")
    parser.add_argument("--clinical-master", type=str, default=str(PROJECT_ROOT / "data" / "clinical_master.csv"))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--planned-bootstraps", type=int, default=2000)
    parser.add_argument("--planned-sim-reps", type=int, default=1000)
    parser.add_argument("--config", type=str, default=None, help="Experiment settings YAML.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_readiness_audit(
        clinical_master_path=args.clinical_master,
        output_dir=args.output_dir,
        n_planned_bootstraps=args.planned_bootstraps,
        n_planned_sim_reps=args.planned_sim_reps,
        config_path=args.config,
    )
    for key, path in outputs.items():
        print(f"[audit_readiness] wrote {key}: {path}")


if __name__ == "__main__":
    main()

