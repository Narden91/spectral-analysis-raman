"""Step 1: Measurement, linkage, input lock, and computational readiness audit.

Verifies the clinical master table, checks preprocessed store availability,
records whether runtime has been measured on the real store, and outputs ``readiness_manifest.json``
and ``limitations_table.csv``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manuscript_ml import (  # noqa: E402
    add_common_stage_args,
    ensure_output_dir,
    load_manuscript_config,
    print_stage_outputs,
    write_json_atomic,
    write_run_provenance,
)
from manuscript_ml.readiness import evaluate_cohort_readiness  # noqa: E402
from src.analysis.io_utils import write_csv_atomic  # noqa: E402
from src.analysis.paired_lipid.cohort import (  # noqa: E402
    build_limitations_table,
    load_clinical_master_verified,
    load_paired_cohort_from_store,
)


def _audit_clinical_dictionary(
    master_path: Path,
    dictionary_path: Path | None = None,
) -> dict[str, Any]:
    """Verify clinical_master_dictionary.csv SHA-256 and exact column-order parity against clinical_master.csv."""
    dict_path = dictionary_path if dictionary_path is not None else (master_path.parent / "clinical_master_dictionary.csv")
    if not dict_path.is_file():
        return {
            "present": False,
            "path": str(dict_path),
            "sha256": None,
            "n_dictionary_columns": None,
            "n_master_columns": None,
            "columns_match_exact": None,
            "missing_in_dictionary": [],
            "extra_in_dictionary": [],
        }

    dict_bytes = dict_path.read_bytes()
    dict_sha256 = hashlib.sha256(dict_bytes).hexdigest()
    with dict_path.open("r", encoding="utf-8-sig", newline="") as f_dict:
        dict_reader = csv.DictReader(f_dict)
        dict_cols = [str(row["column"]).strip() for row in dict_reader if row.get("column")]

    if not master_path.is_file():
        return {
            "present": True,
            "path": str(dict_path),
            "sha256": dict_sha256,
            "n_dictionary_columns": len(dict_cols),
            "n_master_columns": None,
            "columns_match_exact": False,
            "missing_in_dictionary": [],
            "extra_in_dictionary": list(dict_cols),
        }

    with master_path.open("r", encoding="utf-8-sig", newline="") as f_master:
        master_reader = csv.reader(f_master)
        master_cols = [str(c).strip() for c in next(master_reader, [])]

    dict_set = set(dict_cols)
    master_set = set(master_cols)
    return {
        "present": True,
        "path": str(dict_path),
        "sha256": dict_sha256,
        "n_dictionary_columns": len(dict_cols),
        "n_master_columns": len(master_cols),
        "columns_match_exact": bool(master_cols == dict_cols),
        "missing_in_dictionary": [c for c in master_cols if c not in dict_set],
        "extra_in_dictionary": [c for c in dict_cols if c not in master_set],
    }


def run_readiness_audit(
    *,
    clinical_master_path: str | Path = PROJECT_ROOT / "data" / "clinical_master.csv",
    clinical_dictionary_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    n_planned_bootstraps: int = 2000,
    n_planned_sim_reps: int = 1000,
    measured_runtime_seconds: float | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Path]:
    """Audit current measurement identities, dictionary parity, and joins without model fitting."""
    target_dir = ensure_output_dir(output_dir)
    cfg, raw_cfg = load_manuscript_config(config_path)
    readiness_cfg = raw_cfg["readiness"]
    expected_verified = int(readiness_cfg["expected_identity_verified"])

    master_path = Path(clinical_master_path)
    explicit_dict = Path(clinical_dictionary_path) if clinical_dictionary_path is not None else None
    dict_audit = _audit_clinical_dictionary(master_path, explicit_dict)
    dictionary_ok = bool(dict_audit["present"] and dict_audit["columns_match_exact"])

    if master_path.is_file():
        _, clin_audit = load_clinical_master_verified(master_path)
        clin_ready = clin_audit.get("stroke_identity_verified") == expected_verified
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
    join_status: str | None = "not_run"
    observed_summary: dict[str, Any] | None = None
    store_meta: dict[str, Any] = {}
    if clin_ready and store_manifest_exists:
        try:
            source = get_active_data_source() or resolve_data_source()
            store_meta["store_path"] = str(source.source_path)
            store_meta["store_manifest_hash"] = source.manifest_hash
            join_ds, _ = load_paired_cohort_from_store(master_path, data_source=source, cfg=cfg)
            observed_summary = dict(join_ds.manifest_summary)
            store_meta["exclusion_reasons"] = dict(Counter(e["reason"] for e in join_ds.exclusions))
            join_status = None
        except (ValueError, RuntimeError) as exc:
            # RuntimeError: the store failed validation in resolve_data_source.
            join_status = "failed"
            store_meta["error"] = str(exc)

    measured_readiness = evaluate_cohort_readiness(
        clin_audit,
        observed_summary,
        readiness_cfg,
        join_status=join_status,
        store_meta=store_meta,
    )
    cohort_join = measured_readiness["cohort_join"]
    ready_for_cohort = bool(measured_readiness["ready_for_cohort_evaluation"] and dictionary_ok)

    n_outer = int(cfg.n_outer_splits)
    n_inner = int(cfg.n_inner_splits)
    n_inner_divisions = n_outer * n_inner
    n_pca = len(cfg.pca_dimensions)
    n_lam = len(cfg.lambda_grid)
    n_dir = len(cfg.retained_directions)
    n_alpha = len(cfg.ridge_alpha_grid)
    n_pls = len(cfg.pls_component_grid)

    paired_candidates_per_division = n_pca * n_lam * n_dir * n_alpha
    established_candidates_per_division = n_alpha + n_alpha + (n_pca * n_alpha) + n_pls
    total_candidates_per_division = paired_candidates_per_division + established_candidates_per_division
    paired_division_fits = n_inner_divisions * paired_candidates_per_division
    inner_division_fits = n_inner_divisions * total_candidates_per_division
    outer_core_fits = n_outer * 5
    total_division_fits = inner_division_fits + outer_core_fits

    if measured_runtime_seconds is not None:
        comp_status = "measured_in_current_audit"
    elif ready_for_cohort:
        comp_status = "cohort_join_verified_campaign_not_timed_in_audit"
    else:
        comp_status = "cohort_join_incomplete_or_store_unavailable"

    manifest: dict[str, Any] = {
        "study_design_reference": "docs/research_design.md",
        "locked_config": asdict(cfg),
        "measurement_readiness": {
            "clinical_master_present": master_path.is_file(),
            "clinical_audit": clin_audit,
            "clinical_dictionary": dict_audit,
            "dictionary_columns_match_exact": dict_audit["columns_match_exact"],
            "preprocessed_store_present": store_manifest_exists,
            "expected_identity_verified": measured_readiness["expected_identity_verified"],
            "identity_verified_matches_expected": measured_readiness["identity_verified_matches_expected"],
            "ready_for_cohort_join": bool(cohort_join["status"] == "ok"),
            "cohort_join": cohort_join,
            "ready_for_cohort_evaluation": ready_for_cohort,
        },
        "computational_readiness": {
            "status": comp_status,
            "measured_runtime_seconds": (
                float(measured_runtime_seconds) if measured_runtime_seconds is not None else None
            ),
            "requires_separate_campaign_approval": True,
            "n_outer_folds": n_outer,
            "n_inner_folds": n_inner,
            "n_inner_divisions": n_inner_divisions,
            "paired_candidates_per_inner_fold": paired_candidates_per_division,
            "established_comparator_candidates_per_inner_fold": established_candidates_per_division,
            "total_candidates_per_inner_fold": total_candidates_per_division,
            "paired_division_candidate_fits_per_refit": paired_division_fits,
            "inner_division_candidate_fits_per_refit": inner_division_fits,
            "outer_core_model_fits_per_refit": outer_core_fits,
            "total_division_candidate_fits_per_refit": total_division_fits,
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
        settings={
            "n_planned_bootstraps": n_planned_bootstraps,
            "n_planned_sim_reps": n_planned_sim_reps,
            "measured_runtime_seconds": measured_runtime_seconds,
        },
        source_meta={
            "clinical_audit": clin_audit,
            "clinical_dictionary": dict_audit,
            "preprocessed_dir": str(store_path),
        },
    )

    return {
        "run_provenance_readiness_audit": prov_path,
        "readiness_manifest": manifest_file,
        "limitations_table": limitations_file,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit readiness gates and estimate computational budget.")
    parser.add_argument("--clinical-master", type=str, default=str(PROJECT_ROOT / "data" / "clinical_master.csv"))
    parser.add_argument(
        "--clinical-dictionary",
        type=str,
        default=None,
        help="Optional path to clinical_master_dictionary.csv (defaults to sibling of --clinical-master).",
    )
    parser.add_argument("--planned-bootstraps", type=int, default=2000)
    parser.add_argument("--planned-sim-reps", type=int, default=1000)
    parser.add_argument(
        "--measured-runtime-seconds",
        type=float,
        default=None,
        help="Optional measured wall-clock runtime (seconds) from a dry-run timing benchmark.",
    )
    add_common_stage_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_readiness_audit(
        clinical_master_path=args.clinical_master,
        clinical_dictionary_path=args.clinical_dictionary,
        output_dir=args.output_dir,
        n_planned_bootstraps=args.planned_bootstraps,
        n_planned_sim_reps=args.planned_sim_reps,
        measured_runtime_seconds=args.measured_runtime_seconds,
        config_path=args.config,
    )
    print_stage_outputs("audit_readiness", outputs)


if __name__ == "__main__":
    main()
