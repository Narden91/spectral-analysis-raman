"""Step 3: Primary held-out LDL-C evaluation, comparator comparisons, and repeatability.

Produces:
- ``eligibility_and_split_manifest.json``
- ``nested_held_out_predictions.csv``
- ``comparator_errors_and_differences.csv``
- ``fold_selected_settings.csv``
- ``baseline_repeatability_report.csv``
- ``subspace_principal_angles.csv``
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manuscript_ml import ensure_output_dir, resolve_experiment_dataset, write_json_atomic
from src.analysis.io_utils import write_csv_atomic
from src.analysis.paired_lipid_representation import (
    PairedLipidConfig,
    evaluate_nested_cohort,
    run_multiplicity_bootstrap,
)


def _flatten_repeatability_rows(fold_repeatability: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rep in fold_repeatability:
        fold_idx = int(rep["outer_fold"])
        n_paired = int(rep.get("n_paired_test", 0))
        row: dict[str, Any] = {
            "outer_fold": fold_idx,
            "n_paired_test": n_paired,
        }
        for key in ("paired_full_icc", "paired_spectral_icc", "fixed_band_full_icc", "fixed_band_spectral_icc"):
            info = rep.get(key, {})
            row[f"{key}_estimate"] = info.get("icc_a1", float("nan"))
            row[f"{key}_estimable"] = info.get("estimable", False)
            row[f"{key}_mean_diff"] = info.get("mean_diff", float("nan"))
            row[f"{key}_rmsd"] = info.get("rmsd", float("nan"))
        for key in ("paired_discriminability", "pca_discriminability", "band_discriminability"):
            info = rep.get(key, {})
            row[f"{key}_score"] = info.get("discriminability", float("nan"))
            row[f"{key}_retrieval_top1"] = info.get("retrieval_top1", float("nan"))
            row[f"{key}_retrieval_chance"] = info.get("retrieval_chance", float("nan"))
            row[f"{key}_local_retrieval_top1"] = info.get("local_retrieval_top1", float("nan"))
            row[f"{key}_local_retrieval_chance"] = info.get("local_retrieval_chance", float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def run_primary_evaluation(
    *,
    use_synthetic: bool = True,
    approve_cohort_evaluation: bool = False,
    clinical_master_path: str | Path = PROJECT_ROOT / "data" / "clinical_master.csv",
    output_dir: str | Path | None = None,
    n_bootstraps: int = 0,
    compact_grid: bool = False,
) -> dict[str, Path]:
    """Run the primary held-out LDL-C nested cross-validation and repeatability analysis."""
    target_dir = ensure_output_dir(output_dir)
    cfg = PairedLipidConfig()
    dataset, source_meta = resolve_experiment_dataset(
        use_synthetic=use_synthetic,
        approve_cohort_evaluation=approve_cohort_evaluation,
        clinical_master_path=clinical_master_path,
        compact_grid=compact_grid,
        cfg=cfg,
    )

    eval_res = evaluate_nested_cohort(dataset, cfg=cfg, include_band_paired=True)
    outer_splits, _ = eval_res["splits"]
    fold_assignment = [-1] * dataset.n_participants
    for f_idx, (_, te_idx) in enumerate(outer_splits):
        for p_i in te_idx:
            fold_assignment[int(p_i)] = int(f_idx)

    manifest_payload = {
        "source_metadata": source_meta,
        "manifest_summary": dataset.manifest_summary,
        "exclusions": dataset.exclusions,
        "outer_seed": eval_res["outer_seed"],
        "participant_outer_fold": {
            str(pid): int(f_id) for pid, f_id in zip(dataset.participant_ids, fold_assignment)
        },
    }

    # Build held-out predictions dataframe
    pred_df = pd.DataFrame(
        {
            "participant_id": dataset.participant_ids,
            "outer_fold": fold_assignment,
            "n_eligible_samples": dataset.participant_sample_counts,
            "y_true_ldl_mmol_l": dataset.outcomes[cfg.primary_endpoint],
        }
    )
    for m_name, preds in eval_res["oof_predictions"].items():
        pred_df[f"pred_{m_name}"] = preds
    for m_name, s_vals in eval_res["oof_spectral_contributions"].items():
        pred_df[f"spectral_contrib_{m_name}"] = s_vals

    # Build comparator errors and differences table
    comp_rows: list[dict[str, Any]] = []
    for m_name, metrics in eval_res["comparisons_vs_covariate_ridge"].items():
        comp_rows.append({"comparison": f"{m_name}_vs_covariate_ridge", "model": m_name, **metrics})
    comp_rows.append(
        {
            "comparison": "paired_linear_ridge_vs_strongest_comparator",
            "model": "paired_linear_ridge",
            **eval_res["paired_vs_strongest_comparator"],
        }
    )

    if n_bootstraps > 0:
        boot_res = run_multiplicity_bootstrap(dataset, cfg=cfg, n_bootstraps=n_bootstraps)
        manifest_payload["multiplicity_bootstrap"] = {
            "n_bootstraps_requested": boot_res["n_bootstraps_requested"],
            "n_bootstraps_succeeded": boot_res["n_bootstraps_succeeded"],
            "n_failed_refits": boot_res["n_failed_refits"],
            "interval_procedure_succeeded": boot_res["interval_procedure_succeeded"],
            "ci_95_delta_mse": boot_res["ci_95_delta_mse"],
            "ci_95_vs_comparator": boot_res["ci_95_vs_comparator"],
            "monte_carlo_se": boot_res["monte_carlo_se"],
        }

    comp_df = pd.DataFrame(comp_rows)
    rep_df = _flatten_repeatability_rows(eval_res["fold_repeatability"])

    manifest_path = write_json_atomic(manifest_payload, target_dir / "eligibility_and_split_manifest.json")
    preds_path = target_dir / "nested_held_out_predictions.csv"
    comp_path = target_dir / "comparator_errors_and_differences.csv"
    settings_path = target_dir / "fold_selected_settings.csv"
    rep_path = target_dir / "baseline_repeatability_report.csv"
    angles_path = target_dir / "subspace_principal_angles.csv"

    write_csv_atomic(pred_df, preds_path)
    write_csv_atomic(comp_df, comp_path)
    write_csv_atomic(eval_res["fold_settings"], settings_path)
    write_csv_atomic(rep_df, rep_path)
    write_csv_atomic(eval_res["subspace_angles"], angles_path)

    return {
        "eligibility_and_split_manifest": manifest_path,
        "nested_held_out_predictions": preds_path,
        "comparator_errors_and_differences": comp_path,
        "fold_selected_settings": settings_path,
        "baseline_repeatability_report": rep_path,
        "subspace_principal_angles": angles_path,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run primary LDL-C nested evaluation and repeatability report.")
    parser.add_argument(
        "--cohort",
        action="store_true",
        help="Evaluate on the real stroke cohort instead of synthetic data.",
    )
    parser.add_argument(
        "--approve-cohort-evaluation",
        action="store_true",
        help="Confirm that all readiness gates in runbook.md are satisfied.",
    )
    parser.add_argument("--clinical-master", type=str, default=str(PROJECT_ROOT / "data" / "clinical_master.csv"))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--bootstraps", type=int, default=0)
    parser.add_argument("--compact-grid", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_primary_evaluation(
        use_synthetic=not args.cohort,
        approve_cohort_evaluation=args.approve_cohort_evaluation,
        clinical_master_path=args.clinical_master,
        output_dir=args.output_dir,
        n_bootstraps=args.bootstraps,
        compact_grid=args.compact_grid,
    )
    for key, path in outputs.items():
        print(f"[run_primary_evaluation] wrote {key}: {path}")


if __name__ == "__main__":
    main()

